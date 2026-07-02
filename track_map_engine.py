# =============================================================================
# track_map_engine.py
# Onboard Localization Engine — Track Map Engine & Geometric Projection

# Responsibilities:
#   - Load track JSON (nodes + segments)
#   - Convert all geodetic points from LLA → ECEF → ENU (local metric frame)
#   - Build Shapely LineStrings, segment lengths, curvature profiles
#   - Build R-tree spatial index for fast candidate segment queries
#   - Build NetworkX topology graph for undirected adjacency queries
#   - Provide geometric projection: raw GNSS fix → (s_proj, d_cross)

# COORDINATE SYSTEM:
#   All internal geometry is in local ENU (East, North, Up) meters.
#   Reference point ENU(0,0,0) is defined in config.yaml.
#   Conversion pipeline: LLA (EPSG:4326) → ECEF (EPSG:4978) → ENU
#   Raw lat/lon is NEVER used for distance math — always convert first.

# SIGN CONTRACT:
#   All geometric quantities are parameterized in the stored point order
#   (first_node → last_node). s=0 at first_node, s=length at last_node.
#
#   Curvature sign follows stored point order. No runtime sign flip is applied.
#   The KF velocity sign resolves direction:
#     positive velocity = moving toward last_node (nominal direction)
#     negative velocity = moving toward first_node (reverse direction)
#   omega_expected = velocity × kappa handles sign automatically.
#
#   direction_qualifier is used ONLY by distance_to_nearest_switch to determine
#   which end of the segment the train is approaching.

# CURVATURE SIGN CONVENTION:
#   κ_stored(s) = menger_magnitude(s) × imu_sign
#   where imu_sign is fixed at load time from config.yaml imu.z_axis_down.
#   This ensures κ_map(s) is directly comparable to ω_sensor from the IMU.
#   See _build_curvature_profiles() for full derivation.
# =============================================================================

from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import yaml
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
from rtree import index as rtree_index
from pyproj import Transformer

logger = logging.getLogger(__name__)

# Import package modules
from sensor_interface import SensorFix


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TrackNode:
    """A topological node in the track graph.

    Nodes are switches, endpoints, or junctions where segments connect.
    All geometry lives in TrackSegment; nodes carry only position and type.

    Attributes:
        id: Node ID (e.g. "N1").
        type: One of "endpoint", "switch", or "junction".
        lat: WGS84 latitude, degrees.
        lon: WGS84 longitude, degrees.
        alt: WGS84 ellipsoidal height, meters.
    """
    id: str # N1, N2, etc.
    type: str           # "endpoint", "switch", "junction"
    lat: float          # WGS84 degrees
    lon: float          # WGS84 degrees
    alt: float          # meters


@dataclass
class TrackSegment:
    """An undirected track segment between two consecutive nodes with ordered points.

    Raw fields are populated from JSON; derived fields are computed at load
    time by TrackMapEngine. All derived geometry is in local ENU meters.

    Geometry is parameterized in stored point order (first_node → last_node).
    s=0 at first_node, s=length at last_node. Does NOT imply allowed travel
    direction — the train may traverse the segment in either direction.

    Attributes:
        id: Segment ID string.
        first_node: ID of the node at s=0 (start of stored point order).
        last_node: ID of the node at s=length (end of stored point order).
        points: Raw LLA dicts [{"lat":..,"lon":..,"alt":..}, ...] from JSON.
        enu_points: ENU (East, North, Up) coordinates in meters, same order as points.
        linestring: Shapely 2D LineString in (E, N) for geometric operations.
        length: Segment length in meters.
        curvature_profile: List of (s, κ) tuples sorted by s. κ in 1/m,
            IMU-sign-corrected in stored point order.
    """
    # --- From JSON ---
    id: str
    first_node: str
    last_node: str
    points: List[Dict]          # raw LLA dicts: [{"lat":..,"lon":..,"alt":..}, ...]

    # --- Derived at load time (populated by TrackMapEngine) ---
    enu_points: List[Tuple[float, float, float]] = field(default_factory=list)
    # ENU (East, North, Up) coordinates in meters, same order as points.

    linestring: Optional[LineString] = field(default=None)
    # Shapely LineString in (E, N) — used for all 2D geometric operations.
    # Altitude (U) is excluded; vertical geometry not needed for track matching.

    length: float = field(default=0.0)
    # Segment length in meters, derived from linestring.length.

    curvature_profile: List[Tuple[float, float]] = field(default_factory=list)
    # List of (s, kappa) tuples, sorted by s (along-track distance from first_node).
    # kappa units: 1/m (reciprocal meters).
    # kappa sign: imu_sign applied — directly comparable to ω_sensor / v̂.
    # Stored in first_node→last_node order.
    # Query via TrackMapEngine.get_curvature(segment_id, s) for interpolation.


@dataclass
class TrackProjectionCoordinates:
    """Output of project_fix_to_segment().

    Attributes:
        segment_id: Segment this projection is onto.
        s_proj: Along-track distance from first_node, meters. Feeds the KF.
        d_cross: Unsigned perpendicular distance from fix to track, meters.
            Feeds L_pos = gaussian(d_cross, σ_pos). Always >= 0.
        side: +1 if fix is left of first→last direction, -1 if right.
            Diagnostic only — not used in scoring.
        e_proj: Projected ENU easting, meters.
        n_proj: Projected ENU northing, meters.
        track_heading_rad: Local track heading at s_proj, radians from East
            in ENU convention (atan2(dN, dE) of the chord at this position).

    Note:
        s_proj and d_cross must never be combined — doing so contaminates
        track discrimination with KF velocity estimation noise.
    """
    segment_id: str
    s_proj: float
    d_cross: float
    side: int
    e_proj: float
    n_proj: float
    track_heading_rad: float = 0.0


# =============================================================================
# TrackMapEngine
# =============================================================================

class TrackMapEngine:
    """Central map service for the MHMM localization engine.

    Instantiate once at startup. Provides geometric projection, topology
    queries, curvature lookup, cold-start candidate search, and switch
    proximity queries.

    Example:
        engine = TrackMapEngine("track.json", "config.yaml")
        result = engine.project_fix_to_segment(fix, "S1")
        connected = engine.get_connected_segments("S1", "N2")
        kappa = engine.get_curvature("S2", s=45.3)
    """

    def __init__(self, json_path: str, config_path: str):
        """Load track map and build all derived quantities.

        Args:
            json_path: Path to track JSON file.
            config_path: Path to config.yaml.
        """
        # Initialise private metthods in defined order to ensure correct population of derived fields.
        self._load_config(config_path)
        self._load_json(json_path)
        self._init_enu_transform()
        self._build_geometries()
        self._build_curvature_profiles()
        self._build_spatial_index()
        self._build_topology_graph()

        logger.info(
            f"TrackMapEngine ready | "
            f"{len(self.segments)} segments | "
            f"{len(self.nodes)} nodes | "
            f"imu_z_down={self.imu_z_down}"
        )

    # -------------------------------------------------------------------------
    # Initialisation - private methods (called once at startup)
    # -------------------------------------------------------------------------

    def _load_config(self, config_path: str) -> None:
        """Load config.yaml and set ENU reference and IMU sign convention.

        Populates self.ref_lat/lon/alt, self.imu_z_down, and self.imu_sign.
        imu_sign (±1) aligns stored curvature with the IMU yaw rate convention.

        Args:
            config_path: Path to config.yaml.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        ref = cfg["reference"]
        self.ref_lat: float = ref["lat"]    # degrees
        self.ref_lon: float = ref["lon"]    # degrees
        self.ref_alt: float = ref["alt"]    # meters

        self.imu_z_down: bool = cfg["imu"]["z_axis_down"]

        # imu_sign reconciles the geometric curvature (always positive from
        # Menger formula) with the IMU yaw rate sign convention.
        #
        # imu_z_down = True  (z points down, aerospace convention):
        #   right-hand (clockwise from above) turn → positive ω_sensor
        #   We want κ_stored > 0 for clockwise turns → imu_sign = -1
        #   because ENU cross product gives negative z for clockwise turns
        #   and we want menger_magnitude × imu_sign to be positive there.
        #
        # imu_z_down = False (z points up, ENU convention):
        #   left-hand (counterclockwise from above) turn → positive ω_sensor
        #   Menger magnitude is always positive, so imu_sign = +1.
        #
        # VERIFY THIS AGAINST YOUR IMU DATASHEET before field testing.
        self.imu_sign: int = -1 if self.imu_z_down else +1

        # min search radius for get_candidate_segments() — ensures we always return candidates even if GNSS accuracy is poor.
        self.min_search_radius_m: float = cfg["min_search_radius_m"]

        # Number of terminal chords used for circular-mean heading estimation at
        # switch ends. See get_approach_heading() / get_departure_heading().
        # Also used by get_heading() interior smoothing when heading_smoothing=True.
        self.heading_smoothing_chords: int = int(cfg.get("heading_smoothing_chords", 3))

        # When True, get_heading() uses chord-length-weighted circular mean over
        # heading_smoothing_chords nearest chords instead of raw two-chord
        # interpolation.
        self.heading_smoothing: bool = bool(cfg.get("heading_smoothing", False))

        logger.debug(
            f"Config loaded | ref=({self.ref_lat:.6f}, {self.ref_lon:.6f}, "
            f"{self.ref_alt:.2f}) | imu_z_down={self.imu_z_down} | "
            f"imu_sign={self.imu_sign} | min_search_radius_m={self.min_search_radius_m:.2f} | "
            f"heading_smoothing_chords={self.heading_smoothing_chords} | "
            f"heading_smoothing={self.heading_smoothing}"
        )

    def _load_json(self, json_path: str) -> None:
        """Parse track JSON into TrackNode and TrackSegment objects.

        Populates self.nodes (Dict[str, TrackNode]) and self.segments
        (Dict[str, TrackSegment]), both keyed by respective IDs.

        Args:
            json_path: Path to track file (JSON).
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        self.nodes: Dict[str, TrackNode] = {}
        for n in data["nodes"]:
            self.nodes[n["id"]] = TrackNode(
                id=n["id"],
                type=n["type"],
                lat=n["lat"],
                lon=n["lon"],
                alt=n["alt"],
            )

        self.segments: Dict[str, TrackSegment] = {}
        for s in data["segments"]:
            # Detect duplicate segment IDs before the dict assignment would
            # silently overwrite the first entry, making the duplicate
            # invisible to _validate_topology.
            if s["id"] in self.segments:
                logger.error(
                    "_load_json | duplicate segment ID: %s — "
                    "second entry ignored. Fix the track JSON.",
                    s["id"],
                )
                continue
            self.segments[s["id"]] = TrackSegment(
                id=s["id"],
                first_node=s.get("first_node", s.get("from_node")),
                last_node=s.get("last_node", s.get("to_node")),
                points=s["points"],     # [{"lat":..,"lon":..,"alt":..}, ...]
            )

        logger.debug(
            f"JSON loaded | {len(self.nodes)} nodes | {len(self.segments)} segments"
        )

    def _init_enu_transform(self) -> None:
        """Initialise LLA→ECEF transformer and ECEF→ENU rotation matrix.

        Pipeline: LLA (EPSG:4326) → ECEF (EPSG:4978) → ENU. ENU origin is
        at (ref_lat, ref_lon, ref_alt). All subsequent geometry is in ENU
        meters. NOTE: NO UTM zone approximation.

        Rotation: [E, N, U]ᵀ = R · (P_ecef - P_ref_ecef)
        """
        # Transformer: LLA → ECEF
        # always_xy=True: input order is (lon, lat, alt) — be explicit.
        self._lla_to_ecef = Transformer.from_crs(
            "EPSG:4326", "EPSG:4978", always_xy=True
        )

        # Reference point in ECEF
        x_r, y_r, z_r = self._lla_to_ecef.transform(
            self.ref_lon, self.ref_lat, self.ref_alt
        )
        self._ecef_ref = np.array([x_r, y_r, z_r])

        # Precompute ECEF → ENU rotation matrix at reference point
        phi = math.radians(self.ref_lat)
        lam = math.radians(self.ref_lon)
        sp, cp = math.sin(phi), math.cos(phi)
        sl, cl = math.sin(lam), math.cos(lam)

        self._R_ecef_to_enu = np.array([
            [-sl,       cl,       0.0],
            [-sp * cl, -sp * sl,  cp ],
            [ cp * cl,  cp * sl,  sp ],
        ])

        # Inverse transformer: ECEF → LLA (used by enu_to_lla).
        # Built once here alongside the forward transformer — Transformer.from_crs
        # is not cheap to call repeatedly.
        self._ecef_to_lla = Transformer.from_crs(
            "EPSG:4978", "EPSG:4326", always_xy=True
        )

        logger.debug(
            f"ENU transform initialised | "
            f"ECEF ref=({x_r:.2f}, {y_r:.2f}, {z_r:.2f})"
        )

    def _lla_to_enu(self, lat: float, lon: float, alt: float
                    ) -> Tuple[float, float, float]:
        """Convert a single LLA point to local ENU meters.

        Pipeline: LLA → ECEF → ENU. Valid globally, no zone approximation.

        Args:
            lat: WGS84 latitude, degrees.
            lon: WGS84 longitude, degrees.
            alt: WGS84 ellipsoidal height, meters.

        Returns:
            (E, N, U) tuple in meters relative to the reference point.
        """
        x, y, z = self._lla_to_ecef.transform(lon, lat, alt)
        delta_ecef = np.array([x, y, z]) - self._ecef_ref
        enu = self._R_ecef_to_enu @ delta_ecef
        return float(enu[0]), float(enu[1]), float(enu[2])

    def _build_geometries(self) -> None:
        """Convert each segment's LLA points to ENU and build Shapely LineStrings.

        Populates segment.enu_points, segment.linestring (2D, E/N only), and
        segment.length. Altitude is stored in enu_points but excluded from the
        LineString (vertical geometry not needed for track matching).
        """
        for seg in self.segments.values():
            enu_pts = [
                self._lla_to_enu(p["lat"], p["lon"], p["alt"])
                for p in seg.points
            ]
            seg.enu_points = enu_pts

            # NOTE: Build 2D LineString from (E, N) pairs only
            seg.linestring = LineString([(e, n) for e, n, u in enu_pts])
            seg.length = seg.linestring.length

            logger.debug(
                f"Segment {seg.id} | {len(enu_pts)} points | "
                f"length={seg.length:.2f}m"
            )

    def _build_curvature_profiles(self) -> None:
        """Compute signed curvature profiles for all segments.

        For each interior point i, Menger curvature magnitude (meters^-1) is:
            κ_mag = 4 × Area(P_{i-1}, P_i, P_{i+1}) / (|a| × |b| × |c|)
        then multiplied by imu_sign to align with the IMU yaw rate convention:
            κ_stored = κ_mag × imu_sign

        Profile: list of (s, κ) tuples sorted by s (along-track distance from
        first_node). Populates segment.curvature_profile for all segments.

        ENDPOINT HANDLING — clamp to nearest interior value:
            Menger curvature cannot be computed at the first and last points
            (no predecessor/successor respectively). So, the endpoint value is 
            clamped to the nearest computed interior curvature:
                profile[0]  <- κ of first interior point  (index 1)
                profile[-1] <- κ of last  interior point  (index n-2)

        Note:
            κ_stored is in stored point order (first_node→last_node).
            KF velocity sign handles direction in omega_expected = velocity × kappa.
            No sign flip is applied here or in get_curvature().
        """
        for seg in self.segments.values():
            pts = seg.enu_points
            n = len(pts)
            profile: List[Tuple[float, float]] = []

            if n < 3:
                # Degenerate segment — not enough points for curvature.
                # Treat as straight track.
                seg.curvature_profile = [(0.0, 0.0), (seg.length, 0.0)]
                logger.warning(
                    f"Segment {seg.id} has fewer than 3 points — "
                    f"curvature set to zero."
                )
                continue

            # Accumulate along-track distance for each point
            cumulative_s = [0.0]
            for i in range(1, n): # start from second point to n point to compute chord lengths
                dx = pts[i][0] - pts[i-1][0]
                dy = pts[i][1] - pts[i-1][1]
                cumulative_s.append(cumulative_s[-1] + math.hypot(dx, dy))

            # Compute interior curvatures first (indices 1 to n-2).
            # Endpoint entries are deferred until first/last interior values
            # are known so they can be clamped rather than forced to zero.
            interior: List[Tuple[float, float]] = []

            for i in range(1, n - 1):
                p_prev = pts[i - 1]
                p_curr = pts[i]
                p_next = pts[i + 1]

                # Chord vectors in ENU (E, N) — altitude excluded
                ax = p_prev[0] - p_curr[0]
                ay = p_prev[1] - p_curr[1]
                bx = p_next[0] - p_curr[0]
                by = p_next[1] - p_curr[1]
                cx = p_prev[0] - p_next[0]
                cy = p_prev[1] - p_next[1]

                len_a = math.hypot(ax, ay)   # |P_{i-1} - P_i|
                len_b = math.hypot(bx, by)   # |P_{i+1} - P_i|
                len_c = math.hypot(cx, cy)   # |P_{i-1} - P_{i+1}|

                # Check for duplicate points which would cause zero-length chords and division by zero.
                denom = len_a * len_b * len_c
                if denom < 1e-10:
                    # Degenerate triplet (collinear or duplicate points).
                    # NOTE: Assign zero curvature — safe default.
                    interior.append((cumulative_s[i], 0.0))
                    continue

                # Triangle area via cross product (signed, but we take abs)
                # cross_z = chord1 × chord2 (z-component)
                # chord1 = P_i - P_{i-1}, chord2 = P_{i+1} - P_i
                c1x = p_curr[0] - p_prev[0]
                c1y = p_curr[1] - p_prev[1]
                c2x = p_next[0] - p_curr[0]
                c2y = p_next[1] - p_curr[1]
                area_signed = c1x * c2y - c1y * c2x   # = 2 × signed triangle area
                area_abs = abs(area_signed) / 2.0

                # Menger curvature magnitude
                kappa_mag = (4.0 * area_abs) / denom

                # Apply IMU sign convention.
                # κ_stored is positive when the turn direction produces a
                # positive yaw rate on the IMU as configured.
                # Stored in first_node→last_node order. KF velocity sign handles direction.
                kappa_stored = kappa_mag * self.imu_sign

                interior.append((cumulative_s[i], kappa_stored))

            if not interior:
                logger.warning(
                    f"Segment {seg.id} has no valid interior points — "
                    f"curvature set to zero."
                )
            
            # Assemble profile: clamp endpoints to nearest interior value.
            # If interior is empty (n==3, only one interior point degenerate),
            # fall back to zero for both endpoints — identical to prior behaviour.
            kappa_first = interior[0][1]  if interior else 0.0
            kappa_last  = interior[-1][1] if interior else 0.0

            profile.append((cumulative_s[0],  kappa_first))   # first_node -> clamp
            profile.extend(interior)
            profile.append((cumulative_s[-1], kappa_last))    # last_node  -> clamp

            seg.curvature_profile = profile

    def _build_spatial_index(self) -> None:
        """Build an R-tree spatial index over segment bounding boxes (ENU).

        Used by get_candidate_segments() for fast nearest-segment queries at
        cold start. self._rtree_id_map maps integer R-tree IDs to segment ID
        strings (rtree requires integer keys).
        """
        self._rtree_id_map: Dict[int, str] = {}
        p = rtree_index.Property()
        p.dimension = 2 # 2D index over ENU (E, N) — altitude not included for track matching
        self._spatial_index = rtree_index.Index(properties=p)

        for i, seg in enumerate(self.segments.values()):
            # Check if linestring is valid before indexing
            if seg.linestring is None:
                logger.warning(
                    f"Segment {seg.id} has invalid geometry and will be "
                    f"excluded from spatial index."
                )
                raise ValueError(f"Segment {seg.id} has invalid geometry.")
            
            bounds = seg.linestring.bounds   # (min_E, min_N, max_E, max_N)
            self._spatial_index.insert(i, bounds)
            self._rtree_id_map[i] = seg.id

        # LOGGING: report number of indexed segments and any geometry issues
        logger.debug(
            f"R-tree spatial index built | {len(self._rtree_id_map)} entries for {len(self.segments)} segments"
        )

    def _build_topology_graph(self) -> None:
        """Build a NetworkX MultiGraph (undirected) encoding track topology.

        MultiGraph (undirected) is required to support parallel segments between
        the same two nodes (e.g. a main track and a siding both running between
        switch N2 and switch N3). Unlike Graph/ DiGraph, MultiGraph preserves parallel edges between the
        same node pair while edges have no direction. get_connected_segments()
        queries edges incident to a node regardless of stored point order.
        """
        self.graph = nx.MultiGraph()

        # Add nodes
        for node in self.nodes.values():
            self.graph.add_node(node.id, type=node.type)

        # Add edges (one per segment, undirected)
        for seg in self.segments.values():
            self.graph.add_edge(
                seg.first_node,
                seg.last_node,
                segment_id=seg.id
            )

        logger.debug(
            f"Topology graph built | "
            f"{self.graph.number_of_nodes()} nodes | "
            f"{self.graph.number_of_edges()} edges"
        )

        # Load-time validation
        self._validate_topology()

    # -------------------------------------------------------------------------
    # Public API - geometric projection
    # -------------------------------------------------------------------------

    def project_fix_to_segment(
        self,
        fix: SensorFix,
        segment_id: str,
    ) -> TrackProjectionCoordinates:
        """Project a raw GNSS fix onto a track segment.

        Returns s_proj (along-track, feeds KF) and d_cross (cross-track, feeds
        L_pos). These must never be combined — see TrackProjectionCoordinates.

        s_proj is always measured from first_node regardless of travel direction.

        Args:
            fix: SensorFix with lat, lon, alt, horizontal_accuracy.
            segment_id: Segment to project onto.

        Returns:
            TrackProjectionCoordinates with s_proj, d_cross, side, enu_e, enu_n.
        """
        seg = self.segments[segment_id]

        # Check if segment geometry is valid before projection
        if seg.linestring is None:
            logger.error(
                f"Segment {segment_id} has invalid geometry — cannot project fix."
            )
            raise ValueError(f"Segment {segment_id} has invalid geometry.")

        # Step 1: Convert fix to ENU
        fix_e, fix_n, fix_u = self._lla_to_enu(fix.lat, fix.lon, fix.alt)
        fix_point = Point(fix_e, fix_n)

        # Step 2: Along-track projection (Shapely internally does point-to-
        # polyline projection: finds nearest chord, accumulates arc length).
        s_proj = seg.linestring.project(fix_point)

        # Step 3: Cross-track distance (unsigned, perpendicular).
        # Always >= 0. Used directly in L_pos = gaussian(d_cross, σ_pos).
        d_cross = seg.linestring.distance(fix_point)

        # Step 4: Lateral side determination (diagnostic only, not used in scoring).
        # Find nearest point on linestring, then find which chord it falls on,
        # then compute cross product to determine left (+1) or right (-1).
        side = self._compute_side(seg, s_proj, fix_e, fix_n)
        
        # Step 5: Compute projected ENU coordinates using "track_to_enu". NOTE: for diagnostics (not used in scoring).
        e_proj, n_proj, _ = self.track_to_enu(segment_id,s_proj) # u: float = 0.0 (default)

        # Step 6: Local track heading at s_proj for directional GNSS sigma projection.
        # Used in observation_likelihood to project the 2D error ellipse onto
        # cross-track (sigma_cross → L_pos) and along-track (sigma_along → KF R).
        track_heading_rad = self.get_heading(segment_id, s_proj)

        return TrackProjectionCoordinates(
            segment_id=segment_id,
            s_proj=s_proj,
            d_cross=d_cross,
            side=side,
            e_proj=e_proj,
            n_proj=n_proj,
            track_heading_rad=track_heading_rad,
        )

    def _compute_side(
        self,
        segment: TrackSegment,
        s_proj: float,
        fix_e: float,
        fix_n: float,
    ) -> int:
        """Determine which side of the track the fix is on.

        Finds the chord containing s_proj, then computes the cross product of
        the chord vector with the fix-offset vector. Diagnostic only — not
        used in scoring.

        Args:
            segment: The TrackSegment to check against.
            s_proj: Along-track projection distance, meters.
            fix_e: Fix ENU easting, meters.
            fix_n: Fix ENU northing, meters.

        Returns:
            +1 if fix is left of first_node→last_node direction,
            -1 if right, 0 if on the track.
        """
        pts = segment.enu_points
        n = len(pts)

        # Accumulate s along the linestring to find the correct chord
        s_acc = 0.0
        for i in range(n - 1):
            dx = pts[i+1][0] - pts[i][0]
            dy = pts[i+1][1] - pts[i][1]
            chord_len = math.hypot(dx, dy)
            s_next = s_acc + chord_len

            if s_next >= s_proj or i == n - 2:
                # This chord contains the projection point
                # Cross product: chord_vec × fix_offset_vec
                # chord_vec = (dx, dy), fix_offset = (fix - P_i)
                off_x = fix_e - pts[i][0]
                off_y = fix_n - pts[i][1]
                cross_z = dx * off_y - dy * off_x
                if cross_z > 1e-9:
                    return +1   # left of from→to direction
                elif cross_z < -1e-9:
                    return -1   # right of from→to direction
                else:
                    return 0    # on the track
            s_acc = s_next

        return 0

    # -------------------------------------------------------------------------
    # Public API - curvature lookup
    # -------------------------------------------------------------------------

    def get_curvature(
        self,
        segment_id: str,
        s_query: float,
    ) -> float:
        """Return linearly interpolated curvature at along-track position s_query.

        Returns curvature in stored point order (first_node → last_node).
        No sign flip is applied. Caller resolves direction via KF velocity sign
        in omega_expected = velocity × kappa.

        Args:
            segment_id: Segment to query.
            s_query: Along-track position from first_node, meters.

        Returns:
            κ in 1/m, IMU-sign-corrected in stored point order.
            Use as: ω_expected = velocity × get_curvature(segment_id, s_query).
            The KF velocity sign is negative when traveling toward first_node,
            which automatically produces the correct signed omega_expected.
        """
        seg = self.segments[segment_id]
        profile = seg.curvature_profile

        if not profile:
            return 0.0

        # Clamp s_query to valid range. NOTE: To avoid the KF s estimates produced slightly outside [0, length]
        s_query = max(profile[0][0], min(profile[-1][0], s_query))

        # Linear interpolation between the two nearest profile entries. NOTE: This is efficient because the profile [s, value] is pre-sorted by s at build time.
        # Profile is sorted by s (guaranteed by _build_curvature_profiles)
        kappa = self._interpolate_profile(profile, s_query)

        return kappa

    @staticmethod
    def _interpolate_profile(
        profile: List[Tuple[float, float]], s_query: float
    ) -> float:
        """Linearly interpolate a sorted (s, value) profile at position s_query.

        Args:
            profile: Sorted list of (s, value) tuples.
            s_query: Query position.

        Returns:
            Interpolated value at s_query.
        """
        if s_query <= profile[0][0]:
            return profile[0][1]
        if s_query >= profile[-1][0]:
            return profile[-1][1]

        # Binary search for bracketing interval
        lo, hi = 0, len(profile) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if profile[mid][0] <= s_query:
                lo = mid
            else:
                hi = mid

        s0, v0 = profile[lo]
        s1, v1 = profile[hi]
        t = (s_query - s0) / (s1 - s0) if (s1 - s0) > 1e-10 else 0.0
        return v0 + t * (v1 - v0)

    # -------------------------------------------------------------------------
    # Public API - heading lookup
    # -------------------------------------------------------------------------

    def get_heading(
        self,
        segment_id: str,
        s_query: float,
    ) -> float:
        """Return the local track heading at along-track position s_query.

        Two modes selected by self.heading_smoothing (config: heading_smoothing):

        RAW (default, heading_smoothing=False):
            Heading is atan2(dN, dE) of the chord bracketing s_query, giving the
            direction from East in ENU convention (radians). Linear interpolation
            between adjacent chord headings is used for smoothness.

        SMOOTHED (heading_smoothing=True):
            Chord-length-weighted circular mean over the heading_smoothing_chords
            chords nearest to s_query. Suppresses per-point survey noise at the cost of 
            slightly reduced spatial resolution. The circular mean correctly handles ±π wraparound.

        Args:
            segment_id: Segment to query.
            s_query: Along-track position from first_node, meters.

        Returns:
            Track heading in radians from East (ENU convention).
            Range: (-π, π]. East=0, North=π/2, West=±π, South=-π/2.
        """
        seg = self.segments[segment_id]
        pts = seg.enu_points
        n = len(pts)

        if n < 2:
            logger.warning(
                "get_heading | seg=%s has fewer than 2 points — returning 0.0",
                segment_id,
            )
            return 0.0

        # Clamp to valid segment range.
        s_query = max(0.0, min(seg.length, s_query))

        # -------------------------------------------------------------------------
        # SMOOTHED PATH — chord-length-weighted circular mean
        # -------------------------------------------------------------------------
        if self.heading_smoothing:
            # Build cumulative arc-length table for all chords.
            # Each chord i spans [s_acc[i], s_acc[i+1]) and has midpoint at
            # s_acc[i] + chord_len/2. We select the heading_smoothing_chords
            # chords whose midpoints are nearest to s_query.
            chord_mids: List[Tuple[float, float, float]] = []  # (mid_s, dx, dy)
            s_acc = 0.0
            for i in range(n - 1):
                dx = pts[i + 1][0] - pts[i][0]
                dy = pts[i + 1][1] - pts[i][1]
                chord_len = math.hypot(dx, dy)
                chord_mids.append((s_acc + chord_len / 2.0, dx, dy))
                s_acc += chord_len

            if not chord_mids:
                return 0.0

            # Sort chords by distance of their midpoint from s_query and keep nearest N.
            n_chords = min(self.heading_smoothing_chords, len(chord_mids))
            nearest = sorted(chord_mids, key=lambda c: abs(c[0] - s_query))[:n_chords]

            # Chord-length-weighted circular mean (same logic as _terminal_heading).
            sin_sum = 0.0
            cos_sum = 0.0
            for _, dx, dy in nearest:
                length = math.hypot(dx, dy)
                if length < 1e-10:
                    continue  # degenerate chord — skip
                heading = math.atan2(dy, dx)
                sin_sum += length * math.sin(heading)
                cos_sum += length * math.cos(heading)

            if math.hypot(sin_sum, cos_sum) < 1e-10:
                # Degenerate fallback: all chords cancelled — use nearest single chord.
                _, dx, dy = nearest[0]
                return math.atan2(dy, dx)

            return math.atan2(sin_sum, cos_sum)

        # -------------------------------------------------------------------------
        # RAW PATH — two-chord linear interpolation (unchanged default)
        # -------------------------------------------------------------------------
        # Build heading profile: (s, heading_rad) anchored at chord midpoints.
        # heading_rad = atan2(dN, dE) — ENU convention, East=0, ccw positive.
        # Anchoring at chord midpoints ensures interpolation covers [0, length]
        # without extrapolation: the first and last entries are then re-clamped
        # to s=0 and s=length respectively.
        heading_profile: List[Tuple[float, float]] = []
        s_acc = 0.0
        for i in range(n - 1):
            de = pts[i + 1][0] - pts[i][0]
            dn = pts[i + 1][1] - pts[i][1]
            chord_len = math.hypot(de, dn)
            heading_rad = math.atan2(dn, de)
            heading_profile.append((s_acc + chord_len / 2.0, heading_rad))
            s_acc += chord_len

        if not heading_profile:
            return 0.0

        # Clamp first and last anchor to segment endpoints.
        heading_profile[0]  = (0.0,        heading_profile[0][1])
        heading_profile[-1] = (seg.length,  heading_profile[-1][1])

        # Boundary returns.
        if s_query <= heading_profile[0][0]:
            return heading_profile[0][1]
        if s_query >= heading_profile[-1][0]:
            return heading_profile[-1][1]

        # Binary search for bracketing interval (same pattern as _interpolate_profile).
        lo, hi = 0, len(heading_profile) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if heading_profile[mid][0] <= s_query:
                lo = mid
            else:
                hi = mid

        s0, h0 = heading_profile[lo]
        s1, h1 = heading_profile[hi]
        t = (s_query - s0) / (s1 - s0) if (s1 - s0) > 1e-10 else 0.0

        # Wrap angular difference to (-π, π] to correctly interpolate through
        # sharp heading changes (e.g. 170° bends where naive subtraction aliases).
        diff = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
        return h0 + t * diff

    # -------------------------------------------------------------------------
    # Public API - cold start
    # -------------------------------------------------------------------------

    def get_candidate_segments(
        self,
        fix: SensorFix,
        n_sigma: float = 3.0,
    ) -> List[str]:
        """Return segment IDs within n_sigma × horizontal_accuracy of the fix.

        Used at cold start to spawn initial hypotheses. R-tree bounding-box
        query is refined by exact d_cross to filter false positives.

        Args:
            fix: SensorFix — lat, lon, horizontal_accuracy used.
            n_sigma: Search radius multiplier (default 3.0).

        Returns:
            List of segment ID strings. Empty if fix is off-map.
        """
        enu_e, enu_n, _ = self._lla_to_enu(fix.lat, fix.lon, fix.alt)
        r = max(n_sigma * fix.horizontal_accuracy, self.min_search_radius_m)

        # Bounding box query on R-tree
        search_bbox = (enu_e - r, enu_n - r, enu_e + r, enu_n + r)
        candidate_ids_int = list(self._spatial_index.intersection(search_bbox))

        candidates = []
        for int_id in candidate_ids_int:
            seg_id = self._rtree_id_map[int_id]
            result = self.project_fix_to_segment(fix, seg_id)
            if result.d_cross < r:
                candidates.append(seg_id)
        if not candidates:
            logger.warning(
                f"No candidate segments found within {r:.2f}m of fix "
                f"({fix.lat:.6f}, {fix.lon:.6f})."
            )

        logger.debug(
            f"Cold start candidates | fix=({fix.lat:.6f},{fix.lon:.6f}) | "
            f"r={r:.2f}m | found={candidates}"
        )
        return candidates

    # -------------------------------------------------------------------------
    # Public API - topology queries
    # -------------------------------------------------------------------------

    def get_connected_segments(self, segment_id: str, node_id: str) -> List[str]:
        """Return all segment IDs connected to node_id, excluding segment_id.

        Undirected adjacency query. Returns segments regardless of their
        stored point order direction relative to the node. 

        Args:
            segment_id: Current segment (excluded from results).
            node_id: Node to query connections for.

        Returns:
            List of connected segment IDs (empty at isolated endpoints).
        """
        connected = []
        for u, v, data in self.graph.edges(node_id, data=True):
            cand_id = data["segment_id"]
            if cand_id != segment_id:
                connected.append(cand_id)
        return connected

    def get_node_at_boundary(self, segment_id: str, boundary: str) -> str:
        """Return node ID at the specified end of the segment.

        Args:
            segment_id: Segment to query.
            boundary: "start" (s=0, first_node) or "end" (s=length, last_node).

        Returns:
            Node ID string.

        Raises:
            ValueError: If boundary is not "start" or "end".
        """
        seg = self.segments[segment_id]
        if boundary == "start":
            return seg.first_node
        elif boundary == "end":
            return seg.last_node
        else:
            raise ValueError(
                f"boundary must be 'start' or 'end', got {boundary!r}"
            )

    def get_approach_heading(self, segment_id: str, node_id: str) -> float:
        """Return track heading at the node end of segment, oriented TOWARD the node.

        Used for angle-of-approach filtering. Computes a chord-length-weighted
        circular mean over the terminal heading_smoothing_chords chords nearest
        the node, reducing sensitivity to survey point noise at switch ends.

        Args:
            segment_id: Segment to query.
            node_id: Node being approached.

        Returns:
            Heading in radians, measured from East (ENU convention).
        """
        seg = self.segments[segment_id]
        return self._terminal_heading(seg, node_id, toward_node=True)

    def get_departure_heading(self, segment_id: str, node_id: str) -> float:
        """Return track heading at the node end of segment, oriented AWAY from the node.

        Used for angle-of-approach filtering on candidate segments. This is
        the heading the train would have if it entered this segment from node_id.
        Computes a chord-length-weighted circular mean over the terminal
        heading_smoothing_chords chords nearest the node.

        Args:
            segment_id: Candidate segment.
            node_id: Node being departed from.

        Returns:
            Heading in radians, measured from East (ENU convention).
        """
        seg = self.segments[segment_id]
        return self._terminal_heading(seg, node_id, toward_node=False)

    def _terminal_heading(
        self, seg: "TrackSegment", node_id: str, toward_node: bool
    ) -> float:
        """Compute chord-length-weighted circular mean heading at the node end of a segment.

        Takes up to heading_smoothing_chords chords nearest node_id, weights each
        by its chord length, and returns the circular mean heading. Chord direction
        is determined by node_id and toward_node:
          - toward_node=True:  chords oriented TOWARD node_id (approach heading).
          - toward_node=False: chords oriented AWAY FROM node_id (departure heading).

        node_id == seg.last_node → terminal chords are the last N chords of the
        stored point sequence; natural chord direction is first→last.
        node_id == seg.first_node → terminal chords are the first N chords;
        natural chord direction is first→last (reversed for toward_node=True).

        Circular mean avoids wrap-around averaging error at ±π.

        Args:
            seg: TrackSegment to query.
            node_id: The node at the boundary being queried.
            toward_node: True → heading oriented toward node_id; False → away from it.

        Returns:
            Weighted circular mean heading in radians, East-referenced (ENU).
        """
        pts = seg.enu_points
        n = len(pts)
        if n < 2:
            # Logging: warn about degenerate segment geometry and return zero heading.
            logger.warning(
                f"Segment {seg.id} has fewer than 2 points — cannot compute terminal heading. Returning 0.0."
            )
            return 0.0

        n_chords = min(self.heading_smoothing_chords, n - 1)

        if node_id == seg.last_node:
            # Terminal chords are the last n_chords chords (pts[n-n_chords-1] → pts[n-1])
            # Natural chord direction (first→last) points TOWARD last_node.
            sign = 1 if toward_node else -1
            chords = [
                (sign * (pts[i+1][0] - pts[i][0]),
                 sign * (pts[i+1][1] - pts[i][1]))
                for i in range(n - 1 - n_chords, n - 1)
            ]
        else:
            # node_id == seg.first_node
            # Terminal chords are the first n_chords chords (pts[0] → pts[n_chords])
            # Natural chord direction (first→last) points AWAY FROM first_node.
            sign = -1 if toward_node else 1
            chords = [
                (sign * (pts[i+1][0] - pts[i][0]),
                 sign * (pts[i+1][1] - pts[i][1]))
                for i in range(n_chords)
            ]

        # Chord-length-weighted circular mean
        sin_sum = 0.0
        cos_sum = 0.0
        for dx, dy in chords:
            length = math.hypot(dx, dy)
            if length < 1e-10:
                continue  # degenerate chord (duplicate survey point) — skip
            heading = math.atan2(dy, dx)
            sin_sum += length * math.sin(heading)
            cos_sum += length * math.cos(heading)

        if math.hypot(sin_sum, cos_sum) < 1e-10:
            # Degenerate fallback: all chords cancelled — use single terminal chord
            if node_id == seg.last_node:
                sign = 1 if toward_node else -1
                dx = sign * (pts[-1][0] - pts[-2][0])
                dy = sign * (pts[-1][1] - pts[-2][1])
            else:
                sign = -1 if toward_node else 1
                dx = sign * (pts[1][0] - pts[0][0])
                dy = sign * (pts[1][1] - pts[0][1])
            return math.atan2(dy, dx)

        return math.atan2(sin_sum, cos_sum)

    def is_switch_node(self, node_id: str) -> bool:
        """Return True if node has more than 2 segment connections.

        A node with degree 2 is a through-node (one segment in, one out).
        A node with degree 1 is an endpoint.
        A node with degree >= 3 is a switch.

        Args:
            node_id: Node to check.

        Returns:
            True if the node is a switch (undirected degree > 2).
        """
        return self.graph.degree(node_id) > 2

    def _validate_topology(self) -> None:
        """Validate the built topology graph after construction.

        Checks:
        - Every node typed "switch" in the track JSON has undirected degree >= 3.
        - No orphan nodes (degree 0).
        - No duplicate segment IDs.
        - Warns if any node has degree > 6 (unusual, possible data error).
        """
        # track segment IDs must be unique across the entire track — check for duplicates that could cause mapping errors
        seen_seg_ids: set = set()
        for seg in self.segments.values():
            if seg.id in seen_seg_ids:
                logger.error(
                    f"_validate_topology | duplicate segment ID: {seg.id}"
                )
            seen_seg_ids.add(seg.id)

        for node in self.nodes.values():
            deg = self.graph.degree(node.id)

            if deg == 0:
                logger.error(
                    f"_validate_topology | orphan node (degree 0): {node.id}"
                )

            if node.type == "switch" and deg < 3:
                logger.error(
                    f"_validate_topology | node {node.id} typed 'switch' "
                    f"but has undirected degree {deg} (expected >= 3)"
                )

            if deg > 6:
                logger.warning(
                    f"_validate_topology | node {node.id} has unusually high "
                    f"degree {deg} — possible data error in track JSON"
                )

        logger.debug(
            "_validate_topology | passed | %d segments | %d nodes",
            len(self.segments),
            len(self.nodes),
        )



    # -------------------------------------------------------------------------
    # Public API - switch proximity
    # -------------------------------------------------------------------------

    def distance_to_nearest_switch(
        self,
        segment_id: str,
        s: float,
        direction_qualifier: str = "unknown",
    ) -> float:
        """Return along-track distance from s to the nearest switch node along direction_qualifier.

        Used to gate curvature scoring: activate L_curv only when approaching
        a switch (within D_decision meters).

        "nominal": checks last_node (train moving toward s=length), distance = segment.length - s.
        "reverse": checks first_node (train moving toward s=0), distance = s.
        "unknown": checks both ends, returns minimum distance to any switch.
            Safe fallback when direction has not been established (cold start,
            near-zero velocity). May activate L_curv slightly early but never
            misses an approaching switch. Transitions to correct single-end
            measurement once direction_qualifier resolves to nominal/reverse.

        Args:
            segment_id: Current segment.
            s: Along-track position from first_node, meters.
            direction_qualifier: "nominal", "reverse", or "unknown". Derived from
                KF velocity sign with hysteresis by the OLE integration module.

        Returns:
            Distance in meters to the next switch, or math.inf if none.
        """
        seg = self.segments[segment_id]

        if direction_qualifier == "unknown":
            # Check both ends, return minimum distance to any switch
            candidates = []
            d_end = seg.length - s
            d_start = s
            if self.is_switch_node(seg.last_node):
                candidates.append(max(0.0, d_end))
            if self.is_switch_node(seg.first_node):
                candidates.append(max(0.0, d_start))
            if candidates:
                result = min(candidates)
                logger.debug(
                    "distance_to_nearest_switch | seg=%s s=%.2fm dq=unknown | "
                    "min_switch_dist=%.2fm (checked both ends)",
                    segment_id, s, result,
                )
                return result
            # Not a switch at either end — curvature scoring should not activate
            logger.debug(
                "distance_to_nearest_switch | seg=%s s=%.2fm dq=unknown | "
                "no switches at either end → returning inf",
                segment_id, s,
            )
            return math.inf

        if direction_qualifier == "nominal":
            upcoming_node = seg.last_node
            dist = seg.length - s
        else:
            # "reverse": train is moving toward first_node
            upcoming_node = seg.first_node
            dist = s

        # Check if upcoming node is a switch
        if self.is_switch_node(upcoming_node):
            return max(0.0, dist)

        # Not a switch — curvature scoring should not activate
        logger.debug(
            f"distance_to_nearest_switch | segment={segment_id} | s={s:.2f}m | "
            f"direction={direction_qualifier} | upcoming_node={upcoming_node} "
            f"is not a switch → returning inf"
        )
        return math.inf

    # -------------------------------------------------------------------------
    # Public API - output coordinate conversion
    # -------------------------------------------------------------------------

    def track_to_enu(
        self,
        segment_id: str,
        s: float,
        u: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Convert a track position (segment_id, s) to local ENU coordinates.

        Interpolates along the segment's ENU point sequence at arc-length s
        to find the exact point on the track geometry. 

        NOTE: Altitude (U) is not encoded in the 2D segment geometry. u_corrected will
         be passed if required; defaults to 0.0
        (reference surface) until vertical geometry is integrated.

        Args:
            segment_id: Segment containing the position.
            s: Along-track distance from first_node, meters. Clamped to
                [0, segment.length] — out-of-range values are safe.
            u: ENU up component, meters. Optional, defaults to 0.0.

        Returns:
            (E, N, U) tuple in local ENU meters.

        Raises:
            ValueError: If segment has no ENU points (geometry not built).
        """
        seg = self.segments[segment_id]
        pts = seg.enu_points
        n = len(pts)

        # Clamp s to valid range — KF estimates can overshoot slightly at
        # segment boundaries during predict step before rollover is handled.
        s = max(0.0, min(seg.length, s))

        if n == 0:
            raise ValueError(f"Segment {segment_id} has no ENU points.")
        if n == 1:
            return pts[0][0], pts[0][1], u

        # Walk chord sequence to find the chord containing s, then lerp
        # within that chord.
        s_acc = 0.0
        for i in range(n - 1):
            dx = pts[i+1][0] - pts[i][0]
            dy = pts[i+1][1] - pts[i][1]
            chord_len = math.hypot(dx, dy)
            s_next = s_acc + chord_len

            if s_next >= s or i == n - 2:
                # s falls within chord [P_i, P_{i+1}]; t=0 → P_i, t=1 → P_{i+1}
                t = (s - s_acc) / chord_len if chord_len > 1e-10 else 0.0
                t = max(0.0, min(1.0, t))   # numerical safety clamp
                e = pts[i][0] + t * dx
                n_ = pts[i][1] + t * dy
                return e, n_, u

            s_acc = s_next

        # Fallback — unreachable after clamping, but safe
        return pts[-1][0], pts[-1][1], u

    def enu_to_lla(
        self,
        e: float,
        n: float,
        u: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Convert local ENU coordinates to LLA (WGS84).

        NOTE: With u=0.0 (default), the returned altitude equals ref_alt.
        Pass the actual u_corrected if required for a meaningful altitude output.

        Args:
            e: ENU easting, meters.
            n: ENU northing, meters.
            u: ENU up, meters. Optional, defaults to 0.0.

        Returns:
            (lat, lon, alt) — WGS84 degrees / ellipsoidal meters.
        """
        # ENU → ECEF: P_ecef = Rᵀ · [E, N, U]ᵀ + P_ref_ecef
        enu_vec = np.array([e, n, u])
        ecef_vec = self._R_ecef_to_enu.T @ enu_vec + self._ecef_ref

        # ECEF → LLA (always_xy=True → output order: lon, lat, alt)
        lon, lat, alt = self._ecef_to_lla.transform(
            ecef_vec[0], ecef_vec[1], ecef_vec[2]
        )
        return lat, lon, alt

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def get_segment(self, segment_id: str) -> TrackSegment:
        """Return a TrackSegment by ID."""
        return self.segments[segment_id]

    def get_node(self, node_id: str) -> TrackNode:
        """Return a TrackNode by ID."""
        return self.nodes[node_id]

    def summary(self) -> str:
        """Return a human-readable summary of the loaded map."""
        lines = [
            "=== TrackMapEngine Summary ===",
            f"  Segments : {len(self.segments)}",
            f"  Nodes    : {len(self.nodes)}",
            f"  ENU ref  : ({self.ref_lat:.6f}°, {self.ref_lon:.6f}°, "
            f"{self.ref_alt:.2f}m)",
            f"  IMU z↓   : {self.imu_z_down}  |  imu_sign: {self.imu_sign:+d}",
            "",
            "  Segments:",
        ]
        for seg in self.segments.values():
            n_sw = len(seg.curvature_profile)
            lines.append(
                f"    {seg.id}: {seg.first_node}→{seg.last_node} | "
                f"length={seg.length:.2f}m | curvature_pts={n_sw}"
            )
        lines.append("")
        lines.append("  Nodes:")
        for node in self.nodes.values():
            deg = self.graph.degree(node.id)
            lines.append(
                f"    {node.id}: type={node.type} | "
                f"degree={deg}"
            )
        return "\n".join(lines)
