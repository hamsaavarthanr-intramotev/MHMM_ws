# =============================================================================
# sensor_interface.py
# Onboard Localization Engine — Sensor Interface & Data Classes

# Responsibilities:
#   - Define hardware-agnostic SensorFix dataclass consumed by all engine modules
#   - Apply GNSS antenna offset correction to align reported position with the
#     track centerline reference point (axle center)
#   - Subscribe to Certus Redis channels and assemble SensorFix from cached
#     protobuf messages

# ANTENNA OFFSET CORRECTION:
#   The GNSS antenna is physically mounted at some offset from the track
#   centerline / axle center. The engine needs the position of the axle contact
#   point, not the antenna, for accurate cross-track distance computation.
#
#   Offset is defined in the vehicle body frame (config.yaml: antenna section):
#     forward_m  - along vehicle heading (positive = front)
#     lateral_m  - perpendicular to heading (positive = left)
#     vertical_m - vertical (positive = up)
#
#   Correction uses a flat-earth approximation valid for offsets << 1 km.
#   It requires a valid heading in SensorFix; if heading is unreliable (e.g.
#   vehicle stationary), correction is skipped and a warning is logged.

# REDIS / PROTOBUF INTEGRATION:
#   CertusSubscriber subscribes to 9 Certus Redis channels from config.yaml.
#   Each channel carries exactly one protobuf message type (see certus.proto),
#   corresponding to exactly one ANPP packet ID published by the driver.
#
#   Channel --> Proto message --> ANPP source packet:
#     certus_pkt_utc_time           --> CertusTime            (Packet 21)
#     certus_pkt_lla                --> CertusLLA             (Packet 32)
#     certus_pkt_body_velocity      --> CertusBodyVelocity    (Packet 36)
#     certus_pkt_angular_velocity   --> CertusAngularVelocity (Packet 42)
#     certus_pkt_euler_orientation  --> CertusOrientation     (Packet 39)
#     certus_pkt_pos_stddev         --> CertusPositionStdDev  (Packet 24)
#     certus_pkt_euler_stddev       --> CertusEulerStdDev     (Packet 26)
#     certus_pkt_system_state       --> CertusSystemState     (Packet 20)
#     certus_pkt_satellites         --> CertusSatellites      (Packet 30)
#
#   The driver publishes raw per-packet messages. 
#   This module assembles them into SensorFix via build_sensor_fix().
#
#   Redis and protobuf dependencies are isolated to CertusSubscriber.
#   All other OLE modules consume only SensorFix. NOTE: No protobuf awareness!

# PRECISION CONTRACT:
#   lat/lon are 64-bit double throughout:
#   certus.proto CertusLLA (double) --> math.degrees() (Python float = C double 64-bit)
#   --> SensorFix.lat/lon (Python float = C double 64-bit)
# =============================================================================

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml
import dataclasses

import redis
from proto import certus_pb2

logger = logging.getLogger(__name__)

# WGS84 mean Earth radius used for flat-earth lat/lon offset conversion.
_EARTH_RADIUS_M = 6_371_000.0
_MAX_HEADING_ACCURACY_DEG = 30.0  # Threshold above which heading is considered unreliable


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SensorFix:
    """Hardware-agnostic GNSS/IMU observation.

    When hardware changes, only the parser populating this dataclass changes;
    the engine consumes SensorFix fields, not raw hardware output.

    Attributes:
        lat: WGS84 latitude, degrees. 
        lon: WGS84 longitude, degrees.
        alt: WGS84 ellipsoidal height, meters.
        heading: Degrees clockwise from true North, [0, 360).
        horizontal_accuracy: 1-sigma GNSS horizontal accuracy, meters (σ_pos).
        heading_accuracy: 1-sigma heading accuracy, degrees.
        timestamp: POSIX time, seconds (double precision).
        gyro_yaw_rate: Yaw rate, rad/s. Positive = clockwise from above when
            config.yaml imu.z_axis_down = true (Certus Z-down body frame).
        velocity: Forward speed magnitude, m/s (body-frame X).
        fix_quality: GNSS fix quality string — "RTK_FIXED", "RTK_FLOAT",
            "DGNSS", "SBAS", "SPP", or "INVALID".
    """
    lat: float
    lon: float
    alt: float
    heading: float
    horizontal_accuracy: float
    heading_accuracy: float
    timestamp: float
    gyro_yaw_rate: Optional[float] = None
    velocity: Optional[float] = None
    fix_quality: Optional[str] = None
    lat_stddev_m: Optional[float] = None
    lon_stddev_m: Optional[float] = None


@dataclass
class AntennaOffset:
    """GNSS antenna offset from the vehicle reference point in the body frame.
    The vehicle reference point is the track centerline contact point (axle
    center).

    Body frame convention:
        forward_m  — positive toward the front of the vehicle
        lateral_m  — positive toward the left of the vehicle
        vertical_m — positive upward

        NOTE: existing systems has vertical down positive; this class uses up
        positive for consistency with ENU and flat-earth math. Check vertical_m
        sign in your config.yaml and negate if necessary before populating.

    Load via AntennaOffset.from_config().
    """
    forward_m: float = 0.0
    lateral_m: float = 0.0
    vertical_m: float = 0.0

    @classmethod
    def from_config(cls, config_path: str) -> AntennaOffset:
        """Load antenna offset from config.yaml."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        ant = cfg.get("antenna", {})
        return cls(
            forward_m=float(ant.get("forward_m", 0.0)),
            lateral_m=float(ant.get("lateral_m", 0.0)),
            vertical_m=float(ant.get("vertical_m", 0.0)),
        )

    @property
    def is_zero(self) -> bool:
        return self.forward_m == 0.0 and self.lateral_m == 0.0 and self.vertical_m == 0.0


# =============================================================================
# Antenna Offset Correction
# =============================================================================

def apply_antenna_offset(fix: SensorFix, offset: AntennaOffset) -> SensorFix:
    """Correct a raw GNSS fix from antenna position to track centerline.

    Args:
        fix: Raw SensorFix as reported by the GNSS unit (antenna position).
        offset: AntennaOffset loaded from config.yaml.

    Returns:
        A new SensorFix with lat/lon/alt shifted to the axle center.
    """
    if offset.is_zero:
        return fix

    # Heading reliability check based on heading accuracy
    if fix.heading_accuracy is not None and fix.heading_accuracy > _MAX_HEADING_ACCURACY_DEG:
        logger.warning(
            "apply_antenna_offset: heading_accuracy %.2f° exceeds threshold %.1f°; "
            "heading unreliable, skipping antenna offset correction.",
            fix.heading_accuracy,
            _MAX_HEADING_ACCURACY_DEG,
        )
        return fix

    heading_rad = math.radians(fix.heading)
    sin_h = math.sin(heading_rad)
    cos_h = math.cos(heading_rad)

    # Vector from axle center to antenna, expressed in ENU (meters).
    delta_e = offset.forward_m * sin_h + offset.lateral_m * (-cos_h)
    delta_n = offset.forward_m * cos_h + offset.lateral_m * sin_h
    delta_u = offset.vertical_m

    # Convert ENU offset back to geodetic delta using flat-earth approximation.
    lat_rad = math.radians(fix.lat)
    delta_lat_deg = math.degrees(-delta_n / _EARTH_RADIUS_M)
    delta_lon_deg = math.degrees(-delta_e / (_EARTH_RADIUS_M * math.cos(lat_rad)))

    corrected = dataclasses.replace(
        fix,
        lat=fix.lat + delta_lat_deg,
        lon=fix.lon + delta_lon_deg,
        alt=fix.alt - delta_u,
    )

    logger.debug(
        "Antenna offset applied: Δlat=%.9f° Δlon=%.9f° Δalt=%.3fm "
        "(forward=%.3fm lateral=%.3fm vertical=%.3fm heading=%.1f°)",
        delta_lat_deg, delta_lon_deg, -delta_u,
        offset.forward_m, offset.lateral_m, offset.vertical_m, fix.heading,
    )
    return corrected


# =============================================================================
# GNSS Fix Type Mapping
# =============================================================================

# Maps CertusSystemState.gnss_fix_type (0–7, extracted from Packet 20
# filter_status bits 4–6) to SensorFix.fix_quality strings consumed by the
# MHMM observation likelihood scorer.
#
# Source: SDK FilterStatus.gnss_fix_type enum, an_packet_20.py GNSSFixType.
_CERTUS_FIX_TYPE_MAP = {
    0: "INVALID",       # No GNSS fix
    1: "SPP",           # 2D fix
    2: "SPP",           # 3D fix
    3: "SBAS",          # SBAS-augmented
    4: "DGNSS",         # Differential GNSS
    5: "SPP",           # PPP (Omnistar — mapped to SPP for OLE purposes)
    6: "RTK_FLOAT",     # RTK float
    7: "RTK_FIXED",     # RTK fixed
}


# =============================================================================
# CertusSubscriber — Redis protobuf subscriber
# =============================================================================

class CertusSubscriber:
    """Subscribe to Certus Redis channels and build SensorFix objects.

    Caches the latest protobuf message from each of the 9 Certus channels.
    When build_sensor_fix() is called, assembles a SensorFix from cached data,
    applies antenna offset, and returns the result.

    Architecture: one channel per ANPP packet ID. This class is responsible
    for all field combining — e.g. body velocity (Packet 36) and angular
    velocity (Packet 42) are cached separately and read together in
    build_sensor_fix(). The driver publishes them independently.

    Redis and protobuf imports are isolated here — no other OLE module
    needs to import redis or certus_pb2.

    Channel names are read from config.yaml redis section — never hardcoded.

    Usage:
        subscriber = CertusSubscriber(config_path="config.yaml")
        subscriber.start()
        fix = subscriber.build_sensor_fix()
        subscriber.stop()
    """

    def __init__(self, config_path: str):
        """Initialise subscriber with config-driven Redis connection and channels.

        Args:
            config_path: Path to config.yaml.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        redis_cfg = cfg["redis"]

        # Redis connection from config
        self._redis_host = str(redis_cfg["host"])
        self._redis_port = int(redis_cfg["port"])
        self._redis_db   = int(redis_cfg.get("db", 0))

        # Each key maps directly to one ANPP packet ID and one proto message.
        self._ch_utc_time    = str(redis_cfg["subscribe_channel_certus_utc_time"])
        self._ch_lla         = str(redis_cfg["subscribe_channel_certus_lla"])
        self._ch_body_vel    = str(redis_cfg["subscribe_channel_certus_body_velocity"])
        self._ch_angular_vel = str(redis_cfg["subscribe_channel_certus_angular_velocity"])
        self._ch_orientation = str(redis_cfg["subscribe_channel_certus_euler_orientation"])
        self._ch_pos_stddev  = str(redis_cfg["subscribe_channel_certus_pos_stddev"])
        self._ch_euler_stddev= str(redis_cfg["subscribe_channel_certus_euler_stddev"])
        self._ch_system_state= str(redis_cfg["subscribe_channel_certus_system_state"])
        self._ch_satellites  = str(redis_cfg["subscribe_channel_certus_satellites"])

        # Antenna offset
        self._antenna_offset = AntennaOffset.from_config(config_path)

        # Redis client and pubsub handle
        self._certus_pb2 = certus_pb2
        self._redis_client = redis.Redis(
            host=self._redis_host,
            port=self._redis_port,
            db=self._redis_db,
            decode_responses=False,
        )
        self._pubsub = self._redis_client.pubsub()

        # Cached latest messages — None until first received
        # Named to match their source ANPP packet and proto message type.
        self._latest_time         = None    # CertusTime          (Packet 21)
        self._latest_lla          = None    # CertusLLA           (Packet 32)
        self._latest_body_vel     = None    # CertusBodyVelocity  (Packet 36)
        self._latest_angular_vel  = None    # CertusAngularVelocity (Packet 42)
        self._latest_orientation  = None    # CertusOrientation   (Packet 39)
        self._latest_pos_stddev   = None    # CertusPositionStdDev (Packet 24)
        self._latest_euler_stddev = None    # CertusEulerStdDev   (Packet 26)
        self._latest_system_state = None    # CertusSystemState   (Packet 20)
        self._latest_satellites   = None    # CertusSatellites    (Packet 30)

        # Pubsub thread handle and running flag
        self._is_running    = False
        self._pubsub_thread = None
        # Fix callback for event-driven processing. Registered by OLE integration module.
        self._on_fix_ready_cb: Optional[Callable[[SensorFix], None]] = None

        logger.info(
            "CertusSubscriber init | redis=%s:%d/%d "
            "channels=[time=%s, lla=%s, body_vel=%s, angular_vel=%s, "
            "orient=%s, pos_std=%s, euler_std=%s, sys_state=%s, sats=%s] "
            "antenna_offset=%s",
            self._redis_host, self._redis_port, self._redis_db,
            self._ch_utc_time, self._ch_lla, self._ch_body_vel,
            self._ch_angular_vel, self._ch_orientation,
            self._ch_pos_stddev, self._ch_euler_stddev,
            self._ch_system_state, self._ch_satellites,
            "zero" if self._antenna_offset.is_zero else "applied",
        )

    def start(self) -> None:
        """Subscribe to all Certus channels and begin background listening."""
        self._pubsub.subscribe(**{
            self._ch_utc_time:    self._on_utc_time,
            self._ch_lla:         self._on_lla,
            self._ch_body_vel:    self._on_body_velocity,
            self._ch_angular_vel: self._on_angular_velocity,
            self._ch_orientation: self._on_orientation,
            self._ch_pos_stddev:  self._on_pos_stddev,
            self._ch_euler_stddev:self._on_euler_stddev,
            self._ch_system_state:self._on_system_state,
            self._ch_satellites:  self._on_satellites,
        })
        self._pubsub_thread = self._pubsub.run_in_thread(sleep_time=0.001, daemon=True)
        self._is_running = True
        logger.info("CertusSubscriber started | subscribed to 9 channels")

    def stop(self) -> None:
        """Unsubscribe and stop the background listener."""
        if self._is_running and self._pubsub_thread is not None:
            self._pubsub_thread.stop()
            self._pubsub.unsubscribe()
            self._is_running = False
            logger.info("CertusSubscriber stopped")

    def register_fix_callback(self, callback: Callable[[SensorFix], None]) -> None:
        """Register a callback invoked with each fully assembled SensorFix.

        Called from the pubsub thread after _on_lla triggers build_sensor_fix()
        and returns a non-None result. OLE registers its _on_fix_ready method here
        during __init__ to implement event-driven processing.

        Args:
            callback: Callable accepting a single SensorFix argument.
        """
        self._on_fix_ready_cb = callback
        logger.info("CertusSubscriber | fix callback registered: %s", callback.__qualname__)

    # -------------------------------------------------------------------------
    # Redis message handlers — one per channel, one per protobuf type
    # -------------------------------------------------------------------------

    def _on_utc_time(self, message: dict) -> None:
        """Handle certus_pkt_utc_time --> CertusTime (Packet 21)."""
        if message["type"] != "message":
            return
        msg = self._certus_pb2.CertusTime()
        msg.ParseFromString(message["data"])
        self._latest_time = msg

    def _on_lla(self, message: dict) -> None:
        """Handle certus_pkt_lla --> CertusLLA (Packet 32)."""
        if message["type"] != "message":
            return
        msg = self._certus_pb2.CertusLLA()
        msg.ParseFromString(message["data"])
        self._latest_lla = msg

        # Event-driven trigger: attempt to assemble a complete SensorFix and
        # invoke the registered callback. build_sensor_fix() returns None if
        # any required field (time, lla, orientation) is not yet cached.
        if self._on_fix_ready_cb is not None:
            fix = self.build_sensor_fix()
            if fix is not None:
                self._on_fix_ready_cb(fix)
            else:
                logger.debug(
                    "build_sensor_fix returned None — waiting for more data | "
                    "time=%s lla=%s orientation=%s",
                    self._latest_time        is not None,
                    self._latest_lla         is not None,
                    self._latest_orientation is not None,
                )

    def _on_body_velocity(self, message: dict) -> None:
        """Handle certus_pkt_body_velocity --> CertusBodyVelocity (Packet 36).

        Provides forward speed (velocity_x_mps) for SensorFix.velocity and
        antenna offset heading reliability check.
        """
        if message["type"] != "message":
            return
        msg = self._certus_pb2.CertusBodyVelocity()
        msg.ParseFromString(message["data"])
        self._latest_body_vel = msg

    def _on_angular_velocity(self, message: dict) -> None:
        """Handle certus_pkt_angular_velocity --> CertusAngularVelocity (Packet 42).

        Provides yaw rate (angular_velocity_z_rps) for SensorFix.gyro_yaw_rate.
        Z-axis is down (NED body frame) — positive yaw = clockwise from above.
        """
        if message["type"] != "message":
            return
        msg = self._certus_pb2.CertusAngularVelocity()
        msg.ParseFromString(message["data"])
        self._latest_angular_vel = msg

    def _on_orientation(self, message: dict) -> None:
        """Handle certus_pkt_euler_orientation --> CertusOrientation (Packet 39)."""
        if message["type"] != "message":
            return
        msg = self._certus_pb2.CertusOrientation()
        msg.ParseFromString(message["data"])
        self._latest_orientation = msg

    def _on_pos_stddev(self, message: dict) -> None:
        """Handle certus_pkt_pos_stddev --> CertusPositionStdDev (Packet 24).

        Provides lat/lon/height 1-sigma values.
        build_sensor_fix() computes horizontal_accuracy = sqrt(lat² + lon²).
        """
        if message["type"] != "message":
            return
        msg = self._certus_pb2.CertusPositionStdDev()
        msg.ParseFromString(message["data"])
        self._latest_pos_stddev = msg

    def _on_euler_stddev(self, message: dict) -> None:
        """Handle certus_pkt_euler_stddev --> CertusEulerStdDev (Packet 26).

        Provides heading_stddev_rad for SensorFix.heading_accuracy (converted
        to degrees in build_sensor_fix()).
        """
        if message["type"] != "message":
            return
        msg = self._certus_pb2.CertusEulerStdDev()
        msg.ParseFromString(message["data"])
        self._latest_euler_stddev = msg

    def _on_system_state(self, message: dict) -> None:
        """Handle certus_pkt_system_state --> CertusSystemState (Packet 20).

        Provides gnss_fix_type (extracted from filter_status bits 4–6 by the
        driver) for SensorFix.fix_quality via _CERTUS_FIX_TYPE_MAP.
        """
        if message["type"] != "message":
            return
        msg = self._certus_pb2.CertusSystemState()
        msg.ParseFromString(message["data"])
        self._latest_system_state = msg

    def _on_satellites(self, message: dict) -> None:
        """Handle certus_pkt_satellites --> CertusSatellites (Packet 30).

        Provides num_satellites (sum across all constellations). Not currently
        consumed by build_sensor_fix() — available for diagnostics and future
        use (e.g. satellite-count gating in observation likelihood).
        """
        if message["type"] != "message":
            return
        msg = self._certus_pb2.CertusSatellites()
        msg.ParseFromString(message["data"])
        self._latest_satellites = msg

    # -------------------------------------------------------------------------
    # SensorFix assembly
    # -------------------------------------------------------------------------

    def build_sensor_fix(self) -> Optional[SensorFix]:
        """Assemble a SensorFix from the latest cached Certus messages.

        REQUIRED fields (returns None if any are missing):
            CertusTime         — provides timestamp
            CertusLLA          — provides lat, lon, alt
            CertusOrientation  — provides heading

        OPTIONAL fields (use defaults or None if not yet received):
            CertusBodyVelocity    — provides velocity (forward speed)
            CertusAngularVelocity — provides gyro_yaw_rate
            CertusPositionStdDev  — provides lat/lon stddev → horizontal_accuracy
            CertusEulerStdDev     — provides heading stddev → heading_accuracy
            CertusSystemState     — provides gnss_fix_type → fix_quality
            CertusSatellites      — not consumed here (available for diagnostics)

        UNIT CONVERSIONS (all 64-bit safe):
            lat/lon:          radians (double) → degrees  (math.degrees, 64-bit)
            heading:          radians → degrees [0, 360)
            heading_stddev:   radians → degrees
            horizontal_accuracy: sqrt(lat_stddev_m² + lon_stddev_m²) meters
            velocity:         body-frame X (m/s) — forward speed
            gyro_yaw_rate:    angular_velocity_z_rps (rad/s) — Z-down body frame

        Returns:
            SensorFix with antenna offset applied, or None if required data
            has not yet been received on all three mandatory channels.
        """
        if (self._latest_time is None
                or self._latest_lla is None
                or self._latest_orientation is None):
            logger.debug(
                "build_sensor_fix | insufficient data — "
                "time=%s lla=%s orientation=%s",
                self._latest_time        is not None,
                self._latest_lla         is not None,
                self._latest_orientation is not None,
            )
            return None

        # Timestamp: Packet 21
        # Unix seconds + microseconds → POSIX float (64-bit throughout)
        timestamp = (float(self._latest_time.unix_time_s)
                     + float(self._latest_time.microseconds) * 1e-6)

        # Position: Packet 32
        # Radians (double) → degrees (Python float = C double, 64-bit)
        lat_deg = math.degrees(self._latest_lla.latitude_rad)
        lon_deg = math.degrees(self._latest_lla.longitude_rad)
        alt_m   = self._latest_lla.height_m

        # Heading: Packet 39
        # Radians → degrees [0, 360)
        heading_deg = math.degrees(self._latest_orientation.heading_rad) % 360.0

        # Horizontal accuracy: Packet 24
        # sqrt(lat_stddev² + lon_stddev²) — 2D position uncertainty, meters.
        # Also preserve individual axis stddevs for directional sigma projection.
        horizontal_accuracy = 0.0
        lat_stddev_m: Optional[float] = None
        lon_stddev_m: Optional[float] = None
        if self._latest_pos_stddev is not None:
            lat_stddev_m = float(self._latest_pos_stddev.lat_stddev_m)
            lon_stddev_m = float(self._latest_pos_stddev.lon_stddev_m)
            horizontal_accuracy = math.sqrt(lat_stddev_m ** 2 + lon_stddev_m ** 2)

        # Heading accuracy: Packet 26
        # Radians → degrees
        heading_accuracy_deg = 0.0
        if self._latest_euler_stddev is not None:
            heading_accuracy_deg = math.degrees(
                self._latest_euler_stddev.heading_stddev_rad
            )

        # Forward velocity: Packet 36
        # Body-frame X = forward speed magnitude, m/s
        velocity = None
        if self._latest_body_vel is not None:
            velocity = float(self._latest_body_vel.velocity_x_mps)

        # Gyro yaw rate: Packet 42
        # angular_velocity_z_rps --> Z-down body frame, rad/s
        # Positive = clockwise from above (matches imu.z_axis_down = true)
        gyro_yaw_rate = None
        if self._latest_angular_vel is not None:
            gyro_yaw_rate = float(self._latest_angular_vel.angular_velocity_z_rps)

        # Fix quality: Packet 20
        # gnss_fix_type (0–7) mapped to OLE quality string
        fix_quality = None
        if self._latest_system_state is not None:
            fix_quality = _CERTUS_FIX_TYPE_MAP.get(
                self._latest_system_state.gnss_fix_type, "INVALID"
            )

        # Build raw SensorFix (antenna position, not yet corrected)
        raw_fix = SensorFix(
            lat=lat_deg,
            lon=lon_deg,
            alt=alt_m,
            heading=heading_deg,
            horizontal_accuracy=horizontal_accuracy,
            heading_accuracy=heading_accuracy_deg,
            timestamp=timestamp,
            gyro_yaw_rate=gyro_yaw_rate,
            velocity=velocity,
            fix_quality=fix_quality,
            lat_stddev_m=lat_stddev_m,
            lon_stddev_m=lon_stddev_m,
        )

        # Apply antenna offset correction
        corrected_fix = apply_antenna_offset(raw_fix, self._antenna_offset)

        logger.debug(
            "build_sensor_fix | lat=%.9f° lon=%.9f° alt=%.2fm "
            "heading=%.1f° h_acc=%.3fm (lat_std=%.4fm lon_std=%.4fm) heading_acc=%.1f° "
            "gyro_z=%.4f rad/s vel=%.2f m/s fix=%s t=%.6f",
            corrected_fix.lat, corrected_fix.lon, corrected_fix.alt,
            corrected_fix.heading, corrected_fix.horizontal_accuracy,
            corrected_fix.lat_stddev_m or 0.0,
            corrected_fix.lon_stddev_m or 0.0,
            corrected_fix.heading_accuracy or 0.0,
            corrected_fix.gyro_yaw_rate or 0.0,
            corrected_fix.velocity or 0.0,
            corrected_fix.fix_quality or "N/A",
            corrected_fix.timestamp,
        )

        return corrected_fix
    

    # -------------------------------------------------------------------------
    # Public API - Number of satellites getter
    # -------------------------------------------------------------------------
    @property
    def num_satellites(self) -> Optional[int]:
        """Get the latest satellite count from CertusSatellites (Packet 30).
        Used in OLE integration module for final output.

        Returns:
            Number of satellites in view, or None if not yet received.
        """
        if self._latest_satellites is not None:
            return int(self._latest_satellites.num_satellites)
        return None
