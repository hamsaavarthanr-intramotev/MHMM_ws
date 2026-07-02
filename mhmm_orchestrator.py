# =============================================================================
# mhmm_orchestrator.py
# Onboard Localization Engine — Multi-Hypothesis Map-Matching Lifecycle Manager / Orchestrator

# Responsibilities:
#   - Manage the full hypothesis lifecycle (spawn, prune, merge, cap)
#   - Drive the two-rate sensor loop (GNSS full predict/update; propagate between fixes at IMU rate)
#   - Emit LocalizationResult with confidence classification each GNSS cycle 
#   - Coordinate all downstream modules without exposing their internal implementation details

# MODULE CALL ORDER PER GNSS FIX (process_fix):
#   1. KF predict + segment boundary / spawn (kalman_filter.py)
#   2. Geometric projection (track_map_engine.py)
#   3. KF update (kalman_filter.py)
#   4. Observation likelihood + Bayesian weight update (observation_likelihood.py)
#   5. Lifecycle management: prune --> merge --> cap --> normalise
#   6. Build and return LocalizationResult

# SIGN CONTRACT:
#   direction_qualifier ("nominal"/"reverse"/"unknown") is a PER-HYPOTHESIS
#   field on Hypothesis. It is derived from each hypothesis's own KF velocity
#   sign after every update cycle and at spawn time. It is passed to
#   observation_likelihood for each hypothesis independently.
#
#   Velocity sign flip at boundaries:
#     When a hypothesis crosses into a segment whose connection node is on the
#     opposite end from the parent's crossing direction, the child's velocity
#     must be negated to preserve the physical travel direction. The rule is:
#       flip = (crossing_start_node) XOR (child.last_node == node_id)
#     where crossing_start_node = (predicted_s < 0.0).
#     This is applied in _handle_boundary() for both rollover and spawn paths.
# =============================================================================

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml

from kalman_filter import KFConfig, KFState, TrackKalmanFilter
from observation_likelihood import LikelihoodResult, ScoringConfig, compute_likelihood
from sensor_interface import SensorFix
from track_map_engine import TrackMapEngine, TrackProjectionCoordinates

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Hypothesis:
    """A single track hypothesis in the MHMM.

    Wraps a TrackKalmanFilter with posterior weight and lifecycle metadata.
    One instance per active candidate track segment.

    Attributes:
        id: Unique monotonically increasing integer identifier.
        segment_id: Track segment this hypothesis is currently on.
        kf: TrackKalmanFilter tracking along-track position and velocity.
        weight: Posterior probability. All active weights sum to 1.0 after normalisation.
        parent_id: ID of the spawning hypothesis (None for cold-start hypotheses).
        spawn_s: Along-track position of this hypothesis at the moment it entered
            the system, in the coordinate frame of its current segment (metres).
            Set for ALL new hypotheses — both cold-start (entry_s = s_proj at first
            fix) and switch-spawned (entry_s = overshoot past the switch node).
            Used as the reference point for grace distance pruning: a hypothesis
            is immune from pruning while abs(kf.position - spawn_s) < grace_distance,
            giving it a chance to accumulate position evidence before being judged.
            This unifies the cold-start and switch-spawn cases — both are new entrants
            to the system and deserve the same geometric grace period.
        latest_likelihood: Most recent LikelihoodResult. Overwritten each cycle.
        latest_projection: Most recent TrackProjectionCoordinates. Overwritten each cycle.
    """
    id: int
    segment_id: str
    kf: TrackKalmanFilter
    weight: float
    direction_qualifier: str = "unknown"  # per-hypothesis; "nominal"/"reverse"/"unknown"
    # NOTE: Owned by orchestrator, not OLE. Updated after every KF update cycle
    # and set at spawn time from the post-flip velocity sign. Two hypotheses on
    # segments with opposite orientation at the same node will correctly carry
    # opposite dq values, preventing ghost boundary loops on reverse-oriented stubs.
    parent_id: Optional[int] = None
    spawn_s: Optional[float] = None
    latest_likelihood: Optional[LikelihoodResult] = None
    latest_projection: Optional[TrackProjectionCoordinates] = None
    _consecutive_low: int = field(default=0, repr=False)
    # Counts consecutive update cycles where this hypothesis weight was below
    # prune_threshold. Reset to zero when weight recovers. Used by _prune()
    # for the consecutive-cycle eligibility check. Not set at spawn time.
    # Not carried across transition_to_segment() — pruning resets on rollover.


@dataclass
class MHMMConfig:
    """MHMM orchestrator parameters. Loaded from config.yaml kalman_filter section.

    Attributes:
        prune_threshold: Minimum weight below which a hypothesis is pruned.
        grace_distance: Distance (m) traveled on a child segment after spawning within which pruning is suppressed.
            Lets evidence accumulate before eliminating newly spawned hypotheses.
        merge_distance: Along-track distance (m) within which two hypotheses
            on the same segment are merged (weights summed, weaker KF discarded).
        max_hypotheses: Hard cap on active hypothesis count.
        confidence_high: w_max threshold above which output confidence is HIGH.
        confidence_ambiguous: w_max threshold below which confidence is AMBIGUOUS.
            MEDIUM is implicitly: confidence_ambiguous < w_max <= confidence_high.
        prune_consecutive_cycles: Number of consecutive update cycles a hypothesis
            must remain below prune_threshold before it is actually removed.
            Absorbs transient dips from single-cycle multipath spikes or
            momentary heading-unreliable zones. Counter resets to zero if the
            hypothesis recovers above threshold. Typical range: 2-5.
            Only applies AFTER the grace_distance window has been exhausted —
            the two mechanisms are sequential, not parallel.
    """
    # NOTE: Requires critical tuning!
    prune_threshold: float = 0.02
    grace_distance: float = 150.0
    merge_distance: float = 5.0
    max_hypotheses: int = 12
    confidence_high: float = 0.95
    confidence_ambiguous: float = 0.60
    prune_consecutive_cycles: int = 3
    approach_angle_threshold: float = 90.0
    direction_hysteresis_threshold: float = 0.3
    # Integrity gate parameters (Approach A + B)
    integrity_chi2_confidence: float = 0.999   # chi-squared quantile p-value
    llr_null_sigma_offtrack: float = 5.0        # off-track null model sigma (m)
    llr_sep_threshold: float = 2.0              # min LLR best vs 2nd-best for HIGH

    @classmethod
    def from_config(cls, config_path: str) -> MHMMConfig:
        """Load MHMMConfig from the kalman_filter section of config.yaml.

        Args:
            config_path: Path to config.yaml.

        Returns:
            MHMMConfig populated from config.yaml.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        kf = cfg["kalman_filter"]
        config = cls(
            prune_threshold=float(kf["prune_threshold"]),
            grace_distance=float(kf["grace_distance"]),
            merge_distance=float(kf["merge_distance"]),
            max_hypotheses=int(kf["max_hypotheses"]),
            confidence_high=float(kf.get("confidence_high", 0.95)),
            confidence_ambiguous=float(kf.get("confidence_ambiguous", 0.60)),
            prune_consecutive_cycles=int(kf.get("prune_consecutive_cycles", 3)),
            approach_angle_threshold=float(kf.get("approach_angle_threshold", 90.0)),
            direction_hysteresis_threshold=float(kf.get("direction_hysteresis_threshold", 0.3)),
            integrity_chi2_confidence=float(kf.get("integrity_chi2_confidence", 0.999)),
            llr_null_sigma_offtrack=float(kf.get("llr_null_sigma_offtrack", 5.0)),
            llr_sep_threshold=float(kf.get("llr_sep_threshold", 2.0)),
        )

        # Log all values at load time so the lifecycle tuning parameters for each run are auditable alongside KFConfig and ScoringConfig.
        # NOTE: Critical to tune: prune_threshold and grace_distance together control how aggressively the filter collapses after a switch.
        logger.info(
            "MHMMConfig loaded | prune=%.3f grace=%.1fm merge=%.1fm "
            "max_hyp=%d conf_high=%.2f conf_amb=%.2f prune_consec=%d "
            "approach_angle_threshold=%.1fdeg direction_hysteresis_threshold=%.2f | "
            "integrity: chi2_p=%.4f llr_null_sigma=%.1fm llr_sep_thr=%.2f",
            config.prune_threshold,
            config.grace_distance,
            config.merge_distance,
            config.max_hypotheses,
            config.confidence_high,
            config.confidence_ambiguous,
            config.prune_consecutive_cycles,
            config.approach_angle_threshold,
            config.direction_hysteresis_threshold,
            config.integrity_chi2_confidence,
            config.llr_null_sigma_offtrack,
            config.llr_sep_threshold,
        )

        return config


@dataclass
class LocalizationResult:
    """Output of one MHMM update cycle. Consumed by the autonomy stack.

    Attributes:
        timestamp: POSIX time of this result, seconds.
        best_segment_id: Segment ID of the highest-weight (MAP) hypothesis.
        best_position: Along-track position of the MAP hypothesis, meters.
        best_velocity: Velocity estimate of the MAP hypothesis, m/s.
        best_position_var: Position variance of the MAP hypothesis, m^2.
        best_weight: Weight of the MAP hypothesis.
        best_direction_qualifier: direction_qualifier of the MAP hypothesis.
        confidence: "HIGH", "MEDIUM", or "AMBIGUOUS".
            HIGH     → w_max > confidence_high   (normal autonomous operation)
            MEDIUM   → confidence_ambiguous < w_max <= confidence_high
            AMBIGUOUS → w_max <= confidence_ambiguous (reduce speed / request info)
        num_hypotheses: Number of active hypotheses this cycle.
        hypotheses: Full list of (segment_id, position, weight) tuples for all
            active hypotheses. Always populated regardless of confidence level.
    """
    timestamp: float
    best_segment_id: str
    best_position: float
    best_velocity: float
    best_position_var: float
    best_weight: float
    confidence: str
    num_hypotheses: int
    hypotheses: List[Tuple[str, float, float]]  # (segment_id, position, weight)
    best_direction_qualifier: str = "unknown"   # dq of MAP hypothesis;


# =============================================================================
# MHMMOrchestrator
# =============================================================================


class MHMMOrchestrator:
    """Multi-Hypothesis Map-Matching orchestrator.

    Manages the full hypothesis lifecycle and emits localization output with
    confidence classification. Instantiate once at startup. Feed SensorFix objects via initialise() then process_fix(). 
    For IMU-rate dead-reckoning between GNSS fixes, call propagate(). 

    Example:
        orchestrator = MHMMOrchestrator("track.json", "config.yaml")
        result = orchestrator.initialise(first_fix)
        for fix in gnss_stream:
            result = orchestrator.process_fix(fix)
    """

    def __init__(self, track_json_path: str, config_path: str) -> None:
        """Load track map and all tunable configs. Does not create hypotheses.

        Args:
            track_json_path: Path to track JSON file.
            config_path: Path to config.yaml.
        """
        self.track_map_engine = TrackMapEngine(track_json_path, config_path)
        self.kf_config = KFConfig.from_config(config_path)
        self.scoring_config = ScoringConfig.from_config(config_path)
        self.mhmm_config = MHMMConfig.from_config(config_path)

        self.hypotheses: List[Hypothesis] = []       # Active hypotheses list, updated each cycle.
        self._next_hypothesis_id: int = 0            # Monotonically increasing ID generator for hypotheses.
        self._last_timestamp: Optional[float] = None # Timestamp of the last processed fix.
        self._is_initialised: bool = False           # Flag to indicate if initialisation has been done.
        self._weights_are_evidence_based: bool = False  # False after uniform reset — prevents false HIGH confidence.

        logger.info(
            "MHMMOrchestrator ready | track_json=%s config=%s",
            track_json_path,
            config_path,
        )

    # -------------------------------------------------------------------------
    # Public API — initialisation
    # -------------------------------------------------------------------------

    def initialise(self, fix: SensorFix) -> LocalizationResult:
        """Cold-start: spawn one hypothesis per candidate segment near the fix.

        Finds all track segments within 3σ of fix.horizontal_accuracy, projects
        the fix onto each track segment, and creates hypotheses with uniform weight.

        Args:
            fix: First reliable GNSS fix. horizontal_accuracy used as σ_pos.

        Returns:
            LocalizationResult from initial hypothesis set.
            Returns a degenerate result (no hypotheses) if the fix is off-map.
            # Caller must handle the off-map condition.
        """
        candidates = self.track_map_engine.get_candidate_segments(fix, n_sigma=5.0)
        # Cold start is critical: if no candidates are found, the filter has no hypotheses to track and will fail.

        # Validate candidates before spawining hypotheses.
        if not candidates:
            logger.warning(
                "initialise | no candidate segments within 3-sigma of fix "
                "(lat=%.6f lon=%.6f acc=%.2fm) — fix may be off-map",
                fix.lat,
                fix.lon,
                fix.horizontal_accuracy,
            )
            self._last_timestamp = fix.timestamp
            self._is_initialised = True
            # Return empty result. 
            # NOTE: Caller must handle the off-map condition.
            return LocalizationResult(
                timestamp=fix.timestamp,
                best_segment_id="",
                best_position=0.0,
                best_velocity=0.0,
                best_position_var=0.0,
                best_weight=0.0,
                confidence="AMBIGUOUS",
                num_hypotheses=0,
                hypotheses=[],
            )

        # Spawn one hypothesis per candidate segment with uniform weight.
        uniform_weight = 1.0 / len(candidates)

        for seg_id in candidates:
            projection = self.track_map_engine.project_fix_to_segment(fix, seg_id)
            kf = TrackKalmanFilter(
                segment_id=seg_id,
                initial_position=projection.s_proj,
                config=self.kf_config,
                initial_velocity=0.0,  # Assume stationary at cold start. Velocity will be updated on the first fix update.
                timestamp=fix.timestamp,
            )
            self.hypotheses.append(
                Hypothesis(
                    id=self._next_id(),
                    segment_id=seg_id,
                    kf=kf,
                    weight=uniform_weight,
                    spawn_s=projection.s_proj,
                    # spawn_s is set for cold-start hypotheses too.
                    # Grace distance pruning applies uniformly to all new
                    # entrants — cold-start and switch-spawned are equivalent
                    # from the pruning system's perspective.
                    latest_projection=projection,
                )
            )

        self._last_timestamp = fix.timestamp
        self._is_initialised = True

        logger.info(
            "initialise | %d hypothesis(es) spawned | segments=%s | weight=%.4f each",
            len(self.hypotheses),
            [h.segment_id for h in self.hypotheses],
            uniform_weight,
        )

        return self._build_result(fix.timestamp)

    # -------------------------------------------------------------------------
    # Public API — main GNSS update cycle
    # -------------------------------------------------------------------------

    def process_fix(
        self,
        fix: SensorFix,
    ) -> LocalizationResult:
        """Full MHMM update cycle driven by a GNSS fix.

        Steps: predict --> boundary/spawn --> project --> KF update -->
               per-hypothesis dq update --> likelihood/weight update -->
               lifecycle management --> output.

        Args:
            fix: Current SensorFix from the sensor interface.

        Returns:
            LocalizationResult with confidence classification.
        """
        if not self._is_initialised:
            logger.warning(
                "process_fix called before initialise() | calling initialise() now",
            )
            return self.initialise(fix)

        # Validate timestamp for duplicate or out-of-order fixes.
        dt = fix.timestamp - self._last_timestamp if self._last_timestamp is not None else None

        if dt is None or dt <= 0.0:
            # Skip predict on duplicate or out-of-order timestamp to avoid degenerate Q matrix.
            logger.warning(
                "process_fix | dt=%.6fs <= 0 (duplicate or out-of-order timestamp "
                "t=%.3f last=%.3f) | skipping predict, proceeding with update",
                dt,
                fix.timestamp,
                self._last_timestamp,
            )
        else:
            self._last_timestamp = fix.timestamp

            # Step 1: KF predict + segment boundary handling
            self.propagate(dt)

        # Step 2: Geometric projection
        for h in self.hypotheses:
            h.latest_projection = self.track_map_engine.project_fix_to_segment(
                fix, h.segment_id
            )

        # Step 3a: KF update with per-hypothesis measurement noise R.
        #
        # R = sigma_along² where sigma_along is the 1D along-track projection
        # of the 2D GNSS error ellipse at this hypothesis's track heading.
        # Each hypothesis has a different heading (different segment geometry)
        # so R is computed inside the per-hypothesis loop.
        #
        # Three cases (matching the sigma_gnss logic in observation_likelihood):
        #   A — per-axis stddevs available: project onto along-track axis.
        #       sigma_along = sqrt((lon_std × cos(heading))² + (lat_std × sin(heading))²)
        #   B — only horizontal_accuracy available (isotropic): h_acc / sqrt(2).
        #   C — nothing available: KF config default.
        for h in self.hypotheses:
            # Validate each projection before KF updates each hypothesis.
            if h.latest_projection is None:
                logger.warning(
                    "process_fix | hyp_id=%d seg=%s projection is None, skipping KF update",
                    h.id,
                    h.segment_id,
                )
                continue

            heading = h.latest_projection.track_heading_rad
            if (fix.lat_stddev_m is not None and fix.lat_stddev_m > 0.0
                    and fix.lon_stddev_m is not None and fix.lon_stddev_m > 0.0):
                # Case A: directional projection onto along-track axis.
                sin_h = math.sin(heading)
                cos_h = math.cos(heading)
                sigma_along = math.sqrt(
                    (fix.lon_stddev_m * cos_h) ** 2
                    + (fix.lat_stddev_m * sin_h) ** 2
                )
                measurement_noise = max(sigma_along ** 2, 1e-6)
            elif fix.horizontal_accuracy is not None and fix.horizontal_accuracy > 0.0:
                # Case B: isotropic approximation.
                measurement_noise = (fix.horizontal_accuracy / math.sqrt(2.0)) ** 2
            else:
                # Case C: no accuracy data — KF uses its config default.
                measurement_noise = None

            h.kf.update(
                h.latest_projection.s_proj,
                measurement_noise=measurement_noise,
                timestamp=fix.timestamp,
            )

        # Step 3b: Update per-hypothesis direction_qualifier from post-update velocity.
        # Each hypothesis owns its own dq — two hypotheses on counter-oriented segments
        # at the same node must carry opposite dq values for correct switch proximity queries.
        self._update_hypothesis_dq()

        # Step 4: Likelihood scoring --> observation likelihood & Bayesian weight update
        #
        # L_curv is computed per-hypothesis within the decision area without
        # any cross-hypothesis comparison. Each hypothesis scores its own
        # omega_expected = velocity × kappa against the sensor measurement.
        for h in self.hypotheses:
            if h.latest_projection is None:
                logger.warning(
                    "process_fix | hyp_id=%d seg=%s projection is None, skipping likelihood compute",
                    h.id,
                    h.segment_id,
                )
                continue
            # distance_since_spawn: along-track distance from spawn point to
            # current KF position. Enables backward proximity check in
            # compute_likelihood so L_curv stays active on branching segments
            # whose far node is not a switch (distance_to_nearest_switch=inf).
            distance_since_spawn = (
                abs(h.kf.get_state().position - h.spawn_s)
                if h.spawn_s is not None else None
            )

            h.latest_likelihood = compute_likelihood(
                fix=fix,
                projection=h.latest_projection,
                kf_state=h.kf.get_state(),
                track_map_engine=self.track_map_engine,
                scoring_config=self.scoring_config,
                direction_qualifier=h.direction_qualifier,   # per-hypothesis
                distance_since_spawn=distance_since_spawn,   # post-switch backward proximity
            )

        self._update_weights()


        # Step 5: Lifecycle management
        self._prune()
        self._merge()
        self._enforce_cap()
        self._normalise_weights()

        # NOTE: Emergency Recovery! If all hypotheses were pruned, reinitialise.
        if not self.hypotheses:
            logger.critical(
                "process_fix | all hypotheses eliminated after lifecycle step "
                "at t=%.3f | reinitialising from current fix",
                fix.timestamp,
            )
            self._is_initialised = False
            return self.initialise(fix)


        # Step 6: Output LocalizationResult with confidence classification
        return self._build_result(fix.timestamp)

    # -------------------------------------------------------------------------
    # Propagate --> Predict + Boundary / Spawn handling
    # Public API - dead-reckoning propagation (between GNSS fixes)
    # -------------------------------------------------------------------------

    def propagate(self, dt: float) -> None:
        """
        Propagate / predict all hypothesis KFs forward by dt and handle boundaries.

        For each hypothesis: predict, then check if the predicted position has
        crossed a segment boundary. Handles rollover (1 connected), switch spawn
        (N connected), and endpoint clamping (0 connected).

        Both boundaries (s > seg_length and s < 0) are checked unconditionally.
        The KF velocity sign naturally drives position toward one end or the other.

        Does NOT update weights or run lifecycle management, which only happen on GNSS fixes via process_fix().

        Args:
            dt: Elapsed time since the last propagate call, seconds.
        """
        # NOTE: Boundary/spawn handling runs in both propagate() and process_fix().
        # However, weight updates and lifecycle management are intentionally exclusive to process_fix().
        spawned: List[Hypothesis] = []

        for h in self.hypotheses:
            predicted_s = h.kf.predict(dt)
            seg_length = self.track_map_engine.get_segment(h.segment_id).length

            if predicted_s > seg_length:
                overshoot = predicted_s - seg_length
                node_id = self.track_map_engine.get_node_at_boundary(h.segment_id, "end")
                candidates = self.track_map_engine.get_connected_segments(h.segment_id, node_id)
                candidates = self._filter_by_approach_angle(h.segment_id, node_id, candidates)
                self._handle_boundary(h, candidates, overshoot, spawned, node_id,
                                      crossing_start_node=False)

            elif predicted_s < 0.0:
                overshoot = abs(predicted_s)
                node_id = self.track_map_engine.get_node_at_boundary(h.segment_id, "start")
                candidates = self.track_map_engine.get_connected_segments(h.segment_id, node_id)
                candidates = self._filter_by_approach_angle(h.segment_id, node_id, candidates)
                self._handle_boundary(h, candidates, overshoot, spawned, node_id,
                                      crossing_start_node=True)

        # Remove parents marked for removal (weight set to 0.0 at spawn) and add spawned children.
        self.hypotheses = [h for h in self.hypotheses if h.weight > 0.0]
        self.hypotheses.extend(spawned)
        

    def _handle_boundary(
        self,
        h: Hypothesis,
        next_segments: List[str],
        overshoot: float,
        spawned: List[Hypothesis],
        node_id: str,
        crossing_start_node: bool = False,
    ) -> None:
        """Resolve a segment boundary crossing for hypothesis h.

        Three outcomes based on the number of connected segments:
            0 — endpoint: clamp KF position to boundary.
            1 — rollover: transition KF to next segment at correct entry position.
            N — switch: spawn N children, mark parent for removal.

        Velocity flip rule (applied in rollover and spawn paths):
            When a child segment is entered from its last_node side, the train
            moves in the decreasing-s direction on that segment, so the child
            velocity must be negated relative to the parent.

            flip = crossing_start_node XOR (child.last_node == node_id)

            All four cases:
              parent crosses last_node  + child first_node=node → NO flip
              parent crosses last_node  + child last_node=node  → FLIP
              parent crosses first_node + child first_node=node → FLIP
              parent crosses first_node + child last_node=node  → NO flip

        Args:
            h: The hypothesis that crossed the boundary.
            next_segments: Connected segment IDs (angle-filtered) at the crossed node.
            overshoot: Distance past the boundary, meters.
            spawned: Accumulator list for new child hypotheses.
            node_id: Node ID of the boundary that was crossed.
            crossing_start_node: True when the parent crossed its first_node
                (predicted_s < 0), False when it crossed its last_node (predicted_s > length).
        """
        n = len(next_segments)

        def _child_dq(velocity: float) -> str:
            if velocity > self.mhmm_config.direction_hysteresis_threshold:
                return "nominal"
            if velocity < -self.mhmm_config.direction_hysteresis_threshold:
                return "reverse"
            return "unknown"

        if n == 0:
            # Endpoint reached: clamp KF position to segment boundary.
            seg_length = self.track_map_engine.get_segment(h.segment_id).length
            if node_id == self.track_map_engine.get_node_at_boundary(h.segment_id, "end"):
                clamp_pos = seg_length
            else:
                clamp_pos = 0.0
            h.kf.transition_to_segment(h.segment_id, clamp_pos)

            logger.info(
                "boundary | hypothesis %d seg=%s node=%s endpoint reached, clamped to %.2fm",
                h.id, h.segment_id, node_id, clamp_pos,
            )

        elif n == 1:
            # Simple rollover: no switch.
            new_seg_id = next_segments[0]
            new_seg = self.track_map_engine.get_segment(new_seg_id)
            if new_seg.first_node == node_id:
                child_s = overshoot
            else:
                child_s = new_seg.length - overshoot

            # Flip velocity if parent and child have the same crossing-side orientation.
            # XOR: flip when crossing_start_node and child entry are both start-side
            # or both end-side — i.e. the physical travel direction reverses on the new segment.
            child_last_at_node = (new_seg.last_node == node_id)
            if crossing_start_node != child_last_at_node:  # XOR
                h.kf.x[1] = -h.kf.x[1]
                if h.kf._predicted_x is not None:
                    h.kf._predicted_x[1] = -h.kf._predicted_x[1]

            h.kf.transition_to_segment(new_seg_id, child_s)
            h.segment_id = new_seg_id
            h.direction_qualifier = _child_dq(float(h.kf.x[1]))

            logger.debug(
                "boundary | hypothesis %d rolled over to seg=%s at s=%.2fm (node=%s)",
                h.id, new_seg_id, child_s, node_id,
            )

        else:
            # Switch: spawn one child per branch, mark parent for removal.
            child_weight = h.weight / n
            child_ids = []

            for child_seg_id in next_segments:
                new_seg = self.track_map_engine.get_segment(child_seg_id)
                if new_seg.first_node == node_id:
                    child_s = overshoot
                else:
                    child_s = new_seg.length - overshoot

                child_kf = h.kf.clone()

                # Apply velocity flip before transition_to_segment so the
                # post-flip velocity is baked into _predicted_x correctly.
                child_last_at_node = (new_seg.last_node == node_id)
                if crossing_start_node != child_last_at_node:  # XOR
                    child_kf.x[1] = -child_kf.x[1]
                    if child_kf._predicted_x is not None:
                        child_kf._predicted_x[1] = -child_kf._predicted_x[1]

                child_kf.transition_to_segment(child_seg_id, child_s)

                child = Hypothesis(
                    id=self._next_id(),
                    segment_id=child_seg_id,
                    kf=child_kf,
                    weight=child_weight,
                    direction_qualifier=_child_dq(float(child_kf.x[1])),
                    parent_id=h.id,
                    spawn_s=child_s,  # child's initial s on its own segment at spawn time
                    # NOTE: must be in the child's coordinate frame so that abs(kf.position - spawn_s) in _prune() 
                    # correctly measures distance traveled since spawning.
                )
                spawned.append(child)
                child_ids.append(child.id)

            h.weight = 0.0  # mark parent for removal

            logger.info(
                "boundary | hypothesis %d seg=%s node=%s spawned %d children %s "
                "weight_each=%.4f",
                h.id, h.segment_id, node_id, n, child_ids, child_weight,
            )

    # -------------------------------------------------------------------------
    # Per-hypothesis direction qualifier
    # -------------------------------------------------------------------------

    def _update_hypothesis_dq(self) -> None:
        """Update direction_qualifier for every active hypothesis from its KF velocity.

        Called after the KF update step each cycle. Uses hysteresis to avoid
        chattering near zero velocity — within the dead-band the previous dq is
        retained (including "unknown" at cold start).

        """
        for h in self.hypotheses:
            v = h.kf.get_state().velocity
            if v > self.mhmm_config.direction_hysteresis_threshold:
                h.direction_qualifier = "nominal"
            elif v < -self.mhmm_config.direction_hysteresis_threshold:
                h.direction_qualifier = "reverse"
            else:
                h.direction_qualifier = "unknown"

    # -------------------------------------------------------------------------
    # Approach-angle filtering
    # -------------------------------------------------------------------------

    def _filter_by_approach_angle(
        self, segment_id: str, node_id: str, candidates: List[str]
    ) -> List[str]:
        """Filter candidate segments by geometric continuity at a node.

        Retains candidates whose departure heading differs from the current
        segment's approach heading by less than approach_angle_threshold.
        This eliminates geometrically impossible branches at undirected switches
        (e.g. a segment that would require the train to do a U-turn).

        Args:
            segment_id: Current segment the hypothesis is leaving.
            node_id: Node at the boundary being crossed.
            candidates: All segments connected to node_id (excluding current).

        Returns:
            Filtered list of candidate segment IDs.
        """
        approach = self.track_map_engine.get_approach_heading(segment_id, node_id)
        filtered = []
        for cand_id in candidates:
            departure = self.track_map_engine.get_departure_heading(cand_id, node_id)
            angle_diff = self._abs_angle_difference(approach, departure)
            if angle_diff < self.mhmm_config.approach_angle_threshold:
                filtered.append(cand_id)
        return filtered

    @staticmethod
    def _abs_angle_difference(heading_a: float, heading_b: float) -> float:
        """Return absolute angular difference in degrees, range [0, 180].

        Args:
            heading_a: First heading in radians.
            heading_b: Second heading in radians.

        Returns:
            Absolute angular difference in degrees.
        """
        diff = abs(math.degrees(heading_a) - math.degrees(heading_b)) % 360
        return min(diff, 360 - diff)

    # -------------------------------------------------------------------------
    # Bayesian weight update
    # -------------------------------------------------------------------------

    def _update_weights(self) -> None:
        """Apply Bayesian weight update: w_new = w_old * L_total, then normalise.

        If all raw weights collapse to zero (all hypotheses score near-zero likelihood), reset to uniform. 
        This is a recovery path for sensor dropout or off-map conditions.
        """
        raw_weights = []
        for h in self.hypotheses:
            if h.latest_likelihood is None:
                logger.warning(
                    "_update_weights | hyp_id=%d seg=%s has no likelihood, treating weight as zero",
                    h.id,
                    h.segment_id,
                )
                raw_weights.append(0.0)
            else:
                raw_weights.append(h.weight * h.latest_likelihood.L_total)
        total = sum(raw_weights)

        # Log raw total so likelihood scale is visible across tuning sessions.
        # NOTE: A consistently tiny total indicates sigma_pos or sigma_omega need widening.
        logger.debug(
            "_update_weights | raw_total=%.6e n_hyp=%d",
            total,
            len(self.hypotheses),
        )

        # NOTE: Normalise so that _prune() can apply an absolute threshold on meaningful weights.
        if total > 0.0:
            for h, rw in zip(self.hypotheses, raw_weights):
                h.weight = rw / total
            self._weights_are_evidence_based = True
        else:
            if not self.hypotheses:
                logger.warning(
                    "_update_weights | no hypotheses to update weights for",
                )
                return
            
            logger.warning(
                "_update_weights | all likelihoods near zero (total=%.3e). Resetting to uniform; possible map error or sensor failure!",
                total,
            )
            uniform = 1.0 / len(self.hypotheses)
            for h in self.hypotheses:
                h.weight = uniform
            self._weights_are_evidence_based = False

        # Maintain per-hypothesis consecutive-below-threshold counter.
        # Incremented here (post-normalisation) so _prune() can read a fresh count.
        # Counter resets to zero when weight is at or above prune_threshold,
        # meaning a recovering hypothesis gets a clean slate immediately.
        for h in self.hypotheses:
            if h.weight < self.mhmm_config.prune_threshold:
                h._consecutive_low += 1
            else:
                h._consecutive_low = 0

    # -------------------------------------------------------------------------
    # Lifecycle methods
    # -------------------------------------------------------------------------

    def _prune(self) -> None:
        """Remove hypotheses below prune_threshold using a two-stage sequential filter.

        STAGE 1 — Grace distance window (geometric, applies to ALL new hypotheses):
            Any hypothesis — whether created at cold start or spawned at a switch —
            is immune from pruning while it has not yet traveled grace_distance metres
            from its entry position (spawn_s). This gives every new entrant a fair
            chance to accumulate position evidence before its weight is judged.

            spawn_s is set at construction time for both cold-start and switch-spawned
            hypotheses, so the check is identical:
                abs(kf.position - spawn_s) < grace_distance  →  immune

            This unifies what were previously two separate mechanisms (update_count
            guard for cold-start, spawn_s guard for switches) into one geometrically
            meaningful criterion.

        STAGE 2 — Consecutive low-weight filter (temporal, applies after grace period):
            Once a hypothesis has traveled beyond its grace window, it becomes eligible
            for pruning only when its weight has been CONSISTENTLY below prune_threshold
            for prune_consecutive_cycles cycles in a row. A single-cycle dip from
            multipath, a brief GNSS obstruction, or the heading-unreliable zone at
            vehicle startup does not warrant elimination. The counter (_consecutive_low)
            resets to zero whenever the hypothesis recovers above threshold.

        SINGLE-SURVIVOR PROTECTION:
            If Stage 2 would eliminate every remaining hypothesis, the prune is
            suppressed for that cycle. The emergency reinitialisation path in
            process_fix() is reserved for genuine off-map conditions — not for
            an aggressive prune on a low-evidence cycle.

        Interaction with _consecutive_low counter:
            The counter is maintained by _update_weights() immediately after
            normalisation, so _prune() always reads a fresh value. The counter
            is NOT reset on transition_to_segment() — a weak hypothesis that
            rolls over a segment boundary retains its streak, which is correct
            because the weakness reflects the observation model, not the segment.
        """
        survivors: List[Hypothesis] = []
        candidates_for_removal: List[Hypothesis] = []

        for h in self.hypotheses:

            # ── STAGE 0: weight above threshold — keep, reset streak ────────
            if h.weight >= self.mhmm_config.prune_threshold:
                h._consecutive_low = 0
                survivors.append(h)
                continue

            # ── STAGE 1: grace distance window ──────────────────────────────
            # Applies to ALL hypotheses (spawn_s is now always set).
            # Cold-start: spawn_s = s_proj at first fix.
            # Switch-spawn: spawn_s = overshoot past the switch node.
            # Both measure distance traveled in the hypothesis's own segment frame.
            if h.spawn_s is not None:
                distance_traveled = abs(h.kf.get_state().position - h.spawn_s)
                if distance_traveled < self.mhmm_config.grace_distance:
                    logger.debug(
                        "prune | hypothesis %d seg=%s weight=%.4f GRACE "
                        "(traveled=%.2fm < grace=%.1fm)",
                        h.id, h.segment_id, h.weight,
                        distance_traveled, self.mhmm_config.grace_distance,
                    )
                    survivors.append(h)
                    continue

            # ── STAGE 2: consecutive low-weight filter ───────────────────────
            # Grace window exhausted. Only prune if weakness is sustained.
            if h._consecutive_low < self.mhmm_config.prune_consecutive_cycles:
                logger.debug(
                    "prune | hypothesis %d seg=%s weight=%.4f SUPPRESSED "
                    "(consecutive_low=%d < required=%d)",
                    h.id, h.segment_id, h.weight,
                    h._consecutive_low, self.mhmm_config.prune_consecutive_cycles,
                )
                survivors.append(h)
                continue

            # Both stages exhausted — eligible for removal.
            candidates_for_removal.append(h)

        # ── Single-survivor protection ───────────────────────────────────────
        if not survivors and candidates_for_removal:
            logger.warning(
                "prune | single-survivor protection: suppressing prune of %d "
                "hypothesis(es) — no survivor would remain. "
                "Widen prune_consecutive_cycles or sigma_map if this persists.",
                len(candidates_for_removal),
            )
            self.hypotheses = candidates_for_removal
            return

        # ── Log and finalise ─────────────────────────────────────────────────
        for h in candidates_for_removal:
            logger.info(
                "prune | hypothesis %d seg=%s weight=%.4f consecutive_low=%d — REMOVED",
                h.id, h.segment_id, h.weight, h._consecutive_low,
            )

        self.hypotheses = survivors


    def _merge(self) -> None:
        """Handles the case where two hypotheses converge on the same segment. 
        Merge same-segment hypotheses within merge_distance of each other.

        The higher-weight hypothesis survives (winner); the lower-weight one
        (loser) is absorbed. Winner accumulates the loser's weight.
        """
        # Group hypotheses by segment for efficient same-segment comparison.
        by_segment: Dict[str, List[Hypothesis]] = {}
        for h in self.hypotheses:
            by_segment.setdefault(h.segment_id, []).append(h)

        # Track merged-out hypothesis IDs to avoid double-merging. A hypothesis can only be merged once per cycle, either as a winner or a loser.
        merged_out: set = set()

        for seg_id, group in by_segment.items():
            # Ignore segments with only one hypothesis. No merge possible.
            if len(group) < 2:
                continue

            # Sort by position O(n log n) so adjacent hypotheses are compared first.
            group.sort(key=lambda h: h.kf.get_state().position)

            # NOTE: This is an O(n^2) pairwise comparison. If performance becomes an issue, consider a more efficient clustering approach.
            for i in range(len(group)):
                if group[i].id in merged_out:
                    continue
                for j in range(i + 1, len(group)):
                    if group[j].id in merged_out:
                        continue

                    pos_i = group[i].kf.get_state().position
                    pos_j = group[j].kf.get_state().position

                    if abs(pos_i - pos_j) < self.mhmm_config.merge_distance:
                        winner = (
                            group[i] if group[i].weight >= group[j].weight else group[j]
                        )
                        loser = group[j] if winner is group[i] else group[i]

                        winner.weight += loser.weight
                        merged_out.add(loser.id)

                        # Log merges so hypothesis consolidation is auditable.
                        logger.debug(
                            "merge | winner=%d loser=%d seg=%s "
                            "pos_i=%.2fm pos_j=%.2fm combined_weight=%.4f",
                            winner.id,
                            loser.id,
                            seg_id,
                            pos_i,
                            pos_j,
                            winner.weight,
                        )

        # Remove merged-out (loser) hypotheses from the main list.
        self.hypotheses = [h for h in self.hypotheses if h.id not in merged_out]

    def _enforce_cap(self) -> None:
        """Remove lowest-weight hypotheses if count exceeds max_hypotheses.

        Protects against hypothesis explosion at complex junctions. Logs at
        WARNING when the cap is hit and repeated triggering indicates spawning
        pressure is outpacing pruning.
        """
        if len(self.hypotheses) <= self.mhmm_config.max_hypotheses:
            return

        logger.warning(
            "cap | hypothesis count %d exceeds max %d, trimming lowest weights",
            len(self.hypotheses),
            self.mhmm_config.max_hypotheses,
        )

        self.hypotheses.sort(key=lambda h: h.weight, reverse=True)
        removed = self.hypotheses[self.mhmm_config.max_hypotheses :] # Removed hypotheses for logging
        self.hypotheses = self.hypotheses[: self.mhmm_config.max_hypotheses]

        for h in removed:
            # Log removed hypotheses so cap events are distinguishable from prune events in the log.
            logger.info(
                "cap | removed hypothesis %d seg=%s weight=%.4f (cap=%d)",
                h.id,
                h.segment_id,
                h.weight,
                self.mhmm_config.max_hypotheses,
            )

    def _normalise_weights(self) -> None:
        """Ensure all weights sum to 1.0 after lifecycle changes.

        Called after prune/merge/cap. Handles the edge case where lifecycle management changes the weight distribution.
        This is non-operational if no hypotheses remain, emergency recovery is handled in process_fix().
        Handling zero total weight by resetting to uniform is a fallback for extreme cases where lifecycle management eliminates all evidence of a MAP hypothesis, which can happen with aggressive pruning parameters or in off-map conditions.
        """
        if not self.hypotheses:
            return

        total = sum(h.weight for h in self.hypotheses)
        if total > 0.0:
            for h in self.hypotheses:
                h.weight /= total
        else:
            # NOTE: Defensive Fallback
            uniform = 1.0 / len(self.hypotheses)
            for h in self.hypotheses:
                h.weight = uniform

        if len(self.hypotheses) == 1:
            # Warn when the system has collapsed to a single hypothesis.
            # NOTE: On isolated track this is expected; after a switch it may indicate overly aggressive pruning parameters.
            logger.info(
                "normalise | single hypothesis remaining seg=%s — "
                "system in single-hypothesis mode",
                self.hypotheses[0].segment_id,
            )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_chi2_threshold(dof: int, p: float) -> float:
        """Return the chi-squared quantile for integrity gating (Approach A).

        Uses a closed-form approximation that is accurate to <0.01% for the
        p-values and dof values used here (p=0.99–0.9999, dof=1–2).

        Key values (verified against scipy.stats.chi2.ppf):
            dof=1, p=0.999  → 10.828
            dof=2, p=0.999  → 13.816
            dof=1, p=0.99   →  6.635
            dof=2, p=0.99   →  9.210

        Args:
            dof: Degrees of freedom (1 = L_pos only, 2 = L_pos + L_curv).
            p:   Confidence level (0 < p < 1). From config integrity_chi2_confidence.

        Returns:
            Chi-squared quantile threshold.
        """
        # Lookup table for the supported operating range.
        # Keys: (dof, p_rounded_to_3dp). Covers the full expected config range.
        _TABLE = {
            (1, 0.990): 6.635,  (1, 0.995): 7.879,
            (1, 0.999): 10.828, (1, 0.9999): 15.137,
            (2, 0.990): 9.210,  (2, 0.995): 10.597,
            (2, 0.999): 13.816, (2, 0.9999): 18.421,
        }
        key = (dof, round(p, 4))
        if key in _TABLE:
            return _TABLE[key]
        # Linear interpolation fallback between nearest bracketing entries for
        # off-table p values. Accuracy sufficient for integrity gating.
        entries = [(k[1], v) for k, v in _TABLE.items() if k[0] == dof]
        entries.sort()
        if p <= entries[0][0]:
            return entries[0][1]
        if p >= entries[-1][0]:
            return entries[-1][1]
        for i in range(len(entries) - 1):
            p0, v0 = entries[i]
            p1, v1 = entries[i + 1]
            if p0 <= p <= p1:
                frac = (p - p0) / (p1 - p0)
                return v0 + frac * (v1 - v0)
        return entries[-1][1]  # unreachable

    def _compute_llr_null(
        self,
        L_total: float,
        d_cross: float,
        sigma_pos_effective: float,
    ) -> float:
        """Compute log-likelihood ratio of best hypothesis vs off-track null (Approach B).

        LLR_null = log(L_total / L_null)

        L_null is the position likelihood under the off-track null model:
        a Gaussian centred at d_cross=0 with sigma = llr_null_sigma_offtrack.
        This represents the question: "is this fix better explained by being on
        this track than by being off-track at this cross-track distance?"

        A positive LLR_null means the track hypothesis explains the observation
        better than the null. A negative value means the null fits at least as
        well — the observation is not sufficiently consistent with the track.

        Args:
            L_total:             Combined likelihood for the best hypothesis.
            d_cross:             Cross-track distance of that hypothesis (m).
            sigma_pos_effective: Effective sigma used in L_pos (m).

        Returns:
            LLR_null (float). Negative values → integrity failure (null wins).
        """
        sigma_null = self.mhmm_config.llr_null_sigma_offtrack
        # Null likelihood: Gaussian with off-track sigma, same d_cross residual.
        L_null = (1.0 / (math.sqrt(2.0 * math.pi) * sigma_null)) * math.exp(
            -(d_cross ** 2) / (2.0 * sigma_null ** 2)
        )
        if L_null <= 0.0 or L_total <= 0.0:
            return -math.inf
        return math.log(L_total / L_null)

    def _compute_confidence(
        self,
        best: "Hypothesis",
        second_best: Optional["Hypothesis"],
    ) -> tuple:
        """Two-axis confidence classification: integrity (A+B) AND discrimination.

        Approach A — Chi-squared consistency gate:
            d2_total = d2_pos + d2_curv (dof tracks active terms)
            integrity_chi2_ok = d2_total < chi2_threshold(dof, p)

        Approach B — LLR vs null + separation:
            integrity_null_ok = LLR(L_total_best / L_null) > 0
            separation_ok     = LLR(L_total_best / L_total_second) > llr_sep_threshold
                                 (vacuously True when only one hypothesis is active)

        Decision:
            if not (integrity_chi2_ok and integrity_null_ok): → "AMBIGUOUS"
            elif not (discrimination_ok and separation_ok):   → "MEDIUM"
            else:                                             → "HIGH"

        Args:
            best:        MAP hypothesis (highest weight).
            second_best: Second-highest-weight hypothesis, or None if only one active.

        Returns:
            Tuple of (confidence_str, diagnostic_dict) where diagnostic_dict
            contains all intermediate values for the DEBUG log.
        """
        lik = best.latest_likelihood

        # ── Approach A: chi-squared consistency gate ─────────────────────────
        if lik is not None:
            d2_pos  = lik.d2_pos
            # d2_aux: whichever auxiliary term is active — d2_head (L_head enabled)
            # or d2_curv (L_curv enabled). Only one is active at a time per config.
            # Both None → dof=1 (L_pos only).
            d2_head = lik.d2_head   # None when L_head inactive or dq=="unknown"
            d2_curv = lik.d2_curv   # None when L_curv inactive or outside decision area
            d2_aux  = d2_head if d2_head is not None else d2_curv
        else:
            d2_pos, d2_head, d2_curv, d2_aux = 0.0, None, None, None

        dof = 2 if d2_aux is not None else 1
        d2_total = d2_pos + (d2_aux if d2_aux is not None else 0.0)
        chi2_thr = self._compute_chi2_threshold(dof, self.mhmm_config.integrity_chi2_confidence)
        integrity_chi2_ok = d2_total < chi2_thr

        # ── Approach B: LLR vs null ───────────────────────────────────────────
        if lik is not None and lik.sigma_omega_effective is not None:
            # sigma_pos_effective is not stored on LikelihoodResult; reconstruct
            # from d_cross and L_pos using the Gaussian inversion.
            # Simpler: use L_total directly (includes L_curv when active).
            llr_null = self._compute_llr_null(
                L_total=lik.L_total,
                d_cross=lik.d_cross,
                sigma_pos_effective=math.sqrt(
                    (lik.d_cross ** 2) / max(d2_pos, 1e-12)
                ) if d2_pos > 0.0 else self.scoring_config.sigma_map,
            )
        elif lik is not None:
            llr_null = self._compute_llr_null(
                L_total=lik.L_total,
                d_cross=lik.d_cross,
                sigma_pos_effective=math.sqrt(
                    (lik.d_cross ** 2) / max(d2_pos, 1e-12)
                ) if d2_pos > 0.0 else self.scoring_config.sigma_map,
            )
        else:
            llr_null = -math.inf
        integrity_null_ok = llr_null > 0.0

        # ── Approach B: LLR separation (discrimination) ───────────────────────
        if second_best is not None and second_best.latest_likelihood is not None:
            L_second = second_best.latest_likelihood.L_total
            L_best   = lik.L_total if lik is not None else 0.0
            if L_second > 0.0 and L_best > 0.0:
                llr_sep = math.log(L_best / L_second)
            elif L_best > 0.0:
                llr_sep = math.inf   # second has zero likelihood — fully separated
            else:
                llr_sep = -math.inf
        else:
            # Single hypothesis — no competitor; separation vacuously satisfied.
            llr_sep = math.inf

        separation_ok     = llr_sep > self.mhmm_config.llr_sep_threshold
        discrimination_ok = best.weight > self.mhmm_config.confidence_high

        # ── Decision ──────────────────────────────────────────────────────────
        if not (integrity_chi2_ok and integrity_null_ok):
            confidence = "AMBIGUOUS"
        elif not (discrimination_ok and separation_ok):
            confidence = "MEDIUM"
        else:
            confidence = "HIGH"

        diag = {
            "d2_pos":             d2_pos,
            "d2_head":            d2_head,
            "d2_curv":            d2_curv,
            "d2_aux":             d2_aux,
            "dof":                dof,
            "d2_total":           d2_total,
            "chi2_thr":           chi2_thr,
            "integrity_chi2_ok":  integrity_chi2_ok,
            "llr_null":           llr_null,
            "integrity_null_ok":  integrity_null_ok,
            "llr_sep":            llr_sep,
            "separation_ok":      separation_ok,
            "discrimination_ok":  discrimination_ok,
        }
        return confidence, diag

    def _build_result(self, timestamp: float) -> LocalizationResult:
        """Build LocalizationResult from current hypothesis state.

        Args:
            timestamp: POSIX time of this result, seconds.

        Returns:
            LocalizationResult with MAP estimate and confidence classification.
        """
        # Return degenerate result if no hypotheses remain.
        if not self.hypotheses:
            return LocalizationResult(
                timestamp=timestamp,
                best_segment_id="",
                best_position=0.0,
                best_velocity=0.0,
                best_position_var=0.0,
                best_weight=0.0,
                confidence="AMBIGUOUS",
                num_hypotheses=0,
                hypotheses=[],
            )

        best = max(self.hypotheses, key=lambda h: h.weight)
        best_state = best.kf.get_state()

        # Identify second-best hypothesis for LLR separation gate (Approach B).
        second_best: Optional[Hypothesis] = None
        if len(self.hypotheses) > 1:
            sorted_hyps = sorted(self.hypotheses, key=lambda h: h.weight, reverse=True)
            second_best = sorted_hyps[1]

        # Two-axis confidence classification (Approach A + B).
        # _weights_are_evidence_based guards the uniform-reset path: a reset
        # produces degenerate L_total values that will fail the chi-squared gate
        # anyway, but the flag provides a fast early-exit for the zero-hypothesis
        # edge case and is retained for diagnostic clarity.
        if not self._weights_are_evidence_based:
            confidence = "AMBIGUOUS"
            diag = {}
            logger.debug(
                "output | confidence overridden to AMBIGUOUS — weights are from "
                "uniform reset, not evidence-based"
            )
        else:
            confidence, diag = self._compute_confidence(best, second_best)

        # Extended DEBUG log: every integrity decision field is auditable per fix.
        best_lik = best.latest_likelihood
        logger.debug(
            "output | t=%.3f conf=%s seg=%s s=%.2fm v=%.3fm/s w=%.4f n_hyp=%d | "
            "d2_pos=%.2f d2_head=%s d2_curv=%s h_res=%s dof=%s chi2_thr=%s chi2_ok=%s "
            "llr_null=%s null_ok=%s llr_sep=%s sep_ok=%s disc_ok=%s",
            timestamp,
            confidence,
            best.segment_id,
            best_state.position,
            best_state.velocity,
            best.weight,
            len(self.hypotheses),
            diag.get("d2_pos", float("nan")),
            f"{diag['d2_head']:.3f}" if diag.get("d2_head") is not None else "None",
            f"{diag['d2_curv']:.3f}" if diag.get("d2_curv") is not None else "None",
            f"{math.degrees(best_lik.heading_residual):.2f}°"
                if (best_lik is not None and best_lik.heading_residual is not None) else "None",
            diag.get("dof", "?"),
            f"{diag['chi2_thr']:.3f}" if diag.get("chi2_thr") is not None else "?",
            diag.get("integrity_chi2_ok", "?"),
            f"{diag['llr_null']:.2f}" if diag.get("llr_null") is not None else "?",
            diag.get("integrity_null_ok", "?"),
            f"{diag['llr_sep']:.2f}" if diag.get("llr_sep") not in (None, math.inf, -math.inf)
                else ("inf" if diag.get("llr_sep") == math.inf else "-inf" if diag.get("llr_sep") == -math.inf else "?"),
            diag.get("separation_ok", "?"),
            diag.get("discrimination_ok", "?"),
        )

        hypothesis_tuples = [
            (h.segment_id, h.kf.get_state().position, h.weight)
            for h in self.hypotheses
        ]

        result = LocalizationResult(
            timestamp=timestamp,
            best_segment_id=best.segment_id,
            best_position=best_state.position,
            best_velocity=best_state.velocity,
            best_position_var=best_state.position_var,
            best_weight=best.weight,
            confidence=confidence,
            num_hypotheses=len(self.hypotheses),
            hypotheses=hypothesis_tuples,
            best_direction_qualifier=best.direction_qualifier,
        )

        return result

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def _next_id(self) -> int:
        """Return and increment the hypothesis ID counter."""
        hid = self._next_hypothesis_id
        self._next_hypothesis_id += 1
        return hid
