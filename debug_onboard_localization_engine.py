# =============================================================================
# onboard_localization_engine.py (DEBUG)
# Onboard Localization Engine — Top-Level Integration Module
# Architecture version: 2.0 (Event-driven callback)

# Responsibilities:
#   - Event-driven fix processing via CertusSubscriber.register_fix_callback()
#   - MHMM orchestrator lifecycle (auto-init via OR-P06, cooldown, recovery)
#   - direction_of_movement: "forward"/"reverse"/"neutral" via body velocity + hysteresis
#     (sensor-level, globally applicable to all hypotheses)
#   - LocalizationResult + SensorFix → OLEStateEstimate protobuf → Redis publish

# WHAT THIS MODULE DOES NOT DO:
#   - Does not modify SensorFix values (no sign flips, no frame conversions)
#   - Does not compute direction_qualifier (orchestrator owns per-hypothesis dq)
#   - Does not manage hypothesis lifecycle (orchestrator's responsibility)
#   - Does not call TrackMapEngine directly for algorithm work (uses orchestrator's
#     shared track_map_engine reference for ENU/LLA conversion in output path only)
#   - Does not tune parameters at runtime
#   - Does not perform IMU-rate dead-reckoning (Certus handles this internally)
#   - Does not call build_sensor_fix() directly — receives SensorFix via callback
#   - Does not run a GNSS timeout watchdog — deferred to external dockerised service

# THREADING MODEL:
#   pubsub thread (CertusSubscriber)
#       _on_lla fires → build_sensor_fix() → _on_fix_ready(fix)
#           _processing_lock.acquire(blocking=False)
#               acquired  → _process_gnss_fix(fix) → _publish_result(result, fix)
#               not acq.  → WARNING "overrun" + drop fix (perf violation)
#           _processing_lock.release()
#   main thread
#       start() blocks on _stop_event.wait() after setup

# COLD-START: No special path. orchestrator.process_fix() auto-inits (OR-P06).
#   direction_qualifier is per-hypothesis, owned by the orchestrator. Defaults to
#   "unknown" at spawn time and transitions to "nominal"/"reverse" as each
#   hypothesis's KF velocity crosses the hysteresis threshold.

# COOLDOWN PATH (inside _process_gnss_fix):
#   In cooldown → _attempt_cooldown_recovery(fix)
#   Too soon / failed → return _build_cooldown_result (AMBIGUOUS, 0 hyp)
#   Recovered → return result directly; do NOT call process_fix on same fix
#               (fix was consumed by initialise() inside recovery)
# =============================================================================

from __future__ import annotations

import argparse
import math
import signal
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import yaml
import redis

from proto import onboard_localization_engine_pb2 as ole_pb2

from mhmm_orchestrator import MHMMOrchestrator, LocalizationResult
from sensor_interface import CertusSubscriber, SensorFix

# NOTE:DEBUG
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OLEConfig:
    """OLE integration module parameters. Loaded from config.yaml ole section."""
    direction_hysteresis_threshold: float = 0.3
    max_init_failures: int = 5
    init_cooldown_interval_s: float = 5.0

    @classmethod
    def from_config(cls, config_path: str) -> OLEConfig:
        """Load OLEConfig from the ole section of config.yaml."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        kf = cfg["kalman_filter"]
        ole = cfg["ole"]
        config = cls(
            direction_hysteresis_threshold=float(kf["direction_hysteresis_threshold"]),
            max_init_failures=int(ole["max_init_failures"]),
            init_cooldown_interval_s=float(ole["init_cooldown_interval_s"]),
        )
        logger.info(
            "OLEConfig loaded | dq_hyst=%.2fm/s "
            "max_init_fail=%d cooldown_interval=%.1fs",
            config.direction_hysteresis_threshold,
            config.max_init_failures,
            config.init_cooldown_interval_s,
        )
        return config


# =============================================================================
# Internal Runtime State
# =============================================================================

@dataclass
class OLEState: # NOTE: Global OLE state (applicable to all active hypotheses)
    """Mutable runtime state owned by OnboardLocalizationEngine.

    All fields written and read exclusively on the pubsub thread (under
    _processing_lock).
    """
    is_running: bool = False
    direction_of_movement: str = "neutral"  # "forward" / "reverse" / "neutral" via body velocity
    last_fix_timestamp: Optional[float] = None # arrival timestamp of last processed fix (for GNSS timeout diagnostics)
    consecutive_init_failures: int = 0
    in_cooldown: bool = False
    last_cooldown_attempt: float = 0.0
    last_result: Optional[LocalizationResult] = None


# =============================================================================
# Proto enum helpers
# =============================================================================

# Maps SensorFix.fix_quality strings → OLEStateEstimate FixType enum values.
_FIX_TYPE_MAP = {
    "INVALID":   ole_pb2.INVALID,
    "SPP":       ole_pb2.SPP,
    "DGNSS":     ole_pb2.DGNSS,
    "SBAS":      ole_pb2.SBAS,
    "RTK_FLOAT": ole_pb2.RTK_FLOAT,
    "RTK_FIXED": ole_pb2.RTK_FIXED,
}

# Maps LocalizationResult.confidence strings → OLEConfidence enum values.
_CONFIDENCE_MAP = {
    "HIGH":      ole_pb2.HIGH,
    "MEDIUM":    ole_pb2.MEDIUM,
    "AMBIGUOUS": ole_pb2.AMBIGUOUS,
}


# =============================================================================
# Main Class
# =============================================================================

class OnboardLocalizationEngine:
    """Top-level OLE runtime. Owns the event-driven fix processing loop,
    orchestrator lifecycle, and Redis publish path.

    Usage:
        ole = OnboardLocalizationEngine(
            track_json_path="back_parking_lot_track.json",
            config_path="config.yaml",
        )
        ole.start()   # blocks on _stop_event.wait() until SIGINT/SIGTERM/stop()
    """

    def __init__(self, track_json_path: str, config_path: str) -> None:
        """Instantiate all components and register event-driven callback.
        Does NOT start the subscriber until start() is explicitly called.

        Args:
            track_json_path: Path to track JSON file.
            config_path:     Path to config.yaml.
        """
        self._config = OLEConfig.from_config(config_path)

        with open(config_path, "r") as f:
            _cfg = yaml.safe_load(f)

        # Sub-components
        self._certus_subscriber = CertusSubscriber(config_path)
        self._mhmm_orchestrator = MHMMOrchestrator(track_json_path, config_path)
        self._track_map_engine = self._mhmm_orchestrator.track_map_engine  # shared ref — output conversion API only

        # Runtime state
        self._ole_state = OLEState()

        # Redis publisher — separate connection from CertusSubscriber's subscriber connection
        _redis_cfg = _cfg["redis"]
        self._redis_pub = redis.Redis(
            host=str(_redis_cfg["host"]),
            port=int(_redis_cfg["port"]),
            db=int(_redis_cfg.get("db", 0)),
            decode_responses=False,
        )
        self._publish_channel: str = str(_redis_cfg["publish_channel_full_state_estimate"])

        # NOTE: DEBUG
        self._hyp_channel: str = str(_redis_cfg.get(
            "publish_channel_hypotheses_debug", "ole_hypotheses_debug"
        ))

        # ENU reference origin — track_map_engine owns the coordinate math
        _ref = _cfg["reference"]
        self._ref_lat: float = float(_ref["lat"])
        self._ref_lon: float = float(_ref["lon"])
        self._ref_alt: float = float(_ref["alt"])

        # Lock guards _process_gnss_fix + _publish_result (both on pubsub thread).
        # Non-blocking acquire in _on_fix_ready — if held, drop fix + log WARNING.
        self._processing_lock = threading.Lock()

        # Stop event — set by stop() or signal handler; unblocks start()
        self._stop_event = threading.Event()

        # Register event-driven callback — fires on every fully assembled SensorFix
        self._certus_subscriber.register_fix_callback(self._on_fix_ready)

        logger.info(
            "OnboardLocalizationEngine init | track=%s config=%s "
            "publish_channel=%s ref=(%.6f°, %.6f°, %.2fm)",
            track_json_path, config_path, self._publish_channel,
            self._ref_lat, self._ref_lon, self._ref_alt,
        )

    # =========================================================================
    # Entry points
    # =========================================================================

    def start(self) -> None:
        """Start subscriber and block until stop() or signal.

        Does NOT use a polling loop. Main thread parks at _stop_event.wait()
        after setup. All fix processing happens on the pubsub thread via
        _on_fix_ready callback.
        """
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._certus_subscriber.start()

        self._ole_state.is_running = True
        logger.info(
            "OnboardLocalizationEngine started | publish_channel=%s",
            self._publish_channel,
        )

        self._stop_event.wait()  # Main thread blocks here until stop() is called by Signal Interrupt (signal_handler) — all work on pubsub thread
        self._shutdown() # Clean shutdown Certus subscriber and Redis connection

    def stop(self) -> None:
        """Request a clean shutdown. Returns immediately; start() unblocks."""
        self._stop_event.set()

    # =========================================================================
    # Approach B — fix callback (runs on pubsub thread)
    # =========================================================================

    def _on_fix_ready(self, fix: SensorFix) -> None:
        """Callable method called by CertusSubscriber on every fully assembled SensorFix.

        Runs on the pubsub thread. Non-blocking lock attempt — if the previous
        fix is still being processed, this fix is dropped and a WARNING is
        logged. An overrun means process_fix exceeded the inter-fix budget and
        must be surfaced for diagnostics.

        Args:
            fix: Fully assembled, antenna-corrected SensorFix from _on_lla trigger.
        """
        acquired = self._processing_lock.acquire(blocking=False)
        if not acquired:
            logger.warning(
                "_on_fix_ready | overrun — process_fix still running, "
                "dropping fix t=%.3f (inter-fix budget exceeded)",
                fix.timestamp,
            )
            return
        try:
            result = self._process_gnss_fix(fix)
            if result is not None:
                self._publish_result(result, fix)
            else:
                logger.debug(
                    "_on_fix_ready | no result to publish for fix t=%.3f "
                    "(process_fix returned None, likely on first-fix exception)",
                    fix.timestamp,
                )
        # Log any exception from processing/publishing, but do not let it escape to the pubsub thread (which would kill it and stop all future fix callbacks).
        except Exception as exc:  # noqa: BLE001
            logger.error("Exception in _on_fix_ready: %s", exc, exc_info=True)

        finally:
            self._processing_lock.release()

    # =========================================================================
    # Fix processing (always called under _processing_lock)
    # =========================================================================

    def _process_gnss_fix(self, fix: SensorFix) -> Optional[LocalizationResult]:
        """Drive one MHMM cycle from a valid SensorFix.

        Always called under _processing_lock (from _on_fix_ready).

        Two paths:
            1. Cooldown: throttled recovery retries, AMBIGUOUS output while waiting.
            2. Normal: dq/dom update → orchestrator.process_fix (auto-inits on first call).
               No separate cold-start path needed.

        Cooldown recovery note: when _attempt_cooldown_recovery returns a result,
        that fix was already consumed by initialise() inside recovery. Return
        immediately — do NOT call process_fix on the same fix.

        Args:
            fix: Fully assembled, antenna-corrected SensorFix.

        Returns:
            LocalizationResult to publish, or None only on first-fix exception
            (last_result is None, nothing to re-publish).
        """
        self._ole_state.last_fix_timestamp = time.time()

        # ------------------------------------------------------------------
        # 1. Cooldown path
        # ------------------------------------------------------------------
        if self._ole_state.in_cooldown:
            recovered = self._attempt_cooldown_recovery(fix)
            if recovered is None:
                # Too soon to retry or retry failed — publish AMBIGUOUS
                return self._build_cooldown_result(fix.timestamp)
            # Recovery succeeded — fix consumed by initialise() inside recovery.
            # Do NOT fall through to process_fix.
            return recovered

        # ------------------------------------------------------------------
        # 2. Normal path (first fix, steady-state, post-emergency-reinit)
        # ------------------------------------------------------------------
        self._update_direction_of_movement(fix)

        try:
            result = self._mhmm_orchestrator.process_fix(fix)
        except Exception as exc:  # noqa: BLE001
            logger.error("process_fix exception: %s", exc, exc_info=True)
            if self._ole_state.last_result is None:
                logger.warning("process_fix exception on first fix — nothing to re-publish")
            return self._ole_state.last_result  # None on first fix; caller publishes nothing

        if result.num_hypotheses == 0:
            self._ole_state.consecutive_init_failures += 1
            logger.warning(
                "process_fix | 0 hypotheses returned (consecutive=%d/%d)",
                self._ole_state.consecutive_init_failures,
                self._config.max_init_failures,
            )
            if self._ole_state.consecutive_init_failures >= self._config.max_init_failures:
                self._enter_cooldown()
                return self._build_cooldown_result(fix.timestamp)
        else:
            self._ole_state.consecutive_init_failures = 0

        self._ole_state.last_result = result

        logger.debug(
            "process_fix | dq=%s dom=%s n_hyp=%d seg=%s conf=%s w=%.4f",
            result.best_direction_qualifier,
            self._ole_state.direction_of_movement,
            result.num_hypotheses,
            result.best_segment_id,
            result.confidence,
            result.best_weight,
        )

        return result

    # =========================================================================
    # Direction qualifier / movement
    # =========================================================================

    def _update_direction_of_movement(self, fix: SensorFix) -> None:
        """Update direction_of_movement from body-frame velocity in SensorFix.

        Three states using the same hysteresis threshold as direction_qualifier:
            "forward"  — fix.velocity > +threshold
            "reverse"  — fix.velocity < -threshold
            "neutral"  — fix.velocity within ±threshold, or fix.velocity is None

        Args:
            fix: Current SensorFix. fix.velocity is body-frame X (m/s).
        """
        threshold = self._config.direction_hysteresis_threshold
        if fix.velocity is None:
            self._ole_state.direction_of_movement = "neutral"
        elif fix.velocity > threshold:
            self._ole_state.direction_of_movement = "forward"
        elif fix.velocity < -threshold:
            self._ole_state.direction_of_movement = "reverse"
        else:
            self._ole_state.direction_of_movement = "neutral"

    # =========================================================================
    # Off-map cooldown
    # =========================================================================

    def _enter_cooldown(self) -> None:
        """Enter off-map cooldown. Throttles reinitialisation to cooldown interval."""
        self._ole_state.in_cooldown = True
        self._ole_state.last_cooldown_attempt = time.time()
        logger.warning(
            "Entering off-map cooldown after %d consecutive failures. "
            "Retry interval: %.1fs",
            self._ole_state.consecutive_init_failures,
            self._config.init_cooldown_interval_s,
        )

    def _attempt_cooldown_recovery(self, fix: SensorFix) -> Optional[LocalizationResult]:
        """Try to reinitialise while in cooldown, respecting the retry interval.

        Args:
            fix: Current SensorFix.

        Returns:
            LocalizationResult if recovery succeeded (cooldown lifted), else None.
        """
        elapsed = time.time() - self._ole_state.last_cooldown_attempt
        if elapsed < self._config.init_cooldown_interval_s:
            return None  # Too soon — throttle

        self._ole_state.last_cooldown_attempt = time.time()
        logger.debug("cooldown | retry attempt (%.1fs since last)", elapsed)

        result = self._mhmm_orchestrator.initialise(fix)

        if result.num_hypotheses > 0:
            self._ole_state.in_cooldown = False
            self._ole_state.consecutive_init_failures = 0
            logger.info(
                "cooldown | recovered — %d hypotheses seeded",
                result.num_hypotheses,
            )
            return result

        logger.debug("cooldown | recovery failed — still off-map")
        return None

    def _build_cooldown_result(self, timestamp: float) -> LocalizationResult:
        """Build a degenerate AMBIGUOUS LocalizationResult for cooldown output."""
        return LocalizationResult(
            timestamp=timestamp,
            best_segment_id="",
            best_position=0.0,
            best_velocity=0.0,
            best_position_var=0.0,  # matches orchestrator degenerate result convention
            best_weight=0.0,
            confidence="AMBIGUOUS",
            num_hypotheses=0,
            hypotheses=[],
        )

    # =========================================================================
    # Output publishing
    # =========================================================================

    def _publish_result(self, result: LocalizationResult, fix: SensorFix) -> None:
        """Serialize LocalizationResult + SensorFix to OLEStateEstimate and publish.

        Args:
            result: Output of _process_gnss_fix (process_fix, initialise, or cooldown).
            fix:    The SensorFix that produced this result.
        """
        msg = ole_pb2.OLEStateEstimate()

        # Timestamps
        msg.system_timestamp_ms = int(time.time() * 1000)
        msg.gps_timestamp_ms    = int(result.timestamp * 1000)

        # ENU pose + LLA — only populated when MAP estimate is valid
        if result.best_segment_id:
            try:
                e, n, u = self._track_map_engine.track_to_enu(
                    result.best_segment_id, result.best_position
                )
                msg.ole_pose_enu.position_x = float(e)
                msg.ole_pose_enu.position_y = float(n)
                msg.ole_pose_enu.position_z = float(u)
                msg.ole_pose_enu.heading    = float(
                    self._compute_track_heading(result.best_segment_id, result.best_position)
                )

                # enu_to_lla preserves 64-bit double for lat/lon
                lat, lon, alt = self._track_map_engine.enu_to_lla(e, n, u)
                msg.ole_loc_lla.latitude_deg  = lat
                msg.ole_loc_lla.longitude_deg = lon
                msg.ole_loc_lla.altitude_m    = alt
            except ValueError as exc:
                # track_to_enu raises ValueError if a segment has no ENU points
                # (degenerate/malformed track geometry). Log and publish without pose
                # rather than letting ValueError escape to the pubsub thread.
                logger.error(
                    "_publish_result | track_to_enu failed for seg=%s s=%.2fm: %s — "
                    "publishing without ENU/LLA fields",
                    result.best_segment_id, result.best_position, exc,
                )

        # Fix quality
        fix_quality = fix.fix_quality if fix.fix_quality else "INVALID"
        msg.fix_type = _FIX_TYPE_MAP.get(fix_quality, ole_pb2.INVALID)

        # Velocity — body-frame X = forward; yaw rate from gyro
        msg.ole_velocity.velocity_x = float(result.best_velocity)
        msg.ole_velocity.yaw_rate   = float(fix.gyro_yaw_rate) if fix.gyro_yaw_rate is not None else 0.0

        # Satellite count via public property
        num_sats = self._certus_subscriber.num_satellites
        if num_sats is not None:
            msg.num_sats = num_sats

        # Accuracy
        msg.h_accuracy_m         = float(fix.horizontal_accuracy)
        msg.heading_accuracy_rad = float(math.radians(fix.heading_accuracy))

        # Source metadata
        msg.heading_source = ole_pb2.FUSED
        msg.vel_reference  = ole_pb2.BODY

        # Reference frame origin
        msg.ref_origin.reference_lat_deg = self._ref_lat
        msg.ref_origin.reference_lon_deg = self._ref_lon
        msg.ref_origin.reference_alt_m   = self._ref_alt

        # MHMM confidence
        msg.ole_confidence = _CONFIDENCE_MAP.get(result.confidence, ole_pb2.AMBIGUOUS)

        try:
            self._redis_pub.publish(self._publish_channel, msg.SerializeToString())
        except redis.RedisError as exc:
            # Catch all Redis exceptions (ConnectionError, TimeoutError, ResponseError, etc.)
            # A narrower ConnectionError catch lets TimeoutError/ResponseError escape to
            # the pubsub thread, which would kill it and stop all future _on_fix_ready calls.
            logger.error("Redis publish failed: %s", exc)

    
        # NOTE: DEBUG
        # Diagnostics debug payload — consumed by visualiser and performance logger.
        # Published regardless of whether hypotheses list is non-empty so the logger
        # always gets a row per fix (even cooldown / degenerate results).
        # Non-critical: Redis errors are swallowed silently.
        try:
            # --- Dominant hypothesis diagnostics (MAP hypothesis only) ---
            best_hyp = max(self._mhmm_orchestrator.hypotheses,
                           key=lambda h: h.weight) if self._mhmm_orchestrator.hypotheses else None

            # d_cross and likelihood from the dominant hypothesis latest LikelihoodResult
            dom_d_cross       = float(best_hyp.latest_likelihood.d_cross)      if (best_hyp and best_hyp.latest_likelihood)  else None
            dom_P_pos         = float(best_hyp.kf.get_state().position_var)    if best_hyp else None
            dom_L_total       = float(best_hyp.latest_likelihood.L_total)      if (best_hyp and best_hyp.latest_likelihood)  else None
            dom_d2_pos        = float(best_hyp.latest_likelihood.d2_pos)       if (best_hyp and best_hyp.latest_likelihood) else None
            dom_d2_curv       = float(best_hyp.latest_likelihood.d2_curv)      if (best_hyp and best_hyp.latest_likelihood and best_hyp.latest_likelihood.d2_curv is not None) else None
            dom_d2_head       = float(best_hyp.latest_likelihood.d2_head)      if (best_hyp and best_hyp.latest_likelihood and best_hyp.latest_likelihood.d2_head is not None) else None
            dom_heading_res   = float(best_hyp.latest_likelihood.heading_residual) if (best_hyp and best_hyp.latest_likelihood and best_hyp.latest_likelihood.heading_residual is not None) else None

            # Innovation y and S from the KF update on the dominant hypothesis.
            # The KF exposes the last accepted innovation via _innovation_history;
            # we read the raw last entry directly so we get the exact value for this fix.
            # S is not stored on the KF state, so we derive it from P_prior and R at this point.
            # Instead we use get_innovation_stats() for the rolling NIS summary.
            nis_mean, nis_var = best_hyp.kf.get_innovation_stats(window=20) if best_hyp else (None, None)
            dom_innov_y       = float(best_hyp.kf._innovation_history[-1]) if (best_hyp and best_hyp.kf._innovation_history) else None

            # raw_total is the unnormalised weight sum from _update_weights, stored
            # transiently on the orchestrator after each cycle.
            raw_total         = float(getattr(self._mhmm_orchestrator, '_last_raw_total', None) or 0.0)

            # Gate hit count this cycle — number of hypotheses whose update was gated
            n_gated           = int(getattr(self._mhmm_orchestrator, '_last_n_gated', 0))

            # Process latency: wall-clock time from GPS timestamp to now (publish)
            # Correct latency: wall-clock time from fix reception to publish.
            # last_fix_timestamp is set at the top of _process_gnss_fix (line 315)
            # immediately when the fix arrives — before process_fix() is called.
            # This measures actual OLE processing time, independent of GPS epoch.
            process_latency_ms = (time.time() - self._ole_state.last_fix_timestamp) * 1000.0

            diag_payload = json.dumps({
                "hypotheses":        [[h[0], h[1], h[2]] for h in result.hypotheses],
                "dq":                result.best_direction_qualifier,
                "dom":               self._ole_state.direction_of_movement,
                # --- per-fix diagnostics (dominant hypothesis) ---
                "best_seg":          result.best_segment_id,
                "best_s":            result.best_position,
                "best_vel":          result.best_velocity,
                "best_weight":       result.best_weight,
                "confidence":        result.confidence,
                "n_hyp":             result.num_hypotheses,
                "gps_ts":            result.timestamp,
                "sys_ts":            time.time(),
                "process_latency_ms": round(process_latency_ms, 3),
                "h_acc_m":           float(fix.horizontal_accuracy) if fix.horizontal_accuracy else None,
                "gnss_fix_type":     fix.fix_quality,
                "num_sats":          self._certus_subscriber.num_satellites,
                "body_vel_mps":      float(fix.velocity) if fix.velocity else None,
                # --- KF diagnostics ---
                "d_cross":           dom_d_cross,
                "P_pos":             dom_P_pos,
                "L_total":           dom_L_total,
                "d2_pos":            dom_d2_pos,
                "d2_curv":           dom_d2_curv,
                "d2_head":           dom_d2_head,
                "heading_residual_deg": round(math.degrees(dom_heading_res), 3) if dom_heading_res is not None else None,
                "innov_y":           dom_innov_y,
                "nis_mean":          float(nis_mean) if nis_mean is not None else None,
                "nis_var":           float(nis_var)  if nis_var  is not None else None,
                "raw_total":         raw_total,
                "n_gated":           n_gated,
            }).encode()
            self._redis_pub.publish(self._hyp_channel, diag_payload)
        except redis.RedisError:
            pass  # non-critical — logger and visualiser degrade gracefully
        except Exception as _diag_exc:  # noqa: BLE001
            logger.debug("diagnostics publish failed: %s", _diag_exc)

    def _compute_track_heading(self, segment_id: str, s: float) -> float:
        """Compute track heading (radians, ENU) at along-track position s.

        Finite-difference chord (±0.5 m) via track_to_enu → atan2(ΔN, ΔE).
        Returns angle in [−π, π] measured CCW from East (ENU convention).

        Args:
            segment_id: Segment containing the position.
            s:          Along-track distance from first_node, metres.

        Returns:
            Heading in radians, or 0.0 if chord is degenerate (seg_len < 0.01 m).
        """
        step = 0.5  # metres
        seg_len = self._track_map_engine.get_segment(segment_id).length
        s_fwd = min(s + step, seg_len)
        s_bwd = max(s - step, 0.0)

        if s_fwd - s_bwd < 0.01:
            logger.warning(
                "_compute_track_heading | degenerate chord %.4fm on seg=%s "
                "s=%.4fm seg_len=%.4fm — returning 0.0 rad",
                s_fwd - s_bwd, segment_id, s, seg_len,
            )
            return 0.0

        e0, n0, _ = self._track_map_engine.track_to_enu(segment_id, s_bwd)
        e1, n1, _ = self._track_map_engine.track_to_enu(segment_id, s_fwd)

        return math.atan2(n1 - n0, e1 - e0)

    # =========================================================================
    # Shutdown
    # =========================================================================

    def _signal_handler(self, signum: int, frame) -> None:  # noqa: ANN001
        """Handle SIGINT / SIGTERM — request clean shutdown."""
        logger.info("Signal %d received — initiating shutdown", signum)
        self.stop()

    def _shutdown(self) -> None:
        """Release all resources. Called by start() after _stop_event fires."""
        self._ole_state.is_running = False
        try:
            self._certus_subscriber.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("certus_subscriber.stop() error during shutdown: %s", exc)
        try:
            self._redis_pub.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis close error during shutdown: %s", exc)
        logger.info("OnboardLocalizationEngine shut down cleanly")


# =============================================================================
# Entry point - Main Thread
# =============================================================================

if __name__ == "__main__":

    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    # )
    # NOTE: DEBUG
    import os
    import sys
    from datetime import datetime
    from pathlib import Path

    _ws  = Path(__file__).resolve().parent
    _log_dir = _ws / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _log_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = _log_dir / f"ole_{_log_ts}.log"
    _log_fmt  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"

    _root = logging.getLogger()
    _root.setLevel(logging.DEBUG)

    # Terminal handler — INFO and above only
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setLevel(logging.INFO)
    _sh.setFormatter(logging.Formatter(_log_fmt, datefmt="%H:%M:%S"))
    _root.addHandler(_sh)

    # File handler — ALL levels (DEBUG) for post-run analysis
    _fh = logging.FileHandler(_log_file, mode="w", encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter(_log_fmt))
    _root.addHandler(_fh)

    logging.getLogger(__name__).info("OLE log → %s", _log_file)

    parser = argparse.ArgumentParser(description="Onboard Localization Engine")
    parser.add_argument(
        "--track",
        default="data/back_parking_lot_track.json",
        help="Path to track JSON file",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    engine = OnboardLocalizationEngine(
        track_json_path=args.track,
        config_path=args.config,
    )
    engine.start()
