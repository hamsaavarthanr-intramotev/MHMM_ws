# =============================================================================
# kalman_filter.py
# Onboard Localization Engine — Single-Hypothesis Kalman Filter

# Responsibilities:
#   - Track along-track position and velocity for a single track hypothesis
#   - Provide predict(), update()
#   - Provide clone() for hypothesis spawning at switches
#   - Provide transition_to_segment() for segment rollover
#   - Provide get_innovation_stats() for consistency monitoring

# MOTION MODEL — Constant Velocity (CV):
#   State:  x = [s, v]^T   (along-track position m, velocity m/s)
#   F = [[1, dt], [0, 1]]
#   Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]
#
#   NOTE: CV selected over DWPA for this domain. See MHMM_Project_Context.md
#   section "Hypothesis update" for full rationale and upgrade path.

# MEASUREMENT MODEL:
#   z = s_proj  (along-track projection from TrackMapEngine)
#   H = [1, 0],  R = sigma_pos^2  (from fix.horizontal_accuracy or config default)

# SIGN CONTRACT:
#   All quantities (s, curvature, heading) are parameterised in the
#   first_node->last_node stored point order.
#   No runtime sign flip on curvature. The KF velocity sign resolves direction:
#     positive velocity = moving toward last_node ("nominal")
#     negative velocity = moving toward first_node ("reverse")
#   omega_expected = velocity × kappa handles direction automatically.
#   This filter is direction-agnostic — it tracks s and v on whatever
#   segment it is assigned to. Direction handling is the orchestrator's job.
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class KFState:
    """dataclass to store Kalman filter state vector x=[s, v]^T at a single timestep.
    Each state is denoted by a gaussian N(mean, covariance) where mean=[s, v]^T and covariance=P.
    
    Consumed by the MHMM orchestrator for output reporting and diagnostics.

    Attributes: 
        position: Mean along-track position from from_node, meters.
        velocity: Mean along-track velocity, m/s. Positive = forward travel.
        position_var: Position variance P[0,0], meters^2.
        velocity_var: Velocity variance P[1,1], (m/s)^2.
        segment_id: Track segment this filter is currently on.
        timestamp: POSIX time of this state estimate, seconds.
    """
    position: float
    velocity: float
    position_var: float
    velocity_var: float
    segment_id: str
    timestamp: float


@dataclass
class KFUpdate:
    """Kalman state (posterior) and related information of a single predict-update cycle.

    Attributes:
        state: Post-update KFState vector (posterior).
        innovation: y = s_proj - H*x_prior. Signed along-track residual, meters.
        innovation_covariance: S = H*P_prior*H^T + R, scalar, meters^2. (system uncertainty for this measurement)
        predicted_position: x_prior[0] before update, meters. For logging.
        measurement: s_proj value fed into this update, meters.
        measurement_noise: R value used in this update, meters^2.
        accepted: False if measurement was rejected by gating check.
            When False, state holds the predicted (not updated) values.
    """
    state: KFState
    innovation: float
    innovation_covariance: float
    predicted_position: float
    measurement: float
    measurement_noise: float
    accepted: bool


@dataclass
class KFConfig:
    """Kalman filter parameters. Loaded from config.yaml kalman_filter section.

    Attributes: 
        process_noise_density: q (m^2/s^3). Scalar variance of unmodelled 1D acceleration per unit time. NOTE: Tune via innovation variance plot.
        default_measurement_noise: Fallback sigma_pos (m) used as R = sigma^2 when fix.horizontal_accuracy is not available.
        initial_position_variance: P_init[0,0] (m^2). Position uncertainty at cold start. Default 9.0 = 3m sigma, typical GNSS accuracy.
        initial_velocity_variance: P_init[1,1] ((m/s)^2). Velocity uncertainty at cold start. Default 100.0 = 10 m/s sigma,
            reflecting complete ignorance of initial velocity.
        gate_threshold: Reject update if |y|/sqrt(S) exceeds this (sigma). Protects against GNSS multipath spikes and outlier fixes.
        min_dt: Minimum accepted dt (s). Rejects duplicate timestamps.
        max_dt: Maximum accepted dt (s). Rejects stale predictions.
    """
    # Initial values are placeholders; actual values loaded from config.yaml via from_config() class method.
    process_noise_density: float = 1.0
    default_measurement_noise: float = 2.5
    initial_position_variance: float = 9.0
    initial_velocity_variance: float = 100.0
    gate_threshold: float = 5.0
    min_dt: float = 0.01
    max_dt: float = 10.0

    @classmethod
    def from_config(cls, config_path: str) -> KFConfig:
        """Load KFConfig from the kalman_filter section of config.yaml.

        Args:
            config_path: Path to config.yaml.

        Returns:
            KFConfig populated from config.yaml kalman_filter section.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        kf = cfg["kalman_filter"]

        config = cls(
            process_noise_density=float(kf["process_noise_density"]),
            default_measurement_noise=float(kf["default_measurement_noise"]),
            initial_position_variance=float(kf["initial_position_variance"]),
            initial_velocity_variance=float(kf["initial_velocity_variance"]),
            gate_threshold=float(kf["gate_threshold"]),
            min_dt=float(kf["min_dt"]),
            max_dt=float(kf["max_dt"]),
        )

        # Log all config values at load time so the exact parameters used for each run are captured
        # NOTE: Essential for reproducing results when comparing tuning sessions against the same GPS dataset.
        logger.info(
            "KFConfig loaded | q=%.3f default_R_sigma=%.2fm "
            "P0_pos=%.2f P0_vel=%.2f gate=%.1f sigma "
            "min_dt=%.3fs max_dt=%.1fs",
            config.process_noise_density,
            config.default_measurement_noise,
            config.initial_position_variance,
            config.initial_velocity_variance,
            config.gate_threshold,
            config.min_dt,
            config.max_dt,
        )

        return config


# =============================================================================
# TrackKalmanFilter
# =============================================================================

class TrackKalmanFilter:
    """Hypothesis: 1D Kalman filter tracking along-track position and velocity.

    Implements a constant-velocity motion model with process noise on acceleration.
    Each MHMM hypothesis contains one instance of this filter.
    
    Provides predict() and update() methods for the MHMM orchestrator to call at IMU and GNSS rates respectively, 
    and clone() and transition_to_segment() for hypothesis lifecycle management at switches.
    """

    # Observation matrix: z = measure position only (not velocity).
    # z ~ H @ x gives scalar prediction in measurement space.
    _H = np.array([[1.0, 0.0]])

    def __init__(
        self,
        segment_id: str,
        initial_position: float,
        config: KFConfig,
        initial_velocity: float = 0.0,
        timestamp: float = 0.0,
    ) -> None:
        """Initialise the Kalman filter.

        Args:
            segment_id: Track segment this filter starts on.
            initial_position: Along-track position from from_node, meters.
            config: KFConfig loaded from config.yaml.
            initial_velocity: Initial velocity estimate, m/s. Default 0.0
            timestamp: POSIX time of initialisation, seconds.
        """
        self.segment_id = segment_id
        self.config = config
        self.timestamp = timestamp
        self._update_count = 0
        self._innovation_history: list = []  # for diagnostic consistency monitoring

        # State vector: x = [s, v]^T
        self.x = np.array([initial_position, 
                           initial_velocity], dtype=float)

        # Initial state covariance: P = diagonal (position and velocity variances)
        # NOTE: The position and velocity relation will be establlished at the filter updates covariances (non-diagonal P)
        self.P = np.array([[config.initial_position_variance, 0.0],
                           [0.0,                              config.initial_velocity_variance],], dtype=float)

        # Initialise predicted state (prior) placeholders for use in update() from predict() step
        self._predicted_x: Optional[np.ndarray] = None
        self._predicted_P: Optional[np.ndarray] = None

        # Log init so cold-start position and P diagonal are auditable
        # NOTE: initialisation errors propagate through the entire run.
        logger.info(
            "TrackKalmanFilter init | seg=%s s=%.2fm v=%.3fm/s "
            "P_pos=%.2f P_vel=%.2f t=%.3f",
            segment_id, initial_position, initial_velocity,
            config.initial_position_variance,
            config.initial_velocity_variance,
            timestamp,
        )

    # -------------------------------------------------------------------------
    # Public API - KF steps
    # -------------------------------------------------------------------------

    def predict(self, dt: float) -> float:
        """Propagate/ predict state forward by dt seconds.
        Validates dt, builds F (process model)and Q (process noise covariance), with the CV motion model. 
        
        Constant Velocity (CV) model: F = [[1, dt], [0, 1]], Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]
        The structure of F and Q matrices determined by the CV model assumptions and standard discrete-time process noise derivation.
        
        Assumptions:
        - Motion model is constant velocity with process noise on acceleration.
        - State vector is [s, v]^T where s is along-track position and v is velocity.
        - Process noise density q is scalar and applies to the acceleration component 
            as a continuous white noise of the motion model, leading to the specific structure of Q.

        Called at IMU rate (50-200 Hz) for predict-only dead-reckoning between GNSS fixes, and at GNSS rate (1-5 Hz) before update().

        Args:
            dt: Elapsed time since last predict, seconds.

        Returns:
            Predicted along-track position x_prior[0], meters.
        """
        # Validate dt to reject degenerate values before building Q
        if dt < self.config.min_dt:
            # NOTE: Warn on too-small dt to catch duplicate timestamps from
            # the sensor interface that would produce a near-zero Q matrix.
            logger.warning(
                "predict | seg=%s dt=%.4fs below min_dt=%.3fs — skipping",
                self.segment_id, dt, self.config.min_dt,
            )
            return float(self.x[0])

        if dt > self.config.max_dt:
            # Re-initialise rather than skip. Skipping leaves _predicted_x stale,
            # causing the next update() to compute a large innovation against a
            # frozen prior — that innovation is absorbed into velocity via Kalman
            # gain cross-coupling, corrupting all subsequent estimates.
            #
            # Re-initialisation strategy:
            #   - Position: preserved (last accepted posterior — best known value).
            #   - Velocity: reset to 0.0 (genuinely unknown after a long gap;
            #               dead-reckoning over max_dt is unreliable).
            #   - P: reset to P0 (explicitly uninformed; the filter must rebuild
            #        confidence from new measurements).
            #   - _predicted_x/_predicted_P: set to current state so update()
            #     uses the re-initialised prior, not a stale pre-gap prior.
            #
            # NOTE: In production, a watchdog monitors GNSS heartbeat and flags
            # dropout before it reaches this path. This handler covers the edge
            # case gracefully without producing corrupt velocity estimates.
            logger.warning(
                "predict | seg=%s dt=%.2fs exceeds max_dt=%.1fs — "
                "re-initialising KF (position preserved, velocity=0, P=P0)",
                self.segment_id, dt, self.config.max_dt,
            )
            self.x[1] = 0.0
            self.P = np.array([
                [self.config.initial_position_variance, 0.0],
                [0.0,                                   self.config.initial_velocity_variance],
            ], dtype=float)
            self._predicted_x = self.x.copy()
            self._predicted_P = self.P.copy()
            return float(self.x[0])

        q = self.config.process_noise_density 
        # NOTE: process noise density applied to the acceleration component as a continuous white noise of the motion model, 
        # leading to the specific structure of process noise covariance Q.

        # Constant Velocity (CV) model: F = [[1, dt], [0, 1]]
        F = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])
        # Process noise covariance: Q = q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]
        Q = q * np.array([
            [dt**3 / 3.0,  dt**2 / 2.0],
            [dt**2 / 2.0,  dt],
        ])

        # Predict step: x_prior = F*x, P_prior = F*P*F^T + Q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Save prior for update()
        self._predicted_x = self.x.copy()
        self._predicted_P = self.P.copy()

        # Log predicted state at DEBUG to track filter divergence during GNSS dropouts 
        # NOTE: Only dead-reckoning advances the estimate.
        logger.debug(
            "predict | seg=%s dt=%.3fs s_pred=%.2fm v_pred=%.3fm/s "
            "P_pos=%.4f P_vel=%.4f",
            self.segment_id, dt,
            float(self.x[0]), float(self.x[1]),
            float(self.P[0, 0]), float(self.P[1, 1]),
        )

        return float(self.x[0])

    def update(
        self,
        s_proj: float,
        measurement_noise: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> KFUpdate:
        """Incorporate an incoming projected along-track measurement to update/ correct predictions (innovation) and covariances.

        Update step: 
            Computes innovation y = s_proj - H*x_prior, innovation covariance S = H*P_prior*H^T + R, Kalman gain K = P_prior*H^T / S, 
            and updates state x = x_prior + K*y and covariance P using Joseph form for numerical stability.
            Applies gating check on the normalised innovation (|y|/sqrt(S)) against the gate_threshold to reject outliers. 
            If gated, state and covariance are restored to the prior values.

        Called when GNSS fix is available; Applies measurement gating to reject outlier fixes.

        Args:
            s_proj: Along-track projection from from_node, meters.
                From TrackProjectionCoordinates.s_proj. Feeds the KF only and never combined with d_cross (two-layer architecture).
            measurement_noise: R = sigma_pos^2, meters^2. Optional: uses config default when fix does not report horizontal_accuracy.
            timestamp: POSIX time of this measurement. Optional.

        Returns:
            KFUpdate. If gated (accepted=False), state holds prior values.
        """

        if self._predicted_x is not None or self._predicted_P is not None:
            x_prior = self._predicted_x
            P_prior = self._predicted_P
        else:
            # predicted x_prior saved by predict()
            # EDGE CASE: fall back to initial / current state if update() is called without a preceding predict()
            logger.debug(
                "update | Kalman filter update called before predict() with s_proj=%.2fm. "
                "Falling back to current state as prior.",
                s_proj
            )
            x_prior = self.x.copy()
            P_prior = self.P.copy()

        # Measurement noise covariance: R
        # NOTE: Scalar since measurement is 1D (only position measured), meters^2.
        R = measurement_noise if measurement_noise is not None \
            else self.config.default_measurement_noise ** 2       # Default R if None

        # Log which R source was used to distinguish fix-specific vs default noise
        # NOTE: Important for diagnosing measurement quality.
        if measurement_noise is None:
            logger.debug(
                "update | seg=%s using default R=%.4f (no fix accuracy reported)",
                self.segment_id, R,
            )

        # Measurement model: z = H*x + w, where (H) = [1, 0] and white noise (w) ~ N(0, R)
        H = self._H

        # Innovation (residual): y = z - H*x_prior
        y = s_proj - (H @ x_prior).item()  # Scalar innovation in measurement space, meters.

        # Innovation (system) covariance: S = H*P_prior*H^T + R (in measurement space)
        S = (H @ P_prior @ H.T).item() + R # Scalar innovation covariance, meters^2.

        predicted_position = float(x_prior[0])

        # Gating: reject if |y| / sqrt(S) exceeds threshold
        # NOTE: # Avoid division by zero if S is extremely small due to numerical issues. Treat as infinite residual to trigger gating.
        normalised_residual = abs(y) / (S ** 0.5) if S > 1e-10 else float("inf") 
        if normalised_residual > self.config.gate_threshold:
            # Restore state to prior to avoid outlier measurement
            self.x = x_prior.copy()
            self.P = P_prior.copy()

            # NOTE: Warn on gated measurement so GNSS multipath spikes and
            # outlier fixes are visible in logs during GPS data analysis.
            logger.warning(
                "update GATED | seg=%s s_proj=%.2fm y=%.3fm "
                "normalised_residual=%.2f > gate=%.1f sigma",
                self.segment_id, s_proj, y,
                normalised_residual, self.config.gate_threshold,
            )

            return KFUpdate(
                state=self.get_state(),
                innovation=y,
                innovation_covariance=S,
                predicted_position=predicted_position,
                measurement=s_proj,
                measurement_noise=R,
                accepted=False,
            )

        # Kalman gain: K = P_prior*H^T / S
        K = (P_prior @ H.T) / S

        # State and covariance update
        self.x = x_prior + (K * y).flatten() 
        # NOTE: K is (2,1) and y is scalar. Flatten to ensure it's 1D (2,) for multiplication.
        
        # Joseph form covariance update for numerical stability: P = (I - K*H)*P_prior*(I - K*H)^T + K*R*K^T
        I = np.eye(2)
        self.P = (I - K @ H) @ P_prior @ (I - K @ H).T + K * R * K.T
        # NOTE: K * R * K.T since R is scalar. Use K @ R @ K.T if R is ever expanded to a matrix in future iterations.
        
        # self.P = (I - K @ H) @ P_prior  -->  Standard form can lead to numerical issues
        # NOTE: I used Joseph form covariance update instead of P = (I - K*H) @ P_prior, 
        # for enhanced numerical stability and to ensure P remains positive semi-definite, especially when S is small.
        

        # Store innovation for consistency monitoring 
        self._innovation_history.append(y)
        self._update_count += 1

        if timestamp is not None:
            self.timestamp = timestamp

        # Log full update result so innovation, gain, and post-update states
        # are available for NIS plots and P convergence checks during tuning sessions.
        logger.debug(
            "update | seg=%s s_proj=%.2fm R=%.4f | "
            "y=%.3fm S=%.4f K=[%.4f, %.4f] | "
            "s_post=%.2fm v_post=%.3fm/s P_pos=%.4f P_vel=%.4f",
            self.segment_id, s_proj, R,
            y, S, (K[0]).item(), (K[1]).item(),
            float(self.x[0]), float(self.x[1]),
            float(self.P[0, 0]), float(self.P[1, 1]),
        )

        return KFUpdate(
            state=self.get_state(),
            innovation=y,
            innovation_covariance=S,
            predicted_position=predicted_position,
            measurement=s_proj,
            measurement_noise=R,
            accepted=True,
        )

    # -------------------------------------------------------------------------
    # State access
    # -------------------------------------------------------------------------

    def get_state(self) -> KFState:
        """Return the current state stored in the Kalman filter.

        Returns:
            KFState with current position, velocity, variances, segment, timestamp.
        """
        return KFState(
            position=float(self.x[0]),
            velocity=float(self.x[1]),
            position_var=float(self.P[0, 0]),
            velocity_var=float(self.P[1, 1]),
            segment_id=self.segment_id,
            timestamp=self.timestamp,
        )

    # -------------------------------------------------------------------------
    # Hypothesis lifecycle support
    # -------------------------------------------------------------------------

    def clone(self) -> TrackKalmanFilter:
        """Deep copy this filter for hypothesis spawning at a switch.

        The child gets identical and independent x, P, config, segment_id, and update_count.

        Called by the orchestrator at switch detection:
            child_kf = parent_kf.clone()
            child_kf.transition_to_segment(branch_seg_id, position_offset=0.0)

        Returns:
            New TrackKalmanFilter with identical state.
        """
        new_kf = TrackKalmanFilter(
            segment_id=self.segment_id,
            initial_position=float(self.x[0]),
            config=self.config,
            initial_velocity=float(self.x[1]),
            timestamp=self.timestamp,
        )
        # Overwrite P with the current (possibly non-diagonal) covariance
        new_kf.x = self.x.copy()
        new_kf.P = self.P.copy()
        new_kf._update_count = self._update_count
        # NOTE: Do not copy _innovation_history, so its own consistency stats are not contaminated by the parent's history.

        logger.debug(
            "clone | seg=%s s=%.2fm v=%.3fm/s update_count=%d",
            self.segment_id, float(self.x[0]), float(self.x[1]),
            self._update_count,
        )

        return new_kf

    def transition_to_segment(
        self,
        new_segment_id: str,
        position_offset: float,
    ) -> None:
        """Reset position to the start of a new segment after rollover or spawn.

        Updates segment_id and resets s to position_offset. 
        Velocity and covariance carry over unchanged since they are valid across boundaries.

        Called externally by the orchestrator:
            Rollover : position_offset = predicted_s - old_segment_length
            Spawn    : position_offset = 0.0 (start of new branch)

        Args:
            new_segment_id: ID of the segment to transition onto.
            position_offset: New along-track position on the new segment, meters.
        """
        old_seg = self.segment_id
        self.segment_id = new_segment_id
        self.x[0] = position_offset

        # Keep _predicted_x consistent so update() uses the post-transition
        # position as its prior, not the pre-transition one.
        if self._predicted_x is not None:
            self._predicted_x[0] = position_offset

        # Log segment transition so rollover and spawn events are auditable 
        # NOTE: Wrong segment_id here corrupts all subsequent boundary checks and curvature lookups.
        logger.info(
            "transition | %s -> %s position_offset=%.2fm v=%.3fm/s",
            old_seg, new_segment_id, position_offset, float(self.x[1]),
        )

    # -------------------------------------------------------------------------
    # Diagnostics: Consistency monitoring
    # -------------------------------------------------------------------------

    def get_innovation_stats(self, window: int = 20) -> Tuple[float, float]:
        """Return mean and variance of recent innovations for health monitoring.

        Used by the Week 2 consistency monitor to detect systematic bias
        (wrong segment) or model mismatch (noise parameters mistuned).

        Expected healthy values (normalised innovations):
            mean ~= 0   --> unbiased filter, correct segment
            var  ~= 1   --> noise parameters well-tuned
        Diagnostics:
            mean drifts      --> systematic bias: map error or wrong segment
            var >> 1         --> measurement noise underestimated (widen R or q)
            var << 1         --> measurement noise overestimated (tighten R or q)

        Args:
            window: Number of recent innovations to use. Default 20.

        Returns:
            (mean, variance) of the innovation window. Returns (0.0, 1.0)
            if fewer than 3 updates have occurred (insufficient data).
        """
        recent = self._innovation_history[-window:]
        if len(recent) < 3:
            # Return neutral values when history is too short 
            # NOTE: Avoids false consistency alarms in the first few fixes after cold start.
            return 0.0, 1.0

        mean = float(np.mean(recent))
        var  = float(np.var(recent))

        # Log innovation stats so systematic drift or variance anomalies appear in the log.
        if abs(mean) > 1.0 or var > 3.0:
            logger.warning(
                "innovation stats | seg=%s mean=%.3fm var=%.3f "
                "(mean>1m or var>3 indicates bias or model mismatch)",
                self.segment_id, mean, var,
            )

        return mean, var
