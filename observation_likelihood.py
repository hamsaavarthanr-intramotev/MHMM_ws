# =============================================================================
# observation_likelihood.py
# Onboard Localization Engine — Observation Likelihood Model

# Responsibilities:
#   - Evaluate how well a sensor observation fits a given hypothesis
#   - Compute position likelihood L_pos from cross-track distance
#   - Compute heading likelihood L_head from measured vs expected track heading
#   - Compute curvature likelihood L_curv from IMU yaw rate vs map curvature
#   - Combine into L_total for Bayesian weight update by the orchestrator

# ROLE IN THE MHMM:
#   Pure scoring function — no internal state. Does not modify the KF or
#   hypothesis. Called once per hypothesis per update cycle.
#   Orchestrator update pattern:
#       result = compute_likelihood(fix, projection, kf_state,
#                                   track_map_engine, scoring_config, direction_qualifier)
#       h.weight *= result.L_total
#   Weights are then normalised across all active hypotheses.

# TWO-LAYER ARCHITECTURE RESPECTED:
#   L_pos  uses projection.d_cross        — cross-track distance (weight system)
#   L_head uses projection.track_heading_rad — map heading at the projected point
#   L_curv uses kf_state.position         — along-track position (KF estimate)
#   These inputs are never mixed. See MHMM_Project_Context.md.

# L_HEAD DESIGN:
#   L_head compares the sensor heading (fix.heading, compass degrees) against the
#   expected track heading (projection.track_heading_rad, ENU radians) per hypothesis.
#   It activates only when direction_qualifier != "unknown" (the railcar has cleared
#   the direction_hysteresis_threshold in the orchestrator, so dq acts as the
#   velocity gate — no separate heading_velocity_gate is needed).
#
#   DIRECTION RESOLUTION — nearest-heading comparison (NOT dq-based flip):
#   A segment has two candidate expected headings:
#       h_nom = track_heading_rad         (first_node→last_node)
#       h_rev = track_heading_rad + π     (last_node→first_node)
#   The correct expected heading is whichever candidate is geometrically closer to
#   fix.heading. fix.heading (Certus dual-antenna) is an absolute vehicle orientation —
#   it is stable and independent of KF velocity or direction_qualifier.
#   This eliminates the catastrophic L_head=0 failure that occurred when dq
#   flickered at near-zero KF velocity while the physical vehicle heading was unchanged.
#
#   L_head is an absolute Gaussian PDF (no peak-normalisation):
#       L_head = gaussian_likelihood(heading_residual_rad, sigma_heading_eff)
#
#   sigma_heading_eff is a two-term RSS:
#       sigma_heading_eff = sqrt(sigma_heading² + sigma_heading_sensor²)
#   where sigma_heading_sensor = radians(fix.heading_accuracy) when available,
#   else the sensor term is zero and sigma_heading alone is used.

# L_CURV DESIGN:
#   L_curv is computed per-hypothesis independently. Each hypothesis evaluates its own omega_expected
#   against the sensor measurement. The Bayesian likelihood ratio between
#   competing hypotheses emerges naturally from their individual L_curv values:
#   a hypothesis whose map curvature matches the sensor is rewarded; one that
#   disagrees is penalised.
#
#   L_curv is a true Gaussian PDF (absolute scale):
#       L_curv = gaussian_likelihood(omega_res, sigma_omega_eff)
#              = (1/sqrt(2π·sigma²)) · exp(-omega_res²/(2·sigma²))
#
#   Peak-normalisation (dividing by gaussian(0, sigma)) has been deliberately
#   removed. That operation stripped L_curv of its absolute scale, making it
#   impossible to distinguish a perfect match (omega_res≈0) from a mismatch on
#   a track with a different sigma. With absolute scale, L_total = L_pos × L_curv
#   is a proper joint likelihood, and the chi-squared integrity gate in the
#   orchestrator can use d2_curv = (omega_res/sigma_omega_eff)² directly.
#
#   L_curv activates when enable_curvature=True AND:
#       1. in_decision_area — within d_decision of a switch, OR within
#                             d_decision of the hypothesis's spawn point
#       2. gyro_yaw_rate available on the fix
#
#   There is no minimum velocity gate. At v≈0, omega_expected≈0 and
#   omega_residual≈omega_sensor (gyro noise), so L_curv contributes equally
#   to all hypotheses and the ratio is ~1 — no distortion, no gate needed.

# SIGN CONTRACT:
#   direction_qualifier ("nominal"/"reverse") is passed through to
#   track_map_engine.distance_to_nearest_switch(). This module applies no sign
#   corrections itself. Curvature is returned as stored by get_curvature();
#   the KF velocity sign in omega_expected = velocity × kappa handles direction.
#   Heading direction is resolved by the +π flip on "reverse" direction_qualifier.
# =============================================================================

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

import yaml

from sensor_interface import SensorFix
from track_map_engine import TrackMapEngine, TrackProjectionCoordinates
from kalman_filter import KFState

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LikelihoodResult:
    """Output of a single hypothesis likelihood evaluation.

    Consumed by the MHMM orchestrator for Bayesian weight updates and diagnostics.

    Attributes:
        L_total: Combined observation likelihood (L_pos × L_head × L_curv, or
            any active subset). Multiply directly into hypothesis weight.
        L_pos: Position likelihood. Always computed. Absolute Gaussian PDF on d_cross.
        L_head: Heading likelihood. Absolute Gaussian PDF on heading_residual_rad.
            None when direction_qualifier=="unknown" or enable_heading=False.
        L_curv: Curvature likelihood. Absolute Gaussian PDF on omega_residual.
            None when outside decision area, gyro unavailable, or enable_curvature=False.
        in_decision_area: True when hypothesis is within d_decision of a switch
            (forward proximity) or within d_decision of its spawn point
            (backward proximity). Controls whether L_curv is attempted.
        d_cross: Cross-track distance fed to L_pos, meters. For logging.
        heading_expected: Expected track heading in ENU radians selected by nearest-heading
            comparison to fix.heading (whichever of h_nom / h_rev is closer).
            None when L_head was not computed.
        heading_residual: heading_measured_enu − heading_expected, radians, wrapped (−π, π].
            None when L_head was not computed.
        sigma_heading_effective: Effective sigma used for L_head Gaussian, radians.
            None when L_head was not computed.
        omega_expected: Map-derived expected yaw rate, rad/s.
            None when L_curv was not computed.
        omega_residual: omega_sensor − omega_expected, rad/s.
            None when L_curv was not computed.
        sigma_omega_effective: Effective sigma used for L_curv Gaussian, rad/s.
            None when L_curv was not computed.
        d2_pos: (d_cross / sigma_pos_effective)² — chi-squared(1) for integrity gate.
        d2_head: (heading_residual / sigma_heading_eff)² — chi-squared(1) for integrity gate.
            None when L_head was not computed.
        d2_curv: (omega_residual / sigma_omega_eff)² — chi-squared(1) for integrity gate.
            None when L_curv was not computed.
    """
    L_total: float
    L_pos: float
    L_head: Optional[float]
    L_curv: Optional[float]
    in_decision_area: bool
    d_cross: float
    heading_expected: Optional[float]
    heading_residual: Optional[float]
    sigma_heading_effective: Optional[float]
    omega_expected: Optional[float]
    omega_residual: Optional[float]
    sigma_omega_effective: Optional[float]
    d2_pos: float = 0.0
    d2_head: Optional[float] = None
    d2_curv: Optional[float] = None


@dataclass
class ScoringConfig:
    """Observation likelihood parameters. Loaded from config.yaml kalman_filter section.

    Attributes:
        sigma_gyro: Hardware angular rate noise from IMU datasheet (rad/s).
            This is a physical constant — do not tune. Read from spec sheet.
            Certus: gyroscope in-run bias instability ~0.005 rad/s (1-sigma).

        sigma_kappa_map: Irreducible map curvature uncertainty (1/m).
            Reflects RTK survey point spacing and polyline chord approximation
            error in curvature. At velocity v (m/s), contributes sigma_kappa_map×v
            to the omega residual sigma (rad/s). Tune from empirical omega_residual
            distribution on straight track (where kappa=0, so residual = gyro noise
            only, and any excess variance is map kappa error).
            Derived default: 0.002 1/m from 0.012m RMS RTK survey with ~0.5m
            point spacing → chord curvature error ≈ 8×point_error/chord² ≈ 0.002.

        d_decision: Distance from switch node (m) within which L_curv is
            attempted. L_curv is also active within d_decision of the hypothesis's
            spawn point (backward proximity), covering the post-switch zone.
            Typical: 100–200 m.

        default_sigma_pos: Fallback position sigma (m) when SensorFix does not
            report horizontal_accuracy.
            NOTE: Shares kalman_filter.default_measurement_noise from config.

        sigma_map: Irreducible positional uncertainty from track map quality (m).
            Accounts for polyline discretisation error, survey bias, and
            antenna offset calibration uncertainty. Combined with GNSS accuracy
            and KF position variance via three-term RSS:
            sigma_pos_effective = sqrt(sigma_gnss² + sigma_map² + P_pos).
    """
    # NOTE: sigma_gyro and sigma_kappa_map are physical constants, not tuning parameters.
    # sigma_gyro: from IMU datasheet; sigma_kappa_map: from RTK survey quality.
    sigma_gyro: float = 0.005
    sigma_kappa_map: float = 0.002
    d_decision: float = 150.0
    default_sigma_pos: float = 2.5
    sigma_map: float = 0.5
    # Heading likelihood parameters
    enable_heading: bool = True
    sigma_heading: float = 0.0262        # 1.5° in radians — base map heading uncertainty
    # Curvature likelihood enable flag
    enable_curvature: bool = False       # disabled pending survey-noise resolution

    @classmethod
    def from_config(cls, config_path: str) -> ScoringConfig:
        """Load ScoringConfig from the kalman_filter section of config.yaml.

        Args:
            config_path: Path to config.yaml.

        Returns:
            ScoringConfig populated from config.yaml.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        kf = cfg["kalman_filter"]
        config = cls(
            sigma_gyro=float(kf.get("sigma_gyro", 0.005)),
            sigma_kappa_map=float(kf.get("sigma_kappa_map", 0.002)),
            d_decision=float(kf["d_decision"]),
            default_sigma_pos=float(kf.get("default_measurement_noise", 2.5)),
            sigma_map=float(kf.get("sigma_map", 0.5)),
            enable_heading=bool(kf.get("enable_heading", True)),
            sigma_heading=float(kf.get("sigma_heading", 0.0262)),
            enable_curvature=bool(kf.get("enable_curvature", False)),
        )

        logger.info(
            "ScoringConfig loaded | sigma_gyro=%.4f rad/s sigma_kappa_map=%.4f 1/m "
            "d_decision=%.1fm default_sigma_pos=%.2fm sigma_map=%.3fm | "
            "enable_heading=%s sigma_heading=%.4f rad (%.2f°) | "
            "enable_curvature=%s",
            config.sigma_gyro,
            config.sigma_kappa_map,
            config.d_decision,
            config.default_sigma_pos,
            config.sigma_map,
            config.enable_heading,
            config.sigma_heading,
            math.degrees(config.sigma_heading),
            config.enable_curvature,
        )

        return config


# =============================================================================
# Core math
# =============================================================================

def gaussian_likelihood(residual: float, sigma: float) -> float:
    """Evaluate the Gaussian PDF at residual (x-μ) given standard deviation sigma.

    Likelihood = (1/sqrt(2π sigma²)) * exp(-0.5 * (residual/sigma)²)

    The normalisation constant 1/sqrt(2π sigma²) is included so likelihoods
    from different sigma values remain comparable across hypotheses.

    Args:
        residual: Observed minus expected value (meters or rad/s).
        sigma: Standard deviation of the expected noise (meters or rad/s).

    Returns:
        Likelihood value > 0.
    """
    # NOTE: NO mathematical checks for sigma > 0??
    return (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * math.exp(
        -(residual ** 2) / (2.0 * sigma ** 2)
    )



# =============================================================================
# Main entry point
# =============================================================================

def compute_likelihood(
    fix: SensorFix,
    projection: TrackProjectionCoordinates,
    kf_state: KFState,
    track_map_engine: TrackMapEngine,
    scoring_config: ScoringConfig,
    direction_qualifier: str = "nominal",
    distance_since_spawn: Optional[float] = None,
) -> LikelihoodResult:
    """Compute the observation likelihood for a single hypothesis.

    Called once per hypothesis per GNSS update cycle by the MHMM orchestrator.

    L_curv activation requires:
        1. in_decision_area — within d_decision of a switch (forward proximity)
                              OR within d_decision of the spawn point (backward
                              proximity, covering the post-switch zone).
        2. gyro_yaw_rate available on the fix.

    L_curv is an absolute Gaussian PDF value (peak-normalisation removed).
    L_total = L_pos * L_curv is a proper joint likelihood when L_curv is active.

    DECISION AREA — TWO-SIDED CHECK:
        in_decision_area is True when EITHER:
          (a) distance_to_nearest_switch < d_decision — approaching a switch
              ahead in the direction of travel (forward proximity), OR
          (b) distance_since_spawn < d_decision — recently spawned from a
              switch (backward proximity).

    Args:
        fix: Current SensorFix. Provides horizontal_accuracy and gyro_yaw_rate.
        projection: Provides d_cross for L_pos.
        kf_state: Post-update KFState. Provides position, velocity, P_pos, P_vel.
        track_map_engine: For get_curvature() and distance_to_nearest_switch().
        scoring_config: ScoringConfig with all likelihood parameters.
        direction_qualifier: "nominal" or "reverse". For switch proximity query.
        distance_since_spawn: Along-track distance traveled by this hypothesis
            since it was spawned (abs(kf.position - spawn_s)), metres.

    Returns:
        LikelihoodResult with all fields populated.
    """

    # -------------------------------------------------------------------------
    # Step 1: sigma_gnss — directional cross-track projection of GNSS error
    #
    # d_cross is a 1D cross-track measurement. The GNSS error ellipse must be
    # projected onto the cross-track axis to match dimensions correctly.
    #
    # THREE CASES (in priority order):
    #
    # Case A — per-axis stddevs available (fix.lat_stddev_m, fix.lon_stddev_m)
    #   AND track heading available (projection.track_heading_rad):
    #   Project the 2D error ellipse onto the cross-track axis:
    #     sigma_cross = sqrt((lon_stddev_m × sin(heading))²
    #                      + (lat_stddev_m × cos(heading))²)
    #   ENU convention: heading = 0 → East, π/2 → North.
    #   East component of perpendicular = sin(heading),
    #   North component of perpendicular = cos(heading).
    #   This is the physically exact 1D cross-track sigma.
    #
    # Case B — only horizontal_accuracy available (isotropic approximation):
    #   h_acc = sqrt(σ_east² + σ_north²) = sqrt(2) × σ_1d for isotropic error.
    #   σ_cross = h_acc / sqrt(2).
    #   Removes the 41% overestimation from using the 2D RSS as a 1D sigma.
    #
    # Case C — nothing available: fall back to default_sigma_pos.
    #
    # sigma_pos_effective (three-term RSS):
    #   sigma_pos_effective = sqrt(sigma_gnss² + sigma_map² + P_pos)
    #
    #   sigma_gnss — directional projection, isotropic approximation, or default fallback.
    #   sigma_map  — Irreducible map/calibration floor (config, constant).
    #   P_pos      — KF position variance (runtime, self-regulating).
    #                Large at cold start, negligible after convergence.
    # -------------------------------------------------------------------------
    if (fix.lat_stddev_m is not None and fix.lat_stddev_m > 0.0
            and fix.lon_stddev_m is not None and fix.lon_stddev_m > 0.0):
        # Case A: directional projection onto cross-track axis.
        heading = projection.track_heading_rad
        sin_h = math.sin(heading)
        cos_h = math.cos(heading)
        sigma_gnss = math.sqrt(
            (fix.lon_stddev_m * sin_h) ** 2
            + (fix.lat_stddev_m * cos_h) ** 2
        )
    elif fix.horizontal_accuracy is not None and fix.horizontal_accuracy > 0.0:
        # Case B: isotropic approximation — remove 2D-to-1D overestimation.
        sigma_gnss = fix.horizontal_accuracy / math.sqrt(2.0)
    else:
        # Case C: no GNSS accuracy data — use config default.
        logger.warning(
            "compute_likelihood | seg=%s no GNSS accuracy data available — "
            "sigma_gnss fallback to default=%.2fm",
            kf_state.segment_id,
            scoring_config.default_sigma_pos,
        )
        sigma_gnss = scoring_config.default_sigma_pos


    P_pos = kf_state.position_var
    sigma_pos_effective = math.sqrt(
        sigma_gnss ** 2 + scoring_config.sigma_map ** 2 + P_pos
    )

    # Positive sigma_pos_effective guard
    if sigma_pos_effective <= 0.0:
        logger.warning(
            "compute_likelihood | seg=%s sigma_pos_effective=%.6f <= 0 — "
            "clamping to default_sigma_pos=%.2fm",
            kf_state.segment_id, sigma_pos_effective, scoring_config.default_sigma_pos,
        )
        sigma_pos_effective = scoring_config.default_sigma_pos

    # -------------------------------------------------------------------------
    # Step 2: L_pos — always computed
    # -------------------------------------------------------------------------
    L_pos = gaussian_likelihood(projection.d_cross, sigma_pos_effective)
    # d2_pos: squared normalised cross-track residual ~ chi-squared(1).
    # Used by the orchestrator chi-squared integrity gate (Approach A).
    d2_pos = (projection.d_cross / sigma_pos_effective) ** 2

    # -------------------------------------------------------------------------
    # Step 3: Decision area check (two-sided)
    #
    # Case A — forward proximity: approaching a switch node in the direction
    #   of travel. distance_to_nearest_switch uses direction_qualifier.
    #
    # Case B — backward proximity: recently spawned from a switch and still
    #   within d_decision of the spawn point. Handles branching segments whose
    #   far node is not a switch — distance_to_nearest_switch returns inf there,
    #   but L_curv is most useful precisely in the post-switch zone.
    # -------------------------------------------------------------------------
    d_switch = track_map_engine.distance_to_nearest_switch(
        kf_state.segment_id, kf_state.position, direction_qualifier
    )
    # Forward Proximity
    forward_proximity  = d_switch < scoring_config.d_decision
    # Backward Proximity
    if distance_since_spawn is None:
        logger.debug(
            "compute_likelihood | seg=%s distance_since_spawn is None — "
            "backward proximity check skipped",
            kf_state.segment_id,
        )
        backward_proximity = False
    else:
        backward_proximity = distance_since_spawn < scoring_config.d_decision

    in_decision_area = forward_proximity or backward_proximity

    # -------------------------------------------------------------------------
    # Step 4a: L_curv — computed per-hypothesis within the decision area
    #          Gated by enable_curvature config flag.
    #
    # L_curv is an absolute Gaussian PDF (peak-normalisation removed).
    # L_total = L_pos × L_curv is a proper joint likelihood when active.
    # d2_curv = (omega_residual/sigma_omega_eff)² — chi-squared(1) for integrity gate.
    # -------------------------------------------------------------------------
    L_curv: Optional[float] = None
    omega_expected: Optional[float] = None
    omega_residual: Optional[float] = None
    sigma_omega_eff: Optional[float] = None
    d2_curv: Optional[float] = None

    if scoring_config.enable_curvature and in_decision_area:
        if fix.gyro_yaw_rate is None:
            logger.debug(
                "compute_likelihood | seg=%s in decision area but "
                "gyro_yaw_rate is None — L_curv skipped",
                kf_state.segment_id,
            )
        else:
            kappa = track_map_engine.get_curvature(
                kf_state.segment_id, kf_state.position
            )

            omega_expected = kf_state.velocity * kappa
            omega_residual = fix.gyro_yaw_rate - omega_expected

            # sigma_omega_eff: Three-term RSS mirroring sigma_pos_effective:
            #   sigma_gyro²              ← hardware noise floor (constant)
            #   (sigma_kappa_map × |v|)² ← map curvature uncertainty, speed-scaled
            #   (|kappa| × sigma_v)²     ← velocity uncertainty projected onto omega
            P_vel = kf_state.velocity_var
            v = abs(kf_state.velocity)
            sigma_v = math.sqrt(P_vel) if P_vel > 0.0 else 0.0

            sigma_omega_eff = math.sqrt(
                scoring_config.sigma_gyro ** 2
                + (scoring_config.sigma_kappa_map * v) ** 2
                + (abs(kappa) * sigma_v) ** 2
            )

            if sigma_omega_eff <= 0.0:
                logger.warning(
                    "compute_likelihood | seg=%s sigma_omega_eff=%.6f <= 0 — "
                    "clamping to sigma_gyro=%.5f rad/s",
                    kf_state.segment_id, sigma_omega_eff, scoring_config.sigma_gyro,
                )
                sigma_omega_eff = max(scoring_config.sigma_gyro, 1e-6)

            L_curv = gaussian_likelihood(omega_residual, sigma_omega_eff)
            d2_curv = (omega_residual / sigma_omega_eff) ** 2

            # Log Curvature Likelihood
            logger.debug(
                "compute_likelihood | seg=%s kappa=%.6f 1/m "
                "v=%.3fm/s omega_exp=%.4f rad/s omega_sens=%.4f rad/s "
                "omega_res=%.4f rad/s sigma_omega_eff=%.5f rad/s L_curv=%.6e",
                kf_state.segment_id,
                kappa,
                kf_state.velocity,
                omega_expected,
                fix.gyro_yaw_rate,
                omega_residual,
                sigma_omega_eff,
                L_curv,
            )

    # -------------------------------------------------------------------------
    # Step 4b: L_head — heading likelihood
    #          Gated by enable_heading AND direction_qualifier != "unknown".
    #          direction_qualifier acts as the velocity gate: it is set to
    #          "unknown" until the KF velocity clears direction_hysteresis_threshold,
    #          ensuring the vehicle is genuinely in motion before heading is scored.
    #
    # DIRECTION CONVENTION:
    #   fix.heading is an absolute vehicle orientation from the Certus dual-antenna
    #   (compass degrees, 0=N, 90=E, clockwise). It reports the physical chassis
    #   direction regardless of whether the vehicle is moving, slowing, or stopped.
    #   It does NOT depend on KF velocity sign or direction_qualifier.
    #
    #   projection.track_heading_rad is the ENU heading of the segment in the
    #   first_node→last_node direction (atan2(dN, dE), East=0, ccw positive).
    #
    #   A segment can be traversed in either physical direction. The two candidate
    #   expected headings are:
    #       h_nom = track_heading_rad            (first_node→last_node)
    #       h_rev = track_heading_rad + π        (last_node→first_node)
    #
    #   The correct expected heading is whichever candidate is geometrically closer
    #   to fix.heading, determined purely from the sensor measurement without any
    #   dependence on direction_qualifier or KF velocity sign.
    #
    #   Frame conversion:  heading_meas_enu = π/2 − radians(fix.heading)
    #   Residual wrapped to (−π, π] before Gaussian evaluation.
    #
    # sigma_heading_eff: two-term RSS
    #   sigma_heading_eff = sqrt(sigma_heading² + sigma_heading_sensor²)
    #   sigma_heading_sensor = radians(fix.heading_accuracy) when available, else 0.
    # -------------------------------------------------------------------------
    L_head: Optional[float] = None
    heading_expected: Optional[float] = None
    heading_residual: Optional[float] = None
    sigma_heading_eff: Optional[float] = None
    d2_head: Optional[float] = None

    if scoring_config.enable_heading and direction_qualifier != "unknown":
        if fix.heading is None:
            logger.debug(
                "compute_likelihood | seg=%s enable_heading=True but "
                "fix.heading is None — L_head skipped",
                kf_state.segment_id,
            )
        else:
            # Convert measured compass heading → ENU radians
            # ENU convention: 0 rad = East, π/2 rad = North, counter-clockwise positive, range (−π, π]
            heading_meas_enu = math.pi / 2.0 - math.radians(fix.heading)

            # Both candidate expected headings for this segment in ENU radians
            h_nom = projection.track_heading_rad             # first_node→last_node
            h_rev = projection.track_heading_rad + math.pi   # last_node→first_node

            # Residuals toward each candidate, wrapped to (−π, π]
            def _wrap(a: float) -> float:
                return (a + math.pi) % (2.0 * math.pi) - math.pi

            res_nom = _wrap(heading_meas_enu - h_nom)
            res_rev = _wrap(heading_meas_enu - h_rev)

            # Select the candidate geometrically closer to the measured heading.
            # fix.heading (dual-antenna) is the authoritative physical orientation —
            # independent of KF velocity or direction_qualifier.
            if abs(res_nom) <= abs(res_rev):
                heading_expected = _wrap(h_nom)
                heading_residual = res_nom
            else:
                heading_expected = _wrap(h_rev)
                heading_residual = res_rev

            # sigma_heading_eff: base map uncertainty RSS-combined with sensor accuracy
            sigma_sensor = (
                math.radians(fix.heading_accuracy)
                if fix.heading_accuracy is not None and fix.heading_accuracy > 0.0
                else 0.0
            )
            sigma_heading_eff = math.sqrt(
                scoring_config.sigma_heading ** 2 + sigma_sensor ** 2
            )

            if sigma_heading_eff <= 0.0:
                logger.warning(
                    "compute_likelihood | seg=%s sigma_heading_eff=%.6f <= 0 — "
                    "clamping to sigma_heading=%.5f rad",
                    kf_state.segment_id, sigma_heading_eff, scoring_config.sigma_heading,
                )
                sigma_heading_eff = max(scoring_config.sigma_heading, 1e-6)

            L_head = gaussian_likelihood(heading_residual, sigma_heading_eff)
            d2_head = (heading_residual / sigma_heading_eff) ** 2

            # Log Heading Likelihood
            logger.debug(  
                "compute_likelihood | seg=%s dq=%s "
                "heading_meas_enu=%.2f° h_nom=%.4f rad (%.2f°) h_rev=%.4f rad (%.2f°) "
                "heading_exp=%.4f rad (%.2f°) heading_res=%.4f rad (%.2f°) "
                "sigma_head_eff=%.5f rad L_head=%.6e",
                kf_state.segment_id,
                direction_qualifier,
                math.degrees(heading_meas_enu),
                h_nom, math.degrees(_wrap(h_nom)),
                h_rev, math.degrees(_wrap(h_rev)),
                heading_expected, math.degrees(heading_expected),
                heading_residual, math.degrees(heading_residual),
                sigma_heading_eff,
                L_head,
            )

    # -------------------------------------------------------------------------
    # Step 5: Combine — L_total = L_pos × L_head × L_curv (active terms only)
    # -------------------------------------------------------------------------
    L_total = L_pos
    if L_head is not None:
        L_total *= L_head
    if L_curv is not None:
        L_total *= L_curv

    # -------------------------------------------------------------------------
    # Step 6: Log and return
    # -------------------------------------------------------------------------
    # Log Combined Likelihood
    logger.debug(
        "compute_likelihood | seg=%s d_cross=%.3fm sigma_pos_eff=%.3fm "
        "(sigma_gnss=%.3fm [%s] sigma_map=%.3fm P_pos=%.4fm²) "
        "L_pos=%.6e | L_head=%s | in_decision=%s L_curv=%s | L_total=%.6e",
        kf_state.segment_id,
        projection.d_cross,
        sigma_pos_effective,
        sigma_gnss,
        "projected" if (fix.lat_stddev_m is not None and fix.lat_stddev_m > 0.0)
            else "isotropic" if (fix.horizontal_accuracy is not None and fix.horizontal_accuracy > 0.0)
            else "default",
        scoring_config.sigma_map,
        P_pos,
        L_pos,
        f"{L_head:.6e}" if L_head is not None else "None",
        in_decision_area,
        f"{L_curv:.6e}" if L_curv is not None else "None",
        L_total,
    )

    if L_total < 1e-10:
        logger.warning(
            "compute_likelihood | seg=%s L_total=%.3e extremely small; "
            "hypothesis may be on wrong segment or sensor quality is poor",
            kf_state.segment_id,
            L_total,
        )

    return LikelihoodResult(
        L_total=L_total,
        L_pos=L_pos,
        L_head=L_head,
        L_curv=L_curv,
        in_decision_area=in_decision_area,
        d_cross=projection.d_cross,
        heading_expected=heading_expected,
        heading_residual=heading_residual,
        sigma_heading_effective=sigma_heading_eff,
        omega_expected=omega_expected,
        omega_residual=omega_residual,
        sigma_omega_effective=sigma_omega_eff,
        d2_pos=d2_pos,
        d2_head=d2_head,
        d2_curv=d2_curv,
    )
