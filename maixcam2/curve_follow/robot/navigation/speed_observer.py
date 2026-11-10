"""Read-only online speed-profile evaluation; no command or actuator interface."""

from dataclasses import asdict
import math

from robot.core.geometry import arc, describe_curve
from robot.navigation.speed_profile import SpeedLimits, plan_speed_profile
from robot.navigation.supported_path import supported_segment


class SpeedObserver:
    """Compute hypothetical supported-segment limits without changing driving.

    Maximum speed 0.30m/s rounds down the provisional wood-floor PWM200
    reference (about 0.305m/s). Accelerations are analytical assumptions, not
    measured tyre friction. Existing runtime braking and response bounds are
    reused only for this observer. Task stops and body rollout are not solved.
    """

    def __init__(self, settings):
        crossing = settings.crosswalk_policy
        self.limits = SpeedLimits(
            0.30, 0.15, 0.15,
            float(crossing.approach_deceleration or 0.08),
            settings.road_policy.missing_command_s + crossing.speed_response_s,
            settings.turn_lever,
        )
        self.maximum_age_s = settings.road_policy.missing_command_s
        self.calibrated = settings.calibrated
        frame = settings.runtime.get("control_frame", {})
        body = frame.get("body_m")
        self.front_m = (
            float(body[0]) if frame.get("geometry_verified") and body is not None
            and len(body) == 3 and math.isfinite(body[0]) and body[0] > 0 else None
        )
        self.parameters = dict(
            asdict(self.limits), front_margin_m=self.front_m, sigma_multiplier=2.0,
            provisional=True, basis="WOOD_FLOOR_20261110_AND_UNVALIDATED_DYNAMICS",
        )

    def update(self, target, frame, age_s):
        snapshot = dict(
            mode="SHADOW", controls_motor=False, source_frame=frame,
            scope="SUPPORTED_SEGMENT_ENTRY_ONLY", valid=False,
            parameters=dict(self.parameters),
        )
        # Analysis failures must not become driving faults or stale success.
        try:
            if not self.calibrated:
                snapshot["reason"] = "CALIBRATION_REQUIRED"
            elif self.front_m is None:
                snapshot["reason"] = "BODY_GEOMETRY_REQUIRED"
            elif not math.isfinite(age_s) or not 0 <= age_s <= self.maximum_age_s:
                snapshot["reason"] = "OBSERVATION_STALE"
            else:
                segment, reason = supported_segment(target)
                if segment is None:
                    snapshot["reason"] = reason
                    return snapshot
                sigma = target.get("reference_sigma_m")
                curve = describe_curve(segment)
                if (curve is None or sigma is None or not math.isfinite(sigma) or sigma < 0):
                    snapshot["reason"] = "INVALID_GEOMETRY_OR_UNCERTAINTY"
                    return snapshot
                positions = arc(curve["path"])
                profile = plan_speed_profile(
                    positions, curve["curvature"], self.limits,
                    distance_margin_m=self.front_m + 2.0 * sigma,
                )
                snapshot.update(
                    valid=True, reason=profile.entry_reason,
                    entry_limit_m_s=profile.entry_limit_m_s,
                    limiting_station_m=profile.limiting_station_m,
                    path_first_point_m=segment[0].tolist(),
                    supported_length_m=float(positions[-1]),
                    usable_length_m=profile.stations_m[-1],
                    sample_count=len(profile.stations_m),
                    observation_age_s=age_s,
                )
        except Exception as error:
            snapshot.update(valid=False, reason="ANALYSIS_ERROR", error=type(error).__name__)
        return snapshot
