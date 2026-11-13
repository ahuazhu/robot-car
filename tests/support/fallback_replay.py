"""Recorded controller inputs with explicit, incomplete-log state restoration.

CONTROL does not contain optical-flow transforms, raw observed road pixels,
or every preceding controller state. These helpers replay recorded geometry
and twist, with a logged baseline body-envelope oracle. Candidate envelopes
are supplied by each regression so they cannot silently claim camera replay.
"""
import copy
import json
import math
from unittest.mock import patch

import numpy as np

from tests.support import ROOT, APP_ROOT
from robot.config import RuntimeSettings
from robot.control.controller import CommandRamp, compensate_pwm
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.steering_control import SteeringControl


FIXTURE = ROOT / "tests/fixtures/bend_fallback_20261113.json"


def recorded_case(name):
    fixture = json.loads(FIXTURE.read_text())
    return next(case for case in fixture["cases"] if case["name"] == name)


def unboost(wire, boost=37.):
    peak = max(abs(value) for value in wire)
    return tuple(value * max(0., peak - boost) / peak for value in wire) if peak else (0., 0.)


def restore(name):
    """Restore a single recorded step, not hidden history or measured feedback."""
    case = recorded_case(name)
    settings = RuntimeSettings.load(APP_ROOT, {})
    frame = settings.runtime["control_frame"]
    task = settings.crosswalk_policy
    profile = ProfileSpeedControl.from_settings(settings)
    steering = SteeringControl(settings.road_policy, frame["origin_m"], frame["body_m"],
        settings.calibration["road_width_m"], settings.turn_lever,
        settings.steering_gain, settings.cruise_pwm, task.brake_deceleration_m_s2,
        task.speed_per_pwm_m_s, calibration=settings.calibration,
        response_s=task.speed_response_s, straight_cruise=settings.straight_cruise_pwm,
        confirmed_cruise=settings.confirmed_cruise_pwm, speed_control=profile)

    recorded = case["steering"]
    captured = case["captured"]
    dt, speed, yaw = (recorded[key] for key in (
        "motion_interval_s", "body_speed_m_s", "body_yaw_rad_s"))
    angle = -yaw * dt
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    # Pick zero unlogged lateral displacement. This algebra preserves exactly
    # the logged body-forward speed and yaw consumed by steering.observe().
    translation = (-np.array([speed * dt, 0.]) @ rotation.T
                   - steering.origin @ rotation.T + steering.origin)
    gap = recorded.get("motion_gap_replan") or {}
    motion = dict(valid=recorded["motion_measured"], interval_s=dt,
                  rotation=rotation, translation=translation,
                  sigma_m=recorded["motion_sigma_m"],
                  source=recorded.get("motion_source"),
                  prediction_valid=bool(gap.get("active")),
                  prediction_age_s=gap.get("source_age_s"))
    steering.observe(motion, captured, allow_feedback=False)
    steering.trim = recorded.get("trim_m_inv", 0.)
    stable = (recorded.get("speed_plan") or {}).get("road_confirmed_s", 1.)
    steering.speed_ready_since = captured - max(stable, steering.response_s)
    steering.speed_observed_at = captured

    previous = case["previous_trace"]
    old_model = previous["model"]
    previous_wheels = unboost(previous["pwm"], settings.pwm_boost)
    profile.applied(previous["published"][0], previous_wheels,
                    delivered_wheels=previous["pwm"])
    status = recorded.get("speed_profile_control") or {}
    old_reference = (status.get("dwpp") or {}).get("current_linear_m_s",
                                                          old_model["reference_v"])
    old_curvature = old_model.get("predicted_k")
    if old_curvature is None:
        old_curvature = ((previous["pwm"][1] - previous["pwm"][0])
                         / (settings.turn_lever * sum(previous["pwm"])))
    old_angular = (status.get("dwpp") or {}).get("current_angular_rad_s",
                                                          old_reference * old_curvature)
    profile.reference = old_reference
    profile.dwpp_state = (profile.applied_at, old_reference, old_angular)
    # Rounded trace baseline is only a controlled seed when CONTROL did not
    # expose actual regulator mean. It is not asserted equal to real history.
    profile.applied_mean = status.get("applied_mean_pwm",
                              sum(unboost(old_model["base"], settings.pwm_boost)) / 2.)
    if gap.get("active"):
        modeled_previous = profile.bend_response.predict(previous["pwm"])
        profile._bend_gap_source = dict(
            source_at=captured - gap["source_age_s"],
            wheels=previous_wheels, reference=old_reference,
            speed=status["planning_speed_m_s"], yaw=previous["motion"][3],
            speed_cap=min(old_reference, modeled_previous[0]),
        )
    ramp = CommandRamp(settings.motor_rise_pwm_s)
    ramp.last, ramp.previous = profile.applied_at, previous_wheels
    return case, steering, profile, ramp


def command_step(name, *, enabled=True, envelopes=None, target_changes=None):
    """Run actual steering/profile/allocation/ramp with a stated body oracle."""
    case, steering, profile, ramp = restore(name)
    target = copy.deepcopy(case["target"])
    target.update(target_changes or {})
    recorded = case["steering"]
    defaults = [recorded["envelope"]]
    if "dwpp_envelope" in recorded:
        defaults.append(recorded["dwpp_envelope"])
    checked = []

    def envelope(path, curvature, *args, **kwargs):
        checked.append(float(curvature))
        if callable(envelopes):
            return dict(envelopes(len(checked) - 1, float(curvature)))
        choices = defaults if envelopes is None else envelopes
        return dict(choices[min(len(checked) - 1, len(choices) - 1)])

    age = recorded["latency_prediction"]["age_s"]
    with patch.object(steering, "envelope", side_effect=envelope):
        output = steering.command(target, age_s=age,
            lookahead_m=recorded["lookahead_m"], bend_inner_reduction=enabled)
    if output is None:
        applied, wire = None, None
    else:
        applied = ramp.apply(output, case["captured"] + age)
        wire = compensate_pwm(applied)
    return dict(case=case, steering=steering, profile=profile, ramp=ramp,
                output=output, applied=applied, wire=wire, checked_curvatures=checked)
