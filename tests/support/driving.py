"""Shared road observations and steering setup."""

import numpy as np
from robot.navigation.steering_control import SteeringControl
from robot.core.road_policy import RoadPolicy


def road(curvature=0.0, start=-0.2, end=0.8, **changes):
    s = np.linspace(start, end, 100)
    p = (
        np.c_[s, np.zeros(len(s))]
        if not curvature
        else np.c_[np.sin(s * curvature) / curvature, (1 - np.cos(s * curvature)) / curvature]
    )
    result = dict(
        valid=True,
        state="OBSERVED",
        observed=True,
        confidence=1.0,
        control_target_m=[0.18, 0.0],
        control_path_m=p,
        path=p,
        path_observed_mask=[True] * len(p),
        path_curvature_m_inv=curvature,
    )
    result.update(changes)
    return result


def control(result, fast=72.0, measured=True, body=(0.075, 0.105, 0.085)):
    c = SteeringControl(RoadPolicy(), [0.0, 0.0], body, 0.3, cruise=60.0, straight_cruise=fast)
    for now in (1.0, 1.2, 1.4, 1.6):
        c.observe(
            dict(
                valid=measured,
                interval_s=0.2,
                rotation=np.eye(2),
                translation=np.array([-0.02, 0.0]),
                sigma_m=0.0002,
                speed_m_s=0.1,
                yaw_rad_s=0.0,
            ),
            now,
            False,
        )
        command = c.command(result, age_s=0.1)
    return c, command


def flow(valid=True):
    return dict(
        valid=valid,
        interval_s=0.15,
        rotation=np.eye(2),
        translation=np.array([-0.015, 0.0]),
        sigma_m=0.0002,
    )
