"""Crosswalk scene and physical-motion builders."""

import numpy as np
from robot.tasks.crosswalk import CrosswalkPolicy


def motion(speed=0.0, distance=0.0, valid=True):
    return dict(
        valid=valid,
        rotation=np.eye(2),
        translation=np.array([-distance, 0.0]),
        sigma_m=0.0002,
        travel_m=abs(distance),
        speed_m_s=speed,
        yaw_rad_s=0.0,
    )


def observation(now, near=0.27, far=None):
    return dict(
        normal=[1.0, 0.0],
        near_m=near,
        far_m=near + 0.1 if far is None else far,
        sigma_m=0.01,
        observed_at=now,
        bar_count=7,
        confidence=0.9,
        corners=[],
    )


def configured(**kwargs):
    d = dict(
        mode="yield",
        body_front_m=0.12,
        body_rear_m=0.13,
        body_half_width_m=0.10,
        geometry_verified=True,
        brake_lamp_verified=True,
        brake_deceleration_m_s2=0.3,
    )
    d.update(kwargs)
    return CrosswalkPolicy.from_config(d)
