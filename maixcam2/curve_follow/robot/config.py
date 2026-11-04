"""Load deployment files and normalize application settings in one place."""

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

from robot.control.controller import MAX_MOTOR_PWM, can_enable_drive
from robot.core.road_policy import RoadPolicy
from robot.tasks.crosswalk import CrosswalkPolicy


@dataclass(frozen=True)
class RuntimeSettings:
    calibration: dict
    runtime: dict
    road_calibration: dict
    road_policy: RoadPolicy
    crosswalk_policy: CrosswalkPolicy
    preview_only: bool
    calibrated: bool
    pwm_boost: int
    vision_budget_ms: float
    turn_lever: float
    steering_gain: float
    cruise_pwm: float
    straight_cruise_pwm: float
    confirmed_cruise_pwm: float
    search_cruise_pwm: float
    duration_s: float

    @classmethod
    def load(cls, app_dir, environ=None):
        env = os.environ if environ is None else environ
        app_dir = Path(app_dir)
        calibration = json.loads((app_dir / "calibration.json").read_text())
        runtime_path = app_dir / "runtime.json"
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        road_calibration = dict(
            calibration,
            road_model=runtime.get("road_model", {}),
            control_frame=runtime.get("control_frame", {}),
        )
        boost = int(np.clip(runtime.get("motor_peak_boost_pwm", 0), 0, 45))
        ceiling = float(MAX_MOTOR_PWM - boost)
        cruise = float(np.clip(runtime.get("cruise_pwm", 60), 1, ceiling))
        straight = float(np.clip(runtime.get("straight_cruise_pwm", cruise), cruise, ceiling))
        return cls(
            calibration,
            runtime,
            road_calibration,
            RoadPolicy.from_config(road_calibration),
            CrosswalkPolicy.from_config(runtime.get("crosswalk", {})),
            not can_enable_drive(calibration, runtime, env.get("TARGET_PREVIEW_ONLY") == "1"),
            bool(calibration.get("geometry_confirmed") and calibration.get("camera_pose_verified")),
            boost,
            float(np.clip(runtime.get("vision_budget_ms", 200), 80, 280)),
            float(np.clip(runtime.get("turn_lever_m", 0.15), 0.08, 0.25)),
            float(np.clip(runtime.get("steering_gain", 1.0), 0.5, 1.5)),
            cruise,
            straight,
            float(np.clip(runtime.get("confirmed_cruise_pwm", cruise), cruise, straight)),
            float(np.clip(runtime.get("search_cruise_pwm", 60), 1, ceiling)),
            float(env.get("TARGET_TEST_SECONDS", "0")),
        )
