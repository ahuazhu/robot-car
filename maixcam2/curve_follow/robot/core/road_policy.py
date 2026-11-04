"""Validated geometry/observation limits, shared by every road source.

Distances are metres, times are monotonic seconds. These are conservative
engineering bounds, not a claim that chassis dynamics have been measured.
Calibration and actuator configuration remain separate from image thresholds.
"""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RoadPolicy:
    measurement_quality: float = 0.55
    confirmation_quality: float = 0.65
    single_edge_quality_scale: float = 0.85
    association_distance_m: float = 0.055
    association_heading_cos: float = 0.75
    minimum_overlap_m: float = 0.06
    minimum_segment_m: float = 0.075
    sample_step_m: float = 0.015
    observation_sigma_m: float = 0.005
    maximum_sigma_m: float = 0.05
    prediction_noise_m_s: float = 0.04
    maximum_prediction_s: float = 0.60
    maximum_blind_distance_m: float = 0.05
    confidence_decay_s: float = 0.45
    confirmation_evidence: float = 1.5
    stopped_confirmation_frames: int = 4
    forward_min_m: float = 0.06
    curve_analysis_window_m: float = 0.30
    observation_search_near_m: float = 0.40
    maximum_target_m: float = 0.65
    motion_min_features: int = 8
    motion_max_features: int = 100
    motion_fb_error_px: float = 1.0
    motion_inlier_m: float = 0.012
    motion_max_residual_m: float = 0.008
    motion_min_span_m: float = 0.06
    motion_min_inlier_fraction: float = 0.60
    motion_max_speed_m_s: float = 0.60
    motion_max_yaw_rad_s: float = 3.0
    motion_max_interval_s: float = 0.75
    missing_command_s: float = 0.35
    startup_observations: int = 5

    @classmethod
    def from_config(cls, config=None):
        values = (config or {}).get("road_model", {})
        if not isinstance(values, dict):
            raise ValueError("road_model must be an object")
        names = {f.name for f in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError("Unknown road_model options: " + ",".join(sorted(unknown)))
        policy = cls(**values)
        for f in fields(cls):
            value = getattr(policy, f.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Invalid road_model option: " + f.name)
        for name in (
            "motion_min_features",
            "motion_max_features",
            "stopped_confirmation_frames",
            "startup_observations",
        ):
            if not isinstance(getattr(policy, name), int):
                raise ValueError(name + " must be an integer")
        if (
            policy.observation_search_near_m >= policy.maximum_target_m
            or policy.measurement_quality > policy.confirmation_quality
            or policy.confirmation_quality > 1
            or policy.single_edge_quality_scale > 1
            or policy.association_heading_cos > 1
            or policy.motion_min_inlier_fraction > 1
            or policy.motion_min_features < 3
            or policy.motion_min_features > policy.motion_max_features
        ):
            raise ValueError("Inconsistent road_model limits")
        return policy
