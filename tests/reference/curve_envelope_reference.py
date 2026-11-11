from tests.support import ROOT
import math
import numpy as np
from robot.core.geometry import arc, support_projection, pursuit_point


def envelope_reference(
    self,
    path,
    curvature,
    sigma,
    horizon,
    observed_path=None,
    speed_m_s=None,
    sensor_transform=None,
    lookahead_m=None,
):
    if not self.body or len(path) < 5:
        return dict(speed_scale=1.0, coverage=0.0, state="UNCONFIGURED")
    front, rear, half = self.body
    corners = np.array([[front, half], [front, -half], [-rear, half], [-rear, -half]])
    support = path if observed_path is None else observed_path
    if len(support) < 2:
        return dict(speed_scale=1.0, coverage=0.0, state="UNOBSERVED")
    support_arc = arc(support)
    horizon = min(horizon, self.width)
    pose = np.zeros(3)
    first_unsafe = None
    checked = 0
    total = 0
    minimum = float("inf")
    # Receding-horizon geometry: after the first requested turn, recompute
    # pursuit along this same curve. No future observation is invented.
    for i, distance in enumerate(np.linspace(0.0, horizon, 7)):
        if i:
            ds = horizon / 6.0
            theta = curvature * ds
            advance = (
                np.array([ds, 0.0])
                if abs(curvature) < 1e-8
                else np.array([math.sin(theta) / curvature, (1 - math.cos(theta)) / curvature])
            )
            c, s = math.cos(pose[2]), math.sin(pose[2])
            pose[:2] += advance @ np.array([[c, s], [-s, c]])
            pose[2] += theta
        c, s = math.cos(pose[2]), math.sin(pose[2])
        rotation = np.array([[c, -s], [s, c]])
        footprint = corners @ rotation.T + pose[:2]
        distances, positions = support_projection(footprint, support)
        known = (positions > 1e-5) & (positions < support_arc[-1] - 1e-5)
        if self.inverse is not None:
            # Arc projection alone is not sensor coverage: an unseen body
            # corner may project onto a distant interior bend segment.
            camera_footprint = footprint
            if sensor_transform is not None:
                rotation_at_command, translation_at_command = sensor_transform
                camera_footprint = (footprint - translation_at_command) @ rotation_at_command
            ground = camera_footprint + self.origin
            q = np.c_[ground, np.ones(len(ground))] @ self.inverse.T
            denominator = q[:, 2]
            finite = abs(denominator) > 1e-8
            uv = q[:, :2] / np.where(finite, denominator, 1.0)[:, None] * [320.0, 240.0]
            known &= (
                finite
                & np.isfinite(uv).all(axis=1)
                & (uv[:, 0] >= 2)
                & (uv[:, 0] < 318)
                & (uv[:, 1] >= self.roi_top)
                & (uv[:, 1] < 238)
            )
        checked += int(known.sum())
        total += 4
        if known.any():
            clearance = self.width / 2 - distances[known] - sigma
            minimum = min(minimum, float(clearance.min()))
            if float(clearance.min()) < 0 and first_unsafe is None:
                first_unsafe = float(distance)
        relative_path = (path - pose[:2]) @ rotation
        selected = pursuit_point(
            relative_path,
            lookahead_m or self.minimum_lookahead_m,
            self.policy.forward_min_m,
            self.policy.maximum_target_m,
        )
        if selected is not None:
            target = selected["point"]
            curvature = 2 * target[1] / max(float(target @ target), 1e-8)

    coverage = checked / max(1, total)
    # The camera cannot see under the chassis. Missing coverage is reported,
    # not fabricated from an extrapolated prefix or treated as a collision.
    scale = 1.0
    safe_speed = None
    if first_unsafe is not None:
        a = self.deceleration
        t = self.policy.missing_command_s
        safe_speed = max(0.0, math.sqrt((a * t) ** 2 + 2 * a * first_unsafe) - a * t)
        estimated = max(1e-8, self.speed_per_pwm * self.cruise if speed_m_s is None else speed_m_s)
        scale = min(scale, safe_speed / estimated)
    return dict(
        speed_scale=scale,
        coverage=coverage,
        state="FULL" if checked == total else "PARTIAL",
        minimum_clearance_m=None if not math.isfinite(minimum) else minimum,
        first_unsafe_m=first_unsafe,
        speed_limit_m_s=safe_speed,
    )
