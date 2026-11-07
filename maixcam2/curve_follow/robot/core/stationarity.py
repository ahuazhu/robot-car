"""Confirm rest without treating alternating subpixel flow as repeated motion."""

import numpy as np


class StationaryEvidence:
    def __init__(self, speed_limit, yaw_limit, window_s, maximum_gap_s):
        self.speed_limit = speed_limit
        self.yaw_limit = yaw_limit
        self.window_s = window_s
        self.maximum_gap_s = maximum_gap_s
        self.reset()

    def reset(self):
        self.samples = []
        self.rotation = np.eye(2)
        self.translation = np.zeros(2)
        self.previous = None

    def observe(self, motion, now, stopped):
        if not stopped or not motion or not motion.get("valid"):
            self.reset()
            return False
        speed, yaw = motion.get("speed_m_s"), motion.get("yaw_rad_s")
        if speed is None or yaw is None or not np.isfinite([speed, yaw, now]).all():
            self.reset()
            return False
        rotation = np.asarray(motion["rotation"], float)
        translation = np.asarray(motion["translation"], float)
        if (rotation.shape != (2, 2) or translation.shape != (2,)
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()):
            self.reset()
            return False
        if self.previous is not None and not 0 < now-self.previous <= self.maximum_gap_s:
            self.reset()
            return False
        self.previous = now
        self.translation = rotation @ self.translation + translation
        self.rotation = rotation @ self.rotation
        angle = float(np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))
        self.samples.append((now, *self.translation, angle))
        # Keep one sample spanning the window boundary; memory stays bounded.
        while len(self.samples) > 2 and now-self.samples[1][0] >= self.window_s:
            self.samples.pop(0)
        instant = abs(speed) <= self.speed_limit and abs(yaw) <= self.yaw_limit
        if instant:
            return True
        # Only small instantaneous violations may be flow jitter. A pronounced
        # movement, missing flow or lost zero ACK immediately breaks the dwell.
        if abs(speed) > 3*self.speed_limit or abs(yaw) > 3*self.yaw_limit:
            return False
        if len(self.samples) < 3 or now-self.samples[0][0] < self.window_s:
            return False
        samples = np.asarray(self.samples)
        times = samples[:, 0] - samples[:, 0].mean()
        positions = samples[:, 1:].copy()
        positions[:, 2] = np.unwrap(positions[:, 2])
        trend = times @ (positions-positions.mean(axis=0)) / (times @ times)
        return bool(np.linalg.norm(trend[:2]) <= self.speed_limit
                    and abs(trend[2]) <= self.yaw_limit)
