"""Current-frame negative road evidence for the fixed green racing field.

White paint, gray asphalt and unobserved pixels cannot refute a remembered
road. Only an extended region of unambiguous green, outside projection-error
margins, can revoke an old geometric hypothesis. This evidence never creates
or refreshes a road observation.
"""

import cv2
import numpy as np

from robot.core.geometry import contiguous_runs


class RoadSurfaceEvidence:
    def __init__(self, pixels, green, saturation):
        self._sample = pixels.sample
        self.step_m = pixels.step
        self.minimum_segment_m = pixels.policy.minimum_segment_m
        # Existing gray/white tests extend through S=110/105 respectively.
        # Exclude that overlap instead of treating pale paint as green ground.
        self._definite_green = green & (saturation > 110) & pixels.visible
        self._green_clearance = None

    def contradiction(self, path, sigma_m):
        report = dict(contradicted=False, visible_samples=0, green_samples=0, longest_green_m=0.0)
        if path is None:
            return report
        points = np.asarray(path, float)
        if (
            points.ndim != 2
            or points.shape[1:] != (2,)
            or len(points) < 2
            or not np.isfinite(points).all()
        ):
            return report
        sigma = np.broadcast_to(np.asarray(sigma_m, float), (len(points),))
        definite, visible = self._sample(self._definite_green, points)
        report["visible_samples"] = int(visible.sum())
        valid_sigma = np.isfinite(sigma) & (sigma >= 0)
        if not (visible & valid_sigma & (definite > 0)).any():
            return report
        if self._green_clearance is None:
            self._green_clearance = (
                cv2.distanceTransform(self._definite_green.astype(np.uint8), cv2.DIST_L2, 5)
                * self.step_m
            )
        clearance, visible = self._sample(self._green_clearance, points)
        # Two grid cells cover remap/discretization near the color boundary;
        # geometry uncertainty adds its own two-sigma margin per road sample.
        green = visible & valid_sigma & (clearance > 2 * sigma + 2 * self.step_m)
        lengths = [
            float(np.linalg.norm(np.diff(points[ids], axis=0), axis=1).sum())
            for ids in contiguous_runs(green)
            if len(ids) >= 3
        ]
        longest = max(lengths, default=0.0)
        report.update(
            contradicted=longest + 1e-9 >= self.minimum_segment_m,
            visible_samples=int(visible.sum()),
            green_samples=int(green.sum()),
            longest_green_m=longest,
        )
        return report
