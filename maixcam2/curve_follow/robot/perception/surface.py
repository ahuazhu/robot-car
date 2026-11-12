"""Current-frame road topology evidence for the fixed green racing field.

White paint, gray asphalt and unobserved pixels cannot refute a remembered
road. Only an extended region of unambiguous green, outside projection-error
margins, can revoke an old geometric hypothesis. Connectivity can disambiguate existing measured boundaries on separate road
pieces. This evidence never creates or refreshes a road observation.
"""

import cv2
import numpy as np

from robot.core.geometry import contiguous_runs


class RoadSurfaceEvidence:
    def __init__(self, pixels, green, saturation, *, road_value=None, reference_origin=(0.0, 0.0)):
        self._sample = pixels.sample
        self.step_m = pixels.step
        self.minimum_segment_m = pixels.policy.minimum_segment_m
        # Existing gray/white tests extend through S=110/105 respectively.
        # Exclude that overlap instead of treating pale paint as green ground.
        self._definite_green = green & (saturation > 110) & pixels.visible
        self._green_clearance = None
        self._pixels = pixels
        self._green = green
        self._saturation = saturation
        self._road_value = road_value
        self._origin = np.asarray(reference_origin, float)
        self._components = self._component_near = None

    def nearer_separated_component(self, hypotheses, sigma_m):
        """Disambiguate measured boundaries on separate visible road pieces.

        White transverse paint remains part of a road component. Select only
        when their observed interiors belong to different components, one is
        uniquely nearer the chassis, and definite green separates the pieces.
        This returns evidence about existing boundaries, never a driving path.
        """
        if self._road_value is None or len(hypotheses) != 2:
            return None
        p = self._pixels
        if self._components is None:
            road = (self._saturation < 110) & (self._road_value >= 35) & ~self._green & p.visible
            count, self._components = cv2.connectedComponents(road.astype(np.uint8), connectivity=8)
            self._component_near = np.full(count, np.inf)
            distance = np.hypot(p.xs[:, None] - self._origin[0], p.ys[None, :] - self._origin[1])
            np.minimum.at(self._component_near, self._components[road], distance[road])
        ids, probes = [], []
        for path, inward in hypotheses:
            points = np.asarray(path) + 0.025 * np.asarray(inward)
            labels, visible = self._sample(self._components, points)
            labels = labels.astype(np.int32)
            counts = np.bincount(labels[visible], minlength=len(self._component_near))
            counts[0] = 0
            component = int(np.argmax(counts))
            selected = visible & (labels == component)
            if component == 0 or counts[component] < 5 or counts[component] <= visible.sum() / 2:
                return None
            ids.append(component)
            probes.append(points[selected])
        if ids[0] == ids[1]:
            return None
        near = self._component_near[ids]
        if not np.isfinite(near).all() or abs(near[0] - near[1]) <= 2 * sigma_m + 2 * self.step_m:
            return None
        distance2 = ((probes[0][:, None, :] - probes[1][None, :, :]) ** 2).sum(axis=2)
        a, b = np.unravel_index(int(np.argmin(distance2)), distance2.shape)
        start, end = probes[0][a], probes[1][b]
        segment = np.linspace(start, end, max(3, int(np.ceil(np.linalg.norm(end-start) / self.step_m)) + 1))
        separator = self.contradiction(segment, sigma_m)
        if not separator["contradicted"]:
            return None
        return dict(index=int(np.argmin(near)), components=ids, nearest_m=near.tolist(),
                    green_separator_m=separator["longest_green_m"])

    def contradiction(self, path, sigma_m):
        return self.contradictions([(path, sigma_m)])[0]

    def contradictions(self, paths):
        """Sample all current-frame corridors once, retaining each run boundary."""
        reports = [dict(contradicted=False, visible_samples=0, green_samples=0,
                        longest_green_m=0.0) for _ in paths]
        chunks, sigmas, slices = [], [], []
        offset = 0
        for i, (path, sigma_m) in enumerate(paths):
            points = np.asarray(path, float)
            if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2
                    or not np.isfinite(points).all()):
                continue
            chunks.append(points)
            sigmas.append(np.broadcast_to(np.asarray(sigma_m, float), (len(points),)))
            slices.append((i, offset, offset+len(points)))
            offset += len(points)
        if not chunks:
            return reports
        points, sigma = np.concatenate(chunks), np.concatenate(sigmas)
        definite, visible = self._sample(self._definite_green, points)
        for i, begin, end in slices:
            reports[i]['visible_samples'] = int(visible[begin:end].sum())
        valid_sigma = np.isfinite(sigma) & (sigma >= 0)
        if not (visible & valid_sigma & (definite > 0)).any():
            return reports
        if self._green_clearance is None:
            self._green_clearance = (
                cv2.distanceTransform(self._definite_green.astype(np.uint8), cv2.DIST_L2, 5)
                * self.step_m
            )
        clearance, visible = self._sample(self._green_clearance, points)
        # Two grid cells cover remap/discretization near the color boundary;
        # geometry uncertainty adds its own two-sigma margin per road sample.
        green = visible & valid_sigma & (clearance > 2 * sigma + 2 * self.step_m)
        for i, begin, end in slices:
            # Never join a green run across separate stored corridors.
            local, mask = points[begin:end], green[begin:end]
            lengths = [float(np.linalg.norm(np.diff(local[ids], axis=0), axis=1).sum())
                       for ids in contiguous_runs(mask) if len(ids) >= 3]
            longest = max(lengths, default=0.0)
            reports[i].update(contradicted=longest+1e-9 >= self.minimum_segment_m,
                             visible_samples=int(visible[begin:end].sum()),
                             green_samples=int(mask.sum()), longest_green_m=longest)
        return reports
