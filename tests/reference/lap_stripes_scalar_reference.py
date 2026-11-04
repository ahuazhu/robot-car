"""Frozen pre-vectorization stripe detector, 2026-11-01; test oracle only."""

from tests.support import ROOT
import math
import cv2
import numpy as np


def stripes(self, details, road):
    grid = self.pixels
    path = np.asarray(road.get("path", []), float)
    if path.ndim != 2 or path.shape[1:] != (2,) or len(path) < 2 or "bird" not in details:
        return []
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-8)
    path = np.vstack((path[0] - grid.width * tangent[0], path, path[-1] + grid.width * tangent[-1]))
    tangent = np.vstack((tangent[0], tangent, tangent[-1]))
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    half = grid.width * (0.5 - 1 / 12)
    polygon = np.vstack((path + half * normal, (path - half * normal)[::-1]))
    uv = np.rint(
        np.c_[(grid.ys[0] - polygon[:, 1]) / grid.step, (polygon[:, 0] - grid.xs[0]) / grid.step]
    ).astype(np.int32)
    corridor = np.zeros(details["bird"].shape[:2], np.uint8)
    cv2.fillPoly(corridor, [uv], 255)
    candidates = []
    evaluated = []
    for source, level in (("value", min(grid.white_value_levels)), ("contrast_value", 210)):
        mask = (
            (details["saturation"] <= 105)
            & (details[source] >= level)
            & grid.visible
            & (corridor != 0)
        ).astype(np.uint8) * 255
        lines = cv2.HoughLinesP(
            mask, 1, np.pi / 180, 12, minLineLength=int(0.4 * grid.width / grid.step), maxLineGap=2
        )
        segments = (
            []
            if lines is None
            else sorted(
                lines.reshape(-1, 4),
                key=lambda a: -(float(a[2] - a[0]) ** 2 + float(a[3] - a[1]) ** 2),
            )
        )
        for c1, r1, c2, r2 in segments:
            pts = np.array([[grid.xs[r1], grid.ys[c1]], [grid.xs[r2], grid.ys[c2]]])
            delta = pts[1] - pts[0]
            length = np.linalg.norm(delta)
            t = delta / length
            mid = pts.mean(axis=0)
            k = np.argmin(np.linalg.norm(path - mid, axis=1))
            forward = tangent[k]
            if abs(t @ forward) > 0.35:
                continue
            center = mid + ((path[k] - mid) @ t) * t
            n = np.array([-t[1], t[0]])
            if n @ forward < 0:
                n = -n
            if any(
                abs(float((center - e[0]) @ n)) < 2 * grid.step and abs(float(n @ e[1])) > 0.98
                for e in evaluated
            ):
                continue
            if len(evaluated) >= 32:
                break
            evaluated.append((center, n))
            # profile samples a whole transverse strip, including gray before/after
            offs = np.arange(-grid.width / 4, grid.width / 4 + grid.step / 2, grid.step)
            lat = np.linspace(-half, half, max(17, int(2 * half / grid.step) + 1))
            points = center + offs[:, None, None] * n + lat[None, :, None] * t
            vals, valid = grid.sample(details["value"], points.reshape(-1, 2))
            sat, _ = grid.sample(details["saturation"], points.reshape(-1, 2))
            vals = vals.reshape(len(offs), -1)
            valid = valid.reshape(vals.shape)
            sat = sat.reshape(vals.shape)
            profile = np.median(vals, axis=1)
            baseline = cv2.morphologyEx(
                profile.astype(np.float32).reshape(-1, 1),
                cv2.MORPH_OPEN,
                np.ones((2 * int(np.ceil(grid.width / 5 / (2 * grid.step))) + 3, 1), np.uint8),
            ).ravel()
            peak = profile - baseline
            peak[valid.mean(axis=1) < 0.85] = 0
            centerid = int(np.argmax(peak))
            p = float(peak[centerid])
            b = float(baseline[centerid])
            threshold = max(8.0, 0.08 * b)
            if p < threshold:
                continue
            bright = peak >= max(threshold, 0.45 * p)
            lo = hi = centerid
            while lo > 0 and bright[lo - 1]:
                lo -= 1
            while hi + 1 < len(offs) and bright[hi + 1]:
                hi += 1
            thickness = (hi - lo + 1) * grid.step
            if not grid.step / 2 <= thickness <= grid.width / 5:
                continue
            before = lo - 2
            after = hi + 2
            if before < 0 or after >= len(offs):
                continue
            shoulders = np.maximum(vals[before], vals[after])
            core = vals[lo : hi + 1].max(axis=0)
            core_sat = sat[lo : hi + 1].min(axis=0)
            good = valid[before] & valid[after] & valid[lo : hi + 1].all(axis=0)
            support = (
                good & (core_sat <= 105) & (core > shoulders + np.maximum(8.0, 0.08 * shoulders))
            ).mean()
            if support < 0.75:
                continue
            c = center + offs[centerid] * n
            if any(
                np.linalg.norm(c - np.array(a["center"])) < grid.width / 7.5 for a in candidates
            ):
                continue
            candidates.append(
                dict(
                    center=c.tolist(),
                    normal=n.tolist(),
                    distance_m=float(c @ n + thickness / 2),
                    thickness_m=thickness,
                    sigma_m=float(math.hypot(grid.step, half * (1 - support))),
                    qualified=True,
                    support=round(float(support), 3),
                    contrast=round(p, 2),
                    source=source,
                )
            )
    return candidates
