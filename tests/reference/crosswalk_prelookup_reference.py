"""Frozen detector before grid lookup/mask reuse, 2026-11-01."""

from tests.support import ROOT
from statistics import median
import math
import cv2
import numpy as np


class ScalarCrosswalkDetector:
    def __init__(self, pixels, policy):
        self.pixels = pixels
        self.policy = policy
        self.status = dict(reason="NOT_RUN", passes=[])

    def detect(self, details, road, now):
        p = self.policy
        grid = self.pixels
        bird = details.get("bird")
        path = np.asarray(road.get("path", []), float)
        self.status = dict(reason="NO_ROAD_SUPPORT", passes=[])
        if bird is None or len(path) < 2:
            return None
        if road.get("confidence", 1.0) < 0.55:
            self.status["reason"] = "ROAD_CONFIDENCE_LOW"
            return None
        # A road control target may start beyond a close crossing. Extend only
        # the semantic search corridor along the measured near tangent; this
        # does not create lane support or authorize driving into an unseen area.
        direction = path[1] - path[0]
        length = np.linalg.norm(direction)
        if length > 1e-6:
            path = np.vstack((path[0] - self.pixels.width * direction / length, path))
        tangent = np.gradient(path, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-8)
        normals = np.c_[-tangent[:, 1], tangent[:, 0]]
        half = grid.width / 2 - p.road_margin_m
        polygon = np.vstack((path + half * normals, (path - half * normals)[::-1]))
        uv = np.rint(
            np.c_[
                (grid.ys[0] - polygon[:, 1]) / grid.step, (polygon[:, 0] - grid.xs[0]) / grid.step
            ]
        ).astype(np.int32)
        corridor = np.zeros(bird.shape[:2], np.uint8)
        cv2.fillPoly(corridor, [uv], 255)
        if "saturation" in details:
            saturation, value = details["saturation"], details["value"]
        else:
            hsv = cv2.cvtColor(bird, cv2.COLOR_BGR2HSV)
            saturation, value = hsv[:, :, 1], hsv[:, :, 2]
        # Preserve the calibrated raw-light fast path. Shadows/exposure changes
        # can put every white bar below an absolute threshold; reuse the local
        # contrast image already computed for lane extraction only if it fails.
        for source, values in (("raw", value), ("contrast", details.get("contrast_value"))):
            if values is None:
                continue
            best = self._detect_value(values, saturation, corridor, path, tangent, now, source)
            if best is not None:
                self.status.update(
                    reason="DETECTED",
                    source=source,
                    bar_count=best["bar_count"],
                    near_m=best["near_m"],
                    confidence=best["confidence"],
                )
                return best
        passes = self.status["passes"]
        self.status["reason"] = (
            "NO_REPEATED_PATTERN" if any(x["metric_bars"] for x in passes) else "NO_METRIC_BARS"
        )
        return None

    def _detect_value(self, value, saturation, corridor, path, tangent, now, source):
        p = self.policy
        grid = self.pixels
        best = None
        for level in grid.white_value_levels:
            mask = ((saturation <= 105) & (value >= level) & grid.visible & (corridor != 0)).astype(
                np.uint8
            ) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            stage = dict(
                source=source,
                level=int(level),
                contours=len(contours),
                metric_bars=0,
                largest_group=0,
            )
            self.status["passes"].append(stage)
            boxes = []
            for contour in contours:
                if len(contour) < 4 or cv2.contourArea(contour) < 3:
                    continue
                rc = contour[:, 0, :]
                points = np.c_[grid.xs[rc[:, 1]], grid.ys[rc[:, 0]]].astype(np.float32)
                boxes.append(cv2.boxPoints(cv2.minAreaRect(points)))
            bars = []
            if boxes:
                # Same rectangles, thresholds, tangents and selected bars as the
                # scalar loop, batched to avoid tiny NumPy dispatches on ARM.
                boxes = np.asarray(boxes, float)
                sides = np.roll(boxes, -1, axis=1) - boxes
                lengths = np.linalg.norm(sides, axis=2)
                major = np.argmax(lengths, axis=1)
                rows = np.arange(len(boxes))
                long = lengths[rows, major]
                short = lengths.min(axis=1)
                valid = (
                    (long >= p.bar_length_min_m)
                    & (long <= p.bar_length_max_m)
                    & (short >= p.bar_width_min_m)
                    & (short <= p.bar_width_max_m)
                    & (long >= 2.3 * short)
                )
                centers0 = boxes.mean(axis=1)
                directions0 = sides[rows, major] / np.maximum(long, 1e-12)[:, None]
                nearest = np.argmin(
                    np.sum((centers0[:, None, :] - path[None, :, :]) ** 2, axis=2), axis=1
                )
                alignment = np.sum(directions0 * tangent[nearest], axis=1)
                directions0[alignment < 0] *= -1
                valid &= abs(alignment) >= 0.85
                for i in np.flatnonzero(valid):
                    bars.append(
                        dict(
                            center=centers0[i],
                            direction=directions0[i],
                            length=float(long[i]),
                            width=float(short[i]),
                            box=boxes[i],
                        )
                    )
            stage["metric_bars"] = len(bars)
            centers = np.asarray([b["center"] for b in bars])
            directions = np.asarray([b["direction"] for b in bars])
            evaluated = set()
            for anchor in bars:
                direction = anchor["direction"]
                normal = np.array([-direction[1], direction[0]])
                ids = np.flatnonzero(
                    (abs((centers - anchor["center"]) @ direction) <= p.row_alignment_m)
                    & (directions @ direction >= 0.94)
                )
                key = tuple(ids.tolist())
                if key in evaluated:
                    continue
                evaluated.add(key)
                group = [bars[int(i)] for i in ids]
                if len(group) < p.minimum_bars:
                    continue
                group.sort(key=lambda b: np.dot(b["center"], normal))
                lateral = np.array([np.dot(b["center"], normal) for b in group])
                spacing = np.diff(lateral)
                pitch = np.rint(spacing / p.bar_pitch_m)
                good = (
                    (pitch >= 1)
                    & (pitch <= 2)
                    & (abs(spacing - pitch * p.bar_pitch_m) <= p.pitch_tolerance_m)
                )
                # Split nonuniform clusters rather than joining unrelated reflections.
                chunks = np.split(np.arange(len(group)), np.flatnonzero(~good) + 1)
                for ids in chunks:
                    stage["largest_group"] = max(stage["largest_group"], len(ids))
                    if len(ids) < p.minimum_bars:
                        continue
                    selected = [group[int(i)] for i in ids]
                    t = np.mean([b["direction"] for b in selected], axis=0)
                    t /= np.linalg.norm(t)
                    if t[0] < 0.35:
                        continue
                    corners = np.vstack([b["box"] for b in selected])
                    near = float(np.min(corners @ t))
                    far = float(np.max(corners @ t))
                    length = float(median([b["length"] for b in selected]))
                    width = float(median([b["width"] for b in selected]))
                    # Require the rule's bar/gap scale as a repeated pattern.
                    quality = min(1.0, len(selected) / 7) * max(0.0, 1 - abs(length - 0.10) / 0.12)
                    candidate = dict(
                        normal=t.tolist(),
                        near_m=near,
                        far_m=far,
                        observed_at=now,
                        sigma_m=p.distance_error_m + grid.step,
                        bar_count=len(selected),
                        confidence=float(quality),
                        bar_length_m=length,
                        bar_width_m=width,
                        corners=corners.tolist(),
                    )
                    if best is None or candidate["confidence"] > best["confidence"]:
                        best = candidate
            # A complete seven-bar observation needs no second photometric pass.
            if best and best["bar_count"] == 7 and best["confidence"] >= 0.85:
                return best
        return best
