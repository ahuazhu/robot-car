"""One-shot brightness transition trigger inside a configurable image rectangle.

ARM records the current scene. A configured polarity of a persistent compact brightness change
can request a start; the common road/motor gates still decide whether to move.
The normalized search rectangle is shared by detection and preview drawing.
"""

import math
import cv2
import numpy as np


class StartLight:
    def __init__(self, config=None):
        c = config or {}
        self.enabled = c.get("enabled", False)
        auto = c.get("auto_arm_on_launch", False)
        if not isinstance(self.enabled, bool) or not isinstance(auto, bool):
            raise ValueError("Lamp enable/auto-arm must be boolean")
        self.size = tuple(c.get("image_size", [320, 240]))
        if self.size != (320, 240):
            raise ValueError("Start signal image size mismatch")
        rectangle = c.get("search_roi_fraction", [0.4, 0.0, 0.2, 2 / 15])
        if not isinstance(rectangle, (list, tuple)) or len(rectangle) != 4:
            raise ValueError("Signal rectangle must be normalized x,y,width,height")
        self.rectangle = tuple(float(v) for v in rectangle)
        if not all(math.isfinite(v) for v in self.rectangle):
            raise ValueError("Nonfinite signal rectangle")
        x, y, w, h = self.rectangle
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1 or y + h > 1:
            raise ValueError("Signal rectangle outside image")
        x0, y0 = round(x * self.size[0]), round(y * self.size[1])
        x1, y1 = round((x + w) * self.size[0]), round((y + h) * self.size[1])
        self.search_roi = (x0, y0, x1 - x0, y1 - y0)
        self.trigger_direction = c.get("trigger_direction", "BOTH")
        if self.trigger_direction not in ("BOTH", "BRIGHTER", "DARKER"):
            raise ValueError("Invalid start signal direction")
        self.delta = float(c.get("delta_luma", 35.0))
        self.min_pixels = int(c.get("minimum_changed_pixels", 512))
        self.max_region_fraction = float(c.get("maximum_region_fraction", 1.0))
        self.min_fill = float(c.get("minimum_region_fill", 0.25))
        self.max_aspect = float(c.get("maximum_region_aspect", 4.0))
        self.confirm = float(c.get("confirm_s", 0.4))
        self.frames = c.get("confirm_frames", 3)
        self.gap = float(c.get("maximum_frame_gap_s", 0.75))
        self.min_overlap = float(c.get("minimum_confirmation_iou", 0.6))
        if not all(
            math.isfinite(x)
            for x in (
                self.delta,
                self.confirm,
                self.gap,
                self.max_region_fraction,
                self.min_fill,
                self.max_aspect,
                self.min_overlap,
            )
        ):
            raise ValueError("Nonfinite start signal configuration")
        if not 0 < self.delta < 255 or not 0 <= self.confirm <= 2 or not 0 < self.gap <= 2:
            raise ValueError("Invalid start signal limits")
        if (
            not 0 < self.max_region_fraction <= 1
            or not 0 < self.min_fill <= 1
            or self.max_aspect < 1
            or not 0 < self.min_overlap <= 1
        ):
            raise ValueError("Invalid compact signal region limits")
        if (
            isinstance(self.frames, bool)
            or not isinstance(self.frames, int)
            or self.frames < 2
            or self.min_pixels < 4
        ):
            raise ValueError("Start signal needs at least two observations and four pixels")
        if (
            min(self.search_roi[2:]) < 4
            or any(v % 2 for v in self.search_roi[2:])
            or self.min_pixels > self.search_roi[2] * self.search_roi[3] * self.max_region_fraction
        ):
            raise ValueError("Invalid start signal region")
        self.armed = False
        self.fired = False
        self.latched = False
        self.phase = "DISARMED"
        self.reference = None
        self.baseline_observations = 0
        self.roi = None
        self.since = self.last = None
        self.hits = 0
        self.elapsed = 0.0
        self.pending = None
        self.observation = dict(state="IDLE", reason="DISARMED")
        if self.enabled and auto:
            self.arm()

    def _prepare(self, bgr):
        if bgr is None or bgr.shape != (self.size[1], self.size[0], 3):
            return None
        # Crop before filtering, so pixels outside the visible rectangle
        # cannot affect detection, including along its border.
        x, y, w, h = self.search_roi
        small = cv2.resize(
            bgr[y : y + h, x : x + w], (w // 2, h // 2), interpolation=cv2.INTER_AREA
        )
        small = cv2.GaussianBlur(small, (3, 3), 0)
        # Preserve the existing 8-bit Lab lightness scale for delta_luma.
        return cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0]

    def arm(self, bgr=None):
        if not self.enabled:
            return
        self.armed = True
        self.fired = self.latched = False
        self.roi = None
        self.since = self.last = None
        self.hits = 0
        self.elapsed = 0.0
        self.pending = None
        prepared = self._prepare(bgr)
        self.reference = prepared
        self.baseline_observations = self.frames if prepared is not None else 0
        self.phase = "WAIT_CHANGE" if self.reference is not None else "BASELINING"
        self.observation = dict(
            state="STEADY" if self.reference is not None else "BASELINE",
            reason="WAITING_FOR_CHANGE" if self.reference is not None else "WAITING_FOR_FRAME",
        )

    def cancel(self):
        self.armed = False
        self.latched = False
        self.phase = "DISARMED"
        self.roi = None
        self.reference = None
        self.baseline_observations = 0
        self.pending = None
        self.since = self.last = None
        self.hits = 0
        self.elapsed = 0.0
        self.observation = dict(state="IDLE", reason="DISARMED")

    def started(self):
        self.armed = False
        self.fired = True
        self.phase = "RUNNING"
        self.reference = None

    def detect(self, bgr):
        if not self.enabled:
            return dict(state="DISABLED")
        if not self.armed or self.latched:
            return self.observation
        prepared = self._prepare(bgr)
        if prepared is None:
            return dict(state="UNKNOWN", reason="IMAGE_SIZE")
        gray = prepared
        if self.reference is None or self.baseline_observations < self.frames:
            self.reference = gray
            self.baseline_observations += 1
            self.phase = (
                "WAIT_CHANGE" if self.baseline_observations >= self.frames else "BASELINING"
            )
            return dict(state="BASELINE", reason="BASELINE_RECORDED")
        # Brightness only, restricted to the configured polarity. Compact
        # connected regions and temporal confirmation still reject small,
        # scattered or brief fluctuations. No chroma condition remains.
        difference = cv2.subtract(gray, self.reference, dtype=cv2.CV_16S)
        candidates = []
        for direction, polarity in (("BRIGHTER", 1), ("DARKER", -1)):
            if self.trigger_direction != "BOTH" and direction != self.trigger_direction:
                continue
            light_strength = difference * polarity
            changed_mask = (light_strength >= self.delta).astype(np.uint8) * 255
            joined = cv2.morphologyEx(changed_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            _, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
            for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
                selected = (labels[y : y + h, x : x + w] == label) & (
                    changed_mask[y : y + h, x : x + w] != 0
                )
                count = int(np.count_nonzero(selected))
                # Count actual thresholded pixels; morphology must not invent
                # evidence. Do not pool disconnected changes across the image.
                fill = count / float(w * h)
                fraction = w * h / float(gray.size)
                aspect = max(w / h, h / w)
                if (
                    count * 4 < self.min_pixels
                    or fraction > self.max_region_fraction
                    or fill < self.min_fill
                    or aspect > self.max_aspect
                ):
                    continue
                strength = float(np.mean(light_strength[y : y + h, x : x + w][selected]))
                candidates.append(
                    dict(
                        state="CHANGE",
                        reason="COMPACT_BRIGHTNESS_CHANGE",
                        direction=direction,
                        kind="BRIGHTNESS",
                        roi=[
                            int(self.search_roi[0] + x * 2),
                            int(self.search_roi[1] + y * 2),
                            int(w * 2),
                            int(h * 2),
                        ],
                        magnitude=round(strength, 1),
                        changed_pixels=count * 4,
                        region_fraction=round(fraction, 4),
                        region_fill=round(fill, 3),
                        energy=strength * count,
                    )
                )
        if not candidates:
            self.roi = None
            return dict(state="STEADY", reason="NO_SIGNIFICANT_CHANGE")
        # Prefer a region already being confirmed; otherwise use the strongest
        # changed region. Multiple LED cells/lights need not be ambiguous.
        continuing = [c for c in candidates if self._same_change(self.pending, c)]
        best = max(continuing or candidates, key=lambda c: c["energy"])
        best.pop("energy")
        self.roi = best["roi"]
        return best

    def _same_change(self, a, b):
        if not a or a.get("direction") != b.get("direction"):
            return False
        ax, ay, aw, ah = a["roi"]
        bx, by, bw, bh = b["roi"]
        intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
            0, min(ay + ah, by + bh) - max(ay, by)
        )
        union = aw * ah + bw * bh - intersection
        return union > 0 and intersection / union >= self.min_overlap

    def update(self, observation, now, *, active=False, timely=True):
        self.observation = observation
        if not self.enabled:
            return False
        if active:
            self.started()
            return False
        if not self.armed:
            return False
        if self.latched:
            self.phase = "SIGNAL_READY"
            return math.isfinite(now)
        valid_time = math.isfinite(now) and (self.last is None or 0 < now - self.last <= self.gap)
        direction = observation.get("direction")
        allowed_direction = direction in ("BRIGHTER", "DARKER") and (
            self.trigger_direction == "BOTH" or direction == self.trigger_direction
        )
        valid = valid_time and timely and observation.get("state") == "CHANGE" and allowed_direction
        if not valid:
            self.since = None
            self.hits = 0
            self.pending = None
            self.elapsed = 0.0
        else:
            if self.since is None or not self._same_change(self.pending, observation):
                self.since = now
                self.hits = 1
                # Anchor the whole confirmation window to its first region.
                # Comparing only adjacent frames lets a moving reflection
                # accumulate evidence while sweeping across the rectangle.
                self.pending = dict(observation, roi=list(observation["roi"]))
            else:
                self.hits += 1
            self.elapsed = max(0.0, now - self.since)
        self.last = now if math.isfinite(now) else None
        self.latched = valid and self.hits >= self.frames and self.elapsed >= self.confirm
        self.phase = (
            "SIGNAL_READY"
            if self.latched
            else "CONFIRM_CHANGE"
            if self.hits
            else "WAIT_FRESH_FRAME"
            if observation.get("state") == "CHANGE" and not timely
            else "BASELINING"
            if self.reference is None or self.baseline_observations < self.frames
            else "WAIT_CHANGE"
        )
        return self.latched

    def draw_search_region(self, preview):
        """Draw the exact detection rectangle on an already separate UI image."""
        if not self.enabled:
            return
        x, y, w, h = self.search_roi
        cv2.rectangle(preview, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 1)

    def snapshot(self):
        return dict(
            enabled=self.enabled,
            armed=self.armed,
            phase=self.phase,
            fired=self.fired,
            signal_confirmed=self.latched,
            lamp=self.observation,
            confirmed_s=round(self.elapsed, 3),
            required_s=self.confirm,
            confirmation_frames=self.hits,
            required_frames=self.frames,
            search_scope="RECTANGLE",
            search_roi=list(self.search_roi),
            search_roi_fraction=list(self.rectangle),
            trigger_direction=self.trigger_direction,
            delta_luma_min=self.delta,
            minimum_changed_pixels=self.min_pixels,
            change_rule="BRIGHTNESS_ONLY",
            maximum_region_fraction=self.max_region_fraction,
            minimum_confirmation_iou=self.min_overlap,
            confirmation_anchor="FIRST_REGION",
            minimum_region_fill=self.min_fill,
            maximum_region_aspect=self.max_aspect,
            baseline_ready=self.reference is not None and self.baseline_observations >= self.frames,
        )
