"""Observe a transverse white paint band and count measured crossings.

Marker perception is independent of gantry visibility and navigation targets.
A line must cover the lane interior, be locally brighter than gray ground on
both sides, and remain geometrically consistent in successive images. Visits
are transported by measured ground motion; frames and elapsed time are not laps.
"""

import math
import cv2
import numpy as np

from robot.core.race import validate_laps


class LapMarkerDetector:
    def __init__(self, pixels, config):
        self.pixels = pixels
        self.enabled = config.get("enabled", False)
        if not isinstance(self.enabled, bool):
            raise ValueError("race.enabled must be boolean")
        self.status = {}
        self.confirmation_frames = 0
        self.confirmation_since = None
        self.previous = None
        self._profile_cache = None

    def stripes(self, details, road):
        grid = self.pixels
        path = np.asarray(road.get("path", []), float)
        if path.ndim != 2 or path.shape[1:] != (2,) or len(path) < 2 or "bird" not in details:
            return []
        tangent = np.gradient(path, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-8)
        path = np.vstack(
            (path[0] - grid.width * tangent[0], path, path[-1] + grid.width * tangent[-1])
        )
        tangent = np.vstack((tangent[0], tangent, tangent[-1]))
        normal = np.c_[-tangent[:, 1], tangent[:, 0]]
        half = grid.width * (0.5 - 1 / 12)
        polygon = np.vstack((path + half * normal, (path - half * normal)[::-1]))
        uv = np.rint(
            np.c_[
                (grid.ys[0] - polygon[:, 1]) / grid.step, (polygon[:, 0] - grid.xs[0]) / grid.step
            ]
        ).astype(np.int32)
        corridor = np.zeros(details["bird"].shape[:2], np.uint8)
        cv2.fillPoly(corridor, [uv], 255)
        # Sampling geometry depends only on physical lane width/resolution.
        key = (grid.width, grid.step)
        if self._profile_cache is None or self._profile_cache[0] != key:
            offs = np.arange(-grid.width / 4, grid.width / 4 + grid.step / 2, grid.step)
            lat = np.linspace(-half, half, max(17, int(2 * half / grid.step) + 1))
            kernel = np.ones((2 * int(np.ceil(grid.width / 5 / (2 * grid.step))) + 3, 1), np.uint8)
            self._profile_cache = (key, offs, lat, kernel)
        _, offs, lat, kernel = self._profile_cache
        eligible = (details["saturation"] <= 105) & grid.visible & (corridor != 0)
        candidates = []
        evaluated = []
        for source, level in (("value", min(grid.white_value_levels)), ("contrast_value", 210)):
            mask = (eligible & (details[source] >= level)).astype(np.uint8) * 255
            lines = cv2.HoughLinesP(
                mask,
                1,
                np.pi / 180,
                12,
                minLineLength=int(0.4 * grid.width / grid.step),
                maxLineGap=2,
            )
            if lines is None:
                continue
            segments = lines.reshape(-1, 4)
            pixel_delta = segments[:, 2:].astype(float) - segments[:, :2]
            # Python's original sort is stable, including equal-length lines.
            order = np.argsort(-np.sum(pixel_delta * pixel_delta, axis=1), kind="stable")
            segments = segments[order]
            points = np.stack((grid.xs[segments[:, [1, 3]]], grid.ys[segments[:, [0, 2]]]), axis=2)
            delta = points[:, 1] - points[:, 0]
            directions = delta / np.linalg.norm(delta, axis=1)[:, None]
            mids = points.mean(axis=1)
            nearest = np.argmin(np.linalg.norm(path[None, :, :] - mids[:, None, :], axis=2), axis=1)
            forwards = tangent[nearest]
            alignment = np.sum(directions * forwards, axis=1)
            # Preserve scalar dot-product decisions at the acceptance boundary.
            for i in np.flatnonzero(abs(abs(alignment) - 0.35) < 1e-12):
                directions[i] = delta[i] / np.linalg.norm(delta[i])
                alignment[i] = directions[i] @ forwards[i]
            probes = []
            for i in np.flatnonzero(~(abs(alignment) > 0.35)):
                t = directions[i]
                mid = mids[i]
                k = nearest[i]
                forward = forwards[i]
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
                probes.append((center, n, t))
            if not probes:
                continue
            values, visibility, saturations, peaks, baselines = self._profiles(
                probes, details, offs, lat, kernel
            )
            for (center, n, _), vals, valid, sat, peak, baseline in zip(
                probes, values, visibility, saturations, peaks, baselines
            ):
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
                    good
                    & (core_sat <= 105)
                    & (core > shoulders + np.maximum(8.0, 0.08 * shoulders))
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

    def _profiles(self, probes, details, offsets, lateral, kernel):
        """Sample all qualified strip geometries together, preserving order.

        The candidate budget and pixel coordinates are unchanged. Morphology
        runs down each independent profile column; no adjacent candidate is
        mixed into its brightness baseline.
        """
        centers, normals, tangents = (np.asarray(items) for items in zip(*probes))
        points = (
            centers[:, None, None, :]
            + offsets[None, :, None, None] * normals[:, None, None, :]
            + lateral[None, None, :, None] * tangents[:, None, None, :]
        )
        shape = points.shape[:-1]
        values, visible = self.pixels.sample(details["value"], points.reshape(-1, 2))
        saturation, _ = self.pixels.sample(details["saturation"], points.reshape(-1, 2))
        values, visible, saturation = (a.reshape(shape) for a in (values, visible, saturation))
        profiles = np.median(values, axis=2)
        baseline = cv2.morphologyEx(
            profiles.T.astype(np.float32), cv2.MORPH_OPEN, kernel
        ).T
        peaks = profiles - baseline
        peaks[visible.mean(axis=2) < 0.85] = 0
        return values, visible, saturation, peaks, baseline

    def detect(self, bgr, details, road, now, motion=None):
        if not self.enabled:
            return dict(stripes=[], marker_visible=False, confirmed=False, marker=None)
        stripes = self.stripes(details, road)
        current = None
        consistent = False
        if (
            self.previous is not None
            and 0 < now - self.previous["time"] <= self.pixels.policy.motion_max_interval_s
        ):
            predicted = self.previous["center"]
            normal = self.previous["normal"]
            sigma = self.previous["sigma"]
            if motion is not None:
                rotation = np.asarray(motion["rotation"])
                predicted = rotation @ predicted + motion["translation"]
                normal = rotation @ normal
                sigma = math.hypot(sigma, float(motion["sigma_m"]))
            matches = [
                (float(np.linalg.norm(np.asarray(s["center"]) - predicted)), s)
                for s in stripes
                if float(np.asarray(s["normal"]) @ normal) > 0.9
            ]
            if matches:
                error, current = min(matches, key=lambda a: a[0])
                consistent = error < min(0.15, 0.03 + 3 * math.hypot(sigma, current["sigma_m"]))
        if not consistent:
            current = min(stripes, key=lambda s: s["distance_m"]) if stripes else None
            self.confirmation_since = float(now) if current is not None else None
        self.confirmation_frames = (
            self.confirmation_frames + 1 if consistent else int(current is not None)
        )
        self.previous = (
            dict(
                time=now,
                center=np.asarray(current["center"]),
                normal=np.asarray(current["normal"]),
                sigma=current["sigma_m"],
            )
            if current
            else None
        )
        confirmed = self.confirmation_frames >= 3
        reason = (
            "CONFIRMED"
            if confirmed
            else "CONFIRMING_LINE"
            if current
            else "NO_TRANSVERSE_WHITE_BAND"
        )
        result = dict(
            stripes=stripes,
            marker_visible=bool(current),
            confirmed=confirmed,
            marker=current if confirmed else None,
            confirmation_frames=self.confirmation_frames,
            first_observed_at=self.confirmation_since if confirmed else None,
            reason=reason,
        )
        self.status = dict(result, marker=current)
        return result

    def draw(self, bgr):
        marker = self.status.get("marker")
        if marker is None:
            return
        center = np.asarray(marker["center"])
        normal = np.asarray(marker["normal"])
        tangent = np.array([-normal[1], normal[0]])
        points = center + np.array([-0.5, 0.5])[:, None] * self.pixels.width * tangent
        q = np.c_[points, np.ones(2)] @ self.pixels.Hinv.T
        if np.any(abs(q[:, 2]) < 1e-8):
            return
        xy = q[:, :2] / q[:, 2, None] * [320, 240]
        if not np.isfinite(xy).all():
            return
        a, b = np.clip(np.rint(xy), -1000, 1000).astype(int)
        color = (255, 0, 255) if self.status["confirmed"] else (0, 165, 255)
        cv2.line(bgr, tuple(a), tuple(b), color, 2)
        text = (
            "COUNT LINE"
            if self.status["confirmed"]
            else "COUNT LINE %d/3" % self.confirmation_frames
        )
        x = int(np.clip(min(a[0], b[0]), 2, 210))
        y = int(np.clip(min(a[1], b[1]) - 6, 12, 234))
        cv2.putText(bgr, text, (x, y), 0, 0.35, color, 1)


class LapCounter:
    def __init__(self, config, front_m=0.02, rear_m=0.16, half_width_m=0.085,
                 *, reference_origin_m=(0.0, 0.0)):
        self.enabled = config.get("enabled", False)
        self.total = validate_laps(config.get("laps", 3))
        self.front = float(front_m)
        ignore = config.get("initial_line_ignore_s", 30.0)
        if (isinstance(ignore, bool) or not isinstance(ignore, (int, float))
                or not math.isfinite(ignore) or ignore <= 0):
            raise ValueError("Invalid initial white-line ignore window")
        self.initial_line_ignore_s = float(ignore)
        distance = config.get("finish_distance_m", 0.40)
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise ValueError("Invalid finish distance")
        self.finish_distance = float(distance)
        if not math.isfinite(self.finish_distance) or self.finish_distance <= 0:
            raise ValueError("Invalid finish distance")
        self.origin = np.asarray(reference_origin_m, float)
        if self.origin.shape != (2,) or not np.isfinite(self.origin).all():
            raise ValueError("Invalid finish reference origin")
        self.rear = float(rear_m)
        self.half = float(half_width_m)
        if not all(math.isfinite(v) and v > 0 for v in (self.front, self.rear, self.half)):
            raise ValueError("Invalid lap body geometry")
        if not isinstance(self.enabled, bool):
            raise ValueError("Invalid lap counter enable flag")
        self.reset()

    def prepare(self, total=None):
        """Configure a new idle mission; an in-progress race keeps its target."""
        if self.running:
            raise RuntimeError("比赛进行中不能修改目标圈数")
        target = self.total if total is None else validate_laps(total)
        self.total = target
        self.reset()

    def reset(self):
        self.count = 0
        self.running = False
        self.complete = False
        self.phase = "WAIT_START"
        self.track = None
        self.departing = True
        self.started_at = None
        self.first_line_seen_at = None
        self.first_line_ignored = None
        self.visit_done = False
        self.clear_since = None
        self.event = None
        self.missed = 0
        self.release_track = None
        self.finish_motion_verified = False
        self.finish_travel = 0.0
        self.finish_variance = 0.0
        self.finish_unknown = 0.0
        self.rear_cleared = False

    def begin(self, observation, now):
        self.reset()
        self.started_at = float(now)
        if not math.isfinite(self.started_at):
            raise ValueError("Invalid race start time")
        self.running = True
        self.phase = "SEEK_DEPARTURE" if self.enabled else "DEPARTURE"
        if self.enabled and observation.get("confirmed") and observation.get("marker"):
            self._acquire(observation["marker"], now, observation.get("first_observed_at"))
            self.phase = "DEPARTURE"

    def stop(self):
        self.running = False
        if not self.complete:
            self.phase = "STOPPED"

    def _acquire(self, marker, now, first_observed_at=None):
        if self.departing and self.first_line_seen_at is None:
            # Classify the first confirmed visit once, by the beginning of its
            # consistent observations. Crossing later cannot change that vote.
            seen = float(now)
            if (isinstance(first_observed_at, (int, float))
                    and not isinstance(first_observed_at, bool)
                    and math.isfinite(first_observed_at) and first_observed_at <= seen):
                seen = float(first_observed_at)
            self.first_line_seen_at = max(self.started_at, seen)
            self.first_line_ignored = bool(
                self.first_line_seen_at-self.started_at <= self.initial_line_ignore_s
            )
        self.track = dict(
            normal=np.array(marker["normal"], float),
            distance=float(marker["distance_m"]),
            sigma=float(marker["sigma_m"]),
            seen=now,
        )
        self.visit_done = False
        self.clear_since = None

    def _rear_limit(self, normal):
        # Minimum body projection onto the line normal; includes vehicle width
        # when it crosses obliquely. All dimensions use the camera ground frame.
        return min(self.front * normal[0], -self.rear * normal[0]) - self.half * abs(normal[1])

    def finish_position(self):
        r = self.release_track
        if self.count < self.total or r is None:
            return None
        clearance = self._rear_limit(r["normal"]) - r["distance"]
        lower = clearance - 2 * r["sigma"]
        return dict(
            rear_clearance_m=round(float(clearance), 3),
            rear_clearance_lower_m=round(float(lower), 3),
            remaining_m=round(float(max(0.0, -lower)), 3),
            sigma_m=round(float(r["sigma"]), 3),
            motion_verified=self.finish_motion_verified,
        )

    def _finish_update(self, motion):
        """Run at normal road-controlled speed, then stop once, without docking.

        Integrate measured chassis-forward displacement from the confirmed
        final crossing. The crossing frame starts the baseline and contributes
        no pre-line movement. Rotation about the chassis origin is not travel;
        reverse travel undoes progress. Time and motor PWM never create metres.
        """
        self.finish_motion_verified = False
        if not motion:
            return False
        try:
            rotation = np.asarray(motion["rotation"], float)
            translation = np.asarray(motion["translation"], float)
            sigma = float(motion["sigma_m"])
            finite = (rotation.shape == (2, 2) and translation.shape == (2,)
                      and np.isfinite(rotation).all() and np.isfinite(translation).all()
                      and math.isfinite(sigma) and sigma >= 0)
        except (KeyError, ValueError, TypeError):
            finite = False
        if not finite:
            # Malformed motion cannot poison the distance comparison with NaN.
            self.finish_travel = self.finish_variance = self.finish_unknown = 0.0
            return False
        if not motion.get("valid"):
            # A command prior is not measured travel. Its magnitude only widens
            # the bound for motion that could have undone previous progress.
            self.finish_unknown += (
                float(np.linalg.norm(translation)) + sigma
                + 2 * float(np.linalg.norm(self.origin))
                * abs(math.sin(math.atan2(rotation[1, 0], rotation[0, 0]) / 2))
            )
            return False
        self.finish_motion_verified = True
        translated = self.origin @ rotation.T + translation - self.origin
        forward = float((-translated @ rotation)[0])
        self.finish_travel += forward
        self.finish_variance += sigma * sigma
        # Retain the final line only for rear-clearance diagnostics. Neither a
        # new stripe nor a lost line restarts the measured distance runout.
        r = self.release_track
        if r is not None:
            n = rotation @ r["normal"]
            r["distance"] += float(n @ translation)
            r["normal"] = n
            r["sigma"] = math.hypot(r["sigma"], sigma)
            # State flags cross the logging/HTTP boundary and must be Python
            # bools: comparison against NumPy geometry returns numpy.bool_.
            self.rear_cleared = bool(
                self._rear_limit(n) - r["distance"] - 2 * r["sigma"] >= 0
            )
        if self._finish_lower() + 1e-9 < self.finish_distance:
            return False
        self.complete = True
        self.running = False
        self.phase = "FINISHED"
        self.event = "FINISH_DISTANCE_REACHED"
        return True

    def _finish_lower(self):
        return self.finish_travel - 2 * math.sqrt(self.finish_variance) - self.finish_unknown

    def update(self, observation, motion, now, active):
        self.event = None
        if not self.enabled or not active or not self.running or self.complete:
            return False
        if self.phase == "FINISH_DISTANCE":
            return self._finish_update(motion)
        if self.release_track is not None and motion:
            r = self.release_track
            n = np.asarray(motion["rotation"]) @ r["normal"]
            r["distance"] += float(n @ motion["translation"])
            r["normal"] = n
            r["sigma"] = math.hypot(r["sigma"], float(motion["sigma_m"]))
        if self.track is not None:
            t = self.track
            if motion:
                n = np.asarray(motion["rotation"]) @ t["normal"]
                t["distance"] += float(n @ motion["translation"])
                t["normal"] = n
                t["sigma"] = math.hypot(t["sigma"], float(motion["sigma_m"]))
            matches = []
            for m in observation.get("stripes", []):
                n = np.asarray(m["normal"])
                error = abs(float(m["distance_m"]) - t["distance"])
                if n @ t["normal"] > 0.9 and error < min(
                    0.08, 0.03 + 2 * math.hypot(t["sigma"], float(m["sigma_m"]))
                ):
                    matches.append((error, m))
            if matches:
                m = min(matches, key=lambda a: a[0])[1]
                t.update(
                    normal=np.array(m["normal"]),
                    distance=float(m["distance_m"]),
                    sigma=float(m["sigma_m"]),
                    seen=now,
                )
            passed = bool(
                motion
                and motion.get("valid")
                and t["sigma"] < 0.08
                and t["distance"] + 2 * t["sigma"] < self.front
            )
            if passed:
                self.release_track = dict(t, normal=t["normal"].copy())
                self.visit_done = True
                self.track = None
                if self.departing and self.first_line_ignored:
                    self.event = "DEPARTURE_SKIPPED"
                else:
                    self.count += 1
                    self.event = "LAP_COMPLETED"
                    if self.count >= self.total:
                        self.phase = "FINISH_DISTANCE"
                        # Baseline is this confirmed crossing, not the beginning
                        # of the frame interval that approached the white line.
                        return False
                self.phase = "WAIT_CLEAR"
            elif t["sigma"] >= 0.08:
                self.release_track = dict(t, normal=t["normal"].copy())
                self.track = None
                self.visit_done = True
                self.missed += int(not (self.departing and self.first_line_ignored))
                self.phase = "WAIT_CLEAR"
                self.event = "CROSSING_UNVERIFIED"
        if self.visit_done:
            r = self.release_track
            back = None if r is None else self._rear_limit(r["normal"])
            past_line = bool(
                r is not None
                and motion
                and motion.get("valid")
                and r["distance"] + 2 * r["sigma"] < back
            )
            if past_line and not observation.get("marker_visible"):
                if self.clear_since is None:
                    self.clear_since = now
                if now - self.clear_since >= 0.6:
                    self.departing = False
                    self.visit_done = False
                    self.clear_since = None
                    self.release_track = None
                    self.phase = "SEEK_RETURN"
            else:
                self.clear_since = None
        elif self.track is None and observation.get("confirmed") and observation.get("marker"):
            self._acquire(observation["marker"], now, observation.get("first_observed_at"))
            self.phase = "DEPARTURE" if self.departing and self.first_line_ignored else "TRACK_RETURN"
        return False

    def snapshot(self):
        track = self.track
        return dict(
            enabled=self.enabled,
            completed_laps=self.count,
            total_laps=self.total,
            phase=self.phase,
            complete=self.complete,
            event=self.event,
            unverified_crossings=self.missed,
            finish_distance_m=self.finish_distance,
            finish_travel_m=round(self.finish_travel, 3),
            finish_travel_lower_m=round(max(0.0, self._finish_lower()), 3),
            finish_remaining_m=round(max(0.0, self.finish_distance - self._finish_lower()), 3),
            finish_motion_verified=self.finish_motion_verified,
            rear_line_cleared=self.rear_cleared,
            finish_position=self.finish_position(),
            marker_distance_m=None if track is None else round(track["distance"], 3),
            marker_sigma_m=None if track is None else round(track["sigma"], 3),
            clearing_distance_m=None
            if self.release_track is None
            else round(self.release_track["distance"], 3),
            initial_line_ignore_s=self.initial_line_ignore_s,
            first_line_seen_after_start_s=(
                None if self.first_line_seen_at is None
                else round(self.first_line_seen_at-self.started_at, 3)
            ),
            first_line_ignored=self.first_line_ignored,
            departure_not_counted=self.first_line_ignored,
        )
