"""Metric zebra observations and a bounded, separately enabled yielding task.

Rules describe seven 10x2cm longitudinal bars, separated by 2cm. Detection
allows missing bars but requires repeated metric geometry INSIDE the road.
No route, frame number, image row or elapsed lap time is a trigger.
"""

from dataclasses import dataclass, fields
import math
from statistics import median
import cv2
import numpy as np

from robot.core.stationarity import StationaryEvidence


# Original race rules, visual racing IV.1(3): whole-body projection strictly
# before the crossing and less than20cm away. The braking buffer is separate.
CROSSWALK_STOP_LIMIT_M = 0.20


def legal_stop_interval(interval):
    return (
        interval is not None
        and len(interval) == 2
        and all(math.isfinite(x) for x in interval)
        and 0 < interval[0] <= interval[1] < CROSSWALK_STOP_LIMIT_M
    )


@dataclass
class CrosswalkPolicy:
    mode: str = "observe"
    body_front_m: object = None
    body_rear_m: object = None
    body_half_width_m: object = None
    geometry_verified: bool = False
    brake_lamp_verified: bool = False
    brake_deceleration_m_s2: object = None
    hold_s: float = 3.2
    maximum_hold_s: float = 4.8
    stop_clearance_m: float = 0.14
    brake_entry_m: float = 0.15
    distance_error_m: float = 0.015
    observation_max_age_s: float = 1.0
    maximum_position_error_m: float = 0.035
    confirmation_frames: int = 3
    confirmation_gap_s: float = 0.45
    association_m: float = 0.065
    maximum_prediction_travel_m: float = 0.10
    stationary_speed_m_s: float = 0.006
    stationary_yaw_rad_s: float = 0.025
    stationary_confirm_s: float = 0.25
    stop_confirm_timeout_s: float = 1.0
    approach_speed_m_s: float = 0.08
    command_latency_s: float = 0.35
    speed_per_pwm_m_s: float = 0.006
    speed_response_s: float = 0.4
    minimum_bars: int = 4
    bar_length_min_m: float = 0.045
    bar_length_max_m: float = 0.17
    bar_width_min_m: float = 0.008
    bar_width_max_m: float = 0.045
    bar_pitch_m: float = 0.04
    pitch_tolerance_m: float = 0.014
    row_alignment_m: float = 0.035
    road_margin_m: float = 0.018

    @classmethod
    def from_config(cls, config):
        keys = {f.name for f in fields(cls)}
        unknown = set(config) - keys
        if unknown:
            raise ValueError("Unknown crosswalk settings: " + str(sorted(unknown)))
        p = cls(**config)
        if not isinstance(p.geometry_verified, bool) or not isinstance(p.brake_lamp_verified, bool):
            raise ValueError("Verification flags must be boolean")
        if p.mode not in ("observe", "yield"):
            raise ValueError("Invalid crosswalk mode")
        for f in fields(cls):
            value = getattr(p, f.name)
            if f.name in ("mode", "geometry_verified", "brake_lamp_verified") or value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Invalid crosswalk " + f.name)
        if not 3 < p.hold_s < p.maximum_hold_s < 5:
            raise ValueError("Yield hold must be >3s and <5s")
        if not 0 < p.stop_clearance_m < CROSSWALK_STOP_LIMIT_M:
            raise ValueError("Stop clearance must be below 20cm")
        if not 0 < p.brake_entry_m < CROSSWALK_STOP_LIMIT_M:
            raise ValueError("Brake entry must be below 20cm")
        if not isinstance(p.confirmation_frames, int) or p.confirmation_frames < 2:
            raise ValueError("Need multiple observations")
        if not isinstance(p.minimum_bars, int) or not 4 <= p.minimum_bars <= 7:
            raise ValueError("Invalid zebra bar count")
        return p

    def blockers(self):
        missing = []
        for name in ("body_front_m", "body_rear_m", "body_half_width_m", "brake_deceleration_m_s2"):
            if getattr(self, name) is None:
                missing.append(name.upper() + "_REQUIRED")
        if not self.geometry_verified:
            missing.append("BODY_GEOMETRY_UNVERIFIED")
        if not self.brake_lamp_verified:
            missing.append("BRAKE_LAMP_UNVERIFIED")
        return missing


def crosswalk_body_support(normal, policy):
    x, y = normal
    return (x * float(policy.body_front_m) if x >= 0 else -x * float(policy.body_rear_m)) + abs(
        y
    ) * float(policy.body_half_width_m)


class CrosswalkDetector:
    def __init__(self, pixels, policy):
        self.pixels = pixels
        self.policy = policy
        # The calibrated grid is fixed; preserve the original float32 rounding
        # before minAreaRect without rebuilding coordinates for every contour.
        self._metric_points = np.empty((len(pixels.xs), len(pixels.ys), 2), np.float32)
        self._metric_points[:, :, 0] = pixels.xs[:, None]
        self._metric_points[:, :, 1] = pixels.ys[None, :]
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
        eligible = (saturation <= 105) & grid.visible & (corridor != 0)
        for source, values in (("raw", value), ("contrast", details.get("contrast_value"))):
            if values is None:
                continue
            best = self._detect_value(
                values, saturation, corridor, path, tangent, now, source, eligible
            )
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

    def _detect_value(self, value, saturation, corridor, path, tangent, now, source, eligible=None):
        p = self.policy
        grid = self.pixels
        best = None
        if eligible is None:
            eligible = (saturation <= 105) & grid.visible & (corridor != 0)
        for level in grid.white_value_levels:
            mask = (eligible & (value >= level)).astype(np.uint8) * 255
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
                points = self._metric_points[rc[:, 1], rc[:, 0]]
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


class CrosswalkTask:
    """Instance association, metric stop envelope, dwell and spatial rearming.

    Prediction requires measured visual motion; PWM is never used to move a
    crossing. An unlocalized served event cannot veto a newly confirmed
    forward observation; uncertain identity is recorded rather than guessed.
    """

    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        self.track = None
        self.hits = 0
        self.identity = 0
        self.phase = "SEARCH"
        self.failure = None
        self.observation_association = "NONE"
        self.pending_track = None
        self.pending_hits = 0
        self.retired_event = None
        self.issues = []
        self.recovery_reason = None
        self.yield_verified = False
        self.yield_completed = False
        self.stationary_since = None
        self.hold_since = None
        self.stopping_since = None
        self.stop_deadline = None
        self.physical_rest_since = None
        self.hold_completed_s = 0.0
        self.last_update = None
        self.passed = 0
        self.prediction_travel = 0.0
        self.stop_evidence = None
        self.stop_blockers = []
        self.speed_pwm = None
        self.speed_update = None
        self.speed_ceiling = None
        self.last_visual_speed = 0.0
        self.speed_missing_since = None
        self.rest = StationaryEvidence(
            self.policy.stationary_speed_m_s, self.policy.stationary_yaw_rad_s,
            self.policy.stationary_confirm_s, self.policy.confirmation_gap_s,
        )

    def _begin_track(self, observation):
        self.identity += 1
        self.track = dict(observation)
        self.hits = 1
        self.prediction_travel = 0.0
        self.issues = []
        self.recovery_reason = None
        self.yield_verified = False
        self.yield_completed = False
        self.stationary_since = self.hold_since = self.stopping_since = None
        self.stop_deadline = self.physical_rest_since = None
        self.rest.reset()
        self.hold_completed_s = 0.0
        self.stop_evidence = None
        self.stop_blockers = []
        self.speed_pwm = self.speed_update = self.speed_ceiling = self.speed_missing_since = None

    def _pending_crossing(self, observation, now):
        """Confirm fresh geometry independently of an unlocalized served event.

        This does not prove a different physical instance. When identity is
        uncertain, a repeated forward crossing must be yielded to again instead
        of inheriting the old event's completed permission to pass.
        """
        p = self.policy
        clearance = observation["near_m"] - crosswalk_body_support(observation["normal"], p)
        if clearance <= observation["sigma_m"]:
            self.pending_track = None
            self.pending_hits = 0
            return
        old = self.pending_track
        consistent = (
            old is not None
            and now - old["observed_at"] <= p.confirmation_gap_s
            and np.dot(old["normal"], observation["normal"]) > 0.94
            and abs(old["near_m"] - observation["near_m"]) <= p.association_m
        )
        self.pending_hits = self.pending_hits + 1 if consistent else 1
        self.pending_track = dict(observation)
        self.observation_association = "CONFIRMING_REOBSERVED_CROSSWALK"
        if self.pending_hits >= p.confirmation_frames:
            previous = dict(
                event_id=self.identity,
                yield_completed=self.yield_completed,
                yield_verified=self.yield_verified,
                passage_proven=False,
                reason="SERVED_EVENT_POSITION_UNCERTAIN",
            )
            hits = self.pending_hits
            candidate = self.pending_track
            self._begin_track(candidate)
            self.hits = hits
            self.pending_track = None
            self.pending_hits = 0
            self.phase = "APPROACH"
            self.retired_event = previous
            self.observation_association = "REOBSERVED_CROSSWALK"

    def _stop_interval(self, now):
        if self.stop_evidence is None:
            return None
        e = self.stop_evidence
        # Bound any interval after the latest measured rigid transform, including
        # undetectably small translation and rotation of the whole body.
        p = self.policy
        radius = math.hypot(
            max(float(p.body_front_m), float(p.body_rear_m)), float(p.body_half_width_m)
        )
        drift = (p.stationary_speed_m_s + radius * p.stationary_yaw_rad_s) * max(0.0, now - e["at"])
        return [e["lower_m"] - drift, e["upper_m"] + drift]

    def _advance_stop_evidence(self, motion, now):
        if self.stop_evidence is None:
            return
        e = self.stop_evidence
        p = self.policy
        old_normal = np.asarray(e["normal"])
        normal = np.asarray(motion["rotation"]) @ old_normal
        # Carry the verified body-to-crossing interval through actual visual
        # motion. Observation quality gates creation; accumulated uncertainty
        # afterward is checked against the legal stopping window itself.
        clearance = (e["lower_m"] + e["upper_m"]) / 2
        clearance += (
            crosswalk_body_support(old_normal, p)
            + float(np.dot(normal, motion["translation"]))
            - crosswalk_body_support(normal, p)
        )
        sigma = math.hypot(
            (e["upper_m"] - e["lower_m"]) / 2, float(motion["sigma_m"])
        ) + 0.001 * max(0.0, now - e["at"])
        self.stop_evidence = dict(
            lower_m=clearance - sigma, upper_m=clearance + sigma, normal=normal.tolist(), at=now
        )

    def _verify_stop(
        self, now, stationary, zero_ack, lamp_on, known, clearance, sigma, fresh_position
    ):
        """Keep position evidence separate from the continuous rest/dwell clock."""
        p = self.policy
        self.stop_blockers = []
        for ok, label in (
            (stationary, "PHYSICAL_STOP_UNVERIFIED"),
            (zero_ack, "ZERO_ACK_PENDING"),
            (lamp_on, "BRAKE_LAMP_PENDING"),
        ):
            if not ok:
                self.stop_blockers.append(label)
        if self.stop_blockers:
            if self.hold_since is not None:
                self.stop_deadline = now + p.stop_confirm_timeout_s
            self.stop_evidence = None
            self.stationary_since = self.hold_since = None
            self.phase = "STOPPING"
            return
        interval = self._stop_interval(now)
        measured = [clearance - sigma, clearance + sigma]
        # A fresh observation outside the legal window cannot be overridden
        # by a previous legal pose. Changes within the window remain legal.
        if interval is not None and fresh_position and known:
            if legal_stop_interval(interval) and legal_stop_interval(measured):
                # Both measurements prove membership in the legal window. A
                # change of the estimated centimetre value is not body motion
                # and must not restart an otherwise continuous physical dwell.
                self.stop_evidence = dict(
                    lower_m=measured[0],
                    upper_m=measured[1],
                    normal=list(self.track["normal"]),
                    at=now,
                )
                interval = measured
            else:
                low = max(interval[0], measured[0])
                high = min(interval[1], measured[1])
                if low <= high:
                    self.stop_evidence = dict(
                        lower_m=low, upper_m=high, normal=list(self.track["normal"]), at=now
                    )
                    interval = [low, high]
                else:
                    self.stop_evidence = None
                    self.stationary_since = self.hold_since = None
                    self.phase = "STOPPING"
                    interval = None
        if interval is None and known and legal_stop_interval(measured):
            self.stop_evidence = dict(
                lower_m=measured[0], upper_m=measured[1], normal=list(self.track["normal"]), at=now
            )
            self.stationary_since = now
            interval = measured
        if not legal_stop_interval(interval):
            outside = known and (measured[1] <= 0 or measured[0] >= CROSSWALK_STOP_LIMIT_M)
            self.stop_blockers.append(
                "STOP_POSITION_OUTSIDE_WINDOW" if outside else "STOP_POSITION_UNCERTAIN"
            )
            if not known or not all(math.isfinite(value) for value in measured):
                self.stop_blockers.append("STOP_POSITION_UNKNOWN")
            elif measured[0] >= CROSSWALK_STOP_LIMIT_M:
                self.stop_blockers.append("STOP_POSITION_TOO_FAR")
            elif measured[1] <= 0:
                self.stop_blockers.append("STOP_POSITION_AT_OR_PAST_LINE")
            self.stop_evidence = None
            self.stationary_since = self.hold_since = None
            self.phase = "STOPPING"
            return
        if now - self.stationary_since >= p.stationary_confirm_s:
            if self.hold_since is None:
                self.hold_since = now
                self.phase = "HOLD"

    def _fail(self, reason):
        self.failure = reason
        self.phase = "FAULT"

    def _issue(self, reason):
        if reason not in self.issues:
            self.issues.append(reason)
        self.recovery_reason = reason

    def _request_stop(self, now):
        p = self.policy
        speed = max(0.0, self.last_visual_speed)
        self.phase = "STOPPING"
        self.stopping_since = now
        self.stationary_since = None
        self.stop_deadline = now + max(
            p.stop_confirm_timeout_s,
            speed / float(p.brake_deceleration_m_s2) + p.command_latency_s + p.stationary_confirm_s,
        )
        self.physical_rest_since = None
        self.speed_pwm = None
        self.speed_update = None

    def update(
        self,
        observation,
        motion,
        now,
        active=False,
        zero_ack=False,
        lamp_on=False,
        road_ready=False,
    ):
        p = self.policy
        self.observation_association = "NONE"
        if self.failure:
            return self.snapshot(now)
        if p.mode == "yield" and p.blockers():
            self._fail("CROSSWALK SETUP REQUIRED")
            return self.snapshot(now)
        dt = 0 if self.last_update is None else now - self.last_update
        self.last_update = now
        valid_motion = bool(motion and motion.get("valid"))
        if valid_motion and motion.get("speed_m_s") is not None:
            self.last_visual_speed = float(motion["speed_m_s"])
        if valid_motion and self.phase in ("STOPPING", "HOLD", "RECOVERY_HOLD"):
            self._advance_stop_evidence(motion, now)
        if self.track is not None and dt > 0:
            if valid_motion:
                n = np.asarray(motion["rotation"]) @ np.asarray(self.track["normal"])
                shift = float(np.dot(n, motion["translation"]))
                self.track["normal"] = n.tolist()
                self.track["near_m"] += shift
                self.track["far_m"] += shift
                self.track["sigma_m"] = math.hypot(
                    self.track["sigma_m"], float(motion["sigma_m"])
                ) + 0.001 * max(0, dt)
                self.prediction_travel += float(motion["travel_m"])
            else:
                self.track["sigma_m"] += max(p.approach_speed_m_s, self.last_visual_speed) * max(
                    0, dt
                )
        if self.pending_track is not None:
            if valid_motion:
                n = np.asarray(motion["rotation"]) @ np.asarray(self.pending_track["normal"])
                shift = float(np.dot(n, motion["translation"]))
                self.pending_track["normal"] = n.tolist()
                self.pending_track["near_m"] += shift
                self.pending_track["far_m"] += shift
                self.pending_track["sigma_m"] = math.hypot(
                    self.pending_track["sigma_m"], float(motion["sigma_m"])
                ) + 0.001 * max(0, dt)
            else:
                self.pending_track["sigma_m"] += max(
                    p.approach_speed_m_s, self.last_visual_speed
                ) * max(0, dt)
            if now - self.pending_track["observed_at"] > p.confirmation_gap_s:
                self.pending_track = None
                self.pending_hits = 0
        matched = False
        if observation is not None:
            if self.track is None:
                self._begin_track(observation)
                self.observation_association = "NEW"
            else:
                matched = (
                    np.dot(self.track["normal"], observation["normal"]) > 0.94
                    and abs(self.track["near_m"] - observation["near_m"]) <= p.association_m
                )
                if matched:
                    self.pending_track = None
                    self.pending_hits = 0
                    self.observation_association = "MATCHED"
                    gap = observation["observed_at"] - self.track["observed_at"]
                    if self.hits < p.confirmation_frames and gap > p.confirmation_gap_s:
                        self.hits = 0
                    self.hits += 1
                    self.track = dict(observation)
                    self.prediction_travel = 0.0
                elif self.hits < p.confirmation_frames:
                    self._begin_track(observation)
                    self.observation_association = "REPLACED_UNCONFIRMED"
                elif (
                    self.phase == "PASSING"
                    and self.yield_completed
                    and (
                        now - self.track["observed_at"] > p.observation_max_age_s
                        or self.track["sigma_m"] > p.maximum_position_error_m
                    )
                ):
                    self._pending_crossing(observation, now)
                else:
                    self.observation_association = "UNMATCHED_CONFIRMED"
        if self.track is None:
            return self.snapshot(now)
        age = now - self.track["observed_at"]
        confirmed = self.hits >= p.confirmation_frames
        # Preview expiry only. A driving yield event is never erased on a timer.
        if (
            p.mode == "observe" or not active and not self.failure
        ) and age > p.observation_max_age_s:
            self.track = None
            self.hits = 0
            self.phase = "SEARCH"
            return self.snapshot(now)
        if not confirmed and age > p.confirmation_gap_s:
            self.track = None
            self.hits = 0
            self.phase = "SEARCH"
            return self.snapshot(now)
        if confirmed and self.phase == "SEARCH":
            self.phase = "APPROACH"
        if p.mode != "yield" or not active or p.blockers():
            return self.snapshot(now)
        stationary = self.rest.observe(
            motion, now,
            self.phase in ("STOPPING", "HOLD", "RECOVERY_HOLD") and zero_ack and lamp_on,
        )
        # During approach an already stopped legal pose needs no extra move.
        if self.phase == "APPROACH":
            stationary = bool(
                valid_motion and motion.get("speed_m_s") is not None
                and abs(motion["speed_m_s"]) <= p.stationary_speed_m_s
                and abs(motion["yaw_rad_s"]) <= p.stationary_yaw_rad_s
            )
        # A continuously measured stationary scene preserves a stopped pose;
        # it does not refresh the time of the last zebra observation.
        stationary_hold = self.phase in ("STOPPING", "HOLD", "RECOVERY_HOLD") and stationary
        sigma = self.track["sigma_m"]
        known = (
            (age <= p.observation_max_age_s or stationary_hold)
            and sigma <= p.maximum_position_error_m
            and self.prediction_travel <= p.maximum_prediction_travel_m
        )
        # At rest the legal interval, rather than a fixed centimetre accuracy
        # gate, decides whether position is precise enough for a verified yield.
        position_available = (
            age <= p.observation_max_age_s or stationary_hold
        ) and self.prediction_travel <= p.maximum_prediction_travel_m
        clearance = self.track["near_m"] - crosswalk_body_support(self.track["normal"], p)
        legal_rest = (
            stationary
            and position_available
            and legal_stop_interval([clearance - sigma, clearance + sigma])
        )
        if self.phase == "APPROACH" and confirmed and not known and not legal_rest:
            self._issue("CROSSWALK POSITION UNCERTAIN")
            self._request_stop(now)
        # After yielding, the old crossing is retained solely to prove passage
        # and suppress duplicate events. Age alone cannot cause repeated stops.
        if self.phase == "PASSING":
            # The yield has already been served. Uncertainty about an old
            # crossing blocks rearming, not lane following. Keep its tombstone
            # until the entire far-edge uncertainty interval is behind the car.
            # Missing flow expands that interval; neither PWM nor elapsed time
            # can prove passage. Lane/motion safety supervision stays separate.
            known = valid_motion and math.isfinite(sigma)
        if self.phase == "APPROACH" and confirmed:
            n = self.track["normal"]
            clearance = self.track["near_m"] - crosswalk_body_support(n, p)
            if legal_rest and zero_ack and lamp_on:
                self._request_stop(now)
            elif clearance - sigma <= 0:
                self._issue("CROSSWALK STOP WINDOW MISSED")
                self._request_stop(now)
            else:
                speed = max(0.0, self.last_visual_speed)
                stopping_distance = speed * p.command_latency_s + speed * speed / (
                    2 * float(p.brake_deceleration_m_s2)
                )
                # Brake as soon as every possible resting position lies in
                # the legal window. Braking distance bounds forward coasting;
                # it is not an exact offset to subtract from the 20 cm limit.
                # The lower buffer remains an independent last-brake boundary,
                # never a centimetre setpoint to creep toward inside the window.
                stopping_interval = [clearance - sigma - stopping_distance, clearance + sigma]
                if (
                    (clearance <= p.brake_entry_m and legal_stop_interval(stopping_interval))
                    or clearance <= p.stop_clearance_m + sigma + stopping_distance
                ):
                    self._request_stop(now)
        if self.phase in ("STOPPING", "HOLD", "RECOVERY_HOLD"):
            verified_rest = stationary and zero_ack and lamp_on
            if verified_rest and self.physical_rest_since is None:
                self.physical_rest_since = now
            elif not verified_rest:
                self.physical_rest_since = None
            clearance = self.track["near_m"] - crosswalk_body_support(self.track["normal"], p)
            # One stop request owns this event through the complete dwell.
            # A fresh estimate saying the car stopped too far is an exception,
            # not permission to alternate zero output and another approach.
            self._verify_stop(
                now, stationary, zero_ack, lamp_on, position_available,
                clearance, sigma, fresh_position=matched,
            )
            # One physical dwell clock serves both a legally located stop and
            # a recorded position exception. Position verification never adds
            # a second full hold, and ordinary flow jitter cannot restart it.
            rest_elapsed = (0.0 if self.physical_rest_since is None
                            else now-self.physical_rest_since)
            if verified_rest and rest_elapsed >= p.stationary_confirm_s:
                located = legal_stop_interval(self._stop_interval(now))
                self.phase = "HOLD" if located else "RECOVERY_HOLD"
                self.hold_since = self.physical_rest_since
                if not located:
                    self._issue("CROSSWALK STOP UNVERIFIED")
                if rest_elapsed >= p.hold_s:
                    self.hold_completed_s = rest_elapsed
                    self.yield_completed = True
                    self.yield_verified = located
                    if located:
                        self.recovery_reason = None
                    self.phase = "PASSING"
            elif (self.stop_deadline is not None and now > self.stop_deadline
                  and self.stop_evidence is None):
                self._issue("CROSSWALK STOP UNVERIFIED")
            if (
                self.phase in ("STOPPING", "HOLD", "RECOVERY_HOLD")
                and self.physical_rest_since is not None
                and now - self.physical_rest_since >= p.maximum_hold_s
            ):
                self._issue("CROSSWALK HOLD EXCEEDED")
        if self.phase == "PASSING" and known:
            n = self.track["normal"]
            rear = -crosswalk_body_support([-n[0], -n[1]], p)
            if self.track["far_m"] + sigma < rear:
                self.passed += 1
                self.track = None
                self.hits = 0
                self.phase = "SEARCH"
                self.stationary_since = self.hold_since = self.stopping_since = None
        return self.snapshot(now)

    def _position_outcome(self, now):
        """Describe the stop proof using existing evidence and retained blockers.

        After an exceptional dwell, blockers preserve the stopped position's
        outcome while the car advances through the crossing.
        """
        if self.yield_verified or (
            self.phase in ("STOPPING", "HOLD", "RECOVERY_HOLD")
            and legal_stop_interval(self._stop_interval(now))
        ):
            return "LEGAL"
        for blocker, outcome in (
            ("STOP_POSITION_TOO_FAR", "TOO_FAR"),
            ("STOP_POSITION_AT_OR_PAST_LINE", "AT_OR_PAST_LINE"),
            ("STOP_POSITION_UNKNOWN", "UNKNOWN"),
            ("STOP_POSITION_UNCERTAIN", "UNCERTAIN"),
        ):
            if blocker in self.stop_blockers:
                return outcome
        return "NOT_VERIFIED"

    def snapshot(self, now):
        p = self.policy
        t = self.track
        distance = (
            None
            if t is None
            or p.body_front_m is None
            or p.body_rear_m is None
            or p.body_half_width_m is None
            else t["near_m"] - crosswalk_body_support(t["normal"], p)
        )
        return dict(
            mode=p.mode,
            phase=self.phase,
            observation_association=self.observation_association,
            pending_observation=dict(self.pending_track) if self.pending_track is not None else None,
            pending_hits=self.pending_hits,
            retired_event=self.retired_event,
            event_id=self.identity if t else None,
            hits=self.hits,
            confirmed=self.hits >= p.confirmation_frames,
            # Motion compensation updates these records in place next frame.
            # Published status must retain the captured geometry.
            observation=dict(t) if t is not None else None,
            distance_to_body_m=distance,
            position_error_m=t["sigma_m"] if t else None,
            observation_age_s=now - t["observed_at"] if t else None,
            hold_elapsed_s=(
                now - self.hold_since
                if self.phase in ("HOLD", "RECOVERY_HOLD") and self.hold_since is not None
                else self.hold_completed_s
                if self.phase == "PASSING"
                else 0.0
            ),
            stop_verified=self.stop_evidence is not None and self.phase in ("STOPPING", "HOLD"),
            yield_completed=self.yield_completed,
            yield_verified=self.yield_verified,
            position_outcome=self._position_outcome(now),
            recovery_reason=self.recovery_reason,
            issues=list(self.issues),
            stop_clearance_interval_m=self._stop_interval(now)
            if self.phase in ("STOPPING", "HOLD")
            else None,
            stop_blockers=list(self.stop_blockers),
            passed=self.passed,
            blockers=p.blockers(),
            failure=self.failure,
            physical_rest_elapsed_s=(max(0.0, now-self.physical_rest_since)
                                     if self.physical_rest_since is not None else 0.0),
            required_hold_s=p.hold_s,
            brake_entry_m=p.brake_entry_m,
            speed_ceiling_m_s=self.speed_ceiling,
            regulated_pwm=self.speed_pwm,
            legal_stop_window_m=[0.0, CROSSWALK_STOP_LIMIT_M],
            braking_buffer_m=p.stop_clearance_m,
        )

    def allows_road_search(self):
        """Task ownership shared by path search and its restart-envelope logic."""
        return self.track is None or (self.phase == "PASSING" and self.pending_track is None)

    def supervision_command(self, road_command, processing_ms, budget_ms):
        # A planned zero-speed dwell needs fresh vision, but no steering target.
        # This exception cannot feed a stale frame or authorize nonzero motion.
        if (
            self.policy.mode == "yield"
            and self.phase in ("STOPPING", "HOLD", "RECOVERY_HOLD")
            and processing_ms <= budget_ms
        ):
            return (0, 0)
        return road_command

    def limit_command(self, command, snapshot, motion):
        p = self.policy
        if p.mode != "yield":
            return command
        if p.blockers() or self.failure:
            return (0, 0)
        if self.phase in ("STOPPING", "HOLD", "RECOVERY_HOLD"):
            return (0, 0)
        if command is None or self.track is None:
            return command
        # A tentative metric zebra pattern already imposes the braking envelope.
        # Confirmation authorizes the yield event, never full-speed approach.
        if self.phase == "PASSING" and self.pending_track is None:
            return command
        approach = self.pending_track if self.phase == "PASSING" else self.track
        clearance = approach["near_m"] - crosswalk_body_support(approach["normal"], p)
        speed = motion.get("speed_m_s") if motion and motion.get("valid") else None
        if speed is None:
            # No blind nonzero command. A missing speed observation first
            # requests a measured recovery while stopped, rather than latching
            # a driving fault on a single optical-flow rejection.
            if self.speed_missing_since is None:
                self.speed_missing_since = self.last_update
            if self.last_update - self.speed_missing_since >= p.stop_confirm_timeout_s:
                self._issue("CROSSWALK SPEED UNVERIFIED")
            self.speed_update = None
            return (0, 0)
        self.speed_missing_since = None
        if self.recovery_reason == "CROSSWALK SPEED UNVERIFIED":
            self.recovery_reason = None
        available = max(0.0, clearance - approach["sigma_m"] - p.stop_clearance_m)
        decel = float(p.brake_deceleration_m_s2)
        # Speed feedback takes time to settle. Plan the next speed early
        # enough that this response cannot repeatedly hit the emergency-stop
        # boundary while the crossing is still outside the legal window.
        delay = p.command_latency_s + p.speed_response_s
        safe_speed = max(
            0.0, math.sqrt((decel * delay) ** 2 + 2 * decel * available) - decel * delay
        )
        # The task imposes only a distance-dependent braking envelope. A
        # fixed 6cm/s approach cap caused long crawls and repeated early stops.
        ceiling = safe_speed
        previous_ceiling = self.speed_ceiling
        self.speed_ceiling = ceiling
        if (self.speed_pwm is not None and previous_ceiling
                and ceiling < previous_ceiling):
            # Apply a falling reference immediately; integral-only feedback
            # otherwise keeps the old torque until overspeed causes an early
            # emergency stop. Both wheels retain their requested curvature.
            self.speed_pwm *= ceiling / previous_ceiling
        requested = max(0.0, sum(command) / 2.0)
        now = self.last_update
        dt = (
            0.0
            if self.speed_update is None
            else max(0.0, min(p.confirmation_gap_s, now - self.speed_update))
        )
        if self.speed_pwm is None:
            self.speed_pwm = ceiling / p.speed_per_pwm_m_s
        # Integral speed feedback retains the torque needed at low speed. Unlike
        # scaling by current speed, a measured zero cannot restore full cruise.
        self.speed_pwm = max(
            0.0,
            min(
                requested,
                self.speed_pwm
                + (ceiling - speed) * dt / (p.speed_per_pwm_m_s * p.speed_response_s),
            ),
        )
        self.speed_update = now
        scale = min(1.0, self.speed_pwm / requested) if requested > 0 else 0.0
        return tuple(round(value * scale) for value in command)


def draw_crosswalk_overlay(out, snapshot, calibration):
    t = snapshot.get("observation")
    if t:
        corners = np.asarray(t.get("corners", []), float)
        # Boxes belong to the last actual observation; avoid drawing stale boxes.
        if len(corners) and snapshot["observation_age_s"] < 0.05:
            q = (
                np.c_[corners, np.ones(len(corners))]
                @ np.linalg.inv(np.asarray(calibration["H_normalized"])).T
            )
            uv = np.rint(q[:, :2] / q[:, 2, None] * [320, 240]).astype(np.int32)
            for box in uv.reshape(-1, 4, 2):
                cv2.polylines(out, [box], True, (255, 0, 255), 1)
    distance = snapshot.get("distance_to_body_m")
    label = "ZEBRA %s %s" % (snapshot["mode"].upper(), snapshot["phase"])
    if distance is not None:
        label += " %.0fcm" % (distance * 100)
    cv2.putText(out, label, (4, 190), 0, 0.37, (255, 0, 255), 1)
    return out
