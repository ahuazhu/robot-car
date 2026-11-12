"""Short visual search rotations inside a previously measured road corridor.

The corridor is not an extrapolated centreline. Every patch came from accepted
observations and is transported with ground motion; uncertainty keeps growing.
When full map coverage is unavailable, a recent local pose estimate may
authorize a small rotation only if its curvature/distance uncertainty leaves
room for the complete swept body. Neither source creates a driving path.

A separately enabled BoundedVisualScan owns the no-clearance look-around state.
Its telemetry explicitly marks space_verified=False; the map strategy never
borrows that permission, and an active scan cannot switch evidence strategy.
"""

import math
from dataclasses import dataclass
import numpy as np
from robot.core.geometry import (
    arc,
    support_projection,
    contiguous_runs,
    curve_agreement,
    differential,
)
from robot.core.ground_motion import apply_ground_motion
from robot.core.route_confidence import confidence_speed_factor
from robot.navigation.visual_scan import BoundedVisualScan


@dataclass(frozen=True)
class SearchPolicy:
    maximum_yaw_rad: float = math.radians(10.0)
    yaw_rate_rad_s: float = 0.20
    pulse_s: float = 0.08

    def __post_init__(self):
        for value in (self.maximum_yaw_rad, self.yaw_rate_rad_s, self.pulse_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid search policy")
        if self.maximum_yaw_rad >= math.pi / 4 or self.pulse_s > 0.15:
            raise ValueError("Search motion exceeds small-step bounds")


class RoadSearch:
    def __init__(
        self,
        policy,
        origin,
        body,
        width,
        cruise=60.0,
        response_s=0.4,
        search_policy=None,
        *,
        allow_visual_scan=False,
        finish_priority=False,
    ):
        self.policy = policy
        self.origin = np.asarray(origin, float)
        self.body = body
        self.width = float(width)
        self.cruise = cruise
        self.response_s = response_s
        self.limits = search_policy or SearchPolicy()
        self.allow_visual_scan = bool(allow_visual_scan)
        self.finish_priority = bool(finish_priority)
        self.scan = BoundedVisualScan(
            policy, cruise, response_s, self.limits.yaw_rate_rad_s,
            finish_priority=self.finish_priority,
        )
        self.patches = []
        self.pose_hint = None
        self.now = 0.0
        self.space_source = None
        self.pose_error = None
        self.last_hint = 1.0
        self.reset()

    def reset(self, clear_map=False):
        if clear_map:
            self.patches = []
            self.pose_hint = None
        self.began = None
        self.requested_at = None
        self.yaw = 0.0
        self.drift = 0.0
        self.leg = 0
        self.targets = []
        self.last_action = False
        self.awaiting_rest = False
        self._map_proposal = None
        self.finished = False
        self.status = dict(phase="IDLE", reason="", search_pending=False, lease_s=None)
        self.scan.reset()

    def snapshot(self):
        """Expose the selected strategy's evidence and command lease."""
        return self.scan.status if self.scan.began is not None else self.status

    def is_turning(self):
        return self.scan.last_action if self.scan.began is not None else self.last_action

    def applied(self, now, command, issued=True):
        """Advance pulse/rest state only after matching command publication."""
        if self.scan.began is not None:
            return self.scan.applied(now, command, issued=issued)
        proposal = self._map_proposal
        if (
            not issued
            or proposal is None
            or self.began is None
            or self.finished
            or command is None
            or tuple(command) != proposal[1]
            or (proposal[0] == "pulse" and not any(command))
        ):
            return False
        self._map_proposal = None
        self.awaiting_rest = proposal[0] == "pulse"
        return True

    def observe_hint(self, result):
        """Keep the idle scan direction without building or transporting a map."""
        if not result.get("observed") or not self.body:
            return
        path = np.asarray(result.get("path", []), float)
        if path.ndim != 2 or path.shape[1:] != (2,) or not np.isfinite(path).all():
            return
        path = path - self.origin
        mask = np.asarray(result.get("path_observed_mask", []), bool)
        if len(mask) != len(path):
            return
        sigma = float(
            (result.get("road_model") or {}).get("sigma_m") or self.policy.observation_sigma_m
        )
        for ids in contiguous_runs(mask):
            if len(ids) < 5 or arc(path[ids])[-1] < self.policy.minimum_segment_m:
                continue
            fresh = path[ids]
            hint = float(fresh[np.argmin(np.linalg.norm(fresh, axis=1)), 1])
            if abs(hint) > sigma:
                self.last_hint = math.copysign(1.0, hint)

    def observe(self, result, motion, now, *, surface=None):
        self.now = now
        valid = bool(motion and motion.get("valid"))
        # An unmeasured transform breaks the map's registration. A later good
        # relative flow cannot retrospectively repair the missed interval.
        if motion and not valid:
            self.patches = []
            self.pose_hint = None
        if valid and self.pose_hint is not None:
            self.pose_hint["path"] = (
                apply_ground_motion(self.pose_hint["path"] + self.origin, motion) - self.origin
            )
            self.pose_hint["sigma"] = math.hypot(self.pose_hint["sigma"], float(motion["sigma_m"]))
        if motion:
            for patch in self.patches:
                patch["path"] = (
                    apply_ground_motion(patch["path"] + self.origin, motion) - self.origin
                )
                patch["sigma"] = math.hypot(patch["sigma"], float(motion["sigma_m"]))
            if self.began is not None and valid:
                self.yaw -= math.atan2(motion["rotation"][1, 0], motion["rotation"][0, 0])
                rotation = np.asarray(motion["rotation"])
                delta = self.origin @ rotation.T + motion["translation"] - self.origin
                self.drift += float(np.linalg.norm(delta))
        if surface is not None:
            # Stored search geometry uses the body origin; image evidence uses
            # the calibrated camera-ground coordinates after this frame's motion.
            candidates = self.patches + ([] if self.pose_hint is None else [self.pose_hint])
            paths = [(p['path']+self.origin, p['sigma']) for p in candidates]
            reports = (surface.contradictions(paths)
                       if hasattr(type(surface), 'contradictions') else
                       [surface.contradiction(*p) for p in paths])
            count = len(self.patches)
            self.patches = [p for p, r in zip(self.patches, reports[:count])
                            if not r['contradicted']]
            if self.pose_hint is not None and reports[-1]['contradicted']:
                self.pose_hint = None
        radius = (
            self.policy.maximum_target_m + math.hypot(max(self.body[:2]), self.body[2])
            if self.body
            else self.policy.maximum_target_m
        )
        self.patches = [
            p
            for p in self.patches
            if p["sigma"] <= self.policy.maximum_sigma_m
            and np.linalg.norm(p["path"], axis=1).min() < radius
        ]
        if not result.get("observed") or not self.body:
            return
        path = np.asarray(result.get("path", []), float)
        if path.ndim != 2 or path.shape[1:] != (2,) or not np.isfinite(path).all():
            return
        path = path - self.origin
        mask = np.asarray(result.get("path_observed_mask", []), bool)
        if len(mask) != len(path):
            return
        sigma = float(
            (result.get("road_model") or {}).get("sigma_m") or self.policy.observation_sigma_m
        )
        for ids in contiguous_runs(mask):
            if len(ids) < 5 or arc(path[ids])[-1] < self.policy.minimum_segment_m:
                continue
            fresh = path[ids].copy()
            _, curvature = differential(fresh)
            positions = arc(fresh)
            nearest = int(np.argmin(np.linalg.norm(fresh, axis=1)))
            local = abs(positions - positions[nearest]) <= self.policy.curve_analysis_window_m / 2
            self.pose_hint = dict(
                path=fresh.copy(),
                sigma=sigma,
                seen=now,
                curvature=float(np.max(abs(curvature[local]))),
            )
            # A shifted reinitialization must not leave two contradictory free
            # corridors behind. Only accepted geometry may establish a new map.
            conflict = False
            for patch in self.patches:
                match = curve_agreement(patch["path"], fresh)
                if match and (
                    match["distance_m"]
                    > math.hypot(patch["sigma"], sigma) + self.policy.sample_step_m
                    or match["heading_cos"] < self.policy.association_heading_cos
                ):
                    conflict = True
                    break
            if conflict:
                self.patches = []
            retained = []
            for patch in self.patches:
                d, s = support_projection(patch["path"], fresh)
                covered = (
                    float(d.max()) <= self.policy.sample_step_m / 2
                    and s[-1] - s[0] >= arc(patch["path"])[-1] - self.policy.sample_step_m
                )
                if not covered:
                    retained.append(patch)
            self.patches = retained + [dict(path=fresh, sigma=sigma, seen=now)]
            hint = float(fresh[np.argmin(np.linalg.norm(fresh, axis=1)), 1])
            if abs(hint) > sigma:
                self.last_hint = math.copysign(1.0, hint)
        # Bound CPU work by spatial resolution, rather than a lap/frame number.
        count = math.ceil(2 * radius / self.policy.sample_step_m)
        self.patches = sorted(self.patches, key=lambda p: p["seen"], reverse=True)[:count]

    def swept_clear(self, delta_yaw):
        self.space_source = None
        self.pose_error = None
        if not self.body or not math.isfinite(delta_yaw):
            return False
        front, rear, half = self.body
        p = self.policy
        xs = np.linspace(-rear, front, math.ceil((front + rear) / p.sample_step_m) + 1)
        ys = np.linspace(-half, half, math.ceil(2 * half / p.sample_step_m) + 1)
        xx, yy = np.meshgrid(xs, ys)
        foot = np.c_[xx.ravel(), yy.ravel()]
        radius = math.hypot(max(front, rear), half)
        steps = max(2, math.ceil(abs(delta_yaw) * radius / (p.sample_step_m / 2)) + 1)
        angles = np.linspace(0.0, delta_yaw, steps)
        c, s = np.cos(angles), np.sin(angles)
        rotations = np.stack((c, -s, s, c), axis=1).reshape(-1, 2, 2)
        points = (foot[None, :, :] @ rotations.transpose(0, 2, 1)).reshape(-1, 2)
        covered = np.zeros(len(points), bool)
        margin = p.sample_step_m / math.sqrt(2.0)
        for patch in self.patches:
            remaining = np.flatnonzero(~covered)
            if not len(remaining):
                self.space_source = "OBSERVED_MAP"
                return True
            d, pos = support_projection(points[remaining], patch["path"])
            end = arc(patch["path"])[-1]
            inside = (
                (pos > 1e-5) & (pos < end - 1e-5) & (d + patch["sigma"] + margin < self.width / 2)
            )
            covered[remaining[inside]] = True
        if covered.all():
            self.space_source = "OBSERVED_MAP"
            return True
        hint = self.pose_hint
        duration = (
            4 * self.limits.maximum_yaw_rad / self.limits.yaw_rate_rad_s + 3 * self.response_s
        )
        reserved = self.requested_at is not None and self.now - self.requested_at <= duration
        limit = (
            duration + p.maximum_prediction_s + p.missing_command_s
            if reserved
            else p.maximum_prediction_s
        )
        if hint is None or self.now - hint["seen"] > limit:
            return False
        path = hint["path"]
        index = int(np.argmin(np.linalg.norm(path, axis=1)))
        gap = float(np.linalg.norm(path[index]))
        if gap > p.observation_search_near_m:
            return False
        tangent, _ = differential(path)
        normal = np.array([-tangent[index, 1], tangent[index, 0]])
        # This is an explicit local pose inference, never observed free space.
        # Its unobserved-gap error grows quadratically with distance/curvature.
        distance = float(np.linalg.norm(points - path[index], axis=1).max())
        uncertainty = hint["sigma"] + 0.5 * hint["curvature"] * distance * distance
        self.pose_error = uncertainty
        if uncertainty > p.maximum_sigma_m:
            return False
        if np.max(abs((points - path[index]) @ normal)) + uncertainty + margin >= self.width / 2:
            return False
        self.space_source = "LOCAL_POSE_ESTIMATE"
        return True

    def _follow(self, result, follow, *, recovering=False):
        self.reset()
        if result.get('speed_limit_source')=='CENTERLINE_CONFIDENCE':
            if float(result.get('control_confidence',0.))<.8:
                self.status=dict(phase='CAUTIOUS',reason='CENTERLINE CONFIDENCE',
                                 search_pending=True,lease_s=None)
            return follow
        quality = float(result.get("observation_confidence") or 0.0)
        cautious = self.finish_priority and (
            recovering or result.get("state") == "PREDICTED"
            or result.get("transition_pending") or not math.isfinite(quality)
            or result.get("partial_boundary_inference")
            or ((result.get("road_model") or {}).get("persistent_history")
                and not all(result.get("path_observed_mask", [False])))
            or quality < self.policy.confirmation_quality
        )
        if cautious and follow is not None and any(follow):
            scale = min(1.0, self.cruise / max(abs(value) for value in follow))
            follow = tuple(value * scale for value in follow)
            self.status = dict(
                phase="CAUTIOUS", reason="RECOVERED PATH" if recovering else "LOW CONFIDENCE PATH",
                search_pending=True, lease_s=None,
            )
        return follow

    def command(self, result, follow, motion, now, *, active=False, allowed=True, stopped=False):
        self.last_action = False
        self._map_proposal = None
        if not active:
            self.reset()
            return follow
        if (self.finish_priority and result.get('valid')
                and result.get('speed_limit_source')=='CENTERLINE_CONFIDENCE'
                and follow is not None
                and result.get('invalid_reason')!='BODY ENVELOPE BLOCKED'):
            # The accepted path controller owns its constrained motion, whether
            # braking or crawling. A recovery turn cannot replace that envelope.
            return self._follow(result,follow)
        effort_scale=1.
        if self.finish_priority:
            confidence=result.get('control_confidence')
            if confidence is None:confidence=result.get('confidence',0.)
            try:effort_scale=confidence_speed_factor(float(confidence))
            except (TypeError,ValueError):effort_scale=confidence_speed_factor(0.)
        alignment=result.get("route_alignment") or {}
        heading=alignment.get("heading_rad")
        distance=alignment.get("distance_m")
        directional=bool(self.finish_priority and allowed
            and isinstance(heading,(int,float)) and math.isfinite(heading)
            and isinstance(distance,(int,float)) and math.isfinite(distance)
            and 0 <= distance <= self.policy.maximum_target_m)
        if directional and follow is None and abs(heading)>math.radians(20):
            # A retained heading is an observation hint, not a separate motor
            # authority. Use the same bounded pulse/rest scan as every other
            # recovery instead of renewing an unlimited in-place rotation.
            self.last_hint=math.copysign(1.,heading)
        if self.scan.began is not None:
            if (
                self.finish_priority and follow is not None and any(follow)
                and result.get("valid") and (
                    result.get("state") == "PREDICTED"
                    or ((result.get("road_model") or {}).get("persistent_history")
                        and (result.get("target_inferred") or result.get("partial_boundary_inference"))))
                and result.get("invalid_reason") != "BODY ENVELOPE BLOCKED"
            ):
                # Steering has already checked this historical curve's body,
                # motion and remaining lease. Search cannot impose a second
                # fresh-observation streak on a usable prediction. Session
                # still applies task limits and the original prediction lease.
                return self._follow(result, follow, recovering=True)
            command = self.scan.command(
                result, follow, motion, now, allowed=allowed, stopped=stopped, hint=self.last_hint,
                effort_scale=effort_scale,
            )
            if self.scan.began is None:
                return self._follow(result, command, recovering=True)
            return command
        blocked = self.finish_priority and result.get("invalid_reason") == "BODY ENVELOPE BLOCKED"
        if follow is not None and not blocked:
            return self._follow(result, follow)
        if not allowed:
            self.status = dict(
                phase="TASK_WAIT", reason="YIELD TASK OWNS STOP", search_pending=False, lease_s=None
            )
            return None
        if self.finished:
            if self.finish_priority and self.allow_visual_scan:
                return self.scan.start(motion, now, stopped, self.last_hint,effort_scale=effort_scale)
            return None
        if self.allow_visual_scan and result.get("invalid_reason") == "BODY ENVELOPE BLOCKED":
            if self.finish_priority:
                return self.scan.start(motion, now, stopped, self.last_hint,effort_scale=effort_scale)
            self.status = dict(
                phase="WAIT_EVIDENCE",
                reason="BODY ENVELOPE BLOCKED",
                search_pending=False,
                lease_s=None,
            )
            return None
        if (
            self.requested_at is None
            and self.pose_hint is not None
            and now - self.pose_hint["seen"]
            <= self.policy.maximum_prediction_s + self.policy.missing_command_s
        ):
            self.requested_at = now
        valid = bool(motion and motion.get("valid"))
        if not valid:
            if self.allow_visual_scan:
                return self.scan.start(motion, now, stopped, self.last_hint,effort_scale=effort_scale)
            self.status = dict(
                phase="WAIT_EVIDENCE",
                reason="GROUND MOTION UNVERIFIED",
                search_pending=False,
                lease_s=None,
            )
            return None
        if not self.swept_clear(0.0):
            if self.allow_visual_scan:
                return self.scan.start(motion, now, stopped, self.last_hint,effort_scale=effort_scale)
            self.status = dict(
                phase="WAIT_EVIDENCE",
                reason="BODY POSE CLEARANCE UNVERIFIED",
                search_pending=False,
                lease_s=None,
            )
            return None
        speed = abs(float(motion.get("speed_m_s") or 0.0))
        rate = abs(float(motion.get("yaw_rad_s") or 0.0))
        resting = (
            speed < self.policy.sample_step_m / self.response_s
            and rate < self.limits.yaw_rate_rad_s / 4
        )
        if self.began is None and not resting:
            if self.finish_priority and self.allow_visual_scan:
                return self.scan.start(motion, now, stopped, self.last_hint,effort_scale=effort_scale)
            self.status = dict(
                phase="SETTLING", reason="WAIT FOR MEASURED REST", search_pending=True, lease_s=None
            )
            return (0, 0)
        if self.began is None:
            self.began = now
            self.yaw = 0.0
            self.drift = 0.0
            self.leg = 0
            radius = math.hypot(max(self.body[:2]), self.body[2])
            minimum = math.atan2(
                max(self.policy.observation_sigma_m, float(motion["sigma_m"])), radius
            )
            goals = []
            for direction in (self.last_hint, -self.last_hint):
                angle = self.limits.maximum_yaw_rad
                while angle >= minimum:
                    if self.swept_clear(direction * angle):
                        goals.append(direction * angle)
                        break
                    angle /= 2
            if not goals:
                self.began = None
                if self.allow_visual_scan:
                    return self.scan.start(motion, now, stopped, self.last_hint,effort_scale=effort_scale)
                self.status = dict(
                    phase="WAIT_EVIDENCE",
                    reason="NO OBSERVABLE SAFE YAW STEP",
                    search_pending=False,
                    lease_s=None,
                )
                return None
            self.targets = goals + [0.0]
        duration = (
            4 * self.limits.maximum_yaw_rad / self.limits.yaw_rate_rad_s + 3 * self.response_s
        )
        if (
            now - (self.requested_at if self.requested_at is not None else self.began) > duration
            or self.drift > self.policy.sample_step_m
        ):
            self.finished = True
            self.status = dict(
                phase="EXHAUSTED",
                reason="SEARCH BUDGET OR DRIFT LIMIT",
                search_pending=False,
                lease_s=None,
            )
            return None
        while (
            self.leg < len(self.targets)
            and abs(self.targets[self.leg] - self.yaw)
            <= self.limits.yaw_rate_rad_s * self.limits.pulse_s
        ):
            self.leg += 1
        if self.leg >= len(self.targets):
            self.finished = True
            self.status = dict(
                phase="EXHAUSTED",
                reason="SWEEP COMPLETED WITHOUT A CONFIRMED PATH",
                search_pending=False,
                lease_s=None,
            )
            return None
        goal = self.targets[self.leg]
        remaining = goal - self.yaw
        if (
            abs(self.yaw)
            > self.limits.maximum_yaw_rad + self.limits.yaw_rate_rad_s * self.limits.pulse_s
            or not self.swept_clear(remaining)
        ):
            if self.allow_visual_scan:
                return self.scan.start(motion, now, stopped, self.last_hint,effort_scale=effort_scale)
            self.status = dict(
                phase="WAIT_EVIDENCE",
                reason="ROTATION SWEEP NOT INSIDE KNOWN ROAD",
                search_pending=False,
                lease_s=None,
            )
            return None
        speed = abs(float(motion.get("speed_m_s") or 0.0))
        rate = abs(float(motion.get("yaw_rad_s") or 0.0))
        # Each bounded pulse is followed by measured rest before the next one.
        resting = (
            speed < self.policy.sample_step_m / self.response_s
            and rate < self.limits.yaw_rate_rad_s / 4
        )
        self.status = dict(
            phase="SCANNING" if resting else "SETTLING",
            reason="",
            search_pending=True,
            yaw_rad=self.yaw,
            goal_rad=goal,
            leg=self.leg,
            drift_m=self.drift,
            map_patches=len(self.patches),
            space_source=self.space_source,
            pose_error_m=self.pose_error,
            lease_s=None,
        )
        if self.awaiting_rest:
            if resting:
                # The measured-rest frame must also publish zero before another
                # pulse; a slow or unsent zero proposal cannot release the wait.
                self._map_proposal = ("rest_zero", (0, 0))
            return (0, 0)
        if not resting:
            return (0, 0)
        effort = self.cruise / 2 * min(1.0, abs(remaining) / self.limits.maximum_yaw_rad)*effort_scale
        effort = max(self.cruise / 6, effort)
        direction = math.copysign(1.0, remaining)
        self.last_action = True
        self.status["lease_s"] = self.limits.pulse_s
        command = (-round(direction * effort), round(direction * effort))
        # Vision timing, task limits or the control scheduler may suppress this
        # proposal. Only applied() can start the post-pulse rest requirement.
        self._map_proposal = ("pulse", command)
        return command
