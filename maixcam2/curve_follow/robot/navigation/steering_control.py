"""Adaptive physical lookahead, measured latency compensation and body envelope."""

import math
import numpy as np
from robot.core.geometry import arc, support_projection, pursuit_point
from robot.control.controller import target_command


class SteeringControl:
    def __init__(
        self,
        policy,
        origin,
        body,
        road_width,
        lever=0.15,
        gain=1.0,
        cruise=60.0,
        deceleration=0.08,
        speed_per_pwm=0.0067,
        calibration=None,
        response_s=0.4,
        straight_cruise=None,
        confirmed_cruise=None,
    ):
        self.policy = policy
        self.origin = np.asarray(origin, float)
        self.body = body
        self.width = road_width
        self.lever = lever
        self.gain = gain
        self.cruise = cruise
        self.deceleration = deceleration
        self.speed_per_pwm = speed_per_pwm
        if self.origin.shape != (2,) or not np.isfinite(self.origin).all():
            raise ValueError("Invalid control origin")
        if body is not None and (
            np.asarray(body).shape != (3,) or not np.isfinite(body).all() or min(body) <= 0
        ):
            raise ValueError("Invalid body envelope")
        if not all(
            math.isfinite(v) and v > 0
            for v in (road_width, lever, gain, cruise, deceleration, speed_per_pwm)
        ):
            raise ValueError("Invalid steering configuration")
        if not math.isfinite(response_s) or response_s <= 0:
            raise ValueError("Invalid actuator response time")
        self.straight_cruise = cruise if straight_cruise is None else float(straight_cruise)
        if not math.isfinite(self.straight_cruise) or self.straight_cruise < cruise:
            raise ValueError("Straight cruise must be finite and at least baseline cruise")
        self.confirmed_cruise = cruise if confirmed_cruise is None else float(confirmed_cruise)
        if (
            not math.isfinite(self.confirmed_cruise)
            or not cruise <= self.confirmed_cruise <= self.straight_cruise
        ):
            raise ValueError("Confirmed cruise must lie between baseline and straight cruise")
        self.response_s = response_s
        self.minimum_lookahead_m = sum(body[:2]) if body is not None else road_width
        self.inverse = (
            None
            if calibration is None
            else np.linalg.inv(np.asarray(calibration["H_normalized"], float))
        )
        self.roi_top = 0 if calibration is None else int(calibration.get("vision_roi_top_px", 0))
        self.reset()

    def reset(self):
        self.trim = 0.0
        self.commands = []
        self.requested = 0.0
        self.status = {}
        self.preview_speed = 0.0
        self.preview_interval = 0.0
        self.speed_ready_since = None
        self.observed_at = None
        self.filtered_lookahead = None
        self.lookahead_at = None

    def applied(self, now, wheels):
        moving = max(abs(v) for v in wheels) > 0
        saturated = min(wheels) <= 0 and moving
        self.commands.append((now, self.requested, moving, saturated, tuple(wheels)))
        self.commands = self.commands[-128:]

    def observe(self, motion, now, allow_feedback=True):
        self.observed_at = now
        self.status = dict(trim_m_inv=self.trim, feedback_valid=False)
        if not motion.get("valid"):
            return
        dt = motion["interval_s"]
        if dt <= 0:
            return
        rotation = np.asarray(motion["rotation"])
        translation = self.origin @ rotation.T + motion["translation"] - self.origin
        displacement = -translation @ rotation
        forward = float(displacement[0])
        sigma = motion["sigma_m"]
        speed = forward / dt
        yaw = -math.atan2(rotation[1, 0], rotation[0, 0]) / dt
        self.status.update(
            body_speed_m_s=speed,
            body_yaw_rad_s=yaw,
            speed_upper_m_s=max(0.0, speed) + 2 * sigma / dt,
            motion_sigma_m=sigma,
            motion_interval_s=dt,
        )
        # Hold only a conservative horizon-speed bound through a flow dropout.
        # It never transports road geometry or masquerades as measured twist.
        self.preview_speed = max(0.0, speed) + 2 * sigma / dt
        self.preview_interval = dt
        if not allow_feedback or not self.commands or forward <= 2 * sigma:
            return
        # Do not identify a drivetrain bias during startup, a steering transient
        # or a regulator changing speed. One PWM step is command quantization.
        settled_begin = now - max(dt, self.response_s)
        settled_prior = [row for row in self.commands if row[0] <= settled_begin]
        if not settled_prior:
            return
        settled = [settled_prior[-1]] + [
            row for row in self.commands if settled_begin < row[0] < now
        ]
        wheels = np.array([row[4] for row in settled])
        if np.any(np.ptp(wheels, axis=0) > 1):
            return
        begin = now - dt
        history = self.commands
        prior = [row for row in history if row[0] <= begin]
        if not prior:
            return
        segments = (
            [(begin, *prior[-1][1:])]
            + [row for row in history if begin < row[0] < now]
            + [(now, 0.0, False, False, (0, 0))]
        )
        requested = 0.0
        for a, b in zip(segments, segments[1:]):
            if not a[2] or a[3]:
                return  # Do not learn during coast, dwell or saturation.
            requested += a[1] * (b[0] - a[0]) / dt
        speed = forward / dt
        yaw = -math.atan2(rotation[1, 0], rotation[0, 0]) / dt
        error = requested - yaw / speed
        quality = max(0.0, 1.0 - 2 * sigma / forward)
        # Integrate over travelled distance / preview length, not a per-frame
        # gain. Setpoint history aligns the observed interval with past commands.
        step = quality * forward / self.width
        self.trim = float(np.clip(self.trim + step * error, -1 / self.lever, 1 / self.lever))
        self.status.update(
            trim_m_inv=self.trim,
            feedback_valid=True,
            desired_curvature_m_inv=requested,
            measured_curvature_m_inv=yaw / speed,
            body_speed_m_s=speed,
            body_yaw_rad_s=yaw,
            quality=quality,
        )

    def smooth_lookahead(self, desired):
        """Match preview-length changes to actuator bandwidth, once per image.

        This filters only the selection radius, never fitted geometry or wheel
        direction. Every goal is still selected on this frame's supported path.
        Explicit alternate-goal probes do not advance this temporal state.
        """
        now = self.observed_at
        if now is None:
            return desired
        if self.filtered_lookahead is None:
            self.filtered_lookahead = desired
            self.lookahead_at = now
        elif now > self.lookahead_at:
            weight = -math.expm1(-(now - self.lookahead_at) / self.response_s)
            self.filtered_lookahead += weight * (desired - self.filtered_lookahead)
            self.lookahead_at = now
        return self.filtered_lookahead

    def command(self, result, age_s=0.0, *, lookahead_m=None, alternatives=True):
        """Follow geometry at command time, not at frame-receive time.

        Extrapolate ONLY valid measured body twist over the measured processing
        delay. This is a short SE(2) prediction, not a fresh road observation or
        a model of camera buffering / motor response. Invalid flow is reported
        and leaves geometry unchanged. The ordinary freshness gate still owns
        whether this command can be sent.
        """
        point = result.get("control_target_m", result.get("target_m"))
        self.status.pop("envelope", None)
        if (
            not result.get("valid")
            or result.get("state") != "OBSERVED"
            or not result.get("observed", False)
        ):
            self.speed_ready_since = None
        if (
            point is None
            or not result.get("valid")
            or result.get("state") not in ("OBSERVED", "PREDICTED")
        ):
            return None
        if len(point) != 2 or not all(math.isfinite(v) for v in point) or point[0] <= 0:
            return None
        if not math.isfinite(age_s) or age_s < 0:
            return None
        if age_s > self.policy.missing_command_s:
            return None
        receive_point = np.asarray(point, float)
        self.status["receive_curvature_m_inv"] = (
            self.gain * 2 * receive_point[1] / float(receive_point @ receive_point)
        )
        path = np.asarray(result.get("control_path_m", []), float)
        if path.ndim != 2 or path.shape[1:] != (2,) or len(path) < 2 or not np.isfinite(path).all():
            return None
        transform = None
        prediction_sigma = 0.0
        self.status["latency_prediction"] = dict(
            age_s=age_s, applied=False, reason="NO_VALID_MEASURED_MOTION"
        )
        if "body_speed_m_s" in self.status and age_s > 0:
            v = self.status["body_speed_m_s"]
            w = self.status["body_yaw_rad_s"]
            theta = w * age_s
            c, s = math.cos(theta), math.sin(theta)
            rotation = np.array([[c, s], [-s, c]])
            delta = (
                np.array([v * age_s, 0.0])
                if abs(w) < 1e-8
                else np.array([v * math.sin(theta) / w, v * (1 - c) / w])
            )
            translation = -delta @ rotation.T
            transform = (rotation, translation)
            point = np.asarray(point) @ rotation.T + translation
            path = path @ rotation.T + translation
            prediction_sigma = (
                self.status["motion_sigma_m"] * age_s / self.status["motion_interval_s"]
            )
            self.status["latency_prediction"] = dict(
                age_s=age_s,
                applied=True,
                displacement_m=delta.tolist(),
                yaw_rad=theta,
                sigma_m=prediction_sigma,
            )
        # Pure pursuit's correction bandwidth scales as speed / lookahead.
        # Two response horizons cover motor lag, current processing and a whole
        # visual sampling interval. The measured body length supplies a floor;
        # a nearby support sample must never collapse this horizon to 6 cm.
        horizon = 2 * (self.response_s + age_s + self.preview_interval)
        lookahead = min(
            self.policy.maximum_target_m,
            max(self.minimum_lookahead_m, self.preview_speed * horizon),
        )
        raw_lookahead = lookahead
        lookahead = self.smooth_lookahead(lookahead) if lookahead_m is None else float(lookahead_m)
        self.status.update(raw_lookahead_m=raw_lookahead, lookahead_filter_s=self.response_s)
        selected = pursuit_point(
            path, lookahead, self.policy.forward_min_m, self.policy.maximum_target_m
        )
        if selected is None:
            return None
        point = selected["point"]
        receive_target = point if transform is None else (point - transform[1]) @ transform[0]
        self.status.update(
            target_selection="ADAPTIVE_PURSUIT",
            lookahead_m=lookahead,
            lookahead_time_s=horizon,
            horizon_speed_m_s=self.preview_speed,
            target_mode=selected["mode"],
            target_distance_m=selected["distance_m"],
            command_target_m=point.tolist(),
            receive_target_m=receive_target.tolist(),
        )
        positions = arc(path)
        index = min(len(path) - 1, int(np.searchsorted(positions, selected["position_m"])))
        mask = result.get("path_observed_mask", [])
        self.status["target_observed"] = bool(
            len(mask) == len(path) and all(mask[max(0, index - 1) : index + 1])
        )
        self.status["latency_prediction"]["point_m"] = point.tolist()
        raw = dict(result, target_m=point)
        x, y = point
        self.requested = self.gain * 2 * y / (x * x + y * y)
        self.status.update(
            requested_curvature_m_inv=self.requested,
            corrected_curvature_m_inv=self.requested + self.trim,
        )
        raw["steering_trim_m_inv"] = self.trim
        command = target_command(raw, self.lever, self.gain, self.cruise)
        if command is None:
            return None
        observed = np.asarray(result.get("path", []), float)
        if observed.size:
            observed = observed - self.origin
        if transform is not None:
            rotation, translation = transform
            if observed.size:
                observed = observed @ rotation.T + translation
        target_scale = float(selected["speed_scale"])
        if not math.isfinite(target_scale):
            return None
        scale = max(0.0, min(1.0, target_scale))
        support_sigma = float(
            (result.get("road_model") or {}).get("sigma_m") or self.policy.observation_sigma_m
        )
        envelope = self.envelope(
            path,
            self.requested + self.trim,
            support_sigma + prediction_sigma,
            self.width,
            observed_path=observed,
            speed_m_s=self.status.get("speed_upper_m_s"),
            sensor_transform=transform,
            lookahead_m=lookahead,
        )
        scale = min(scale, envelope["speed_scale"])
        if result.get("history_continuity_evidence") is not None:
            candidate = np.asarray(result.get("candidate_path_m", []), float)
            if candidate.ndim != 2 or candidate.shape[1:] != (2,) or len(candidate) < 5:
                return None
            candidate = candidate - self.origin
            if transform is not None:
                candidate = candidate @ transform[0].T + transform[1]
            shared = self.envelope(
                path,
                self.requested + self.trim,
                support_sigma + prediction_sigma,
                self.width,
                observed_path=candidate,
                speed_m_s=self.status.get("speed_upper_m_s"),
                sensor_transform=transform,
                lookahead_m=lookahead,
            )
            scale = min(scale, shared["speed_scale"])
            self.status["continuation_envelope"] = shared
        if result.get("transition_pending"):
            # Use the SAME supported historical curve and quality-based drive
            # law as an ordinary occlusion. Bound its motor command lifetime,
            # rather than treating the watchdog timeout as braking delay or an
            # old PWM/speed proxy as a new speed measurement. Those conversions
            # repeatedly reduced useful commands below the motor dead zone.
            speed = self.status.get("speed_upper_m_s")
            model = result.get("road_model") or {}
            room = (
                min(
                    self.policy.maximum_blind_distance_m
                    - float(model.get("blind_distance_m", 0.0)),
                    float(model.get("length_m", 0.0)),
                )
                - support_sigma
                - prediction_sigma
            )
            remaining = self.policy.maximum_prediction_s - float(model.get("age_s", 0.0)) - age_s
            lease = (
                0.0
                if speed is None
                or speed <= 0
                or self.status.get("body_speed_m_s", 0.0) <= 0
                or room <= 0
                or remaining <= 0
                else min(0.15, remaining, room / speed)
            )
            self.status["handoff"] = dict(
                remaining_distance_m=room,
                remaining_time_s=remaining,
                lease_s=lease,
                speed_scale=1.0 if lease > 0 else 0.0,
            )
            if lease <= 0:
                return None
        # Extra effort is allowed only on a currently measured, supported
        # pursuit segment with measured motion and a clear body envelope.
        # Reuse the ordinary dimensionless bend load: the straight ceiling
        # decays to the confirmed-road cruise before a sharp turn. The ordinary
        # target law still reduces speed with curvature; uncertain observations
        # retain baseline cruise and never inherit the confirmed-road increase.
        # Historical tails keep their decaying quality for prediction and
        # envelope checks. They must not label a freshly observed pursuit
        # segment weak. Unconfirmed candidates never populate this field.
        observation_quality = float(
            result.get("observation_confidence", result.get("confidence", 0.0))
        )
        road_eligible = (
            result.get("state") == "OBSERVED"
            and result.get("observed", False)
            and observation_quality >= self.policy.confirmation_quality
            and not result.get("transition_pending")
            and self.status["target_observed"]
            and selected["mode"] == "LOOKAHEAD"
            and envelope["speed_scale"] == 1.0
            and envelope.get("minimum_clearance_m") is not None
            and envelope["minimum_clearance_m"] > 0.0
            and self.observed_at is not None
        )
        # Use measured time, not a frame counter. A single accepted frame must
        # not immediately restore extra speed after a weak/occluded segment.
        if not road_eligible:
            self.speed_ready_since = None
        elif self.speed_ready_since is None:
            self.speed_ready_since = self.observed_at
        stable_s = (
            0.0
            if self.speed_ready_since is None
            else max(0.0, self.observed_at - self.speed_ready_since)
        )
        # Motion remains mandatory for this command; an isolated flow rejection
        # cannot erase otherwise continuous road observation history.
        boost_ready = (
            road_eligible and "body_speed_m_s" in self.status and stable_s >= self.response_s
        )
        bend = max(
            abs(self.requested + self.trim),
            self.gain * abs(float(result.get("path_curvature_m_inv", 0.0))),
        )
        load = min(1.0, self.lever * bend)
        straight_candidate = (
            max(self.cruise, self.straight_cruise / (1.0 + load)) if boost_ready else self.cruise
        )
        cruise = max(self.confirmed_cruise, straight_candidate) if boost_ready else self.cruise
        if cruise > self.cruise:
            # Preserve the existing quality source. Raising bend cruise must
            # not also replace decayed history quality with fresh quality,
            # which would multiply effort beyond the configured cruise step.
            quality = (
                observation_quality
                if straight_candidate > self.cruise
                else raw.get("confidence", 0.0)
            )
            command = target_command(dict(raw, confidence=quality), self.lever, self.gain, cruise)
        self.status["speed_plan"] = dict(
            baseline_cruise_pwm=self.cruise,
            confirmed_cruise_pwm=self.confirmed_cruise,
            straight_cruise_pwm=self.straight_cruise,
            selected_cruise_pwm=cruise,
            bend_load=load,
            boost_eligible=boost_ready,
            road_confirmed_s=stable_s,
            required_road_confirm_s=self.response_s,
            observation_confidence=observation_quality,
            road_eligible=road_eligible,
        )
        self.status.update(
            envelope=envelope, target_selection="ADAPTIVE_PURSUIT", target_speed_scale=target_scale
        )
        output = tuple(round(v * scale) for v in command) if scale > 0 else None
        if output is not None and any(output):
            return output
        if alternatives:
            # Try other goals on the same supported curve before asking the
            # visual searcher to rotate. No extrapolated path is introduced.
            count = max(2, math.ceil(self.policy.maximum_target_m / self.minimum_lookahead_m))
            radii = np.linspace(self.policy.forward_min_m, self.policy.maximum_target_m, count)
            for radius in sorted(radii, key=lambda r: abs(r - lookahead)):
                if abs(radius - lookahead) < self.policy.sample_step_m:
                    continue
                alternate = self.command(result, age_s, lookahead_m=radius, alternatives=False)
                if alternate is not None and any(alternate):
                    self.status["target_selection"] = "FEASIBLE_PURSUIT"
                    return alternate
        return None

    def envelope(
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
        poses = []
        distances_ahead = np.linspace(0.0, horizon, 7)
        # Future steering is sequential; corner projection and sensor coverage
        # are independent once those poses are known, so process all 28 corners
        # in one batch rather than repeating tiny NumPy/C calls seven times.
        for i, distance in enumerate(distances_ahead):
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
            poses.append(pose.copy())
            c, s = math.cos(pose[2]), math.sin(pose[2])
            rotation = np.array([[c, -s], [s, c]])
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
        poses = np.asarray(poses)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        rotations = np.stack((c, -s, s, c), axis=1).reshape(-1, 2, 2)
        footprint = (
            corners[None, :, :] @ rotations.transpose(0, 2, 1) + poses[:, None, :2]
        ).reshape(-1, 2)
        distances, positions = support_projection(footprint, support)
        known = (positions > 1e-5) & (positions < support_arc[-1] - 1e-5)
        if self.inverse is not None:
            camera_footprint = footprint
            if sensor_transform is not None:
                rotation_at_command, translation_at_command = sensor_transform
                camera_footprint = (footprint - translation_at_command) @ rotation_at_command
            ground = camera_footprint + self.origin
            q = np.column_stack((ground, np.ones(len(ground)))) @ self.inverse.T
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
        clearance = self.width / 2 - distances - sigma
        checked = int(known.sum())
        total = len(footprint)
        minimum = float(clearance[known].min()) if checked else float("inf")
        unsafe = np.flatnonzero((known & (clearance < 0)).reshape(-1, 4).any(axis=1))
        first_unsafe = float(distances_ahead[unsafe[0]]) if len(unsafe) else None
        coverage = checked / max(1, total)
        # The camera cannot see under the chassis. Missing coverage is reported,
        # not fabricated from an extrapolated prefix or treated as a collision.
        scale = 1.0
        if first_unsafe is not None:
            a = self.deceleration
            t = self.policy.missing_command_s
            safe_speed = max(0.0, math.sqrt((a * t) ** 2 + 2 * a * first_unsafe) - a * t)
            estimated = max(
                1e-8, self.speed_per_pwm * self.cruise if speed_m_s is None else speed_m_s
            )
            scale = min(scale, safe_speed / estimated)
        return dict(
            speed_scale=scale,
            coverage=coverage,
            state="FULL" if checked == total else "PARTIAL",
            minimum_clearance_m=None if not math.isfinite(minimum) else minimum,
            first_unsafe_m=first_unsafe,
        )
