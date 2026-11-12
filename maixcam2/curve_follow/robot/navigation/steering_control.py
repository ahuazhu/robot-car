"""Adaptive physical lookahead, measured latency compensation and body envelope."""

import math
import numpy as np
from robot.core.geometry import arc, support_projection, pursuit_point, nearest_forward_point
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
        speed_control=None,
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
        self.speed_control = speed_control
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
        if self.speed_control is not None:
            self.speed_control.reset()
        self.trim = 0.0
        self.commands = []
        self.requested = 0.0
        self.status = {}
        self.preview_speed = 0.0
        self.preview_interval = 0.0
        self.speed_ready_since = None
        self.speed_observed_at = None
        self.observed_at = None
        self.filtered_lookahead = None
        self.lookahead_at = None
        self._motion_prediction_valid = False
        self._motion_prediction_age_s = None

    def applied(self, now, wheels, *, delivered_wheels=None):
        if self.speed_control is not None:
            self.speed_control.applied(now, wheels, delivered_wheels=delivered_wheels)
        # Feedback describes the command actually sent after boost/quantization.
        # The speed regulator separately retains its logical effort units.
        if delivered_wheels is not None:
            wheels = delivered_wheels
        moving = max(abs(v) for v in wheels) > 0
        saturated = min(wheels) <= 0 and moving
        expected=self.requested
        plan=self.status.get('speed_profile_control') or {}
        if (plan.get('controller')=='DWPP' and plan.get('selected')
                and plan.get('effective_curvature_m_inv') is not None):
            # Deliberate dynamic-window turn limiting is not drivetrain bias.
            expected=plan.get('applied_curvature_m_inv',plan['effective_curvature_m_inv'])-self.trim
        self.commands.append((now, expected, moving, saturated, tuple(wheels)))
        self.commands = self.commands[-128:]

    def observe(self, motion, now, allow_feedback=True):
        self.observed_at = now
        self._motion_prediction_valid = bool(motion.get('prediction_valid'))
        self._motion_prediction_age_s = motion.get('prediction_age_s')
        measured = bool(motion.get("valid"))
        self.status = dict(trim_m_inv=self.trim, feedback_valid=False, motion_measured=measured)
        if not measured and not motion.get("prediction_valid"):
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
            motion_source=motion.get("source"),
        )
        # Hold only a conservative horizon-speed bound through a flow dropout.
        # It never transports road geometry or masquerades as measured twist.
        self.preview_speed = max(0.0, speed) + 2 * sigma / dt
        self.preview_interval = dt
        if not measured or not allow_feedback or not self.commands or forward <= 2 * sigma:
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

    def _confirm_observed_road(self, result, age_s):
        """Observation continuity and this command's feasibility are separate.

        A short/far pursuit goal or a temporarily missing envelope sample cannot
        erase an otherwise accepted road observation. Those conditions still
        forbid boost on that command. Lost/weak/conflicting/stale observations,
        a new adopted road and long image gaps start a new confirmation window.
        """
        now = self.observed_at
        quality = float(result.get("observation_confidence", result.get("confidence", 0.0)))
        accepted = bool(
            result.get("valid") and result.get("state") == "OBSERVED"
            and result.get("observed", False) and not result.get("transition_pending")
            and quality >= self.policy.confirmation_quality and now is not None
            and math.isfinite(age_s) and 0 <= age_s <= self.policy.missing_command_s
        )
        if not accepted:
            self.speed_ready_since = self.speed_observed_at = None
            return 0.0
        if self.speed_observed_at != now:
            if (self.speed_observed_at is None or now < self.speed_observed_at
                    or now-self.speed_observed_at > self.policy.missing_command_s
                    or result.get("reacquired")):
                self.speed_ready_since = now
            self.speed_observed_at = now
        return max(0.0, now-self.speed_ready_since)

    def _forward_span(self, path, selected, positions):
        anchor = nearest_forward_point(path, self.policy.forward_min_m)
        if anchor is None:
            return None
        front = self.body[0] if self.body is not None else self.width
        end = max(selected["position_m"] + front, anchor[1] + self.width + front)
        first = max(0, int(np.searchsorted(positions, anchor[1])) - 1)
        last = min(len(path), int(np.searchsorted(positions, end, side="right")) + 1)
        return slice(first, last)

    def _forward_uncertainty(self, result, path, selected, positions):
        """Use uncertainty where the controller will drive, retaining old tails.

        A pointwise profile stays attached to the same transported road samples.
        Without that evidence, preserve the conservative scalar fallback.
        """
        fallback = float((result.get("road_model") or {}).get("sigma_m")
                         or self.policy.observation_sigma_m)
        try:
            profile = np.asarray(result.get("path_sigma_m", []), float)
        except (ValueError, TypeError):
            return fallback, None
        span = self._forward_span(path, selected, positions)
        if (profile.shape != (len(path),) or not np.isfinite(profile).all()
                or np.any(profile < 0) or span is None):
            return fallback, None
        return float(profile[span].max()), profile

    def _forward_quality(self, result, path, selected, positions):
        """Keep low quality inside the driven prefix; ignore unrelated tails.

        The selected pursuit point and the body rollout horizon both contribute
        to the prefix extent. No per-point evidence means the original global
        quality. Geometry expiry, uncertainty and prediction gates are unchanged.
        """
        original = float(result.get("confidence", 0.0))
        try:
            profile = np.asarray(result.get("path_confidence", []), float)
        except (ValueError, TypeError):
            return original, "GLOBAL"
        if (profile.shape != (len(path),) or not np.isfinite(profile).all()
                or not (0 < original <= 1) or np.any(profile <= 0) or np.any(profile > 1)
                or abs(float(profile.min())-original) > 1e-6):
            return original, "GLOBAL"
        span = self._forward_span(path, selected, positions)
        if span is None:
            return original, "GLOBAL"
        quality = float(profile[span].min())
        return quality, "FORWARD_PROFILE"

    def _history_output(self, command, age_s, sigma, length, envelope):
        lease=min(self.policy.maximum_prediction_command_s,
                  max(0.,self.policy.missing_command_s-age_s))
        self.status['prediction']=dict(persistent_history=True,remaining_time_s=None,
            remaining_distance_m=float(length),lease_s=lease,uncertainty_m=sigma,
            envelope_unverified=envelope['speed_scale']<1.,
            measured_motion=bool(self.status.get('motion_measured',False)),
            motion_source=self.status.get('motion_source'))
        # A speed-controller brake is still a valid road-following decision.
        # Discarding its zero would falsely authorize a visual search turn.
        return command if not any(command) or lease>0 else None

    def command(self, result, age_s=0.0, *, lookahead_m=None, alternatives=True,
                speed_profile_allowed=True,motion_gap_hold=False):
        """Follow geometry at command time, not at frame-receive time.

        Extrapolate ONLY valid measured body twist over the measured processing
        delay. This is a short SE(2) prediction, not a fresh road observation or
        a model of camera buffering / motor response. Invalid flow is reported
        and leaves geometry unchanged. The ordinary freshness gate still owns
        whether this command can be sent.
        """
        point = result.get("control_target_m", result.get("target_m"))
        self.status.pop('motion_gap_hold',None)
        if self.speed_control is not None and not motion_gap_hold:
            self.speed_control.cancel_motion_gap_hold()
        if motion_gap_hold:
            self.status['motion_gap_hold']=dict(active=False,
                reason='MEASURED_MOTION' if self.status.get('motion_measured') else 'NO_ACCEPTED_PURSUIT',
                prediction_age_s=self._motion_prediction_age_s,
                maximum_age_s=self.policy.maximum_prediction_command_s)
        self.status.pop("envelope", None)
        self.status.pop("prediction", None)
        self.status.pop("handoff", None)
        self.status.pop("confidence_speed_limit_m_s", None)
        if self.speed_control is not None:
            self.status['speed_profile_control'] = dict(
                mode='ACTIVE', selected=False, reason='NO_ACCEPTED_PURSUIT'
            )
        stable_s = self._confirm_observed_road(result, age_s)
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
            if motion_gap_hold:
                self.status['motion_gap_hold']['reason']='COMMAND_EXPIRED'
                if self.speed_control is not None:self.speed_control.cancel_motion_gap_hold()
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
        persistent_recovery = bool(
            (result.get("road_model") or {}).get("persistent_history")
            and (not self.status["target_observed"] or result.get("state") == "PREDICTED"
                 or result.get("transition_pending") or result.get("partial_boundary_inference"))
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
        support_sigma, sigma_profile = self._forward_uncertainty(result, path, selected, positions)
        footprint_sigma = (sigma_profile + prediction_sigma
                           if sigma_profile is not None and observed.shape == path.shape else None)
        self.status["support_uncertainty"] = dict(
            scope="FORWARD_PROFILE" if sigma_profile is not None else "GLOBAL",
            sigma_m=support_sigma, prediction_sigma_m=prediction_sigma,
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
            support_sigma_m=footprint_sigma,
        )
        scale = min(scale, envelope["speed_scale"])
        clearance_limit = envelope.get('speed_limit_m_s')
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
            shared_limit=shared.get('speed_limit_m_s')
            if shared_limit is not None:
                clearance_limit=shared_limit if clearance_limit is None else min(clearance_limit,shared_limit)
            scale = min(scale, shared["speed_scale"])
            self.status["continuation_envelope"] = shared
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
        observed_pursuit = (
            result.get("state") == "OBSERVED"
            and result.get("observed", False)
            and observation_quality >= self.policy.confirmation_quality
            and not result.get("transition_pending")
            and self.status["target_observed"]
            and selected["mode"] == "LOOKAHEAD"
            and self.observed_at is not None
        )
        road_eligible = (
            observed_pursuit
            and envelope["speed_scale"] == 1.0
            and envelope.get("minimum_clearance_m") is not None
            and envelope["minimum_clearance_m"] > 0.0
        )
        # The elapsed road-observation proof is maintained independently of
        # pursuit/envelope feasibility; every boost still needs road_eligible.
        # Motion remains mandatory for this command; an isolated flow rejection
        # cannot erase otherwise continuous road observation history.
        boost_ready = (
            road_eligible and "body_speed_m_s" in self.status and stable_s >= self.response_s
            and self.status.get("motion_measured", True)
            and not persistent_recovery
        )
        # Extra turning is distinct from permission to accelerate on a clear
        # baseline arc. Its calibrated arc receives its own body check below.
        bend_ready = (
            observed_pursuit and "body_speed_m_s" in self.status
            and stable_s >= self.response_s and self.status.get("motion_measured",False)
            and not persistent_recovery
        )
        self.status['bend_eligible']=bend_ready
        if self.speed_control is not None:
            control_confidence,_ = self._forward_quality(result,path,selected,positions)
            resolved_forward_motion = (
                self.status.get('body_speed_m_s', 0.)
                * self.status.get('motion_interval_s', 0.)
                > 2 * self.status.get('motion_sigma_m', float('inf'))
            )
            measured_standstill=bool(self.status.get('motion_measured')
                and abs(self.status.get('body_speed_m_s',1.))<=.006
                and abs(self.status.get('body_yaw_rad_s',1.))<=.025)
            planning_target = dict(result, control_confidence=control_confidence)
            if sigma_profile is not None:
                planning_target["reference_sigma_m"] = support_sigma
            eligible=(speed_profile_allowed and target_scale == 1.
                      and self.status.get('motion_measured',False)
                      and (resolved_forward_motion or measured_standstill))
            planned=None
            holding=False
            if motion_gap_hold and not self.status.get('motion_measured',False):
                prediction_age=self._motion_prediction_age_s
                gap_reason=('TASK_OWNS_SPEED' if not speed_profile_allowed else
                    'CURRENT_ROAD_NOT_CONFIRMED' if not observed_pursuit or persistent_recovery else
                    'PURSUIT_SPEED_REDUCED' if target_scale!=1. else
                    'NO_VALID_MOTION_PREDICTION' if not self._motion_prediction_valid else
                    'MOTION_PREDICTION_EXPIRED' if not isinstance(prediction_age,(int,float))
                        or not math.isfinite(prediction_age) or prediction_age<0
                        or prediction_age>=self.policy.maximum_prediction_command_s else None)
                if gap_reason is None:
                    planned,details=self.speed_control.hold_motion_gap(
                        planning_target,path,self.observed_at+age_s,self.requested+self.trim,
                        support_sigma+prediction_sigma,self.body[0] if self.body else self.width,
                        self.policy.maximum_prediction_command_s,captured=self.observed_at)
                    self.status['motion_gap_hold'].update(details)
                    holding=planned is not None
                else:
                    self.status['motion_gap_hold']['reason']=gap_reason
            if not holding:
                planned = self.speed_control.command(
                    planning_target, path, self.observed_at + age_s,
                    self.status.get('body_speed_m_s'), self.requested + self.trim,
                    support_sigma + prediction_sigma, self.body[0] if self.body else self.width,
                    eligible=eligible,
                    high_confidence=(boost_ready and observation_quality >= .75
                        and abs(self.lever * (self.requested + self.trim)) <= .18),
                    clearance_limit_m_s=clearance_limit,
                    measured_yaw_rate=self.status.get('body_yaw_rad_s'),
                    bend_allowed=bend_ready,
                    hold_source_at=(self.observed_at if motion_gap_hold and eligible
                        and observed_pursuit and not persistent_recovery and resolved_forward_motion
                        and self.status.get('body_speed_m_s',0.)>=.10 else None),
                )
            if not speed_profile_allowed:
                self.speed_control.status['reason'] = 'TASK_OWNS_SPEED'
            elif not holding and not resolved_forward_motion and not measured_standstill:
                self.speed_control.status['reason'] = 'MEASURED_FORWARD_SPEED_UNRESOLVED'
            if motion_gap_hold:
                self.speed_control.status['motion_gap_hold']=dict(self.status['motion_gap_hold'])
            self.status['speed_profile_control'] = self.speed_control.status
            self.status['confidence_speed_limit_m_s']=self.speed_control.status.get('confidence_ceiling_m_s')
            self.status.pop('dwpp_envelope',None)
            if planned is not None:
                plan=self.speed_control.status
                if (plan.get('controller')=='DWPP' and any(planned)
                        and (holding or plan.get('bend_allocation',{}).get('active')
                             or abs(plan['effective_curvature_m_inv']-(self.requested+self.trim))>1e-9)):
                    # Braking may withdraw outside-wheel compensation and
                    # restore the baseline arc. Check that arc once as well;
                    # the baseline's common braking scale preserves it.
                    for _ in range(2):
                        checked_curvature=plan['effective_curvature_m_inv']
                        actual=self.envelope(path,checked_curvature,
                            support_sigma+prediction_sigma,self.width,observed_path=observed,
                            speed_m_s=self.status.get('speed_upper_m_s'),sensor_transform=transform,
                            lookahead_m=lookahead,support_sigma_m=footprint_sigma)
                        self.status['dwpp_envelope']=actual
                        checked_envelopes=[actual]
                        limits=[actual.get('speed_limit_m_s')]
                        if result.get('history_continuity_evidence') is not None:
                            actual_candidate=self.envelope(path,checked_curvature,
                                support_sigma+prediction_sigma,self.width,observed_path=candidate,
                                speed_m_s=self.status.get('speed_upper_m_s'),sensor_transform=transform,
                                lookahead_m=lookahead)
                            limits.append(actual_candidate.get('speed_limit_m_s'))
                            checked_envelopes.append(actual_candidate)
                        if holding:
                            required_speed=max(plan['target_speed_m_s'],
                                (self.speed_control.bend_proposal or {}).get('estimated_speed_m_s',0.))
                            unknown=any(e.get('minimum_clearance_m') is None
                                or not math.isfinite(e['minimum_clearance_m']) for e in checked_envelopes)
                            tighter=any(v is not None and v<required_speed-1e-9 for v in limits)
                            if unknown or tighter:
                                reason=('CURRENT_CLEARANCE_UNOBSERVED' if unknown
                                        else 'CURRENT_CLEARANCE_LIMIT')
                                self.status['motion_gap_hold'].update(active=False,reason=reason)
                                # Refresh the original fallback's current
                                # confidence/body cap; never retain an old cap.
                                self.speed_control.command(planning_target,path,
                                    self.observed_at+age_s,self.status.get('body_speed_m_s'),
                                    self.requested+self.trim,support_sigma+prediction_sigma,
                                    self.body[0] if self.body else self.width,eligible=False,
                                    clearance_limit_m_s=clearance_limit)
                                self.speed_control.status['reason']='MOTION_GAP_'+reason
                                self.speed_control.status['motion_gap_hold']=dict(self.status['motion_gap_hold'])
                                self.status['speed_profile_control']=self.speed_control.status
                                planned=None
                                break
                        if (plan.get('bend_allocation',{}).get('active') and any(
                                e.get('minimum_clearance_m') is None
                                or not math.isfinite(e['minimum_clearance_m'])
                                for e in checked_envelopes)):
                            planned=self.speed_control.withdraw_bend(
                                planned,'BODY_CLEARANCE_UNOBSERVED')
                            continue
                        known=[v for v in limits if v is not None]
                        if known:
                            cap=min(known)
                            if (result.get('road_model') or {}).get('persistent_history'):
                                cap=max(cap,self.speed_control.limits.maximum_m_s*.1)
                            planned=self.speed_control.constrain_clearance(planned,cap)
                        if (not any(planned) or
                                abs(plan['effective_curvature_m_inv']-checked_curvature)<=1e-9):
                            break
                if planned is not None:
                    self.status.update(envelope=envelope, target_selection='ADAPTIVE_PURSUIT',
                                       target_speed_scale=target_scale,
                                       speed_plan=dict(mode='DWPP' if plan.get('controller')=='DWPP'
                                                       else 'PROFILE', boost_eligible=boost_ready))
                    return (self._history_output(planned,age_s,support_sigma+prediction_sigma,
                                                 positions[-1],envelope)
                            if persistent_recovery else planned)
        bend = max(
            abs(self.requested + self.trim),
            self.gain * abs(float(result.get("path_curvature_m_inv", 0.0))),
        )
        load = min(1.0, self.lever * bend)
        control_quality = float(raw.get("confidence", 0.0))
        quality_source = "GLOBAL"
        # Reusing local quality raises only normal baseline effort on an
        # already feasible gentle segment. It never enables the extra cruise
        # step, changes sharp-turn effort, or grants speed to unseen road.
        if (road_eligible and "body_speed_m_s" in self.status
                and self.straight_cruise / (1.0+load) > self.cruise):
            control_quality, quality_source = self._forward_quality(result, path, selected, positions)
        straight_candidate = (
            max(self.cruise, self.straight_cruise / (1.0 + load)) if boost_ready else self.cruise
        )
        cruise = max(self.confirmed_cruise, straight_candidate) if boost_ready else self.cruise
        if persistent_recovery:
            cruise = self.cruise
        applied_quality = control_quality
        applied_quality_source = quality_source
        if cruise > self.cruise:
            # Preserve the existing quality source. Raising bend cruise must
            # not also replace decayed history quality with fresh quality,
            # which would multiply effort beyond the configured cruise step.
            quality = (
                observation_quality
                if straight_candidate > self.cruise
                else control_quality
            )
            applied_quality = float(quality)
            if straight_candidate > self.cruise:
                applied_quality_source = "CURRENT_OBSERVATION"
        # Select the quality and effort once, then evaluate the coupled wheel
        # law once. Provisional baseline/quality/boost commands are not outputs
        # and used to repeat the same pursuit and curvature arithmetic.
        command = target_command(
            dict(raw, confidence=1. if self.speed_control is not None else applied_quality),
            self.lever, self.gain, cruise
        )
        if command is None:
            return None
        if self.speed_control is not None:
            command=self.speed_control.limit_fallback(command)
            # Both confidence and clearance have already constrained amplitude.
            # Preserve only the pursuit-geometry scale; do not apply the old
            # current-speed-dependent clearance ratio a second time.
            scale=max(0.,min(1.,target_scale))
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
            global_confidence=float(result.get("confidence", 0.0)),
            control_confidence=applied_quality,
            control_quality_source=applied_quality_source,
            forward_confidence=control_quality,
        )
        self.status.update(
            envelope=envelope, target_selection="ADAPTIVE_PURSUIT", target_speed_scale=target_scale
        )
        if persistent_recovery:
            output=tuple(v*scale for v in command)
            return self._history_output(output,age_s,support_sigma+prediction_sigma,positions[-1],envelope)
        if result.get("state") == "PREDICTED" or result.get("transition_pending"):
            # Short dropouts and pending handoffs share the same physical
            # budget. Processing latency already consumes unseen travel; a
            # fresh UART heartbeat cannot extend this road evidence.
            speed = self.status.get("speed_upper_m_s")
            model = result.get("road_model") or {}
            room = min(
                self.policy.maximum_blind_distance_m - float(model.get("blind_distance_m", 0.0)),
                float(model.get("length_m", 0.0)),
            ) - support_sigma - prediction_sigma
            remaining = self.policy.maximum_prediction_s - float(model.get("age_s", 0.0)) - age_s
            measured = speed is not None and math.isfinite(speed) and speed > 0
            if measured:
                room -= speed * age_s
            lease = (
                min(self.policy.maximum_prediction_command_s, remaining, room / speed)
                if measured and self.status.get("body_speed_m_s", 0.0) > 0
                and room > 0 and remaining > 0 else 0.0
            )
            horizon = max(float(self.status.get("motion_interval_s") or 0.0), age_s)
            prediction_scale = min(1.0, room / (speed * horizon)) if lease > 0 and horizon > 0 else 0.0
            budget = dict(
                remaining_distance_m=room, remaining_time_s=remaining,
                lease_s=lease, speed_scale=prediction_scale,
                measured_motion=bool(measured and self.status.get("motion_measured", True)),
                motion_source=self.status.get("motion_source"),
            )
            self.status["prediction"] = budget
            if result.get("transition_pending"):
                self.status["handoff"] = budget
            if lease <= 0:
                return None
            scale = min(scale, prediction_scale)
        output = tuple(v * scale for v in command) if scale > 0 else None
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
                alternate = self.command(result, age_s, lookahead_m=radius, alternatives=False,
                                         speed_profile_allowed=speed_profile_allowed)
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
        support_sigma_m=None,
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
            # Pose zero must keep the caller's arc; the final pose has no
            # following advance. Neither needs another pursuit calculation.
            if i == 0 or i == len(distances_ahead) - 1:
                continue
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
                # The first advance must use the caller's selected curvature.
                # Recomputing at pose zero would silently replace DWPP's
                # constrained turn with the ideal pursuit before checking it.
                target = selected["point"]
                curvature = self.gain * 2 * target[1] / max(float(target @ target), 1e-8) + self.trim
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
        corner_sigma = sigma
        if support_sigma_m is not None:
            values = np.asarray(support_sigma_m, float)
            if (values.shape != (len(support),) or not np.isfinite(values).all()
                    or np.any(values < 0)):
                raise ValueError("Invalid pointwise road uncertainty")
            # A projected corner lies on a segment. Use both endpoints' error
            # bounds, not the worst sample from an unrelated distant tail.
            right = np.clip(np.searchsorted(support_arc, positions), 1, len(support) - 1)
            corner_sigma = np.maximum(values[right - 1], values[right])
        clearance = self.width / 2 - distances - corner_sigma
        checked = int(known.sum())
        total = len(footprint)
        minimum = float(clearance[known].min()) if checked else float("inf")
        unsafe = np.flatnonzero((known & (clearance < 0)).reshape(-1, 4).any(axis=1))
        first_unsafe = float(distances_ahead[unsafe[0]]) if len(unsafe) else None
        coverage = checked / max(1, total)
        # The camera cannot see under the chassis. Missing coverage is reported,
        # not fabricated from an extrapolated prefix or treated as a collision.
        scale = 1.0
        safe_speed = None
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
            speed_limit_m_s=safe_speed,
        )
