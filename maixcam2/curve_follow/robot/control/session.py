"""Mission lifecycle and the single owner of motor output.

Perception proposes commands. This layer applies STOP/start precedence,
mission restrictions, command freshness, recovery, ramps and motor leases.
It has no camera, touchscreen or HTTP dependency.
"""

from dataclasses import dataclass
import json
import time

from robot.control.requests import ControlRequest
from robot.control.controller import (
    assess_vision_command,
    compensate_pwm,
    motor_startup_blockers,
    startup_blockers,
)


@dataclass
class FrameDecision:
    result: dict
    details: dict
    captured: float
    command: object
    follow_command: object
    processing_ms: float
    lap_observation: object
    lap_finished: bool
    motor_snapshot: object
    zero_ack: bool
    light_start: bool
    crosswalk_state: dict


class DrivingSession:
    def __init__(
        self,
        settings,
        link,
        control,
        steering,
        search,
        start_light,
        laps,
        crosswalk,
        ramp,
        recovery,
        *,
        clock=time.monotonic,
    ):
        self.settings = settings
        self.link = link
        self.control = control
        self.steering = steering
        self.search = search
        self.start_light = start_light
        self.laps = laps
        self.crosswalk = crosswalk
        self.ramp = ramp
        self.recovery = recovery
        self.clock = clock
        self.active = False
        self.pwm = (0, 0)
        self.link_ok = False
        self.consecutive = 0
        self.next_control = self.next_status = 0.0
        self.reason = (
            ("CALIBRATION REQUIRED" if not settings.calibrated else "PREVIEW ONLY")
            if settings.preview_only
            else "WAIT START"
        )

    def connect(self):
        try:
            print(
                "PREVIEW_NO_UART" if self.settings.preview_only else "STOP_ACK",
                self.link.send(0),
                flush=True,
            )
            self.link_ok = True
        except RuntimeError as exc:
            self.reason = "UNO NOT CONNECTED"
            print(self.reason, str(exc), flush=True)

    def task_blockers(self, motor):
        if self.settings.crosswalk_policy.mode != "yield":
            return []
        blocked = self.settings.crosswalk_policy.blockers()
        if not (motor and motor.get("brake_lamp_supported")):
            blocked.append("BRAKE_LAMP_UNAVAILABLE")
        return blocked

    def readiness(self, frame):
        s = self.settings
        blocked = startup_blockers(
            frame.result,
            frame.command,
            self.consecutive,
            frame.processing_ms,
            s.vision_budget_ms,
            self.link_ok,
            s.preview_only,
            s.road_policy.measurement_quality,
            required=s.road_policy.startup_observations,
        )
        if not self.active and not s.preview_only:
            blocked.extend(motor_startup_blockers(frame.motor_snapshot))
        blocked.extend(self.task_blockers(frame.motor_snapshot))
        if self.crosswalk.failure:
            blocked.append(self.crosswalk.failure)
        configuring = bool(getattr(self.start_light, "configuration_pending", False))
        if configuring:
            blocked.append("LAMP_CONFIGURATION_PENDING")
        direct = bool(not self.active and not blocked)
        arm = (
            not self.active and not s.preview_only and not self.start_light.armed and self.start_light.available
            if self.start_light.enabled
            else direct
        )
        return dict(blocked_by=blocked, direct_start_ready=direct,
                    arm_start_light_ready=bool(arm and not configuring))

    def stop(self, reason, *, reset_tasks=False, clear_start_permission=False):
        # Local STOP/finish reset must not depend on receiving a UNO reply.
        self.start_light.cancel(clear_permission=clear_start_permission)
        self.active = False
        self.pwm = (0, 0)
        self.link.stop()
        self.ramp.reset()
        self.search.reset(clear_map=True)
        self.laps.stop()
        self.reason = reason
        self.control.stopped()
        if reset_tasks:
            self.recovery.reset()
            self.crosswalk.reset()
            self.steering.reset()

    def _start(self, frame, direct, target=None):
        was_armed = self.start_light.armed
        if not direct and self.start_light.enabled and not self.start_light.consume():
            self.reason = "ESP32 PERMISSION REQUIRED"
            return
        if not was_armed:
            self.laps.prepare(target)
        try:
            self.link.send(1)
            # Begin only once UNO has accepted zero-output ARM. Frame capture
            # predates vision processing and the UART round trip.
            started_at = self.clock()
            self.active = True
            self.next_control = 0.0
            self.recovery.reset()
            self.ramp.reset()
            self.steering.reset()
            self.search.reset(clear_map=True)
            self.reason = frame.result["mode"]
            self.start_light.started()
            self.laps.begin(frame.lap_observation, started_at)
            print(
                "START",
                json.dumps(
                    dict(
                        pose=frame.result,
                        auto=self.start_light.enabled and not direct,
                        source="MANUAL"
                        if direct or not self.start_light.enabled
                        else "ESP32_SIGNAL",
                        start_light=self.start_light.snapshot(),
                        total_laps=self.laps.total,
                    )
                ),
                flush=True,
            )
        except RuntimeError as exc:
            self.stop("UNO NOT CONNECTED")
            self.link_ok = False
            print("START REFUSED", str(exc), flush=True)

    def update(self, frame, actions=(), bgr=None):
        s = self.settings
        frame.command, self.consecutive = assess_vision_command(
            frame.result,
            frame.command,
            frame.processing_ms,
            s.vision_budget_ms,
            self.consecutive,
            s.road_policy.measurement_quality,
        )
        requests = [
            action if isinstance(action, ControlRequest) else ControlRequest(action)
            for action in actions
            if action is not None
        ]
        # STOP wins over all starts, target changes and completion events.
        if any(request.action == "stop" for request in requests):
            self.stop("STOPPED", reset_tasks=True, clear_start_permission=True)
        else:
            direct_request = next((r for r in requests if r.action == "start_now"), None)
            arm_request = next((r for r in requests if r.action == "start"), None)
            request = direct_request or arm_request
            target = request.laps if request is not None else None
            target_locked = bool(
                self.start_light.armed and target is not None and target != self.laps.total
            )
            mode_locked = bool(
                arm_request and not direct_request
                and (self.start_light.armed or getattr(self.start_light, "permission", False))
                and arm_request.initial_lamp_state != getattr(self.start_light, "initial_lamp_state", "OFF")
            )
            if mode_locked:
                print("START REFUSED", "已ARM，修改准备灯态请先STOP", flush=True)
            if (
                arm_request
                and not direct_request
                and not mode_locked
                and self.start_light.enabled
                and not self.active
                and not s.preview_only
                and not self.start_light.armed
            ):
                self.laps.prepare(target)
                accepted = self.start_light.arm(initial_lamp_state=arm_request.initial_lamp_state)
                frame.light_start = False
                self.reason = "WAIT START LIGHT" if accepted is not False else "ESP32 NOT READY"
                print(
                    "WAIT_START_LIGHT",
                    json.dumps(dict(self.start_light.snapshot(), total_laps=self.laps.total)),
                    flush=True,
                )
            wants_start = bool(direct_request) or (
                frame.light_start if self.start_light.enabled else bool(arm_request)
            )
            if wants_start and not self.active and not s.preview_only:
                gates = self.readiness(frame)
                if target_locked:
                    print("START REFUSED", "已ARM，修改圈数请先STOP", flush=True)
                elif gates["direct_start_ready"]:
                    self._start(frame, bool(direct_request), target)
                else:
                    self.reason = self.crosswalk.failure or (
                        "CROSSWALK SETUP REQUIRED"
                        if self.task_blockers(frame.motor_snapshot)
                        else "ROAD NOT READY"
                        if self.link_ok
                        else "UNO NOT CONNECTED"
                    )
                    if request:
                        print(
                            "START REFUSED",
                            json.dumps(
                                dict(
                                    reason=self.reason,
                                    observed_streak=self.consecutive,
                                    detection_ms=round(frame.processing_ms, 1),
                                    command_available=frame.command is not None,
                                    link_ok=self.link_ok,
                                )
                            ),
                            flush=True,
                        )
            if frame.lap_finished and self.active:
                self.stop("LAPS COMPLETE", clear_start_permission=True)
                print("RACE COMPLETE", json.dumps(self.laps.snapshot()), flush=True)
        if self.active:
            self._drive(frame)
        else:
            self._poll_idle()
        if s.preview_only:
            self.reason = "CALIBRATION REQUIRED" if not s.calibrated else "PREVIEW ONLY"

    def _drive(self, frame):
        s = self.settings
        now = self.clock()
        motion = frame.details.get("ground_motion") or {}
        p = s.crosswalk_policy
        rest = bool(
            motion.get("valid")
            and motion.get("speed_m_s") is not None
            and abs(motion["speed_m_s"]) <= p.stationary_speed_m_s
            and abs(motion["yaw_rad_s"]) <= p.stationary_yaw_rad_s
        )
        speed = float(motion.get("speed_m_s") or 0.0)
        rolling = bool(
            motion.get("valid")
            and speed > p.stationary_speed_m_s
            and speed * float(motion.get("interval_s") or 0.0)
            > 2 * float(motion.get("sigma_m") or 0.0)
        )
        failure = self.recovery.update(
            frame.result,
            self.crosswalk.supervision_command(
                frame.command, frame.processing_ms, s.vision_budget_ms
            ),
            now,
            frame.processing_ms,
            s.vision_budget_ms,
            stop_verified=frame.zero_ack and rest,
            road_ready=(
                self.consecutive >= s.road_policy.startup_observations
                or s.road_policy.confirmed_reacquisition(frame.result)
            ),
            search_pending=self.search.snapshot().get("search_pending", False),
        )
        if frame.command is None:
            self.pwm = self.ramp.pause(now, rolling=rolling)
            self.reason = {"VISION_WAIT": "VISION WAIT", "ROAD_SEARCH": "ROAD SEARCH"}.get(
                self.recovery.phase, "REACQUIRING ROAD"
            )
        command = self.crosswalk.limit_command(frame.command, frame.crosswalk_state, motion)
        failure = self.crosswalk.failure or failure
        if failure and failure != "ROAD NOT FOUND":
            self.stop(failure)
            print(
                "STOP EVENT",
                failure,
                json.dumps(dict(recovery=self.recovery.snapshot(), steering=self.steering.status)),
                flush=True,
            )
            return
        if now < self.next_control:
            return
        if failure == "ROAD NOT FOUND":
            command = (0, 0)
            self.reason = "ROAD NOT FOUND"
        search_state = self.search.snapshot()
        if self.search.is_turning() and command is not None:
            self.pwm = command
        elif (
            command == (0, 0) and frame.follow_command is None and failure is None
            and search_state.get("search_pending") and search_state.get("phase") == "SETTLING"
            and self.crosswalk.allows_road_search()
        ):
            # A search-generated zero is still published immediately. While
            # measured forward motion remains, decay the restart envelope just
            # as for a missing follow command instead of pretending rest. A
            # stopped/uncertain chassis or long command gap still resets it.
            self.pwm = self.ramp.pause(now, rolling=rolling)
        else:
            self.pwm = self.ramp.apply(command, now) if command is not None else (0, 0)
        if self.search.is_turning():
            self.ramp.reset()
        lease = None
        if any(self.pwm):
            lease = (
                self.search.snapshot().get("lease_s")
                if self.search.is_turning()
                else (self.steering.status.get("prediction") or {}).get("lease_s")
            )
        try:
            self.link.send(2, *compensate_pwm(self.pwm, s.pwm_boost), lease_s=lease)
            self.search.applied(self.clock(), self.pwm)
            self.steering.applied(self.clock(), self.pwm)
            self.next_control = self.clock() + 0.05
            if frame.command is not None and failure is None:
                self.reason = self._driving_reason(frame)
        except RuntimeError as exc:
            self.stop("MOTOR LINK STOP")
            self.link_ok = False
            print("STOP EVENT", self.reason, str(exc), flush=True)

    def _driving_reason(self, frame):
        if any(self.pwm) and self.laps.phase == "FINISH_DISTANCE":
            return self.laps.phase.replace("_", " ")
        if frame.follow_command is None and self.search.snapshot().get("search_pending"):
            return "ROAD SEARCH " + self.search.snapshot()["phase"]
        if self.settings.crosswalk_policy.mode == "yield" and self.crosswalk.track:
            return "CROSSWALK " + self.crosswalk.phase
        if self.search.snapshot().get("phase") == "CAUTIOUS":
            return "CAUTIOUS FOLLOW"
        return "CURVE HANDOFF" if frame.result.get("transition_pending") else frame.result["mode"]

    def _poll_idle(self):
        self.pwm = (0, 0)
        self.ramp.reset()
        if self.clock() >= self.next_status:
            try:
                ack = self.link.send(3 if self.link_ok else 0)
                if ack[1:4] != [0, 0, 0]:
                    self.link.stop()
                    raise RuntimeError("Unexpected armed state")
                if not self.link_ok:
                    self.reason = "WAIT START"
                self.link_ok = True
            except RuntimeError as exc:
                self.link_ok = False
                self.reason = "UNO NOT CONNECTED"
                print("UNO STATUS", str(exc), flush=True)
            self.next_status = self.clock() + 1.0
        if self.start_light.armed and self.link_ok:
            self.reason = "WAIT START LIGHT"
        elif self.reason in ("WAIT START", "ROAD NOT READY"):
            self.reason = (
                "WAIT START"
                if self.consecutive >= self.settings.road_policy.startup_observations
                else "ROAD NOT READY"
            )
