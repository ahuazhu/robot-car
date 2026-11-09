import math

# Must match the active UNO firmware; the UART returns applied duty directly.
MAX_MOTOR_PWM = 200


class CommandRamp:
    """Limit drive effort with a common scale, preserving requested curvature.

    Bound both peak and mean absolute effort at the existing rise rate. A
    steering change redistributes effort between wheels; limiting each wheel
    independently would also brake the already fast wheel when leaving a turn.
    Deceleration and an exact zero always take priority. This shapes commanded
    effort, not measured wheel acceleration.
    """

    # MotorChannel sends zero once a published command reaches 300 ms old.
    # A gap at that boundary starts a new ramp, never banked acceleration.
    MAX_CONTINUOUS_INTERVAL_S = 0.30

    def __init__(self, rise_pwm_s=180.0):
        self.rise_pwm_s = rise_pwm_s
        self.reset()

    def reset(self):
        self.previous = (0.0, 0.0)
        self.last = None

    def pause(self, now, *, rolling=False):
        """Output zero, retaining a decaying restart envelope only while rolling.

        Explicit zero/STOP still resets immediately. A single missing target
        must not erase all acceleration history while the chassis is visibly
        moving; stale/no-motion evidence always takes the normal zero restart.
        """
        if (
            not rolling
            or self.last is None
            or not 0.0 <= now - self.last < self.MAX_CONTINUOUS_INTERVAL_S
        ):
            self.reset()
            return (0, 0)
        dt = now - self.last
        peak = max(abs(v) for v in self.previous)
        scale = max(0.0, 1.0 - self.rise_pwm_s * dt / peak) if peak else 0.0
        self.previous = tuple(v * scale for v in self.previous)
        self.last = now
        return (0, 0)

    def apply(self, command, now):
        if command is None or command == (0, 0):
            self.reset()
            return (0, 0)
        dt = None if self.last is None else now - self.last
        if dt is None or not 0.0 <= dt < self.MAX_CONTINUOUS_INTERVAL_S:
            self.reset()
            dt = 0.05
        self.last = now
        increase = self.rise_pwm_s * dt
        previous = tuple(abs(value) for value in self.previous)
        desired = tuple(abs(value) for value in command)
        # Mean effort owns forward acceleration; peak effort also bounds the
        # outside wheel when a turn is requested from straight running. Both
        # budgets share elapsed time and cannot be banked across a stale gap.
        scale = min(
            1.0,
            (sum(previous) + 2 * increase) / sum(desired),
            (max(previous) + increase) / max(desired),
        )
        self.previous = tuple(float(value) * scale for value in command)
        return tuple(round(value) for value in self.previous)


def can_enable_drive(calibration, runtime, force_preview=False):
    """Explicit current-pose calibration and two intentional config switches."""
    return (
        not force_preview
        and calibration.get("geometry_confirmed") is True
        and calibration.get("camera_pose_verified") is True
        and runtime.get("allow_drive") is True
        and runtime.get("preview_only", True) is False
    )


def compensate_pwm(command, peak_boost=37):
    """Raise starting effort with one common scale, preserving wheel ratio.

    A fixed additive offset on BOTH nonzero wheels changes curvature. The
    boost applies to peak effort, then scales the complete vector; saturation
    also uses one scale. This is not a calibrated linear dead-zone model.
    """
    magnitude = max(abs(value) for value in command)
    if magnitude == 0:
        return (0, 0)
    peak = min(float(MAX_MOTOR_PWM), magnitude + peak_boost)
    return tuple(round(value * peak / magnitude) for value in command)


class RoadSupervisor:
    """A running mission may pause and reacquire without losing its START.

    A failed association is not evidence of a missing road. Candidate geometry
    and ambiguous boundaries keep perception running at zero command until the
    tracker supplies a confirmed target. Attempt counts are diagnostics only.
    Declare road loss only after continuous timely observations find neither
    a boundary candidate nor usable historical support. Hardware watchdogs and
    explicit STOP remain independent and cannot be undone by this supervisor.
    """

    def __init__(self, missing_seconds=0.35):
        self.missing_seconds = missing_seconds
        self.reset()

    def reset(self):
        self.frames = 0
        self.missing_since = None
        self.phase = "IDLE"
        self.failed = None

    def update(
        self,
        result,
        command,
        now,
        processing_ms,
        budget_ms,
        *,
        stop_verified=False,
        road_ready=False,
        search_pending=False,
    ):
        if self.failed:
            if self.failed == "ROAD NOT FOUND" and road_ready and command is not None:
                self.reset()
                return None
            if self.failed == "ROAD NOT FOUND" and search_pending and command is not None:
                self.phase = "SEARCHING"
                return None
            return self.failed
        if command is not None:
            self.reset()
            return None
        self.frames += 1
        if processing_ms > budget_ms:
            # A slow frame authorizes zero only, but cannot prove the road is
            # absent. The motor process separately enforces its command lease.
            self.phase = "VISION_WAIT"
            self.missing_since = None
            return None
        evidence = result.get("road_evidence")
        if evidence not in ("OBSERVED", "CANDIDATE", "AMBIGUOUS", "ABSENT"):
            self.failed = "INVALID ROAD EVIDENCE"
            return self.failed
        if result.get("state") == "CONFIRMING":
            required = result.get("confirmation_required_frames", 0)
            if not isinstance(required, int) or isinstance(required, bool) or required < 2:
                self.failed = "INVALID CONFIRMATION STATE"
                return self.failed
        if evidence != "ABSENT" or result.get("valid"):
            self.missing_since = None
            self.phase = (
                ("CONFIRMING" if stop_verified else "SETTLING")
                if result.get("state") == "CONFIRMING"
                else "REACQUIRING"
            )
            return None
        self.phase = "ROAD_SEARCH"
        if self.missing_since is None:
            self.missing_since = now
        if now - self.missing_since >= self.missing_seconds:
            self.failed = "ROAD NOT FOUND"
            self.phase = "LOST_WAIT"
        return self.failed

    def snapshot(self):
        return dict(
            attempt_frames=self.frames,
            phase=self.phase,
            missing_since=self.missing_since,
            failure=self.failed,
        )


def target_command(result, turn_lever=0.15, steering_gain=1.0, cruise_pwm=60.0):
    """Curve geometry -> Pure Pursuit curvature -> coupled wheel commands.

    k=2*y/(x*x+y*y), turn/base=lever*k*gain. Slow continuously as either the
    pursuit curvature or supported road curvature increases. Confidence is
    already decayed by the tracker on predicted frames; provenance is unused.
    PWM speed/lever remain a provisional chassis model, not wheel odometry.
    """
    if not result.get("valid") or result.get("state") not in ("OBSERVED", "PREDICTED"):
        return None
    point = result.get("target_m")
    if point is None or len(point) != 2:
        return None
    x, y = point
    confidence = result.get("confidence", 0.0)
    path_curvature = result.get("path_curvature_m_inv", 0.0)
    if (
        not all(
            math.isfinite(value)
            for value in (x, y, confidence, path_curvature, turn_lever, steering_gain, cruise_pwm)
        )
        or x <= 0
        or turn_lever <= 0
        or steering_gain <= 0
        or cruise_pwm <= 0
        or confidence <= 0
    ):
        return None
    curvature = 2 * y / (x * x + y * y)
    corrected_curvature = steering_gain * curvature + result.get("steering_trim_m_inv", 0.0)
    if not math.isfinite(corrected_curvature):
        return None
    ratio = turn_lever * corrected_curvature
    load = min(1.0, turn_lever * steering_gain * max(abs(curvature), abs(path_curvature)))
    base = cruise_pwm * math.sqrt(min(1.0, confidence)) / (1.0 + load)
    turn = base * max(-1.0, min(1.0, ratio))
    return round(base - turn), round(base + turn)


def motion_from_pwm(pwm, turn_lever=0.15, speed_per_pwm=0.0015):
    """Bounded prediction's provisional wheel-speed model, shared with pursuit."""
    left, right = pwm
    return (
        max(0.0, (left + right) / 2) * speed_per_pwm,
        (right - left) * speed_per_pwm / (2 * turn_lever),
    )


def assess_vision_command(
    result, proposed, detection_ms, budget_ms, consecutive, minimum_confidence
):
    """Track geometry stability separately from the current command's age.

    A slow computation cannot drive, but it is not evidence that an otherwise
    accepted road curve changed. Predicted/missing/weak geometry still resets
    the observed-road streak. Startup also requires a timely current command.
    """
    observed = (
        proposed is not None
        and result.get("observed", False)
        and result.get("confidence", 0.0) >= minimum_confidence
    )
    consecutive = consecutive + 1 if observed else 0
    command = proposed if detection_ms <= budget_ms else None
    return command, consecutive


# Idle STATUS polls run once a second. Allow one 300 ms UART exchange and
# ordinary frame scheduling, but never let an old successful ACK imply READY.
MOTOR_STATUS_MAX_AGE_MS = 1500.0


def motor_startup_blockers(motor):
    """Require a live worker and fresh, stopped, fault-free UNO evidence.

    STOP/STATUS can legitimately acknowledge a latched emergency stop. That
    proves communication, not permission to ARM; keep those concepts separate.
    The ARM exchange still independently validates the actual returned state.
    """
    if not isinstance(motor, dict):
        return ["UNO_STATUS_UNAVAILABLE"]
    blocked = []
    if motor.get("worker_alive") is not True:
        blocked.append("UNO_WORKER_STOPPED")
    if motor.get("error"):
        blocked.append("UNO_LINK_ERROR")
    age = motor.get("ack_age_ms")
    if (isinstance(age, bool) or not isinstance(age, (int, float))
            or not math.isfinite(age)):
        blocked.append("UNO_STATUS_INVALID")
    elif not 0 <= age < MOTOR_STATUS_MAX_AGE_MS:
        blocked.append("UNO_STATUS_STALE")
    ack = motor.get("ack")
    if (not isinstance(ack, (list, tuple)) or len(ack) < 5
            or any(type(value) is not int for value in ack[:5])
            or not 0 <= ack[0] <= 65535 or ack[1] not in (0, 1)
            or any(abs(value) > MAX_MOTOR_PWM for value in ack[2:4])
            or not 0 <= ack[4] <= 255):
        if "UNO_STATUS_INVALID" not in blocked:
            blocked.append("UNO_STATUS_INVALID")
        return blocked
    if motor.get("armed") is True or any(ack[1:4]):
        blocked.append("UNO_NOT_STOPPED")
    if ack[4]:
        blocked.append("UNO_EMERGENCY_STOP" if ack[4] == 4 else "UNO_FAULT_%d" % ack[4])
    return blocked


def startup_blockers(
    result,
    command,
    consecutive,
    detection_ms,
    budget_ms,
    link_ok,
    preview_only,
    minimum_confidence,
    required=5,
):
    """Explain the existing startup gates without changing or issuing commands."""
    blocked = []
    if preview_only:
        blocked.append("PREVIEW_ONLY")
    if not link_ok:
        blocked.append("UNO_DISCONNECTED")
    if detection_ms > budget_ms:
        blocked.append("VISION_TOO_SLOW")
    elif command is None:
        blocked.append(
            "PATH_CONFIRMING" if result.get("state") == "CONFIRMING" else "ROAD_NO_TARGET"
        )
    if not result.get("observed"):
        blocked.append("ROAD_UNOBSERVED")
    if result.get("confidence", 0.0) < minimum_confidence:
        blocked.append("ROAD_WEAK")
    if consecutive < required:
        blocked.append("ROAD_STABILIZING")
    return blocked
