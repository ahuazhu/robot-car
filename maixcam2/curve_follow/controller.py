import math

# Must match the active UNO firmware; the UART returns applied duty directly.
MAX_MOTOR_PWM = 150


class CommandRamp:
    """Limit wheel acceleration with a common scale, preserving path curvature.

    Independent turn/base ramps used to straighten a requested sharp turn
    during acceleration. Deceleration and an exact zero always take priority.
    This is command shaping, not measured acceleration feedback.
    """
    def __init__(self,rise_pwm_s=180.):
        self.rise_pwm_s=rise_pwm_s
        self.reset()
    def reset(self):
        self.previous=(0.,0.);self.last=None
    def apply(self,command,now):
        if command is None or command==(0,0):
            self.reset();return (0,0)
        dt=.05 if self.last is None else max(0.,min(.1,now-self.last))
        self.last=now
        increase=self.rise_pwm_s*dt
        scale=min([1.]+[(old+increase)/desired for old,desired in zip(self.previous,command)
                         if desired>0 and desired>old+increase])
        self.previous=tuple(float(value)*scale for value in command)
        return tuple(round(value) for value in self.previous)


def can_enable_drive(calibration, runtime, force_preview=False):
    """Explicit current-pose calibration and two intentional config switches."""
    return (not force_preview and calibration.get('geometry_confirmed') is True
            and calibration.get('camera_pose_verified') is True
            and runtime.get('allow_drive') is True
            and runtime.get('preview_only', True) is False)


def compensate_pwm(command, peak_boost=37):
    """Raise starting effort with one common scale, preserving wheel ratio.

    A fixed additive offset on BOTH nonzero wheels changes curvature. The
    boost applies to peak effort, then scales the complete vector; saturation
    also uses one scale. This is not a calibrated linear dead-zone model.
    """
    magnitude=max(abs(value) for value in command)
    if magnitude==0:return (0,0)
    peak=min(float(MAX_MOTOR_PWM),magnitude+peak_boost)
    return tuple(round(value*peak/magnitude) for value in command)


class RoadSupervisor:
    """A running mission may pause and reacquire without losing its START.

    A failed association is not evidence of a missing road. Candidate geometry
    and ambiguous boundaries keep perception running at zero command until the
    tracker supplies a confirmed target. Attempt counts are diagnostics only.
    Declare road loss only after continuous timely observations find neither
    a boundary candidate nor usable historical support. Hardware watchdogs and
    explicit STOP remain independent and cannot be undone by this supervisor.
    """
    def __init__(self, missing_seconds=.35):
        self.missing_seconds = missing_seconds
        self.reset()

    def reset(self):
        self.frames = 0
        self.missing_since = None
        self.phase = 'IDLE'
        self.failed = None

    def update(self, result, command, now, processing_ms, budget_ms, *, stop_verified=False, road_ready=False, search_pending=False):
        if self.failed:
            if self.failed=='ROAD NOT FOUND' and road_ready and command is not None:
                self.reset()
                return None
            if self.failed=='ROAD NOT FOUND' and search_pending and command is not None:
                self.phase='SEARCHING'
                return None
            return self.failed
        if command is not None:
            self.reset()
            return None
        self.frames += 1
        if processing_ms > budget_ms:
            # A slow frame authorizes zero only, but cannot prove the road is
            # absent. The motor process separately enforces its command lease.
            self.phase = 'VISION_WAIT'
            self.missing_since = None
            return None
        evidence = result.get('road_evidence')
        if evidence not in ('OBSERVED','CANDIDATE','AMBIGUOUS','ABSENT'):
            self.failed = 'INVALID ROAD EVIDENCE'
            return self.failed
        if result.get('state') == 'CONFIRMING':
            required = result.get('confirmation_required_frames',0)
            if not isinstance(required,int) or isinstance(required,bool) or required < 2:
                self.failed = 'INVALID CONFIRMATION STATE'
                return self.failed
        if evidence != 'ABSENT' or result.get('valid'):
            self.missing_since = None
            self.phase = ('CONFIRMING' if stop_verified else 'SETTLING') if result.get('state') == 'CONFIRMING' else 'REACQUIRING'
            return None
        self.phase = 'ROAD_SEARCH'
        if self.missing_since is None:
            self.missing_since = now
        if now-self.missing_since >= self.missing_seconds:
            self.failed = 'ROAD NOT FOUND'
            self.phase = 'LOST_WAIT'
        return self.failed

    def snapshot(self):
        return dict(attempt_frames=self.frames,phase=self.phase,
                    missing_since=self.missing_since,failure=self.failed)


# Compatibility for offline traces recorded before the supervisor was named.
VisionRecovery=RoadSupervisor


def target_command(result, turn_lever=.15, steering_gain=1., cruise_pwm=60.):
    """Curve geometry -> Pure Pursuit curvature -> coupled wheel commands.

    k=2*y/(x*x+y*y), turn/base=lever*k*gain. Slow continuously as either the
    pursuit curvature or supported road curvature increases. Confidence is
    already decayed by the tracker on predicted frames; provenance is unused.
    PWM speed/lever remain a provisional chassis model, not wheel odometry.
    """
    if not result.get('valid') or result.get('state') not in ('OBSERVED', 'PREDICTED'):
        return None
    point = result.get('target_m')
    if point is None or len(point) != 2:
        return None
    x, y = point
    confidence = result.get('confidence', 0.)
    path_curvature = result.get('path_curvature_m_inv', 0.)
    if (not all(math.isfinite(value) for value in
                (x, y, confidence, path_curvature, turn_lever, steering_gain, cruise_pwm))
            or x <= 0 or turn_lever <= 0 or steering_gain <= 0 or cruise_pwm <= 0
            or confidence <= 0):
        return None
    curvature = 2*y/(x*x+y*y)
    corrected_curvature=steering_gain*curvature+result.get('steering_trim_m_inv',0.)
    if not math.isfinite(corrected_curvature):return None
    ratio = turn_lever*corrected_curvature
    load = min(1., turn_lever*steering_gain*max(abs(curvature), abs(path_curvature)))
    base = cruise_pwm*math.sqrt(min(1., confidence))/(1.+load)
    turn = base*max(-1., min(1., ratio))
    return round(base-turn), round(base+turn)


def motion_from_pwm(pwm, turn_lever=.15, speed_per_pwm=.0015):
    """Bounded prediction's provisional wheel-speed model, shared with pursuit."""
    left, right = pwm
    return (max(0., (left+right)/2)*speed_per_pwm,
            (right-left)*speed_per_pwm/(2*turn_lever))


def assess_vision_command(result, proposed, detection_ms, budget_ms,
                          consecutive, minimum_confidence):
    """Track geometry stability separately from the current command's age.

    A slow computation cannot drive, but it is not evidence that an otherwise
    accepted road curve changed. Predicted/missing/weak geometry still resets
    the observed-road streak. Startup also requires a timely current command.
    """
    observed = (proposed is not None and result.get('observed', False)
                and result.get('confidence', 0.) >= minimum_confidence)
    consecutive = consecutive+1 if observed else 0
    command = proposed if detection_ms <= budget_ms else None
    return command, consecutive


def startup_blockers(result, command, consecutive, detection_ms, budget_ms,
                     link_ok, preview_only, minimum_confidence, required=5):
    """Explain the existing startup gates without changing or issuing commands."""
    blocked=[]
    if preview_only:blocked.append('PREVIEW_ONLY')
    if not link_ok:blocked.append('UNO_DISCONNECTED')
    if detection_ms>budget_ms:blocked.append('VISION_TOO_SLOW')
    elif command is None:
        blocked.append('PATH_CONFIRMING' if result.get('state')=='CONFIRMING' else 'ROAD_NO_TARGET')
    if not result.get('observed'):blocked.append('ROAD_UNOBSERVED')
    if result.get('confidence',0.)<minimum_confidence:blocked.append('ROAD_WEAK')
    if consecutive<required:blocked.append('ROAD_STABILIZING')
    return blocked
