"""Sampled path-speed envelope in metres/seconds, independent of PWM and tasks.

Offline replay, the read-only observer and the active scalar speed controller
consume this module. Callers supply a supported path starting at station zero,
physical limits and conservative distance margins; no unseen gaps are joined.
"""

from dataclasses import dataclass
import math
from robot.core.route_confidence import confidence_speed_factor


@dataclass(frozen=True)
class SpeedLimits:
    maximum_m_s: float
    lateral_acceleration_m_s2: float
    acceleration_m_s2: float
    deceleration_m_s2: float
    delay_s: float
    turn_lever_m: float

    def __post_init__(self):
        positive = (self.maximum_m_s, self.lateral_acceleration_m_s2,
                    self.acceleration_m_s2, self.deceleration_m_s2, self.turn_lever_m)
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Speed, acceleration and turn lever must be positive and finite")
        if not math.isfinite(self.delay_s) or self.delay_s < 0:
            raise ValueError("Delay must be finite and nonnegative")


@dataclass(frozen=True)
class SpeedProfile:
    stations_m: tuple
    speeds_m_s: tuple
    point_limits_m_s: tuple
    entry_limit_m_s: float
    entry_reason: str
    limiting_station_m: float
    initial_overspeed: bool


def plan_speed_profile(stations_m, curvature_m_inv, limits, *, stop_at_m=None,
                       distance_margin_m=0.0, initial_speed_m_s=None, centerline_confidence=1.,
                       clearance_limit_m_s=None, exact_curvature_required=True):
    """Plan a forward-only sampled envelope with an explicit zero-speed end.

    Each sample is bounded by lateral acceleration and nominal differential
    outer-wheel speed, centerline confidence and an optional absolute chassis
    clearance speed limit. A backward pass brakes before future limits.
    Reaction/actuation delay is charged ONCE at entry, then a forward pass
    enforces acceleration. Complexity and memory are O(number of samples).

    A requested stop is inserted exactly, never rounded to the next sample.
    The end of supported road is also a stop constraint. The distance margin
    is subtracted from whichever comes first. Curvature at an inserted end
    uses the larger adjacent magnitude. No claim is made about unsampled
    curvature peaks, footprint clearance, calibrated friction or jerk limits.

    initial_speed_m_s optionally fixes a reachable starting speed. If it is
    above the entry envelope, initial_overspeed reports infeasibility instead
    of pretending the vehicle can instantly decelerate to the planned value.

    Exact pure pursuit cannot follow curvature beyond the forward-wheel cone.
    DWPP may instead select a bounded approximate arc, subsequently checked by
    the caller's body sweep; it disables only this exact-curvature feasibility
    constraint. All distance, confidence and braking limits remain in force.
    """
    s = tuple(float(v) for v in stations_m)
    k = tuple(float(v) for v in curvature_m_inv)
    if (len(s) < 2 or len(k) != len(s) or s[0] != 0.0
            or not all(math.isfinite(v) for v in s + k)
            or any(b <= a for a, b in zip(s, s[1:]))):
        raise ValueError("Need finite, increasing stations starting at zero and matching curvature")
    if not math.isfinite(distance_margin_m) or distance_margin_m < 0:
        raise ValueError("Distance margin must be finite and nonnegative")
    if stop_at_m is not None and (not math.isfinite(stop_at_m) or stop_at_m < 0):
        raise ValueError("Stop distance must be finite and nonnegative")
    if initial_speed_m_s is not None and (
            not math.isfinite(initial_speed_m_s) or initial_speed_m_s < 0):
        raise ValueError("Initial speed must be finite and nonnegative")
    if clearance_limit_m_s is not None and (
            not math.isfinite(clearance_limit_m_s) or clearance_limit_m_s < 0):
        raise ValueError("Clearance speed limit must be finite and nonnegative")

    task_stop = stop_at_m is not None and stop_at_m <= s[-1]
    end = max(0.0, min(s[-1], stop_at_m if task_stop else s[-1]) - distance_margin_m)
    end_reason = "STOP_POINT" if task_stop else "SUPPORT_END"
    cut = next((i for i, v in enumerate(s) if v >= end), len(s) - 1)
    if s[cut] == end:
        s, k = s[:cut + 1], k[:cut + 1]
    else:
        boundary_k = max((k[cut - 1], k[cut]), key=abs)
        s, k = s[:cut] + (end,), k[:cut] + (boundary_k,)

    confidence_limit=limits.maximum_m_s*confidence_speed_factor(centerline_confidence)
    caps, reasons = [], []
    for curvature in k:
        load = abs(limits.turn_lever_m * curvature)
        lateral = math.sqrt(limits.lateral_acceleration_m_s2 / abs(curvature)) if curvature else math.inf
        wheel = limits.maximum_m_s / (1.0 + load)
        choices = [(limits.maximum_m_s, "MAXIMUM_SPEED"), (lateral, "CURVATURE"), (wheel, "OUTER_WHEEL"),
                   (confidence_limit,'CENTERLINE_CONFIDENCE')]
        if clearance_limit_m_s is not None:
            choices.append((clearance_limit_m_s,'BODY_CLEARANCE'))
        if exact_curvature_required and load > 1.0:
            # Forward-only tracking cannot ask the inside wheel to reverse.
            choices.append((0.0, "FORWARD_KINEMATICS"))
        cap, reason = min(choices, key=lambda item: item[0])
        caps.append(cap)
        reasons.append(reason)
    caps[-1], reasons[-1] = 0.0, end_reason

    # The exact entry delay constraint for each future cap u at distance d is
    # v*T + (v*v-u*u)/(2*b) <= d when v > u. For v <= u no braking is needed.
    # Reapplying T on every backward edge would depend on sampling density and
    # incorrectly collapse speed as more points are added.
    deceleration = limits.deceleration_m_s2
    bt = deceleration * limits.delay_s
    entry, limiting = caps[0], 0
    for i, (distance, cap) in enumerate(zip(s, caps)):
        root = math.sqrt(bt * bt + cap * cap + 2.0 * deceleration * distance)
        # Rationalized root avoids cancellation for tiny stopping distances.
        delayed = (cap * cap + 2.0 * deceleration * distance) / (root + bt) if root + bt else 0.0
        candidate = max(cap, delayed)
        if candidate < entry:
            entry, limiting = candidate, i

    speeds = list(caps)
    for i in range(len(s) - 2, -1, -1):
        speeds[i] = min(speeds[i], math.sqrt(speeds[i + 1] ** 2 + 2.0 * deceleration * (s[i + 1] - s[i])))
    speeds[0] = min(speeds[0], entry)
    overspeed = initial_speed_m_s is not None and initial_speed_m_s > entry + 1e-9
    if initial_speed_m_s is not None:
        speeds[0] = min(speeds[0], initial_speed_m_s)
    for i in range(1, len(s)):
        speeds[i] = min(speeds[i], math.sqrt(speeds[i - 1] ** 2 + 2.0 * limits.acceleration_m_s2 * (s[i] - s[i - 1])))
    return SpeedProfile(
        s, tuple(speeds), tuple(caps), entry, reasons[limiting], s[limiting], overspeed
    )
