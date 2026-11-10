"""Offline interpolation of measured actual-UNO-PWM references, not a motor PID.

The wood-floor measurements are provisional steady straight-line references.
They do not identify low-duty deadband, reverse motion or either wheel in a
turn. Unsupported values return None; there is no extrapolation to PWM 0..200.
"""

import bisect
import math


class DriveReference:
    def __init__(self, record):
        self.points = tuple((float(pwm), float(block["speed_reference_m_s"]))
                            for block in record["offline_monotone_reference_blocks"]
                            for pwm in block["pwms"])
        if (len(self.points) < 2 or not all(math.isfinite(p) and math.isfinite(v) and p > 0 and v > 0
                                          for p, v in self.points)
                or any(p2 <= p1 or v2 < v1 for (p1, v1), (p2, v2) in zip(self.points, self.points[1:]))
                or self.points[-1][0] > record["actual_uno_pwm_limit"]):
            raise ValueError("Need ordered, monotone measured PWM/speed reference blocks")
        self.pwms, self.speeds = zip(*self.points)

    def speed_for_pwm(self, actual_pwm):
        return self._interpolate(actual_pwm, self.pwms, self.speeds)

    def pwm_for_speed(self, speed_m_s):
        # bisect_left chooses the least duty at an equal-speed plateau.
        return self._interpolate(speed_m_s, self.speeds, self.pwms)

    @staticmethod
    def _interpolate(value, source, destination):
        if not math.isfinite(value) or not source[0] <= value <= source[-1]:
            return None
        i = bisect.bisect_left(source, value)
        if source[i] == value:
            return destination[i]
        fraction = (value - source[i - 1]) / (source[i] - source[i - 1])
        return destination[i - 1] + fraction * (destination[i] - destination[i - 1])
