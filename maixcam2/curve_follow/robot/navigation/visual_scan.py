"""Explicitly enabled, bounded look-around without verified swept free space.

This state machine requires actual zero/pulse publication acknowledgements.
It never manufactures a road path or treats opposite pulses as measured yaw.
"""

import math
from robot.core.stationarity import StationaryEvidence
from robot.navigation.supported_path import supported_segment


class BoundedVisualScan:
    def __init__(self, policy, cruise, response_s, yaw_rate_rad_s, *, finish_priority=False):
        self.policy = policy
        self.cruise = cruise
        self.response_s = response_s
        self.yaw_rate_rad_s = yaw_rate_rad_s
        self.finish_priority = bool(finish_priority)
        self.measured_rest = StationaryEvidence(0.006, 0.025, 0.25, 0.45)
        self.reset()

    def reset(self):
        self.began = None
        self.pulse_index = 0
        self.round_index = 0
        self.total_pulses = 0
        self.hint = 1.0
        self.zero_required = False
        self.zero_sent_at = None
        self.sent_at = None
        self.rest_since = None
        self.proposal = None
        self.recovery_frames = 0
        self.recovery_at = None
        self.finished = False
        self.retryable = False
        self.measured_since = None
        self.measured_rest.reset()
        self.last_action = False
        self.status = dict(phase="IDLE", reason="", search_pending=False, lease_s=None)

    def start(self, motion, now, stopped, hint):
        """Enter only after the coordinator verifies task/user permission."""
        self.last_action = False
        return self._advance(motion, now, stopped, hint)

    def command(self, result, follow, motion, now, *, allowed, stopped, hint):
        """Continue one scan, including stopped confirmation of a fresh road."""
        self.last_action = False
        # Task ownership cancels rotation, not accepted road following. Finish
        # once so subsequent fresh observations can complete recovery; repeating
        # _finish() would erase that evidence on every crosswalk approach frame.
        # DrivingSession applies the crosswalk's stop/speed limit afterwards.
        if not allowed and not self.finished:
            self._finish("YIELD TASK CANCELLED VISUAL SCAN", retryable=True)
        blocked = result.get("invalid_reason") == "BODY ENVELOPE BLOCKED"
        if blocked and not self.finish_priority:
            return self._finish("BODY ENVELOPE BLOCKED")
        if not self.finished and now - self.began >= 6.0:
            self._finish("VISUAL SCAN TIME BUDGET EXHAUSTED", retryable=True)
        if not blocked and self._fresh_follow(result, follow):
            # The road model has already confirmed this replacement across
            # observations. Reuse that proof instead of starting a second
            # startup streak in the search coordinator.
            if self.policy.confirmed_reacquisition(result) or (
                self.finish_priority and self._observed_control_path(result)
            ):
                self.reset()
                return follow
            if self.recovery_at is None or now > self.recovery_at:
                self.recovery_frames += 1
                self.recovery_at = now
            if self.recovery_frames >= self.policy.startup_observations:
                self.reset()
                return follow
            self._set_status("EXHAUSTED" if self.finished else "SETTLING", "CONFIRMING FRESH ROAD")
            self.status["recovery_frames"] = self.recovery_frames
            return self._propose((0, 0), "zero")
        self.recovery_frames = 0
        self.recovery_at = None
        if self.finished:
            if not (allowed and self.retryable):
                return None
            # Each retry starts with a newly published zero and fresh stopped
            # ACK. Alternating the first direction is not a measured yaw reset.
            hint = -self.hint
            rounds, pulses = self.round_index, self.total_pulses
            self.reset()
            self.round_index, self.total_pulses = rounds, pulses
        return self._advance(motion, now, stopped, hint)

    def _set_status(self, phase, reason="", lease_s=None):
        self.status = dict(
            phase=phase,
            reason=reason,
            search_pending=not self.finished,
            lease_s=lease_s,
            space_source="BOUNDED_VISUAL_SCAN",
            space_verified=False,
            pulse_index=self.pulse_index,
            pulse_count=4,
            round_index=self.round_index,
            total_pulses=self.total_pulses,
            budget_s=6.0,
            sent_at=self.sent_at,
        )

    def _finish(self, reason, *, retryable=False):
        self.finished = True
        self.retryable = self.finish_priority and retryable
        self.proposal = None
        self.last_action = False
        self.recovery_frames = 0
        self.recovery_at = None
        self._set_status("RETRY_WAIT" if self.retryable else "EXHAUSTED", reason)
        self.status["search_pending"] = self.retryable
        return None

    def _observed_control_path(self, result):
        """Use the current steering-approved measured path, never a candidate."""
        path, _ = supported_segment(result, skip_inferred_prefix=True)
        sigma = result.get("reference_sigma_m")
        return bool(
            result.get("target_inferred") is False
            and result.get("road_evidence") == "OBSERVED"
            and path is not None
            and sigma is not None and math.isfinite(float(sigma))
            and 0 <= float(sigma) <= self.policy.maximum_sigma_m
        )

    def _fresh_follow(self, result, follow):
        if follow is None or not any(abs(float(value)) > 0 for value in follow):
            return False
        quality = float(result.get("observation_confidence") or 0.0)
        age = float(result.get("age_s") or 0.0)
        return bool(
            result.get("valid")
            and result.get("observed")
            and result.get("state") == "OBSERVED"
            and not result.get("transition_pending")
            and math.isfinite(quality)
            and quality >= self.policy.measurement_quality
            and math.isfinite(age)
            and 0 <= age <= self.policy.missing_command_s
        )

    def _propose(self, command, kind):
        self.proposal = (kind, tuple(command))
        self.last_action = kind == "pulse"
        return command

    def applied(self, now, command, issued=True):
        """Confirm a scan proposal only after its motor command was sent.

        The caller reports the proposed logical PWM pair after successful
        publication. Rejected/unsent proposals consume no pulse. A successful
        zero publication is also required before counting a settling interval.
        """
        proposal = self.proposal
        if (
            not issued
            or proposal is None
            or self.began is None
            or self.finished
            or command is None
            or tuple(command) != proposal[1]
        ):
            return False
        kind, _ = proposal
        self.proposal = None
        if kind == "pulse":
            self.pulse_index += 1
            self.total_pulses += 1
            self.sent_at = now
            self.zero_required = True
            self.zero_sent_at = None
            self.rest_since = None
            self.measured_since = None
            self.measured_rest.reset()
            self.status["pulse_index"] = self.pulse_index
            self.status["total_pulses"] = self.total_pulses
            self.status["sent_at"] = now
        elif self.zero_sent_at is None:
            self.zero_required = False
            self.zero_sent_at = now
            self.rest_since = None
        return True

    def _advance(self, motion, now, stopped, hint):
        # This is a user-enabled bounded look-around, not verified free space.
        # Four equal opposite PWM pulses do not imply a measured return to yaw.
        if self.began is None:
            self.began = now
            self.round_index += 1
            self.hint = 1.0 if hint >= 0 else -1.0
            self.zero_required = True
            self.pulse_index = 0
            self.sent_at = None
            self.zero_sent_at = None
            self.rest_since = None
        if now - self.began >= 6.0:
            return self._finish("VISUAL SCAN TIME BUDGET EXHAUSTED", retryable=True)
        if self.finished:
            return None
        self._set_status("SETTLING", "WAIT FOR SENT ZERO AND STOPPED ACK")
        if self.zero_required or self.zero_sent_at is None:
            return self._propose((0, 0), "zero")
        valid = bool(motion and motion.get("valid"))
        speed = abs(float(motion.get("speed_m_s") or 0.0)) if valid else 0.0
        rate = abs(float(motion.get("yaw_rad_s") or 0.0)) if valid else 0.0
        resting = not valid or (
            math.isfinite(speed)
            and math.isfinite(rate)
            and speed < self.policy.sample_step_m / self.response_s
            and rate < self.yaw_rate_rad_s / 4
        )
        # A noisy visual velocity must not veto every recovery round forever.
        # Finish priority may use a continuously published zero plus a fresh
        # stopped ACK after the full response/watchdog settling interval. The
        # resulting turn remains short and explicitly has unverified space.
        zero_settled = bool(
            self.finish_priority and stopped and self.zero_sent_at is not None
            and now-self.zero_sent_at >= self.response_s+self.policy.missing_command_s
        )
        if not stopped or not resting and not zero_settled:
            self.rest_since = None
            self.measured_since = None
            self.measured_rest.reset()
            self.status["reason"] = (
                "WAIT FOR MEASURED REST" if not resting else "WAIT FOR STOPPED ACK"
            )
            return self._propose((0, 0), "zero")
        if not resting and zero_settled:
            self.rest_since = self.zero_sent_at
            self.status["reason"] = "ZERO ACK SETTLING FALLBACK"
            self.status["rest_source"] = "ZERO_ACK_FALLBACK"
        if self.rest_since is None:
            self.rest_since = max(
                now, self.zero_sent_at, self.sent_at + 0.12 if self.sent_at is not None else now
            )
        # Waiting beyond the .12s lease plus response/watchdog time separates
        # any pre-pulse zero ACK from the fresh (<200ms) ACK supplied by main.
        wait = self.response_s + self.policy.missing_command_s
        measured = bool(
            self.finish_priority and valid
            and "rotation" in motion and "translation" in motion
            and self.measured_rest.observe(motion, now, stopped)
        )
        if measured:
            if self.measured_since is None:
                self.measured_since = now
        else:
            self.measured_since = None
            self.measured_rest.reset()
        settled = now - self.rest_since + 1e-9 >= wait or (
            measured and now - self.measured_since + 1e-9 >= 0.25
            and now - self.rest_since + 1e-9 >= 0.25
        )
        if not settled:
            self.status["reason"] = "WAIT FOR STOP SETTLING INTERVAL"
            return self._propose((0, 0), "zero")
        if self.pulse_index >= 4:
            return self._finish("VISUAL SCAN COMPLETED WITHOUT A CONFIRMED PATH", retryable=True)
        if now - self.began + 0.12 > 6.0:
            return self._finish("VISUAL SCAN TIME BUDGET EXHAUSTED", retryable=True)
        effort = math.floor(min(30.0, self.cruise / 2))
        if effort <= 0:
            return self._finish("VISUAL SCAN HAS NO MOTOR EFFORT")
        direction = (self.hint, -self.hint, -self.hint, self.hint)[self.pulse_index]
        self._set_status("SCANNING", lease_s=0.12)
        self.status["rest_source"] = "ZERO_ACK_FALLBACK" if not resting else "VISUAL_REST"
        return self._propose((-int(direction * effort), int(direction * effort)), "pulse")
