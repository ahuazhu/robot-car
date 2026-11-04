"""Temporal road tracking, motion compensation and supported target selection."""

import math
import time
import numpy as np
from robot.core.geometry import arc, choose_target, geometry_frame
from robot.core.road_policy import RoadPolicy
from robot.core.ground_motion import GroundMotion, motion_summary
from robot.perception.boundaries import CurvePixels
from robot.navigation.road_model import LocalRoadModel, BoundaryTrackBank


class TargetTracker:
    """Adapter: observe -> compensate motion -> update road -> select target.

    LocalRoadModel owns association, partial freshness and confirmation.
    Controller readiness cannot suppress otherwise valid road observations.
    """

    def __init__(self, cal):
        self.policy = RoadPolicy.from_config(cal)
        frame = cal.get("control_frame", {})
        self.reference_origin = np.asarray(frame.get("origin_m", [0.0, 0.0]), float)
        self.control_body = frame.get("body_m") if frame.get("geometry_verified") else None
        if self.control_body is not None and (
            len(self.control_body) != 3
            or not all(math.isfinite(v) and v > 0 for v in self.control_body)
        ):
            raise ValueError("Invalid body geometry")
        if self.reference_origin.shape != (2,) or not np.isfinite(self.reference_origin).all():
            raise ValueError("Invalid control frame")
        self.pixels = CurvePixels(cal, self.policy)
        self.model = LocalRoadModel(self.policy)
        self.boundaries = BoundaryTrackBank(self.policy)
        self.motion = GroundMotion(cal, self.policy)

    def update(self, bgr, now, v=0.0, omega=0.0, *, require_current=False):
        with geometry_frame():
            return self._update(bgr, now, v, omega, require_current=require_current)

    def _update(self, bgr, now, v=0.0, omega=0.0, *, require_current=False):
        p = self.policy
        tick = time.perf_counter()
        motion = self.motion.update(bgr, now, v, omega)
        motion_ms = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        self.model.predict(motion)
        self.boundaries.predict(motion, now)
        self.pixels.boundary_prior = self.boundaries.tracks
        prior, _ = self.model.support(now)
        measurement, details = self.pixels.measure(
            bgr, self.model.pending if self.model.pending is not None else prior
        )
        perception_ms = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        surface = details.get("surface")
        surface_rejection = self.model.reject_surface(surface)
        self.boundaries.reject_surface(surface)
        self.boundaries.update(details.get("candidates", {}), now)
        # Actual observed scene motion wins over a newly published zero command.
        stationary = (
            motion["speed_m_s"] < 0.005 and abs(motion["yaw_rad_s"]) < 0.02
            if motion["valid"]
            else abs(v) < 0.005 and abs(omega) < 0.02
        )
        continuation_margin = None
        if (
            self.control_body is not None
            and measurement is not None
            and motion["valid"]
            and not require_current
        ):
            _, old_support = self.model.support(now)
            if not old_support["reason"]:
                continuation_margin = max(
                    0.0,
                    self.pixels.width / 2
                    - float(self.control_body[2])
                    - float(old_support["sigma_m"])
                    - p.observation_sigma_m / max(measurement["confidence"], 0.1),
                )
        event = self.model.update(
            measurement,
            now,
            stationary,
            allow_handoff=bool(motion["valid"] and not require_current),
            continuation_clearance_m=continuation_margin,
        )
        path, support = self.model.support(now, current_only=require_current)
        model_ms = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        target = (
            choose_target(
                path,
                forward_min_m=p.forward_min_m,
                maximum_target_m=p.maximum_target_m,
                reference_origin_m=self.reference_origin,
                observation_sigma_m=float(support.get("sigma_m") or p.observation_sigma_m),
                maximum_reference_sigma_m=p.maximum_sigma_m,
                curve_analysis_window_m=p.curve_analysis_window_m,
            )
            if path is not None
            else None
        )
        observed_mask = support.get("observed_mask", [])
        target_observed = False
        if target is not None:
            position = target["position_m"]
            positions = arc(target["reference_path"])
            index = min(len(positions) - 1, int(np.searchsorted(positions, position)))
            target_observed = all(observed_mask[max(0, index - 1) : index + 1])
        reason = support["reason"] or ("NO FORWARD CURVE POINT" if target is None else "")
        transition_pending = bool(
            self.model.pending_count
            and event["continuation_allowed"]
            and target is not None
            and not reason
            and not require_current
        )
        state = (
            "PREDICTED"
            if transition_pending
            else "CONFIRMING"
            if self.model.pending_count
            else "LOST"
            if reason
            else "OBSERVED"
            if event["observed"]
            else "PREDICTED"
        )
        confidence = support.get("confidence", 0.0)
        # Position uncertainty and track existence are separate from quality.
        # Freshness remains per sample even after a distant accepted observation.
        evidence = (
            "OBSERVED"
            if event["observed"]
            else "CANDIDATE"
            if event["measurement_usable"]
            else "AMBIGUOUS"
            if measurement is not None or any(details.get("counts", {}).values())
            else "ABSENT"
        )
        result = dict(
            valid=not bool(reason),
            state=state,
            road_evidence=evidence,
            mode=self.model.mode if event["observed"] else "LOST" if reason else "PREDICT",
            confidence=float(confidence),
            observed=event["observed"],
            invalid_reason=reason,
            observation_confidence=float(measurement["confidence"])
            if event["observed"] and measurement is not None
            else 0.0,
            measurement_reject=None
            if event["observed"]
            else (
                event["reject"]
                if measurement is not None
                else details.get("measurement_reason", "NO EDGE")
            ),
            age_s=round(support["age_s"], 3),
            reacquired=event["reacquired"],
            pending_frames=self.model.pending_count,
            pending_confidence=self.model.pending_confidence,
            transition_pending=transition_pending,
            association_kind=event["association_kind"],
            handoff_evidence=event["handoff_evidence"],
            history_continuity_evidence=event["history_continuity_evidence"]
            if self.model.pending_count
            else None,
            association_prior_path_m=event["prior_path"] if self.model.pending_count else None,
            candidate_path_m=self.model.pending.tolist() if self.model.pending_count else None,
            candidate_mode=measurement["mode"] if measurement is not None else None,
            candidate_confidence=measurement["confidence"] if measurement is not None else 0.0,
            confirmation_required_frames=event["required"] if self.model.pending_count else 0,
            measurement_alignment=event["alignment"],
            confirmation_alignment=event["confirmation_alignment"],
            blind_distance_m=support["blind_distance_m"],
            path=path.tolist() if path is not None else [],
            target_m=(target["point"] + self.reference_origin).tolist() if target else None,
            control_target_m=target["point"].tolist() if target else None,
            control_reference_origin_m=self.reference_origin.tolist(),
            control_path_m=target["reference_path"].tolist() if target else [],
            reference_sigma_m=target["reference_sigma_m"] if target else None,
            reference_inferred=bool(observed_mask and not all(observed_mask)),
            target_inferred=bool(target is not None and not target_observed),
            path_observed_mask=observed_mask,
            control_target_source=("OBSERVED" if target_observed else "HISTORY PREDICTION")
            if target
            else None,
            target_distance_m=target["distance_m"] if target else None,
            target_selection=target["target_selection"] if target else None,
            target_speed_scale=target["target_speed_scale"] if target else 0.0,
            path_heading_rad=target["path_heading_rad"] if target else None,
            path_curvature_m_inv=target["curvature"] if target else 0.0,
            near_support_m=target["near_support_m"] if target else None,
            reliable_length_m=target["reliable_length_m"] if target else 0.0,
            road_model=dict(
                support,
                version=2,
                support_scope="CURRENT" if require_current else "HISTORY_ALLOWED",
                geometry_observed=event["observed"],
                model_age_s=None
                if self.model.last_seen is None
                else round(now - self.model.last_seen, 3),
            ),
            timings_ms=dict(
                motion=round(motion_ms, 1),
                perception=round(perception_ms, 1),
                model=round(model_ms, 1),
                target=round((time.perf_counter() - tick) * 1000, 1),
            ),
            boundary_tracks=self.boundaries.summary(now),
            surface_rejection=surface_rejection,
            motion=motion_summary(motion),
            counts=details.get("counts", {}),
            boundary_fallback=details.get("boundary_fallback"),
            white_value_min=details.get("white_value_min"),
            photometric_passes=details.get("photometric_passes", 0),
            exposure_normalized=details.get("exposure_normalized", False),
        )
        details["measured_road"] = measurement
        details["ground_motion"] = motion
        return result, details
