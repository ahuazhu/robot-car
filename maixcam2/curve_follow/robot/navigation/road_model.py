"""Persistent local road geometry, independent of target selection and PWM.

Each sample retains its own observation time, quality, uncertainty and blind
travel. Updating a far portion cannot make an unseen near portion fresh.
Single-edge/dual-edge provenance does not change association or confirmation.
"""

import math
import numpy as np
from robot.core.geometry import (
    arc,
    fit_curve,
    describe_curve,
    merge_centers,
    curve_agreement,
    contiguous_runs,
    support_projection,
    curve_handoff,
    contained_observation,
    choose_target,
    differential,
)
from robot.core.ground_motion import apply_ground_motion
from robot.core.road_policy import RoadPolicy
from robot.core.route_confidence import confidence_profile,innovation_likelihood


class LocalRoadModel:
    def __init__(self, policy=None, *, persistent_history=False, reference_origin_m=(0.,0.)):
        self.policy = policy or RoadPolicy()
        self.persistent_history = bool(persistent_history)
        self.reference_origin = np.asarray(reference_origin_m,float)
        self.reset()

    def reset(self):
        """Revoke the complete old road, including pending replacement evidence."""
        self.path = None
        self.seen = self.quality = self.variance = self.travel = None
        self.mode = "LOST"
        self.last_seen = None
        self.confidence = 0.0
        self.retired_support = None
        self.surface_contradicted = False
        self.observation_agreement = 1.
        self.surface_support = 1.
        self._pruned_at = None
        self.reset_pending()

    def reset_pending(self):
        self.pending = None
        self.pending_count = 0
        self.pending_confidence = 0.0

    def reject_surface(self, surface):
        """Current visible exterior can disprove history, never refresh it.

        Conservative mode revokes a contradicted curve. Persistent prediction
        records counterevidence and lowers confidence without deleting it;
        independently confirmed current geometry can then correct the route.
        """
        if surface is None or self.path is None:
            return None
        evidence = surface.contradiction(self.path, np.sqrt(self.variance))
        summary = dict(
            contradicted=bool(evidence["contradicted"]),
            visible_samples=int(evidence["visible_samples"]),
            green_samples=int(evidence["green_samples"]),
            longest_green_m=float(evidence["longest_green_m"]),
        )
        self.surface_contradicted = (
            self.surface_contradicted or summary["contradicted"]
            if self.persistent_history else summary["contradicted"]
        )
        if self.persistent_history and summary['contradicted']:
            self.surface_support=1. / (1. + (summary['longest_green_m']/self.policy.minimum_segment_m)**2)
        if summary["contradicted"] and not self.persistent_history:
            self.reset()
        return summary

    def predict(self, motion):
        self._pruned_at = None
        if self.path is not None:
            self.path = apply_ground_motion(self.path, motion)
            self.variance += motion["sigma_m"] ** 2
            self.travel += motion["travel_m"]
        if self.pending is not None:
            self.pending = apply_ground_motion(self.pending, motion)

    def initialize(self, path, now, quality, mode):
        agreement=(self.observation_agreement
                   if self.persistent_history and self.path is not None else 1.)
        self.path = np.asarray(path, float).copy()
        count = len(self.path)
        self.seen = np.full(count, now, float)
        self.quality = np.full(count, quality, float)
        self.variance = np.full(count, (self.policy.observation_sigma_m / max(quality, 0.1)) ** 2)
        self.travel = np.zeros(count)
        self.mode = mode
        self.last_seen = now
        self.confidence = quality
        self.retired_support = None
        self.surface_contradicted = False
        # Confirmation authorizes replacement geometry, not instant recovery
        # from the innovation that made that replacement uncertain.
        self.observation_agreement = agreement
        self.surface_support = 1.

    def accept_agreement(self, quality, likelihood=1.):
        """Decrease immediately; restore trust with independent accepted evidence."""
        if not self.persistent_history:
            self.observation_agreement=1.
        elif likelihood < self.observation_agreement:
            self.observation_agreement=likelihood
        else:
            evidence=min(1.,quality/self.policy.confirmation_evidence)
            self.observation_agreement+=evidence*(likelihood-self.observation_agreement)

    def retire_expired(self, now):
        """Compact expiring conservative support; retain persistent predictions.

        Preserve independent candidate confirmation, but retain only diagnostic
        ages/distances from a retired road. Expired points cannot veto a new
        observation or authorize driving through a gap.
        """
        if self._pruned_at == now:
            return
        self._pruned_at = now
        if self.path is None:
            return
        if self.persistent_history:
            # Retention and command confidence are separate. Keep the accepted
            # curve until measured replacement or an explicit reset; time,
            # travel and uncertainty do not erase the prediction.
            return
        path, support = self.support(now)
        if path is None:
            self.path = None
            self.seen = self.quality = self.variance = self.travel = None
            self.mode = "LOST"
            self.last_seen = None
            self.confidence = 0.0
            self.retired_support = support
            return
        # support() returns one continuous eligible component in original order.
        start = int(np.argmin(np.linalg.norm(self.path - path[0], axis=1)))
        stop = start + len(path)
        self.path = self.path[start:stop]
        self.seen, self.quality, self.variance, self.travel = [
            values[start:stop] for values in (self.seen, self.quality, self.variance, self.travel)
        ]

    def confirm(self, measurement):
        p = self.policy
        q = measurement["confidence"]
        alignment = (
            curve_agreement(self.pending, measurement["path"]) if self.pending is not None else None
        )
        if q < p.confirmation_quality:
            self.reset_pending()
            return alignment, 0
        consistent = (
            alignment is not None
            and alignment["distance_m"] < 0.035
            and alignment["heading_cos"] > 0.90
        )
        self.pending_count = self.pending_count + 1 if consistent else 1
        self.pending_confidence = min(self.pending_confidence, q) if consistent else q
        self.pending = np.asarray(measurement["path"]).copy()
        return alignment, math.ceil(p.confirmation_evidence / self.pending_confidence)

    def accept_contained(self, candidate, proof, prior, measurement, now):
        """Replace only the actually observed interval, preserving both tails."""
        p = self.policy
        offset = int(np.argmin(np.linalg.norm(self.path - prior[0], axis=1)))
        positions = arc(self.path)
        start = positions[offset] + proof["start_m"]
        end = positions[offset] + proof["end_m"]
        before = np.flatnonzero(positions < start - 1e-6)
        after = np.flatnonzero(positions > end + 1e-6)
        path = np.vstack((self.path[before], candidate, self.path[after]))
        # A compatible local observation must not introduce a cusp, loop or
        # reversed splice into the stored road. No fitted gap is manufactured.
        if describe_curve(path) is None:
            return False
        q = measurement["confidence"]
        count = len(candidate)
        self.seen = np.concatenate((self.seen[before], np.full(count, now), self.seen[after]))
        self.quality = np.concatenate(
            (self.quality[before], np.full(count, q), self.quality[after])
        )
        self.variance = np.concatenate(
            (
                self.variance[before],
                np.full(count, (p.observation_sigma_m / q) ** 2),
                self.variance[after],
            )
        )
        self.travel = np.concatenate((self.travel[before], np.zeros(count), self.travel[after]))
        self.path = path
        self.mode = measurement["mode"]
        self.last_seen = now
        self.confidence = q
        self.accept_agreement(q)
        self.reset_pending()
        return True

    def update(
        self,
        measurement,
        now,
        stationary=False,
        *,
        allow_handoff=False,
        continuation_clearance_m=None,
        boundary_confirmed=False,
    ):
        """Accept geometry without asking whether it already has a drive target."""
        p = self.policy
        event = dict(
            observed=False,
            reacquired=False,
            reject="NO EDGE",
            alignment=None,
            confirmation_alignment=None,
            required=0,
            association_kind="UNOBSERVED",
            handoff_evidence=None,
            history_continuity_evidence=None,
            continuation_allowed=False,
            prior_path=None,
            measurement_usable=False,
        )
        self.retire_expired(now)
        if measurement is None or measurement["confidence"] < p.measurement_quality:
            self.reset_pending()
            return event
        candidate = np.asarray(measurement["path"], float)
        curve = describe_curve(candidate)
        if curve is None:
            self.reset_pending()
            event["reject"] = "INVALID ROAD GEOMETRY"
            return event
        event["measurement_usable"] = True
        if (self.persistent_history and self.path is not None
                and choose_target(candidate, reference_origin_m=self.reference_origin,
                    forward_min_m=p.forward_min_m, maximum_target_m=p.maximum_target_m) is None):
            # A stable observation of another road beyond control range is
            # not a correction of the route being followed. Retain the primary
            # prediction; confirmation alone cannot confer route identity.
            self.reset_pending()
            event["reject"] = "CANDIDATE OUTSIDE CONTROL RANGE"
            event["association_kind"] = "UNRELATED_REMOTE_OBSERVATION"
            return event
        if self.path is None and self.retired_support is not None:
            # Cold startup and recovery are distinct. A fully expired reference
            # supplies no geometry proof; confirm fresh candidates against each
            # other rather than comparing them to the retired path.
            event["association_kind"] = "REINITIALIZATION"
            if boundary_confirmed:
                self.initialize(candidate, now, measurement["confidence"], measurement["mode"])
                self.reset_pending()
                event.update(
                    observed=True, reacquired=True, reject=None,
                    association_kind="CONFIRMED_BOUNDARY_REINITIALIZATION",
                )
                return event
            event["confirmation_alignment"], required = self.confirm(measurement)
            event["required"] = required
            if not required or self.pending_count < required:
                event["reject"] = "FRESH ROAD CONFIRMATION"
                return event
            self.initialize(candidate, now, measurement["confidence"], measurement["mode"])
            self.reset_pending()
            event.update(observed=True, reacquired=True, reject=None)
            return event
        agreement = curve_agreement(self.path, candidate) if self.path is not None else None
        event["alignment"] = agreement
        # Association has an uncertainty term, not a growing arbitrary jump
        # allowance. Outliers are confirmed while commands are paused.
        sigma = 0.0 if self.variance is None else float(np.sqrt(np.median(self.variance)))
        gate = p.association_distance_m + min(p.maximum_sigma_m, sigma)
        consistent = self.path is None or (
            agreement is not None
            and agreement["distance_m"] < gate
            and agreement["heading_cos"] > 0.65
        )
        geometry_consistent = consistent
        fused = None
        if consistent and self.path is not None:
            old_curve = fit_curve(self.path)
            old_weight = 1.0 / max(float(np.median(self.variance)), 1e-8)
            new_weight = (measurement["confidence"] / p.observation_sigma_m) ** 2
            fused = (
                merge_centers(
                    old_curve,
                    curve,
                    old_weight,
                    new_weight,
                    gate,
                    p.association_heading_cos,
                    p.minimum_overlap_m,
                )
                if old_curve
                else None
            )
            # A usable observation that cannot be spliced enters the same
            # bounded confirmation path as any other geometry conflict. Never
            # reset confirmation forever merely because fusion is unavailable.
            consistent = fused is not None
        if not consistent:
            prior, support = self.support(now)
            if self.persistent_history and prior is not None:
                if agreement is not None:
                    likelihood=innovation_likelihood(
                        agreement['distance_m'],agreement['heading_cos'],p.association_distance_m)
                else:
                    distance,position= support_projection(candidate,prior)
                    nearest=int(np.argmin(distance))
                    old_index=min(len(prior)-1,int(np.searchsorted(arc(prior),position[nearest])))
                    old_tangent,_=differential(prior)
                    new_tangent,_=differential(candidate)
                    heading=float(old_tangent[old_index]@new_tangent[nearest])
                    likelihood=innovation_likelihood(float(distance[nearest]),heading,p.association_distance_m)
                # A conflicting candidate is counterevidence, not accepted
                # support for the road still being followed. Only an accepted
                # observation may restore trust through accept_agreement().
                self.observation_agreement=min(self.observation_agreement,likelihood)
            event["prior_path"] = None if prior is None else prior.tolist()
            # A small current patch inside a valid road is a partial update,
            # not a disconnected replacement. Short samples must not force the
            # entire old road to be discarded after a stationary confirmation.
            if prior is not None and measurement["confidence"] >= p.confirmation_quality:
                tolerance = math.hypot(
                    support["sigma_m"], p.observation_sigma_m / measurement["confidence"]
                )
                patch = contained_observation(
                    prior, candidate, tolerance, p.association_heading_cos, p.minimum_overlap_m
                )
                if patch is not None and self.accept_contained(
                    candidate, patch, prior, measurement, now
                ):
                    event.update(
                        observed=True,
                        reject=None,
                        association_kind="CONTAINED_OBSERVATION",
                        alignment=patch,
                    )
                    return event
            # Missing long overlap is distinct from contradictory geometry.
            # Only a valid old component and uncertainty-bounded exit/entry
            # agreement can authorize moving confirmation of a replacement.
            handoff = None
            if (
                agreement is None
                and prior is not None
                and measurement["confidence"] >= p.confirmation_quality
            ):
                tolerance = math.hypot(
                    support["sigma_m"], p.observation_sigma_m / measurement["confidence"]
                )
                handoff = curve_handoff(
                    prior, candidate, tolerance, p.association_heading_cos, p.minimum_overlap_m
                )
            event["handoff_evidence"] = handoff
            event["association_kind"] = (
                "COVERAGE_HANDOFF"
                if handoff
                else "INSUFFICIENT_OVERLAP"
                if agreement is None
                else "SUPPORT_FUSION_UNAVAILABLE"
                if geometry_consistent
                else "GEOMETRY_CONFLICT"
            )
            # A second, physical proof handles sparse overlap: both lane
            # hypotheses must admit the measured chassis and all shared
            # samples must agree in direction/order. This does not extend the
            # old curve or make it fresh. Candidate adoption STILL requires
            # independent repeated observations through confirm().
            history_continuity = None
            if (
                continuation_clearance_m is not None
                and continuation_clearance_m > 0
                and prior is not None
                and (agreement is None or geometry_consistent)
                and measurement["confidence"] >= p.confirmation_quality
            ):
                history_continuity = curve_handoff(
                    prior,
                    candidate,
                    min(p.association_distance_m, continuation_clearance_m),
                    p.association_heading_cos,
                    max(arc(prior)[-1], arc(candidate)[-1]),
                )
            event["history_continuity_evidence"] = history_continuity
            moving_handoff = allow_handoff and (
                handoff is not None or history_continuity is not None
            )
            event["continuation_allowed"] = moving_handoff
            if moving_handoff and handoff is not None and boundary_confirmed:
                # The current selected boundary already has independent
                # repeated-observation proof. With a strict ordered endpoint
                # handoff, confirming that same candidate again only forces a
                # stop while the old short support expires. Adopt only the
                # observed candidate; never stitch across an unobserved gap.
                self.initialize(candidate, now, measurement["confidence"], measurement["mode"])
                self.reset_pending()
                event.update(
                    observed=True, reacquired=True, reject=None,
                    association_kind="CONFIRMED_BOUNDARY_HANDOFF",
                    continuation_allowed=False,
                )
                return event
            event["confirmation_alignment"], required = self.confirm(measurement)
            plausible = moving_handoff or (
                agreement is not None
                and agreement["distance_m"] < 0.15
                and agreement["heading_cos"] > 0.5
            )
            if self.pending_count and not plausible:
                required = max(p.stopped_confirmation_frames, required)
            event["required"] = required
            event["reacquired"] = bool(
                self.pending_count >= required > 0
                and (plausible or stationary or self.persistent_history)
            )
            if not event["reacquired"]:
                event["reject"] = "COVERAGE HANDOFF PENDING" if moving_handoff else "ROAD CONFLICT"
                return event
            # A confirmed disconnected replacement does not inherit old tails.
            if self.persistent_history and not plausible and not stationary:
                event["association_kind"] = "CONFIRMED_PREDICTION_CORRECTION"
            self.initialize(candidate, now, measurement["confidence"], measurement["mode"])
        elif self.path is None:
            self.initialize(candidate, now, measurement["confidence"], measurement["mode"])
        else:
            path = fused["path"]
            old_dist, old_s = support_projection(path, self.path)
            new_dist, new_s = support_projection(path, candidate)
            old_arc, new_arc = arc(self.path), arc(candidate)
            # Only points actually covered by the new measurement are fresh;
            # neither a fitted extension nor a nearby clamped endpoint counts.
            covered = (new_dist <= gate) & (new_s > 1e-5) & (new_s < new_arc[-1] - 1e-5)
            covered |= np.linalg.norm(path - candidate[0], axis=1) < p.sample_step_m * 0.55
            covered |= np.linalg.norm(path - candidate[-1], axis=1) < p.sample_step_m * 0.55
            seen = np.interp(old_s, old_arc, self.seen)
            quality = np.interp(old_s, old_arc, self.quality)
            variance = np.interp(old_s, old_arc, self.variance) + old_dist**2
            travel = np.interp(old_s, old_arc, self.travel)
            seen[covered] = now
            quality[covered] = measurement["confidence"]
            variance[covered] = (
                p.observation_sigma_m / max(measurement["confidence"], 0.1)
            ) ** 2 + new_dist[covered] ** 2
            travel[covered] = 0.0
            self.path, self.seen, self.quality, self.variance, self.travel = (
                path,
                seen,
                quality,
                variance,
                travel,
            )
            self.mode = measurement["mode"]
            self.confidence = measurement["confidence"]
            self.last_seen = now
            likelihood=innovation_likelihood(agreement['distance_m'],agreement['heading_cos'],
                                            p.association_distance_m) if agreement else 1.
            self.accept_agreement(measurement['confidence'],likelihood)
        if event["association_kind"] == "UNOBSERVED":
            event["association_kind"] = "OVERLAP" if agreement else "INITIAL"
        event["continuation_allowed"] = False
        self.reset_pending()
        event["observed"] = True
        event["reject"] = None
        return event

    def support(self, now, *, current_only=False):
        """Query a continuous component without changing persistent evidence.

        Arming requires a sufficiently long, trusted current observation.
        Driving may use transported history through an occlusion. Persistent
        mode has no age/travel/uncertainty expiry; those remain honest telemetry
        and confidence inputs. Current-only startup never borrows old samples.
        Old tails remain in the model regardless of which component is queried.
        """
        p = self.policy
        if self.path is None:
            return None, dict(self.retired_support) if self.retired_support is not None else dict(
                reason="NO ROAD MODEL", age_s=0.0, blind_distance_m=0.0, sigma_m=None
            )
        age = np.maximum(0.0, now - self.seen)
        sigma = np.sqrt(self.variance)
        if self.persistent_history:
            q = confidence_profile(self.quality,sigma,self.observation_agreement,
                                   self.surface_support,p.maximum_sigma_m)
            good = np.isfinite(self.path).all(axis=1) & np.isfinite(sigma)
        else:
            q = self.quality * np.exp(-age / p.confidence_decay_s)
            good = (age <= p.maximum_prediction_s) & (self.travel <= p.maximum_blind_distance_m)
            good &= (sigma <= p.maximum_sigma_m) & (q >= 0.22)
        if current_only:
            # Current accepted geometry is eligible for the startup streak.
            # confirmation_quality belongs to adoption of conflicting roads;
            # applying it here rejects a stable accepted road forever.
            good &= (age < 1e-6) & (q >= p.measurement_quality)
        runs = [
            ids
            for ids in contiguous_runs(good)
            if len(ids) >= 5 and arc(self.path[ids])[-1] >= p.minimum_segment_m
        ]
        if not runs:
            return None, dict(
                reason="NO CURRENT ROAD SUPPORT" if current_only else
                    "NO USABLE ROAD GEOMETRY" if self.persistent_history else "ROAD SUPPORT EXPIRED",
                age_s=float(age.max()),
                blind_distance_m=float(self.travel.max()),
                sigma_m=float(sigma.max()),
            )
        ids = min(runs, key=lambda x: float(np.linalg.norm(self.path[x], axis=1).min()))
        path = self.path[ids]
        return path, dict(
            reason="",
            age_s=float(age[ids].max()),
            blind_distance_m=float(self.travel[ids].max()),
            sigma_m=float(sigma[ids].max()),
            sigma_profile=sigma[ids].tolist(),
            confidence=float(q[ids].min()),
            confidence_profile=q[ids].tolist(),
            length_m=float(arc(path)[-1]),
            fresh_fraction=float(np.mean(age[ids] < 1e-6)),
            sample_count=len(path),
            oldest_seen=float(self.seen[ids].min()),
            observed_mask=(age[ids] < 1e-6).tolist(),
            persistent_history=self.persistent_history,
            surface_contradicted=self.surface_contradicted,
            observation_agreement=self.observation_agreement,
            surface_support=self.surface_support,
        )


class BoundaryTrackBank:
    """Bounded side-specific tracks, separate from a frame's contour candidates.

    Tracks keep identity through partial visibility. Missing contours do not
    become fresh detections, and ambiguous candidates remain separate tracks.
    """

    def __init__(self, policy=None):
        self.policy = policy or RoadPolicy()
        self.tracks = {"left": [], "right": []}
        self.next_id = 1

    def predict(self, motion, now):
        for side, tracks in self.tracks.items():
            self.tracks[side] = [
                t for t in tracks if now - t["seen"] <= self.policy.maximum_prediction_s
            ]
            for track in self.tracks[side]:
                track["path"] = apply_ground_motion(track["path"], motion)
                track["sigma_m"] = math.hypot(track["sigma_m"], motion["sigma_m"])

    def reject_surface(self, surface):
        """Drop transported boundary identities disproved by the current image."""
        if surface is None:
            return 0
        rejected = 0
        for side, tracks in self.tracks.items():
            retained = []
            for track in tracks:
                if surface.contradiction(track["path"], track["sigma_m"])["contradicted"]:
                    rejected += 1
                else:
                    retained.append(track)
            self.tracks[side] = retained
        return rejected

    def update(self, candidates, now):
        for side in ("left", "right"):
            used = set()
            for observation in sorted(
                candidates.get(side, []), key=lambda c: c["score"], reverse=True
            ):
                if observation["confidence"] < self.policy.measurement_quality:
                    continue
                matches = []
                for track in self.tracks[side]:
                    if track["id"] in used:
                        continue
                    agreement = curve_agreement(track["path"], observation["path"])
                    if (
                        agreement is not None
                        and agreement["distance_m"]
                        < self.policy.association_distance_m + track["sigma_m"]
                        and agreement["heading_cos"] > self.policy.association_heading_cos
                    ):
                        matches.append((agreement["distance_m"], track))
                if matches:
                    track = min(matches, key=lambda item: item[0])[1]
                    track["hits"] += 1
                else:
                    track = dict(id=self.next_id, hits=1)
                    self.next_id += 1
                    self.tracks[side].append(track)
                track.update(
                    path=observation["path"].copy(),
                    seen=now,
                    sigma_m=self.policy.observation_sigma_m,
                    quality=observation["confidence"],
                )
                used.add(track["id"])
            # Bound work while preferring recently observed, confirmed tracks.
            self.tracks[side] = sorted(
                self.tracks[side], key=lambda t: (t["seen"], t["hits"]), reverse=True
            )[:4]

    def confirms(self, measurement, fits, now):
        """Reuse current boundary identity only for the selected road geometry.

        Two confirmed sides prove a high-quality paired observation. A lone
        confirmed side may also support a paired road if the resulting centre
        agrees within the two observation errors; an unrelated or weak edge
        cannot confer confirmation on another candidate.
        """
        p = self.policy
        if measurement is None or measurement["confidence"] < p.measurement_quality:
            return False
        sides = {
            "ONE LEFT": ("left",), "ONE RIGHT": ("right",),
            "TWO EDGES": ("left", "right"),
        }.get(measurement.get("mode"), ())
        confirmed = [
            side for side in sides if fits.get(side) is not None and any(
                track["seen"] == now and track["hits"] >= 2
                and np.array_equal(track["path"], fits[side]["path"])
                for track in self.tracks[side]
            )
        ]
        if confirmed and len(confirmed) == len(sides) and measurement["confidence"] >= p.confirmation_quality:
            return True
        for side in confirmed:
            edge = fits[side]
            if p.single_edge_quality_scale * edge["confidence"] < p.confirmation_quality:
                continue
            center = edge.get("center")
            agreement = curve_agreement(center["path"], measurement["path"]) if center else None
            tolerance = math.hypot(
                p.observation_sigma_m / edge["confidence"],
                p.observation_sigma_m / measurement["confidence"],
            )
            if agreement is not None and agreement["distance_m"] <= tolerance and agreement["heading_cos"] > 0.90:
                return True
        return False

    def summary(self, now):
        return {
            side: [
                dict(
                    id=t["id"],
                    confirmed=t["hits"] >= 2,
                    age_s=round(now - t["seen"], 3),
                    sigma_m=round(t["sigma_m"], 4),
                )
                for t in tracks
            ]
            for side, tracks in self.tracks.items()
        }
