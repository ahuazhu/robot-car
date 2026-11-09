"""Single-frame white-boundary detection and measured road geometry."""

import math
import cv2
import numpy as np
from robot.core.geometry import (
    arc,
    fit_curve,
    inward_center,
    merge_centers,
    curve_distance,
    contiguous_runs,
    curve_agreement,
)
from robot.core.native_geometry import native_boundary_chains
from robot.core.road_policy import RoadPolicy
from robot.perception.surface import RoadSurfaceEvidence
from robot.perception.photometry import road_regions


class CurvePixels:
    def __init__(self, cal, policy=None):
        self.policy = policy or RoadPolicy.from_config(cal)
        self.exterior_distances = np.array([0.025, 0.035, 0.045])
        self.boundary_prior = {"left": [], "right": []}
        self.H = np.asarray(cal["H_normalized"], float)
        self.Hinv = np.linalg.inv(self.H)
        self.width = float(cal.get("road_width_m", 0.30))
        self.reference_origin = np.asarray(cal.get("control_frame", {}).get("origin_m", [0., 0.]), float)
        if self.reference_origin.shape != (2,) or not np.isfinite(self.reference_origin).all():
            raise ValueError("Invalid road surface reference origin")
        # Photometric calibration, shared by every frame and curve source.
        self.white_value_levels = tuple(cal.get("white_value_levels", (210, 180)))
        if not self.white_value_levels or any(
            not 0 <= value <= 255 for value in self.white_value_levels
        ):
            raise ValueError("Invalid white boundary brightness levels")
        self.step = float(np.clip(cal.get("vision_grid_step_m", 0.005), 0.005, 0.01))
        self.xs = np.arange(0.06, 0.805, self.step)
        self.ys = np.arange(0.65, -0.655, -self.step)
        xx, yy = np.meshgrid(self.xs, self.ys, indexing="ij")
        p = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ self.Hinv.T
        u = p[:, :, 0] / p[:, :, 2] * 320
        v = p[:, :, 1] / p[:, :, 2] * 240
        # ROI belongs to the calibrated camera pose; fixed old crops truncate
        # otherwise usable boundaries after a mount-angle change.
        roi_top = int(np.clip(cal.get("vision_roi_top_px", 100), 0, 220))
        self.visible = (
            (u >= 2) & (u < 318) & (v >= roi_top) & (v < 238) & np.isfinite(u) & np.isfinite(v)
        )
        self.visible_u8 = self.visible.astype(np.uint8)
        rows, cols = np.where(self.visible)
        self.visible_near_m = float(np.hypot(self.xs[rows], self.ys[cols]).min())
        self.mapx = u.astype(np.float32)
        self.mapy = v.astype(np.float32)
        self.photometric_window = max(3, int(self.width / self.step / 4) | 1)
        radius = int(math.ceil(float(self.exterior_distances.max()) / self.step)) + 1
        self.exterior_kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        self.minimum_samples = max(5, int(math.ceil(self.policy.minimum_segment_m / self.step)))

    def sample(self, mask, points):
        row = np.rint((points[:, 0] - self.xs[0]) / self.step).astype(int)
        col = np.rint((self.ys[0] - points[:, 1]) / self.step).astype(int)
        valid = (row >= 0) & (row < len(self.xs)) & (col >= 0) & (col < len(self.ys))
        r = np.clip(row, 0, len(self.xs) - 1)
        c = np.clip(col, 0, len(self.ys) - 1)
        valid &= self.visible[r, c]
        return mask[r, c].astype(float), valid

    def prepare(self, bgr):
        bird = cv2.remap(
            bgr, self.mapx, self.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        h, s, v = cv2.split(cv2.cvtColor(bird, cv2.COLOR_BGR2HSV))
        # A quarter-lane window spans paint and its context, so a white stripe
        # in shadow can compete with a specular road highlight.
        green, contrast, raw_gray, gray = road_regions(
            h, s, v, self.visible_u8, self.photometric_window
        )
        # Necessary condition for the existing normal-probe exterior test.
        # Reject only entire contours with no possible green sample, never
        # clip contours or require green at every point of a valid chain.
        vicinity = cv2.dilate(green.view(np.uint8), self.exterior_kernel)
        return (
            bird,
            s,
            v,
            raw_gray,
            green,
            contrast,
            gray,
            vicinity,
        )

    def contour_chains_reference(self, pixels, region_codes, gray, minimum_samples):
        chains = []
        p = np.column_stack((self.xs[pixels[:, 1]], self.ys[pixels[:, 0]]))
        tangent = np.roll(p, -4, axis=0) - np.roll(p, 4, axis=0)
        norm = np.linalg.norm(tangent, axis=1)
        t = tangent / np.maximum(norm[:, None], 1e-8)
        normal = np.column_stack((-t[:, 1], t[:, 0]))
        # Sample both sides, all three distances and both region masks in
        # one indexed pass; repeated tiny numpy calls dominate on ARM.
        offsets = (
            np.array([1.0, -1.0])[:, None, None, None]
            * self.exterior_distances[None, :, None, None]
        )
        samples = p[None, None, :, :] + offsets * normal[None, None, :, :]
        # White paint counts as road only with green on the opposite
        # side of the same normal sample. Without that exterior evidence,
        # preserve the original gray-road test. Keep gray-only immediate
        # facing checks below so the outer stripe edge is still rejected.
        codes, checks = self.sample(region_codes, samples.reshape(-1, 2))
        codes = codes.astype(np.uint8).reshape(2, 3, len(p))
        outside_bits = (codes & 2) > 0
        roads = (((codes & 1) > 0) | (((codes & 4) > 0) & outside_bits[::-1])).mean(axis=1)
        outsides = outside_bits.mean(axis=1)
        visibility = checks.reshape(2, 3, len(p)).all(axis=1)
        # Gray on road side and contrasting exterior: stops/crosswalks with gray
        # on BOTH sides are deliberately not interpreted as lane boundaries.
        evidence = roads[0] - roads[1] + 0.7 * (outsides[1] - outsides[0])
        for inward_sign in (1, -1):
            road_index = 0 if inward_sign == 1 else 1
            good = (inward_sign * evidence > 0.65) & (roads[road_index] >= 0.66)
            good &= visibility[0] & visibility[1] & (norm > 0.012)
            # A cyclic contour can have a valid chain across its array seam.
            if good.all():
                continue
            start = int(np.flatnonzero(~good)[0])
            pp = np.roll(p, -start, axis=0)
            nn = np.roll(normal * inward_sign, -start, axis=0)
            gg = np.roll(good, -start)
            for ids in contiguous_runs(gg):
                if len(ids) < minimum_samples:
                    continue
                chain, inward = pp[ids], nn[ids]
                # Require exterior evidence over a majority of this whole
                # chain, while allowing paint joins and short occlusions.
                # A reflected stripe entirely inside the road cannot pass.
                exterior = np.roll(outsides[1 - road_index], -start)[ids] > 0
                if float(exterior.mean()) < 0.5:
                    continue
                if arc(chain)[-1] < self.policy.minimum_segment_m:
                    continue
                if np.linalg.norm(chain[-1]) < np.linalg.norm(chain[0]):
                    chain = chain[::-1]
                    inward = inward[::-1]
                # Select the road-facing edge of the white stripe, not its
                # outer edge (otherwise half-width inference is biased 2cm).
                immediate, visible = self.sample(gray, chain + 0.012 * inward)
                facing = float(np.mean(immediate[visible])) if visible.any() else 0.0
                if facing < 0.45:
                    continue
                chains.append((chain, inward, facing))
        return chains

    def extract(self, bgr, prior=None, white_value_min=180, prepared=None, normalized=False):
        bird, s, raw_value, gray, green, contrast, normalized_gray, vicinity = (
            self.prepare(bgr) if prepared is None else prepared
        )
        v = contrast if normalized else raw_value
        if normalized:
            gray = normalized_gray
        white = (s <= 105) & (v >= white_value_min) & self.visible
        contours, _ = cv2.findContours(
            white.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        fits = {"left": [], "right": []}
        region_codes = (
            gray.astype(np.uint8) + 2 * green.astype(np.uint8) + 4 * white.astype(np.uint8)
        )
        gray_u8 = gray.astype(np.uint8)
        minimum_samples = self.minimum_samples
        for contour in contours:
            pixels = contour[:, 0, :]
            if len(pixels) < 2 * minimum_samples or not vicinity[pixels[:, 1], pixels[:, 0]].any():
                continue
            chains = native_boundary_chains(
                pixels,
                region_codes,
                gray_u8,
                self.visible_u8,
                self.xs,
                self.ys,
                self.step,
                self.exterior_distances,
                minimum_samples,
                self.policy.minimum_segment_m,
            )
            if chains is None:
                chains = self.contour_chains_reference(pixels, region_codes, gray, minimum_samples)
            for chain, inward, facing in chains:
                edge = fit_curve(chain)
                if edge is None or edge["length"] < self.policy.minimum_segment_m:
                    continue
                direction = chain[-1] - chain[0]
                left_normal = np.array([-direction[1], direction[0]])
                side = "right" if np.dot(left_normal, np.mean(inward, axis=0)) > 0 else "left"
                center = inward_center(edge, side, self.width)
                if center is None:
                    continue
                confidence = min(1.0, edge["length"] / 0.18) * (
                    1 - min(0.3, edge["residual"] / 0.025)
                )
                score = confidence * (0.5 + 0.5 * facing)
                # Observation relevance is continuous in distance; a strong
                # far fragment must not eclipse a usable near boundary.
                near = float(np.linalg.norm(center["path"], axis=1).min())
                score *= self.width / (self.width + near)
                if prior is not None:
                    diff = curve_distance(prior, center["path"])
                    if diff is not None:
                        score *= math.exp(-diff / 0.12)
                edge.update(side=side, center=center, confidence=confidence, score=score)
                # Associate only with this side's established boundary tracks.
                # Unmatched observations remain candidates for reacquisition.
                agreements = [
                    curve_agreement(track["path"], edge["path"])
                    for track in self.boundary_prior[side]
                ]
                agreements = [
                    a
                    for a in agreements
                    if a is not None and a["heading_cos"] > self.policy.association_heading_cos
                ]
                if agreements:
                    score *= math.exp(-min(a["distance_m"] for a in agreements) / self.width)
                    edge["score"] = score
                fits[side].append(edge)
        # White strips have two contours. Choose a single supported chain per side;
        # matching uses normal offsets and tangent agreement, not image left/right.
        selected = {s: max(fs, key=lambda f: f["score"]) if fs else None for s, fs in fits.items()}
        return selected, dict(
            bird=bird,
            mask=white,
            saturation=s,
            value=raw_value,
            contrast_value=contrast,
            exposure_normalized=normalized,
            fits=selected,
            candidates=fits,
            counts={s: len(fs) for s, fs in fits.items()},
        )

    def measure(self, bgr, prior=None):
        # Contrast levels are calibration data, not scene-dependent driving
        # modes. Every level uses the same boundary and curve validity checks.
        # Cache the ground projection and region masks once for this frame.
        prepared = self.prepare(bgr)
        surface = RoadSurfaceEvidence(
            self, prepared[4], prepared[1], road_value=prepared[5], reference_origin=self.reference_origin
        )
        best = None
        best_score = -1.0
        first_failure = None
        # Coverage is a sensor/road observation property, not target policy.
        visible_near = self.visible_near_m
        for passes, (normalized, cutoff) in enumerate(
            [(n, c) for n in (True, False) for c in self.white_value_levels], 1
        ):
            fits, details = self.extract(
                bgr, prior, white_value_min=cutoff, prepared=prepared, normalized=normalized
            )
            details["surface"] = surface
            details["white_value_min"] = cutoff
            details["photometric_passes"] = passes
            measurement, details = self.measure_fits(fits, details, prior)
            if first_failure is None:
                first_failure = (measurement, details)
            if measurement is None or measurement["confidence"] < self.policy.measurement_quality:
                continue
            near = float(np.linalg.norm(measurement["path"], axis=1).min())
            score = measurement["confidence"] * self.width / (self.width + near)
            if score > best_score:
                best, best_score = (measurement, details), score
            # A far-only strict mask may hide a continuous darker near edge.
            # Try the other calibrated levels before declaring it the road.
            if near <= max(visible_near, self.policy.observation_search_near_m):
                return measurement, details
        selected = best if best is not None else first_failure
        selected[1]["photometric_passes"] = passes
        return selected

    def measure_fits(self, fits, details, prior=None):
        left, right = fits["left"], fits["right"]
        curve = None
        mode = "LOST"
        confidence = 0.0
        reason = "NO EDGE"
        if left is not None and right is not None:
            curve = merge_centers(
                left["center"],
                right["center"],
                left["confidence"],
                right["confidence"],
                self.policy.association_distance_m,
                self.policy.association_heading_cos,
                self.policy.minimum_overlap_m,
            )
            reason = "EDGES DISAGREE"
            if curve is None:
                # Do not commit independently to one contour on each side.
                # Compare a bounded set of alternatives before rejecting a pair.
                options = []
                for lf in sorted(
                    details.get("candidates", {}).get("left", []),
                    key=lambda f: f["score"],
                    reverse=True,
                )[:3]:
                    for rf in sorted(
                        details.get("candidates", {}).get("right", []),
                        key=lambda f: f["score"],
                        reverse=True,
                    )[:3]:
                        if lf is left and rf is right:
                            continue
                        merged = merge_centers(
                            lf["center"],
                            rf["center"],
                            lf["confidence"],
                            rf["confidence"],
                            self.policy.association_distance_m,
                            self.policy.association_heading_cos,
                            self.policy.minimum_overlap_m,
                        )
                        if merged is not None:
                            options.append((lf["score"] + rf["score"], merged, lf, rf))
                if options:
                    _, curve, left, right = max(options, key=lambda option: option[0])
                    details["fits"] = dict(left=left, right=right)
            if curve is not None:
                mode = "TWO EDGES"
                confidence = min(left["confidence"], right["confidence"])
            else:
                # Apply the existing tracker acceptance threshold to each
                # independent single-edge measurement. An individually unusable
                # fragment must not veto the other, usable boundary. If both are
                # usable but contradictory, retain EDGES DISAGREE rejection.
                eligible = [
                    f
                    for f in (left, right)
                    if self.policy.single_edge_quality_scale * f["confidence"]
                    >= self.policy.measurement_quality
                ]
                if len(eligible) == 1:
                    f = eligible[0]
                    curve = f["center"]
                    mode = "ONE " + f["side"].upper()
                    confidence = self.policy.single_edge_quality_scale * f["confidence"]
                    details["boundary_fallback"] = "IGNORED INELIGIBLE " + (
                        "RIGHT" if f is left else "LEFT"
                    )
                elif len(eligible) == 2 and prior is not None:
                    # A far conflicting fragment cannot veto a uniquely
                    # associated near boundary. If both/neither agree, retain
                    # the conflict; never choose a side arbitrarily.
                    associated = []
                    for f in eligible:
                        agreement = curve_agreement(prior, f["center"]["path"])
                        if (
                            agreement is not None
                            and agreement["distance_m"] < self.policy.association_distance_m
                            and agreement["heading_cos"] > self.policy.association_heading_cos
                        ):
                            associated.append(f)
                    if len(associated) == 1:
                        f = associated[0]
                        curve = f["center"]
                        mode = "ONE " + f["side"].upper()
                        confidence = self.policy.single_edge_quality_scale * f["confidence"]
                        details["boundary_fallback"] = "ASSOCIATED " + f["side"].upper()
                if curve is None and len(eligible) == 2 and details.get("surface") is not None:
                    hypotheses = []
                    for f in eligible:
                        normal = np.c_[-f["tangent"][:, 1], f["tangent"][:, 0]]
                        hypotheses.append((f["path"], normal if f["side"] == "right" else -normal))
                    sigma = self.policy.observation_sigma_m / min(f["confidence"] for f in eligible)
                    proof = details["surface"].nearer_separated_component(hypotheses, sigma)
                    if proof is not None:
                        f = eligible[proof["index"]]
                        if not details["surface"].contradiction(f["center"]["path"], sigma)["contradicted"]:
                            curve = f["center"]
                            mode = "ONE " + f["side"].upper()
                            confidence = self.policy.single_edge_quality_scale * f["confidence"]
                            details["fits"] = {side: f if side == f["side"] else None for side in ("left", "right")}
                            details["boundary_fallback"] = "NEAREST CONNECTED ROAD"
                            details["road_component_selection"] = dict(proof, selected_side=f["side"])
        elif left is not None or right is not None:
            f = left if left is not None else right
            curve = f["center"]
            mode = "ONE " + f["side"].upper()
            confidence = self.policy.single_edge_quality_scale * f["confidence"]
        details["measurement_reason"] = reason if curve is None else ""
        if curve is None:
            return None, details
        return dict(
            path=curve["path"],
            mode=mode,
            confidence=confidence,
            boundaries={
                side: [
                    dict(path=f["path"], confidence=f["confidence"])
                    for f in details.get("candidates", {}).get(side, [])
                ]
                for side in ("left", "right")
            },
        ), details
