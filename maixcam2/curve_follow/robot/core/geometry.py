"""Metric parametric curves. Coordinates: forward x, left y, metres.
No scipy dependency. Local robust quadratic fits use arc length, not y(x).
"""

import math
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_frame_geometry_cache = ContextVar("frame_geometry_cache", default=None)


@contextmanager
def geometry_frame():
    """Share pure geometry work within one frame, never across observations."""
    token = _frame_geometry_cache.set({})
    try:
        yield
    finally:
        _frame_geometry_cache.reset(token)


def frame_cached(function):
    @wraps(function)
    def cached(*paths):
        cache = _frame_geometry_cache.get()
        if cache is None:
            return function(*paths)
        arrays = tuple(np.asarray(points) for points in paths)
        key = (function.__name__, tuple((a.shape, a.dtype.str, a.tobytes()) for a in arrays))
        if key not in cache:
            cache[key] = function(*paths)
        return cache[key]

    return cached


import numpy as np
from robot.core.native_geometry import (
    native_local_fit,
    native_curve_agreement,
    native_chain_fit,
    native_support_projection,
    native_pursuit_selection,
    native_curve_agree_paths,
    native_nearest_samples,
)


def norm2(values, axis=-1):
    """Euclidean length of planar vectors; avoid generic linalg dispatch."""
    return np.hypot(values[..., 0], values[..., 1])


@frame_cached
def arc(path):
    delta = path[1:] - path[:-1]
    return np.concatenate(([0.0], np.cumsum(norm2(delta))))


def resample(path, step=0.015):
    p = np.asarray(path, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        return None
    keep = np.empty(len(p), bool)
    keep[0] = True
    keep[1:] = norm2(p[1:] - p[:-1], axis=1) > 1e-5
    p = p[keep]
    if len(p) < 3:
        return None
    s = arc(p)
    if s[-1] < 0.075:
        return None
    if s[-1] > 2.0:  # Reject looping/noisy contours; bound quadratic fitting cost.
        return None
    q = np.linspace(0, s[-1], min(160, max(10, int(s[-1] / step) + 1)))
    result = np.empty((len(q), 2))
    result[:, 0] = np.interp(q, s, p[:, 0])
    result[:, 1] = np.interp(q, s, p[:, 1])
    return result


def fit_curve(points):
    # Fit arrays are read-only geometric values within a frame. Return a new
    # metadata dictionary so score/side updates cannot alias another view.
    result = _fit_curve_geometry(points)
    return dict(result) if result is not None else None


@frame_cached
def _fit_curve_geometry(points):
    """Smooth an ordered observed chain, with bounded local support only."""
    handled, native_curve = native_chain_fit(points)
    if handled:
        return native_curve
    p = resample(points)
    if p is None:
        return None
    s = arc(p)
    native = native_local_fit(p, s)
    if native is not None:
        fitted, residual = native
    else:
        # Batch the small least-squares systems: a Python loop over dozens of
        # np.linalg.lstsq calls was too slow on the camera CPU.
        delta = (s[None, :] - s[:, None]) / 0.12
        support = abs(delta) <= 1.0
        for i in np.flatnonzero(support.sum(axis=1) < 5):
            support[i, np.argsort(abs(delta[i]))[: min(7, len(p))]] = True
        # Pack only nonzero local support. The omitted samples have exactly zero
        # weight in every IRLS iteration; this preserves the same normal equations
        # while bounding work by the local window rather than full path length.
        count = support.sum(axis=1)
        columns = np.broadcast_to(np.arange(len(p)), support.shape)
        indices = np.sort(np.where(support, columns, len(p)), axis=1)[:, : int(count.max())]
        valid = indices < len(p)
        indices = np.minimum(indices, len(p) - 1)
        delta = np.take_along_axis(delta, indices, axis=1)
        neighbours = p[indices]
        design = np.stack((np.ones_like(delta), delta, delta * delta), axis=-1)
        base = np.exp(-delta * delta) * valid
        weight = base.copy()
        coef = None
        for _ in range(3):
            weighted = design.transpose(0, 2, 1) * weight[:, None, :]
            normal = weighted @ design
            rhs = weighted @ neighbours
            normal += np.eye(3)[None, :, :] * 1e-10
            coef = np.linalg.solve(normal, rhs)
            estimates = design @ coef
            residuals = norm2(estimates - neighbours, axis=2)
            weight = base * np.minimum(1.0, 0.008 / np.maximum(residuals, 1e-6))
        fitted = coef[:, 0, :]
        residual = norm2(fitted - p, axis=1)
    if _quantile_1d(residual, 0.9) > 0.025:
        return None
    fitted = resample(fitted)
    if fitted is None:
        return None
    tangent, curvature = differential(fitted)
    # Reject reversals/cusps and isolated chains that double back on themselves.
    if np.min(np.sum(tangent[1:] * tangent[:-1], axis=1)) < 0.5:
        return None
    return dict(
        path=fitted,
        tangent=tangent,
        curvature=curvature,
        residual=float(_quantile_1d(residual, 0.5)),
        length=float(arc(fitted)[-1]),
    )


@frame_cached
def describe_curve(path):
    """Validate and describe an already fitted path without smoothing it again."""
    path = np.asarray(path, float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 5 or not np.isfinite(path).all():
        return None
    s = arc(path)
    if np.any(np.diff(s) <= 1e-6) or not 0.075 <= s[-1] <= 2.0:
        return None
    tangent, curvature = differential(path)
    if np.min(np.sum(tangent[1:] * tangent[:-1], axis=1)) < 0.5:
        return None
    return dict(path=path, tangent=tangent, curvature=curvature, residual=0.0, length=float(s[-1]))


@frame_cached
def differential(path):
    # The same nonuniform, first-order-endpoint formula as np.gradient,
    # specialized to the two coordinates along arc length. Reuse its weights
    # for both derivatives instead of parsing the generic API twice per fit.
    s = arc(path)
    dx = np.diff(s)
    before, after = dx[:-1], dx[1:]
    a = (-after / (before * (before + after)))[:, None]
    b = ((after - before) / (before * after))[:, None]
    c = (before / (after * (before + after)))[:, None]

    def gradient(values):
        result = np.empty_like(values)
        result[0] = (values[1] - values[0]) / dx[0]
        result[-1] = (values[-1] - values[-2]) / dx[-1]
        result[1:-1] = a * values[:-2] + b * values[1:-1] + c * values[2:]
        return result

    derivative = gradient(path)
    tangent = derivative / np.maximum(norm2(derivative, axis=1)[:, None], 1e-8)
    second = gradient(tangent)
    curvature = tangent[:, 0] * second[:, 1] - tangent[:, 1] * second[:, 0]
    return tangent, curvature


def _quantile_1d(values, fraction):
    """Linear quantile for finite 1-D geometry samples, matching NumPy's default."""
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    selected = np.partition(values, (low, high))
    difference = selected[high] - selected[low]
    weight = index - low
    return (
        selected[low] + difference * weight
        if weight < 0.5
        else selected[high] - difference * (1 - weight)
    )


def inward_center(edge, side, width=0.30):
    result = _inward_center_geometry(edge["path"], edge["tangent"], edge["curvature"], side, width)
    return dict(result) if result is not None else None


@frame_cached
def _inward_center_geometry(path, tangent, curvature, side, width):
    """Offset along ground-plane normals, never a fixed pixel/horizontal shift."""
    edge = dict(path=path, tangent=tangent, curvature=curvature)
    if side not in ("left", "right") or not 0.1 <= width <= 0.8:
        return None
    offset = width / 2 if side == "right" else -width / 2
    # A noisy far endpoint must not invalidate the whole near boundary. Keep only
    # contiguous supported portions with a regular offset (no normal cusp).
    regular = 1 - offset * edge["curvature"] > 0.20
    t = edge["tangent"]
    shifted = edge["path"] + offset * np.column_stack((-t[:, 1], t[:, 0]))
    # A wholly regular boundary has one component; avoid splitting/copying it.
    if regular.all():
        if len(shifted) < 8 or arc(shifted)[-1] < 0.075:
            return None
        return fit_curve(shifted)
    options = []
    for ids in contiguous_runs(regular):
        if len(ids) < 8 or arc(shifted[ids])[-1] < 0.075:
            continue
        curve = fit_curve(shifted[ids])
        if curve is not None:
            options.append(curve)
    if not options:
        return None
    # Prefer support near the vehicle; never extrapolate over discarded portions.
    return min(options, key=lambda c: np.min(norm2(c["path"], axis=1)))


def _nearest_samples(query, reference):
    accelerated = native_nearest_samples(query, reference)
    if accelerated is not None:
        return accelerated
    distance = norm2(query[:, None] - reference[None, :], axis=2)
    index = np.argmin(distance, axis=1)
    return index, distance[np.arange(len(query)), index]


def merge_centers(
    left,
    right,
    left_weight=1.0,
    right_weight=1.0,
    distance_limit=0.055,
    heading_cos=0.75,
    minimum_overlap=0.06,
):
    """Fuse compatible connected support, retaining the near/far single tails.

    Pick a reference by geometry, not left/right identity. Blend only genuine
    ordered overlap; unknown gaps and inconsistent branches are never bridged.
    Weights are fit quality, not calibrated probabilities.
    """
    a, b = left["path"], right["path"]
    # Canonical nearer reference makes fusion invariant to input order.
    if (float(norm2(b[0])), float(b[0, 0]), float(b[0, 1])) < (
        float(norm2(a[0])),
        float(a[0, 0]),
        float(a[0, 1]),
    ):
        left, right = right, left
        a, b = b, a
        left_weight, right_weight = right_weight, left_weight
    j, distance = _nearest_samples(a, b)
    interior = (j > 0) & (j < len(b) - 1)
    # Check all geometrically overlapping samples, not only the subset that
    # happens to be close: a crossing cannot fabricate a valid short overlap.
    tangent_dot = np.sum(left["tangent"] * right["tangent"][j], axis=1)
    ok = interior & (distance < distance_limit) & (tangent_dot > heading_cos)
    runs = [
        ids for ids in contiguous_runs(ok) if len(ids) >= 5 and arc(a[ids])[-1] >= minimum_overlap
    ]
    if not runs:
        return None
    ids = min(runs, key=lambda run: run[0])
    if np.any(np.diff(j[ids]) < 0):
        return None
    total = max(1e-8, left_weight + right_weight)
    merged = a.copy()
    merged[ids] = (a[ids] * left_weight + b[j[ids]] * right_weight) / total
    for endpoint in (0, -1):
        if (
            abs(norm2(a[endpoint]) - norm2(b[endpoint])) < 1e-6
            and norm2(a[endpoint] - b[endpoint]) < distance_limit
            and np.dot(left["tangent"][endpoint], right["tangent"][endpoint]) > heading_cos
        ):
            merged[endpoint] = (a[endpoint] * left_weight + b[endpoint] * right_weight) / total
    # Keep the reference near prefix and its connected supported suffix. If
    # the other curve extends farther, continue from its matched far endpoint.
    last = int(ids[-1])
    other_last = int(j[last])
    arc_a, arc_b = arc(a), arc(b)
    remaining_a = arc_a[-1] - arc_a[last]
    remaining_b = arc_b[-1] - arc_b[other_last]
    # Beyond overlap, conflicting continuations are not a connected union.
    if last < len(a) - 2 and other_last < len(b) - 2:
        overlap = interior & (np.arange(len(a)) > last)
        if np.any(overlap & ((distance >= distance_limit) | (tangent_dot <= heading_cos))):
            # Retain the consistent near component. Do not average or extend
            # through a divergent far branch, and do not discard the near road.
            return fit_curve(merged[: last + 1])
    if remaining_b > remaining_a:
        merged = np.concatenate((merged[: last + 1], b[other_last + 1 :]))
    result = fit_curve(merged)
    if result is not None:
        result["overlap_m"] = float(arc(a[ids])[-1])
    return result


def support_projection(query, reference):
    """Nearest point on an ordered polyline, returning distance and arc position.

    Segment interpolation avoids treating every clamped endpoint as overlapping
    support. Consumers must also check tangent/order and their own error bounds.
    """
    q, r = np.asarray(query, float), np.asarray(reference, float)
    accelerated = native_support_projection(q, r)
    if accelerated is not None:
        return accelerated
    delta = np.diff(r, axis=0)
    length2 = np.maximum(np.sum(delta * delta, axis=1), 1e-12)
    fraction = np.clip(
        np.sum((q[:, None] - r[None, :-1]) * delta[None], axis=2) / length2, 0.0, 1.0
    )
    projected = r[None, :-1] + fraction[:, :, None] * delta[None]
    d2 = np.sum((q[:, None] - projected) ** 2, axis=2)
    index = np.argmin(d2, axis=1)
    rows = np.arange(len(q))
    s = arc(r)
    position = s[index] + fraction[rows, index] * np.sqrt(length2[index])
    return np.sqrt(d2[rows, index]), position


def contiguous_runs(mask):
    m = np.asarray(mask, bool)
    changes = np.diff(np.concatenate(([False], m, [False])).astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [np.arange(a, b) for a, b in zip(starts, ends)]


def nearest_forward_point(path, forward_min_m=0.06):
    """Closest point on the fitted polyline within the forward half-plane.

    Project onto segments rather than choosing a sampling vertex. Clip each
    segment to the forward domain before projection, so a behind-car minimum
    cannot hide an available forward part of this same fitted curve.
    """
    handled, selected = native_pursuit_selection(path, 0.0, forward_min_m, float("inf"))
    if handled:
        return selected
    path = np.asarray(path, float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all():
        return None
    start = path[:-1]
    delta = path[1:] - start
    length2 = np.sum(delta * delta, axis=1)
    dx = delta[:, 0]
    cross = (forward_min_m - start[:, 0]) / np.where(abs(dx) > 1e-12, dx, 1.0)
    lower = np.where(dx > 1e-12, np.maximum(0.0, cross), 0.0)
    upper = np.where(dx < -1e-12, np.minimum(1.0, cross), 1.0)
    allowed = (
        (length2 > 1e-12) & (lower <= upper) & ((abs(dx) > 1e-12) | (start[:, 0] >= forward_min_m))
    )
    if not allowed.any():
        return None
    fraction = np.clip(-np.sum(start * delta, axis=1) / np.maximum(length2, 1e-12), lower, upper)
    points = start + fraction[:, None] * delta
    distance2 = np.sum(points * points, axis=1)
    i = int(np.argmin(np.where(allowed, distance2, np.inf)))
    point = points[i].copy()
    point[0] = max(forward_min_m, point[0])
    position = float(arc(path)[i] + fraction[i] * math.sqrt(length2[i]))
    return point, position


def _preview_far_support(path, selected, lookahead_m, forward_min_m, maximum_target_m):
    """Read direction from existing far support without jumping at its threshold.

    The unseen approach gap supplies a continuous, bounded arc advance. As the
    anchor reaches the ordinary lookahead circle, this advance tends to zero.
    Never cross a forward-domain break or leave the supported segment/range.
    Both native and reference selectors use this postprocessing step.
    """
    if selected is None or selected["mode"] != "FAR_SUPPORT":
        return selected
    budget = min(lookahead_m, max(0.0, selected["distance_m"] - lookahead_m))
    if budget <= 1e-12:
        return selected
    path = np.asarray(path, float)
    positions = arc(path)
    position = float(selected["position_m"])
    start = np.asarray(selected["point"], float)
    index = max(0, min(len(path) - 2, int(np.searchsorted(positions, position, side="right") - 1)))
    for end in path[index + 1 :]:
        end = end.copy()
        last = False
        if end[0] < forward_min_m:
            fraction = (forward_min_m - start[0]) / (end[0] - start[0])
            end = start + np.clip(fraction, 0.0, 1.0) * (end - start)
            last = True
        delta = end - start
        length = float(norm2(delta))
        if length <= 1e-12:
            if last:
                break
            continue
        fraction = min(1.0, budget / length)
        candidate = start + fraction * delta
        if float(candidate @ candidate) > maximum_target_m**2:
            a = float(delta @ delta)
            b = float(start @ delta)
            disc = max(0.0, b * b - a * (float(start @ start) - maximum_target_m**2))
            fraction = min(fraction, max(0.0, (-b + math.sqrt(disc)) / a))
            last = True
            candidate = start + fraction * delta
        used = fraction * length
        position += used
        budget -= used
        start = candidate
        if last or fraction < 1.0 or budget <= 1e-12:
            break
    distance = float(norm2(start))
    return dict(
        point=start.copy(),
        position_m=position,
        distance_m=distance,
        mode="FAR_SUPPORT",
        speed_scale=min(selected["speed_scale"], lookahead_m / max(distance, 1e-12)),
    )


def pursuit_point(path, lookahead_m, forward_min_m=0.06, maximum_target_m=0.65):
    """First physical lookahead-circle intersection after the nearest anchor.

    Only walk the same continuous forward part of the supported polyline.
    Far-only support reads ahead on that same supported branch at reduced
    speed; short support uses its last forward point at reduced speed. Neither case extends a fit
    beyond its actual support or jumps to a later branch of a looping curve.
    """
    if not math.isfinite(lookahead_m) or lookahead_m <= 0:
        return None
    handled, selected = native_pursuit_selection(path, lookahead_m, forward_min_m, maximum_target_m)
    if handled:
        return _preview_far_support(path, selected, lookahead_m, forward_min_m, maximum_target_m)
    path = np.asarray(path, float)
    nearest = nearest_forward_point(path, forward_min_m)
    if nearest is None:
        return None
    anchor, at = nearest
    distance = float(norm2(anchor))
    positions = arc(path)
    if distance > maximum_target_m:
        return None
    if distance >= lookahead_m:
        selected = dict(
            point=anchor,
            position_m=at,
            distance_m=distance,
            mode="FAR_SUPPORT",
            speed_scale=min(1.0, lookahead_m / distance),
        )
        return _preview_far_support(path, selected, lookahead_m, forward_min_m, maximum_target_m)
    index = min(len(path) - 2, int(np.searchsorted(positions, at, side="right") - 1))
    start = anchor
    position = at
    for end in path[index + 1 :]:
        delta = end - start
        length = float(norm2(delta))
        last = False
        if end[0] < forward_min_m:
            fraction = (forward_min_m - start[0]) / delta[0]
            end = start + np.clip(fraction, 0.0, 1.0) * delta
            delta = end - start
            length = float(norm2(delta))
            last = True
        a = float(delta @ delta)
        if a > 1e-14:
            b = float(start @ delta)
            c = float(start @ start) - lookahead_m**2
            discriminant = b * b - a * c
            if discriminant >= -1e-12:
                fraction = (-b + math.sqrt(max(0.0, discriminant))) / a
                if -1e-10 <= fraction <= 1.0 + 1e-10:
                    fraction = float(np.clip(fraction, 0.0, 1.0))
                    point = start + fraction * delta
                    return dict(
                        point=point,
                        position_m=position + fraction * length,
                        distance_m=float(norm2(point)),
                        mode="LOOKAHEAD",
                        speed_scale=1.0,
                    )
        position += length
        start = end
        if last:
            break
    distance = float(norm2(start))
    if distance < forward_min_m or distance > maximum_target_m:
        return None
    return dict(
        point=start.copy(),
        position_m=position,
        distance_m=distance,
        mode="SHORT_SUPPORT",
        speed_scale=min(1.0, distance / lookahead_m),
    )


def choose_target(
    path,
    forward_min_m=0.06,
    maximum_target_m=0.65,
    reference_origin_m=(0.0, 0.0),
    observation_sigma_m=0.005,
    maximum_reference_sigma_m=0.05,
    curve_analysis_window_m=0.30,
):
    """Nearest supported road reference, NOT the final pursuit steering goal.

    The input already contains measured geometry and motion-compensated history
    selected by LocalRoadModel. Unseen-road prediction transports that known
    curve; a stable fit to a far straight must not invent an unobserved bend.
    """
    origin = np.asarray(reference_origin_m, float)
    if (
        origin.shape != (2,)
        or not np.isfinite(origin).all()
        or not all(
            math.isfinite(v) and v > 0
            for v in (
                forward_min_m,
                maximum_target_m,
                observation_sigma_m,
                maximum_reference_sigma_m,
                curve_analysis_window_m,
            )
        )
    ):
        raise ValueError("Invalid nearest-curve geometry")
    observed = np.asarray(path, float)
    if (
        observed.ndim != 2
        or observed.shape[1] != 2
        or len(observed) < 5
        or not np.isfinite(observed).all()
        or observation_sigma_m > maximum_reference_sigma_m
    ):
        return None
    curve = observed - origin
    s = arc(curve)
    if np.any(np.diff(s) <= 1e-6):
        return None
    selected = nearest_forward_point(curve, forward_min_m)
    if selected is None:
        return None
    point, at = selected
    distance = float(norm2(point))
    if distance > maximum_target_m:
        return None
    tangent, curvature = differential(curve)
    index = min(len(curve) - 1, int(np.searchsorted(s, at)))
    local = (s >= at) & (s <= at + curve_analysis_window_m)
    if not local.any():
        local[index] = True
    return dict(
        point=point,
        position_m=at,
        distance_m=distance,
        curvature=float(_quantile_1d(abs(curvature[local]), 0.75)),
        path_heading_rad=float(math.atan2(tangent[index, 1], tangent[index, 0])),
        near_support_m=float(norm2(observed).min()),
        reliable_length_m=float(s[-1]),
        reference_origin_m=origin.tolist(),
        reference_path=curve,
        reference_sigma_m=observation_sigma_m,
        target_selection="NEAREST_CURVE",
        target_speed_scale=min(1.0, curve_analysis_window_m / distance),
    )


def contained_observation(old, new, distance_limit, heading_cos, minimum_overlap_m):
    """Recognize a short observation entirely inside an existing road segment.

    Compare continuous polyline positions rather than counting nearest sample
    indices. Every new point must agree in distance and direction, with ordered
    projections spanning the unchanged minimum physical overlap.
    """
    a, b = np.asarray(old, float), np.asarray(new, float)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1:] != (2,)
        or b.shape[1:] != (2,)
        or min(len(a), len(b)) < 5
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        return None
    sa, sb = arc(a), arc(b)
    if np.any(np.diff(sa) <= 1e-6) or np.any(np.diff(sb) <= 1e-6):
        return None
    distances, positions = support_projection(b, a)
    if positions[0] <= 1e-6 or positions[-1] >= sa[-1] - 1e-6:
        return None
    if np.any(np.diff(positions) <= 1e-6) or positions[-1] - positions[0] < minimum_overlap_m:
        return None
    ta = np.diff(a, axis=0) / np.diff(sa)[:, None]
    tb = np.diff(b, axis=0) / np.diff(sb)[:, None]
    ids = np.clip(np.searchsorted(sa, positions[:-1], side="right") - 1, 0, len(ta) - 1)
    cosines = np.sum(tb * ta[ids], axis=1)
    if float(distances.max()) > distance_limit or float(cosines.min()) < heading_cos:
        return None
    return dict(
        distance_m=float(distances.max()),
        heading_cos=float(cosines.min()),
        overlap_m=float(positions[-1] - positions[0]),
        start_m=float(positions[0]),
        end_m=float(positions[-1]),
    )


def curve_handoff(old, new, distance_limit, heading_cos, end_window_m):
    """Prove an ordered end-to-start handoff with short/no sampled overlap.

    This is not a gap-filling fit: it returns only geometry evidence. Both
    endpoints must lie within observation uncertainty of the other support,
    ordered overlap must agree throughout, and the candidate must extend the
    old exit. Crossing, reversal and competing overlapping branches fail.
    """
    a, b = np.asarray(old, float), np.asarray(new, float)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1:] != (2,)
        or b.shape[1:] != (2,)
        or min(len(a), len(b)) < 5
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        return None
    sa, sb = arc(a), arc(b)
    if np.any(np.diff(sa) <= 1e-6) or np.any(np.diff(sb) <= 1e-6):
        return None
    da, pa = support_projection(b[:1], a)
    db, pb = support_projection(a[-1:], b)
    if max(da[0], db[0]) > distance_limit:
        return None
    if sa[-1] - pa[0] > end_window_m or pb[0] > end_window_m:
        return None
    ta = np.diff(a, axis=0) / np.diff(sa)[:, None]
    tb = np.diff(b, axis=0) / np.diff(sb)[:, None]
    if float(ta[-1] @ tb[0]) < heading_cos:
        return None
    if float((b[-1] - a[-1]) @ ta[-1]) <= distance_limit:
        return None
    # Every sample in the shared end region must agree; no percentile may
    # hide an actual competing branch. Segment projection avoids making a
    # six-centimetre overlap require extra samples beyond both endpoints.
    worst_distance = 0.0
    worst_heading = 1.0
    for query, reference, sq, sr, tq, tr, mask in (
        (a, b, sa, sb, ta, tb, sa >= pa[0] - 1e-6),
        (b, a, sb, sa, tb, ta, sb <= pb[0] + 1e-6),
    ):
        ids = np.flatnonzero(mask)
        if not len(ids):
            return None
        distances, positions = support_projection(query[ids], reference)
        if np.any(np.diff(positions) < -1e-6):
            return None
        qi = np.minimum(ids, len(tq) - 1)
        ri = np.clip(np.searchsorted(sr, positions, side="right") - 1, 0, len(tr) - 1)
        cosines = np.sum(tq[qi] * tr[ri], axis=1)
        worst_distance = max(worst_distance, float(distances.max()))
        worst_heading = min(worst_heading, float(cosines.min()))
    if worst_distance > distance_limit or worst_heading < heading_cos:
        return None
    return dict(
        endpoint_error_m=float(max(da[0], db[0])),
        distance_m=worst_distance,
        heading_cos=worst_heading,
        old_exit_overlap_m=float(sa[-1] - pa[0]),
        new_entry_overlap_m=float(pb[0]),
        distance_limit_m=float(distance_limit),
    )


def curve_distance(old, new):
    a, b = np.asarray(old), np.asarray(new)
    nearest = native_nearest_samples(b, a)
    if nearest is None:
        # Preserve the original subtraction order for non-float64 inputs too.
        matrix = norm2(a[:, None] - b[None, :], axis=2)
        j = np.argmin(matrix, axis=0)
        dist = matrix[j, np.arange(len(b))]
    else:
        j, dist = nearest
    valid = (j > 0) & (j < len(a) - 1)
    if valid.sum() < 5:
        return None
    return float(_quantile_1d(dist[valid], 0.8))


@frame_cached
def curve_agreement(old, new):
    """Compare overlapping road geometry, independent of adaptive target choice.

    Ignore clamped endpoints in both directions and require at least 6cm of
    overlap. Distance and ordered tangents reject shifted/crossing/reversed
    paths even if their chosen pursuit points happen to coincide.
    """
    handled, result = native_curve_agree_paths(old, new)
    if handled:
        return result
    a, b = np.asarray(old, float), np.asarray(new, float)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != 2
        or b.shape[1] != 2
        or min(len(a), len(b)) < 5
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        return None
    sa, sb = arc(a), arc(b)
    if np.any(np.diff(sa) <= 1e-6) or np.any(np.diff(sb) <= 1e-6):
        return None
    ta, _ = differential(a)
    tb, _ = differential(b)
    handled, result = native_curve_agreement(a, b, ta, tb, sa, sb)
    if handled:
        return result
    dx = a[:, 0, None] - b[None, :, 0]
    dy = a[:, 1, None] - b[None, :, 1]
    distance = dx * dx + dy * dy
    measures = []
    for matrix, t1, t2, support in ((distance, ta, tb, sa), (distance.T, tb, ta, sb)):
        j = np.argmin(matrix, axis=1)
        ids = np.flatnonzero((j > 0) & (j < len(t2) - 1))
        if len(ids) < 5:
            return None
        overlap = float(support[ids[-1]] - support[ids[0]])
        if overlap < 0.06:
            return None
        measures.append(
            (
                float(_quantile_1d(np.sqrt(matrix[ids, j[ids]]), 0.8)),
                float(_quantile_1d(np.sum(t1[ids] * t2[j[ids]], axis=1), 0.2)),
                overlap,
            )
        )
    return dict(
        distance_m=max(m[0] for m in measures),
        heading_cos=min(m[1] for m in measures),
        overlap_m=min(m[2] for m in measures),
    )
