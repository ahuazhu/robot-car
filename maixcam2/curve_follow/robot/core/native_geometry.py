"""Optional compiled curve geometry and single-frame photometric arithmetic.

Resampling, support policy and validation remain in geometry.py; only the
validated numerical association and local fit run in C. The installer
compiles the small C99 source on the target architecture before launching.
Without a library the same Python reference implementation remains available.
"""

import ctypes
import os
from pathlib import Path
import numpy as np

_library = None
_load_error = None
_agreement_function = None
_chain_function = None
_projection_function = None
_boundary_function = None
_pursuit_function = None
_paths_agreement_function = None
_nearest_samples_function = None
_road_regions_function = None
_path = Path(
    os.environ.get(
        "CURVE_NATIVE_LIBRARY", str(Path(__file__).resolve().parents[2] / "curve_fit_kernel.so")
    )
)
try:
    _library = ctypes.CDLL(str(_path))
    # Wrappers below own contiguous typed buffers and shape checks. Passing
    # their addresses avoids repeating NumPy's ndpointer/flags dispatch for
    # every small C argument; locals keep buffers alive through each call.
    _array = ctypes.c_void_p
    _library.curve_fit_local.argtypes = [_array, _array, ctypes.c_int, _array, _array]
    _library.curve_fit_local.restype = ctypes.c_int
    try:
        _agreement_function = _library.curve_agree_local
        _agreement_function.argtypes = [_array] * 6 + [ctypes.c_int, ctypes.c_int, _array]
        _agreement_function.restype = ctypes.c_int
    except AttributeError:
        pass
    try:
        _chain_function = _library.curve_fit_chain
        _chain_function.argtypes = [_array, ctypes.c_int, _array, _array]
        _chain_function.restype = ctypes.c_int
    except AttributeError:
        pass
    try:
        _projection_function = _library.curve_project_polyline
        _projection_function.argtypes = [_array, ctypes.c_int, _array, ctypes.c_int, _array, _array]
        _projection_function.restype = ctypes.c_int
    except AttributeError:
        pass
    try:
        _boundary_function = _library.curve_boundary_chains
        _i32 = ctypes.c_void_p
        _u8 = ctypes.c_void_p
        _boundary_function.argtypes = [
            _i32,
            ctypes.c_int,
            _u8,
            _u8,
            _u8,
            ctypes.c_int,
            ctypes.c_int,
            _array,
            _array,
            ctypes.c_double,
            _array,
            ctypes.c_int,
            ctypes.c_double,
            _array,
            _array,
        ]
        _boundary_function.restype = ctypes.c_int
    except AttributeError:
        pass
    try:
        _pursuit_function = _library.curve_select_pursuit
        _pursuit_function.argtypes = [
            _array,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            _array,
        ]
        _pursuit_function.restype = ctypes.c_int
    except AttributeError:
        pass
    try:
        _paths_agreement_function = _library.curve_agree_paths
        _paths_agreement_function.argtypes = [_array, ctypes.c_int, _array, ctypes.c_int, _array]
        _paths_agreement_function.restype = ctypes.c_int
    except AttributeError:
        pass
    try:
        _nearest_samples_function = _library.curve_nearest_samples
        _nearest_samples_function.argtypes = [
            _array,
            ctypes.c_int,
            _array,
            ctypes.c_int,
            _array,
            _array,
        ]
        _nearest_samples_function.restype = ctypes.c_int
    except AttributeError:
        pass
    try:
        _road_regions_function = _library.curve_road_regions
        _road_regions_function.argtypes = [_array] * 6 + [ctypes.c_int, _array]
        _road_regions_function.restype = ctypes.c_int
    except AttributeError:
        pass
except (OSError, AttributeError) as exc:
    _load_error = str(exc)
    _library = None


def native_fit_status():
    return dict(
        enabled=_library is not None,
        agreement_enabled=_agreement_function is not None,
        chain_enabled=_chain_function is not None,
        projection_enabled=_projection_function is not None,
        boundary_enabled=_boundary_function is not None,
        road_regions_enabled=_road_regions_function is not None,
        pursuit_enabled=_pursuit_function is not None,
        paths_agreement_enabled=_paths_agreement_function is not None,
        nearest_samples_enabled=_nearest_samples_function is not None,
        path=str(_path),
        error=_load_error,
    )


def native_local_fit(points, distances):
    if _library is None:
        return None
    points = np.ascontiguousarray(points, dtype=np.float64)
    distances = np.ascontiguousarray(distances, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or distances.shape != (len(points),):
        raise ValueError("Invalid local quadratic input")
    fitted = np.empty_like(points)
    residual = np.empty(len(points), dtype=np.float64)
    if not _library.curve_fit_local(
        points.ctypes.data,
        distances.ctypes.data,
        len(points),
        fitted.ctypes.data,
        residual.ctypes.data,
    ):
        return None
    return fitted, residual


def native_curve_agreement(a, b, ta, tb, sa, sb):
    if _agreement_function is None:
        return False, None
    arrays = [np.ascontiguousarray(v, dtype=np.float64) for v in (a, b, ta, tb, sa, sb)]
    if any(v.shape != (len(a), 2) for v in (arrays[0], arrays[2])) or any(
        v.shape != (len(b), 2) for v in (arrays[1], arrays[3])
    ):
        raise ValueError("Invalid agreement geometry")
    if arrays[4].shape != (len(a),) or arrays[5].shape != (len(b),):
        raise ValueError("Invalid agreement arc lengths")
    result = np.empty(3, dtype=np.float64)
    code = _agreement_function(*(v.ctypes.data for v in arrays), len(a), len(b), result.ctypes.data)
    if code == 0:
        return False, None
    if code == 2:
        return True, None
    return True, dict(
        distance_m=float(result[0]), heading_cos=float(result[1]), overlap_m=float(result[2])
    )


def native_chain_fit(points):
    if _chain_function is None:
        return False, None
    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return True, None
    out = np.empty((160, 5), dtype=np.float64)
    stats = np.empty(4, dtype=np.float64)
    n = _chain_function(points.ctypes.data, len(points), out.ctypes.data, stats.ctypes.data)
    if n == 0:
        return False, None
    if n < 0 or stats[0] > 0.025 or stats[3] < 0.5:
        return True, None
    return True, dict(
        path=out[:n, :2].copy(),
        tangent=out[:n, 2:4].copy(),
        curvature=out[:n, 4].copy(),
        residual=float(stats[1]),
        length=float(stats[2]),
    )


def native_support_projection(query, reference):
    if _projection_function is None:
        return None
    q = np.ascontiguousarray(query, dtype=np.float64)
    r = np.ascontiguousarray(reference, dtype=np.float64)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != 2 or r.shape[1] != 2:
        return None
    distance = np.empty(len(q), dtype=np.float64)
    position = np.empty_like(distance)
    if not _projection_function(
        q.ctypes.data, len(q), r.ctypes.data, len(r), distance.ctypes.data, position.ctypes.data
    ):
        return None
    return distance, position


def native_boundary_chains(
    pixels, codes, gray, visible, xs, ys, step, distances, minimum_samples, minimum_length
):
    if _boundary_function is None:
        return None
    pixels = np.ascontiguousarray(pixels, dtype=np.int32)
    codes, gray, visible = [np.ascontiguousarray(a, dtype=np.uint8) for a in (codes, gray, visible)]
    xs, ys, distances = [np.ascontiguousarray(a, dtype=np.float64) for a in (xs, ys, distances)]
    if (
        pixels.ndim != 2
        or pixels.shape[1] != 2
        or codes.ndim != 2
        or gray.shape != codes.shape
        or visible.shape != codes.shape
        or xs.shape != (codes.shape[0],)
        or ys.shape != (codes.shape[1],)
        or distances.shape != (3,)
        or not np.isfinite(distances).all()
    ):
        return None
    count = len(pixels)
    points = np.empty((count, 4))
    runs = np.empty((count, 3))
    result = _boundary_function(
        pixels.ctypes.data,
        count,
        codes.ctypes.data,
        gray.ctypes.data,
        visible.ctypes.data,
        *codes.shape,
        xs.ctypes.data,
        ys.ctypes.data,
        step,
        distances.ctypes.data,
        minimum_samples,
        minimum_length,
        points.ctypes.data,
        runs.ctypes.data,
    )
    if result == 0:
        return None
    return [
        (
            points[int(offset) : int(offset + length), :2],
            points[int(offset) : int(offset + length), 2:],
            float(facing),
        )
        for offset, length, facing in runs[: result - 1]
    ]


def native_pursuit_selection(path, radius, forward, maximum):
    if _pursuit_function is None:
        return False, None
    path = np.ascontiguousarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        return True, None
    out = np.empty(6, dtype=np.float64)
    code = _pursuit_function(path.ctypes.data, len(path), radius, forward, maximum, out.ctypes.data)
    if code == 0:
        return False, None
    if code == 2:
        return True, None
    if radius == 0.0:
        return True, (out[:2].copy(), float(out[2]))
    return True, dict(
        point=out[:2].copy(),
        position_m=float(out[2]),
        distance_m=float(out[3]),
        mode=("FAR_SUPPORT", "LOOKAHEAD", "SHORT_SUPPORT")[int(out[4])],
        speed_scale=float(out[5]),
    )


def native_curve_agree_paths(a, b):
    if _paths_agreement_function is None:
        return False, None
    a, b = [np.ascontiguousarray(p, dtype=np.float64) for p in (a, b)]
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 2 or b.shape[1] != 2:
        return True, None
    out = np.empty(3, dtype=np.float64)
    code = _paths_agreement_function(a.ctypes.data, len(a), b.ctypes.data, len(b), out.ctypes.data)
    if code == 0:
        return False, None
    if code == 2:
        return True, None
    return True, dict(distance_m=float(out[0]), heading_cos=float(out[1]), overlap_m=float(out[2]))


def native_nearest_samples(query, reference):
    """First nearest sampled vertex, preserving float64 NumPy/hypot semantics."""
    if _nearest_samples_function is None:
        return None
    q = np.asarray(query)
    r = np.asarray(reference)
    # Float32/integer subtraction has different rounding/overflow semantics;
    # leave those uncommon inputs to the original NumPy implementation.
    if (
        q.dtype != np.float64
        or r.dtype != np.float64
        or q.ndim != 2
        or r.ndim != 2
        or q.shape[1] != 2
        or r.shape[1] != 2
    ):
        return None
    q = np.ascontiguousarray(q)
    r = np.ascontiguousarray(r)
    indices = np.empty(len(q), dtype=np.int32)
    distances = np.empty(len(q), dtype=np.float64)
    if not _nearest_samples_function(
        q.ctypes.data, len(q), r.ctypes.data, len(r), indices.ctypes.data, distances.ctypes.data
    ):
        return None
    return indices, distances


def native_road_regions(hue, saturation, value, visible, mean, second):
    """Classify finite uint8 HSV data and its float32 OpenCV box moments.

    Caller supplies finite box moments computed from uint8 HSV, as in
    perception.photometry; arbitrary/nonfinite moment fields are unsupported.
    Four output planes share one allocation; boolean views contain only 0/1.
    Shape checks bound C access. Older libraries simply use the NumPy fallback.
    """
    if _road_regions_function is None:
        return None
    h, s, v, seen = (
        np.ascontiguousarray(a, dtype=np.uint8) for a in (hue, saturation, value, visible)
    )
    mean, second = (np.ascontiguousarray(a, dtype=np.float32) for a in (mean, second))
    if v.ndim != 2 or not v.size or any(a.shape != v.shape for a in (h, s, seen, mean, second)):
        return None
    if v.size > 2_147_483_647:
        return None
    output = np.empty((4, *v.shape), np.uint8)
    if not _road_regions_function(
        *(a.ctypes.data for a in (h, s, v, seen, mean, second)), v.size, output.ctypes.data
    ):
        return None
    return output[0].view(np.bool_), output[1], output[2].view(np.bool_), output[3].view(np.bool_)
