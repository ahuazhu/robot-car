"""Current contiguous road support shared by live control and offline analysis."""

import numpy as np


def supported_segment(target, *, skip_inferred_prefix=False, include_history=False):
    """First observed forward segment; never bridge gaps or count a blind prefix.

    Station zero is the first supported sample, not the vehicle origin. The
    caller must independently authorize pursuit and whole-body clearance. A
    live caller with that proof may start at the first current sample after
    an inferred prefix; the skipped prefix never adds usable braking distance.
    """
    historical=bool(include_history and (target.get('road_model') or {}).get('persistent_history'))
    if (not target.get('valid') or (not historical and (not target.get('observed')
            or target.get('state') != 'OBSERVED' or target.get('transition_pending')))):
        return None, 'NOT_CURRENT_OBSERVATION'
    raw_mask = target.get('path_observed_mask', [])
    if not isinstance(raw_mask, list) or any(type(v) is not bool for v in raw_mask):
        return None, 'INVALID_PATH_OR_MASK'
    try:
        path = np.asarray(target.get('control_path_m', []), dtype=float)
    except (ValueError, TypeError):
        return None, 'INVALID_PATH_OR_MASK'
    mask = np.asarray(raw_mask, dtype=bool)
    if (path.ndim != 2 or path.shape[1:] != (2,) or not np.isfinite(path).all()
            or mask.shape != (len(path),)):
        return None, 'INVALID_PATH_OR_MASK'
    if historical:
        # These are previously measured route samples, not a polynomial
        # extension. Observation freshness and geometric support are separate.
        mask=np.ones(len(path),dtype=bool)
    start = next((i for i, point in enumerate(path)
                  if point[0] > 0 and (mask[i] or not skip_inferred_prefix)), len(path))
    end = start
    while end < len(path) and mask[end]:
        if end > start and np.hypot(*(path[end] - path[end - 1])) > .05:
            break
        end += 1
    segment = path[start:end]
    return (segment, '') if len(segment) >= 5 else (None, 'NO_CONTIGUOUS_SUPPORT')
