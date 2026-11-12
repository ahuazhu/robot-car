"""JSON encoding at telemetry boundaries, preserving NumPy value types."""
import json
import numpy as np


def _numeric_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray) and value.dtype.kind in 'biuf':
        return value.tolist()
    # Never stringify arbitrary objects or silently change an unknown schema.
    raise TypeError(f'Unsupported telemetry type: {type(value).__name__}')


def dumps(value, **kwargs):
    """Only unsupported numeric leaves need conversion; no second tree walk."""
    return json.dumps(value, default=_numeric_default, **kwargs)
