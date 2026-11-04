"""Shared bounds for configured and on-site race targets."""

MIN_LAPS = 1
MAX_LAPS = 100


def validate_laps(value):
    if isinstance(value, bool) or not isinstance(value, int) or not MIN_LAPS <= value <= MAX_LAPS:
        raise ValueError("目标圈数须为1至100的整数")
    return value
