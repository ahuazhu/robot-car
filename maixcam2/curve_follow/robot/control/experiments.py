"""Session-only, independently selectable trials against the v0.7.1 baseline."""


def baseline_experiments():
    # Never persist trial selections: restarting the app restores the baseline.
    return dict(motion_gap_hold=False, crosswalk_recovery_dwpp=False)


def normalize_experiments(flags):
    if (not isinstance(flags, dict) or set(flags) != set(baseline_experiments())
            or any(type(value) is not bool for value in flags.values())):
        raise ValueError("实验开关必须完整提供，且只能为开启或关闭")
    return dict(flags)
