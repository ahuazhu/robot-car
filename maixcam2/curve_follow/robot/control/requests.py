"""Immutable commands passed from operator interfaces to the driving loop."""

from dataclasses import dataclass
from typing import Optional

from robot.core.race import validate_laps


@dataclass(frozen=True)
class ControlRequest:
    action: str
    laps: Optional[int] = None

    def __post_init__(self):
        if self.action not in ("start", "start_now", "stop"):
            raise ValueError("不支持的操作")
        if self.action != "stop" and self.laps is not None:
            validate_laps(self.laps)
