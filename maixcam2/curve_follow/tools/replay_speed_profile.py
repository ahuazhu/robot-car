"""Evaluate speed envelopes on recorded road segments; never connect to motors.

Example (from repo root):
  python maixcam2/curve_follow/tools/replay_speed_profile.py \
    tests/fixtures/curve_oscillation.json --output .maixpy/speed-profile.json
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys
import time

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[1]
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from robot.core.geometry import arc, describe_curve
from robot.core.road_policy import RoadPolicy
from robot.navigation.speed_profile import SpeedLimits, plan_speed_profile
from robot.navigation.speed_observer import supported_segment
from maixcam2.curve_follow.tools.drive_reference import DriveReference


def load_frames(path):
    content = Path(path).read_text()
    if content.lstrip().startswith("["):
        rows = json.loads(content)
    else:
        rows = [
            json.loads(line.split("CONTROL ", 1)[1])
            for line in content.splitlines() if "CONTROL " in line
        ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Expected telemetry JSON array or CONTROL log lines: " + str(path))
    return rows


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(count=len(values), minimum=ordered[0], median=statistics.median(ordered),
                p95=ordered[math.ceil(0.95 * len(ordered)) - 1], maximum=ordered[-1])


def replay(paths, limits, reference, *, front_margin_m, sigma_multiplier=2.0, include_frames=False):
    counts, reasons = Counter(), Counter()
    ceilings, plan_ms, total_ms, old_references, recorded_loop_ms, details = [], [], [], [], [], []
    per_input = []
    for path in paths:
        source_count = Counter()
        for row in load_frames(path):
            counts["frames"] += 1
            source_count["frames"] += 1
            if not row.get("active"):
                counts["inactive"] += 1
                continue
            counts["active"] += 1
            target = row.get("target", {})
            segment, rejected = supported_segment(target)
            if rejected:
                counts[rejected] += 1
                continue
            started = time.perf_counter()
            curve = describe_curve(segment)
            sigma = target.get("reference_sigma_m")
            if curve is None or sigma is None or not math.isfinite(sigma) or sigma < 0:
                counts["INVALID_GEOMETRY_OR_UNCERTAINTY"] += 1
                continue
            stations = arc(curve["path"])
            planning = time.perf_counter()
            profile = plan_speed_profile(stations, curve["curvature"], limits,
                                         distance_margin_m=front_margin_m + sigma_multiplier * sigma)
            finished = time.perf_counter()
            plan_ms.append((finished - planning) * 1000)
            total_ms.append((finished - started) * 1000)
            ceilings.append(profile.entry_limit_m_s)
            reasons[profile.entry_reason] += 1
            counts["segments_evaluated"] += 1
            source_count["segments_evaluated"] += 1
            loop_ms = row.get("loop_processing_ms")
            if loop_ms is not None and math.isfinite(loop_ms):
                recorded_loop_ms.append(loop_ms)
            wheels = row.get("pwm", [])
            if len(wheels) == 2 and wheels[0] == wheels[1]:
                old = reference.speed_for_pwm(wheels[0])
                if old is not None:
                    old_references.append(old)
                else:
                    counts["straight_pwm_outside_measured_domain"] += 1
            else:
                counts["turn_pwm_has_no_calibrated_speed_reference"] += 1
            if include_frames:
                details.append(dict(
                    source=Path(path).name, frame=row.get("frame"),
                    path_first_point_m=segment[0].tolist(),
                    supported_length_m=float(stations[-1]),
                    usable_length_m=profile.stations_m[-1],
                    entry_limit_m_s=profile.entry_limit_m_s,
                    entry_reason=profile.entry_reason,
                    limiting_station_m=profile.limiting_station_m,
                    crosswalk_phase=row.get("crosswalk", {}).get("phase"),
                    actual_uno_pwm=wheels,
                ))
        per_input.append(dict(source=str(path), **source_count))
    report = dict(
        schema_version=1, scope="OFFLINE_OBSERVED_SEGMENT_ENTRY_ONLY", production_commands_changed=False,
        constraints=asdict(limits), front_margin_m=front_margin_m, sigma_multiplier=sigma_multiplier,
        counts=dict(counts), sources=per_input, entry_limit_reasons=dict(reasons),
        segment_entry_ceiling_m_s=distribution(ceilings),
        recorded_equal_wheel_pwm_speed_reference_m_s=distribution(old_references),
        host_planner_ms=distribution(plan_ms), host_geometry_and_planner_ms=distribution(total_ms),
        recorded_full_loop_ms=distribution(recorded_loop_ms),
        limitations=[
            "Entry is the first supported point, not the car. Near-road gaps are not traversed.",
            "Only current contiguous observations are used; prediction/transitions are excluded.",
            "Geometry retains the recording's calibration; optical-flow scale is not corrected.",
            "Lateral/forward acceleration are provisional. Braking uses the configured bound.",
            "Delay assumes constant speed before braking. Footprint clearance is not proved.",
            "Task stops/dwell, road handoffs, start permission and watchdogs are not simulated.",
            "Wood-floor PWM references are not historical measured speeds or lap predictions.",
        ])
    if include_frames:
        report["segments"] = details
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-frames", action="store_true")
    parser.add_argument("--lateral-acceleration", type=float, default=0.15,
                        help="Provisional m/s², not measured friction")
    parser.add_argument("--acceleration", type=float, default=0.15, help="Provisional forward m/s²")
    parser.add_argument("--deceleration", type=float,
                        help="Sensitivity analysis only; defaults to runtime bound, m/s²")
    parser.add_argument("--delay", type=float,
                        help="Sensitivity analysis only; defaults to freshness plus response, seconds")
    args = parser.parse_args(argv)
    record = json.loads((REPO_ROOT / "docs/动力测量参考-20261110.json").read_text())
    reference = DriveReference(record)
    runtime = json.loads((APP_ROOT / "runtime.json").read_text())
    crosswalk = runtime["crosswalk"]
    delay = RoadPolicy.from_config(runtime).missing_command_s + crosswalk["speed_response_s"]
    deceleration = crosswalk["brake_deceleration_m_s2"]
    limits = SpeedLimits(
        reference.speeds[-1], args.lateral_acceleration, args.acceleration,
        deceleration if args.deceleration is None else args.deceleration,
        delay if args.delay is None else args.delay, runtime["turn_lever_m"],
    )
    report = replay(args.inputs, limits, reference, front_margin_m=runtime["control_frame"]["body_m"][0],
                    include_frames=args.include_frames)
    report["drive_reference"] = {
        key: record[key] for key in ("date", "surface", "motor_supply", "race_surface_validated")
    }
    report["delay_assumption"] = (
        "Explicit sensitivity override" if args.delay is not None
        else "Control freshness plus motor response; not measured end-to-end latency"
    )
    report["deceleration_assumption"] = (
        "Explicit sensitivity override, not measured" if args.deceleration is not None
        else "Conservative production bound, not validated race-surface braking"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary_keys = ("counts", "entry_limit_reasons", "segment_entry_ceiling_m_s", "host_planner_ms")
    print(json.dumps({key: report[key] for key in summary_keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
