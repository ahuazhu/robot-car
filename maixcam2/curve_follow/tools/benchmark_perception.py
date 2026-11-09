#!/usr/bin/env python3
"""Replay checked-in road frames; compare fused photometry with its fallback.

Build the C kernel and set CURVE_NATIVE_LIBRARY before running this script.
The benchmark never connects to a camera, serial port or running vehicle.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "maixcam2/curve_follow"))
from robot.core.geometry import geometry_frame
from robot.core.native_geometry import native_fit_status
from robot.perception import photometry
from robot.perception.boundaries import CurvePixels


@contextmanager
def backend(fused):
    original = photometry.native_road_regions
    if not fused:
        photometry.native_road_regions = lambda *args: None
    try:
        yield
    finally:
        photometry.native_road_regions = original


def frames():
    fixtures = ROOT / "tests/fixtures"
    sequence = CurvePixels(json.loads((fixtures / "road_model_recording_calibration.json").read_text()))
    for name in sorted((fixtures / "road_model_sequence").glob("*.png")):
        yield sequence, cv2.imread(str(name))[:240, :320]
    diagnostic = CurvePixels(json.loads((fixtures / "camera_pose_20261031_191327.json").read_text()))
    for name in ("curve_shadow_glare.jpg", "curve_zebra_turn.jpg", "right_bend.jpg", "curve_white_markings.png"):
        yield diagnostic, cv2.imread(str(fixtures / name))[:240, :320]


def quantiles(values):
    return dict(median_ms=round(float(np.median(values)), 4),
                p95_ms=round(float(np.quantile(values, .95)), 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if not native_fit_status().get("road_regions_enabled"):
        parser.error("compile curve_fit_kernel.c and set CURVE_NATIVE_LIBRARY to the resulting library")
    inputs = list(frames())
    timings = {mode: dict(prepare=[], measure=[]) for mode in ("reference", "fused")}
    passes = Counter()
    # Warm both paths before collecting paired rounds; alternate order to
    # reduce bias from CPU frequency changes during a longer replay.
    for fused in (False, True):
        with backend(fused):
            for pixels, image in inputs[:8]:
                with geometry_frame():
                    pixels.measure(image)
    for repeat in range(args.repeat):
        for fused in ((False, True) if repeat % 2 == 0 else (True, False)):
            mode = "fused" if fused else "reference"
            with backend(fused):
                for pixels, image in inputs:
                    start = time.perf_counter_ns()
                    pixels.prepare(image)
                    timings[mode]["prepare"].append((time.perf_counter_ns() - start) / 1e6)
                    start = time.perf_counter_ns()
                    with geometry_frame():
                        _, details = pixels.measure(image)
                    timings[mode]["measure"].append((time.perf_counter_ns() - start) / 1e6)
                    if repeat == 0 and fused:
                        passes[details["photometric_passes"]] += 1
    report = dict(platform=platform.platform(), frames=len(inputs), repeat=args.repeat,
                  photometric_passes=dict(passes),
                  timings={mode: {stage: quantiles(data) for stage, data in result.items()}
                           for mode, result in timings.items()},
                  limit="Host CPU replay; not measured device FPS, car speed or lap time.")
    for stage in ("prepare", "measure"):
        before = np.median(timings["reference"][stage])
        after = np.median(timings["fused"][stage])
        report.setdefault("median_reduction_percent", {})[stage] = round(100 * (1 - after/before), 2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
