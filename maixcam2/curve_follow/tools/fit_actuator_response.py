"""Read-only system identification from ACK timing and integrated ground flow.
Fit on whole trials; reserve repeated/changed-duty trials for validation.
No runtime config is written. Camera-coordinate speeds are not encoder speeds.
"""

import argparse
import json, math
from pathlib import Path
import numpy as np

KNOTS = (50, 60, 70, 80)


def load_trial(path):
    d = json.loads(path.read_text())
    rows = d["samples"]
    start = next(r["powered_since"] for r in rows if r["powered_since"] is not None)
    events = []
    previous = (0, 0)
    for r in rows:
        m = r.get("motor") or {}
        a = m.get("ack") or []
        if len(a) < 5:
            continue
        pwm = tuple(a[2:4]) if a[1] else (0, 0)
        if pwm != previous:
            events.append((r["at"] - m["ack_age_ms"] / 1000 - start, pwm))
            previous = pwm
    # The response model explicitly separates cold 50 (observed stalled) from
    # 50 following an established 70 phase (observed rolling).
    active = [False, False]
    segments = []
    for i, (at, pwm) in enumerate(events):
        end = events[i + 1][0] if i + 1 < len(events) else rows[-1]["captured"] - start
        sides = []
        for side, p in enumerate(pwm):
            if p >= 60:
                active[side] = True
            if p == 0:
                active[side] = False
            sides.append(p if p in KNOTS and active[side] else 0)
        segments.append((at, end, tuple(sides)))
    samples = []
    distance = 0.0
    last_t = -1e9
    for r in rows:
        if r["phase"] not in ("DRIVE", "COAST", "DONE"):
            continue
        t = r["captured"] - start
        if r.get("speed_m_s") is not None and r["motion"].get("valid"):
            distance += r["speed_m_s"] * r["motion"]["interval_s"]
        # 20Hz observations, complete trial through final coast, not long idle.
        if t - last_t >= 0.05 and -0.05 <= t <= events[-1][0] + 0.8:
            samples.append((t, distance, r["pose"][2]))
            last_t = t
    if not events or len(samples) < 3 or not np.isfinite(np.asarray(samples)).all():
        raise ValueError("Missing or non-finite probe motion: " + str(path))
    return dict(
        run=path.name.removesuffix("-device.json"),
        samples=np.asarray(samples),
        segments=segments,
        aborted=d["aborted"],
        paired=all((l > 0) == (r > 0) for _, _, (l, r) in segments),
    )


def basis(trial, tau, delay):
    t = trial["samples"][:, 0]
    b = np.zeros((len(t), 4))
    yaw = np.zeros_like(b)

    def integral(edge):
        u = np.maximum(0.0, t - edge - delay)
        return u + tau * np.expm1(-u / tau)

    for a, z, sides in trial["segments"]:
        response = integral(a) - integral(z)
        for side, p in enumerate(sides):
            if p:
                k = KNOTS.index(p)
                b[:, k] += 0.5 * response
                yaw[:, k] += (1 if side else -1) * response / 0.3
    return b, yaw


def fit_trials(trials, training_runs, *, paired_only=False):
    train = [r for r in trials if r["run"] in training_runs and (not paired_only or r["paired"])]
    if len(train) < 4:
        raise ValueError("At least four complete training trials required")
    best = None
    for tau in np.linspace(0.04, 0.8, 39):
        for delay in np.linspace(0, 0.20, 21):
            matrices = [basis(r, tau, delay)[0] for r in train]
            # Each trial has equal total weight; positions integrate noisy flow.
            A = np.concatenate([a / math.sqrt(len(a)) for a in matrices])
            y = np.concatenate([r["samples"][:, 1] / math.sqrt(len(r["samples"])) for r in train])
            speeds = np.linalg.lstsq(A, y, rcond=None)[0]
            if np.any(speeds <= 0) or np.any(np.diff(speeds) <= 0):
                continue
            loss = float(np.mean((A @ speeds - y) ** 2))
            if best is None or loss < best[0]:
                best = (loss, float(tau), float(delay), speeds)
    if best is None:
        raise ValueError("No positive monotone fit")
    _, tau, delay, speeds = best
    gains = {}
    for sign, name in ((1, "left"), (-1, "right")):
        x = []
        y = []
        for r in train:
            nominal = basis(r, tau, delay)[1] @ speeds
            if sign * nominal[-1] > 0.02:
                x.extend(nominal)
                y.extend(r["samples"][:, 2])
        if not x or np.dot(x, x) <= 1e-10:
            raise ValueError("Both turn directions need observations")
        gains[name] = float(np.dot(x, y) / np.dot(x, x))
    results = []
    for r in trials:
        b, yaw = basis(r, tau, delay)
        pred = b @ speeds
        angle = yaw @ speeds
        gain = gains["left" if angle[-1] >= 0 else "right"]
        angle *= gain
        observed = r["samples"]
        results.append(
            dict(
                run=r["run"],
                split=(
                    "out_of_domain"
                    if paired_only and not r["paired"]
                    else "fit"
                    if r["run"] in {v["run"] for v in train}
                    else "replay"
                ),
                aborted=r["aborted"],
                forward_actual_mm=float(observed[-1, 1] * 1000),
                forward_predicted_mm=float(pred[-1] * 1000),
                forward_error_mm=float((pred[-1] - observed[-1, 1]) * 1000),
                position_rmse_mm=float(np.sqrt(np.mean((pred - observed[:, 1]) ** 2)) * 1000),
                yaw_actual_deg=float(np.degrees(observed[-1, 2])),
                yaw_predicted_deg=float(np.degrees(angle[-1])),
                yaw_error_deg=float(np.degrees(angle[-1] - observed[-1, 2])),
            )
        )
    # Explicit, declared offline gates. Passing is necessary, not sufficient for
    # physical use; these tolerate 20% travel / 5deg yaw and 15mm trajectory RMSE.
    for r in results:
        r["passes_offline_gate"] = (
            abs(r["forward_error_mm"]) <= max(10.0, 0.2 * abs(r["forward_actual_mm"]))
            and abs(r["yaw_error_deg"]) <= 5
            and r["position_rmse_mm"] <= 15
        )
    report = dict(
        scope="Exploratory replay in optical coordinates; model selection has seen the replay data, so this is not blind validation",
        domain="paired_forward" if paired_only else "mixed_paired_and_stalled",
        calibration_verified=False,
        assumptions=[
            "Both sides use a common equivalent-speed curve; independent left-side low-duty behavior is not fully identified",
            "Cold PWM50 is stationary; rolling PWM50 follows an established PWM70 phase in these specific records",
            "ACK receive times approximate actuation times; camera/transport delay is fitted separately",
            "First-order time constant is shared across acceleration and deceleration",
        ],
        fitted=dict(
            knots_m_s=list(zip(KNOTS, speeds.tolist())),
            response_s=tau,
            ack_to_observation_delay_s=delay,
            direction_gains=gains,
        ),
        criteria=dict(
            relative_travel_error=0.2,
            minimum_travel_tolerance_mm=10,
            maximum_yaw_error_deg=5,
            maximum_position_rmse_mm=15,
        ),
        trials=results,
        all_replays_pass=all(r["passes_offline_gate"] for r in results if r["split"] == "replay"),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--training-runs", type=Path, required=True, help="JSON array of exact training run IDs"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trials = [load_trial(p) for p in sorted(args.input_dir.glob("*-device.json"))]
    training = json.loads(args.training_runs.read_text())
    if not isinstance(training, list) or not all(isinstance(v, str) for v in training):
        raise ValueError("Expected explicit training run IDs")
    missing = set(training) - {r["run"] for r in trials}
    if missing:
        raise ValueError("Training trials missing: " + repr(sorted(missing)))
    report = {
        name: fit_trials(trials, set(training), paired_only=paired)
        for name, paired in (("mixed", False), ("paired", True))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, data in report.items():
        print(name, json.dumps(data["fitted"]))
        for row in data["trials"]:
            if row["split"] == "replay":
                print(
                    row["run"],
                    "travel_error_mm",
                    round(row["forward_error_mm"], 2),
                    "yaw_error_deg",
                    round(row["yaw_error_deg"], 2),
                    "pass",
                    row["passes_offline_gate"],
                )
    print("Wrote", args.output, "(no runtime settings changed)")


if __name__ == "__main__":
    main()
