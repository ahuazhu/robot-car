# Offline tests

Run from the repository root, without connecting the car:

```sh
python -m unittest discover -s tests -t .
```

For the complete native/reference comparison, compile the current C source for the host first:

```sh
cc -std=c99 -O3 -fPIC -shared maixcam2/curve_follow/curve_fit_kernel.c -o /tmp/robot-test-kernel.so -lm
CURVE_NATIVE_LIBRARY=/tmp/robot-test-kernel.so python -m unittest discover -s tests -t .
```

Dependencies: NumPy and OpenCV. `tests.support` adds the application source root once; all production imports use the real `robot` package. A single suite can be run with `python -m unittest tests.test_session`.

## Coverage

- `test_session.py`, `test_startup.py`: STOP/start ordering, shared readiness gates, command freshness, recovery, and motor failure.
- `test_curve_*`: road perception, physical target geometry, road memory, handoff, steering, bounded search, motor protocol, and preview controls.
- `test_native_geometry.py`, `test_curve_nearest_fast.py`: native kernels against their independent Python/NumPy paths, including malformed input and fallback limits.
- `test_crosswalk*`: observed bars, brightness changes, stop positions, verified dwell, spatial rearming, and recovery.
- `test_curve_laps.py`, `test_curve_confirmed_forward.py`: departure suppression, measured crossings, rear clearance and confirmed forward-output runout time.
- `test_curve_start_light.py`: archived Maix detector reference (offline only).
- `test_esp_start_signal.py`: actual embedded detector, recorded lamp replay, remote event freshness, cancellation and one-shot consumption.
- `test_build.py`: package build and offline installer behavior.

Shared scene/clock/protocol constructors live under `support/`; tests do not import helpers from other test modules. Frozen numerical reference implementations live under `reference/`; images and captured calibration/telemetry remain under `fixtures/`. Machine-local `.maixpy` captures are benchmark inputs, not optional regression cases.
