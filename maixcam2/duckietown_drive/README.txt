Duckietown-derived miniAuto lane drive — 2026-10-21

Camera app: standalone.py (build with build.py). UNO firmware remains uno/uno_lane_drive.
UART4 A21 TX / A22 RX -> level converter -> UNO A4 RX / A5 TX, 9600 baud.
Calibration: ../duckietown_test/calibration.json; keep camera mount fixed.

The native Duckietown line detector and ground projector are reused. Green exterior
classification, quadratic boundary fitting and PWM steering are custom adaptations;
this is not a full Duckietown stack or its native Bayesian lane filter.

Default is stopped. Camera screen and http://192.168.3.244:8000/ have START/STOP.
START requires five valid frames and a responsive UNO. Each manual run is limited
to five seconds. Missing vision sends zero speed immediately; persistent loss for
0.35 seconds disarms. UNO watchdog remains 350 ms. EXIT stops and closes the app.
Debug images are disabled unless explicitly enabled. No boot autostart configured.

Tests: 21 checks cover projection, missing boundaries, curved path, steering sign,
PWM bounds, proportional startup steering and immediate stop.
Device stationary curve test 20261021-205959: 114/114 valid frames. This is not
proof of geometric accuracy or driving success.
Device drive 20261021-210446 and 210553: each reached its three-second time limit,
final STOP acknowledged, but user reported excessive left turn / possible strafe.
Steering gain and cap then halved; startup now scales steering with forward PWM.
Full-lap tracking and physical wheel slip remain unvalidated.

Installation helper: .maixpy/maixpy run .maixpy/install_duckietown_drive.py --timeout 30
The generated installer contains the most recently built app; regenerate after edits.
Runtime path: /maixapp/apps/robot_duckietown_drive/main.py
Log: /tmp/robot_duckietown_drive.log

Latest field test 20261021-210902 reached time limit with STOP ACK. User reported
left turn began slightly early. Curve lookahead reduced from 0.45 m to 0.40 m;
this timing adjustment still requires another driving observation.

2026-10-21 ROAD LOST diagnosis:
- Saved event log: .maixpy/road_lost_2113.log; image: .maixpy/road_lost_2113.jpg.
- UNO remained connected; two-boundary fit failed when left edge left the useful field of view.
- Experimental single-boundary normal-offset model, recent-pair gate (1.5 s),
  5 cm target continuity gate and lower PWM added locally. 26 tests pass.
- Stationary run 211932: 8/12 logged samples SINGLE RIGHT, 4/12 invalid; all motor
  commands zero. NOT stable enough to claim driving validation.
- Device restored with the previous two-boundary/manual-START build from
  .maixpy/install_duckietown_drive.py; new single-side logic remains local only.
- Do not regenerate/deploy the experimental drive build as validated firmware.
