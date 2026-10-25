Duckietown / MaixCAM2 实机视觉测试（2026-10-21）

当前完成：官方 LineDetector 移植、miniAuto 双白边分类、原始 GroundProjector 接口、四点标定工具、实拍回放、实时预览。
已完成用户确认的实物四点标定；未完成镜头畸变标定和完整赛道行驶验收。
独立行驶测试程序见 ../duckietown_drive/；本目录 main.py 仍只做视觉预览。
本测试没有打开 UART、没有发送电机命令；网页 START 不提供。不要将识别到线段等同于允许行驶。

代码：
 duckietown_core.py：从官方源码提取的检测器、点类型及 GroundProjector，版本和改动见 UPSTREAM.json。
 vision.py：本车的颜色范围、绿色场外验证、左右边分类及标定矩阵接口。
 main.py：MaixCAM2 入口，屏幕右下角 EXIT 退出。
 build.py：复用原程序的 HTTP 传输，生成可单文件执行的 standalone.py。
 preview.html：检测预览页面。
 replay.py：离线回放图片，输出拼图和 report.json。线段数量不代表正确率。
 calibrate.py：输入实测纸条四个端点生成 calibration.json，顺序近左、近右、远右、远左。

标定：
 两条纸条均长30厘米，横向平行摆放，中点对准车头正前方。
 近条距镜头正下方地面点30厘米，远条60厘米，四端构成30x30厘米矩形。
 车与相机固定不动。标定原点为镜头地面投影；相机到车轴的偏移尚未用于控制。
 不能把未碰到白边的纸条端点直接当作道路宽度，也不能把弯道边界当矩形角点。
 当前四点法直接映射原始画面，lens_rectified=false，只作初步投影验证，不等同完整内外参标定。

运行：
 本机 Python: /Users/yang/work/robot-car/.maixpy/.venv/bin/python
 本机执行 build.py 后，MaixVision 可运行 standalone.py。
 辅助工具：.maixpy/maixpy run maixcam2/duckietown_test/standalone.py
 设备检测预览安装路径：/maixapp/apps/robot_duckietown_test/main.py
 状态与画面：http://192.168.3.244:8000/
 原程序仍在 maixcam2/lane_follow/，UNO 固件未修改。不要同时运行两个占用摄像头的应用。

验证证据：
 .maixpy/runs/20261021-185144-duckietown_probe/：原始320分辨率筛选约180ms。
 .maixpy/runs/20261021-185333-duckietown_probe/：批量筛选约50ms。
 .maixpy/runs/20261021-185532-duckietown_probe/：160宽检测/320画面约21ms，含显示循环累计约30.8fps。
 这是静止画面的10秒测试，不是行驶帧率或巡线成功率。
 tests/test_duckietown.py：6项检查（米制投影、左右方向、空图、横线、错序、分辨率）。
 .maixpy/duckietown-results/replay/：7张实拍图回放；需继续人工检查假线段。

2026-10-21 near-ground pose preview
- lane_pose.py uses Duckietown segment-normal / half-lane-width measurement
  geometry with the two white borders classified by green exterior pixels.
- Uses ground x=0.20..0.65 m, balanced left/right consensus; no encoder
  prediction and no persistent posterior. This is not the full native filter.
- Positive d means vehicle left of lane center; positive phi means vehicle
  heading left relative to lane. Green overlay is inferred road center.
- Missing/inconsistent pairs return invalid immediately, never stale values.
- Device run 20261021-200344: stationary 15-second perception run completed;
  ~20-21 processing frames/s after initialization, ~40 ms detector median.
  Once-per-second samples include offset jitter and an invalid estimate.
  Steering is NOT enabled or validated. All test runs send no motor commands.
- Tests: python -m unittest discover -s tests -p 'test_duckietown*.py' -v
  Eleven coordinate, blank-image, missing-side and clutter checks passed.

Recalibration 2026-10-21 20:08:31: permanent near printed line at 30 cm, temporary far line at 60 cm; inner border intersections with 30 cm road width. Previous JSON backed up under .maixpy/calibrations. Independent image rows give 0.299 / 0.303 m widths. Device validation 201107: 123/257 frames had consistent paired pose (47.9%), valid-frame lateral std 3.07 cm; not steering-ready. No motor command sent. Preserve permanent printed line; remove only far temporary marker for next assessment.

Confirmed-square calibration 2026-10-21 20:39: user explicitly confirmed four intersections form 30x30 cm square after diagonal clarification. Updated points [[48,196],[282,194],[219,145],[104,146]], near .30/far .60 m. geometry_confirmed=true. Four-point mapping check is mathematical consistency, not independent lens validation. Device run 204121: 157/188 frames valid (83.5%), lateral std 1.58 cm, angle std 0.0348 rad. This does NOT validate autonomous driving. All motors remain off in this app.
