# robot-car

miniAuto 麦克纳姆轮小车：MaixCAM2 视觉巡线，Arduino UNO 电机控制，ESP32-S3 摄像头预览及后续路牌识别。

## 当前版本

**v0.6.12-crosswalk-boundary，2026-10-25**，已部署到 MaixCAM2。

- 道路中线采用局部参数二次曲线，支持双边线与单边线推算。
- 使用自适应前瞻追踪控制，支持有限历史预测及有道路余量约束的小幅转向寻路。
- 横道礼让：接近、停车确认、约 3.3 秒等待、继续通过；刹车灯由 UNO 控制。
- 本版本修正横道前反复起停：只有停车距离的整个误差区间都在 20cm 之外，才允许再次接近。位置不确定时保持停车确认，不冒充已验证的合法停车。
- 网页提供实时原图、识别结果、地面投影，以及 START / STOP。部署后默认停车，手动 START 后无固定运行时长限制。

此前版本已实车跑完全程。v0.6.12 已完成 281 项巡线及横道测试和部署后的零速度、通信检查；本版本减少横道顿挫的实际效果仍待下一轮实测。

## 代码位置

| 目录 / 文件 | 用途 |
| --- | --- |
| `maixcam2/curve_follow/` | 当前 MaixCAM2 巡线与横道礼让程序 |
| `maixcam2/curve_follow/road_model.py` | 道路中线模型与观测关联 |
| `maixcam2/curve_follow/steering_control.py` | 前瞻追踪与转向控制 |
| `maixcam2/curve_follow/crosswalk.py` | 横道检测、接近与礼让状态机 |
| `maixcam2/curve_follow/motor_channel.py` | 独立电机通信进程 |
| `maixcam2/curve_follow/runtime.json` | 当前车辆参数 |
| `maixcam2/curve_follow/calibration.json` | 当前镜头姿态的地面投影标定 |
| `uno/uno_lane_drive/` | 当前 UNO 固件，含电机看门狗和刹车灯 |
| `esp32s3/video_preview/` | ESP32-S3 USB 视频预览 |
| `tests/` | 控制、视觉、通信及横道回归测试和精简图像样本 |
| `docs/` | 接线、阶段测试和参赛任务资料 |

其余 MaixCAM2 子目录和 `README.txt` 保留早期实验及诊断记录；当前入口以 `curve_follow` 为准。

## 当前接线

- MaixCAM2 UART4：A21 TX → 电平转换 LV1/HV1 → UNO A0 RX。
- UNO A1 TX → 电平转换 HV2/LV2 → MaixCAM2 A22 RX。
- 摄像头串口速率 9600；UNO USB 调试串口 115200。
- 刹车灯信号接 UNO A2。供电和地线按实际模块定义连接，3.3V 与 5V 逻辑通过转换板连接。

标定和车身尺寸针对本车；改变镜头安装位置或角度后需要重新标定。

## 本地测试

主机 Python 环境需要 NumPy 和 OpenCV。设备端程序使用 MaixPy，不能直接作为桌面 Python 摄像头应用运行。

```sh
python3 -m venv .maixpy/.venv
.maixpy/.venv/bin/python -m pip install numpy opencv-python
mkdir -p .maixpy/speed-profile
cc -std=c99 -O3 -fPIC -shared maixcam2/curve_follow/curve_fit_kernel.c \
  -o .maixpy/speed-profile/host_curve_kernel.so -lm
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  CURVE_NATIVE_LIBRARY="$PWD/.maixpy/speed-profile/host_curve_kernel.so" \
  .maixpy/.venv/bin/python -m unittest discover -s tests -p 'test_curve*.py'
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .maixpy/.venv/bin/python -m unittest discover -s tests -p 'test_crosswalk*.py'
```

当前结果：236 项巡线测试、45 项横道测试通过。部分历史实车录像仅保存在本机；依赖这些录像的测试在没有录像时会跳过。

## 生成部署程序

```sh
mkdir -p .maixpy
python3 maixcam2/curve_follow/build.py --drive
```

生成的 `.maixpy/install_curve_follow_drive.py` 通过已配置的 MaixPy 部署工具在设备上运行：备份旧版本、编译设备端 C 内核、启动应用，并检查多帧遥测与初始停车状态。省略 `--drive` 则生成强制预览版本。

UNO 使用 Arduino IDE 打开 `uno/uno_lane_drive/uno_lane_drive.ino`，开发板选择 Arduino UNO。

设备凭据、部署工具配置、完整录像、调试日志和旧固件备份保存在本机，不提交到 GitHub。仓库保留回归测试使用的精简样本。
