# robot-car

miniAuto 麦克纳姆轮小车：MaixCAM2 视觉巡线，Arduino UNO 电机控制，ESP32-S3 摄像头预览及后续路牌识别。

## 当前版本

**v0.6.17-full-speed-experiment，2026-10-25**，分支 `experiment/full-speed`。

稳定基线是提交 `f86de47`、标签 `baseline-v0.6.15`；横道速度反馈 v0.6.16 尚未部署，已单独保存。此分支只对比全速巡航。

- 道路中线采用局部参数二次曲线，支持双边线与单边线推算。
- 已接受的当前观测置信度至少 0.65、当前追踪点有观测支持、有效运动估计及车身余量检查通过，并连续稳定一个电机响应周期（当前 0.4 秒）后，允许提高巡航指令。基准 60，直道上限可配置；随曲率增加撤销额外提速。全速实验将 `straight_cruise_pwm` 设为 83，加既有起步补偿后的实际输出上限为 120/255。旧尾迹仍按原规则过期，但不再用整条历史路径最弱的点否定一个有当前观测支持的追踪点。偶发运动估计失效立即取消该次额外提速，连续道路观测的计时单独保留；实际道路丢失会重新计时。条件变弱退回基准巡航，不因此新增停车。横道和寻路限速保持独立。
- 使用自适应前瞻追踪控制，支持有限历史预测及有道路余量约束的小幅转向寻路。前瞻距离按电机响应时间平滑变化，每张图仅更新一次，防止瞬时测速波动让追踪点在弯道前后跳变。中线及目标支持范围不做延伸。
- 横道礼让：接近、停车确认、约 3.3 秒等待、继续通过；刹车灯由 UNO 控制。
- 本版本修正横道前反复起停：只有停车距离的整个误差区间都在 20cm 之外，才允许再次接近。位置不确定时保持停车确认，不冒充已验证的合法停车。
- 网页提供实时原图、识别结果、地面投影，以及 START / STOP。部署后默认停车，手动 START 后无固定运行时长限制。

此前版本已实车跑完全程。v0.6.15 已完成 298 项巡线及横道测试；实测记录证明前瞻距离突变可以在同一中线上产生反向指令，固定几何回放验证平滑后消除了其中两次明显反向。v0.6.14 这一轮的记录未显示实际额外提速，横道前后仍有抖动，因此暂不继续加速。修正后的横道前后抖动已获用户正、反两个行驶方向的改善反馈。全速实验仍需实车对比，尚未证明整圈速度收益。此前 v0.6.13 提速试跑经用户确认到达终点、未压线或出界，但有明显停顿，未证明整圈提速；该轮存在障碍物影响，不能直接用它确定加速与停顿的因果关系。

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

当前分支结果：259 项巡线测试、45 项横道测试通过。新增测试验证输出上限、当前观测与候选隔离、测速缺测回退和丢路后重新计时。部分历史实车录像仅保存在本机；依赖这些录像的测试在没有录像时会跳过。

## 生成部署程序

```sh
mkdir -p .maixpy
python3 maixcam2/curve_follow/build.py --drive
```

生成的 `.maixpy/install_curve_follow_drive.py` 通过已配置的 MaixPy 部署工具在设备上运行：备份旧版本、编译设备端 C 内核、启动应用，并检查多帧遥测与初始停车状态。省略 `--drive` 则生成强制预览版本。

UNO 使用 Arduino IDE 打开 `uno/uno_lane_drive/uno_lane_drive.ino`，开发板选择 Arduino UNO。

设备凭据、部署工具配置、完整录像、调试日志和旧固件备份保存在本机，不提交到 GitHub。仓库保留回归测试使用的精简样本。
