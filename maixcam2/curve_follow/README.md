# 小车应用

`main.py` 是 MaixPy 入口。应用源码是正常的 `robot` Python 包；本地测试与设备运行使用相同的导入关系。当前功能、接线、标定和参数说明见项目根目录的 [README](../../README.md)。

## 分层与状态归属

```text
curve_follow/
├── main.py                 # 设置计算线程数，启动应用
├── robot/
│   ├── application.py      # 组装组件、采集帧、连接输入与输出
│   ├── config.py           # 读取配置与统一参数边界
│   ├── core/               # 共享几何、运动变换、数值加速和道路策略
│   ├── perception/         # 原始图像 → 单帧道路边界与候选
│   ├── navigation/         # 时序道路模型、目标、转向与失路恢复
│   ├── tasks/              # 发车信号、横道礼让、计圈
│   ├── control/            # 驾驶会话、轮速、独立电机进程和UART
│   └── interface/          # 触屏、图像叠加、HTTP及网页资源
├── build.py                # 生成普通包和单文件安装器
├── tools/install_app.py    # 可独立验证的设备安装模板
├── calibration.json        # 当前镜头姿态的标定
├── runtime.json            # 当前驾驶和任务参数
└── curve_fit_kernel.c      # 可选的C99数值加速
```

- `perception.boundaries.CurvePixels` 负责光照处理、地面投影和单帧测量；不持有历史道路或驾驶状态。
- `navigation.tracker.TargetTracker` 将单帧测量与地面运动交给 `LocalRoadModel`。道路历史、确认窗口和置信度只由模型持有，已经删除旧的镜像字段和回灌接口。
- `navigation.steering_control` 选择实际前瞻目标并检查车身包络。几何锚点和驾驶目标分别记录，绘制层不修改导航结果。
- `navigation.road_recovery.RoadSearch` 协调恢复策略：有证据时使用已观测道路地图；证据不足且配置允许时委托 `BoundedVisualScan`。有限扫视独立持有脉冲、实际发送确认、停稳等待及重新确认道路的状态，不冒充已验证空间。
- `control.session.DrivingSession` 是应用层唯一的电机命令入口。手动STOP优先，然后处理启动或完赛、任务限制、恢复、爬升和租约；相机、触屏和HTTP均不直接发轮速。视觉发车和直接发车共用同一份启动检查。
- `tasks` 各自持有任务状态；网页和屏幕读取其快照。道路识别、任务停止和传输故障不互相替代。

当前固定场地的颜色互斥与预测反证由 `perception.surface.RoadSurfaceEvidence` 提供。只有扣除投影误差余量后仍连续明确为绿色的旧路段，才能撤销历史；白横纹、灰路面和未知区域不产生反证。回放结果及剩余限制见 [评估报告](../../docs/路面识别评估-20261104.md)。

核心几何的Python实现是原生库缺失时的回退，也是数值等价验证的参考，不属于无用代码。构建不再删除导入或拼接全局命名空间。

v0.6.86增加 `navigation.speed_observer.SpeedObserver`：只读当前连续可见路段，调用统一速度包络计算，并在 `/vision`、`/status` 的readiness与 `CONTROL` 日志发布同帧 `speed_profile_shadow`。`mode=SHADOW`、`controls_motor=false`；`source_frame`对应原始帧，`entry_limit_m_s`表示可见路段入口的分析上限，不是当前车速命令。道路不满足条件时明确标记无结果，不沿用旧结果。分析异常不向控制链抛出；耗时计入 `profile_ms.speed_profile_shadow` 及视觉预算。旧的巡线、横道、计圈、发车与看门狗逻辑保持，详见[在线对照记录](../../docs/统一速度规划回放-20261110.md)。

## 现场设置圈数

网页控制台底部的“目标圈数”接受1–100的整数，默认取 `runtime.json` 的 `race.laps`（当前为2）。填写后点击ARM或直接启动，圈数随本次启动请求一起生效；ARM等待期间及行驶中锁定目标。需要修改时先STOP，再输入新值。设置仅保存在当前应用进程，应用重启后使用配置默认值。

现场准备集中在控制台完成：巡线默认显示叠加画面；ESP32 预览直接显示四条黄线，拖动线或圆点即可调整灯面范围。点击“ARM 准备等灯”会一次提交圈数和灯面设置，Maix 等待 ESP32 确认保存后再 ARM。无需单独编辑或保存；点击 ARM 立即锁定当前显示范围，实际检测也使用该固定范围，保存、等灯和行驶期间均不跟随新候选或检测网格移动。“自动定位”只启用辅助找灯，不会 ARM；“撤销调整”放弃本地草稿。准备、等灯或行驶中禁止继续修改范围；保存失败、超时或 STOP 均取消该次准备，不会因迟到保存回执重新 ARM。STOP 常驻底部，诊断和手动测试入口在折叠区。

触屏ARM沿用当前目标；视觉信号与直接启动共用计圈逻辑。实际发车后首次可靠识别的白线若在30秒内出现则忽略该次越线，否则首次实测越线计为第1圈；达到目标并确认最后越线后，按地面光流累计40cm实测前进行程，再直接停车。非法圈数不能触发启动，也不能阻止STOP。

## 本地验证

在项目根目录执行：

```sh
mkdir -p .maixpy/build
cc -std=c99 -O3 -fPIC -shared maixcam2/curve_follow/curve_fit_kernel.c \
  -o .maixpy/build/host_curve_kernel.so -lm
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  CURVE_NATIVE_LIBRARY="$PWD/.maixpy/build/host_curve_kernel.so" \
  .maixpy/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

将 `CURVE_NATIVE_LIBRARY` 指向不存在的文件可验证Python回退路径；原生专用测试会明确跳过。测试共享构造器放在 `tests/support/`，冻结的独立数值参考放在 `tests/reference/`，实拍样本及对应标定放在 `tests/fixtures/`。测试不得依赖 `.maixpy/` 下某台电脑的临时图片。

`test_application` 在模拟硬件边界执行完整预览帧；`test_session` 验证STOP优先、共同启动门槛、慢帧、失路探路租约、通信失败及完赛停车；`test_build` 离线验证部署包、安装回滚与未ARM启动。

## 构建与安装

```sh
.maixpy/.venv/bin/python maixcam2/curve_follow/build.py
.maixpy/.venv/bin/python maixcam2/curve_follow/build.py --drive
```

输出为 `.maixpy/build/curve_follow.zip` 及 `.maixpy/install_curve_follow.py` 或 `.maixpy/install_curve_follow_drive.py`。包内保留 `robot/` 的模块边界及HTML资源，不再生成 `standalone.py`。默认包强制预览；`--drive` 允许使用原配置的驾驶许可，仍默认未ARM。

通过MaixPy技能助手运行安装器：

```sh
./.maixpy/maixpy run .maixpy/install_curve_follow_drive.py --timeout 60
```

安装前由助手停止旧应用。安装器先暂存并检查文件、编译原生库并验证全部符号，再备份和替换本应用文件；注册或启动失败会恢复文件。安装后检查三个新帧及零输出，不自动发车。UNO固件独立烧录。

桌面显示名为 AAA，但启动器按应用 ID 排序。安装器将稳定 ID `robot_curve_follow` 加入 SDK 列表生成器的首个优先项，并核验生成后的顺序，避免后续部署时又排到后面；其余应用保持原有相对顺序。生成器原文随安装备份，安装失败时与应用列表一起恢复。实机修复记录见[AAA排序修复](../../docs/AAA排序修复-20261110.md)。

## 诊断与恢复

`/status` 提供运行和启动状态，`/vision` 原子发布完整当前帧、任务、道路证据及 `profile_ms`；任务快照每帧采集一次，在状态、日志和绘图之间复用。`/status` 的性能快照截止控制发布时，完整绘图与日志耗时见 `/vision`。`control_pipeline` 已包含算法子阶段，不能重复求和；`frame_period` 还包含帧间等待。预览原图、叠加图、地面投影分别用于检查可见性、识别与几何。

设备日志：`/tmp/robot_curve_follow.log`；桌面启动失败日志：`/maixapp/tmp/last_run.log`。本地设备配置、原始录像和部署证据仅放 `.maixpy/`，不入库。清理前副本在 `.maixpy/cleanup-20261103/before.tar.gz`；源码提交不代表设备已更新，应通过 `/vision` 的 `app_version` 核对实际部署版本。
