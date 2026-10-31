# 机器人小车换机交接

交接日期：2026 年 10 月 31 日（Asia/Shanghai）。主项目是 `robot-car`，当前应用版本 **v0.6.34**，原开发分支为 `experiment/speed-200`。本包保存源码、未提交工作、Git 历史、相关 Codex 会话、MaixPy 技能及本机调试证据。

**接手先处理：确认 MaixCAM2 的当前 IP 和连接；部署本地尚未上车的发车灯性能优化；复验发车灯、白线计圈和三圈停车。** 当前没有完整的三圈实车验收结论。读取本交接及最新报告后，再恢复硬件调试。

## 包内内容

| 路径 | 内容 |
| --- | --- |
| `projects/robot-car/` | 主仓库完整工作快照、Git 分支及标签、`.maixpy` 调试记录；最新工作已保存到交接分支 |
| `projects/uno-legacy/` | 原 `~/work/uno`，早期 Arduino 实验和比赛规则 DOCX |
| `skills/maixpy/` | 正在使用的完整 MaixPy 技能，包含已验证的 launcher 兼容补丁 |
| `skills/reference/` | 本机仍可读取的相关通用技能参考快照，不自动安装 |
| `sessions/` | 相关 session 原始记录、分页历史导出、会话索引和可读转录，详见其说明 |
| `references/` | 能找到的项目会话附件及补充资料 |
| `metadata/` | 源环境版本、Git 差异、排除清单、外部资源清单、测试日志及提交信息 |
| `restore-project.py` | 在新电脑复制项目及技能、重建本机启动入口；支持预览和校验 |
| `MANIFEST.sha256` | 包内文件逐项 SHA256 校验清单 |

设备密码文件、Codex 登录文件、SSH 私钥、全局账号配置没有打包，也没有上传 GitHub。虚拟环境、Python 缓存不跨机复制，按下述步骤重建。其余 `.maixpy` 录像、图片、标定、运行日志、旧固件和生成程序均保留；历史本机二进制只作证据，换架构后需重编译。

旧聊天中部分原始临时附件已被系统清理，现存附件会另存并建立映射；session 内嵌图像仍保留。具体缺失项见 `sessions` 附件清单及 `metadata/external-resources.json`。

## 在新电脑恢复

先解压 ZIP，保留隐藏文件。在解压目录执行以下命令；Windows 可以把 `python3` 换成 `py -3`。

```sh
python3 restore-project.py --verify --dry-run
python3 restore-project.py --destination ~/work
```

默认恢复到 `~/work/robot-car`、`~/work/uno-legacy`，安装技能到 `~/.agents/skills/maixpy`。可用 `--skills-dir` 改技能目录。脚本会预检目标冲突，已有不同内容时拒绝覆盖。只想查看代码时，也可以直接打开包内 `projects/robot-car`。

脚本不会运行小车、联网安装软件或改写现有 Codex 数据库。恢复后的 `.maixpy/maixpy`、Windows `.cmd` 会使用新机路径，旧入口保留为 `maixpy.source-host.txt`。

安装 Python 3.11、Git、C/C++ 编译器、Arduino IDE 或 CLI，以及系统 SSH/SCP。随后在恢复后的 `robot-car` 中：

```sh
python3 -m venv .maixpy/.venv
.maixpy/.venv/bin/python -m pip install -r requirements-handover.txt
```

Windows 虚拟环境解释器为 `.maixpy\.venv\Scripts\python.exe`。精确源环境版本见包内 `metadata/requirements-source-host.txt`；其中包含本机辅助工具，跨平台安装遇到不兼容时以项目所需的最小依赖为起点。不要同时装多种 OpenCV Python 发行包。

原环境为 macOS x86_64、Python 3.11.17；主机 NumPy 2.4.6、opencv-python-headless 5.0.0.93、pyserial 3.5。旧设备配置报告为 Python 3.13.2、OpenCV 4.12.0、NumPy 2.2.6，换机时未重新连接设备核对。

Arduino 主固件选择 `arduino:avr:uno`，本机 AVR core 为 1.8.8。ESP32 预览是历史独立分支功能，其板型和参数见 `esp32s3/video_preview/README.txt`；旧环境为 esp32 core 2.0.12-cn，不要把 UNO 配置用于 ESP32。

## 继续 Codex 工作

最稳妥的接续方式是在新电脑登录 Codex，打开恢复后的 `robot-car` 作为项目，并把下文接手提示发到新聊天。需要追查历史时读取 `sessions` 中的索引和转录。

主调试聊天标题：**Troubleshoot Arduino upload error**；ID：`01a13f97-d63b-7533-9da6-5ec32a7be0a4`。标题来自早期 UNO 问题，后续主要开发都在这条聊天里。

比赛选型聊天 ID：`01a132fc-ff45-74a1-985e-4fc11da61a81`；另有早期 ESP32 选题聊天和相关辅助排查会话，均见索引。

这些会话使用分页存储，单独复制 rollout JSONL 不足以保证新机桌面聊天列表恢复。本包同时保存按项目筛选的历史数据，没有复制包含其他项目的全局数据库。跨电脑原生导入尚未验证，不应覆盖新机整个 `~/.codex`。若日后在兼容版本中完成原生导入，可使用 `codex resume <会话ID> --cd <新项目路径>`；该命令用于继续已登记的会话，并不是 ZIP 导入器。官方参考：[Codex CLI](https://learn.chatgpt.com/docs/codex/cli)、[App Server 分页会话](https://learn.chatgpt.com/docs/app-server)。

可直接发送给新聊天的提示：

> 这是 miniAuto 机器人视觉竞速小车项目。请先读 HANDOVER.md、maixcam2/curve_follow/runtime.json、app.yaml，以及 .maixpy/lamp-start-focus-20261031/report.json。当前 v0.6.34，最新发车灯性能优化已构建但未部署，旧设备 IP 已超时。先核对本机环境、MaixPy 技能和设备连接，保持静止检查；随后续接尚未部署优化的预览验证，再安排实际发车、白线计圈与三圈停车验证。不要把历史版本的成功反馈当作最新版本完整验收。历史聊天及图片在迁移包 sessions 中。

## 当前代码与实际行为

主入口 `maixcam2/curve_follow/main.py`，设备应用 ID `robot_curve_follow`，桌面名称 `AAA`。`app.yaml` 为 0.6.34，遥测版本 0.6.34-lamp-start。旧 README 开头停在 v0.6.29，后面逐条追加到 v0.6.34；以当前源码、runtime 和本交接为准。

| 模块 | 职责 |
| --- | --- |
| `tracker.py`、`geometry.py`、`road_model.py`、`ground_motion.py` | 道路边线、中线拟合、地面投影、运动估计和历史关联 |
| `steering_control.py`、`road_policy.py`、`road_recovery.py`、`controller.py` | 前瞻追踪、转向和速度控制、有限丢线恢复 |
| `crosswalk.py` | 横道检测、合法停车窗口、3.3 秒礼让与事件关联 |
| `start_light.py` | 全画面发车灯匹配、0.90 匹配门槛、0.35 秒连续及时画面确认 |
| `lap_counter.py` | 横向白线独立计圈，首次驶过不计，此后三次返回后停车 |
| `drive_link.py`、`motor_channel.py` | UART 协议与独立电机通信进程 |
| `curve_fit_kernel.c`、`native_geometry.py` | C 内核及 ctypes 接口 |
| `uno/uno_lane_drive/` | UNO 电机输出、CRC/序号检查、350ms 看门狗、刹车灯 |

网页地址为 `http://设备IP:8000/`。**直接启动**及触屏 START 立即请求进入同一套启动检查；**ARM** 等待灯光确认；**STOP** 取消启动/等灯并停车。计圈当前只依赖白线，龙门架已移走，不再要求龙门架匹配。

`runtime.start_light.auto_arm_on_launch=true`：从设备桌面打开程序会等待发车灯，可能在亮灯后自行发车。远程安装器使用 `START_LIGHT_DISARMED=1`，部署后保持未布防停车。不要把“部署后停车”推广成“任何启动方式都不会自动发车”。新一轮比赛计数清零；第三圈停车后不因灯仍亮重启。

## 硬件和当前参数

底盘是 miniAuto 麦克纳姆轮小车，MaixCAM2 负责视觉，Arduino UNO 负责电机；ESP32-S3 摄像头预览代码保留作历史参考。

- Maix A21 UART4 TX → 电平转换 → UNO A0 RX；UNO A1 TX → 电平转换 → Maix A22 RX。UNO A4/A5 保留 I2C。旧 `docs/串口测试说明.txt` 的 A4/A5 串口接法已过期。
- 设备间串口 9600，UNO USB 日志 115200；刹车灯接 UNO A2，当前高电平有效。
- 轮序 RF=M2、LF=M3、LR=M4、RR=M1；PWM 10/9/6/11；DIR 12/8/7/13。
- Maix 与 UNO 输出上限均为 200；350ms 无有效 DRIVE 时 UNO 停车。USB `s` 急停，`u` 清除急停，`b` 静止灯测试，`i` 静止 I2C 探测。
- 当前转向增益 0.85，弯道基础巡航 120，直道 163，加补偿后封顶 200；寻路基础 60，视觉时效预算 280ms。
- 车长 18cm、宽 17cm、轴距 11cm。当前地面标定采用 10 月 31 日镜头姿态及 30/60cm 横线、30cm 路宽。

镜头重装或角度改变需要重新标定。当前标定仍记为 `drive_validated=false`，横道停车也保留待实车验证标记。横向光学偏置已经吸收在投影里，不要重复平移。

## 最后进展与接手顺序

最后部署记录是 `.maixpy/runs/20261031-134138-install_curve_follow_drive/`，报告初始静止和三帧遥测检查通过。之后本地又优化了发车灯候选检查、积分图亮度统计等，已经构建，因设备离线而没有部署。具体事实见 `.maixpy/lamp-start-focus-20261031/report.json`。

已部署预览达到 `SIGNAL_READY`；灯检测中位 29.95ms，整帧中位 236.55ms，峰值 311.4ms。峰值仍超过 280ms 预算，不能说性能已完全达标。报告 `source_sha256` 是 `start_light.py` 的哈希，不是整台设备固件哈希。

最后连接地址 `10.127.108.209` 的 SSH 22、HTTP 80/8000 均超时；旧地址 `192.168.3.244` 仅作历史参考。最后记录为停车、未布防、输出 0；这是历史状态，不是本次迁移的实时遥测。

1. 新机打开项目，读取已安装 `maixpy/SKILL.md`；运行 `doctor`、`device get`。确认设备开机及同一网络，必要时用 `device scan`，从设备“设置 → 设备信息”核对地址。
2. 在本地重新配置 `.maixpy/device_credentials.local.json` 或在终端输入密码，不要把密码发到聊天。使用 `device set` 更新 IP，再 `device check`。主机没有 sshpass 时可使用终端交互登录。
3. 先保持车静止，核对相机姿态、标定、UNO 固件、应用当前版本及通信。摄像头不能同时被 MaixVision 和当前应用占用。
4. 用 `build.py --drive` 生成本地最新安装器，经 MaixPy helper 部署，核对静止遥测和实际版本。先做发车灯亮/灭、图像边缘、过曝及耗时验证，再试实际发车。
5. 接续白线计圈：确认首次越线跳过，后续每圈只计一次，三圈停车。历史曾有计至 2/3 的记录，也有漏圈；v0.6.34 没改计圈逻辑。
6. 综合复验当前姿态下直线偏摆、弯道连续性和连续横道礼让。横道停车要求整车不压线且在 20cm 内，工程等待 3.3 秒。既有“来回很好”等用户反馈不代表新版完整赛程验收。

关键证据目录：`.maixpy/lamp-start-focus-20261031`、`lap-fix-20261031`、`start-readiness-20261031`、`road-search-stop-20261031`、`calibrations/20261031-new-pose`、`baselines` 和 `runs`。

ESP32 的 `esp32s3/backups/before_preview_first_1MiB_20261022.bin` 仅备份闪存前 1MiB，不是完整 8MiB 恢复镜像。

## 本次验证和遗留问题

本次在独立迁移快照中执行验证，没有部署或驱动车辆。

| 检查 | 结果 |
| --- | --- |
| 巡线 `test_curve*.py` | 328 项中 323 通过，3 failure、2 error，均位于 `test_curve_follow.py` 的历史图像用例 |
| 原目录复核 `test_curve_follow.py` | 同样 3 failure、2 error，证明并非复制目录才出现 |
| 横道 `test_crosswalk*.py` | 69 项通过 |
| MaixPy 技能 | 24 项通过 |
| UNO C++ 边界/看门狗 | 通过，含两段模拟 2 小时运行 |
| UNO Arduino 实际编译 | 通过；10740B flash、644B RAM，未烧录 |
| `build.py --drive` | 成功生成 standalone 和安装器，未运行安装器 |

历史图像失败的初步线索是旧照片使用当前 10 月 31 日新姿态标定；需按拍摄日期固定测试标定后验证，尚未完全证明根因。此次迁移保留原代码，不擅自改变控制算法或测试断言。

直接把所有 `test_*.py` 放一个进程发现测试，还会遇到早期实验同名 `controller`/`tracker` 导入冲突及旧路径问题；应按系列分进程。首次沙箱内测试因 HTTP 端口权限另报错，后续允许本机端口的巡线复测已排除该因素。详细日志在包内 `metadata/verification/`。

macOS/Linux 下可按以下方式复查：

```sh
mkdir -p .maixpy/speed-profile
cc -std=c99 -O3 -fPIC -shared maixcam2/curve_follow/curve_fit_kernel.c \
  -o .maixpy/speed-profile/host_curve_kernel.so -lm
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  CURVE_NATIVE_LIBRARY="$PWD/.maixpy/speed-profile/host_curve_kernel.so" \
  .maixpy/.venv/bin/python -m unittest discover -s tests -p 'test_curve*.py'
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .maixpy/.venv/bin/python -m unittest discover -s tests -p 'test_crosswalk*.py'
c++ -std=c++11 -Iuno/uno_lane_drive tests/test_drive_safety.cpp -o .maixpy/test_drive_safety
.maixpy/test_drive_safety
python3 maixcam2/curve_follow/build.py --drive
```

Windows 本机 C 库构建需适配编译器，或在有完整工具链的 Linux/WSL 环境做 native 回归。历史二进制不能直接复用到不同系统。

## GitHub 和离线历史

代码仓库：[ahuazhu/robot-car](https://github.com/ahuazhu/robot-car)，保持私有。交接分支 `handover/2026-10-31` 从源分支 HEAD `fe7d3ff` 保存当前工作，并纳入本交接、最小主机依赖和 `legacy/uno` 早期源码。原开发目录、分支和未提交工作不改动；`main` 也不改动。

GitHub 只保存代码及项目文档，完整会话、`.maixpy` 证据和比赛 DOCX 随本地包交付。包内主仓库 `.git` 保留原本所有本地分支及标签；另有 `metadata/robot-car.bundle` 可离线恢复 Git 历史。确切提交与远程核验结果见 `metadata/git-publication.json`。

仅从 GitHub 接手时：

```sh
git clone --branch handover/2026-10-31 git@github.com:ahuazhu/robot-car.git
```

完整接手优先使用迁移包恢复，以保留未在 Git 中保存的实验照片、录像、标定证据和会话。
