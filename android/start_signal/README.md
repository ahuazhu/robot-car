# Android 发车灯识别原型

版本 `0.1.1-usbcheck`。这是本机识别、数据采集及独立 USB 诊断应用，**灯光识别尚未接入 UNO/Maix 的发车协议**。主页面 ARM 只启动手机传感器，STOP 只重置本机许可；USB 检查与识别状态机独立，不发送 ARM、运动或解除急停命令。现有小车固件未改变。

## 使用

安装后允许相机权限，使用后置主摄。把整个信号灯放进青框，拖动画面可移动框，滑条可改变大小。正式目标是计时器左侧的独立正方形灯面，右侧圈数和时间数字属于干扰。

按 ARM 后，模型先定位灯面，再确认稳定亮灯至少 4 帧、300ms；锁定实际发光像素。熄灯时同一区域至少 60% 发光像素下降，连续至少 3 帧、400ms 后，本机许可锁存。模型未找到目标本身不算熄灯。曝光锁定在设备支持时启用，背景用于亮度增益与偏移补偿。矩阵灯黑色缝隙不参与熄灯像素比例计算，RGB 最大通道强度用于避免红、蓝灯被低亮度权重排除。

许可确认后，断帧或相机暂停不会清除此 Activity 内的许可；STOP 清除，应用重启回到未 ARM。录像按钮保存 2FPS JPEG 抽帧及约 10Hz JSONL 状态到应用专用目录 `Android/data/cn.robotcar.startsignal/files/recordings/`。当前不是 MP4 编码录像。

## Android OTG / UNO 通信检查

将 USB-C OTG 数据线接手机、USB-B 接 UNO，打开“USB / UNO 通信测试”，首次在 Android 系统弹窗中允许设备访问。USB-B 使用 115200、8N1；Maix 的 A0/A1 通道仍为 9600，两者协议不同，不能把 `@...CRC` 帧写入 USB。

默认检查只接收 UNO 每秒输出的 `DRIVE` 状态。调试构建可通过下列 Intent 启动一次安全收发检查：先确认 Maix 已停车、未 ARM、没有出发许可；手机收到两条相隔至少 500ms 的新鲜零输出状态后，只发送单字节 `i`，等待 UNO 的 I2C 地址探测文本应答，再确认两条新的零输出状态。收到 ACK 或 NO ACK 都证明 USB 往返成功；I2C NO ACK 指向另一条 I2C 总线，不代表 USB 通信失败。

```sh
adb -s PHONE_IP:ADB_PORT shell am start \
  -n cn.robotcar.startsignal/.UsbCheckActivity --ez run_test true
```

诊断最长 14 秒，不提供任意串口发送框；遇到 armed、非零输出、fault、重启或超时就结束，不自动清除故障。USB 打开或 DTR 变化可能使 UNO 复位，因此只在小车停着时检查。诊断不改变 Maix/ESP32 发车链路，手机识别成功也不会直接控制小车。

报告和原始串口日志保存在应用外部专用目录 `Android/data/cn.robotcar.startsignal/files/usb-check/` 的 `latest.json`、`latest-serial.log`。每次重试建立新日志；完成后离开页面不会覆盖最终结果。`success` 是本次诊断整体结果，`bidirectional` 独立表示是否收到本次探测回执；只读成功不能当作双向发送已验证。

驱动使用 [usb-serial-for-android 3.11.0](https://github.com/mik3y/usb-serial-for-android/tree/3.11.0)，USB 授权遵循 [Android USB Host API](https://developer.android.com/develop/connectivity/usb/host)。纯 Java 门禁回归可执行：

```sh
javac -d /tmp/robot-usb-check \
  app/src/main/java/cn/robotcar/startsignal/UsbCheckProtocol.java \
  tests/UsbCheckProtocolCheck.java
java -cp /tmp/robot-usb-check cn.robotcar.startsignal.UsbCheckProtocolCheck
```

2026-11-10 实物测试通过：小米14（Android 16）以 OTG 主机模式识别 Arduino UNO `2341:0043`，CDC ACM 串口为 115200、8N1。约 5.6 秒接收 651 字节、发送一次 `i`，收到对应探测回执；探测前后各两条状态均为未 ARM、左右输出零、fault=0。最初 148 字节启动杂字已排除，未作为状态或回执证据。I2C 回执为 `NO ACK error=2 timeout=0`，它不影响 USB 双向测试通过。Maix 随后的 UNO 零输出回执也持续增长、没有通信错误。没有发送运动指令或改写 UNO 固件；手机原有模型与安装前完全一致。纯 Java 协议回归为119项，APK构建、安装通过。

固件一致性另需核对：实物启动文本显示 `maxPWM=255`，仓库 `DriveSafety.h` 当前为200。因此本次仅证明该实物的USB诊断通信和采样期间零输出，不能据此声称UNO固件与当前仓库完全一致。手机识灯仍未接入正式发车链路。

## 模型与数据边界

- `LampNet160`：ImageNet MobileNetV2 alpha 0.35 冻结编码器，加可训练的分割解码器。
- 输入 float32 `[1,160,160,3]`，RGB `/127.5-1`；输出 float32 `[1,40,40,1]` 灯面概率。FP16 是权重存储格式，输入输出仍为 float32。
- 176,353 参数，FP16 TFLite 文件 367,672 字节。实际模型 SHA256 和逐样本诊断见 `model/model-card.json`。
- 训练来自一段 ESP32 室内试灯录像、早期 Maix 小灯样本，以及用户提供的一张真实龙门灯 **OFF** 截图；数字屏裁剪作为负样本。960 张增强训练图不代表 960 个独立场景。
- Maix 大灯与路人场景按拍摄来源隔离，但已用于观察和调整模型，属于验证集，**不是未接触的测试集**。
- 暂无实际龙门灯 ON 样本，也没有 Xiaomi 14 相机训练数据。不能据此声称比赛发车成功率、跨相机鲁棒性，或已验证计时数字全部不会误检。仍需完整的真实 ON→OFF、保持 ON、遮挡、曝光变化等独立序列。

当前模型的独立场景泛化仍有限。完全遮住灯面与熄灯存在视觉歧义，不能仅凭此原型替换现有比赛发车链路。

## 结构

- `app/.../UsbCheckActivity.java`：USB 设备授权、独占串口读取、单次安全探测及报告。
- `app/.../UsbCheckProtocol.java`：独立的零输出证据、时序与回执门禁，不执行 I/O。
- `app/.../MainActivity.java`：Camera2 采集、同源预览与坐标、青框、录制和状态显示。只分析最新图像，及时关闭 ImageReader 帧。
- `app/.../LampModel.java`：TFLite 推理、灯面连通域和多候选检查。
- `app/.../SignalDetector.java`：不依赖 Android 的亮灭时序、曝光补偿和许可锁存。
- `model/train.py`：本地数据索引、手工关键帧标注、分来源验证、训练与 FP16 导出。
- `model/quantize.py`：INT8 校准和主机回放，生成 ESP32 移植研究用模型；不等同于固件或板上可运行验证。
- `model/SignalDetectorCheck.java`：稳亮、亮灭确认时长、全局曝光、断帧、稀疏 LED 和许可生命周期检查。

## 构建与训练

构建使用 JDK 17、Android SDK 35、Gradle 8.9。配置 `ANDROID_HOME` 或本地 `local.properties`，运行 `./gradlew :app:assembleDebug`，安装 `app/build/outputs/apk/debug/app-debug.apk`。

训练使用 Python 3.11 和 `model/requirements.txt` 的独立环境。在仓库根目录执行：

```sh
python android/start_signal/model/train.py --repo "$PWD" --output .maixpy/android-lamp-model --epochs 36 --samples 960
python android/start_signal/model/quantize.py --repo "$PWD" --model-dir .maixpy/android-lamp-model
```

原始录像与用户截图保留在 `.maixpy/`，不会随源码上传。训练脚本中的真实截图分支在本地参考图不存在时跳过。`diagnostic/` 设备自检图片及 `golden_tests.json` 为本地生成数据，已忽略。

预训练骨干来源：[Keras MobileNetV2](https://keras.io/api/applications/mobilenet/)。Android 推理使用 TensorFlow Lite Interpreter，处理自定义分割输出。

## ESP32-S3

本模型需要另行验证 INT8 量化精度、TFLite Micro 或 ESP-DL 的算子版本、推理 arena、PSRAM 与相机/图传并行内存、实际延迟。`int8-report.json` 的最大单张量字节数不能当作运行峰值内存。本次没有刷写 ESP32，也没有改变现有无线发车功能。


## 代码复核（2026-11-09）

锁定后的背景范围只计算一次，后续帧复用；曝光偏移与增益统计合并遍历。STOP、断帧和重新定位都会清除旧范围。亮灭阈值及本机许可生命周期不变；这仍是独立ON→OFF原型，尚未加入ESP32正式链路的完整OFF→ON→OFF门控与配对通信。

Java状态机检查覆盖换位置后重新ARM及锁定后定位器丢失；与冻结旧版的48组、2,160帧对照中，许可、状态、像素掩膜及诊断数值完全一致。离线APK构建通过。本轮只构建，没有重新安装手机或部署小车。
