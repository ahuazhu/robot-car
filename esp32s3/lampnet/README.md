# ESP32-S3 LampNet 定位与发车检测

当前源码构建为 `--model-locator --full-frame`（0.4.13-exposure-cache，2026-11-09已部署，独立Flash校验与6组板上数值自检通过），支持全画面自动定位与持久化手动框灯。此前 `--model-locator` 构建（0.4.9-lamp-face）：模型先粗定位，C++灯面细化后限制原亮度/颜色检测器的搜索范围；稳定亮灯后冻结灯面核心，仅做快速熄灯判断。发车会话、配对及Maix许可锁存沿用原协议。**开启此选项后，模型参与正式ARM的定位，不再只是旁路观察**。省略此选项仍可生成 `SHADOW_ONLY` 旁路诊断版。普通无模型构建仍使用原规则。

模型来源为 `android/start_signal/model/lamp_model.int8.tflite`，285,240 字节。输入 INT8 `[1,160,160,3]`，输出 INT8 `[1,40,40,1]`。它包含 MobileNetV2 编码器和训练的分割头，使用的算子为 Conv2D、DepthwiseConv2D、Add、Concatenation、Pad、ResizeBilinear 和 Logistic。

## 构建

本机使用 Arduino ESP32 `2.0.12-cn`、Chirale_TensorFLowLite `2.0.0`。后者为 Arduino 适配的 TFLite Micro。不带 `ROBOT_LAMPNET_ESP_NN` 时使用可移植内核作为对照；开启该选项时接入下述 ESP32-S3 SIMD 内核。库只放在本地工作目录中，不修改全局 Arduino 库。

库来源：https://downloads.arduino.cc/libraries/github.com/spaziochirale/Chirale_TensorFLowLite-2.0.0.zip

Arduino 官方库索引 SHA256：`b17dc541ac9ef7fa3766fc1f91b9ca5ab1760970663921f7d55522dbef644322`。

在仓库根目录，用安装了 TensorFlow 2.16 的独立 Python 环境生成模型数组和数值对照数据：

```sh
python esp32s3/lampnet/export_probe.py --repo "$PWD" --out .maixpy/esp32-lampnet-20261109
arduino-cli compile \
  --fqbn 'esp32:esp32:esp32s3:PSRAM=opi,FlashSize=8M,PartitionScheme=huge_app,FlashMode=qio,USBMode=hwcdc,CDCOnBoot=default,UploadSpeed=115200' \
  --jobs 4 \
  --build-property 'compiler.cpp.extra_flags=-std=gnu++17 -DROBOT_LAMPNET_SHADOW=1' \
  --libraries .maixpy/esp32-lampnet-20261109/libraries \
  --build-path .maixpy/esp32-lampnet-20261109/build \
  --output-dir .maixpy/esp32-lampnet-20261109/firmware \
  .maixpy/esp32-lampnet-20261109/sketch/video_preview
```

生成数值对照输入需要本地 `android/start_signal/app/src/main/assets/diagnostic/real-off.png`。这张诊断图由用户提供的截图裁剪而来，已被 Git 忽略。缺失时应先提供真实参考素材，不自动替换成伪造亮灯图。

不带 `ROBOT_LAMPNET_SHADOW` 的普通 `video_preview` 构建不包含模型依赖，仍为原有发车检测固件。

## 运行与诊断

当前推理工作区预留 1MiB PSRAM（早期对照版为3MiB），实际 TFLM 使用量由板子返回；预留量、最大单张量和系统峰值内存是不同概念。

启动先运行已知输入（当前六组，早期两组），与电脑端 `BUILTIN_REF` 参考内核的 INT8 输出比较数值误差。XNNPACK 与参考内核在量化计算中可能不同，不能把两者的逐像素差异当成芯片运行故障。早期固定ROI构建向模型提供青框 `[80,0,160,128]` 的RGB565裁剪；当前 `--full-frame` 构建改用整张 `[0,0,320,240]`，不受青框限制。手动范围保存后跳过模型，仅对所选灯面做时序检测。模型主任务低优先级运行在 core 0；可移植版在节点边界让出 CPU，加速版按8ms时间片让出 CPU。可选的 ESP-NN 双核助手在 core 1 分担支持的内核；相机和原检测继续运行。不会持有发车控制互斥锁等待推理。

- HTTP `/status` 增加顶层 `lampnet`，不改变 `signal` 协议。
- USB `M` 输出模型内存、推理次数、时间、候选框、置信度、自检及错误。
- USB `S` 检查相机帧是否持续增加，`Q` 检查网络。
- `monitor_probe.py` 只发送上述三个只读命令，不发送 ARM 或电机命令。

```sh
python esp32s3/lampnet/monitor_probe.py --port /dev/cu.usbserial-14240 --out .maixpy/esp32-lampnet-20261109/device --seconds 90
```

训练数据边界与 Android 原型相同：训练时实际龙门灯只有一张 OFF 参考图，没有同款 ON 录像；历史来源验证集的泛化仍有限。下述新增实物录像仅作验证，未加入当前模型训练。板上数值一致与速度测试不能当作实际发车成功率。

## 刷写与回滚

先备份当前固件并校验版本。原分区 app0 在 `0x10000`，上限 `0x330000`。只写 app0；保留原引导、分区表、NVS Wi-Fi/配对与 OTA 选择。备份可能含私有配置，只保存在 `.maixpy/` 并限制访问权限。

固件、构建日志、原版镜像、写入验证与板上诊断均放在 `.maixpy/esp32-lampnet-20261109/`。模型故障时不得把“没有目标”当成发车事件。

## 本次板上记录（2026-11-09）

已以 app0 方式部署 `0.4.7-lampnet-probe`，写入校验通过，Wi-Fi/配对 NVS 保留。两个与电脑参考内核的对照输入逐字节一致。实际 TFLM 工作区 658,512 字节，预留3MiB PSRAM；稳定实时推理约3.54秒/次，40秒检查中原相机帧持续增加，约6.8FPS。模式仍为 `SHADOW_ONLY`，模型未接管发车定位。后续实物验证结果见下节。

只读实测控制台：`verify_console.py --serial /dev/cu.usbserial-14240 --out .maixpy/<trial> --port 8130`，浏览器打开 `http://127.0.0.1:8130/`。青框表示搜索区域，绿框表示模型判定的灯面候选；服务器只发送 M/J，不发送 ARM。JPEG 与模型状态持续保存到本地试验目录。

## 实物灯验证（2026-11-09）

已记录用户确认的实物比赛灯“亮→灭→亮”。本轮未发送 ARM、发车许可或电机命令；模型始终仅旁路观察。结论是**定位稳定性未通过**，不应据此让模型接管发车。

JPEG 回放显示模型掩膜越过实际灯面，覆盖光晕、天花板与横梁背景；约 0.93–0.95 的置信度没有对应准确的灯面边界。板上候选几何检查也反复失败。单次板上推理中位数约 3.55 秒，不能作为逐帧熄灯检测器。

用户回复是状态确认，存在延迟，并非开关瞬间的时间戳。已在录像中核对并选用三个稳定短区间：首次亮灯 425–450 秒、熄灯 540–570 秒、再次亮灯 760–790 秒；不能用本轮估算熄灯响应延迟。独立推理次数分别为 7、8、8，候选通过几何检查分别为 3、4、3；这些相关重复样本的通过比例不是定位准确率，也不是泛化召回率。

在人工选择的固定灯面核心区域中，灰度中位数约为 251→205→251，最大 RGB 中位数约为 253→235→253。该区域不是模型输出，说明画面存在可测的亮灭变化；不代表现有自动定位、熄灯确认或整车发车已经通过。后续应修正真实灯面分割边界，并在锁定后联合灰度、颜色与背景补偿进行快速时序确认。

原始 JPEG、模型状态、用户状态标记及报告保存在本机 `.maixpy/esp32-lampnet-20261109/trial-122729/`；报告为 `report/index.html`，掩膜对照图为 `report/comparison.jpg`。绿色掩膜来自 JPEG 的参考内核回放，不等同于板上 RGB565 的逐像素输出。当前录像未用于训练；若后续纳入训练，必须另外录制独立验证片段。

## 硬件加速版（2026-11-09）

已部署 `0.4.8-lampnet-espnn`。模型文件、输入尺寸与识别阈值保持不变，仍为旁路诊断，不接管发车许可。此项优化解决运行开销，未解决前述灯面分割不准的问题。

| 构建 | 持续推理中位数 | 采样次数 | 相机帧率 |
| --- | ---: | ---: | ---: |
| 原可移植内核 | 3551ms | 5 | 6.76FPS |
| 单核 SIMD 卷积 | 823ms | 24 | 6.61FPS |
| 双核 SIMD、加法与 O2 | 743ms | 53 | 6.34FPS |

最终持续测速120秒，推理中位数约快4.8倍；统计为USB周期采样，未记录每次推理，不是检测准确率或严格实时上限。预处理约24ms另计；冷启动的自检/联网不包含在持续测速中。六组固定输入（含实物亮灯、灭灯、背景）的输出均与 TFLite `BUILTIN_REF` 逐字节一致。10次USB JPEG完整性检查和68项现有发车/预览回归检查通过。测速期间未发现重启或堆内存持续下降；Wi-Fi连接保持。电脑不在热点网段，本次没有完成HTTP端到端访问验证。

实现包含：

- 31个卷积、13个深度卷积和8个同形状INT8残差加法接入 ESP-NN；不支持的张量类型、广播或膨胀配置在准备阶段明确失败，不猜测执行。
- 使用乐鑫 S3 向量内核，并可选择双核分工。保留相机与发车检测任务，未关闭看门狗。
- 模型层间共用137,888字节PSRAM工作缓冲；内核优先使用32KiB、16字节对齐、DMA可访问的内部RAM缓冲。TFLM实际工作区664,128字节，预留从3MiB降为1MiB。
- 移除模型工作线程的主动串口日志，数值自检结果由USB `M`查询返回，避免和JPEG数据交错。
- 增加卷积、深度卷积、加法、缩放及调度让出时间的独立计时。

ESP-NN 固定为 [v1.4.2 / 744d48d](https://github.com/espressif/esp-nn/tree/744d48deae6aa4a4d7b93b6a80f548d5298813a4)，Apache-2.0。归档SHA256为 `3a895a6d0f35d0eda8adbe368c34e0615825191c6d9e49411cb06cd4b910fbdd`。`prepare_esp_nn.py`验证归档、提取官方 common/S3 源文件及许可证；C源正文保持不变，通过生成的编译包装启用官方建议的 `O2`、`no-unroll-loops`。不修改全局Arduino库。

在上述TFLM依赖准备好后，可复现加速构建：

```sh
curl -fL https://codeload.github.com/espressif/esp-nn/tar.gz/744d48deae6aa4a4d7b93b6a80f548d5298813a4 -o /tmp/esp-nn.tar.gz
python esp32s3/lampnet/prepare_esp_nn.py --archive /tmp/esp-nn.tar.gz --out .maixpy/esp-nn
python esp32s3/lampnet/export_probe.py --repo "$PWD" --out .maixpy/lampnet-accelerated --dual-core
arduino-cli compile \
  --fqbn 'esp32:esp32:esp32s3:PSRAM=opi,FlashSize=8M,PartitionScheme=huge_app,FlashMode=qio,USBMode=hwcdc,CDCOnBoot=default,UploadSpeed=115200' \
  --jobs 4 \
  --build-property 'compiler.cpp.extra_flags=-std=gnu++17 -DROBOT_LAMPNET_SHADOW=1 -DROBOT_LAMPNET_ESP_NN=1 -DCONFIG_NN_OPTIMIZED=1' \
  --build-property 'compiler.c.extra_flags=-DCONFIG_NN_OPTIMIZED=1' \
  --build-property 'compiler.S.extra_flags=-DCONFIG_NN_OPTIMIZED=1' \
  --libraries .maixpy/esp32-lampnet-20261109/libraries \
  --libraries .maixpy/esp-nn/libraries \
  --build-path .maixpy/lampnet-accelerated/build \
  --output-dir .maixpy/lampnet-accelerated/firmware \
  .maixpy/lampnet-accelerated/sketch/video_preview
```

省略 `--dual-core` 可构建单核加速版。可重复添加 `--golden-image /absolute/path/to/320x240.jpg`，数值自检裁剪区域为 `[80,0,160,128]`；加上默认两组，最多八组。无需重新训练。相同模型的数值验证不能替代灯定位及亮灭动作验证。

本机完整记录在 `.maixpy/esp32-opt-20261109/report.json`，最终镜像在 `firmware-dual/video_preview.ino.bin`。写入时仅更新app0 `0x10000`，未覆盖NVS、分区表或引导。回滚镜像为该目录 `backup/restore-0.4.7-app0.bin`；用相同串口与115200波特率仅写回app0即可。

## 加速后定位回放（2026-11-09）

使用相同INT8模型，按固件的连通域选择与几何过滤规则，回放9张历史实物灯JPEG（亮、灭、再次亮各3张），并与逐张查看后的近似人工灯面框对照。7/9候选通过原几何过滤；框IoU中位数0.398，2/9达到0.50，0/9达到用于贴合判断的0.70。框宽约为灯面的1.8–2.0倍，各状态框内背景占比中位数约51%–65%。人工框每条边±3像素的敏感性检查也没有达到0.70。结论仍为定位不准确；加速未修复分割边界。

这些是同一摆位的相关画面，仅能评价本轮录像，不是泛化准确率或召回率。本轮是电脑端参考内核回放，未新录制板上视频；此前六组固定输入的板上数值一致检查独立保留。黄色为人工参考框，红色为候选框；被几何过滤拒绝的候选也显示，不能把图上所有红框都当作控制台已接受的框。

报告、人工标注和逐帧对照位于本机 `.maixpy/esp32-opt-20261109/localization-retest/`。可用以下命令复现（需安装TensorFlow与OpenCV的Python环境）：

```sh
python esp32s3/lampnet/replay_localization.py \
  --model android/start_signal/model/lamp_model.int8.tflite \
  --trial .maixpy/esp32-lampnet-20261109/trial-122729 \
  --annotations .maixpy/esp32-opt-20261109/localization-retest/annotations.json \
  --out .maixpy/esp32-opt-20261109/localization-retest
```


## 灯面细化版（2026-11-09）

`lamp_face_locator.h` 是固件和电脑回放共用的有界C++实现。低阈值保留完整弱候选；自适应高置信核心提供亮度/颜色参考，通过连通域及开运算去掉光晕连接，再留一格量化边界。细化不可用时回退原有效候选，多目标歧义仍拒绝。模型工作区和细化工作区都在PSRAM，模型权重未变。

正式ARM时，只有新鲜（源图不超过2.5秒）的有效模型框可以用于找灯。原亮灯基准仍要求4帧且300ms；锁定后忽略模型框变化及丢失，停止新增模型推理，固定核心像素继续以相机节奏检查曝光补偿后的亮度变化。熄灯仍要求至少60%核心变暗、3帧且400ms。确认后结果保持到cancel；修复主循环中的2秒租约清理也必须跳过已确认事件，与相机任务保持一致。

导出时加入 `--dual-core --model-locator`，然后按上节相同ESP-NN编译参数编译即可。生成草图内定义 `ROBOT_LAMPNET_FACE`，不要直接改全局库或添加与缓存冲突的编译选项。

新增9项本机C++契约检查，加上已有发车/预览检查共77项通过。旧实物录像9张调参图的框IoU中位数0.398→0.895；另外28张同摆位图0.441→0.886；122张缓存图中未丢失原有效正候选。两组实物图片均在开发中查看，不能算独立泛化验证；历史样本也与训练重叠，仅用于回归。历史大灯仍有定位不准及漏候选，干扰图仍会产生候选。

157张旧录像JPEG按160ms间隔赋时的C++流程回放能够锁定、确认熄灯并保持结果到重新亮灯，亮灯稳定段未误触发。原USB录像约2秒一张，超出检测器750ms连续性要求，所以原时间戳回放不锁定；这项赋时测试不作为实测延迟。17张旧Maix抖动图重新按固件ROI推理并进入同一时序内核，没有熄灯事件；不能据此估计现场误发车率。

`native_locator.py` 和 `replay_face.py` 可在不打开硬件的情况下重放缓存模型输出，生成逐帧JSON和对照页：

```sh
python -m esp32s3.lampnet.replay_face --cache .maixpy/esp32-refine-20261109/cache.npz \
  --samples .maixpy/esp32-refine-20261109/samples.json --out .maixpy/esp32-refine-20261109/report
```

独立实物验证：

```sh
python esp32s3/lampnet/verify_console.py --serial /dev/cu.usbserial-14240 \
  --out .maixpy/<new-trial> --port 8131 --diagnostic
```

`--diagnostic` 仅通过USB T开启另一份检测器、退出用U清理，不创建正式Maix会话，不发送发车许可或电机命令；正式检测已ARM时拒绝开启。正式ARM会取消独立测试。M返回 `diagnostic.native_armed`、`phase`、`roi` 和熄灯确认信息。绿框在锁定后显示冻结灯面，不再追随新的候选。

本轮固件、回放及板上记录保存在 `.maixpy/esp32-refine-20261109/`；回滚镜像 `backup/restore-0.4.8-app0.bin`，仅写回app0 `0x10000`，不要改NVS、原分区或引导。新的距离/摆位实物测试结果以该目录的现场录像为准。


## 全画面与手动兜底（0.4.11）

用户要求取消青框限制后，`--full-frame` 将模型搜索范围改为整张320×240，不再裁去青框外的灯。输入仍为160×160，框坐标按整张图映射。时序检测采用4×4像素块保持工作区规模约80×60，避免扩大到占满内部RAM；旧固定框构建仍用2×2块。全画面缩小会降低极小灯的细节，前述0.4.9固定ROI的IoU统计不能直接套用到此模式。

当前USB预览 `verify_console.py` 加载同目录的 `verify_console.html`。在 http://127.0.0.1:8131/ 默认由模型自动放置左、右、上、下四条黄线；拖动任意黄线即手工调整，再点“保存调整”。范围通过USB R配置保存为一条原子NVS记录，拔掉电脑或重启后仍保留。配置时不发送ARM或电机命令；正式检测已ARM时拒绝改范围，必须先STOP。可点击“重新自动定位”恢复全画面模型定位。

手动模式绕过模型和自动连通域找灯，固定使用矩形内部核心，避免边缘混入背景；初始仍需用户保持亮灯建立稳定基准。它不要求灯比背景更亮。确认阶段复用曝光补偿、至少60%核心变暗、3帧且400ms的条件；Lab亮度是熄灯主判据，同时提供灰度下降、RGB最大值下降、饱和度变化作为诊断。仅颜色变化不单独触发。手动框选错误、遮挡或移动灯仍可能误判，操作时应避开数字屏、亮横梁，并固定灯与相机。

预览“重新锁定（独立验证）”只启动独立检测器，不影响Maix会话；每次改变范围会取消旧验证基准，需重新锁定。正式比赛仍在Maix按ARM。模型推理在手动模式停用；ESP32独立执行判断并按原配对协议向Maix提供一次性确认。

本机新增全画面映射、青框外亮灭、手动弱对比灯、框外干扰和整体变暗的契约检查，总计83项通过。17张旧抖动图在全画面模式中未产生熄灯事件；6张当前新摆位和9张旧摆位JPEG均可定位并在重复帧基准检查中锁定。这些是开发过程中的回归回放，不是独立检测率或现场响应延迟。现场实测、固件SHA、配置保存验证以 `.maixpy/esp32-refine-20261109/` 的报告为准。


现场独立验证已完成：用户手调范围 `[61,59,64,53]` 保存成功；亮灯稳定，熄灯后4帧、543ms确认，重新亮灯保持结果。确认时核心变暗比例100%，灰度下降65.1、RGB最大值只下降0.6、饱和度变化0.489，因此没有要求各项指标同时下降。这里的543ms是板上确认跨度，不是物理开关到车辆启动延迟。USB重连重启后范围及手动模式仍保留，六组模型对照仍逐字节一致、相机帧持续更新。全程正式Maix检测未ARM，没有车辆运动。

预览已移除绿色内部锁定框，只显示四条黄线。自动模式下线跟随候选，开始拖动后停止跟随，保存后按手调范围检测；页面刷新保留ESP32已保存的配置。最新实物比赛照片显示黑灯面有相邻数字屏的蓝色漏光，不能把熄灯定义为RGB全零。当前仍用亮灯基准与曝光补偿后的持续变暗判断，颜色及灰度作为诊断，尚无同款赛场灯的完整多色ON/OFF序列泛化验证。


## 现场发车顺序修正（0.4.12）

现场顺序由用户确认：摆车及准备时灯灭，选手示意ready后停止操作；裁判亮灯，几秒后熄灯，此时发车。正式ARM和USB独立验证均改为完整OFF→ON→OFF流程，准备阶段无需先亮灯或在亮灯后操作。

0.4.11手动模式只以绝对亮度下限24判断可建立基准，初始灰色背板可能被当成ON，随后真正亮灯时又不会更新该基准。这会漏掉后续熄灯。现在ARM先保存初始灭灯的整个视野作为参考，即使模型尚未定位目标也保留该参考；在手调范围或后来找到的模型范围内，以背景曝光补偿后的持续变亮证据门控原4帧/300ms亮灯确认。稳定ON后冻结灯面及ON基准，后续才允许原3帧/400ms的OFF确认。初始灭灯继续变暗不会放行；单帧/短闪光不能成为ON基准；STOP或确认前图像连续性中断会清除旧周期证据。

复用现有参考工作区，未增加整帧缓存。手动模式仍跳过模型推理。另修复模型自检成功后未清掉SELFTEST_RUNNING诊断文本的问题。

Maix 0.6.77增加对ESP32全画面ROI `[0,0,320,240]` 的合法校验，并显示WAIT_ON“已准备，等待裁判亮灯”。旧客户端只接受128×112和160×128，会拒收0.4.11起的全画面响应；因此正式使用全画面固件前，需要两端同步。目标框必须仍在已声明搜索区域内，认证、会话及许可锁存规则不变。

92项原生定位、完整灯周期、发车协议及预览检查通过；覆盖白/红/绿ON、长时间初始灰板、短闪光、整幅变亮、模型在ON后才找到灯、STOP重来及断帧不能复用旧ON。此为离线C++/协议检查，板上实物完整周期与Maix部署结果另记录在 `.maixpy/esp32-cycle-20261109/`。


## 热路径去重（0.4.13，未部署）

OFF→ON→OFF、所有阈值、模型权重及配对/会话协议保持不变。等待亮灯和确认熄灯共用背景曝光统计，一次扫描同时累积加性、增益及拟合直方图，去掉原来额外一次遍历。手动范围和锁定阶段不再计算仅自动定位使用的颜色分类；全画面模式每帧少做4,800个颜色分类。固定范围的背景/光晕掩膜按几何位置缓存，改变范围或重新ARM时重新建立。自动模式断帧后先清除旧锁定，再准备新定位所需的颜色数据。

USB配对和Wi-Fi配置原先分别硬编码0.4.3与0.4.6，导致当前固件被拒绝。两者现在复用 `video_preview/device_protocol.py` 中的已知固件清单；未知固件仍不会收到密码或配对密钥。

96项定位、完整灯周期、发车协议及预览检查通过。冻结0.4.12原生内核与新内核对照112组、3,086帧，导出的状态、诊断浮点数、灯面掩膜逐帧完全一致，覆盖固定/全画面、自动/模型/手动、不同颜色、曝光、断帧、STOP重来、时间回绕与旧实物JPEG。旧JPEG重放使用赋予的160ms间隔，不能当成现场延迟。

同机直接C++ `-O2` 交替测试11轮、每轮700帧：手动等待亮灯297→256µs（约14%），手动/自动锁定后328→293µs（约11%），自动定位668→621µs（约7%）。这是主机热路径基准，不能当作ESP32板上收益或圈速提升；模型推理本身没有改变。编译产物Flash 1,711,053字节、全局RAM 132,780字节。对照脚本、结果与固件仅保存在本地 `.maixpy/signal-review-20261109/`。

本轮未刷机、未ARM、未移动小车。正式使用前仍需部署后做完整实灯OFF→ON→OFF及整车回归；当前版本没有新增实灯成功率或整车速度的证据。
