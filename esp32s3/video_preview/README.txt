待部署模型构建0.4.13-exposure-cache（已编译，未刷机；设备仍是0.4.12）：自动模式搜索整张画面，不受青框限制；USB预览可用四条线手动框灯并保存到ESP32，跳过自动定位。灯灭时准备并按ARM，自动等待稳定亮灯，再确认熄灯。全画面响应需Maix 0.6.77或以上版本。普通无模型构建仍为0.4.6。构建、独立验证、限制和回滚见 ../lampnet/README.md。

ESP32-S3 发车灯检测与无线预览（2026-11-07）

硬件：Hiwonder ESP32-S3 / GC2145，8MB flash、OPI PSRAM。
固件：0.4.6-larger-roi。搭配 Maix 巡线应用0.6.72。
ESP32 上仰，只识别龙门架发车灯；Maix 下视巡线并决定是否启动。
UNO 继续执行 Maix 电机/刹车灯指令，不需要修改。本固件不提供原0x52颜色数据。

日常使用：
  打开 http://10.73.124.209:8000/，两路画面和控制同页。
  原电脑入口 http://127.0.0.1:8123/ 跳转到该控制页。
  ESP32和Maix连接同一可互访的2.4GHz网络，发车许可经Wi-Fi传送，不依赖电脑。
  USB只需供电，可改接稳定5V充电宝；换电源时会重启，自动重新入网。
  ESP32原图320x240；青色定位框 x80,y0,w160,h128，中心保持(160,64)不变。黄色框是锁定灯面的外包矩形，实际只检测分割得到的灯面核心像素。
  让完整灯面处于框中央、四周留余量，车和相机摆稳后在Maix按ARM。
  先确认稳定亮灯，再对同一灯面连续至少3帧、0.4秒确认熄灯；变亮不发车。
  发车还需道路和电机通信就绪；Maix收到有效确认后锁存许可，等待就绪不设两秒过期。
  STOP或最终圈数停车清除本次许可；许可锁存后ESP32断线、重启不影响本轮运行。直接启动可人工跳过等灯。
  ESP32网页只观察，不直接控制电机。

首次无线配置 / 更换Wi-Fi：
  ESP32的USB接电脑，执行：
  .maixpy/.venv/bin/python esp32s3/video_preview/wireless_preview.py \
    --serial /dev/cu.usbserial-14240 --ssid 'Xiaomi 14' \
    --maix-url http://10.73.124.209:8000/ --port 8123
  在 http://127.0.0.1:8123/setup 自行输入密码并连接。
  密码只保存于ESP32 NVS，不写入项目、聊天、日志或主机状态文件。
  电脑状态 .maixpy/esp32_wifi.local.json 只保存SSID与设备URL。
  原热点仍保留为 HW_ESP32S3CAM，地址192.168.5.1；日常使用连接Maix同一网络。

首次 ESP32 / Maix 配对：
  .maixpy/.venv/bin/python esp32s3/video_preview/pair_signal.py \
    --serial /dev/cu.usbserial-14240 --peer-url http://robot-esp32cam.local/ \
    --output .maixpy/start_signal.local.json
  通过安全传输将私有配对文件复制到Maix应用目录并设权限600。
  文件不提交Git、不打包到公共应用包；ESP32端密钥保存NVS。
  Maix应用更新保留该私有文件。更换配对后需同步更新两端。

编译（本机已验证Arduino ESP32 2.0.12-cn）：
  arduino-cli compile \
    --fqbn 'esp32:esp32:esp32s3:PSRAM=opi,FlashSize=8M,PartitionScheme=huge_app,FlashMode=qio,USBMode=hwcdc,CDCOnBoot=default,UploadSpeed=115200' \
    --jobs 4 --build-path .maixpy/permission-latch-20261107/build \
    --output-dir .maixpy/permission-latch-20261107/firmware esp32s3/video_preview
  注意：设备保留原OTA分区，app0为0x10000、大小0x330000。
  常规更新只写app0，不用编译生成的分区表覆盖原表、不整片擦除。
  不刷写UNO串口；当前ESP32是 /dev/cu.usbserial-14240，端口变化时重新确认。

早期发车固件回滚材料（本地忽略目录）：
  .maixpy/esp32-start-signal-20261106/flash-record.json
  记录旧0.2.0-wifi应用文件及SHA256、写入前OTA选择备份、最终固件SHA256。
  恢复旧应用时按记录写app0与OTA选择，不触碰NVS、bootloader和原分区表。
  更早制造商固件备份见 esp32s3/backups/；不要使用不完整的8MiB读取结果。

接口：
  80端口 /capture：缓存JPEG，直接页面短请求刷新预览。
  81端口 /status、POST /signal：独立发车控制服务；不与JPEG发送排队。
  /status：相机和发车状态，不含密码/配对密钥。
  81端口 POST /signal：经过配对认证的 arm / poll / cancel，会话及重启编号匹配才接受。
  USB J=JPEG、S=状态、Q=无线状态、C+JSON=Wi-Fi配置、P+JSON=专用配对。
  检测取帧独立于预览；无客户端时仍能检测。Maix主循环不等待无线请求。

回测：
  .maixpy/.venv/bin/python esp32s3/video_preview/replay.py \
    --output .maixpy/esp32-rule-start-20261107/replay.json
  编译同一C++检测内核；扩框后历史小灯原始构图熄灯样本通过，17张未切灯抖动无误触发。
  大灯原始构图出框仍不锁定；保留完整上下文的整体平移后，大小灯亮→灭均通过，反向均不触发。
  样本来自旧Maix镜头，不能代替最终上仰安装后的ESP32实测。
  分工、失效处理、数据范围和限制详见 docs/双摄发车信号评审-20261106.md。

自恢复：主循环／相机任务8秒看门狗；3秒无新鲜相机帧重启；断网每5秒重连。
/status diagnostics提供重启原因、堆内存、RSSI及注册状态。重启后必须重新ARM。

自恢复0.4.2历史回滚记录：.maixpy/esp32-recovery-20261107/flash-record.json，前版0.4.1镜像与SHA256已保留。

录制工具：record_signal.py，仅ARM ESP32检测器，持续检查Maix停车；JPEG序列和状态日志保存在.maixpy/lamp-recordings。

放大框版本的回滚记录：.maixpy/esp32-larger-roi-20261107/flash-record.json，前版0.4.2已保存。

灯面分割：0.4.4按亮度及颜色寻找连通域，冻结实际像素掩膜；颜色类别每帧只计算一次。
背景仅辅助曝光补偿及结构检查，不再因背景明显变暗直接否决。
0.4.4历史回放：405帧，初次锁定约0.818秒，熄灯候选到确认659ms。当前0.4.5同一录像确认449ms；JPEG回放不等于现场延迟。
复现录制数据：replay.py --recording .maixpy/lamp-recordings/lamp-trial-20261107-a --output .maixpy/esp32-lamp-mask-20261107/recording-replay.json
纯色背景下增益与偏移不可同时辨识，遮挡或移走灯也可能误判；现场灯仍需验证。
灯面分割0.4.4历史回滚记录：.maixpy/esp32-lamp-mask-20261107/flash-record.json，保留0.4.3前版镜像。
详见 docs/灯面分割与回测-20261107.md。

当前发车协议：Maix收到本次会话的有效熄灯确认后锁存permission=true，直到STOP或完成圈数最终停车。
ESP32后续离线、重启、画面中断均不撤销Maix许可；首次许可仍要求匹配会话及有效确认。
ESP32确认前2秒无通信取消；确认后保持到认证cancel，不再因租约过期清掉结果。
当前OFF条件：3帧且400ms，至少60%灯面像素变暗，Lab L8变化下限8；ON基准仍为4帧且300ms。
当前部署及回滚记录：.maixpy/permission-latch-20261107/flash-record.json，前版0.4.4已保存。
协议与验证详见 docs/出发许可锁存-20261107.md。

当前扩框部署、回测及回滚记录：.maixpy/esp32-roi-160x128-20261107/flash-record.json；详细说明 docs/ESP32定位框扩大-20261107.md。当前维护USB端口为/dev/cu.usbserial-14140（端口变化时重新确认）。
