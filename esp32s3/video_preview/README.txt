ESP32-S3近场摄像头视频预览（2026-10-22）

目标硬件：Hiwonder ESP32-S3 GC2145，8MB flash、OPI PSRAM。
用途：安装高度、左右覆盖、俯角与弯道视野测试。不会控制电机。
本固件取代颜色识别固件，不提供原0x52颜色数据或侧摄边线数据。

电脑USB预览（本次已启动并验证）：
  http://127.0.0.1:8123/
  保持ESP32-S3自己的USB线接电脑；不需要切换Wi-Fi。
  这是低帧率调视角预览（115200串口传JPEG），不是巡线控制的数据通道。
  重启服务：在 /Users/yang/work/robot-car 执行
  .maixpy/.venv/bin/python esp32s3/video_preview/usb_preview.py --serial /dev/cu.usbserial-14210 --port 8123
  端口名若改变，以当前ESP32-S3实际串口为准；不要选择UNO端口。
  Ctrl+C停止本机服务；不要同时用Arduino串口监视器占用ESP32串口。
  已验证：相机PID 0x2145、连续3张可解码且哈希不同的320x240 JPEG、浏览器显示实际画面。
  Wi-Fi图传服务已启动，热点端实际播放尚待用户连接验证。

备份与恢复：
  ../backups/before_preview_first_1MiB_20261022.bin 保存原闪存前1MiB。
  它不是8MiB全盘备份，但包含本次写入涉及的引导/分区/当前应用区域；原应用校验与SHA均有效。
  初始8MiB高速读取因串口数据损坏失败，不可使用该次不完整结果恢复。
  恢复时仅向ESP32-S3地址0写回上述1MiB镜像，不执行整片擦除。
  其余闪存区域本次未写入。

手机Wi-Fi预览：
1. 摄像头上电，等待约5秒。
2. 手机或电脑连接 Wi-Fi HW_ESP32S3CAM（开放热点，无密码）。
3. 浏览器打开 http://192.168.5.1/ 。没有互联网提示时选择保持连接。
4. 网页自动预览，可停止、重新开始、开关中心参考线。
   参考线仅用于观察，不是边线识别结果。
5. 原始MJPEG：http://192.168.5.1:81/stream
   单张JPEG：http://192.168.5.1/capture
手机预览不影响电脑继续连接原Wi-Fi。不要把Maix的192.168.3.244当成这块摄像头。

画面：QVGA 320x240，RGB565采集后软件JPEG编码。实际帧率以实测为准。
仅调试预览；比赛运行时应关闭无线图传。
USB串口115200：S返回状态；J返回“JPEG 字节数\n”及其后的JPEG二进制。

编译环境：esp32 by Espressif Systems 2.0.12-cn。
FQBN：esp32:esp32:esp32s3:PSRAM=opi,FlashSize=8M,PartitionScheme=huge_app,FlashMode=qio,USBMode=hwcdc,CDCOnBoot=default,UploadSpeed=115200

相机引脚和15MHz时钟依据厂家源码：
https://github.com/Hiwonder/MiniHexa/tree/main/arduino-programming-projects
4.4 AI视觉项目课程/7.2 颜色识别/esp32s3/ColorDetection/camera_setting.h
HTTP和JPEG接口依据本机Espressif 2.0.12-cn CameraWebServer示例。
厂家原厂图传使用方法：
https://docs.hiwonder.com/projects/miniAuto/en/latest/docs/7.ai_vision_games_lesson.html#live-camera-feed
此工程是针对该硬件编写的独立预览程序，不是厂家的image_transmit.bin。
