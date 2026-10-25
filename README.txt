持续巡线更新（2026-10-23）：
取消当前曲线巡线 Maix 55秒和 UNO 60秒固定运行上限；本次需同步更新UNO固件。
手动START后持续运行，STOP、丢线、视觉过期及350ms通信看门狗仍有效。
下方早期版本的12秒限时等说明为历史记录。

最新状态（2026-10-23）：
曲线巡线代码：maixcam2/curve_follow/；摄像头 192.168.3.3。
已修复视觉阻塞串口造成的停车，独立电机进程实测指令间隔<91ms。
已验证90PWM可使四轮前进，增加+37起转补偿；已增强急弯差速、减速和移动中视觉重捕获。
27项曲线/控制测试通过。完整过弯/整圈尚未验证，当前停车等待本轮实车反馈。
操作及证据见 maixcam2/curve_follow/README.txt。网页 http://192.168.3.3:8000/。

当前开发版本：曲线中线巡线（2026-10-23）
MaixCAM2继续巡线，ESP32-S3后续负责路牌。独立代码：maixcam2/curve_follow/。
采用参数曲线、单双边法线中线、自适应预瞄；见该目录README.txt。
摄像头新IP：192.168.3.3；预览：http://192.168.3.3:8000/。
当前仅预览：旧视角标定在新版本中标为无效，START锁定，需固定镜头重新标定。
旧代码未覆盖，UNO无需重新烧录；ESP32路牌识别尚未实现。

ESP32-S3预览更新（2026-10-22）：
  已烧录 esp32s3/video_preview/video_preview.ino；USB预览 http://127.0.0.1:8123/ 正在运行。
  本轮仅预览：原ESP32颜色识别/I2C 0x52服务已被替换，不再由此固件提供。
  UNO正式A0/A1固件保持不变，未启动小车或Maix巡线应用。
  说明与恢复备份：esp32s3/video_preview/README.txt。

接线迁移（2026-10-22，A0/A1通信实测通过）：
  MaixCAM2 TX -> LV1/HV1（蓝线）-> UNO A0 (RX)
  MaixCAM2 RX <- LV2/HV2（紫线）<- UNO A1 (TX)
  原 ESP32-S3 使用原厂四线插头 -> 白色 SCL/SDA/GND/5V 插槽。
  A4=SDA、A5=SCL 留给 I2C；不要把 Maix 串口信号并接到白色口。
  UNO 已恢复正式 uno_lane_drive 固件；USB串口115200，Maix UART9600。
  Maix重启后的纯PING/PONG测试30/30通过；恢复正式固件后的STOP测试50/50通过。
  全程未发ARM/DRIVE；当前DISARMED，left=right=0，fault=0；摄像头巡线应用未启动。
  之前曾出现50次仅19次成功，重启前后的对照不能单独确定根因；运动负载下的稳定性仍待验证。
  原ESP32 I2C 0x52在最终复核中ACK，仅证明设备在场；侧摄边线固件及数据转发尚未实现。
  验证日志：.maixpy/runs/20261022-062527-a0_a1_ping_check/（30/30）
            .maixpy/runs/20261022-062649-a0_a1_stability/（50/50）
            .maixpy/a0_a1_formal_after_restart_usb.log、.maixpy/i2c_after_uart_restore.log。
  target_follow v0.3 已完成本地测试，尚未实机部署。下方早期进度为历史记录。

当前巡线程序（2026-10-21）
  uno/uno_lane_drive/uno_lane_drive.ino：已烧录；电机方向实测正确。
  maixcam2/lane_follow/main.py：手动 START，低速巡线；每轮最多 12 秒。
  摄像头应用：小车巡线 / Lane Follow。
  详细步骤：docs/巡线测试说明.txt
  当前进度：架空电机测试通过；v0.1.10 已修复3/4条边线切换门槛冲突及可见道路3秒过期误判，已增加网页START/STOP及浏览器实时预览（http://192.168.3.244:8000），并降低转弯速度、加入转向渐变与近边线优先；22项离线检查通过，完整过弯实跑尚未通过。

以下为保留的分阶段诊断程序，勿用串口测试固件覆盖当前巡线固件：

robot-car：miniAuto + MaixCAM2

当前两端程序分开保存：
  uno/uno_uart_test/uno_uart_test.ino
    Arduino IDE 打开；开发板 Arduino UNO；上传到 UNO。
    始终保持电机停止。A0=RX，A1=TX，通信 9600。
    电脑串口监视器使用 115200。

  maixcam2/uart_test/main.py
    MaixVision 打开 uart_test 文件夹或 main.py，连接 MaixCAM2 后运行。
    使用侧边 UART4，A21=TX，A22=RX，/dev/ttyS4，通信 9600。
    每秒发送 PING，验证 UNO 返回同编号 PONG。

  maixcam2/camera_preview/main.py
    独立画面预览程序；先停止串口测试，再运行此程序。

完整步骤与排障：docs/串口测试说明.txt

目录约定：
  uno/ 只存放本次新写、当前需要的 UNO 程序。
  原有 wheel_check、motor_standalone、color_follow、colour_led 已移回 /Users/yang/work/uno/。

Arduino 每个草图的文件夹名与其中主 .ino 文件名保持一致。
MaixCAM2 的 Python 程序通过 MaixVision 在摄像头上运行，不在电脑 Python 中运行。
UNO 串口测试已编译通过，实机 PING/PONG 双向通信已验证（13 次连续成功，0 超时）。

  maixcam2/lane_preview/main.py
    灰白道路、绿色场外的边界及中心预览；不发送运动指令。
    测试步骤：docs/赛道预览说明.txt
    实机识别效果仍需根据车载镜头画面验证。

AI 直接连接 MaixCAM2 的环境已配置；参阅 docs/MaixPy技能配置说明.txt。
运行日志、设备配置及私有凭据存放在 .maixpy/（不入库）。
