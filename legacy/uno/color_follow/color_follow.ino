#include <Arduino.h>
#include <Wire.h>
#include "FollowController.h"
#include "TurnPacer.h"

/*
 * miniAuto UNO 自主颜色跟随（无需蓝牙、手机或持续连接电脑）
 * 必须配合 camera_color/ColorDetection 摄像头程序及原厂 I2C 连线。
 * 协议依据：Hiwonder miniAuto 7.5 官方 ColorDetection / color_tracking 源码。
 * https://docs.hiwonder.com/projects/miniAuto/en/latest/docs/7.ai_vision_games_lesson.html#color-tracking
 *
 * 每次上电/RESET：把同一红色物体放在车头正前方希望保持的距离，
 * 例如 30 cm。取消固定开机等待和倒计时；居中目标确认约 100 ms 后开始跟随。
 * 至少 3 次观测且稳定 100 ms；摄像头未就绪时保持停车。
 * 距离以物体画面宽度为参考，并非厘米测距；物体靠近时停车，不倒车。
 * 换物体或希望重新设置距离时，按 RESET，重新标定。
 *
 * 首次架空车轮检查左右转向；串口监视器 115200 可看状态。
 * 串口发送 s 可锁定停车，发送 c 可停车并重新标定。
 * 发送 f/l/r 可分别测试前进/左转/右转 1 秒，结束后保持暂停。
 * 转向按 TurnPacer.h 的 100 ms 转、120 ms 停执行，保持 TURN_PWM=100。
 * 不要使用 LED_BUILTIN：UNO 的 13 号脚也是该车的电机方向脚。
 */

const uint8_t TARGET_COLOR = 0x00;  // 0x00 红色；改成 0x02 追踪蓝色
const uint8_t CAMERA_ADDRESS = 0x52;
const uint8_t FORWARD_PWM = 75;     // 0～255；车轮负载下不转可小幅增加
const uint8_t TURN_PWM = 100;      // 与已验证的单轮点动 PWM 一致，改善低占空比启动
const uint16_t CAMERA_POLL_MS = 50;
const int8_t TURN_SIGN = 1;         // 若实测左右转向相反，改为 -1
// 2026-10-15 逐轮实测：测试1右前、2左前、3左后、4右后。
// 四路在下方 FORWARD_LEVEL 电平下均为前进。
enum Wheel : uint8_t { FRONT_RIGHT = 0, FRONT_LEFT = 1, REAR_LEFT = 2, REAR_RIGHT = 3 };
const uint8_t PWM_PINS[4] = {10, 9, 6, 11};
const uint8_t DIR_PINS[4] = {12, 8, 7, 13};
const uint8_t FORWARD_LEVEL[4] = {HIGH, LOW, LOW, HIGH};

follow::Controller controller;
follow::TurnPacer turnPacer;
follow::Motion requestedMotion = follow::STOP;
bool paused = false;
bool manualPulse = false;
uint32_t manualStarted = 0;
uint32_t lastPoll = 0, lastLog = 0;
follow::Motion lastLoggedMotion = follow::STOP;

void stopMotors() {
  for (uint8_t i = 0; i < 4; ++i) analogWrite(PWM_PINS[i], 0);
}

void setWheel(uint8_t i, int16_t pwm) {
  const uint8_t direction = pwm < 0 ? !FORWARD_LEVEL[i] : FORWARD_LEVEL[i];
  // 改变转向前先撤去 PWM。
  if (digitalRead(DIR_PINS[i]) != direction) analogWrite(PWM_PINS[i], 0);
  digitalWrite(DIR_PINS[i], direction);
  analogWrite(PWM_PINS[i], uint8_t(pwm < 0 ? -pwm : pwm));
}

void moveMotors(follow::Motion motion) {
  if (motion == follow::STOP) {
    stopMotors();
    return;
  }
  if (motion == follow::FORWARD) {
    for (uint8_t i = 0; i < 4; ++i) setWheel(i, FORWARD_PWM);
    return;
  }
  // 实测 1/2 是左轮，0/3 是右轮（这里是从 0 开始的数组下标）。
  // 左转时左轮后退、右轮前进。
  const int16_t left = (motion == follow::LEFT ? -int16_t(TURN_PWM)
                                                             : int16_t(TURN_PWM)) * TURN_SIGN;
  setWheel(FRONT_LEFT, left);
  setWheel(REAR_LEFT, left);
  setWheel(FRONT_RIGHT, -left);
  setWheel(REAR_RIGHT, -left);
}

bool readTarget(follow::Target &target) {
  target = {0, 0, 0, 0};
  Wire.clearWireTimeoutFlag();
  Wire.beginTransmission(CAMERA_ADDRESS);
  Wire.write(TARGET_COLOR);
  if (Wire.endTransmission() != 0 || Wire.getWireTimeoutFlag()) return false;
  const uint8_t count = Wire.requestFrom(CAMERA_ADDRESS, uint8_t(4));
  if (count != 4 || Wire.available() != 4 || Wire.getWireTimeoutFlag()) {
    while (Wire.available()) Wire.read();
    return false;
  }
  target.x = Wire.read();
  target.y = Wire.read();
  target.width = Wire.read();
  target.height = Wire.read();
  return true;
}

void setup() {
  for (uint8_t i = 0; i < 4; ++i) {
    pinMode(PWM_PINS[i], OUTPUT);
    analogWrite(PWM_PINS[i], 0);
    pinMode(DIR_PINS[i], OUTPUT);
    digitalWrite(DIR_PINS[i], FORWARD_LEVEL[i]);
  }
  Serial.begin(115200);
  Wire.begin();                     // UNO: SDA=A4，SCL=A5
  Wire.setClock(100000);
  Wire.setWireTimeout(25000, true);  // 总线卡住时返回，随后停车
  controller.begin(millis());
  Serial.println(F("Put target at desired distance, centered and still. s=stop c=calibrate"));
  Serial.println(F("f=forward l=left r=right: 1-second test, then PAUSED. c=resume calibration"));
}

void loop() {
  if (manualPulse && uint32_t(millis() - manualStarted) >= 1000) {
    stopMotors();
    requestedMotion = follow::STOP;
    turnPacer.reset();
    manualPulse = false;
    Serial.println(F("TEST STOPPED. PAUSED; c=calibrate."));
  }
  while (Serial.available()) {
    const char command = Serial.read();
    if (command == 's' || command == 'c') {
      stopMotors();
      requestedMotion = follow::STOP;
      turnPacer.reset();
      manualPulse = false;
      paused = command == 's';
      controller.begin(millis());
      Serial.println(paused ? F("PAUSED") : F("CALIBRATION RESTART"));
    } else if (command == 'f' || command == 'l' || command == 'r') {
      // 不排队执行连续输入；每个测试必须等上一个停止后再发送。
      if (manualPulse) continue;
      stopMotors();
      turnPacer.reset();
      paused = true;
      const follow::Motion test = command == 'f' ? follow::FORWARD :
                                 command == 'l' ? follow::LEFT : follow::RIGHT;
      manualStarted = millis();
      manualPulse = true;
      requestedMotion = test;
      Serial.println(command == 'f' ? F("TEST FORWARD") :
                     command == 'l' ? F("TEST LEFT") : F("TEST RIGHT"));
    }
  }
  if (paused) {
    moveMotors(turnPacer.update(millis(), requestedMotion));
    return;
  }
  const uint32_t now = millis();
  // 即使未到读取时间，也继续执行转/停时序，避免被 50 ms 采样节拍限制。
  moveMotors(turnPacer.update(now, requestedMotion));
  if (uint32_t(now - lastPoll) < CAMERA_POLL_MS) return;
  lastPoll = now;
  follow::Target target;
  const bool received = readTarget(target);
  const follow::Motion motion = controller.update(millis(), received, target);
  requestedMotion = motion;
  moveMotors(turnPacer.update(millis(), requestedMotion));

  if (motion != lastLoggedMotion || uint32_t(now - lastLog) >= 500) {
    lastLog = now;
    lastLoggedMotion = motion;
    Serial.print(F("phase="));
    switch (controller.phase()) {
      case follow::CALIBRATING: Serial.print(F("CALIBRATING")); break;
      case follow::TRACKING: Serial.print(F("TRACKING")); break;
    }
    Serial.print(F(" camera="));
    Serial.print(received ? F("OK") : F("I2C_ERROR"));
    Serial.print(F(" x=")); Serial.print(target.x);
    Serial.print(F(" width=")); Serial.print(target.width);
    Serial.print(F(" reference=")); Serial.print(controller.referenceWidth());
    Serial.print(F(" motion=")); Serial.print(uint8_t(motion));
    Serial.print(F(" action="));
    switch (motion) {
      case follow::STOP: Serial.println(F("STOP")); break;
      case follow::FORWARD: Serial.println(F("FORWARD")); break;
      case follow::LEFT: Serial.println(F("ALIGN_LEFT(x<100)")); break;
      case follow::RIGHT: Serial.println(F("ALIGN_RIGHT(x>140)")); break;
    }
  }
}
