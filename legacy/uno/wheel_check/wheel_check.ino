#include <Arduino.h>

/*
 * miniAuto 逐轮点动：只上传到 Arduino UNO。
 * 先架空四个车轮，用电池供电，串口监视器设为 115200。
 * 上电不转；每发送一个数字 1/2/3/4，仅对应的一路转动 1 秒。
 * 发送 s 立即停止。运动期间的其他数字被忽略，需停车后再发。
 *
 * 这里 TEST 1～4 是程序测试编号，不等同于扩展板 M1～M4 标号。
 * 每路沿用 color_follow 当前认为“前进”的电平，用来核对轮子位置和极性。
 * 以摄像头朝向为车头，记录：左前/右前/左后/右后，以及大轮最上缘
 * 朝车头还是车尾运动（看大轮绕电机轴转动，不是麦轮小滚子的自转）。
 */

const uint8_t PWM_PINS[4] = {10, 9, 6, 11};
const uint8_t DIR_PINS[4] = {12, 8, 7, 13};
const uint8_t TEST_LEVELS[4] = {HIGH, LOW, LOW, HIGH};
const uint8_t TEST_PWM = 100;
const uint32_t PULSE_MS = 1000;
int8_t activeWheel = -1;
uint32_t pulseStarted = 0;

void stopAll() {
  for (uint8_t i = 0; i < 4; ++i) analogWrite(PWM_PINS[i], 0);
  activeWheel = -1;
}

void setup() {
  for (uint8_t i = 0; i < 4; ++i) {
    pinMode(PWM_PINS[i], OUTPUT);
    analogWrite(PWM_PINS[i], 0);
    pinMode(DIR_PINS[i], OUTPUT);
    digitalWrite(DIR_PINS[i], TEST_LEVELS[i]);
  }
  Serial.begin(115200);
  Serial.println(F("WHEELS OFF GROUND. Send 1, 2, 3 or 4 individually; s = stop."));
  Serial.println(F("Test numbers are NOT M1-M4 socket labels. Each pulse lasts 1 second."));
}

void loop() {
  if (activeWheel >= 0 && uint32_t(millis() - pulseStarted) >= PULSE_MS) {
    stopAll();
    Serial.println(F("STOP. Ready for next test."));
  }
  while (Serial.available()) {
    const char command = Serial.read();
    if (command == 's' || command == 'S') {
      stopAll();
      Serial.println(F("STOP."));
    } else if (command >= '1' && command <= '4') {
      if (activeWheel >= 0) {
        Serial.println(F("BUSY: wait for STOP, then send the next number."));
        continue;
      }
      const uint8_t i = command - '1';
      stopAll();
      digitalWrite(DIR_PINS[i], TEST_LEVELS[i]);
      Serial.print(F("TEST ")); Serial.print(i + 1);
      Serial.print(F(" PWM pin=")); Serial.print(PWM_PINS[i]);
      Serial.print(F(" DIR pin=")); Serial.print(DIR_PINS[i]);
      Serial.print(F(" level=")); Serial.println(TEST_LEVELS[i] ? F("HIGH") : F("LOW"));
      pulseStarted = millis();
      activeWheel = i;
      analogWrite(PWM_PINS[i], TEST_PWM);
    }
  }
}
