#include <Arduino.h>

// miniAuto 的四路电机引脚，沿用原电机测试程序。
const uint8_t PWM_PINS[4] = {10, 9, 6, 11};
const uint8_t DIR_PINS[4] = {12, 8, 7, 13};
const uint8_t FORWARD_LEVEL[4] = {HIGH, LOW, LOW, HIGH};

const uint8_t SPEED_PERCENT = 50;       // 速度：0～100
const unsigned long START_DELAY_MS = 3000;
const unsigned long RUN_TIME_MS = 2000;

void stopMotors() {
  for (uint8_t i = 0; i < 4; ++i) {
    analogWrite(PWM_PINS[i], 0);
  }
}

void forward(uint8_t percent) {
  if (percent > 100) percent = 100;
  const uint8_t pwm = (uint8_t)map(percent, 0, 100, 0, 255);
  for (uint8_t i = 0; i < 4; ++i) {
    digitalWrite(DIR_PINS[i], FORWARD_LEVEL[i]);
    analogWrite(PWM_PINS[i], pwm);
  }
}

void setup() {
  for (uint8_t i = 0; i < 4; ++i) {
    pinMode(PWM_PINS[i], OUTPUT);
    analogWrite(PWM_PINS[i], 0);
    pinMode(DIR_PINS[i], OUTPUT);
    digitalWrite(DIR_PINS[i], FORWARD_LEVEL[i]);
  }

  // 每次上电或复位：等待 3 秒，前进 2 秒，然后停车。
  delay(START_DELAY_MS);
  forward(SPEED_PERCENT);
  delay(RUN_TIME_MS);
  stopMotors();
}

void loop() {
  // 保持停止；按 RESET 或重新上电可再执行一次。
}
