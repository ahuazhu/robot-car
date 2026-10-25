#include <Arduino.h>
#include <SoftwareSerial.h>
#include <string.h>

// UNO only. Camera side connector uses UART4, NOT front UART2.
// Camera TX -> LV1/HV1 -> A0; camera RX <- LV2/HV2 <- A1.
// A4/SDA and A5/SCL are reserved for the original ESP32 camera; this test does not poll it.
SoftwareSerial cameraSerial(A0, A1); // RX, TX
const uint32_t LINK_BAUD = 9600;
const uint8_t PWM_PINS[] = {10, 9, 6, 11};
const uint8_t DIR_PINS[] = {12, 8, 7, 13};
char line[40];
uint8_t used = 0;
bool dropping = false;
uint32_t received = 0;
uint32_t lastStatus = 0;

void setup() {
  // Hold all four motors stopped. Do not use LED_BUILTIN (D13 is motor DIR).
  for (uint8_t i = 0; i < 4; ++i) {
    digitalWrite(PWM_PINS[i], LOW);
    pinMode(PWM_PINS[i], OUTPUT);
    digitalWrite(DIR_PINS[i], LOW);
    pinMode(DIR_PINS[i], OUTPUT);
  }
  Serial.begin(115200); // Computer's Serial Monitor
  cameraSerial.begin(LINK_BAUD);
  Serial.println(F("UNO READY | camera RX=A0 TX=A1 baud=9600 | motors OFF"));
}

void processLine() {
  line[used] = '\0';
  Serial.print(F("RX: "));
  Serial.println(line);
  if (strncmp(line, "PING ", 5) != 0 || used == 5) return;
  for (uint8_t i = 5; i < used; ++i) {
    if (line[i] < '0' || line[i] > '9') return;
  }
  ++received;
  cameraSerial.print(F("PONG "));
  cameraSerial.println(line + 5);
  Serial.print(F("TX: PONG "));
  Serial.println(line + 5);
}

void loop() {
  while (cameraSerial.available()) {
    char c = cameraSerial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (!dropping && used) processLine();
      used = 0;
      dropping = false;
    } else if (!dropping) {
      if (c >= 32 && c <= 126 && used < sizeof(line) - 1) line[used++] = c;
      else {
        dropping = true;
        Serial.println(F("BAD FRAME: check wiring/baud/power"));
      }
    }
  }
  if (cameraSerial.overflow()) Serial.println(F("RX OVERFLOW"));
  if (uint32_t(millis() - lastStatus) >= 2000) {
    lastStatus = millis();
    Serial.print(F("UNO ALIVE | valid PING="));
    Serial.println(received);
  }
}
