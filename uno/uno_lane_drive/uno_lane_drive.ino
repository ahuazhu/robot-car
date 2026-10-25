#include <Arduino.h>
#include <SoftwareSerial.h>
#include <Wire.h>
#include "DriveSafety.h"
// Set only after checking the actual LED wiring. -1 reports unavailable.
#ifndef BRAKE_LED_PIN
#define BRAKE_LED_PIN 16 // UNO A2; user confirmed signal wiring
#endif
#ifndef BRAKE_LED_ACTIVE_HIGH
#define BRAKE_LED_ACTIVE_HIGH 1
#endif
static_assert(BRAKE_LED_PIN==-1 || (BRAKE_LED_PIN>=2 && BRAKE_LED_PIN<=5) ||
              BRAKE_LED_PIN==A2 || BRAKE_LED_PIN==A3,
              "Brake LED pin conflicts with motor/UART/I2C pins");
bool brakeLampOn=false;
drive::LampBlink lampTest;

// Proven wheel order: RF=M2, LF=M3, LR=M4, RR=M1.
const uint8_t PWM_PINS[4]={10,9,6,11};
const uint8_t DIR_PINS[4]={12,8,7,13};
const uint8_t FORWARD_LEVEL[4]={HIGH,LOW,LOW,HIGH};
// MaixCAM2 UART via level shifter. A4/A5 reserved for original ESP32 I2C.
SoftwareSerial cameraSerial(A0,A1); // RX, TX
drive::State state;
char line[48]; uint8_t used=0; bool dropping=false;
uint32_t lastLog=0, rxBytes=0, replies=0;

void outputs() {
  if (lampTest.update(millis())) Serial.println(F("BRAKE TEST DONE | motors OFF | normal brake ON"));
  if (lampTest.active) state.stop(state.emergency?4:0);
  brakeLampOn=BRAKE_LED_PIN>=0 && lampTest.level(millis(),state.brakeRequired());
#if BRAKE_LED_PIN >= 0
  digitalWrite(BRAKE_LED_PIN,brakeLampOn==bool(BRAKE_LED_ACTIVE_HIGH)?HIGH:LOW);
#endif
  int values[4]={state.right,state.left,state.left,state.right};
  for (uint8_t i=0;i<4;++i) {
    int value=values[i];
    uint8_t level=value<0 ? !FORWARD_LEVEL[i] : FORWARD_LEVEL[i];
    if (digitalRead(DIR_PINS[i])!=level) analogWrite(PWM_PINS[i],0);
    digitalWrite(DIR_PINS[i],level);
    analogWrite(PWM_PINS[i],uint8_t(value<0?-value:value));
  }
}
void reply(uint16_t sequence) {
  char payload[40], out[48];
  snprintf(payload,sizeof(payload),"%u,%u,%d,%d,%u,%u",sequence,state.armed?1:0,state.left,state.right,state.fault,
           4u | (BRAKE_LED_PIN>=0?1u:0u) | (brakeLampOn?2u:0u));
  snprintf(out,sizeof(out),"!%s*%02X\n",payload,drive::crc8(payload));
  cameraSerial.print(out); ++replies;
}
void setup() {
#if BRAKE_LED_PIN >= 0
  digitalWrite(BRAKE_LED_PIN,BRAKE_LED_ACTIVE_HIGH?HIGH:LOW);
  pinMode(BRAKE_LED_PIN,OUTPUT);
#endif
  for (uint8_t i=0;i<4;++i) {
    digitalWrite(PWM_PINS[i],LOW); pinMode(PWM_PINS[i],OUTPUT);
    digitalWrite(DIR_PINS[i],FORWARD_LEVEL[i]); pinMode(DIR_PINS[i],OUTPUT);
  }
  Serial.begin(115200); cameraSerial.begin(9600);
  Wire.begin(); // A4=SDA, A5=SCL
  Wire.setClock(100000);
  Wire.setWireTimeout(25000, true);
  Serial.println(F("LANE DRIVE READY | DISARMED | A0=RX A1=TX | maxPWM=120 | watchdog=350ms"));
  Serial.println(F("USB s=estop; u=clear; b=brake lamp test; i=probe ESP32 I2C (disarmed only)"));
}
void loop() {
  while (Serial.available()) {
    char c=Serial.read();
    if (c=='s'||c=='S') state.estop();
    if (c=='b'||c=='B') {
      if (BRAKE_LED_PIN<0 || !lampTest.start(state.armed,millis())) Serial.println(F("BRAKE TEST REFUSED"));
      else { state.stop(state.emergency?4:0);outputs();Serial.println(F("BRAKE TEST START | 3 flashes / 6 seconds | motors locked OFF")); }
    }
    if (c=='i'||c=='I') {
      if (state.armed) Serial.println(F("I2C probe refused: stop first"));
      else {
        outputs();
        Wire.clearWireTimeoutFlag();
        Wire.beginTransmission(0x52); // Address-only probe, no camera settings changed.
        uint8_t error=Wire.endTransmission();
        bool timeout=Wire.getWireTimeoutFlag();
        Serial.print(F("ESP32 I2C 0x52: "));
        if (!error && !timeout) Serial.println(F("ACK (presence only; not lane data)"));
        else { Serial.print(F("NO ACK error=")); Serial.print(error); Serial.print(F(" timeout=")); Serial.println(timeout); }
      }
    }
    if ((c=='u'||c=='U') && !state.armed) { state.emergency=false; state.stop(); }
  }
  state.tick(millis()); outputs();
  while (cameraSerial.available()) {
    char c=cameraSerial.read(); ++rxBytes;
    if (c=='\r') continue;
    if (c=='\n') {
      if (!dropping && used) {
        line[used]='\0'; uint16_t seq=0;
        bool ok=state.accept(line,millis(),seq); outputs();
        if (ok) reply(seq);
      }
      used=0; dropping=false;
    } else if (!dropping) {
      if (c>=32 && c<=126 && used<sizeof(line)-1) line[used++]=c;
      else { dropping=true; state.stop(1); outputs(); }
    }
    state.tick(millis()); outputs();
  }
  if (cameraSerial.overflow()) { state.stop(1); outputs(); }
  if (uint32_t(millis()-lastLog)>=1000) {
    lastLog=millis();
    Serial.print(F("DRIVE armed=")); Serial.print(state.armed);
    Serial.print(F(" left=")); Serial.print(state.left);
    Serial.print(F(" right=")); Serial.print(state.right);
    Serial.print(F(" fault=")); Serial.print(state.fault);
    Serial.print(F(" brake=")); Serial.print(brakeLampOn);
    Serial.print(F(" lampTest=")); Serial.print(lampTest.active);
    Serial.print(F(" rxBytes=")); Serial.print(rxBytes);
    Serial.print(F(" replies=")); Serial.println(replies);
  }
}
