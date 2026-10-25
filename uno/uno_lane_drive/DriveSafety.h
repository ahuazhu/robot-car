#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace drive {
const int MAX_PWM = 200;
static_assert(MAX_PWM>0 && MAX_PWM<=255,"PWM limit must fit analogWrite duty range");
const uint32_t WATCHDOG_MS = 350;
inline uint8_t crc8(const char* s) {
  uint8_t crc = 0;
  while (*s) {
    crc ^= uint8_t(*s++);
    for (uint8_t i=0; i<8; ++i) crc = crc & 0x80 ? uint8_t((crc<<1)^7) : uint8_t(crc<<1);
  }
  return crc;
}
inline int hex(char c) {
  if (c>='0' && c<='9') return c-'0';
  if (c>='A' && c<='F') return c-'A'+10;
  return -1;
}
struct State {
  bool armed=false, emergency=false;
  int left=0, right=0;
  uint8_t fault=0;
  uint16_t sequence=0;
  uint32_t lastDrive=0;
  bool brakeRequired() const { return !armed || (left==0 && right==0); }
  void stop(uint8_t why=0) { armed=false; left=right=0; fault=why; }
  void tick(uint32_t now) {
    // Continuous driving has no session limit; fresh commands are still required.
    if (armed && uint32_t(now-lastDrive)>=WATCHDOG_MS) stop(2);
  }
  void estop() { emergency=true; stop(4); }
  // @sequence,operation,leftPWM,rightPWM*CRC8
  // 0=stop/disarm, 1=arm at zero, 2=drive, 3=status (does not feed watchdog).
  bool accept(char* line, uint32_t now, uint16_t& replySequence) {
    tick(now);
    char* star=strchr(line,'*');
    if (line[0]!='@' || !star || strlen(star)!=3 || hex(star[1])<0 || hex(star[2])<0) { stop(1); return false; }
    int checksum=hex(star[1])*16+hex(star[2]);
    *star='\0';
    if (crc8(line+1)!=checksum) { stop(1); return false; }
    long values[4]; char* p=line+1;
    for (uint8_t i=0; i<4; ++i) {
      char* begin=p;
      if (*p=='-' && i>=2) ++p;
      if (*p<'0'||*p>'9') { stop(1); return false; }
      uint8_t digits=0;
      while (*p>='0' && *p<='9') { ++p; if (++digits>5) { stop(1); return false; } }
      if (*p!=(i==3?'\0':',')) { stop(1); return false; }
      values[i]=strtol(begin,0,10);
      if (i<3) ++p;
    }
    if (values[0]<0 || values[0]>65535 || values[1]<0 || values[1]>3 ||
        values[2]<-MAX_PWM || values[2]>MAX_PWM || values[3]<-MAX_PWM || values[3]>MAX_PWM) { stop(1); return false; }
    replySequence=uint16_t(values[0]);
    uint8_t op=uint8_t(values[1]);
    if (op!=2 && (values[2] || values[3])) { stop(1); return false; }
    if (op==0) { stop(emergency?4:0); return true; }
    if (op==3) return true;
    if (emergency) { stop(4); return true; }
    if (op==1) { left=right=0; armed=true; fault=0; sequence=replySequence; lastDrive=now; return true; }
    uint16_t delta=uint16_t(replySequence-sequence);
    if (!armed || delta==0 || delta>=32768) { stop(3); return true; }
    sequence=replySequence; left=int(values[2]); right=int(values[3]); lastDrive=now; fault=0;
    return true;
  }
};
// Bounded USB-only lamp diagnostic. The sketch locks motors while active.
struct LampBlink {
  bool active=false;
  uint32_t began=0;
  bool start(bool motorArmed,uint32_t now) {
    if (motorArmed) return false;
    began=now;active=true;return true;
  }
  bool update(uint32_t now) {
    if (active && uint32_t(now-began)>=6000) { active=false;return true; }
    return false;
  }
  bool level(uint32_t now,bool normalBrake) const {
    return active ? (uint32_t(now-began)/1000)%2==0 : normalBrake;
  }
};

}
