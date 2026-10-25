#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include "DriveSafety.h"
bool send(drive::State& s,unsigned seq,unsigned op,int l,int r,uint32_t now) {
  char payload[48],frame[56];
  std::snprintf(payload,sizeof(payload),"%u,%u,%d,%d",seq,op,l,r);
  std::snprintf(frame,sizeof(frame),"@%s*%02X",payload,drive::crc8(payload));
  uint16_t ack=0; return s.accept(frame,now,ack);
}
int main() {
  drive::State s;
  assert(!s.armed && s.left==0); assert(s.brakeRequired());
  send(s,1,2,80,80,0); assert(!s.armed && s.left==0);
  send(s,2,1,0,0,100); send(s,3,2,80,90,110);
  assert(s.armed && s.left==80 && s.right==90); assert(!s.brakeRequired());
  send(s,3,2,80,80,120); assert(!s.armed && s.left==0); // replay
  send(s,4,1,0,0,130); send(s,5,2,80,80,140);
  send(s,6,3,0,0,400); s.tick(490); assert(!s.armed); // status cannot feed lease
  send(s,7,1,0,0,500); send(s,8,2,drive::MAX_PWM+1,0,510); assert(!s.armed); // over limit
  send(s,9,1,0,0,520); char corrupt[]="@10,2,90,90*00"; uint16_t ack;
  s.accept(corrupt,530,ack); assert(!s.armed && s.left==0);
  send(s,65535,1,0,0,540); send(s,0,2,70,70,550); assert(s.armed); // sequence wrap
  s.estop(); send(s,1,1,0,0,560); assert(!s.armed && s.fault==4);
  s.emergency=false; send(s,1,1,0,0,0xfffffff0UL);
  send(s,2,2,70,70,0xfffffff5UL); s.tick(0x200UL); assert(!s.armed); // millis wrap
  send(s,1,1,0,0,1000); send(s,2,2,-100,100,1100);
  assert(s.left==-100 && s.right==100); send(s,0,0,0,0,1110); assert(s.left==0 && !s.armed);
  // Continuous fresh commands must survive the old 60s cap, sequence wrap,
  // and a millis() rollover. A missing command must still stop the car.
  for (uint32_t start : {uint32_t(1000), uint32_t(0xfffff000UL)}) {
    drive::State continuous;
    send(continuous,0,1,0,0,start);
    uint16_t seq=0;
    for (uint32_t elapsed=100; elapsed<=7200000; elapsed+=100) {
      uint32_t now=start+elapsed;
      assert(send(continuous,++seq,2,90,80,now));
      continuous.tick(now);
      assert(continuous.armed && continuous.left==90 && continuous.right==80);
    }
    continuous.tick(uint32_t(start+7200000+drive::WATCHDOG_MS-1));
    assert(continuous.armed);
    continuous.tick(uint32_t(start+7200000+drive::WATCHDOG_MS));
    assert(!continuous.armed && continuous.left==0 && continuous.right==0);
  }
  // The active configured duty cap is accepted in both directions, never clamped.
  // Out-of-range requests must disarm and zero rather than truncate to uint8.
  for (int duty : {-drive::MAX_PWM,drive::MAX_PWM}) {
    drive::State edge;
    assert(send(edge,1,1,0,0,0));
    assert(send(edge,2,2,duty,-duty,10));
    assert(edge.armed && edge.left==duty && edge.right==-duty);
  }
  for (int duty : {-drive::MAX_PWM-1,drive::MAX_PWM+1,255,256}) {
    drive::State edge;
    assert(send(edge,1,1,0,0,0));
    assert(!send(edge,2,2,duty,0,10));
    assert(!edge.armed && edge.left==0 && edge.right==0);
  }
  drive::LampBlink lamp;
  assert(!lamp.start(true,100)); assert(lamp.start(false,100));
  for (uint32_t elapsed=0;elapsed<6000;elapsed+=100) {
    assert(!lamp.update(100+elapsed)); assert(lamp.active);
    assert(lamp.level(100+elapsed,true)==((elapsed/1000)%2==0));
  }
  assert(lamp.update(6100)); assert(!lamp.active); assert(lamp.level(6100,true));
  assert(lamp.start(false,0xfffffff0UL)); assert(lamp.update(uint32_t(0xfffffff0UL+6000)));
  puts("Drive safety tests passed (including two simulated 2-hour sessions)");
}
