#ifndef MINIAUTO_TURN_PACER_H
#define MINIAUTO_TURN_PACER_H

#include "FollowController.h"

namespace follow {

// 保持电机启动所需的 PWM，通过短转、短停降低平均转向速度。
// 每次 loop 都更新，不阻塞摄像头读取或串口停车。
class TurnPacer {
 public:
  static const uint16_t ON_MS = 100;
  static const uint16_t OFF_MS = 120;

  void reset() {
    previous_ = STOP;
    phaseStarted_ = 0;
    reversing_ = false;
  }

  Motion update(uint32_t now, Motion desired) {
    if (desired != LEFT && desired != RIGHT) {
      reset();
      return desired;
    }
    if (desired != previous_) {
      // 左右方向直接切换时，先停一下再反向。
      reversing_ = previous_ == LEFT || previous_ == RIGHT;
      previous_ = desired;
      phaseStarted_ = now;
    }
    if (reversing_) {
      if (uint32_t(now - phaseStarted_) < OFF_MS) return STOP;
      phaseStarted_ += OFF_MS;
      reversing_ = false;
    }
    const uint32_t elapsed = uint32_t(now - phaseStarted_);
    return elapsed % (ON_MS + OFF_MS) < ON_MS ? desired : STOP;
  }

 private:
  Motion previous_ = STOP;
  uint32_t phaseStarted_ = 0;
  bool reversing_ = false;
};

}  // namespace follow
#endif
