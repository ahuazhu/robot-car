#ifndef MINIAUTO_FOLLOW_CONTROLLER_H
#define MINIAUTO_FOLLOW_CONTROLLER_H

#include <stdint.h>

namespace follow {

struct Target {
  uint8_t x, y, width, height;
};

enum Motion : uint8_t { STOP, FORWARD, LEFT, RIGHT };
enum Phase : uint8_t { CALIBRATING, TRACKING };

// Hiwonder ColorDetection 源码输出 240 x 240 图像的包围框。
inline bool validTarget(const Target &t) {
  return t.x < 240 && t.y < 240 && t.width >= 10 &&
         t.width <= 230 && t.height >= 10 && t.height <= 240 &&
         int(t.x) - int(t.width) / 2 > 2 &&
         int(t.x) + int(t.width) / 2 < 237;
}

class Controller {
 public:
  void begin(uint32_t) {
    phase_ = CALIBRATING;
    reference_ = 0;
    goodFrames_ = 0;
    advancing_ = false;
    clearCalibration();
  }

  Motion update(uint32_t now, bool received, const Target &t) {
    const bool valid = received && validTarget(t);
    if (phase_ == CALIBRATING) {
      // 至少 3 次观测且稳定持续 100 ms；多次 I2C 读取可能来自同一帧。
      if (!valid || t.x < 102 || t.x > 138 ||
          t.width < 20 || t.width > 180) {
        clearCalibration();
        return STOP;
      }
      if (!samples_) {
        minWidth_ = maxWidth_ = t.width;
        calibrationStarted_ = now;
      }
      uint8_t lo = t.width < minWidth_ ? t.width : minWidth_;
      uint8_t hi = t.width > maxWidth_ ? t.width : maxWidth_;
      uint8_t tolerance = lo / 10;
      if (tolerance < 3) tolerance = 3;
      if (hi - lo > tolerance) {
        clearCalibration();
        lo = hi = t.width;
        calibrationStarted_ = now;
      }
      minWidth_ = lo;
      maxWidth_ = hi;
      if (samples_ < 16) {
        widthSum_ += t.width;
        ++samples_;
      }
      if (samples_ >= 3 && uint32_t(now - calibrationStarted_) >= 100) {
        reference_ = uint8_t(widthSum_ / samples_);
        phase_ = TRACKING;
        goodFrames_ = 2;  // 标定已确认目标，当次读取即可执行跟随。
      } else {
        return STOP;
      }
    }

    if (!valid) {
      goodFrames_ = 0;
      advancing_ = false;
      return STOP;
    }
    if (goodFrames_ < 2) ++goodFrames_;
    if (goodFrames_ < 2) return STOP;

    uint8_t closeMargin = reference_ / 8;
    if (closeMargin < 5) closeMargin = 5;
    // 目标明显更近时，停止所有动作，包括转向。
    if (int(t.width) >= int(reference_) + closeMargin) {
      advancing_ = false;
      return STOP;
    }
    if (t.x < 100 || t.x > 140) {
      advancing_ = false;
      return t.x < 100 ? LEFT : RIGHT;
    }
    // 启动与停止使用不同阈值，减少尺寸抖动引起的反复启停。
    uint8_t startMargin = reference_ / 12;
    if (startMargin < 4) startMargin = 4;
    if (t.width >= reference_) advancing_ = false;
    else if (int(t.width) < int(reference_) - startMargin) advancing_ = true;
    return advancing_ ? FORWARD : STOP;
  }

  Phase phase() const { return phase_; }
  uint8_t referenceWidth() const { return reference_; }

 private:
  void clearCalibration() {
    samples_ = 0;
    widthSum_ = 0;
    minWidth_ = maxWidth_ = 0;
    calibrationStarted_ = 0;
  }

  Phase phase_ = CALIBRATING;
  uint32_t calibrationStarted_ = 0;
  uint16_t widthSum_ = 0;
  uint8_t samples_ = 0, minWidth_ = 0, maxWidth_ = 0;
  uint8_t reference_ = 0, goodFrames_ = 0;
  bool advancing_ = false;
};

}  // namespace follow
#endif
