"""Native-screen layout: complete camera frame, separate status and touch buttons."""

import cv2
import numpy as np


class DeviceScreen:
    def __init__(self, width, height, frame_size=(320, 240)):
        self.width, self.height = int(width), int(height)
        self.frame_width, self.frame_height = frame_size
        if min(self.width, self.height) < 96 or min(frame_size) <= 0:
            raise ValueError("Invalid display dimensions")
        self.padding = max(3, round(min(self.width, self.height) * 0.012))
        self.font_scale = max(0.35, min(self.width / 320, self.height / 240) * 0.4)
        (_, text_height), baseline = cv2.getTextSize(
            "Ag", cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
        )
        self.text_height = text_height
        self.line_height = text_height + baseline + self.padding
        self.header_height = 2 * self.line_height + self.padding
        self.footer_height = max(round(self.height * 0.1), self.line_height + 2 * self.padding)
        self.footer_top = self.height - self.footer_height
        available = self.footer_top - self.header_height
        fit = min(self.width / self.frame_width, available / self.frame_height)
        w, h = round(self.frame_width * fit), round(self.frame_height * fit)
        self.frame_rect = ((self.width - w) // 2, self.header_height + (available - h) // 2, w, h)
        middle = round(self.width / 2)
        self.buttons = [
            dict(
                action="start", label="ARM", rect=(0, self.footer_top, middle, self.footer_height)
            ),
            dict(
                action="stop",
                label="STOP",
                rect=(middle, self.footer_top, self.width - middle, self.footer_height),
            ),
        ]

        (exit_text_width, _), _ = cv2.getTextSize(
            "EXIT", cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
        )
        exit_width = max(round(self.width * 0.15), exit_text_width + 2 * self.padding)
        self.exit_rect = (
            self.width - self.padding - exit_width,
            self.padding,
            exit_width,
            self.header_height - 2 * self.padding,
        )
        self.buttons.append(dict(action="exit", label="EXIT", rect=self.exit_rect))

    def action_at(self, x, y):
        for button in self.buttons:
            bx, by, w, h = button["rect"]
            if bx <= x < bx + w and by <= y < by + h:
                return button["action"]
        return None

    def compose(self, preview, reason, pwm, frame, arm_ready=False, armed=False):
        if preview.shape != (self.frame_height, self.frame_width, 3):
            raise ValueError("Device preview shape mismatch")
        # The analysis image is never modified. Resize the full image only;
        # controls and text draw exclusively outside this viewport.
        canvas = np.zeros((self.height, self.width, 3), np.uint8)
        x, y, w, h = self.frame_rect
        method = cv2.INTER_AREA if w < self.frame_width else cv2.INTER_LINEAR
        canvas[y : y + h, x : x + w] = cv2.resize(preview, (w, h), interpolation=method)
        pad = self.padding
        header = canvas[: self.header_height, : self.exit_rect[0] - pad]
        cv2.putText(
            header,
            reason,
            (pad, pad + self.text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            header,
            "L=%d R=%d frame=%d" % (*pwm, frame),
            (pad, pad + self.text_height + self.line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            1,
        )
        colors = {
            "start": (0, 120, 180) if armed else (170, 80, 0) if arm_ready else (55, 55, 55),
            "stop": (0, 0, 180),
            "exit": (90, 50, 30),
        }
        for button in self.buttons:
            bx, by, bw, bh = button["rect"]
            label = button["label"]
            cv2.rectangle(
                canvas, (bx, by), (bx + bw - 1, by + bh - 1), colors[button["action"]], -1
            )
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
            cv2.putText(
                canvas,
                label,
                (bx + (bw - tw) // 2, by + (bh + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
            )
        return canvas

    def snapshot(self):
        return dict(
            width=self.width,
            height=self.height,
            frame_rect=list(self.frame_rect),
            header_height=self.header_height,
            footer_top=self.footer_top,
            buttons=self.buttons,
            fit="CONTAIN",
            camera_frame_complete=True,
        )
