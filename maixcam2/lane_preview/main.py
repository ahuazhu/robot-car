"""MaixCAM2 lane preview. Run this single file in MaixVision.

No UART is opened and no motion command is sent. UNO should retain the
uno_uart_test firmware, which keeps motor outputs off.
The first version assumes gray/white road with green outside both edges.
"""
import time
import cv2
import numpy as np

WIDTH, HEIGHT = 320, 240
# OpenCV HSV: H=0..179, S/V=0..255. Tune against the mounted camera image.
GREEN_LOW = (32, 55, 30)
GREEN_HIGH = (95, 255, 255)
ROW_FRACTIONS = (0.88, 0.77, 0.66, 0.55, 0.44)


def runs(bits):
    """Return half-open intervals of True pixels."""
    edges = np.diff(np.r_[False, bits, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def detect_road(bgr):
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array(GREEN_LOW), np.array(GREEN_HIGH))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    points = []
    previous = None
    rows = []
    for fraction in ROW_FRACTIONS:
        y = min(h - 1, int(h * fraction))
        lo, hi = max(0, y - 2), min(h, y + 3)
        profile = np.mean(green[lo:hi] > 0, axis=0) >= 0.5
        # Close tiny holes in green caused by glare; do not close across road.
        profile = cv2.morphologyEx(profile.astype(np.uint8)[None, :],
                                  cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))[0] > 0
        candidates = []
        for left, end in runs(~profile):
            width = end - left
            # Both boundaries need green evidence. Do not invent a clipped edge.
            if left < 3 or end > w - 3 or not (w * 0.10 <= width <= w * 0.92):
                continue
            if not (profile[left-3:left].all() and profile[end:end+3].all()):
                continue
            neutral = np.mean(hsv[lo:hi, left:end, 1] < 100)
            if neutral < 0.50:
                continue
            center = (left + end - 1) / 2.0
            if previous is not None:
                old_center, old_width = previous
                if abs(center - old_center) > max(w * 0.20, old_width * 0.65):
                    continue
                if not (0.30 <= width / old_width <= 1.65):
                    continue
                score = width * 0.25 - abs(center - old_center)
            else:
                score = width - 0.5 * abs(center - w / 2)
            candidates.append((score, int(left), int(end - 1), center, width))
        if not candidates:
            rows.append((y, None))
            continue
        _, left, right, center, width = max(candidates)
        point = (y, left, right, center)
        points.append(point)
        rows.append((y, point))
        previous = (center, width)
    # A near-field pair and several consistent rows are required for diagnostics.
    valid = len(points) >= 3 and points[0][0] >= h * 0.75
    # Positive = road center to camera-right. Diagnostic only, not steering.
    offset = points[0][3] - (w - 1) / 2.0 if valid else None
    return {"valid": valid, "offset": offset, "rows": rows, "points": points,
            "green_mask": green}


def annotate(bgr, result):
    out = bgr.copy()
    h, w = out.shape[:2]
    cv2.line(out, (w // 2, h // 3), (w // 2, h - 1), (255, 255, 0), 1)
    for y, point in result["rows"]:
        if point is None:
            cv2.putText(out, "?", (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            continue
        _, left, right, center = point
        cv2.line(out, (left, y), (right, y), (255, 180, 0), 1)
        cv2.circle(out, (left, y), 3, (0, 255, 0), -1)
        cv2.circle(out, (right, y), 3, (0, 255, 0), -1)
        cv2.circle(out, (round(center), y), 3, (0, 255, 255), -1)
    cv2.rectangle(out, (0, 0), (w, 33), (0, 0, 0), -1)
    cv2.putText(out, "PREVIEW ONLY - NO MOTOR COMMAND", (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
    status = "ROAD OK offset=%+.0fpx" % result["offset"] if result["valid"] else "ROAD UNCERTAIN"
    cv2.putText(out, status, (4, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 255) if result["valid"] else (0, 0, 255), 1)
    return out


def main():
    from maix import app, camera, display, image
    cam = camera.Camera(WIDTH, HEIGHT, image.Format.FMT_BGR888)
    screen = display.Display()
    next_log = 0
    print("LANE PREVIEW READY | no UART / no motion commands", flush=True)
    while not app.need_exit():
        frame = cam.read()
        bgr = image.image2cv(frame, ensure_bgr=True, copy=True)
        result = detect_road(bgr)
        screen.show(image.cv2image(annotate(bgr, result), bgr=True, copy=True))
        now = time.monotonic()
        if now >= next_log:
            offset = "unknown" if result["offset"] is None else "%+.1f" % result["offset"]
            print("ROAD %s | rows=%d/5 | offset_px=%s | preview only" %
                  ("OK" if result["valid"] else "UNCERTAIN", len(result["points"]), offset), flush=True)
            next_log = now + 1.0


if __name__ == "__main__":
    main()
