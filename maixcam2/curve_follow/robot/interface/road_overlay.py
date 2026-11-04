"""Draw measured and remembered road support without changing navigation."""

import cv2
import numpy as np
from robot.core.geometry import contiguous_runs


def render_target(bgr, result, details, cal):
    out = bgr.copy()
    inverse = np.linalg.inv(np.asarray(cal["H_normalized"]))

    def project(path):
        q = np.column_stack((path, np.ones(len(path)))) @ inverse.T
        good = np.abs(q[:, 2]) > 1e-6
        uv = np.zeros((len(q), 2))
        uv[good] = q[good, :2] / q[good, 2, None] * [320, 240]
        good &= np.isfinite(uv).all(axis=1) & (abs(uv).max(axis=1) < 10000)
        return np.rint(np.clip(uv, -10000, 10000)).astype(np.int32), good

    def draw(path, color, thickness):
        pts, valid = project(path)
        for ids in contiguous_runs(valid):
            if len(ids) > 1:
                cv2.polylines(out, [pts[ids]], False, color, thickness)

    for side, f in details["fits"].items():
        if f is not None:
            draw(f["path"], (0, 220, 255) if side == "left" else (255, 180, 0), 2)
    path = np.asarray(result.get("path", []))
    if len(path) >= 2:
        fresh = np.asarray(result.get("path_observed_mask", np.ones(len(path), bool)), bool)
        if len(fresh) != len(path):
            fresh = np.ones(len(path), bool)
        # An orange segment has actual past observation support. It is not a
        # polynomial drawn outside the local road model's geometric support.
        for ids in contiguous_runs(fresh):
            if len(ids) >= 2:
                draw(path[ids], (0, 230, 0), 2)
        for i in range(0, len(path) - 1, 2):
            if not (fresh[i] and fresh[i + 1]):
                draw(path[i : i + 2], (0, 165, 255), 2)
    if result.get("target_m") is not None:
        pts, good = project(np.asarray([result["target_m"]]))
        if good[0]:
            x, y = pts[0]
            inside = 0 <= x < 320 and 38 <= y < 209
            mark = (int(np.clip(x, 8, 311)), int(np.clip(y, 65, 195)))
            cv2.drawMarker(
                out,
                mark,
                (0, 0, 255),
                cv2.MARKER_CROSS if inside else cv2.MARKER_TRIANGLE_UP,
                16,
                2,
            )
            cv2.putText(
                out,
                "TARGET" if inside else "OFF VIEW",
                (max(0, min(230, mark[0] - 30)), max(65, mark[1] - 10)),
                0,
                0.4,
                (0, 0, 255),
                1,
            )
    bird = cv2.resize(cv2.flip(details["bird"], 0), (320, 240))
    return out, bird
