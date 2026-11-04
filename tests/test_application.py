"""Exercise configuration and a complete preview frame across package boundaries."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from tests.support import APP_ROOT
from robot import application
from robot.config import RuntimeSettings
from robot.control.drive_link import PreviewLink
from robot.interface.preview_transport import WebControl


class RuntimeSettingsTests(unittest.TestCase):
    def test_deployed_speed_and_finish_settings_survive_loading(self):
        settings = RuntimeSettings.load(APP_ROOT, {})
        self.assertEqual(
            (
                settings.cruise_pwm,
                settings.confirmed_cruise_pwm,
                settings.straight_cruise_pwm,
                settings.pwm_boost,
            ),
            (120.0, 140.0, 163.0, 37),
        )
        self.assertEqual(settings.runtime["race"]["finish_runout_s"], 5.0)

    def test_preview_override_and_calibration_gate_both_prevent_driving(self):
        self.assertTrue(RuntimeSettings.load(APP_ROOT, {"TARGET_PREVIEW_ONLY": "1"}).preview_only)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cal = json.loads((APP_ROOT / "calibration.json").read_text())
            cal["camera_pose_verified"] = False
            (root / "calibration.json").write_text(json.dumps(cal))
            (root / "runtime.json").write_text((APP_ROOT / "runtime.json").read_text())
            self.assertTrue(RuntimeSettings.load(root, {}).preview_only)


class ApplicationBoundaryTests(unittest.TestCase):
    def test_preview_frame_updates_ui_and_closes_without_drive_commands(self):
        frame = np.full((240, 320, 3), 120, np.uint8)
        shown = []
        events = []

        class Camera:
            def __init__(self, *args):
                pass

            def read(self):
                return frame.copy()

            def close(self):
                events.append("camera_closed")

        class Display:
            def width(self):
                return 320

            def height(self):
                return 240

            def show(self, picture):
                shown.append(picture)

        class Touch:
            def clear(self):
                pass

            def read(self):
                return 0, 0, False

        class Stream:
            def __init__(self, *args):
                self.control = WebControl()
                self.metrics = {}
                self.preview_error = None

            def write(self, picture):
                self.picture = picture

            def stop(self):
                events.append("stream_closed")

        class Link(PreviewLink):
            def send(self, op, left=0, right=0):
                events.append(("send", op, left, right))
                return super().send(op, left, right)

            def stop(self):
                events.append("stopped")
                return super().stop()

            def close(self):
                events.append("link_closed")

        maix = types.ModuleType("maix")
        maix.app = types.SimpleNamespace(need_exit=unittest.mock.Mock(side_effect=[False, True]))
        maix.camera = types.SimpleNamespace(Camera=Camera)
        maix.display = types.SimpleNamespace(Display=Display)
        maix.touchscreen = types.SimpleNamespace(TouchScreen=Touch)
        maix.image = types.SimpleNamespace(
            Format=types.SimpleNamespace(FMT_BGR888=1),
            image2cv=lambda picture, **kwargs: picture,
            cv2image=lambda picture, **kwargs: picture,
        )
        stream = Stream()
        with (
            patch.dict("sys.modules", {"maix": maix}),
            patch.dict(
                "os.environ",
                {
                    "TARGET_PREVIEW_ONLY": "1",
                    "TARGET_TEST_SECONDS": "0",
                    "MAIXPY_DEBUG_IMAGES": "0",
                },
            ),
            patch.object(application, "LiveStream", return_value=stream),
            patch.object(application, "PreviewLink", Link),
            patch.object(
                application, "DriveLink", side_effect=AssertionError("Preview opened UART")
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            application.main(APP_ROOT)
        self.assertEqual(len(shown), 1)
        self.assertEqual(stream.picture.shape, (240, 960, 3))
        self.assertEqual(stream.metrics["frame"], 1)
        self.assertFalse(stream.control.status()["active"])
        self.assertTrue(stream.metrics["preview_only"])
        self.assertTrue(all(event[1] in (0, 3) for event in events if isinstance(event, tuple)))
        self.assertEqual(events[-4:], ["stopped", "stream_closed", "camera_closed", "link_closed"])


if __name__ == "__main__":
    unittest.main()
