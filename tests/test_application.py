"""Exercise configuration and a complete preview frame across package boundaries."""

import contextlib
from dataclasses import replace
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
            (135.0, 155.0, 163.0, 37),
        )
        self.assertEqual(settings.runtime["race"]["finish_distance_m"], 0.4)
        self.assertEqual(settings.motor_rise_pwm_s, 400)
        self.assertEqual(settings.runtime['speed_profile']['acceleration_m_s2'], .4)

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
    def test_disabled_counting_skips_marker_pipeline_but_keeps_crosswalk_detection(self):
        settings=RuntimeSettings.load(APP_ROOT,{'TARGET_PREVIEW_ONLY':'1'})
        settings=replace(settings,runtime=dict(settings.runtime,race=dict(
            settings.runtime['race'],enabled=False)))
        with patch.object(application.RuntimeSettings,'load',return_value=settings), \
             patch.object(application.LapMarkerDetector,'detect',side_effect=AssertionError) as detect, \
             patch.object(application.LapMarkerDetector,'draw',side_effect=AssertionError) as draw, \
             patch.object(application.LapCounter,'update',side_effect=AssertionError) as count, \
             patch.object(application.CrosswalkDetector,'detect',autospec=True,
                          side_effect=application.CrosswalkDetector.detect) as crosswalk:
            run=self.run_application(3)
        detect.assert_not_called();draw.assert_not_called();count.assert_not_called()
        self.assertEqual(crosswalk.call_count,3)
        self.assertFalse(run.stream.metrics['laps']['enabled'])
        self.assertEqual(run.stream.metrics['laps']['detector']['reason'],'DISABLED_FOR_DRIVING_TEST')
        self.assertEqual(run.stream.metrics['profile_ms']['lap_counter'],0.)
        self.assertFalse(run.stream.metrics['laps']['complete'])

    def test_counting_pipeline_can_be_restored_with_its_original_enable_switch(self):
        settings=RuntimeSettings.load(APP_ROOT,{'TARGET_PREVIEW_ONLY':'1'})
        settings=replace(settings,runtime=dict(settings.runtime,race=dict(
            settings.runtime['race'],enabled=True)))
        with patch.object(application.RuntimeSettings,'load',return_value=settings), \
             patch.object(application.LapMarkerDetector,'detect',autospec=True,
                          side_effect=application.LapMarkerDetector.detect) as detect, \
             patch.object(application.LapCounter,'update',autospec=True,
                          side_effect=application.LapCounter.update) as count:
            run=self.run_application(2)
        self.assertEqual(detect.call_count,2)
        self.assertEqual(count.call_count,2)
        self.assertTrue(run.stream.metrics['laps']['enabled'])

    def run_application(
        self, frames=1, *, show_delay=0.0, touch_actions=None, preview=True,
        camera_intervals=None, debug=False,
    ):
        """Run the real application/session with clocked camera and output ports."""
        clock = types.SimpleNamespace(now=100.0, frame=0)
        events, shown, written, rendered, composed = [], [], [], [], []
        control_frames, touch_frames, stops, saved = [], [], [], []
        touch_actions = touch_actions or {}
        layout = application.DeviceScreen(320, 240)
        real_screen = application.DeviceScreen
        real_session = application.DrivingSession
        real_render = application.render_target

        def advance(seconds):
            clock.now = round(clock.now + seconds, 9)

        class Camera:
            def __init__(self, *args):
                pass

            def read(self):
                interval = camera_intervals[clock.frame] if camera_intervals else 0.05
                advance(interval)
                clock.frame += 1
                return np.full((240, 320, 3), 120 + clock.frame, np.uint8)

            def close(self):
                events.append("camera_closed")

        class Display:
            def width(self):
                return 320

            def height(self):
                return 240

            def show(self, picture):
                started = clock.now
                advance(show_delay)
                shown.append((clock.frame, started, clock.now, picture.copy()))

        class Touch:
            def clear(self):
                pass

            def read(self):
                touch_frames.append(clock.frame)
                action = touch_actions.get(clock.frame)
                if action is None:
                    return 0, 0, False
                button = next(b for b in layout.buttons if b["action"] == action)
                x, y, width, height = button["rect"]
                return x + width // 2, y + height // 2, True

        class Control(WebControl):
            def update(self, *args, **kwargs):
                super().update(*args, **kwargs)
                control_frames.append(self.state["readiness"]["frame"])

        class Stream:
            def __init__(self):
                self.control = Control()
                self.publications = []
                self.metrics = {}
                self.preview_error = None

            @property
            def metrics(self):
                return self._metrics

            @metrics.setter
            def metrics(self, value):
                self._metrics = value
                if value:
                    # Equivalent to an HTTP read exactly when a new frame appears.
                    self.publications.append(json.loads(json.dumps(value)))

            def write(self, picture, *, owned=False):
                self.picture = picture.copy()
                written.append((clock.frame, clock.now, self.picture))

            def stop(self):
                events.append("stream_closed")

        class Link(PreviewLink):
            def __init__(self):
                self.armed = False
                self.ack = [0, 0, 0, 0, 0, 7]

            def send(self, op, left=0, right=0, *, lease_s=None):
                events.append(("send", op, left, right))
                if preview:
                    return super().send(op, left, right)
                if op in (0, 1):
                    self.armed = op == 1
                self.ack = [self.ack[0] + 1, int(self.armed), left, right, 0, 7]
                return self.ack

            def snapshot(self):
                return dict(
                    ack=self.ack[:], armed=self.armed, ack_age_ms=0,
                    brake_lamp_supported=True, brake_lamp_on=True,
                    confirmed_forward_s=0.0,
                )

            def stop(self):
                answer = super().stop() if preview else self.send(0)
                stops.append(clock.frame)
                events.append("stopped")
                return answer

            def close(self):
                events.append("link_closed")

        class Screen(real_screen):
            def compose(self, picture, reason, pwm, frame, **kwargs):
                composed.append((frame, reason, pwm, kwargs["armed"]))
                return super().compose(picture, reason, pwm, frame, **kwargs)

        def render(bgr, result, details, calibration, **kwargs):
            rendered.append(clock.frame)
            out, bird = real_render(bgr, result, details, calibration, **kwargs)
            if debug:
                # Distinguish the renderer's current control image from raw data.
                out[-1, -1] = 200 + clock.frame
            return out, bird

        def save(path, picture):
            saved.append((Path(path).name, clock.frame, picture.copy()))
            return True

        maix = types.ModuleType("maix")
        maix.app = types.SimpleNamespace(need_exit=lambda: clock.frame >= frames)
        maix.camera = types.SimpleNamespace(Camera=Camera)
        maix.display = types.SimpleNamespace(Display=Display)
        maix.touchscreen = types.SimpleNamespace(TouchScreen=Touch)
        maix.image = types.SimpleNamespace(
            Format=types.SimpleNamespace(FMT_BGR888=1),
            image2cv=lambda picture, **kwargs: picture,
            cv2image=lambda picture, **kwargs: picture,
        )
        stream, link = Stream(), Link()
        with tempfile.TemporaryDirectory() as run_dir, contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {"maix": maix}))
            stack.enter_context(patch.dict("os.environ", {
                "TARGET_PREVIEW_ONLY": "1" if preview else "0",
                "TARGET_TEST_SECONDS": "0", "START_LIGHT_DISARMED": "1",
                "MAIXPY_DEBUG_IMAGES": "1" if debug else "0",
                "MAIXPY_SKILL_RUN_DIR": run_dir,
            }))
            stack.enter_context(patch.object(application, "time", types.SimpleNamespace(
                monotonic=lambda: clock.now, perf_counter=lambda: clock.now,
            )))
            signal = application.StartSignal({"enabled": True}, clock=lambda: clock.now)
            signal.key = "a" * 64
            signal.peer = dict(boot="1234567890abcdef", paired=True, armed=False)
            signal.received, signal.error = clock.now, ""
            # The disabled worker has completed its initial idle-peer check.
            signal.start = lambda: setattr(signal, "reset_remote_pending", False)
            signal.close = lambda: None
            stack.enter_context(patch.object(application, "StartSignal", return_value=signal))
            stack.enter_context(patch.object(application, "DrivingSession", side_effect=(
                lambda *args: real_session(*args, clock=lambda: clock.now)
            )))
            stack.enter_context(patch.object(application, "LiveStream", return_value=stream))
            stack.enter_context(patch.object(application, "DeviceScreen", Screen))
            stack.enter_context(patch.object(application, "render_target", side_effect=render))
            stack.enter_context(patch.object(application, "PreviewLink", return_value=link))
            drive = stack.enter_context(patch.object(application, "DriveLink", return_value=link))
            if preview:
                drive.side_effect = AssertionError("Preview opened UART")
            stack.enter_context(patch.object(application, "MotorChannel", side_effect=lambda x: x))
            stack.enter_context(patch.object(application.cv2, "imwrite", side_effect=save))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            application.main(APP_ROOT)
        return types.SimpleNamespace(
            shown=shown, written=written, rendered=rendered, composed=composed,
            controls=control_frames, touches=touch_frames, stops=stops,
            saved=saved, events=events, stream=stream,
            output=output.getvalue(),
        )

    def test_preview_frame_updates_ui_and_closes_without_drive_commands(self):
        run = self.run_application()
        self.assertEqual(len(run.shown), 1)
        self.assertEqual(run.stream.picture.shape, (240, 960, 3))
        self.assertEqual(run.stream.metrics["frame"], 1)
        self.assertFalse(run.stream.control.status()["active"])
        self.assertTrue(run.stream.metrics["preview_only"])
        self.assertTrue(all(e[1] in (0, 3) for e in run.events if isinstance(e, tuple)))
        self.assertEqual(run.events[-4:], ["stopped", "stream_closed", "camera_closed", "link_closed"])

    def test_idle_log_throttle_keeps_every_control_and_preview_frame(self):
        run = self.run_application(4, camera_intervals=[.6]*4)
        logs = [line for line in run.output.splitlines() if line.startswith('CONTROL ')]
        self.assertEqual(len(logs), 1)
        self.assertEqual(run.controls, [1,2,3,4])
        self.assertEqual(len(run.stream.publications), 4)

    def test_arm_and_stop_log_immediately_despite_idle_throttle(self):
        run = self.run_application(4, preview=False, touch_actions={2:'start',4:'stop'})
        logs = [json.loads(line[len('CONTROL '):]) for line in run.output.splitlines()
                if line.startswith('CONTROL ')]
        self.assertEqual([entry['frame'] for entry in logs], [1,2,4])

    def test_published_vision_frame_is_complete_before_http_can_read_it(self):
        run = self.run_application(4)
        self.assertEqual([p["frame"] for p in run.stream.publications], [1, 2, 3, 4])
        for published in run.stream.publications:
            self.assertIn("frame_processing", published["profile_ms"])
            self.assertIn("logging", published["profile_ms"])
            readiness = published["readiness"]
            for key in ("start_light", "crosswalk", "laps", "steering", "speed_profile_shadow"):
                self.assertEqual(published[key], readiness[key])
            shadow = published["speed_profile_shadow"]
            self.assertEqual(shadow["source_frame"], published["frame"])
            self.assertFalse(shadow["controls_motor"])
            self.assertEqual(shadow["mode"], "DISABLED")
            self.assertEqual(shadow["reason"], "ACTIVE_PROFILE_OWNS_SPEED")
            self.assertIn("speed_profile_shadow", published["profile_ms"])
        self.assertEqual(run.stream.publications[-1], json.loads(json.dumps(run.stream.metrics)))

    def test_control_and_touch_update_every_frame_while_both_previews_are_throttled(self):
        run = self.run_application(20)
        self.assertEqual(run.controls, list(range(1, 21)))
        self.assertEqual(run.touches, run.controls)
        self.assertEqual(run.stream.metrics["frame"], 20)
        self.assertTrue(4 <= len(run.shown) <= 6)
        self.assertTrue(8 <= len(run.written) <= 10)
        self.assertLess(len(run.shown), len(run.written))
        self.assertNotIn(2, run.rendered)
        self.assertEqual(set(run.rendered), {r[0] for r in run.shown + run.written})
        for _, _, picture in run.written:
            self.assertEqual(picture.shape, (240, 960, 3))

    def test_slow_display_schedules_after_completion_without_catchup_frames(self):
        run = self.run_application(12, show_delay=0.30)
        self.assertEqual(run.controls, list(range(1, 13)))
        self.assertGreaterEqual(len(run.shown), 2)
        self.assertLessEqual(len(run.shown), 4)
        self.assertNotIn(2, run.rendered)
        for before, after in zip(run.shown, run.shown[1:]):
            self.assertGreaterEqual(after[1] - before[2], 0.20 - 1e-8)
            self.assertGreater(after[0] - before[0], 1)

    def test_touch_stop_is_handled_on_a_frame_with_no_rendering(self):
        run = self.run_application(4, touch_actions={2: "stop"})
        self.assertNotIn(2, run.rendered)
        self.assertIn(2, run.stops)
        self.assertEqual(run.controls, [1, 2, 3, 4])
        self.assertFalse(run.stream.control.status()["active"])

    def test_active_mode_never_enters_shadow_geometry_or_changes_stop_handling(self):
        options = dict(frames=4, preview=False, touch_actions={2: "start", 4: "stop"})
        ordinary = self.run_application(**options)
        with patch("robot.navigation.speed_observer.supported_segment", side_effect=RuntimeError) as shadow:
            failed = self.run_application(**options)
        shadow.assert_not_called()
        self.assertEqual(ordinary.events, failed.events)
        self.assertEqual(ordinary.controls, failed.controls)
        self.assertEqual(ordinary.stops, failed.stops)
        self.assertEqual(failed.stream.metrics["speed_profile_shadow"]["reason"], "ACTIVE_PROFILE_OWNS_SPEED")

    def test_arm_and_stop_refresh_screen_immediately_between_display_deadlines(self):
        run = self.run_application(4, preview=False, touch_actions={2: "start", 4: "stop"})
        composed = {row[0]: row for row in run.composed}
        self.assertIn(2, composed)
        self.assertTrue(composed[2][3])
        self.assertIn(4, composed)
        self.assertFalse(composed[4][3])
        self.assertIn(4, run.stops)
        self.assertTrue(all(e[1] in (0, 3) for e in run.events if isinstance(e, tuple)))

    def test_debug_deadline_renders_current_frame_even_when_previews_are_not_due(self):
        run = self.run_application(2, camera_intervals=[1.95, 0.05], debug=True)
        self.assertEqual([r[0] for r in run.shown], [1])
        self.assertEqual([r[0] for r in run.written], [1])
        self.assertEqual(run.rendered, [1, 2])
        images = {name: (frame, picture) for name, frame, picture in run.saved}
        self.assertEqual(set(images), {"raw-2.jpg", "control-2.jpg"})
        self.assertEqual(images["raw-2.jpg"][0], 2)
        self.assertEqual(images["control-2.jpg"][0], 2)
        self.assertEqual(int(images["raw-2.jpg"][1][-1, -1, 0]), 122)
        self.assertEqual(int(images["control-2.jpg"][1][-1, -1, 0]), 202)

if __name__ == "__main__":
    unittest.main()
