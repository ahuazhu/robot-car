from tests.support import ROOT
import unittest, threading, json
from urllib.request import urlopen
from unittest.mock import patch
import numpy as np

from robot.interface.preview_transport import LiveStream
from robot.tasks.lap_counter import LapCounter


class PreviewFaultTests(unittest.TestCase):
    def test_jpeg_failure_preserves_control_and_recovers_next_frame(self):
        stream = LiveStream(0)
        try:
            with patch(
                "robot.interface.preview_transport.cv2.imencode", return_value=(False, None)
            ):
                self.assertTrue(stream.write(np.zeros((8, 8, 3), np.uint8)))
                with stream.condition:
                    self.assertTrue(
                        stream.condition.wait_for(
                            lambda: stream.preview_error is not None, timeout=2
                        )
                    )
            self.assertIsNotNone(stream.preview_error)
            self.assertEqual(stream.sequence, 0)
            self.assertTrue(stream.write(np.zeros((8, 8, 3), np.uint8)))
            with stream.condition:
                self.assertTrue(stream.condition.wait_for(lambda: stream.sequence == 1, timeout=2))
            self.assertIsNone(stream.preview_error)
            self.assertEqual(stream.sequence, 1)
        finally:
            stream.stop()

    def test_finish_frames_keep_logging_status_and_vision_json_available(self):
        # The observed final crossing was at -8.2cm, sigma 3.6cm. The first
        # measured runout frame produced a NumPy boolean and killed v0.6.73.
        counter = LapCounter(dict(enabled=True, laps=2, finish_distance_m=.4))
        counter.begin({}, 0.)
        counter.count = 2
        counter.phase = "FINISH_DISTANCE"
        counter.release_track = dict(normal=np.array([1., 0.]), distance=-.082, sigma=.036)
        stream = LiveStream(0)
        try:
            base = "http://127.0.0.1:%d" % stream.server.server_address[1]
            for i, travel in enumerate((.02, .15, .15, .10)):
                done = counter.update({}, dict(valid=True, rotation=np.eye(2),
                    translation=np.array([-travel, 0.]), sigma_m=.001), i*.1, True)
                snapshot = counter.snapshot()
                json.dumps(dict(active=not done, laps=snapshot), allow_nan=False)
                stream.metrics = dict(laps=snapshot)
                stream.control.update(not done, True, counter.phase,
                                      (0, 0) if done else (100, 100), readiness=dict(laps=snapshot))
                for path in ("/vision", "/status"):
                    with urlopen(base+path, timeout=2) as response:
                        self.assertEqual(response.status, 200)
                        data = json.load(response)
                    laps = data["laps"] if path == "/vision" else data["readiness"]["laps"]
                    self.assertIs(type(laps["rear_line_cleared"]), bool)
                    self.assertEqual(laps["complete"], done)
            self.assertTrue(done)
            with urlopen(base+"/", timeout=2) as response:
                self.assertEqual(response.status, 200)
        finally:
            stream.stop()

    def test_preview_bind_failure_leaves_local_controls_usable(self):
        with patch("http.server.ThreadingHTTPServer", side_effect=OSError("port unavailable")):
            stream = LiveStream(0)
        self.assertIsNotNone(stream.preview_error)
        stream.control.update(False, True, "WAIT START", (0, 0))
        self.assertTrue(stream.control.submit("start", 0)[0])
        self.assertEqual(stream.control.take().action, "start")
        self.assertFalse(stream.write(None))
        stream.stop()

    def test_slow_encoder_drops_pending_old_frames_without_blocking_controls(self):
        entered = threading.Event()
        release = threading.Event()
        encoded = []

        def encode(kind, frame, settings):
            self.assertEqual(settings, [1, 85])
            value = int(frame[0, 0, 0])
            encoded.append(value)
            if len(encoded) == 1:
                entered.set()
                release.wait(2)
            return True, np.array([value], np.uint8)

        stream = LiveStream(0)
        try:
            with patch("robot.interface.preview_transport.cv2.imencode", side_effect=encode):
                self.assertTrue(stream.write(np.full((8, 8, 3), 1, np.uint8)))
                self.assertTrue(entered.wait(1))
                self.assertTrue(stream.write(np.full((8, 8, 3), 2, np.uint8)))
                newest = np.full((8, 8, 3), 3, np.uint8)
                self.assertTrue(stream.write(newest))
                newest[:] = 99
                self.assertEqual(stream.sequence, 0)
                self.assertTrue(stream.control.submit("stop")[0])
                self.assertEqual(stream.control.take().action, "stop")
                release.set()
                with stream.condition:
                    self.assertTrue(
                        stream.condition.wait_for(lambda: stream.sequence == 2, timeout=2)
                    )
            self.assertEqual(encoded, [1, 3])
            self.assertEqual(stream.jpeg, bytes([3]))
        finally:
            release.set()
            stream.stop()

    def test_fresh_composite_can_transfer_ownership_to_encoder(self):
        stream = LiveStream(0)
        composite = np.full((240, 960, 3), 71, np.uint8)
        try:
            # Keep the worker asleep until ownership has been observed. The caller
            # releases this buffer after write; default write still copies (above).
            with stream.condition:
                self.assertTrue(stream.write(composite, owned=True))
                self.assertIs(stream.pending_frame, composite)
            with stream.condition:
                self.assertTrue(stream.condition.wait_for(lambda: stream.sequence == 1, timeout=2))
            self.assertIsNotNone(stream.jpeg)
        finally:
            stream.stop()

    def test_stop_wakes_idle_encoder_and_rejects_late_frames(self):
        stream = LiveStream(0)
        stream.stop()
        self.assertFalse(stream.encoder.is_alive())
        self.assertFalse(stream.write(np.zeros((8, 8, 3), np.uint8)))


if __name__ == "__main__":
    unittest.main()
