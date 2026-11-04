from tests.support import ROOT
import unittest, threading
from unittest.mock import patch
import numpy as np

from robot.interface.preview_transport import LiveStream


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

    def test_stop_wakes_idle_encoder_and_rejects_late_frames(self):
        stream = LiveStream(0)
        stream.stop()
        self.assertFalse(stream.encoder.is_alive())
        self.assertFalse(stream.write(np.zeros((8, 8, 3), np.uint8)))


if __name__ == "__main__":
    unittest.main()
