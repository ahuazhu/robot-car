"""USB relay failures must invalidate frames without taking down local HTTP."""

import argparse
import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

from esp32s3.video_preview import usb_preview as preview


JPEG = b"\xff\xd8" + b"a" * 120 + b"\xff\xd9"
NEW_JPEG = b"\xff\xd8" + b"b" * 121 + b"\xff\xd9"


class CameraPort:
    def __init__(self, jpeg=JPEG, *, close_error=False, reset_error=False, block=None):
        self.jpeg = jpeg
        self.close_error, self.reset_error, self.block = close_error, reset_error, block
        self.disconnected = threading.Event()
        self.reading = threading.Event()
        self.calls = []
        self.close_count = 0
        self.buffer = b""

    def record(self, operation):
        self.calls.append((operation, threading.get_ident()))

    def reset_input_buffer(self):
        self.record("reset")
        if self.reset_error:
            raise OSError("reset failed after unplug")

    def write(self, data):
        self.record(data)
        if self.disconnected.is_set():
            raise OSError("USB disconnected")
        self.header = b"JPEG %d\n" % len(self.jpeg)
        self.buffer = self.jpeg
        return len(data)

    def readline(self, size=128):
        self.record("readline")
        if self.block == "header":
            self.reading.set()
            time.sleep(0.01)
            return b""
        result, self.header = self.header, b""
        return result

    def read(self, size):
        self.record("read")
        if self.block == "body":
            self.reading.set()
            time.sleep(0.01)
            return b""
        result, self.buffer = self.buffer[:size], self.buffer[size:]
        return result

    def close(self):
        self.record("close")
        self.close_count += 1
        if self.close_error:
            raise OSError("device already gone during close")


def request(relay, path, *, body=None):
    """Exercise HTTP routing without opening sockets or serial devices."""
    handler = object.__new__(preview.make_handler(relay, "camera page"))
    handler.path = path
    handler.wfile = io.BytesIO() if body is None else body
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock()
    handler.do_GET()
    return handler.send_response.call_args.args[0], handler.wfile.getvalue()


class PreviewRelayTests(unittest.TestCase):
    def setUp(self):
        quiet = contextlib.redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def relay(self, factory, **kwargs):
        options = dict(startup_s=0.0, retry_s=0.01, frame_interval_s=0.01)
        options.update(kwargs)
        relay = preview.PreviewRelay("/dev/test-esp", serial_factory=factory, **options)
        self.addCleanup(relay.stop)
        return relay

    def wait_for(self, relay, predicate):
        with relay.condition:
            self.assertTrue(relay.condition.wait_for(predicate, timeout=2.0))

    def test_disconnect_and_failing_close_clear_cache_then_reopen_same_port(self):
        first, second = CameraPort(close_error=True), CameraPort(NEW_JPEG)
        reopen_attempted, allow_reopen = threading.Event(), threading.Event()
        opens = []

        def factory(path, baud, **kwargs):
            opens.append((path, baud, kwargs, threading.get_ident()))
            if len(opens) == 1:
                return first
            reopen_attempted.set()
            if not allow_reopen.wait(2.0):
                raise OSError("reconnect held by test")
            return second

        relay = self.relay(factory)
        self.addCleanup(allow_reopen.set)
        relay.start()
        self.wait_for(relay, lambda: relay.sequence >= 1)
        self.assertEqual(request(relay, "/capture"), (200, JPEG))
        first.disconnected.set()
        self.assertTrue(reopen_attempted.wait(2.0))
        failed_count = relay.sequence
        self.assertIsNone(relay.latest)
        self.assertFalse(relay.status()["connected"])
        self.assertIn("disconnected", relay.status()["error"])
        self.assertEqual(request(relay, "/capture")[0], 503)
        self.assertEqual(request(relay, "/stream")[0], 503)
        self.assertTrue(relay.worker.is_alive())
        self.assertEqual(first.close_count, 1)
        allow_reopen.set()
        self.wait_for(relay, lambda: relay.sequence > failed_count)
        self.assertEqual(request(relay, "/capture"), (200, NEW_JPEG))
        self.assertTrue(relay.status()["connected"])
        self.assertEqual(relay.status()["error"], "")
        self.assertTrue(all(row[0:2] == ("/dev/test-esp", 115200) for row in opens))
        relay.stop()
        operations = first.calls + second.calls
        self.assertEqual({ident for _, ident in operations}, {relay.worker.ident})
        self.assertTrue(all(item == b"J" for item, _ in operations if isinstance(item, bytes)))

    def test_reset_error_and_second_close_error_do_not_kill_worker(self):
        bad = CameraPort(reset_error=True, close_error=True)
        good = CameraPort()
        factory = Mock(side_effect=[bad, good])
        relay = self.relay(factory)
        relay.start()
        self.wait_for(relay, lambda: relay.sequence >= 1)
        self.assertTrue(relay.status()["connected"])
        self.assertEqual(bad.close_count, 1)
        self.assertEqual(factory.call_count, 2)

    def test_missing_camera_keeps_http_available_and_retry_wait_is_interruptible(self):
        attempted = threading.Event()

        def absent(*args, **kwargs):
            attempted.set()
            raise OSError("serial port not found")

        relay = self.relay(absent, retry_s=30.0)
        relay.start()
        self.assertTrue(attempted.wait(2.0))
        self.wait_for(relay, lambda: "not found" in relay.failure)
        self.assertEqual(request(relay, "/")[0], 200)
        status_code, raw = request(relay, "/status")
        self.assertEqual(status_code, 200)
        self.assertEqual(set(json.loads(raw)), {"frames", "error", "connected", "age_ms"})
        self.assertFalse(json.loads(raw)["connected"])
        self.assertEqual(request(relay, "/capture")[0], 503)
        self.assertEqual(request(relay, "/control")[0], 404)
        started = time.monotonic()
        relay.stop()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertFalse(relay.worker.is_alive())

    def test_five_second_old_frame_is_never_served_as_live(self):
        clock = types.SimpleNamespace(now=10.0)
        relay = self.relay(Mock(), clock=lambda: clock.now)
        relay._publish(JPEG)
        self.assertTrue(relay.status()["connected"])
        clock.now = 15.0
        self.assertFalse(relay.status()["connected"])
        self.assertEqual(relay.status()["age_ms"], 5000)
        self.assertIn("stale", relay.status()["error"])
        self.assertIsNone(relay.wait_frame(0))
        self.assertEqual(request(relay, "/capture")[0], 503)
        self.assertEqual(request(relay, "/stream")[0], 503)

    def test_existing_stream_ends_on_disconnect_without_repeating_last_jpeg(self):
        relay = self.relay(Mock())
        relay._publish(JPEG)

        class DisconnectAfterFrame(io.BytesIO):
            def flush(self):
                relay._clear("USB disconnected")

        code, body = request(relay, "/stream", body=DisconnectAfterFrame())
        self.assertEqual(code, 200)
        self.assertEqual(body.count(JPEG), 1)
        self.assertEqual(request(relay, "/capture")[0], 503)

    def test_stop_interrupts_boot_wait_and_worker_owns_close(self):
        port, opened = CameraPort(), threading.Event()

        def factory(*args, **kwargs):
            opened.set()
            return port

        relay = self.relay(factory, startup_s=30.0)
        relay.start()
        self.assertTrue(opened.wait(2.0))
        started = time.monotonic()
        relay.stop()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(port.calls, [("close", relay.worker.ident)])
        self.assertFalse(relay.status()["connected"])

    def test_stop_interrupts_header_and_image_reads_without_publishing_partial_frames(self):
        for phase in ("header", "body"):
            with self.subTest(phase=phase):
                port = CameraPort(block=phase)
                relay = self.relay(lambda *args, **kwargs: port)
                relay.start()
                self.assertTrue(port.reading.wait(2.0))
                started = time.monotonic()
                relay.stop()
                self.assertLess(time.monotonic() - started, 0.5)
                self.assertFalse(relay.worker.is_alive())
                self.assertEqual(port.close_count, 1)
                self.assertEqual(relay.sequence, 0)
                self.assertIsNone(relay.latest)

    def test_invalid_jpeg_is_rejected_before_publication(self):
        port = CameraPort(b"x" * len(JPEG))
        with self.assertRaisesRegex(RuntimeError, "Incomplete JPEG"):
            preview.capture(port, threading.Event())
        self.assertEqual([op for op, _ in port.calls if isinstance(op, bytes)], [b"J"])


class PreviewConfigurationTests(unittest.TestCase):
    def test_module_import_does_not_parse_args_open_serial_or_start_server(self):
        with (
            patch.object(preview.serial, "Serial", side_effect=AssertionError("opened serial")),
            patch.object(argparse.ArgumentParser, "parse_args", side_effect=AssertionError("CLI")),
        ):
            importlib.reload(preview)

    def test_maix_url_accepts_only_http_hosts_without_credentials(self):
        for value in ("", "http://10.0.0.2:8000/", "https://camera.example/view?mode=live"):
            self.assertEqual(preview.validate_maix_url(value), value)
        for value in (
            "javascript:alert(1)", "file:///etc/passwd", "//camera.example/", "http:///x",
            "http://user:secret@camera/", "http://@camera/", "http://camera:bad/",
            "http://camera:65536/", "http://camera/\ncontrol", "http://camera\\other/",
        ):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                preview.validate_maix_url(value)

    def test_page_configuration_cannot_terminate_script_element(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "preview.html"
            path.write_text('<script>const config=__PREVIEW_CONFIG__;</script>', encoding="utf-8")
            value = "http://camera/?x=</script><script>alert(1)</script>"
            page = preview.load_page(value, path)
            self.assertEqual(page.count("</script>"), 1)
            self.assertNotIn("__PREVIEW_CONFIG__", page)
            config = page.removeprefix("<script>const config=").removesuffix(";</script>")
            self.assertEqual(json.loads(config), {"maixUrl": value})
            self.assertEqual(json.loads(preview.load_page("", path).split("=", 1)[1][:-10]),
                             {"maixUrl": ""})

    def test_main_binds_only_localhost_and_cleans_up_after_interrupt(self):
        relay, server = Mock(), Mock()
        server.serve_forever.side_effect = KeyboardInterrupt
        with (
            patch.object(preview, "PreviewRelay", return_value=relay) as constructor,
            patch.object(preview, "ThreadingHTTPServer", return_value=server) as http,
            patch.object(preview, "load_page", return_value="page") as page,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            preview.main(["--serial", "/dev/test", "--port", "8123", "--maix-url", "http://maix/"])
        constructor.assert_called_once_with("/dev/test")
        self.assertEqual(http.call_args.args[0], ("127.0.0.1", 8123))
        page.assert_called_once_with("http://maix/")
        relay.start.assert_called_once()
        relay.stop.assert_called_once()
        server.server_close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
