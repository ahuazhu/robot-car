"""Wireless frames and provisioning must never depend on ongoing USB capture."""

import argparse
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

from esp32s3.video_preview import wireless_preview as wireless

JPEG = b"\xff\xd8" + b"wireless-frame" * 12 + b"\xff\xd9"


class QuickEvent:
    def is_set(self):
        return False

    def wait(self, seconds):
        return False


class WirelessPreviewTests(unittest.TestCase):
    def test_credentials_validate_utf8_bytes_and_do_not_echo_password(self):
        self.assertEqual(wireless.validate_credentials("test24", "test-pass")["ssid"], "test24")
        self.assertEqual(wireless.validate_credentials("open-network", "")["password"], "")
        self.assertEqual(len(wireless.validate_credentials("test24", "a" * 64)["password"]), 64)
        for name, password in (("中" * 11, "test-pass"), ("test", "short"), ("test", "z" * 64), ("a\n", "test-pass")):
            with self.subTest(name=name), self.assertRaises(wireless.SetupError) as error:
                wireless.validate_credentials(name, password)
            self.assertNotIn(password, str(error.exception))

    def test_wifi_receiver_uses_http_and_never_opens_a_serial_port(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read.return_value = JPEG
        opener = Mock(return_value=response)
        relay = wireless.WifiRelay("http://10.0.0.9/", opener=opener)
        with patch.object(wireless.serial, "Serial", side_effect=AssertionError("USB used")):
            relay.start()
            try:
                with relay.condition:
                    self.assertTrue(relay.condition.wait_for(lambda: relay.latest is not None, timeout=2))
                self.assertEqual(relay.frame()[1], JPEG)
                self.assertTrue(relay.status()["connected"])
            finally:
                relay.stop()
        self.assertEqual(opener.call_args.args, ("http://10.0.0.9/capture",))

    def test_wireless_failure_invalidates_old_frame_and_recovers(self):
        entered, release = threading.Event(), threading.Event()
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read.return_value = JPEG
        count = 0

        def opener(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                return response
            if count == 2:
                raise OSError("network disconnected")
            entered.set()
            release.wait(2)
            return response

        relay = wireless.WifiRelay("http://10.0.0.9/", opener=opener, retry_s=.01, frame_interval_s=.01)
        relay.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertIsNone(relay.frame())
            self.assertFalse(relay.status()["connected"])
            release.set()
            with relay.condition:
                self.assertTrue(relay.condition.wait_for(lambda: relay.sequence >= 2, timeout=2))
            self.assertTrue(relay.status()["connected"])
        finally:
            release.set()
            relay.stop()

    def manager(self, path):
        port = Mock()
        port.__enter__ = Mock(return_value=port)
        port.__exit__ = Mock(return_value=None)
        port.write.side_effect = len
        factory = Mock(return_value=port)
        manager = wireless.WirelessPreview("/dev/test-esp", path, serial_factory=factory)
        manager.stopping = QuickEvent()
        return manager, port, factory

    def test_setup_confirms_firmware_then_closes_serial_before_wireless_capture(self):
        secret = "not-a-real-password"
        with tempfile.TemporaryDirectory() as folder:
            state_path = Path(folder) / "wifi.local.json"
            manager, port, factory = self.manager(state_path)
            relay = types.SimpleNamespace(base_url="http://10.0.0.9/", latest=JPEG, condition=threading.Condition(), start=Mock(), stop=Mock(), status=lambda: {"connected": True})
            def create(url):
                port.__exit__.assert_called_once()
                return relay
            states = [{"firmware": wireless.FIRMWARE_VERSION}, {"connected": True, "pending": False, "ip": "10.0.0.9", "error": ""}]
            credentials = {"ssid": "test24", "password": secret}
            with patch.object(wireless, "query_network", side_effect=states), patch.object(wireless, "WifiRelay", side_effect=create):
                manager._configure(credentials)
            self.assertEqual(manager.config_status()["phase"], "ready")
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(json.loads(port.write.call_args.args[0][1:]), {"ssid": "test24", "password": secret})
            self.assertEqual(credentials, {})
            stored = state_path.read_text()
            self.assertNotIn(secret, stored)
            self.assertNotIn("password", stored)
            self.assertEqual(json.loads(stored)["esp_url"], "http://10.0.0.9/")
            if os.name == "posix":
                self.assertEqual(state_path.stat().st_mode & 0o777, 0o600)

    def test_redirect_mode_does_not_capture_unused_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            relay=Mock(base_url='http://10.0.0.9/')
            with patch.object(wireless,'WifiRelay',return_value=relay):
                manager=wireless.WirelessPreview('/dev/test-esp',Path(folder)/'state.json',esp_url='http://10.0.0.9/',preview_enabled=False)
            self.assertEqual(manager.config_status()['phase'],'ready')
            relay.start.assert_not_called()
            manager.stop();relay.stop.assert_called_once()

    def test_redirect_setup_only_captures_during_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            manager,port,factory=self.manager(Path(folder)/'state.json');manager.preview_enabled=False
            relay=types.SimpleNamespace(base_url='http://10.0.0.9/',latest=JPEG,condition=threading.Condition(),start=Mock(),stop=Mock(),status=lambda:{'connected':True})
            states=[{'firmware':wireless.FIRMWARE_VERSION},{'connected':True,'pending':False,'ip':'10.0.0.9','error':''}]
            with patch.object(wireless,'query_network',side_effect=states),patch.object(wireless,'WifiRelay',return_value=relay):
                manager._configure({'ssid':'test24','password':'not-a-real-password'})
            self.assertEqual(manager.config_status()['phase'],'ready')
            relay.start.assert_called_once();relay.stop.assert_called_once()

    def test_unrecognized_firmware_never_receives_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            manager, port, _ = self.manager(Path(folder) / "state.json")
            credentials = {"ssid": "test24", "password": "not-a-real-password"}
            with patch.object(wireless, "query_network", side_effect=wireless.SetupError("固件未更新")):
                manager._configure(credentials)
            port.write.assert_not_called()
            self.assertEqual(manager.config_status()["phase"], "error")
            self.assertEqual(credentials, {})
            self.assertFalse(manager.state_path.exists())

    def test_saved_wireless_address_does_not_open_serial_on_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wifi.local.json"
            path.write_text(json.dumps({"ssid": "test24", "esp_url": "http://10.0.0.9/"}))
            factory = Mock(side_effect=AssertionError("USB used after setup"))
            relay = Mock(base_url="http://10.0.0.9/")
            with patch.object(wireless, "WifiRelay", return_value=relay):
                manager = wireless.WirelessPreview(None, path, serial_factory=factory)
            self.assertEqual(manager.config_status()["phase"], "ready")
            relay.start.assert_called_once()
            factory.assert_not_called()

    def test_wifi_settings_require_token_and_same_origin(self):
        manager = Mock()
        handler_type = wireless.make_wireless_handler(manager, "http://maix/", "known-token")
        for token, origin, expected in (("wrong", "http://127.0.0.1:8123", 403), ("known-token", "http://elsewhere", 403), ("known-token", "http://127.0.0.1:8123", 202)):
            with self.subTest(token=token, origin=origin):
                handler = object.__new__(handler_type)
                body = json.dumps({"ssid": "test24", "password": "test-pass"}).encode()
                handler.path = "/setup"
                handler.headers = {"Origin": origin, "Host": "127.0.0.1:8123", "X-Setup-Token": token, "Content-Length": str(len(body))}
                handler.connection = Mock()
                handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
                handler.send_response = Mock()
                handler.send_header = Mock()
                handler.end_headers = Mock()
                manager.reset_mock()
                handler.do_POST()
                self.assertEqual(handler.send_response.call_args.args[0], expected)
                if expected == 403:
                    manager.configure.assert_not_called()
                else:
                    manager.configure.assert_called_once_with("test24", "test-pass")


if __name__ == "__main__":
    unittest.main()
