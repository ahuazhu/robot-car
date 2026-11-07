"""Local dual-camera page using ESP32 Wi-Fi; USB is used only for first setup."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import secrets
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

import serial

if __package__:
    from .usb_preview import PreviewRelay, load_page, make_handler, validate_maix_url
else:
    from usb_preview import PreviewRelay, load_page, make_handler, validate_maix_url


FIRMWARE_VERSION = "0.4.6-larger-roi"


class SetupError(Exception):
    """Only intentional, credential-free messages may be shown to the browser."""


def validate_credentials(ssid, password):
    if not isinstance(ssid, str) or not isinstance(password, str):
        raise SetupError("请输入 Wi-Fi 名称和密码。")
    if not 1 <= len(ssid.encode("utf-8")) <= 32:
        raise SetupError("Wi-Fi 名称应为 1–32 字节。")
    if any(ord(c) < 32 for c in ssid + password):
        raise SetupError("名称或密码中包含不支持的控制字符。")
    size = len(password.encode("utf-8"))
    if size and not (8 <= size <= 63 or size == 64 and all(c in "0123456789abcdefABCDEF" for c in password)):
        raise SetupError("Wi-Fi 密码需为 8–63 字节；开放网络可留空。")
    return {"ssid": ssid, "password": password}


def query_network(link, stopping, timeout=3.0):
    link.write(b"Q")
    deadline = time.monotonic() + timeout
    while not stopping.is_set() and time.monotonic() < deadline:
        line = link.readline(1024)
        if line.startswith(b"WIFI "):
            try:
                state = json.loads(line[5:])
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(state, dict) and state.get("firmware") == FIRMWARE_VERSION:
                return state
    raise SetupError("摄像头未响应无线设置，请检查 USB 或固件。")


class WifiRelay(PreviewRelay):
    """Latest-frame relay sourced only from HTTP, with no serial dependency."""

    def __init__(self, base_url, *, opener=None, **kwargs):
        super().__init__("", **kwargs)
        self.base_url = validate_maix_url(base_url).rstrip("/") + "/"
        self.opener = build_opener(ProxyHandler({})).open if opener is None else opener

    def _produce(self):
        try:
            while not self.stopping.is_set():
                try:
                    with self.opener(self.base_url + "capture", timeout=2.0) as response:
                        data = response.read(150001)
                    if not 100 < len(data) < 150000 or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
                        raise ValueError("Invalid wireless JPEG")
                    self._publish(data)
                except Exception:
                    self._clear("Wi-Fi camera is unavailable")
                    if self.stopping.wait(self.retry_s):
                        break
                if self.stopping.wait(self.frame_interval_s):
                    break
        finally:
            self._clear("Preview stopped")


class WirelessPreview:
    def __init__(self, serial_path, state_path, *, ssid="", esp_url="", serial_factory=serial.Serial, preview_enabled=True):
        self.serial_path = serial_path
        self.state_path = Path(state_path)
        self.ssid = ssid
        self.serial_factory = serial_factory
        self.preview_enabled = preview_enabled
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.relay = None
        self.job = None
        self.phase = "setup"
        self.message = "请填写 2.4GHz Wi-Fi 密码。USB 仅在首次设置时使用。"
        self.esp_url = ""
        if self.state_path.exists():
            try:
                previous = json.loads(self.state_path.read_text(encoding="utf-8"))
                if not isinstance(previous, dict):
                    previous = {}
            except (OSError, ValueError):
                previous = {}
            self.ssid = ssid or previous.get("ssid", "")
            esp_url = esp_url or previous.get("esp_url", "")
        if esp_url:
            self._use_wifi(esp_url)
            self.phase = "ready"

    def _use_wifi(self, url):
        new = WifiRelay(url)
        with self.lock:
            old = self.relay
            self.relay = new
            self.esp_url = new.base_url
        if old is not None:
            old.stop()
        if self.preview_enabled:
            new.start()
        return new

    def config_status(self):
        with self.lock:
            return dict(phase=self.phase, message=self.message, esp_url=self.esp_url)

    def status(self):
        with self.lock:
            relay = self.relay
        return relay.status() if relay else dict(frames=0, error="Wi-Fi setup required", connected=False, age_ms=None)

    def frame(self):
        with self.lock:
            relay = self.relay
        return relay.frame() if relay else None

    def wait_frame(self, after):
        with self.lock:
            relay = self.relay
        return relay.wait_frame(after) if relay else None

    def configure(self, ssid, password):
        credentials = validate_credentials(ssid, password)
        if not self.serial_path:
            raise SetupError("首次设置请通过 USB 连接 ESP32。")
        with self.lock:
            if self.job and self.job.is_alive():
                raise SetupError("正在连接 Wi-Fi，请稍候。")
            self.phase, self.message = "connecting", "正在把 Wi-Fi 设置发送到摄像头…"
            self.job = threading.Thread(target=self._configure, args=(credentials,), daemon=True)
            self.job.start()

    def _configure(self, credentials):
        verification_relay = None
        try:
            with self.serial_factory(self.serial_path, 115200, timeout=0.25, write_timeout=2, exclusive=True) as link:
                if self.stopping.wait(3.0):
                    return
                link.reset_input_buffer()
                query_network(link, self.stopping)  # Confirm firmware before sending credentials.
                packet = b"C" + json.dumps(credentials, ensure_ascii=False).encode("utf-8") + b"\n"
                if len(packet) > 760:
                    raise SetupError("Wi-Fi 设置过长。")
                if link.write(packet) != len(packet):
                    raise SetupError("USB 传输未完成，请重试。")
                deadline = time.monotonic() + 35.0
                while not self.stopping.is_set() and time.monotonic() < deadline:
                    if self.stopping.wait(0.5):
                        return
                    state = query_network(link, self.stopping)
                    if state.get("connected") and not state.get("pending") and not state.get("error"):
                        try:
                            ip = ipaddress.IPv4Address(state["ip"])
                        except (KeyError, ValueError):
                            raise SetupError("摄像头返回了无效的网络地址。") from None
                        if ip.is_loopback or ip.is_unspecified or ip.is_multicast:
                            raise SetupError("摄像头网络地址不可用。")
                        esp_url = "http://%s/" % ip
                        break
                    if state.get("error") and not state.get("pending"):
                        raise SetupError("连接 Wi-Fi 失败，请检查密码和 2.4GHz 网络。")
                else:
                    raise SetupError("连接 Wi-Fi 超时，请检查热点是否已开启。")
            # From this point onward the serial handle is CLOSED. Images use HTTP only.
            if self.stopping.is_set():
                return
            self.ssid = credentials["ssid"]
            credentials.clear()
            record = {"ssid": self.ssid, "esp_url": esp_url}
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            staged = self.state_path.with_name(self.state_path.name + ".tmp")
            with os.fdopen(os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as out:
                json.dump(record, out, ensure_ascii=False)
            staged.chmod(0o600)
            os.replace(staged, self.state_path)
            relay = self._use_wifi(esp_url)
            if not self.preview_enabled:
                verification_relay = relay
                relay.start()
            with relay.condition:
                relay.condition.wait_for(lambda: relay.latest is not None or self.stopping.is_set(), timeout=7.0)
            if not relay.status()["connected"]:
                raise SetupError("摄像头已入网，但图像尚不可达，请检查网络隔离设置后重试。")
            with self.lock:
                self.phase, self.message = "ready", "无线画面已连接，可以移除与电脑之间的 USB 数据线。"
        except SetupError as exc:
            with self.lock:
                self.phase, self.message = "error", str(exc)
        except Exception:
            with self.lock:
                self.phase, self.message = "error", "设置失败，请检查 USB 连接后重试。"
        finally:
            if verification_relay is not None:
                verification_relay.stop()
            credentials.clear()

    def stop(self):
        self.stopping.set()
        if self.job:
            self.job.join(timeout=4.0)
        if self.relay:
            self.relay.stop()


def make_wireless_handler(manager, maix_url, token):
    page = load_page(maix_url, transport="Wi-Fi", setup_url="/setup")
    parent = make_handler(manager, page)

    class Handler(parent):
        def _get(self):
            path = self.path.split("?", 1)[0]
            if path == "/setup" or path == "/" and manager.config_status()["phase"] != "ready":
                config = json.dumps({"ssid": manager.ssid, "token": token}, ensure_ascii=True).replace("<", "\\u003c")
                content = Path(__file__).with_name("wifi_setup.html").read_text(encoding="utf-8")
                self._respond(200, content.replace("__SETUP_CONFIG__", config).encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/" and maix_url:
                self.send_response(302)
                self.send_header("Location", maix_url)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif path == "/setup/status":
                self._respond(200, json.dumps(manager.config_status(), ensure_ascii=False).encode("utf-8"), "application/json")
            else:
                super()._get()

        def do_POST(self):
            origin = self.headers.get("Origin")
            if self.path != "/setup" or not secrets.compare_digest(self.headers.get("X-Setup-Token", ""), token) or (origin and urlsplit(origin).netloc != self.headers.get("Host")):
                self._respond(403, b"Forbidden", "text/plain")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024:
                    raise ValueError()
                self.connection.settimeout(2.0)
                data = json.loads(self.rfile.read(size))
                manager.configure(data.get("ssid"), data.get("password"))
            except SetupError as exc:
                self._respond(400, json.dumps({"message": str(exc)}, ensure_ascii=False).encode(), "application/json")
                return
            except (ValueError, TypeError, AttributeError, OSError):
                self._respond(400, b'{"message":"Invalid setup request"}', "application/json")
                return
            self._respond(202, b'{"accepted":true}', "application/json")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial")
    parser.add_argument("--maix-url", default="", type=validate_maix_url)
    parser.add_argument("--esp-url", default="", type=validate_maix_url)
    parser.add_argument("--ssid", default="")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--state-file", type=Path, default=Path(".maixpy/esp32_wifi.local.json"))
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    manager = WirelessPreview(args.serial, args.state_file, ssid=args.ssid, esp_url=args.esp_url, preview_enabled=not bool(args.maix_url))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_wireless_handler(manager, args.maix_url, secrets.token_urlsafe(32)))
    print("Wireless camera setup/preview http://127.0.0.1:%d/" % args.port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()
        server.server_close()


if __name__ == "__main__":
    main()
