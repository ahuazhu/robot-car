"""Local USB JPEG relay. Requires pyserial; never sends UNO/motor commands."""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import serial


MAX_FRAME_AGE_S = 5.0


class CaptureStopped(Exception):
    """Normal interruption of a camera request during service shutdown."""


def capture(link, stopping, clock=time.monotonic):
    """Read exactly one bounded JPEG response, with interruptible read waits."""
    if stopping.is_set():
        raise CaptureStopped()
    if link.write(b"J") != 1:
        raise RuntimeError("Incomplete camera request")
    deadline = clock() + 5.0
    size = None
    while not stopping.is_set() and clock() < deadline:
        line = link.readline(128)
        if line.startswith(b"JPEG "):
            fields = line.split()
            if len(fields) != 2 or not line.endswith(b"\n"):
                raise RuntimeError("Invalid JPEG header")
            size = int(fields[1])
            break
        if line.startswith(b"ERROR"):
            raise RuntimeError(line.decode(errors="replace").strip())
    if stopping.is_set():
        raise CaptureStopped()
    if size is None or not 100 < size < 150000:
        raise RuntimeError("No valid JPEG header")
    data = bytearray()
    deadline = clock() + max(5.0, size / 8000.0)
    while not stopping.is_set() and len(data) < size and clock() < deadline:
        data.extend(link.read(size - len(data)))
    if stopping.is_set():
        raise CaptureStopped()
    data = bytes(data)
    if len(data) != size or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise RuntimeError("Incomplete JPEG")
    return data


class PreviewRelay:
    """One worker owns the serial handle; HTTP readers only see frame snapshots."""

    def __init__(
        self, serial_path, *, serial_factory=None, clock=time.monotonic,
        startup_s=3.0, retry_s=0.5, frame_interval_s=0.1,
    ):
        self.serial_path = serial_path
        self.serial_factory = serial.Serial if serial_factory is None else serial_factory
        self.clock = clock
        self.startup_s = startup_s
        self.retry_s = retry_s
        self.frame_interval_s = frame_interval_s
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.latest = None
        self.sequence = 0
        self.frame_at = None
        self.failure = "Waiting for camera"
        self.worker = None

    def start(self):
        if self.worker is not None:
            raise RuntimeError("Camera worker already started")
        self.worker = threading.Thread(target=self._produce, name="esp-usb-preview", daemon=True)
        self.worker.start()

    def _clear(self, error):
        with self.condition:
            self.latest = None
            self.frame_at = None
            self.failure = error
            self.condition.notify_all()

    def _publish(self, data):
        with self.condition:
            if self.stopping.is_set():
                return
            self.latest = data
            self.sequence += 1
            self.frame_at = self.clock()
            self.failure = ""
            self.condition.notify_all()

    def _status_locked(self):
        age = None if self.frame_at is None else max(0.0, self.clock() - self.frame_at)
        connected = bool(
            self.latest is not None and not self.failure and not self.stopping.is_set()
            and age is not None and age < MAX_FRAME_AGE_S
        )
        error = self.failure
        if not connected and not error:
            error = "Preview stopped" if self.stopping.is_set() else "Camera frame is stale"
        return dict(
            frames=self.sequence, error=error, connected=connected,
            age_ms=None if age is None else round(age * 1000),
        )

    def status(self):
        with self.condition:
            return self._status_locked()

    def frame(self):
        with self.condition:
            if self._status_locked()["connected"]:
                return self.sequence, self.latest
            return None

    def wait_frame(self, after):
        """Return only a newer live frame; disconnect/expiry closes old streams."""
        with self.condition:
            while self._status_locked()["connected"]:
                if self.sequence != after:
                    return self.sequence, self.latest
                self.condition.wait(timeout=0.25)
            return None

    @staticmethod
    def _close_serial(link):
        if link is not None:
            try:
                link.close()
            except Exception:
                # A disconnected device may fail again during close. It must
                # never prevent the worker from opening a replacement handle.
                pass

    def _produce(self):
        link = None
        last_error = None
        try:
            while not self.stopping.is_set():
                try:
                    if link is None:
                        link = self.serial_factory(
                            self.serial_path, 115200, timeout=0.25, write_timeout=2
                        )
                        if self.stopping.wait(self.startup_s):
                            break
                        link.reset_input_buffer()
                    data = capture(link, self.stopping, self.clock)
                    self._publish(data)
                    last_error = None
                except CaptureStopped:
                    break
                except Exception as exc:
                    error = str(exc) or type(exc).__name__
                    self._clear(error)
                    if error != last_error:
                        print("Capture error:", error, flush=True)
                        last_error = error
                    self._close_serial(link)
                    link = None
                    if self.stopping.wait(self.retry_s):
                        break
                    continue
                if self.stopping.wait(self.frame_interval_s):
                    break
        finally:
            self._close_serial(link)
            self._clear("Preview stopped" if self.stopping.is_set() else "Camera worker stopped")

    def stop(self):
        self.stopping.set()
        self._clear("Preview stopped")
        if self.worker is not None:
            self.worker.join(timeout=4.0)
            if self.worker.is_alive():
                raise RuntimeError("Camera worker did not stop within its serial timeout")


def validate_maix_url(value):
    value = value.strip()
    if not value:
        return ""
    if "\\" in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise argparse.ArgumentTypeError("Maix URL contains invalid characters")
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in ("http", "https") and parsed.hostname
            and parsed.username is None and parsed.password is None
        )
        parsed.port  # Reject malformed/out-of-range ports before handing the URL to a browser.
    except ValueError:
        valid = False
    if not valid:
        raise argparse.ArgumentTypeError("Maix URL must be HTTP(S), with a host and no userinfo")
    return value


def load_page(maix_url="", path=None, *, transport=None, setup_url=None):
    values = {"maixUrl": validate_maix_url(maix_url)}
    if transport is not None:
        values["transport"] = transport
    if setup_url is not None:
        values["setupUrl"] = setup_url
    config = json.dumps(values, ensure_ascii=True)
    config = config.replace("<", "\\u003c")
    page = (Path(path) if path is not None else Path(__file__).with_name("preview.html")).read_text(
        encoding="utf-8"
    )
    if "__PREVIEW_CONFIG__" not in page:
        raise ValueError("Preview page is missing its configuration placeholder")
    return page.replace("__PREVIEW_CONFIG__", config)


def make_handler(relay, page):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _respond(self, status, data, kind):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                self._get()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _get(self):
            route = self.path.split("?", 1)[0]
            if route == "/":
                self._respond(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif route == "/status":
                self._respond(200, json.dumps(relay.status()).encode(), "application/json")
            elif route in ("/capture", "/stream"):
                packet = relay.frame()
                if packet is None:
                    self._respond(503, b"Camera is not live", "text/plain; charset=utf-8")
                    return
                if route == "/capture":
                    self._respond(200, packet[1], "image/jpeg")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while packet is not None:
                    sequence, data = packet
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                    )
                    self.wfile.flush()
                    packet = relay.wait_frame(sequence)
            else:
                self._respond(404, b"Not found", "text/plain; charset=utf-8")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--maix-url", type=validate_maix_url, default="")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    page = load_page(args.maix_url)
    relay = PreviewRelay(args.serial)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(relay, page))
    try:
        relay.start()
        print("ESP32 USB preview http://127.0.0.1:%d/" % args.port, flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            relay.stop()
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
