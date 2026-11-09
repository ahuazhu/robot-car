from pathlib import Path
import cv2
import time, json

from robot import APP_VERSION
from robot.control.requests import ControlRequest

LIVE_HTML = Path(__file__).with_name("preview.html").read_text(encoding="utf-8").replace(
    "__APP_VERSION__", APP_VERSION
)

CONTROL_FRESH_SECONDS = 1.0


class WebControl:
    """Queue commands for the motor loop; STOP invalidates old START requests."""

    def __init__(self):
        import threading

        self.lock = threading.Lock()
        self.generation = 0
        self.pending = None
        self.updated = 0
        self.state = dict(active=False, ready=False, reason="INITIALIZING", left=0, right=0)

    def update(self, active, ready, reason, pwm, readiness=None):
        with self.lock:
            self.updated = time.monotonic()
            self.state = dict(
                active=active,
                ready=ready,
                reason=reason,
                left=pwm[0],
                right=pwm[1],
                readiness=readiness or {},
            )

    def status(self):
        with self.lock:
            return dict(
                self.state,
                generation=self.generation,
                age_ms=int((time.monotonic() - self.updated) * 1000),
            )

    def submit(self, action, generation=None, *, laps=None):
        with self.lock:
            if action == "stop":
                self.generation += 1
                self.pending = (ControlRequest("stop"), time.monotonic())
                return True, "停车请求已接收"
            try:
                request = ControlRequest(action, laps)
            except ValueError as exc:
                return False, str(exc)
            if generation != self.generation:
                return False, "状态已变化，请重试"
            if self.pending is not None:
                return False, "上一条操作尚未完成"
            readiness = self.state.get("readiness", {})
            if (
                readiness.get("start_light", {}).get("armed")
                and laps is not None
                and laps != readiness.get("laps", {}).get("total_laps")
            ):
                return False, "已ARM，修改圈数请先STOP"
            ready = (
                readiness.get("direct_start_ready", self.state["ready"])
                if action == "start_now"
                else readiness.get("arm_start_light_ready", self.state["ready"])
            )
            if (
                time.monotonic() - self.updated >= CONTROL_FRESH_SECONDS
                or not ready
                or self.state["active"]
            ):
                return False, "当前不能启动：请检查道路识别和车辆状态"
            self.pending = (request, time.monotonic())
            return True, "启动请求已接收"

    def take(self):
        with self.lock:
            pending, self.pending = self.pending, None
            if pending is None:
                return None
            request, created = pending
            if (
                request.action in ("start", "start_now")
                and time.monotonic() - created >= CONTROL_FRESH_SECONDS
            ):
                print(
                    "WEB START EXPIRED",
                    json.dumps(dict(age_ms=round((time.monotonic() - created) * 1000))),
                    flush=True,
                )
                return None
            return request

    def stopped(self):
        with self.lock:
            self.generation += 1
            self.pending = None


class LiveStream:
    """Latest-frame HTTP preview. Slow viewers never block motor commands."""

    def __init__(self, port=8000):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        import secrets

        self.token = secrets.token_urlsafe(24)
        self.control = WebControl()
        self.condition = threading.Condition()
        self.closed = False
        self.jpeg = None
        self.pending_frame = None
        self.encoder = None
        self.sequence = 0
        self.metrics = {}
        self.preview_error = None
        self.server = self.worker = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_json(self, code, value):
                body = json.dumps(value).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                from urllib.parse import urlsplit

                if self.path != "/control":
                    self.send_json(404, dict(message="不存在的接口"))
                    return
                origin = self.headers.get("Origin")
                if (
                    origin and urlsplit(origin).netloc != self.headers.get("Host")
                ) or not secrets.compare_digest(
                    self.headers.get("X-Control-Token", ""), owner.token
                ):
                    self.send_json(403, dict(message="页面已失效，请刷新后重试"))
                    return
                self.connection.settimeout(2)
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 256:
                        raise ValueError("invalid request size")
                    data = json.loads(self.rfile.read(size))
                    if not isinstance(data, dict):
                        raise ValueError("invalid command")
                except (ValueError, OSError):
                    self.send_json(400, dict(message="无效请求"))
                    return
                ok, message = owner.control.submit(
                    data.get("action"), data.get("generation"), laps=data.get("laps")
                )
                self.send_json(202 if ok else 409, dict(message=message))

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/vision":
                    self.send_json(200, owner.metrics)
                    return
                if path == "/capture":
                    with owner.condition:
                        data = owner.jpeg
                    if data is None:
                        self.send_error(503)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if path == "/status":
                    self.send_json(200, owner.control.status())
                    return
                if path == "/":
                    body = LIVE_HTML.replace("__CONTROL_TOKEN__", owner.token).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path != "/stream":
                    self.send_error(404)
                    return
                self.connection.settimeout(2)
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                try:
                    while not owner.closed:
                        with owner.condition:
                            owner.condition.wait_for(
                                lambda: owner.closed or owner.sequence != last, timeout=1
                            )
                            if owner.closed:
                                break
                            if owner.jpeg is None or owner.sequence == last:
                                continue
                            frame, last = owner.jpeg, owner.sequence
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(frame)).encode()
                            + b"\r\n\r\n"
                            + frame
                            + b"\r\n"
                        )
                        self.wfile.flush()
                except (OSError, TimeoutError):
                    pass

        try:
            self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        except OSError as exc:
            self.preview_error = "HTTP preview unavailable: " + str(exc)
            print("PREVIEW ERROR", self.preview_error, flush=True)
            return
        self.server.daemon_threads = True
        self.encoder = threading.Thread(target=self._encode_latest, daemon=True)
        self.encoder.start()
        self.worker = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        self.worker.start()

    def write(self, bgr, *, owned=False):
        """Queue the latest frame; owned transfers a fresh buffer to the encoder."""
        if self.server is None or self.closed:
            return False
        # Default callers may reuse their image immediately. A newly composed
        # preview can transfer ownership and avoid copying all three panels again.
        try:
            frame = bgr if owned else bgr.copy()
        except (AttributeError, TypeError, ValueError, OSError) as exc:
            self.preview_error = str(exc)
            return False
        with self.condition:
            if self.closed:
                return False
            self.pending_frame = frame
            self.condition.notify_all()
        return True

    def _encode_latest(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending_frame is not None)
                if self.closed:
                    return
                frame = self.pending_frame
                self.pending_frame = None
            try:
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise ValueError("JPEG encode failed")
                encoded = jpeg.tobytes()
            except (cv2.error, ValueError, OSError) as exc:
                message = str(exc)
                with self.condition:
                    if self.closed:
                        return
                    if self.preview_error != message:
                        print("PREVIEW ERROR", message, flush=True)
                    self.preview_error = message
                    self.condition.notify_all()
                continue
            with self.condition:
                if self.closed:
                    return
                self.jpeg = encoded
                self.sequence += 1
                self.preview_error = None
                self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.closed = True
            self.pending_frame = None
            self.condition.notify_all()
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=0.5)
        if self.encoder:
            self.encoder.join(timeout=0.2)
