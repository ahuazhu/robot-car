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
    """Serialize STOP, starts, and an optional save-before-ARM preparation."""

    PREPARATION_TIMEOUT_S = 5.0

    def __init__(self, lamp_signal=None):
        import threading

        self.lock = threading.Lock()
        self.lamp_signal = lamp_signal
        self.start_transition = False
        self.generation = 0
        self.pending = None
        self.preparing = None
        self.preparation = dict(pending=False, phase="IDLE", error=None, request_id=None, generation=None)
        self.updated = 0
        self.state = dict(active=False, ready=False, reason="INITIALIZING", left=0, right=0)

    def update(self, active, ready, reason, pwm, readiness=None):
        with self.lock:
            self.updated = time.monotonic()
            self.start_transition = False
            self.state = dict(active=active, ready=ready, reason=reason, left=pwm[0], right=pwm[1],
                              readiness=readiness or {})
            if self.preparation["phase"] == "ARM_QUEUED":
                armed = self.state["readiness"].get("start_light", {}).get("armed")
                self.preparation = dict(self.preparation, pending=False,
                                        phase="ARMED" if armed or active else "ERROR",
                                        error=None if armed or active else "arm_not_accepted")

    def status(self):
        with self.lock:
            return dict(self.state, generation=self.generation,
                        age_ms=int((time.monotonic()-self.updated)*1000),
                        preparation=dict(self.preparation))

    def _end_preparation(self, phase, error=None):
        self.preparing = None
        self.preparation = dict(self.preparation, pending=phase == "ARM_QUEUED", phase=phase, error=error)

    def submit(self, action, generation=None, *, laps=None, lamp_configuration=None):
        with self.lock:
            if action == "stop":
                self.generation += 1
                if self.preparing is not None or self.preparation["phase"] == "ARM_QUEUED":
                    self._end_preparation("CANCELLED", "stopped")
                self.pending = (ControlRequest("stop"), time.monotonic())
                return True, "停车请求已接收"
            try:
                request = ControlRequest(action, laps)
            except ValueError as exc:
                return False, str(exc)
            if generation != self.generation:
                return False, "状态已变化，请重试"
            if self.pending is not None or self.preparing is not None or self.start_transition:
                return False, "上一条操作尚未完成"
            if self.lamp_signal is not None and self.lamp_signal.configuration_pending:
                return False, "灯面配置正在保存，请稍候"
            if lamp_configuration is not None and request.action != "start":
                return False, "灯面配置请通过ARM准备操作提交"
            readiness = self.state.get("readiness", {})
            if (readiness.get("start_light", {}).get("armed") and laps is not None
                    and laps != readiness.get("laps", {}).get("total_laps")):
                return False, "已ARM，修改圈数请先STOP"
            ready = (readiness.get("direct_start_ready", self.state["ready"])
                     if action == "start_now" else readiness.get("arm_start_light_ready", self.state["ready"]))
            if (time.monotonic()-self.updated >= CONTROL_FRESH_SECONDS
                    or not ready or self.state["active"]):
                return False, "当前不能启动：请检查道路识别和车辆状态"
            if lamp_configuration is not None:
                if self.lamp_signal is None:
                    return False, "发车灯配置不可用"
                try:
                    target = request.laps if request.laps is not None else readiness.get("laps", {}).get("total_laps")
                    if target is None:
                        return False, "目标圈数尚未就绪"
                    request = ControlRequest("start", target)
                    desired = self.lamp_signal.normalize_configuration(lamp_configuration)
                    request_id = self.lamp_signal.queue_configuration(desired)
                except (RuntimeError, ValueError) as exc:
                    return False, "当前不能准备发车：" + str(exc)
                self.preparing = dict(request=request, created=time.monotonic(), generation=self.generation,
                                      request_id=request_id, configuration=desired)
                self.preparation = dict(pending=True, phase="SAVING", error=None, request_id=request_id, generation=self.generation)
                return True, "正在保存灯面，完成后进入等灯状态"
            self.pending = (request, time.monotonic())
            return True, "启动请求已接收"

    def submit_lamp_configuration(self, configuration, generation=None):
        with self.lock:
            if self.lamp_signal is None:
                return False, "发车灯配置不可用", None
            if generation != self.generation:
                return False, "状态已变化，请重试", None
            if (time.monotonic()-self.updated >= CONTROL_FRESH_SECONDS
                    or self.state["active"] or self.start_transition or self.pending is not None
                    or self.preparing is not None
                    or self.state.get("readiness", {}).get("start_light", {}).get("armed")):
                return False, "请先停车并取消ARM，再调整灯面", None
            try:
                request_id = self.lamp_signal.queue_configuration(configuration)
            except RuntimeError as exc:
                return False, "当前不能调整灯面：" + str(exc), None
            return True, "配置已排队", request_id

    def _take_preparation(self):
        job = self.preparing
        now = time.monotonic()
        if job["generation"] != self.generation:
            error = "state_changed"
        elif not 0 <= now-job["created"] < self.PREPARATION_TIMEOUT_S:
            error = "preparation_expired"
        elif now-self.updated >= CONTROL_FRESH_SECONDS:
            error = "status_stale"
        elif self.start_transition:
            error = "start_in_progress"
        elif self.state["active"] or self.state.get("readiness", {}).get("start_light", {}).get("armed"):
            error = "stop_first"
        else:
            error = None
        if error:
            self._end_preparation("ERROR", error)
            return None
        light = self.lamp_signal.snapshot()
        result = light.get("configuration_update") or {}
        if result.get("request_id") != job["request_id"]:
            self._end_preparation("ERROR", "configuration_result_changed")
            return None
        if result.get("pending"):
            return None
        if result.get("state") != "SAVED":
            self._end_preparation("ERROR", result.get("error") or "configuration_failed")
            return None
        applied, desired = light.get("lamp_configuration") or {}, job["configuration"]
        if (applied.get("manual") is not desired["manual"]
                or desired["manual"] and applied.get("roi") != desired["roi"]):
            self._end_preparation("ERROR", "configuration_mismatch")
            return None
        if (not light.get("connected") or light.get("armed") or light.get("permission")
                or not self.lamp_signal.available):
            self._end_preparation("ERROR", "signal_not_ready")
            return None
        request = job["request"]
        self._end_preparation("ARM_QUEUED")
        self.start_transition = True
        return request

    def take_actions(self, touch_action=None):
        """Resolve web preparation and touch actions together under one lock."""
        with self.lock:
            pending, self.pending = self.pending, None
            if touch_action == "stop":
                if self.preparing is not None:
                    self._end_preparation("CANCELLED", "stopped")
                return None, touch_action
            if pending is not None and pending[0].action == "stop":
                return pending[0], touch_action
            if self.preparing is not None:
                # Do not let a second touch start change this job's captured
                # lap target or bypass its configuration completion proof.
                if touch_action in ("start", "start_now"):
                    touch_action = None
                return self._take_preparation(), touch_action
            if touch_action in ("start", "start_now"):
                self.start_transition = True
            if pending is None:
                return None, touch_action
            request, created = pending
            if (request.action in ("start", "start_now")
                    and time.monotonic()-created >= CONTROL_FRESH_SECONDS):
                print("WEB START EXPIRED", json.dumps(dict(age_ms=round((time.monotonic()-created)*1000))),
                      flush=True)
                return None, touch_action
            if request.action in ("start", "start_now"):
                if self.lamp_signal is not None and self.lamp_signal.configuration_pending:
                    return None, touch_action
                self.start_transition = True
            return request, touch_action

    def take(self, *, touch_action=None):
        # Retain the original single-request API for existing callers.
        return self.take_actions(touch_action)[0]

    def stopped(self):
        with self.lock:
            self.generation += 1
            self.pending = None
            if self.preparing is not None or self.preparation["phase"] == "ARM_QUEUED":
                self._end_preparation("CANCELLED", "stopped")

class LiveStream:
    """Latest-frame HTTP preview. Slow viewers never block motor commands."""

    def __init__(self, port=8000, *, lamp_signal=None):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        import secrets

        self.token = secrets.token_urlsafe(24)
        self.control = WebControl(lamp_signal)
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

                if self.path not in ("/control", "/lamp-config"):
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
                if self.path == "/lamp-config":
                    try:
                        ok, message, request_id = owner.control.submit_lamp_configuration(
                            data.get("configuration"), data.get("generation")
                        )
                    except ValueError:
                        self.send_json(400, dict(message="灯面范围无效"))
                        return
                    self.send_json(202 if ok else 409, dict(message=message, request_id=request_id))
                    return
                ok, message = owner.control.submit(
                    data.get("action"), data.get("generation"), laps=data.get("laps"),
                    lamp_configuration=data.get("lamp_configuration"),
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
