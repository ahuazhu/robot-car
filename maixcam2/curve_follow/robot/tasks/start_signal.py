"""Receive a fresh, authenticated ESP32 event; no Maix image processing.

All network I/O runs in one bounded worker. STOP invalidates the local session
synchronously, so a late response cannot start the car. Before permission, stale or mismatched data cancels ARM. Once a matching OFF
event is accepted, permission is held locally until explicit mission reset;
ESP32 health is then telemetry only and cannot revoke it.
"""
import json
import math
import os
import socket
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import HTTPError


class StartSignal:
    MAX_AGE_S = 0.75
    MIN_CONFIRM_FRAMES = 3
    MIN_CONFIRM_MS = 400
    SEARCH_ROIS = ((96, 8, 128, 112), (80, 0, 160, 128), (0, 0, 320, 240))

    def __init__(self, config=None, *, app_dir=None, clock=time.monotonic, transport=None):
        c = config or {}
        self.enabled = bool(c.get("enabled", False))
        self.clock = clock
        self.url = c.get("peer_url", "http://robot-esp32cam.local/").rstrip("/")
        self.endpoint = None
        self.last_error_kind = ""
        self.last_error_reason = ""
        self.last_request_ms = 0.0
        self.last_endpoint = None
        self.key = ""
        if app_dir is not None:
            try:
                local = json.loads((Path(app_dir) / "start_signal.local.json").read_text())
                self.url = local["peer_url"].rstrip("/")
                self.key = local["key"]
            except (OSError, ValueError, KeyError, TypeError):
                pass
        parts = urlsplit(self.url)
        if parts.scheme != "http" or not parts.hostname or parts.username or parts.password or parts.path or parts.query or parts.fragment:
            raise ValueError("Invalid ESP32 signal peer URL")
        if not isinstance(self.key, str) or len(self.key) != 64 or any(c not in "0123456789abcdef" for c in self.key):
            self.key = ""
        self.transport = transport or self._http
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.thread = None
        self.armed = self.fired = self.permission = False
        self.initial_lamp_state = "OFF"
        self.phase = "DISARMED"
        self.error = "PAIRING_REQUIRED" if not self.key else "CONNECTING"
        self.session = self.boot = ""
        self.peer = {}
        self.received = None
        self.progress_at = None
        self.acknowledged = False
        self.cancel_queue = []
        self.reset_remote_pending = True
        self.confirmation = {}
        self.fault_reason = self.fault_kind = ""
        self.configuration_sequence = 0
        self.configuration_request = None
        self.configuration_update = dict(pending=False, state="IDLE", error=None, request_id=0)

    CONFIGURATION_ERRORS = frozenset((
        "invalid_configuration", "invalid_request", "pairing_required", "boot_changed",
        "native_armed_stop_first", "storage_failed", "configuration_mismatch",
        "configuration_response_invalid", "configuration_failed", "stop_first", "not_ready",
    ))

    @staticmethod
    def normalize_configuration(configuration):
        if not isinstance(configuration, dict) or type(configuration.get("manual")) is not bool:
            raise ValueError("invalid_configuration")
        if set(configuration) - {"manual", "roi"}:
            raise ValueError("invalid_configuration")
        if not configuration["manual"] and "roi" not in configuration:
            return {"manual": False}
        roi = configuration.get("roi")
        if (not isinstance(roi, list) or len(roi) != 4
                or any(type(value) is not int for value in roi)):
            raise ValueError("invalid_configuration")
        x, y, width, height = roi
        if x < 0 or y < 0 or width < 16 or height < 16 or x+width > 320 or y+height > 240:
            raise ValueError("invalid_configuration")
        return dict(manual=True, roi=list(roi)) if configuration["manual"] else {"manual": False}

    @staticmethod
    def _lamp_configuration(peer):
        configuration = peer.get("lamp_configuration")
        if not isinstance(configuration, dict):
            return None
        try:
            normalized = StartSignal.normalize_configuration(dict(configuration, manual=True))
        except ValueError:
            return None
        if type(configuration.get("manual")) is not bool:
            return None
        return dict(manual=configuration["manual"], roi=normalized["roi"])

    def _candidate_hint(self):
        """Forward a fresh locator hint, never formal signal/permission evidence."""
        hint = self.peer.get("candidate_hint")
        if (not isinstance(hint, dict) or hint.get("valid") is not True
                or type(hint.get("age_ms")) is not int or not 0 <= hint["age_ms"] <= 2500
                or self.received is None):
            return None
        elapsed = self.clock() - self.received
        if not math.isfinite(elapsed) or elapsed < 0:
            return None
        age = hint["age_ms"] + math.ceil(elapsed * 1000)
        if age > 2500:
            return None
        try:
            roi = self.normalize_configuration(dict(manual=True, roi=hint.get("roi")))["roi"]
        except ValueError:
            return None
        return dict(valid=True, roi=roi, age_ms=age)

    @property
    def configuration_pending(self):
        with self.lock:
            return self.configuration_request is not None

    def queue_configuration(self, configuration):
        """Reserve one edit locally; only the existing ESP worker performs I/O."""
        configuration = self.normalize_configuration(configuration)
        with self.lock:
            if self.configuration_request is not None:
                raise RuntimeError("configuration_pending")
            if self.armed or self.permission or self.phase == "RUNNING" or self.peer.get("armed"):
                raise RuntimeError("stop_first")
            if (not self.enabled or not self.key or not self._fresh() or self.error
                    or self.peer.get("paired") is not True or self.cancel_queue
                    or self.reset_remote_pending):
                raise RuntimeError("not_ready")
            if self._lamp_configuration(self.peer) is None:
                raise RuntimeError("configuration_unsupported")
            self.configuration_sequence += 1
            request = dict(request_id=self.configuration_sequence, boot=self.peer["boot"],
                           configuration=configuration)
            self.configuration_request = request
            self.configuration_update = dict(pending=True, state="QUEUED", error=None,
                                             request_id=request["request_id"])
            return request["request_id"]

    def _configure_step(self):
        """Recheck the reservation before I/O, and never turn a save into ARM."""
        with self.lock:
            request = self.configuration_request
            if request is None:
                return
            if self.armed or self.permission or self.phase == "RUNNING" or self.peer.get("armed"):
                refusal = "stop_first"
            elif (not self._fresh() or self.error or self.peer.get("paired") is not True
                  or self.peer.get("boot") != request["boot"]):
                refusal = "not_ready"
            else:
                refusal = None
            if refusal:
                self.configuration_request = None
                self.configuration_update = dict(pending=False, state="ERROR", error=refusal,
                                                 request_id=request["request_id"])
                return
            self.configuration_update = dict(self.configuration_update, state="SENDING")
        try:
            peer = self.transport("configure", "", request["boot"],
                                  configuration=request["configuration"])
            if not self._valid(peer) or self._lamp_configuration(peer) is None:
                raise ValueError("configuration_response_invalid")
            if peer["boot"] != request["boot"]:
                raise ValueError("boot_changed")
            if peer["armed"]:
                raise ValueError("native_armed_stop_first")
            applied = self._lamp_configuration(peer)
            expected = request["configuration"]
            if (applied["manual"] != expected["manual"]
                    or expected["manual"] and applied["roi"] != expected["roi"]):
                raise ValueError("configuration_mismatch")
            with self.lock:
                self.peer, self.received, self.error = peer, self.clock(), ""
                self.configuration_request = None
                self.configuration_update = dict(pending=False, state="SAVED", error=None,
                                                 request_id=request["request_id"])
        except Exception as exc:
            error = str(exc) if str(exc) in self.CONFIGURATION_ERRORS else (
                "configuration_offline" if isinstance(exc, OSError) else "configuration_failed")
            with self.lock:
                # A timeout may follow a successful device save. Require a
                # fresh status before ARM; never guess which ROI is installed.
                self.received = None
                self.error = "ESP32_CONFIGURATION_UNCONFIRMED"
                if isinstance(exc, OSError):
                    self.endpoint = None
                self.configuration_request = None
                self.configuration_update = dict(pending=False, state="ERROR", error=error,
                                                 request_id=request["request_id"])

    def start(self):
        if self.enabled and self.key and self.thread is None:
            self.thread = threading.Thread(target=self._worker, name="esp32-start-signal", daemon=True)
            self.thread.start()

    def _fresh(self):
        return self.received is not None and 0 <= self.clock()-self.received <= self.MAX_AGE_S

    @property
    def available(self):
        with self.lock:
            if self.configuration_request is not None:
                return False
            if self.permission:
                return not self.armed
            return self._fresh() and not self.error and self.peer.get("paired") is True and not self.peer.get("armed") and not self.cancel_queue and not self.reset_remote_pending

    @staticmethod
    def normalize_initial_lamp_state(value):
        if type(value) is not str or value not in ("OFF", "ON"):
            raise ValueError("invalid_initial_lamp_state")
        return value

    def arm(self, initial_lamp_state="OFF"):
        initial_lamp_state = self.normalize_initial_lamp_state(initial_lamp_state)
        with self.lock:
            if self.configuration_request is not None:
                return False
            if self.armed:
                return initial_lamp_state == self.initial_lamp_state
            if self.permission:
                if initial_lamp_state != self.initial_lamp_state:
                    return False
                self.armed, self.fired, self.phase = True, False, "SIGNAL_READY"
                return True
            if not self.enabled or not self._fresh() or self.error or self.peer.get("armed") or not self.key or self.peer.get("paired") is not True or self.cancel_queue or self.reset_remote_pending:
                return False
            self.session = os.urandom(16).hex()
            self.boot = self.peer["boot"]
            self.initial_lamp_state = initial_lamp_state
            self.armed, self.fired, self.acknowledged = True, False, False
            self.phase = "ARMING"
            self.confirmation = {}
            self.fault_reason = self.fault_kind = ""
            return True

    def _release_peer(self):
        if self.session:
            item = (self.session, self.boot)
            if item not in self.cancel_queue:
                self.cancel_queue.append(item)
        self.session = ""
        self.acknowledged = False

    def cancel(self, *, clear_permission=True):
        with self.lock:
            self._release_peer()
            self.armed = False
            if clear_permission:
                self.permission = self.fired = False
                self.confirmation = {}
                self.reset_remote_pending = True
            self.phase = "PERMITTED" if self.permission else "DISARMED"
            self.fault_reason = self.fault_kind = ""

    def _grant(self, peer):
        self.permission = True
        self.confirmation = {k:v for k,v in peer.items() if k not in ("session", "boot")}
        self.phase = "SIGNAL_READY"
        self._release_peer()

    def started(self):
        with self.lock:
            self._release_peer()
            self.armed, self.fired, self.phase = False, True, "RUNNING"

    def consume(self):
        """Use the local permission once per ARM; ESP32 is no longer a gate."""
        with self.lock:
            if not self.permission or not self.armed:
                return False
            self._release_peer()
            self.armed, self.fired, self.phase = False, True, "RUNNING"
            return True

    def update(self, now, *, active=False, timely=True):
        with self.lock:
            if active or not self.armed:
                return False
            if self.permission:
                return bool(timely)
            if not self._fresh():
                self.error = "ESP32_WAITING_FOR_RECOVERY"
                self.phase = "WAIT_CONNECTION"
            return False

    def _http(self, op, session, boot, *, configuration=None, initial_lamp_state="OFF"):
        opener = build_opener(ProxyHandler({}))
        # Resolve only once per healthy connection. Some mDNS stacks retry AAAA
        # for several seconds per request; IPv4 pinning also keeps UI video fast.
        if self.endpoint is None:
            parts = urlsplit(self.url)
            self.endpoint = "http://%s:%d" % (socket.gethostbyname(parts.hostname), parts.port or 81)
        if op == "status":
            request = Request(self.endpoint+"/status")
        else:
            payload = dict(op=op, session=session, boot=boot)
            if op == "arm":
                payload["initial_lamp_state"] = self.normalize_initial_lamp_state(initial_lamp_state)
            if op == "configure":
                payload = dict(op=op, boot=boot, configuration=configuration)
            body = json.dumps(payload).encode()
            request = Request(self.endpoint+"/signal", data=body, headers={"X-Signal-Key": self.key, "Content-Type": "application/json"})
        try:
            with opener.open(request, timeout=0.3) as response:
                data = response.read(4097)
        except HTTPError as exc:
            if op != "configure":
                raise
            try:
                error = json.loads(exc.read(4097)).get("error")
            except (ValueError, AttributeError):
                error = None
            raise ValueError(error if error in self.CONFIGURATION_ERRORS else "configuration_failed") from exc
        if len(data)>4096:
            raise ValueError("Oversized ESP32 response")
        data = json.loads(data)
        return data["signal"] if op == "status" else data

    @staticmethod
    def _search_roi_valid(peer):
        roi = peer.get("roi")
        return (isinstance(roi, list) and len(roi) == 4
                and all(type(n) is int for n in roi) and tuple(roi) in StartSignal.SEARCH_ROIS)

    @staticmethod
    def _target_valid(peer):
        if peer.get("target_locked") is not True:
            return True
        if not StartSignal._search_roi_valid(peer):
            return False
        x, y, width, height = peer["roi"]
        a = peer.get("target_roi")
        return (isinstance(a, list) and len(a) == 4 and all(type(n) is int for n in a)
                and a[0] >= x and a[1] >= y and a[2] > 0 and a[3] > 0
                and a[0]+a[2] <= x+width and a[1]+a[3] <= y+height)

    @staticmethod
    def _valid(peer):
        if not isinstance(peer, dict):
            return False
        frames, duration = peer.get("required_frames"), peer.get("required_ms")
        return (isinstance(peer.get("boot"), str) and len(peer["boot"]) == 16
                and all(c in "0123456789abcdef" for c in peer["boot"])
                and type(peer.get("sequence")) is int and peer["sequence"] > 0
                and type(peer.get("frame_age_ms")) is int and 0 <= peer["frame_age_ms"] <= 400
                and type(peer.get("armed")) is bool and type(peer.get("confirmed")) is bool
                and type(peer.get("paired")) is bool and StartSignal._search_roi_valid(peer)
                and peer.get("protocol") == 2 and peer.get("control_port") == 81
                and peer.get("trigger_direction") == "DARKER"
                and type(peer.get("target_locked")) is bool and StartSignal._target_valid(peer)
                and type(peer.get("event_age_ms")) is int and peer["event_age_ms"] >= 0
                and type(frames) is int and frames >= StartSignal.MIN_CONFIRM_FRAMES
                and type(duration) is int and duration >= StartSignal.MIN_CONFIRM_MS
                and (not peer["confirmed"] or peer["target_locked"] is True
                     and type(peer.get("confirmation_frames")) is int and peer["confirmation_frames"] >= frames
                     and type(peer.get("confirmed_ms")) is int and peer["confirmed_ms"] >= duration))

    @staticmethod
    def _remote_session(peer):
        session = peer.get("session")
        return isinstance(session,str) and len(session)==32 and all(c in "0123456789abcdef" for c in session)

    def _step(self):
        with self.lock:
            configuring = self.configuration_request is not None and not self.cancel_queue
        if configuring:
            self._configure_step()
            return
        with self.lock:
            cancel = self.cancel_queue.pop(0) if self.cancel_queue else None
            session, boot = self.session, self.boot
            initial_lamp_state = self.initial_lamp_state
            op = "cancel" if cancel else "poll" if session and self.acknowledged else "arm" if session else "status"
        target_session, target_boot = cancel or (session, boot)
        began = self.clock()
        try:
            # Keep existing OFF-only transport adapters compatible. Explicit
            # ON requires the new request and a matching device acknowledgement.
            peer = (self.transport(op, target_session, target_boot, initial_lamp_state=initial_lamp_state)
                    if op == "arm" and initial_lamp_state == "ON"
                    else self.transport(op, target_session, target_boot))
            self.last_request_ms = round((self.clock()-began)*1000,1)
            if self.clock()-began > .6:
                raise ValueError("RESPONSE_TOO_SLOW")
            if not self._valid(peer):
                if isinstance(peer,dict) and peer.get("frame_age_ms",999999)>400:
                    raise ValueError("CAMERA_FRAME_STALE")
                raise ValueError("INVALID_SIGNAL_RESPONSE")
            with self.lock:
                if cancel:
                    if not peer["armed"]:
                        self.reset_remote_pending = False
                    return
                if session != self.session:
                    return
                if session:
                    if peer["boot"] != boot:
                        raise ValueError("DEVICE_REBOOTED")
                    if peer.get("session") != session or peer["armed"] is not True:
                        raise ValueError("SESSION_CHANGED")
                    if peer.get("initial_lamp_state", "OFF") != initial_lamp_state:
                        raise ValueError("INITIAL_LAMP_STATE_MISMATCH")
                    if self.acknowledged and peer["sequence"] < self.peer.get("sequence",0):
                        raise ValueError("FRAME_SEQUENCE_REGRESSED")
                    self.acknowledged = True
                    self.phase = peer.get("phase","WAIT_CHANGE")
                if peer["sequence"] != self.peer.get("sequence") or peer["boot"] != self.peer.get("boot"):
                    self.progress_at = self.clock()
                if session and peer["confirmed"] and (self.progress_at is None or self.clock()-self.progress_at>self.MAX_AGE_S):
                    raise ValueError("CAMERA_SEQUENCE_STALLED")
                self.last_error_kind = self.last_error_reason = ""
                self.peer, self.received, self.error = peer, self.clock(), "" if peer["paired"] else "PAIRING_REQUIRED"
                if session and peer["confirmed"]:
                    self._grant(peer)
                elif self.reset_remote_pending and not self.armed and not self.permission:
                    if peer["armed"] and self._remote_session(peer):
                        item = (peer["session"],peer["boot"])
                        if item not in self.cancel_queue:
                            self.cancel_queue.append(item)
                    elif not peer["armed"]:
                        self.reset_remote_pending = False
        except Exception as exc:
            with self.lock:
                if not cancel and session != self.session:
                    return
                self.last_request_ms = round((self.clock()-began)*1000,1)
                self.last_error_kind = type(exc).__name__
                self.last_endpoint = self.endpoint
                codes = ("RESPONSE_TOO_SLOW","CAMERA_FRAME_STALE","INVALID_SIGNAL_RESPONSE","DEVICE_REBOOTED","SESSION_CHANGED","INITIAL_LAMP_STATE_MISMATCH","FRAME_SEQUENCE_REGRESSED","CAMERA_SEQUENCE_STALLED")
                self.last_error_reason = str(exc) if str(exc) in codes else type(exc).__name__
                if isinstance(exc,OSError):
                    self.endpoint = None
                if cancel:
                    # A lost cancellation must be retried now that a confirmed
                    # ESP32 event is intentionally held beyond its old lease.
                    if getattr(exc,"code",None) != 409 and cancel not in self.cancel_queue:
                        self.cancel_queue.insert(0,cancel)
                    return
                transient = (isinstance(exc,OSError) and getattr(exc,"code",None) not in (400,401,403,409)
                             or str(exc) in ("RESPONSE_TOO_SLOW","CAMERA_FRAME_STALE","CAMERA_SEQUENCE_STALLED"))
                self.error = "ESP32_WAITING_FOR_RECOVERY" if session and transient else "ESP32_OFFLINE_REARM_REQUIRED" if session else "ESP32_OFFLINE"
                self.received = None
                if session and not self.permission:
                    if transient:
                        # Preserve the nonce so a confirmed ESP32 event can be
                        # collected after a brief lost reply. No permit yet.
                        self.phase = "WAIT_CONNECTION"
                    else:
                        self.fault_reason, self.fault_kind = self.last_error_reason,self.last_error_kind
                        self._release_peer()
                        self.armed, self.phase = False,"FAULT"

    def _worker(self):
        while not self.stopping.is_set():
            self._step()
            self.stopping.wait(.1)

    def close(self):
        self.cancel()
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=1)
        # Remote cancellation is best effort. Its 2 s lease also expires.
        if self.key and self.cancel_queue:
            session, boot = self.cancel_queue[-1]
            try:
                self.transport("cancel", session, boot)
            except Exception:
                pass

    def snapshot(self):
        with self.lock:
            p = self.confirmation if self.permission else self.peer
            evidence = self.permission or self.armed and self.acknowledged
            return dict(enabled=self.enabled, source="ESP32", armed=self.armed, fired=self.fired,
                        initial_lamp_state=self.initial_lamp_state,
                        permission=self.permission, phase=self.phase, connected=self._fresh() and not self.error,
                        error=self.error, error_kind=self.last_error_kind, error_reason=self.last_error_reason,
                        fault_reason=self.fault_reason, fault_kind=self.fault_kind,
                        lamp_configuration=self._lamp_configuration(self.peer),
                        candidate_hint=self._candidate_hint(),
                        configuration_update=dict(self.configuration_update),
                        request_ms=self.last_request_ms, control_endpoint=self.endpoint or self.last_endpoint,
                        event_confirmed=self.permission, signal_confirmed=self.permission,
                        confirmation_frames=p.get("confirmation_frames",0) if evidence else 0,
                        required_frames=p.get("required_frames",self.MIN_CONFIRM_FRAMES),
                        confirmed_s=p.get("confirmed_ms",0)/1000 if evidence else 0,
                        required_s=p.get("required_ms",self.MIN_CONFIRM_MS)/1000,
                        search_roi=list(p.get("roi") or self.SEARCH_ROIS[-1]), trigger_direction="DARKER",
                        target_locked=bool(p.get("target_locked")) and self.armed and not self.permission,
                        target_roi=p.get("target_roi"), contrast=p.get("contrast"), baseline_contrast=p.get("baseline_contrast"),
                        rejection_reason=p.get("rejection_reason"), background_change=p.get("background_change"),
                        background_residual=p.get("background_residual"), darkened_fraction=p.get("darkened_fraction"),
                        lamp=dict(direction=p.get("direction","")),
                        peer_url=self.peer.get("preview_url") or ("http://%s/" % urlsplit(self.endpoint).hostname if self.endpoint else self.url))
