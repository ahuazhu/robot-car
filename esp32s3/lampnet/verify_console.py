"""USB preview and a separate lamp trial; never creates a Maix start session."""
import argparse
import io
import json
import queue
import re
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import serial
from PIL import Image

PAGE = Path(__file__).with_name('verify_console.html')
MODEL_STATUS_MAX_AGE_S = 3.0
WAIT_MODEL = '自动定位尚未通过模型自检'


def lamp_configuration(body):
    """Match the existing firmware's manual-ROI bounds before queueing USB R."""
    if not isinstance(body, dict) or type(body.get('manual')) is not bool:
        raise ValueError('请选择手动范围或自动定位')
    if not body['manual']:
        return dict(manual=False)
    roi = body.get('roi')
    if not isinstance(roi, list) or len(roi) != 4 or any(type(n) is not int for n in roi):
        raise ValueError('灯面范围须为四个整数')
    x, y, w, h = roi
    if not (x >= 0 and y >= 0 and w >= 16 and h >= 16 and x+w <= 320 and y+h <= 240):
        raise ValueError('灯面范围超出画面或尺寸过小')
    return dict(manual=True, roi=roi[:])


class Recorder:
    def __init__(self, port, out, diagnostic=False, *, serial_factory=None,
                 startup_s=.7, interval_s=.8):
        self.port, self.out = port, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out/'frames').mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.model, self.model_at, self.model_revision = {}, None, 0
        self.jpeg, self.jpeg_at, self.frame = None, 0, 0
        self.error = ''
        self.began = time.monotonic()
        self.stream = None
        self.serial_factory, self.startup_s, self.interval_s = serial_factory or serial.Serial, startup_s, interval_s
        self.events = (self.out/'events.jsonl').open('a')
        self.serial_log = (self.out/'serial.log').open('a')
        self.commands = queue.Queue()
        self.inflight_configuration = None
        self.config_error = self.config_result = self.test_error = ''
        self.diagnostic, self.test_started = diagnostic, False
        self.test_request = threading.Event()
        self.test_origin, self.test_configuration = 'startup' if diagnostic else '', None
        if diagnostic:
            self.test_request.set()

    def event(self, kind, **values):
        with self.lock:
            self.events.write(json.dumps(dict(kind=kind, t=time.monotonic()-self.began, **values), ensure_ascii=False)+'\n')
            self.events.flush()

    def _native_blocker(self):
        """Caller holds lock; unknown/stale formal state cannot authorize USB control."""
        if self.stop.is_set() or self.error or self.model_at is None or time.monotonic()-self.model_at > MODEL_STATUS_MAX_AGE_S:
            return 'ESP32 状态尚未就绪或已过期，预览恢复后再试'
        diagnostic = self.model.get('diagnostic')
        if not isinstance(diagnostic, dict) or type(diagnostic.get('native_armed')) is not bool:
            return '固件未提供独立验证状态'
        if diagnostic['native_armed']:
            return 'Maix 已 ARM：正式等灯期间不能开启独立验证或修改范围，请先 STOP'
        return ''

    def _test_blocker(self):
        blocked = self._native_blocker()
        if blocked:
            return blocked
        if self.inflight_configuration is not None or not self.commands.empty():
            return '灯面范围正在保存，请等待设置生效'
        try:
            configuration = lamp_configuration(self.model['diagnostic'].get('configuration'))
        except ValueError as exc:
            return str(exc)
        # Manual detection has the same saved bounds and formal-ARM exclusion
        # as the firmware, without depending on unrelated model self-tests.
        if not configuration['manual'] and self.model.get('selftest_ok') is not True:
            return WAIT_MODEL
        return ''

    def request_test(self):
        with self.lock:
            blocked = self._test_blocker()
            if blocked:
                self.test_error = blocked
                return False, blocked
            if not self.test_request.is_set():
                self.test_configuration = lamp_configuration(self.model['diagnostic']['configuration'])
                self.test_origin = 'operator'
                self.test_request.set()
            self.test_error = ''
        self.event('isolated_test_queued', controls_start=False)
        return True, ''

    def request_configuration(self, configuration):
        with self.lock:
            blocked = self._native_blocker()
            if blocked:
                return False, blocked
            self.test_request.clear()
            self.test_configuration = None
            self.test_error = ''
            self.commands.put(configuration)
        return True, ''

    def status(self):
        with self.lock:
            blocked = self._test_blocker()
            return dict(model=self.model, frame=self.frame, error=self.error, recording=not self.stop.is_set(),
                        controls_start=False, session_kind='ISOLATED_USB_TEST',
                        frame_age_ms=int((time.monotonic()-self.jpeg_at)*1000) if self.jpeg_at else 999999,
                        config_error=self.config_error, config_result=self.config_result,
                        configuration_pending=self.inflight_configuration is not None or not self.commands.empty(),
                        control_ready=not self._native_blocker(), test_ready=not blocked,
                        test_blocked_reason=blocked, test_error=self.test_error,
                        test_pending=self.test_request.is_set())

    def lines(self, data):
        for line in data.decode(errors='replace').splitlines():
            self.serial_log.write(line+'\n')
            self.serial_log.flush()
            if line.startswith('ROI ERROR '):
                with self.lock:
                    self.inflight_configuration = None
                    self.config_error = line[10:]
                self.event('configuration_error', reason=line[10:])
            elif line.startswith('ROI OK '):
                try:
                    configuration = lamp_configuration(json.loads(line[7:]))
                except (ValueError, TypeError):
                    continue
                with self.lock:
                    self.inflight_configuration = None
                    self.config_error, self.config_result = '', '设置已保存到 ESP32'
                self.event('configuration_saved', configuration=configuration)
            match = re.search(r'LAMP_MODEL (\{.*)', line)
            if match:
                try:
                    model = json.JSONDecoder().raw_decode(match.group(1))[0]
                    if not isinstance(model, dict):
                        continue
                except json.JSONDecodeError:
                    continue
                with self.lock:
                    self.model, self.model_at = model, time.monotonic()
                    self.model_revision += 1
                    self.error = ''
                    diagnostic = model.get('diagnostic', {})
                    if not isinstance(diagnostic, dict):
                        diagnostic = {}
                    if diagnostic.get('native_armed') or diagnostic.get('enabled') is False:
                        self.test_started = False
                    if self.inflight_configuration is not None:
                        try:
                            observed = lamp_configuration(diagnostic.get('configuration'))
                        except ValueError:
                            observed = None
                        if observed == self.inflight_configuration:
                            # A fresh M also acknowledges persisted configuration
                            # if an interleaved serial ROI OK line was lost.
                            self.inflight_configuration = None
                            self.config_error, self.config_result = '', '设置已保存到 ESP32'
                        elif diagnostic.get('native_armed'):
                            self.inflight_configuration = None
                            self.config_error = 'Maix 已 ARM，范围保存未确认；请先 STOP 后检查范围'
                self.event('model', state=model)

    def model_status(self):
        before = self.model_revision
        self.stream.write(b'M\n')
        end, data = time.monotonic()+.65, b''
        while time.monotonic() < end:
            data += self.stream.read(4096)
            if data.endswith(b'\n') and b'LAMP_MODEL {' in data:
                break
        self.lines(data)
        return self.model_revision != before

    def process_requests(self, status_updated):
        """Single serial owner rechecks queued work against this iteration's M reply."""
        events = []
        with self.lock:
            try:
                configuration = self.commands.get_nowait()
            except queue.Empty:
                configuration = None
            if configuration is not None:
                self.test_request.clear()
                self.test_configuration = None
                blocked = self._native_blocker() if status_updated else '未收到新的 ESP32 状态，范围未修改'
                if blocked:
                    self.config_error = blocked
                    events.append(('configuration_rejected', dict(reason=blocked)))
                else:
                    self.config_error, self.config_result = '', ''
                    self.stream.write(b'R'+json.dumps(configuration, separators=(',', ':')).encode()+b'\n')
                    self.inflight_configuration = configuration
                    events.append(('configuration_requested', dict(configuration=configuration)))
            if self.test_request.is_set():
                blocked = self._test_blocker() if status_updated else '未收到新的 ESP32 状态，独立验证未启动'
                actual = lamp_configuration(self.model['diagnostic']['configuration']) if not blocked else None
                if not blocked and self.test_configuration is not None and actual != self.test_configuration:
                    blocked = '灯面范围已变化，请检查范围后重新准备独立验证'
                # --diagnostic may wait through initial connection/self-test.
                # Operator requests are never kept for a later, surprising T.
                wait_startup = self.test_origin == 'startup' and (not status_updated or blocked == WAIT_MODEL)
                if blocked:
                    self.test_error = blocked
                    if not wait_startup:
                        self.test_request.clear()
                        self.test_configuration = None
                        events.append(('isolated_test_rejected', dict(reason=blocked, controls_start=False)))
                else:
                    self.stream.write(b'T\n')
                    self.test_started = True
                    self.test_request.clear()
                    self.test_configuration = None
                    self.test_error = ''
                    events.append(('isolated_test_requested', dict(controls_start=False)))
        for kind, values in events:
            self.event(kind, **values)

    def image(self):
        self.stream.write(b'J\n')
        end, data, match = time.monotonic()+2.0, b'', None
        while time.monotonic() < end:
            data += self.stream.read(4096)
            match = re.search(rb'JPEG (\d+)\r?\n', data)
            if match:
                break
        if not match:
            self.lines(data)
            return False
        self.lines(data[:match.start()])
        size, start = int(match.group(1)), match.end()
        if size <= 0 or size > 150000:
            raise ValueError('Unexpected JPEG size')
        end = time.monotonic()+max(3.0, size*10/115200+1.0)
        while len(data)-start < size and time.monotonic() < end:
            data += self.stream.read(min(4096, size-(len(data)-start)))
        image = data[start:start+size]
        if len(image) != size or b'LAMP_MODEL {' in image or not image.startswith(b'\xff\xd8') or not image.endswith(b'\xff\xd9'):
            self.event('dropped_frame', reason='serial framing or concurrent diagnostic line')
            return False
        try:
            decoded = Image.open(io.BytesIO(image))
            decoded.load()
            if decoded.size != (320, 240):
                raise ValueError('Unexpected preview dimensions')
        except Exception as exc:
            self.event('dropped_frame', reason=str(exc))
            return False
        with self.lock:
            self.jpeg = image
            self.frame += 1
            frame, self.jpeg_at = self.frame, time.monotonic()
        filename = 'frames/%06d.jpg' % frame
        (self.out/filename).write_bytes(image)
        self.event('frame', file=filename, jpeg_bytes=len(image))
        self.lines(data[start+size:])
        return True

    def run(self):
        try:
            stream = self.serial_factory()
            stream.port, stream.baudrate, stream.timeout = self.port, 115200, .05
            stream.dtr = stream.rts = False
            stream.open()
            self.stream = stream
            if self.stop.wait(self.startup_s):
                return
            self.event('recording_started', manual_configuration_available=True, mode='ISOLATED_USB_TEST' if self.diagnostic else 'PREVIEW_ONLY', controls_start=False)
            while not self.stop.is_set():
                start = time.monotonic()
                self.process_requests(self.model_status())
                self.image()
                self.stop.wait(max(0, self.interval_s-(time.monotonic()-start)))
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
            self.event('error', message=str(exc))
        finally:
            if self.stream:
                if self.test_started:
                    self.stream.write(b'U\n')
                    self.stream.flush()
                    self.event('isolated_test_cancelled', controls_start=False)
                self.stream.close()


def make_handler(rec):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, content, mime, status=200):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def json(self, value, status=200):
            return self.send(json.dumps(value, ensure_ascii=False).encode(), 'application/json', status)

        def do_GET(self):
            if self.path == '/':
                return self.send(PAGE.read_text().encode(), 'text/html; charset=utf-8')
            if self.path.startswith('/status'):
                return self.json(rec.status())
            if self.path.startswith('/frame'):
                with rec.lock:
                    image = rec.jpeg
                if image:
                    return self.send(image, 'image/jpeg')
                self.send_error(503)
                return
            self.send_error(404)

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and origin != 'http://127.0.0.1:%d' % self.server.server_address[1]:
                self.send_error(403)
                return
            if self.path == '/roi':
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 160:
                        raise ValueError('Invalid configuration size')
                    configuration = lamp_configuration(json.loads(self.rfile.read(size)))
                except (ValueError, TypeError):
                    return self.json(dict(error='灯面范围格式不正确'), 400)
                accepted, error = rec.request_configuration(configuration)
                return self.json(dict(queued=True, controls_start=False) if accepted else dict(error=error), 200 if accepted else 409)
            if self.path == '/test':
                accepted, error = rec.request_test()
                return self.json(dict(queued=True, controls_start=False, session_kind='ISOLATED_USB_TEST') if accepted else dict(error=error, controls_start=False), 200 if accepted else 409)
            if self.path != '/marker':
                self.send_error(404)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 128:
                    raise ValueError('Invalid marker size')
                label = json.loads(self.rfile.read(size))['label']
                if label not in ('ON', 'OFF', 'MOVED'):
                    raise ValueError('Invalid marker')
            except (ValueError, TypeError, KeyError):
                self.send_error(400)
                return
            rec.event('marker', label=label)
            return self.json(dict(ok=True))
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8130)
    parser.add_argument('--diagnostic', action='store_true', help='Separate detector only; never creates a Maix start session')
    args = parser.parse_args()
    rec = Recorder(args.serial, args.out, args.diagnostic)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(rec))
    worker = threading.Thread(target=rec.run)
    worker.start()

    def shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print('MODEL VERIFY http://127.0.0.1:%d/ recording=%s' % (args.port, args.out), flush=True)
    try:
        server.serve_forever()
    finally:
        rec.stop.set()
        worker.join(timeout=5)
        server.server_close()
        rec.events.close()
        rec.serial_log.close()


if __name__ == '__main__':
    main()
