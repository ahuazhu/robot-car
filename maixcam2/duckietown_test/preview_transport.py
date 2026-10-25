import time, json
LIVE_HTML = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<title>Duckietown 视觉测试</title><style>body{margin:0;padding:24px;background:#111827;color:#e5e7eb;font:16px system-ui}main{max-width:1280px;margin:auto}h1{font-size:24px}.labels{display:flex}.labels span{width:33.33%;padding:12px 0}img{width:100%;border:1px solid #475569;border-radius:8px}p{line-height:1.7}button{padding:10px 18px;font:inherit}#status{color:#facc15}</style>\n<main><h1>Duckietown 视觉测试</h1><p>当前仅检查识别效果，轮子不会转动。黄色为左边线，蓝色为右边线。</p><p id="status">连接中…</p>\n<div class="labels"><span>摄像头原图</span><span>边线检测</span><span>地面投影（需标定）</span></div>\n<img id="video" src="/stream"><p>标定时保持小车和镜头不动，让纸条四个端点清晰入镜。未标定时，右侧为空是正常现象。</p>\n<button onclick="document.getElementById(\'video\').src=\'/stream?t=\'+Date.now()">重新连接画面</button><p>退出测试：点击摄像头屏幕右下角 EXIT。</p></main>\n<script>async function update(){try{let r=await fetch(\'/status\',{cache:\'no-store\',signal:AbortSignal.timeout(1500)});let s=await r.json();document.getElementById(\'status\').textContent=s.age_ms<1000?\'检测正在运行 · \'+s.reason:\'画面更新已中断\';}catch(e){document.getElementById(\'status\').textContent=\'连接已断开，请检查摄像头\';}}update();setInterval(update,1000);</script></html>\n'

class WebControl:
    """Queue commands for the motor loop; STOP invalidates old START requests."""
    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.generation = 0
        self.pending = None
        self.updated = 0
        self.state = dict(active=False, ready=False, reason='INITIALIZING', left=0, right=0)

    def update(self, active, ready, reason, pwm):
        with self.lock:
            self.updated = time.monotonic()
            self.state = dict(active=active, ready=ready, reason=reason, left=pwm[0], right=pwm[1])

    def status(self):
        with self.lock:
            return dict(self.state, generation=self.generation,
                        age_ms=int((time.monotonic()-self.updated)*1000))

    def submit(self, action, generation=None):
        with self.lock:
            if action == 'stop':
                self.generation += 1
                self.pending = ('stop', time.monotonic())
                return True, '停车请求已接收'
            if action != 'start':
                return False, '不支持的操作'
            if generation != self.generation:
                return False, '状态已变化，请重试'
            if self.pending is not None:
                return False, '上一条操作尚未完成'
            if time.monotonic()-self.updated >= 1 or not self.state['ready'] or self.state['active']:
                return False, '当前不能启动：请检查道路识别和车辆状态'
            self.pending = ('start', time.monotonic())
            return True, '启动请求已接收'

    def take(self):
        with self.lock:
            pending, self.pending = self.pending, None
            if pending is None:
                return None
            action, created = pending
            if action == 'start' and time.monotonic()-created > 0.5:
                return None
            return action

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
        self.sequence = 0
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def send_json(self, code, value):
                body = json.dumps(value).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                from urllib.parse import urlsplit
                if self.path != '/control':
                    self.send_json(404, dict(message='不存在的接口'))
                    return
                origin = self.headers.get('Origin')
                if (origin and urlsplit(origin).netloc != self.headers.get('Host')) or not secrets.compare_digest(self.headers.get('X-Control-Token',''), owner.token):
                    self.send_json(403, dict(message='页面已失效，请刷新后重试'))
                    return
                self.connection.settimeout(2)
                try:
                    size = int(self.headers.get('Content-Length','0'))
                    if not 0 < size <= 256:
                        raise ValueError('invalid request size')
                    data = json.loads(self.rfile.read(size))
                    if not isinstance(data,dict):
                        raise ValueError('invalid command')
                except (ValueError, OSError):
                    self.send_json(400, dict(message='无效请求'))
                    return
                ok, message = owner.control.submit(data.get('action'),data.get('generation'))
                self.send_json(202 if ok else 409, dict(message=message))

            def do_GET(self):
                path = self.path.split('?', 1)[0]
                if path == '/status':
                    self.send_json(200, owner.control.status())
                    return
                if path == '/':
                    body = LIVE_HTML.replace('__CONTROL_TOKEN__',owner.token).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path != '/stream':
                    self.send_error(404)
                    return
                self.connection.settimeout(2)
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                last = -1
                try:
                    while not owner.closed:
                        with owner.condition:
                            owner.condition.wait_for(lambda: owner.closed or owner.sequence != last, timeout=1)
                            if owner.closed:
                                break
                            if owner.jpeg is None or owner.sequence == last:
                                continue
                            frame, last = owner.jpeg, owner.sequence
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(frame)).encode()+b'\r\n\r\n'+frame+b'\r\n')
                        self.wfile.flush()
                except (OSError, TimeoutError):
                    pass
        self.server = ThreadingHTTPServer(('0.0.0.0',port), Handler)
        self.server.daemon_threads = True
        self.worker = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval':0.1},daemon=True)
        self.worker.start()

    def write(self, bgr):
        ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY,75])
        if not ok:
            raise RuntimeError('JPEG encode failed')
        with self.condition:
            self.jpeg = jpeg.tobytes()
            self.sequence += 1
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=0.5)
