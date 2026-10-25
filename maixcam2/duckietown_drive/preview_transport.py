import time,json
LIVE_HTML = '<!doctype html><html lang="zh-CN"><meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n<title>小车实时画面</title><style>\nbody{margin:0;background:#111827;color:#e5e7eb;font:16px system-ui;padding:20px}\nmain{max-width:1440px;margin:auto}h1{font-size:24px}p{line-height:1.6}\nimg{display:block;width:100%;height:auto;border:1px solid #374151;border-radius:10px}\n.labels{display:flex}.labels span{width:33.33%;padding:12px 0;font-weight:600}\nsmall{color:#9ca3af}button{padding:12px 22px;border:0;border-radius:6px;cursor:pointer;font:inherit}\n#start{background:#16a34a;color:white}#stop{background:#dc2626;color:white;font-weight:bold}\nbutton:disabled{opacity:.4;cursor:not-allowed}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}\n#status{font-weight:600}#notice{min-height:1.6em;color:#fcd34d}\n</style><main><h1>小车实时画面</h1>\n<p>左侧为原图，中间为道路识别与轮速指令，右侧为地面投影。首次使用低速测试。</p>\n<div class="controls"><button id="start" disabled onclick="control(\'start\')">START 开始</button>\n<button id="stop" onclick="control(\'stop\')">STOP 停车</button><span id="status">正在连接…</span></div>\n<p id="notice" role="status" aria-live="polite"></p><div class="labels"><span>原始画面</span><span>巡线识别</span><span>地面投影</span></div>\n<img id="video" src="/stream" alt="连接摄像头中">\n<p>画面角落的数字持续增加，表示仍在更新。上方 START／STOP 可以控制小车；视频图像中的按钮仅在摄像头触摸屏上有效。每轮最多运行5秒。</p>\n<button onclick="document.getElementById(\'video\').src=\'/stream?t=\'+Date.now()">重新连接画面</button>\n<p><small>电脑或手机需与摄像头连接同一 Wi-Fi。退出巡线应用后，视频会断开。</small></p></main><script>\nconst token="__CONTROL_TOKEN__";\nlet generation=0,canStart=false,busy=false;\nconst startButton=document.getElementById(\'start\'),statusBox=document.getElementById(\'status\'),notice=document.getElementById(\'notice\');\nasync function updateStatus(){\n try{\n  const r=await fetch(\'/status\',{cache:\'no-store\',signal:AbortSignal.timeout(1500)});\n  if(!r.ok)throw Error(\'状态连接失败\');\n  const s=await r.json();generation=s.generation;\n  const fresh=s.age_ms<1000;\n  canStart=fresh&&s.ready&&!s.active;\n  startButton.disabled=!canStart||busy;\n  statusBox.textContent=fresh?`${s.active?\'运行中\':\'已停止\'} · ${s.reason} · L=${s.left} R=${s.right}`:\'状态更新中断\';\n }catch(e){canStart=false;startButton.disabled=true;statusBox.textContent=\'连接中断，请检查摄像头和Wi-Fi\';}\n}\nasync function control(action){\n if(action===\'start\'&&(!canStart||busy))return;\n if(action===\'start\'){busy=true;startButton.disabled=true;}\n notice.textContent=action===\'stop\'?\'正在发送停车请求…\':\'正在发送启动请求…\';\n try{\n  const r=await fetch(\'/control\',{method:\'POST\',headers:{\'Content-Type\':\'application/json\',\'X-Control-Token\':token},\n   body:JSON.stringify({action,generation}),signal:AbortSignal.timeout(2000)});\n  const result=await r.json();\n  if(!r.ok)throw Error(result.message||\'请求失败\');\n  notice.textContent=action===\'stop\'?\'停车请求已接收，以上方实际状态为准。\':\'启动请求已接收，以上方实际状态为准。\';\n }catch(e){notice.textContent=e.message+\'；请检查上方状态，必要时使用车上STOP或电源开关。\';}\n finally{if(action===\'start\')busy=false;await updateStatus();}\n}\nupdateStatus();setInterval(updateStatus,500);\n</script></html>'

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
