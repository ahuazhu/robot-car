import cv2
import time,json
LIVE_HTML = '<!doctype html><html lang="zh-CN"><meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n<title>曲线中线巡线 v0.6.34</title><style>\nbody{margin:0;background:#111827;color:#e5e7eb;font:16px system-ui;padding:20px}\nmain{max-width:1440px;margin:auto}h1{font-size:24px}p{line-height:1.6}\nimg{display:block;width:100%;height:auto;border:1px solid #374151;border-radius:10px}\n.labels{display:flex}.labels span{width:33.33%;padding:12px 0;font-weight:600}\nsmall{color:#9ca3af}button{padding:12px 22px;border:0;border-radius:6px;cursor:pointer;font:inherit}\n#direct-start{background:#16a34a;color:white}#start{background:#2563eb;color:white}#stop{background:#dc2626;color:white;font-weight:bold}\nbutton:disabled{opacity:.4;cursor:not-allowed}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}\n#status{font-weight:600}#notice{min-height:1.6em;color:#fcd34d}\n.video-window{overflow:hidden;border:1px solid #374151;border-radius:10px}\n.video-window img{border:0;border-radius:0;max-width:none}\n.video-window[data-view="all"]{aspect-ratio:4/1}\n.video-window[data-view="all"] img{width:100%;transform:none}\n.video-window:not([data-view="all"]){aspect-ratio:4/3}\n.video-window:not([data-view="all"]) img{width:300%}\n.video-window[data-view="raw"] img{transform:translateX(0)}\n.video-window[data-view="annotated"] img{transform:translateX(-33.333333%)}\n.video-window[data-view="ground"] img{transform:translateX(-66.666667%)}\n.view-controls{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}\n.view-controls button{padding:8px 12px;background:#374151;color:#e5e7eb}\n.view-controls button.selected{background:#2563eb}\n#diagnostics{min-height:1.6em;color:#9ca3af}\n</style><main><h1>曲线中线巡线 v0.6.34</h1>\n<p>左侧为原图，中间为完整画面上的巡线叠加，右侧为地面投影（未标定时仅参考）。黄线为左边界，蓝线为右边界，绿线为已见道路中线，橙色虚线为此前已见中线经运动补偿得到的预测，红十字为动态前瞻追踪点。</p>\n<div class="controls"><button id="direct-start" disabled onclick="control(\'start_now\')">▶ 直接启动</button><button id="start" disabled onclick="control(\'start\')">START 开始</button>\n<button id="stop" onclick="control(\'stop\')">STOP 停车</button><span id="status">正在连接…</span></div>\n<p id="notice" role="status" aria-live="polite"></p><p id="lap-status" role="status"></p><p id="start-light-status" role="status"></p><p id="crosswalk-status" role="status"></p><p id="diagnostics"></p><div class="view-controls" aria-label="选择画面">\n<button data-view="all" onclick="setView(\'all\')">三图对照</button>\n<button data-view="raw" onclick="setView(\'raw\')">完整原图</button>\n<button data-view="annotated" onclick="setView(\'annotated\')">巡线叠加</button>\n<button data-view="ground" onclick="setView(\'ground\')">地面投影</button></div>\n<div class="labels" id="labels"><span>原始画面</span><span>巡线识别</span><span>地面投影</span></div>\n<div id="video-window" class="video-window" data-view="all"><img id="video" src="/stream" alt="连接摄像头中"></div>\n<p>上方帧号持续增加表示画面正在更新。原图保留完整视野，巡线叠加图不再被状态栏或按钮遮挡。上方 START／STOP 控制小车；相机触摸屏仍可直接操作。启用发车灯时，应用打开后在整个原始视野搜索发车灯，灯光外观匹配并稳定亮起约0.35秒，且道路和通信就绪后自动发车。“直接启动”无需等灯，仍检查道路和通信。STOP 会取消等待并停车；ARM 可先进入等灯状态，无需先等道路就绪；亮灯后仍检查道路和通信。两种发车共用同一套检查及三圈逻辑；计圈线可在启动后识别。每次只触发一次。关闭发车灯模式时 START 直接开始巡线。本轮完成3圈后自动停车，起步越线不计圈；按横跨道路的计圈白线计圈，不依赖龙门架，重复画面不重复计数。短暂丢线或中线待确认时减速或零速寻回，确认后自动继续；道路暂时找不到时保持零速，连续确认可信道路后自动继续；电机通信故障或手动 STOP 会结束本次运行。PREVIEW ONLY 表示仅预览；CALIBRATION REQUIRED 表示需按当前镜头角度重新标定。</p>\n<button onclick="document.getElementById(\'video\').src=\'/stream?t=\'+Date.now()">重新连接画面</button>\n<p><small>电脑或手机需与摄像头连接同一 Wi-Fi。退出巡线应用后，视频会断开。</small></p></main><script>\nconst token="__CONTROL_TOKEN__";\nlet generation=0,canStart=false,canDirect=false,busy=false;\nconst directButton=document.getElementById(\'direct-start\'),startButton=document.getElementById(\'start\'),statusBox=document.getElementById(\'status\'),notice=document.getElementById(\'notice\');\nasync function updateStatus(){\n try{\n  const r=await fetch(\'/status\',{cache:\'no-store\',signal:AbortSignal.timeout(1500)});\n  if(!r.ok)throw Error(\'状态连接失败\');\n  const s=await r.json();generation=s.generation;\n  const fresh=s.age_ms<1000;\n  canStart=fresh&&!s.active&&!!(s.readiness?.arm_start_light_ready??s.ready);\n  canDirect=fresh&&!s.active&&!!(s.readiness?.direct_start_ready??s.ready);\n  directButton.disabled=!canDirect||busy;\n  startButton.disabled=!canStart||busy;\n  statusBox.textContent=fresh?`${s.active?\'运行中\':\'已停止\'} · ${s.reason} · L=${s.left} R=${s.right}`:\'状态更新中断\';\n   const roadStates={"ROAD SEARCH SCANNING":"小幅转向寻路","ROAD SEARCH SETTLING":"寻路中，等待停稳","REACQUIRING ROAD":"正在寻回道路，确认后自动继续","VISION WAIT":"等待及时画面，恢复后自动继续","ROAD SEARCH":"正在重新搜索道路","ROAD NOT FOUND":"暂未找到道路，零速等待恢复"};\n   if(fresh&&roadStates[s.reason])statusBox.textContent=roadStates[s.reason]+` · L=${s.left} R=${s.right}`;\n   const failures={"CROSSWALK STOP UNVERIFIED":"横道停车位置未能确认","CROSSWALK POSITION UNCERTAIN":"横道距离不确定","CROSSWALK STOP WINDOW MISSED":"未能在横道前及时停车"};\n   if(fresh&&failures[s.reason])statusBox.textContent=failures[s.reason]+\' · 已停车\';\n   const d=s.readiness||{};\n   const laps=d.laps||{};\n   document.getElementById(\'lap-status\').textContent=laps.enabled?`圈数：${laps.completed_laps}/${laps.total_laps}圈 · ${laps.complete?\'三圈完成，已停车\':laps.phase===\'SEEK_DEPARTURE\'?\'等待识别计圈线，首次越线不计\':laps.phase===\'DEPARTURE\'?\'起步越线不计圈\':laps.phase===\'WAIT_START\'||laps.phase===\'STOPPED\'?\'等待新一轮启动\':\'比赛进行中\'}${laps.unverified_crossings?\' · 有越线未能确认，请核对\':\'\'}`:\'\';\n   const lineMarker=laps.detector||{};\n   if(laps.enabled)document.getElementById(\'lap-status\').textContent+=lineMarker.confirmed?\' · 计圈白线已确认\':lineMarker.marker_visible?\' · 计圈白线确认中\':\' · 未看到计圈白线\';\n   if(fresh&&s.reason===\'LAPS COMPLETE\')statusBox.textContent=`已完成${laps.total_laps}圈，自动停车 · L=0 R=0`;\n   const lamp=d.start_light||{};\n   startButton.textContent=lamp.enabled?\'ARM 等待发车灯\':\'START 开始\';\n   const lampPhases={WAIT_FRESH_FRAME:\'已看到亮灯，等待及时画面\',WAIT_LIGHT:\'等待发车灯亮起\',CONFIRM_LIGHT:\'正在确认灯光\',SIGNAL_READY:\'发车信号已确认，等待道路和通信就绪\',RUNNING:\'本次运行已启动\',DISARMED:\'灯光识别已暂停，按 ARM 开始等灯\'};\n   document.getElementById(\'start-light-status\').textContent=lamp.enabled?`发车灯（全画面搜索）：${lampPhases[lamp.phase]||lamp.phase} · ${lamp.phase===\'DISARMED\'?\'未开启等灯\':lamp.lamp?.state===\'ON\'?\'亮\':lamp.lamp?.state===\'OFF\'?\'灭\':\'未确认\'} · 确认 ${lamp.confirmed_s||0}/${lamp.required_s}秒`:\'\';\n   if(fresh&&s.reason===\'WAIT START LIGHT\')statusBox.textContent=\'等待龙门架发车灯 · L=0 R=0\';\n   const crossing=d.crosswalk;\n   if(crossing){\n    const phases={SEARCH:\'搜索横道\',APPROACH:\'接近横道\',STOPPING:\'确认停稳\',HOLD:\'停车礼让\',RECOVERY_HOLD:\'补充礼让（停车位置未核实）\',PASSING:\'通过横道\',FAULT:\'任务故障\'};\n    const distance=crossing.distance_to_body_m;\n    const observed=crossing.observation;\n    const range=distance!==null&&distance!==undefined?`距车身 ${Math.round(distance*100)}cm`:observed?`距镜头参考点 ${Math.round(observed.near_m*100)}cm`:\'\';\n    const mode=crossing.mode===\'yield\'?\'自动礼让\':\'仅识别，自动礼让尚未启用\';\n    document.getElementById(\'crosswalk-status\').textContent=[`横道：${mode}`,phases[crossing.phase]||crossing.phase,range,crossing.phase===\'HOLD\'?`已停 ${crossing.hold_elapsed_s.toFixed(1)}秒`:\'\',crossing.failure||crossing.recovery_reason||\'\',crossing.yield_completed?(crossing.yield_verified?\'礼让已验证\':\'礼让已完成，位置未验证\'):\'\'].filter(Boolean).join(\' · \');\n   }\n\n   const labels={LAP_START_NOT_READY:\'请在计圈白线前摆正，等待龙门架与白线确认\',PREVIEW_ONLY:\'当前仅预览\',UNO_DISCONNECTED:\'UNO未连接\',VISION_TOO_SLOW:\'视觉处理超时\',PATH_CONFIRMING:\'正在确认中线\',ROAD_NO_TARGET:\'暂无有效追踪点\',ROAD_UNOBSERVED:\'等待有效道路观测\',ROAD_WEAK:\'道路置信度不足\',ROAD_STABILIZING:\'等待道路稳定\',CROSSWALK_STOP_UNVERIFIED:\'横道停车位置尚未确认\'};\n   const search=(d.recovery||{}).search||{};\n   const searchLabels={"GROUND MOTION UNVERIFIED":"等待有效运动估计","BODY POSE CLEARANCE UNVERIFIED":"车身与道路余量尚未确认","NO OBSERVABLE SAFE YAW STEP":"暂无可确认的小幅转向空间","ROTATION SWEEP NOT INSIDE KNOWN ROAD":"转向范围超过可信道路余量","SEARCH BUDGET OR DRIFT LIMIT":"试探达到角度、漂移或时长界限","SWEEP COMPLETED WITHOUT A CONFIRMED PATH":"左右试探后仍无可信追踪路径"};\n   const searchNotice=searchLabels[search.reason]||\'\';\n   const waiting=(!s.active&&!s.ready)?(d.blocked_by||[]).map(k=>failures[k]||labels[k]||k).join(\'；\'):\'\';\n   document.getElementById(\'diagnostics\').textContent=[waiting,searchNotice,d.observed_streak!==undefined?`道路确认 ${Math.min(d.observed_streak,d.required_observations)}/${d.required_observations}`:\'\',d.processing_ms!==undefined?`视觉 ${d.processing_ms}ms`:\'\',d.confidence!==undefined?`置信度 ${Math.round(d.confidence*100)}%`:\'\',d.frame!==undefined?`帧 ${d.frame}`:\'\'].filter(Boolean).join(\' · \');\n\n }catch(e){canStart=canDirect=false;directButton.disabled=startButton.disabled=true;statusBox.textContent=\'连接中断，请检查摄像头和Wi-Fi\';}\n}\nasync function control(action){\n const starting=action===\'start\'||action===\'start_now\';\n if(starting&&(busy||(action===\'start_now\'?!canDirect:!canStart)))return;\n if(starting){busy=true;directButton.disabled=startButton.disabled=true;}\n notice.textContent=action===\'stop\'?\'正在发送停车请求…\':\'正在发送启动请求…\';\n try{\n  const r=await fetch(\'/control\',{method:\'POST\',headers:{\'Content-Type\':\'application/json\',\'X-Control-Token\':token},\n   body:JSON.stringify({action,generation}),signal:AbortSignal.timeout(2000)});\n  const result=await r.json();\n  if(!r.ok)throw Error(result.message||\'请求失败\');\n  notice.textContent=action===\'stop\'?\'停车请求已接收，以上方实际状态为准。\':\'启动请求已接收，以上方实际状态为准。\';\n }catch(e){notice.textContent=e.message+\'；请检查上方状态，必要时使用车上STOP或电源开关。\';}\n finally{if(starting)busy=false;await updateStatus();}\n}\nfunction setView(view){\n if(![\'all\',\'raw\',\'annotated\',\'ground\'].includes(view))view=\'all\';\n document.getElementById(\'video-window\').dataset.view=view;\n document.getElementById(\'labels\').style.display=view===\'all\'?\'flex\':\'none\';\n document.querySelectorAll(\'.view-controls button\').forEach(b=>b.classList.toggle(\'selected\',b.dataset.view===view));\n try{localStorage.setItem(\'curve-camera-view\',view)}catch(e){}\n}\nlet initialView=\'all\';try{initialView=localStorage.getItem(\'curve-camera-view\')||\'all\'}catch(e){}\nsetView(initialView);updateStatus();setInterval(updateStatus,500);\n</script></html>'

CONTROL_FRESH_SECONDS = 1.0


class WebControl:
    """Queue commands for the motor loop; STOP invalidates old START requests."""
    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.generation = 0
        self.pending = None
        self.updated = 0
        self.state = dict(active=False, ready=False, reason='INITIALIZING', left=0, right=0)

    def update(self, active, ready, reason, pwm, readiness=None):
        with self.lock:
            self.updated = time.monotonic()
            self.state = dict(active=active, ready=ready, reason=reason, left=pwm[0], right=pwm[1], readiness=readiness or {})

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
            if action not in ('start','start_now'):
                return False, '不支持的操作'
            if generation != self.generation:
                return False, '状态已变化，请重试'
            if self.pending is not None:
                return False, '上一条操作尚未完成'
            readiness=self.state.get('readiness',{})
            ready=readiness.get('direct_start_ready',self.state['ready']) if action=='start_now' else readiness.get('arm_start_light_ready',self.state['ready'])
            if time.monotonic()-self.updated >= CONTROL_FRESH_SECONDS or not ready or self.state['active']:
                return False, '当前不能启动：请检查道路识别和车辆状态'
            self.pending = (action, time.monotonic())
            return True, '启动请求已接收'

    def take(self):
        with self.lock:
            pending, self.pending = self.pending, None
            if pending is None:
                return None
            action, created = pending
            if action in ('start','start_now') and time.monotonic()-created >= CONTROL_FRESH_SECONDS:
                print('WEB START EXPIRED', json.dumps(dict(age_ms=round((time.monotonic()-created)*1000))), flush=True)
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
        self.metrics = {}
        self.preview_error = None
        self.server = self.worker = None
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
                if path == '/vision':
                    self.send_json(200, owner.metrics)
                    return
                if path == '/capture':
                    with owner.condition:data=owner.jpeg
                    if data is None:
                        self.send_error(503);return
                    self.send_response(200)
                    self.send_header('Content-Type','image/jpeg')
                    self.send_header('Content-Length',str(len(data)))
                    self.send_header('Cache-Control','no-store')
                    self.end_headers();self.wfile.write(data);return
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
        try:self.server = ThreadingHTTPServer(('0.0.0.0',port), Handler)
        except OSError as exc:
            self.preview_error='HTTP preview unavailable: '+str(exc)
            print('PREVIEW ERROR',self.preview_error,flush=True)
            return
        self.server.daemon_threads = True
        self.worker = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval':0.1},daemon=True)
        self.worker.start()

    def write(self, bgr):
        if self.server is None:return False
        try:
            ok,jpeg=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,75])
            if not ok:raise ValueError('JPEG encode failed')
            encoded=jpeg.tobytes()
        except (cv2.error,ValueError,OSError) as exc:
            message=str(exc)
            if self.preview_error!=message:print('PREVIEW ERROR',message,flush=True)
            self.preview_error=message
            return False
        with self.condition:
            self.jpeg=encoded;self.sequence+=1;self.preview_error=None
            self.condition.notify_all()
        return True

    def stop(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.server is None:return
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=0.5)
