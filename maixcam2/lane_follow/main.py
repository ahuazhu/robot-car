"""MaixCAM2 visual lane following with UNO A4/A5 UART motor control.

Requires uno_lane_drive firmware. Starts stopped; touch START to drive.
Each trial lasts at most 12 seconds. Road loss or link failure stops motors.
The road detector assumes gray/white road with green outside both edges.
"""
import time
import cv2
import numpy as np

WIDTH, HEIGHT = 320, 240
# OpenCV HSV: H=0..179, S/V=0..255. Tune against the mounted camera image.
GREEN_LOW = (32, 55, 30)
GREEN_HIGH = (95, 255, 255)
ROW_FRACTIONS = tuple(y / 240 for y in range(224, 107, -4))


def runs(bits):
    """Return half-open intervals of True pixels."""
    edges = np.diff(np.r_[False, bits, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def detect_road(bgr):
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array(GREEN_LOW), np.array(GREEN_HIGH))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    points = []
    partial = []
    previous = None
    rows = []
    for fraction in ROW_FRACTIONS:
        y = min(h - 1, int(h * fraction))
        lo, hi = max(0, y - 2), min(h, y + 3)
        profile = np.mean(green[lo:hi] > 0, axis=0) >= 0.5
        # Close tiny holes in green caused by glare; do not close across road.
        profile = cv2.morphologyEx(profile.astype(np.uint8)[None, :],
                                  cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))[0] > 0
        candidates = []
        for left, end in runs(~profile):
            width = end - left
            # Keep observed single boundaries separately; a temporal width
            # model may complete them later, never the per-frame detector.
            left_clipped, right_clipped = left < 3, end > w-3
            if left_clipped != right_clipped and width >= w*0.20:
                edge_green = profile[end:end+3].all() if left_clipped else profile[left-3:left].all()
                if edge_green and np.mean(hsv[lo:hi,left:end,1] < 100) >= 0.50:
                    partial.append((y, 'right' if left_clipped else 'left', int(end-1 if left_clipped else left)))
            # Both boundaries need green evidence. Do not invent a clipped edge.
            if left < 3 or end > w - 3 or not (w * 0.10 <= width <= w * 0.98):
                continue
            if not (profile[left-3:left].all() and profile[end:end+3].all()):
                continue
            neutral = np.mean(hsv[lo:hi, left:end, 1] < 100)
            if neutral < 0.50:
                continue
            center = (left + end - 1) / 2.0
            if previous is not None:
                old_center, old_width = previous
                if abs(center - old_center) > max(w * 0.20, old_width * 0.65):
                    continue
                if not (0.30 <= width / old_width <= 1.65):
                    continue
                score = width * 0.25 - abs(center - old_center)
            else:
                score = width - 0.5 * abs(center - w / 2)
            candidates.append((score, int(left), int(end - 1), center, width))
        if not candidates:
            rows.append((y, None))
            continue
        _, left, right, center, width = max(candidates)
        point = (y, left, right, center)
        points.append(point)
        rows.append((y, point))
        previous = (center, width)
    # Keep one continuous band of observed left/right boundaries. A distant
    # unrelated patch must not supply the span needed to accept a near band.
    groups = []
    for point in points:
        if not groups or groups[-1][-1][0] - point[0] > h * 0.05:
            groups.append([])
        groups[-1].append(point)
    usable = []
    for g in groups:
        if len(g) < 3 or g[0][0] < h*0.55 or g[0][0]-g[-1][0] < h/30:
            continue
        local = [p for p in g if g[0][0]-p[0] <= h*0.20]
        ys = np.array([p[0] for p in local], dtype=float)
        widths = np.array([p[2]-p[1] for p in local], dtype=float)
        # A short visible band is enough when both boundaries are wide,
        # smooth and agree over multiple independent rows.
        if len(local) < 5 and widths.min() < w*0.30:
            continue
        if np.any(np.abs(np.diff(widths)) > np.maximum(widths[:-1]*0.25, w*0.08)):
            continue
        reliable = True
        for edge in (1,2):
            xs = np.array([p[edge] for p in local], dtype=float)
            fit = np.polyfit(ys-ys[0],xs,1)
            residual = np.sqrt(np.mean((np.polyval(fit,ys-ys[0])-xs)**2))
            if residual > max(3.0, widths.mean()*0.035):
                reliable = False
        if reliable:
            usable.append(g)
    points = usable[0] if usable else (groups[0] if groups else [])
    selected_y = {p[0] for p in points}
    rows = [(y, p if y in selected_y else None) for y, p in rows]
    valid = bool(usable)
    # Positive = road center to camera-right. Diagnostic only, not steering.
    offset = points[0][3] - (w - 1) / 2.0 if valid else None
    return {"valid": valid, "offset": offset, "rows": rows, "points": points,
            "green_mask": green, "partial": partial, "mode": "BOTH" if valid else "NONE"}


class RoadTracker:
    """Brief single-edge tracking using widths measured from BOTH edges.
    Never initialize from a lone edge; never keep driving on an old image.
    """
    def __init__(self):
        self.width_fit = None
        self.last_both = None

    def update(self, result, now):
        # Fresh valid double edges still refresh evidence even when a nearer
        # single edge is selected for steering. Do not expire visible roads.
        if result['valid']:
            self.last_both = now
        # A nearer observed edge outranks a more distant pair of edges.
        # In a bend, that distant pair can even belong to a roadside sign.
        if result['valid'] and self.width_fit is not None:
            near_y = result['points'][0][0]
            for side in ('left', 'right'):
                closer = [p for p in result.get('partial', [])
                          if p[1] == side and p[0] > near_y]
                # A shorter clipped band cannot satisfy single-edge tracking;
                # retain the valid nearby pair until a replacement is usable.
                if len(closer) >= 4 and closer[0][0]-near_y >= 12:
                    result = dict(result, valid=False, mode='NONE', offset=None)
                    break
        if result['valid']:
            pts = result['points']
            self.last_both = now
            if len(pts) >= 6 and pts[0][0]-pts[-1][0] >= 24:
                ys = np.array([p[0] for p in pts],float)
                widths = np.array([p[2]-p[1] for p in pts],float)
                fit = np.polyfit(ys,widths,1)
                reference = float(np.polyval(fit,180))
                # Width must grow toward the camera; a crossing or strong
                # curve must not corrupt the straight-road calibration.
                if 0.3 <= fit[0] <= 5 and 80 <= reference <= 480:
                    if self.width_fit is None:
                        self.width_fit = fit
                    else:
                        old = float(np.polyval(self.width_fit,180))
                        if 0.75 <= reference/old <= 1.30:
                            self.width_fit = 0.9*self.width_fit+0.1*fit
            return result
        if self.width_fit is None or self.last_both is None or now-self.last_both > 3.0:
            return result
        bands = []
        for side in ('left','right'):
            group = []
            for y, observed_side, edge in result.get('partial',[]):
                if side != observed_side or y < HEIGHT*0.60:
                    continue
                if group and group[-1][0]-y > 12:
                    if len(group) >= 4:
                        bands.append((side,group))
                    group = []
                width = float(np.polyval(self.width_fit,y))
                if not 80 <= width <= WIDTH*2:
                    continue
                left = float(edge) if side=='left' else float(edge)-width
                right = float(edge)+width if side=='left' else float(edge)
                group.append((y,left,right,(left+right)/2))
            if len(group) >= 4:
                bands.append((side,group))
        if not bands:
            return result
        side,pts = max(bands,key=lambda pair: (pair[1][0][0],len(pair[1])))
        near_y = pts[0][0]
        pts = [p for p in pts if near_y-p[0] <= HEIGHT*0.20]
        if len(pts)<4 or pts[0][0]-pts[-1][0]<12:
            return result
        # Reject disconnected or jagged edges rather than steering on glare.
        ys=np.array([p[0] for p in pts],float)
        xs=np.array([p[1] if side=='left' else p[2] for p in pts],float)
        fit=np.polyfit(ys-ys[0],xs,1)
        if np.sqrt(np.mean((np.polyval(fit,ys-ys[0])-xs)**2))>6:
            return result
        out=dict(result)
        out.update(valid=True,points=pts,rows=[(p[0],p) for p in pts],
                   offset=pts[0][3]-(WIDTH-1)/2,mode='EDGE '+side.upper())
        return out


def annotate(bgr, result):
    out = bgr.copy()
    h, w = out.shape[:2]
    cv2.line(out, (w // 2, h // 3), (w // 2, h - 1), (255, 255, 0), 1)
    for y, point in result["rows"]:
        if point is None:
            cv2.putText(out, "?", (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            continue
        _, left, right, center = point
        left,right=int(round(left)),int(round(right))
        cv2.line(out, (left, y), (right, y), (255, 180, 0), 1)
        cv2.circle(out, (left, y), 3, (0, 255, 0), -1)
        cv2.circle(out, (right, y), 3, (0, 255, 0), -1)
        cv2.circle(out, (round(center), y), 3, (0, 255, 255), -1)
    cv2.rectangle(out, (0, 0), (w, 33), (0, 0, 0), -1)
    cv2.putText(out, "PREVIEW ONLY - NO MOTOR COMMAND", (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
    status = "ROAD OK offset=%+.0fpx" % result["offset"] if result["valid"] else "ROAD UNCERTAIN"
    cv2.putText(out, status, (4, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 255, 255) if result["valid"] else (0, 0, 255), 1)
    return out


# Shared into the single-file MaixCAM2 app by the build step.
import time

def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ (7 if crc & 128 else 0)) & 255
    return crc

class DriveLink:
    def __init__(self):
        from maix import uart, pinmap, err
        err.check_raise(pinmap.set_pin_function('A21', 'UART4_TX'), 'UART4 TX')
        err.check_raise(pinmap.set_pin_function('A22', 'UART4_RX'), 'UART4 RX')
        self.port = uart.UART('/dev/ttyS4', 9600)
        self.seq = 0

    def send(self, op, left=0, right=0):
        self.seq = (self.seq + 1) & 65535
        payload = ('%d,%d,%d,%d' % (self.seq, op, left, right)).encode('ascii')
        self.port.read(timeout=0)
        self.port.write_str('@' + payload.decode('ascii') + '*%02X\n' % crc8(payload))
        deadline = time.monotonic() + 0.16
        buf = b''
        while time.monotonic() < deadline:
            data = self.port.read(timeout=0)
            if data:
                buf += data
                if len(buf) > 256:
                    raise RuntimeError('UART receive overflow')
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    line = line.rstrip(b'\r')
                    try:
                        body, checksum = line[1:].split(b'*')
                        if not line.startswith(b'!') or len(checksum) != 2 or crc8(body) != int(checksum, 16):
                            continue
                        values = [int(v) for v in body.split(b',')]
                        if len(values) == 5 and values[0] == self.seq:
                            if values[1] not in (0,1) or any(abs(v)>120 for v in values[2:4]):
                                raise RuntimeError('Invalid UNO status')
                            if op == 0 and values[1:4] != [0,0,0]:
                                raise RuntimeError('UNO did not confirm stopped')
                            if op in (1,2) and (values[1] != 1 or values[4] != 0 or values[2:4] != [left,right]):
                                raise RuntimeError('UNO rejected motion: %s' % values)
                            return values
                    except (ValueError, IndexError):
                        continue
            time.sleep(0.002)
        raise RuntimeError('UNO acknowledgement timeout')

    def stop(self):
        try:
            return self.send(0)
        except Exception as exc:
            print('STOP ACK unavailable; UNO watchdog will stop: %s' % exc, flush=True)
            return None

    def close(self):
        self.stop()
        self.port.close()

def lane_command(result, image_width=320):
    """Differential drive: positive image error -> stronger left -> turn right.
    Use lane-width-normalized near and look-ahead centers; never strafe.
    """
    # If a coherent near boundary crosses the optical center, stop rather
    # than navigating using a distant road patch. No assumed road width.
    for side in ('left', 'right'):
        edges = [p[2] for p in result.get('partial', [])
                 if p[1] == side and p[0] >= HEIGHT-28]
        if len(edges) >= 3:
            outside = [e >= (image_width-1)/2 if side == 'left'
                       else e <= (image_width-1)/2 for e in edges]
            if all(outside[:3]):
                return None
    if not result['valid']:
        return None
    near = result['points'][0]
    # Fit the local center direction instead of treating scan count as confidence.
    local = [p for p in result['points'] if near[0]-p[0] <= HEIGHT*0.20]
    ys = np.array([near[0]-p[0] for p in local], dtype=float)
    xs = np.array([p[3] for p in local], dtype=float)
    near_width = near[2] - near[1]
    if near_width < image_width * 0.15 or len(local)<3:
        return None
    slope = float(np.polyfit(ys,xs,1)[0])
    lateral = (near[3] - (image_width-1)/2) / (near_width/2)
    heading = slope * HEIGHT*0.20 / (near_width/2)
    if abs(lateral) >= 1.0:
        return None
    error = 0.65*lateral + 0.75*heading
    near_boundary = near[0] >= HEIGHT*0.80 and abs(lateral) > 0.55
    if near_boundary:
        # Containment takes priority over the curve's distant heading.
        error = lateral*0.8
    if abs(error) < 0.04:
        error = 0.0
    # Slow down progressively BEFORE increasing differential steering.
    # PWM values are duty-cycle commands, not measured wheel velocities.
    edge_only = result.get('mode','BOTH').startswith('EDGE') or near_boundary
    limit = 20 if edge_only else 25
    turn = max(-limit, min(limit, round(error*45)))
    base = 45 if edge_only else 80 - round(30*abs(turn)/25)
    return base+turn, base-turn, error


class SteeringRamp:
    """Bound steering changes; STOP and uncertainty always bypass the ramp."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_time = None
        self.base = self.turn = 0.0

    def apply(self, command, now):
        if command is None or command[:2] == (0, 0):
            self.reset()
            return (0, 0)
        dt = 0.05 if self.last_time is None else max(0, min(0.10, now-self.last_time))
        self.last_time = now
        target_base = (command[0]+command[1])/2.0
        target_turn = (command[0]-command[1])/2.0
        # Deceleration is immediate; acceleration and increasing steering
        # are gradual. Never retain steering opposite to a fresh command.
        self.base = min(target_base, self.base+120*dt)
        if self.turn*target_turn <= 0:
            self.turn = 0.0
        if abs(target_turn) <= abs(self.turn):
            self.turn = target_turn
        else:
            step = min(abs(target_turn)-abs(self.turn), 60*dt)
            self.turn += step if target_turn > 0 else -step
        self.turn = max(-self.base, min(self.base, self.turn))
        return round(self.base+self.turn), round(self.base-self.turn)

class MotionGuard:
    """Brief uncertain frames pause output; recovery never uses stale motion."""
    def __init__(self):
        self.lost_at = None

    def resolve(self, command, active, now, good_frames):
        if not active:
            self.lost_at = None
            return command, False
        # Fresh valid geometry wins over an old dropout deadline. Otherwise
        # a frame that has recovered exactly at the deadline falsely stops.
        if command is not None:
            self.lost_at = None
            return command, False
        if self.lost_at is None:
            self.lost_at = now
        if now-self.lost_at > 0.35:
            return None, False
        return (0,0,0.0), True


import os
import json

# Wheel directions and UNO watchdog verified on hardware. Manual START required.
ENABLE_DRIVE = True
SESSION_SECONDS = 12
TEST_RUN_SECONDS = 0

LIVE_HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>小车实时画面</title><style>
body{margin:0;background:#111827;color:#e5e7eb;font:16px system-ui;padding:20px}
main{max-width:1280px;margin:auto}h1{font-size:24px}p{line-height:1.6}
img{display:block;width:100%;height:auto;border:1px solid #374151;border-radius:10px}
.labels{display:flex}.labels span{width:50%;padding:12px 0;font-weight:600}
small{color:#9ca3af}button{padding:12px 22px;border:0;border-radius:6px;cursor:pointer;font:inherit}
#start{background:#16a34a;color:white}#stop{background:#dc2626;color:white;font-weight:bold}
button:disabled{opacity:.4;cursor:not-allowed}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
#status{font-weight:600}#notice{min-height:1.6em;color:#fcd34d}
</style><main><h1>小车实时画面</h1>
<p>左侧为摄像头原始画面；右侧为道路识别、运行状态和左右轮指令。</p>
<div class="controls"><button id="start" disabled onclick="control('start')">START 开始</button>
<button id="stop" onclick="control('stop')">STOP 停车</button><span id="status">正在连接…</span></div>
<p id="notice" role="status" aria-live="polite"></p><div class="labels"><span>原始画面</span><span>巡线识别</span></div>
<img id="video" src="/stream" alt="连接摄像头中">
<p>画面角落的数字持续增加，表示仍在更新。上方 START／STOP 可以控制小车；视频图像中的按钮仅在摄像头触摸屏上有效。每轮最多运行12秒。</p>
<button onclick="document.getElementById('video').src='/stream?t='+Date.now()">重新连接画面</button>
<p><small>电脑或手机需与摄像头连接同一 Wi-Fi。退出巡线应用后，视频会断开。</small></p></main><script>
const token="__CONTROL_TOKEN__";
let generation=0,canStart=false,busy=false;
const startButton=document.getElementById('start'),statusBox=document.getElementById('status'),notice=document.getElementById('notice');
async function updateStatus(){
 try{
  const r=await fetch('/status',{cache:'no-store',signal:AbortSignal.timeout(1500)});
  if(!r.ok)throw Error('状态连接失败');
  const s=await r.json();generation=s.generation;
  const fresh=s.age_ms<1000;
  canStart=fresh&&s.ready&&!s.active;
  startButton.disabled=!canStart||busy;
  statusBox.textContent=fresh?`${s.active?'运行中':'已停止'} · ${s.reason} · L=${s.left} R=${s.right}`:'状态更新中断';
 }catch(e){canStart=false;startButton.disabled=true;statusBox.textContent='连接中断，请检查摄像头和Wi-Fi';}
}
async function control(action){
 if(action==='start'&&(!canStart||busy))return;
 if(action==='start'){busy=true;startButton.disabled=true;}
 notice.textContent=action==='stop'?'正在发送停车请求…':'正在发送启动请求…';
 try{
  const r=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':token},
   body:JSON.stringify({action,generation}),signal:AbortSignal.timeout(2000)});
  const result=await r.json();
  if(!r.ok)throw Error(result.message||'请求失败');
  notice.textContent=action==='stop'?'停车请求已接收，以上方实际状态为准。':'启动请求已接收，以上方实际状态为准。';
 }catch(e){notice.textContent=e.message+'；请检查上方状态，必要时使用车上STOP或电源开关。';}
 finally{if(action==='start')busy=false;await updateStatus();}
}
updateStatus();setInterval(updateStatus,500);
</script></html>"""


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


def main():
    from maix import app, camera, display, image, touchscreen
    link = DriveLink()
    cam = None
    screen = None
    stream = None
    next_stream = 0
    stream_frames = 0
    active = False
    tracker = RoadTracker()
    guard = MotionGuard()
    steering = SteeringRamp()
    started = time.monotonic()
    next_control = next_log = next_status = 0
    consecutive = 0
    was_pressed = False
    pwm = (0, 0)
    reason = 'WAIT START' if ENABLE_DRIVE else 'PREVIEW - DRIVE DISABLED'
    saved = set()
    try:
        ack = link.send(0)
        if ack[1:4] != [0,0,0]:
            raise RuntimeError('UNO did not confirm stopped')
        cam = camera.Camera(WIDTH, HEIGHT, image.Format.FMT_BGR888)
        screen = display.Display()
        touch = touchscreen.TouchScreen()
        touch.clear()
        try:
            stream = LiveStream(port=8000)
            print('LIVE VIEW ready on port 8000', flush=True)
        except Exception as exc:
            print('LIVE VIEW unavailable:', str(exc), flush=True)
            if stream is not None:
                stream.stop()
            stream = None
        scale = min(screen.width()/WIDTH, screen.height()/HEIGHT)
        ox = (screen.width()-WIDTH*scale)/2
        oy = (screen.height()-HEIGHT*scale)/2
        print('LANE CONTROL READY drive_enabled=%s screen=%dx%d' %
              (ENABLE_DRIVE, screen.width(), screen.height()), flush=True)
        while not app.need_exit():
            now = time.monotonic()
            if TEST_RUN_SECONDS and now-started >= TEST_RUN_SECONDS:
                break
            frame = cam.read()
            bgr = image.image2cv(frame, ensure_bgr=True, copy=True)
            result = tracker.update(detect_road(bgr), time.monotonic())
            command = lane_command(result)
            consecutive = consecutive+1 if command else 0
            x,y,pressed = touch.read()
            tx,ty = (x-ox)/scale, (y-oy)/scale
            web_action = stream.control.take() if stream is not None else None
            touch_action = None
            if pressed and not was_pressed and 208 <= ty < HEIGHT:
                touch_action = 'stop' if 106 <= tx < 213 else 'exit' if 213 <= tx < WIDTH else 'start' if 0 <= tx < 106 else None
            # Both UI paths share one serialized UART owner and readiness check.
            if touch_action == 'exit':
                active=False; link.stop(); app.set_exit_flag(True); break
            if 'stop' in (web_action, touch_action):
                active=False; pwm=(0,0); link.send(0); reason='STOPPED'
                if stream is not None:
                    stream.control.stopped()
            elif 'start' in (web_action, touch_action):
                if ENABLE_DRIVE and consecutive>=5 and command is not None and not active:
                    link.send(1)
                    active=True; arm_started=time.monotonic(); next_control=0
                    reason='FOLLOW'
                elif not active:
                    reason='ROAD NOT READY' if ENABLE_DRIVE else 'PREVIEW - DRIVE DISABLED'
            was_pressed = pressed
            if not active and reason in ('WAIT START', 'ROAD NOT READY'):
                reason = 'WAIT START' if consecutive >= 5 else 'ROAD NOT READY'
            now=time.monotonic()
            command,holding = guard.resolve(command,active,now,consecutive)
            if active and (command is None or now-arm_started>=SESSION_SECONDS):
                reason='ROAD LOST' if command is None else 'TIME LIMIT - STOP'
                print('STOP EVENT',json.dumps({'reason':reason,'points':result['points'],'previous_pwm':pwm,'vision_valid':result['valid'],'mode':result.get('mode')}),flush=True)
                active=False; pwm=(0,0); link.send(0)
                if stream is not None:
                    stream.control.stopped()
            if not active:
                steering.reset()
            if active and now>=next_control:
                reason='VISION HOLD' if holding else 'FOLLOW '+result.get('mode','BOTH')
                pwm=steering.apply(command,now)
                link.send(2,*pwm)
                next_control=time.monotonic()+0.05
            elif not active and now>=next_status:
                ack=link.send(3)
                if ack[1] or ack[2] or ack[3]:
                    link.stop()
                    raise RuntimeError('Unexpected armed state')
                next_status=time.monotonic()+1.0
            if stream is not None:
                stream.control.update(active, ENABLE_DRIVE and consecutive>=5 and command is not None and not active,
                                      reason, pwm)
            out=annotate(bgr,result)
            cv2.rectangle(out,(0,0),(WIDTH,33),(0,0,0),-1)
            cv2.putText(out,reason,(3,12),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,255,255),1)
            offset='?' if result['offset'] is None else '%+.0f' % result['offset']
            cv2.putText(out,'offset=%s L=%d R=%d' % (offset,*pwm),(3,27),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1)
            for left,right,label,color in [(0,105,'START',(0,100,0) if ENABLE_DRIVE and consecutive >= 5 else (70,70,70)),(106,212,'STOP',(0,0,160)),(213,319,'EXIT',(90,50,30))]:
                cv2.rectangle(out,(left,208),(right,239),color,-1)
                cv2.putText(out,label,(left+14,230),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            shown=image.cv2image(out,bgr=True,copy=True)
            screen.show(shown)
            if stream is not None and time.monotonic() >= next_stream:
                try:
                    combined = np.hstack((bgr, out))
                    stream_frames += 1
                    cv2.putText(combined, 'LIVE %d' % stream_frames, (4, 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
                    stream.write(combined)
                    next_stream = time.monotonic()+0.2
                except Exception as exc:
                    print('LIVE VIEW stopped:', str(exc), flush=True)
                    stream.stop()
                    stream = None
            if os.environ.get('MAIXPY_DEBUG_IMAGES')=='1':
                elapsed=time.monotonic()-started
                for second in (2,8):
                    if elapsed>=second and second not in saved:
                        directory=os.path.join(os.environ['MAIXPY_SKILL_RUN_DIR'],'artifacts')
                        os.makedirs(directory,exist_ok=True)
                        for label,img in [('raw',frame),('control',shown)]:
                            path=os.path.join(directory,'%s-%d.jpg' % (label,second))
                            img.save(path); print('MAIXPY_SKILL_IMAGE',path,flush=True)
                        saved.add(second)
            if now>=next_log:
                print('CONTROL',json.dumps({'active':active,'reason':reason,'rows':len(result['points']),
                      'offset':result['offset'],'left':pwm[0],'right':pwm[1],'mode':result.get('mode'),'width_model':tracker.width_fit is not None}),flush=True)
                next_log=now+1
    except Exception as exc:
        print('CONTROL FAULT:',str(exc),flush=True)
        raise
    finally:
        link.close()
        if stream is not None:
            stream.stop()
        print('CONTROL EXIT | stop requested',flush=True)

if __name__=='__main__':
    main()
