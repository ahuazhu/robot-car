"""MaixCAM2 raw web preview only. No UART or motor-control commands."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
from maix import app, camera, display, image, touchscreen

PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MaixCAM2 视角预览</title><style>body{background:#111827;color:#eee;font:17px system-ui;max-width:960px;margin:20px auto;padding:0 12px}#box{position:relative}img{display:block;width:100%}#grid{position:absolute;inset:0;pointer-events:none;background:linear-gradient(90deg,transparent 49.8%,cyan 50%,transparent 50.2%),linear-gradient(transparent 49.8%,cyan 50%,transparent 50.2%)}button{font:inherit;margin:8px 5px 8px 0;padding:10px}</style><h2>MaixCAM2 视角预览</h2><p>只看原始画面，不启动巡线。参考线不代表道路识别结果。</p><div id="box"><img id="cam" src="/stream"><div id="grid"></div></div><button onclick="document.getElementById('cam').src='/stream?t='+Date.now()">重连画面</button><button onclick="let g=document.getElementById('grid');g.hidden=!g.hidden">中心参考线</button><p id="status"></p><script>setInterval(async()=>{try{let s=await(await fetch('/status',{cache:'no-store'})).json();document.getElementById('status').textContent='帧数 '+s.frame+' · 原始画面 320×240'}catch(e){}},1000)</script></html>'''
condition=threading.Condition()
latest=None
sequence=0
closed=False
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        route=self.path.split('?',1)[0]
        if route=='/':data=PAGE.encode();kind='text/html; charset=utf-8'
        elif route=='/status':
            with condition:data=json.dumps({'frame':sequence,'preview_only':True}).encode()
            kind='application/json'
        elif route=='/capture':
            with condition:data=latest
            if data is None:self.send_error(503);return
            kind='image/jpeg'
        elif route=='/stream':
            self.connection.settimeout(2)
            self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace;boundary=frame');self.send_header('Cache-Control','no-store');self.end_headers()
            last=-1
            try:
                while not closed:
                    with condition:
                        condition.wait_for(lambda:closed or sequence!=last,timeout=1)
                        if closed:break
                        if latest is None or sequence==last:continue
                        data=latest;last=sequence
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(data)).encode()+b'\r\n\r\n'+data+b'\r\n');self.wfile.flush()
            except OSError:pass
            return
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)

def main():
    global latest,sequence,closed
    server=ThreadingHTTPServer(('0.0.0.0',8000),Handler);server.daemon_threads=True
    worker=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.1},daemon=True);worker.start()
    cam=None;screen=None
    try:
        cam=camera.Camera(320,240,image.Format.FMT_BGR888)
        screen=display.Display();touch=touchscreen.TouchScreen();touch.clear()
        scale=min(screen.width()/320,screen.height()/240)
        ox=(screen.width()-320*scale)/2;oy=(screen.height()-240*scale)/2
        next_frame=0;next_log=0
        print('RAW PREVIEW READY port=8000 | no UART / no motor commands',flush=True)
        while not app.need_exit():
            frame=cam.read();bgr=image.image2cv(frame,ensure_bgr=True,copy=True)
            x,y,pressed=touch.read()
            if pressed and 230<=(x-ox)/scale<320 and 208<=(y-oy)/scale<240:break
            now=time.monotonic()
            if now>=next_frame:
                ok,jpg=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,80])
                if ok:
                    with condition:latest=jpg.tobytes();sequence+=1;condition.notify_all()
                next_frame=now+.15
            cv2.rectangle(bgr,(230,208),(319,239),(60,45,30),-1)
            cv2.putText(bgr,'EXIT',(249,231),0,.55,(255,255,255),1)
            screen.show(image.cv2image(bgr,bgr=True,copy=True))
            if now>=next_log:
                print('PREVIEW frames=%d'%sequence,flush=True);next_log=now+10
    finally:
        with condition:closed=True;condition.notify_all()
        server.shutdown();server.server_close();worker.join(timeout=1)
        if cam is not None:cam.close()
        print('PREVIEW EXIT',flush=True)
if __name__=='__main__':main()
