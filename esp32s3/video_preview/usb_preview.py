"""Local USB preview relay. Requires pyserial. No UNO/motor commands."""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import serial

parser=argparse.ArgumentParser()
parser.add_argument('--serial',required=True)
parser.add_argument('--port',type=int,default=8123)
args=parser.parse_args()
link=serial.Serial(args.serial,115200,timeout=.25,write_timeout=2)
time.sleep(3)
link.reset_input_buffer()
condition=threading.Condition()
latest=None
sequence=0
failure=''
stopping=threading.Event()
page=Path(__file__).with_name('video_preview.ino').read_text().split('R"HTML(',1)[1].split(')HTML"',1)[0]
page=page.replace("'http://'+location.hostname+':81/stream?t='","'/stream?t='")
page=page.replace('320 × 240 ·','USB预览 · 320 × 240 ·')

def capture():
    link.write(b'J')
    deadline=time.monotonic()+5
    size=None
    while time.monotonic()<deadline:
        line=link.readline()
        if line.startswith(b'JPEG '):
            size=int(line.split()[1]);break
        if line.startswith(b'ERROR'):raise RuntimeError(line.decode(errors='replace').strip())
    if size is None or not 100<size<150000:raise RuntimeError('No valid JPEG header')
    data=bytearray()
    deadline=time.monotonic()+max(5,size/8000)
    while len(data)<size and time.monotonic()<deadline:
        data.extend(link.read(size-len(data)))
    data=bytes(data)
    if len(data)!=size or not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):
        raise RuntimeError('Incomplete JPEG')
    return data

def producer():
    global latest,sequence,failure
    while not stopping.is_set():
        try:
            data=capture()
            with condition:
                latest=data;sequence+=1;failure='';condition.notify_all()
            if sequence==1:print('First JPEG received: %d bytes'%len(data),flush=True)
        except Exception as exc:
            with condition:failure=str(exc);condition.notify_all()
            print('Capture error:',exc,flush=True)
            stopping.wait(.5);link.reset_input_buffer()
        stopping.wait(.1)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*a):pass
    def do_GET(self):
        route=self.path.split('?',1)[0]
        if route=='/':
            data=page.encode();kind='text/html; charset=utf-8'
        elif route=='/status':
            with condition:data=json.dumps({'frames':sequence,'error':failure}).encode()
            kind='application/json'
        elif route=='/capture':
            with condition:data=latest
            if not data:self.send_error(503,'Camera warming up');return
            kind='image/jpeg'
        elif route=='/stream':
            self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace;boundary=frame');self.send_header('Cache-Control','no-store');self.end_headers()
            last=0
            try:
                while not stopping.is_set():
                    with condition:
                        condition.wait_for(lambda:sequence!=last or stopping.is_set(),timeout=5)
                        if sequence==last:continue
                        data=latest;last=sequence
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(data)).encode()+b'\r\n\r\n'+data+b'\r\n')
                    self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError,OSError):pass
            return
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)

server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
worker=threading.Thread(target=producer,daemon=True);worker.start()
print('ESP32 USB preview http://127.0.0.1:%d/'%args.port,flush=True)
try:server.serve_forever()
except KeyboardInterrupt:pass
finally:
    stopping.set()
    with condition:condition.notify_all()
    server.server_close();worker.join(timeout=8);link.close()
