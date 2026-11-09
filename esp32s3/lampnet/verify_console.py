"""USB model preview; optional isolated T/U detector test cannot grant a start."""
import argparse,io,json,re,signal,threading,time,queue
from datetime import datetime
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import serial
from PIL import Image

PAGE=Path(__file__).with_name('verify_console.html')

class Recorder:
    def __init__(self,port,out,diagnostic=False):
        self.port,self.out=port,out;self.out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
        self.lock=threading.Lock();self.stop=threading.Event();self.model={};self.jpeg=None;self.frame=0;self.error='';self.began=time.monotonic();self.stream=None
        self.events=(out/'events.jsonl').open('a');self.serial_log=(out/'serial.log').open('a');self.buffer=b''
        self.commands=queue.Queue();self.config_error='';self.config_result='';self.jpeg_at=0
        self.diagnostic=diagnostic;self.test_started=False;self.test_request=threading.Event()
        if diagnostic:self.test_request.set()
    def event(self,kind,**values):
        with self.lock:self.events.write(json.dumps(dict(kind=kind,t=time.monotonic()-self.began,**values),ensure_ascii=False)+'\n');self.events.flush()
    def lines(self,data):
        for line in data.decode(errors='replace').splitlines():
            self.serial_log.write(line+'\n');self.serial_log.flush()
            if line.startswith('ROI ERROR '):self.config_error=line[10:];self.event('configuration_error',reason=self.config_error)
            elif line.startswith('ROI OK '):self.config_error='';self.config_result='设置已保存到 ESP32';self.event('configuration_saved',configuration=json.loads(line[7:]))
            match=re.search(r'LAMP_MODEL (\{.*)',line)
            if match:
                try:model=json.JSONDecoder().raw_decode(match.group(1))[0]
                except json.JSONDecodeError:continue
                with self.lock:self.model=model;self.error=''
                self.event('model',state=model)
    def model_status(self):
        self.stream.write(b'M\n');end=time.monotonic()+.65;data=b''
        while time.monotonic()<end:
            chunk=self.stream.read(4096);data+=chunk
            if data.endswith(b'\n') and b'LAMP_MODEL {' in data:break
        self.lines(data)
    def image(self):
        self.stream.write(b'J\n');end=time.monotonic()+2.0;data=b'';match=None
        while time.monotonic()<end:
            data+=self.stream.read(4096);match=re.search(rb'JPEG (\d+)\r?\n',data)
            if match:break
        if not match:self.lines(data);return False
        self.lines(data[:match.start()]);size=int(match.group(1));start=match.end()
        if size<=0 or size>150000:raise ValueError('Unexpected JPEG size')
        end=time.monotonic()+max(3.0,size*10/115200+1.0)
        while len(data)-start<size and time.monotonic()<end:data+=self.stream.read(min(4096,size-(len(data)-start)))
        image=data[start:start+size]
        if len(image)!=size or b'LAMP_MODEL {' in image or not image.startswith(b'\xff\xd8') or not image.endswith(b'\xff\xd9'):
            self.event('dropped_frame',reason='serial framing or concurrent diagnostic line');return False
        try:
            decoded=Image.open(io.BytesIO(image));decoded.load()
            if decoded.size!=(320,240):raise ValueError('Unexpected preview dimensions')
        except Exception as e:self.event('dropped_frame',reason=str(e));return False
        with self.lock:self.jpeg=image;self.frame+=1;frame=self.frame;self.jpeg_at=time.monotonic()
        filename='frames/%06d.jpg'%frame;(self.out/filename).write_bytes(image);self.event('frame',file=filename,jpeg_bytes=len(image))
        self.lines(data[start+size:]);return True
    def run(self):
        try:
            s=serial.Serial();s.port=self.port;s.baudrate=115200;s.timeout=.05;s.dtr=False;s.rts=False;s.open();self.stream=s;time.sleep(.7)
            self.event('recording_started',manual_configuration_available=True,mode='ISOLATED_MODEL_OFF_TEST' if self.diagnostic else 'MODEL_LOCATION_ONLY',commands=['M','J','T','U'] if self.diagnostic else ['M','J'],controls_start=False)
            while not self.stop.is_set():
                start=time.monotonic();self.model_status()
                try:command=self.commands.get_nowait()
                except queue.Empty:command=None
                if command is not None:
                    self.test_request.clear();self.config_error='';self.config_result='';self.stream.write(b'R'+json.dumps(command,separators=(',',':')).encode()+b'\n');self.event('configuration_requested',configuration=command)
                if self.test_request.is_set() and self.model.get('selftest_ok'):
                    diagnostic=self.model.get('diagnostic',{})
                    if 'native_armed' not in diagnostic:raise RuntimeError('Firmware has no isolated test mode')
                    if diagnostic['native_armed']:raise RuntimeError('Maix ARM is active; isolated lamp test refused')
                    self.stream.write(b'T\n');self.test_started=True;self.test_request.clear();self.event('isolated_test_requested',controls_start=False)
                self.image();self.stop.wait(max(0,.8-(time.monotonic()-start)))
        except Exception as e:
            with self.lock:self.error=str(e)
            self.event('error',message=str(e))
        finally:
            if self.stream:
                if self.test_started:
                    self.stream.write(b'U\n');self.stream.flush();self.event('isolated_test_cancelled',controls_start=False)
                self.stream.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--serial',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--port',type=int,default=8130);ap.add_argument('--diagnostic',action='store_true',help='Separate detector only; never creates a Maix start session');a=ap.parse_args();rec=Recorder(a.serial,a.out,a.diagnostic)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,content,mime):
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
        def do_GET(self):
            if self.path=='/':return self.send(PAGE.read_text().encode(),'text/html; charset=utf-8')
            if self.path.startswith('/status'):
                with rec.lock:d=dict(model=rec.model,frame=rec.frame,error=rec.error,recording=True,controls_start=False,frame_age_ms=int((time.monotonic()-rec.jpeg_at)*1000) if rec.jpeg_at else 999999,config_error=rec.config_error,config_result=rec.config_result)
                return self.send(json.dumps(d).encode(),'application/json')
            if self.path.startswith('/frame'):
                with rec.lock:image=rec.jpeg
                if image:return self.send(image,'image/jpeg')
                self.send_error(503);return
            self.send_error(404)
        def do_POST(self):
            origin=self.headers.get('Origin')
            if origin and origin!='http://127.0.0.1:%d'%a.port:self.send_error(403);return
            if self.path=='/roi':
                try:
                    size=int(self.headers.get('Content-Length','0'));assert 0<size<=160
                    body=json.loads(self.rfile.read(size));assert type(body.get('manual')) is bool
                    if body['manual']:
                        r=body['roi'];assert len(r)==4 and all(type(n) is int for n in r)
                        x,y,w,h=r;assert x>=0 and y>=0 and w>=16 and h>=16 and x+w<=320 and y+h<=240
                        body=dict(manual=True,roi=r)
                    else:body=dict(manual=False)
                except Exception:self.send_error(400);return
                if rec.model.get('diagnostic',{}).get('native_armed'):
                    content=b'{"error":"Maix is armed; STOP before changing the lamp area"}';self.send_response(409);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(content);return
                rec.commands.put(body);return self.send(b'{"queued":true}','application/json')
            if self.path=='/test':
                rec.test_request.set();rec.event('isolated_test_queued',controls_start=False);return self.send(b'{"ok":true}','application/json')
            if self.path!='/marker':self.send_error(404);return
            if int(self.headers.get('Content-Length','0'))>128:self.send_error(413);return
            try:d=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))));label=d['label'];assert label in ['ON','OFF','MOVED']
            except Exception:self.send_error(400);return
            rec.event('marker',label=label);self.send(b'{"ok":true}','application/json')
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);worker=threading.Thread(target=rec.run);worker.start()
    def shutdown(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
    print('MODEL VERIFY http://127.0.0.1:%d/ recording=%s'%(a.port,a.out),flush=True)
    try:server.serve_forever()
    finally:
        rec.stop.set();worker.join(timeout=5);server.server_close();rec.events.close();rec.serial_log.close()
if __name__=='__main__':main()
