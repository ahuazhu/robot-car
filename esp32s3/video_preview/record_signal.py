"""Record ESP32 JPEGs and sensor telemetry while Maix remains STOPPED.

Only the ESP32 detector is armed. Never sends Maix ARM/START or motor commands.
JPEGs and state requests have separate timestamps; they are not exact RGB565
sensor-frame pairs. Private pairing keys and session nonces are never recorded.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.request import Request, ProxyHandler, build_opener
from zoneinfo import ZoneInfo


class Recorder:
    def __init__(self, folder, pair_file, *, esp_url, maix_url):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False,mode=0o700)
        (self.folder/'frames').mkdir(mode=0o700)
        self.key=json.loads(Path(pair_file).read_text())['key']
        self.esp=esp_url.rstrip('/');self.control=self.esp+':81';self.maix=maix_url.rstrip('/')
        self.nonce=secrets.token_hex(16);self.boot='';self.started=time.monotonic()
        self.stopping=threading.Event();self.lock=threading.Lock();self.latest={};self.failure=None;self.confirmed_at=None
        self.events=(self.folder/'events.jsonl').open('w');self.phase=None

    def log(self,kind,**data):
        with self.lock:
            self.events.write(json.dumps(dict(kind=kind,t=round(time.monotonic()-self.started,6),**data),ensure_ascii=False)+'\n');self.events.flush()

    @staticmethod
    def request(url, *, body=None, headers=None, timeout=1):
        request=Request(url,data=None if body is None else json.dumps(body).encode(),headers=headers or {})
        with build_opener(ProxyHandler({})).open(request,timeout=timeout) as response:return response.read(150001)

    def get(self,url):return json.loads(self.request(url))

    def command(self,op):
        return json.loads(self.request(self.control+'/signal',body=dict(op=op,session=self.nonce,boot=self.boot),headers={'Content-Type':'application/json','X-Signal-Key':self.key}))

    def ensure_stopped(self):
        s=self.get(self.maix+'/status');p=s.get('readiness',{}).get('start_light',{})
        self.log('maix',active=s['active'],armed=p.get('armed'),outputs=[s['left'],s['right']])
        if s['active'] or p.get('armed') or s['left'] or s['right']:
            raise RuntimeError('Maix must stay STOPPED and unarmed during recording')

    def heartbeat(self):
        try:
            while not self.stopping.is_set():
                began=time.monotonic();s=self.command('poll')
                self.log('signal',request_ms=round((time.monotonic()-began)*1000,2),state={k:v for k,v in s.items() if k!='session'})
                self.latest=s
                if s['phase']!=self.phase:
                    self.phase=s['phase'];print('PHASE',self.phase,'target',s.get('target_roi'),'contrast',s.get('contrast'),'baseline',s.get('baseline_contrast'),flush=True)
                if s.get('confirmed') and self.confirmed_at is None:self.confirmed_at=time.monotonic()
                self.stopping.wait(max(0,.1-(time.monotonic()-began)))
        except Exception as exc:
            self.failure=type(exc).__name__;self.log('signal_error',error=self.failure);self.stopping.set()

    def run(self,duration=90,fps=5):
        worker=None;count=0;next_check=0
        try:
            self.ensure_stopped();status=self.get(self.control+'/status');s=status['signal']
            if s.get('armed'):raise RuntimeError('ESP32 has an existing session; STOP it before recording')
            self.boot=s['boot'];self.command('arm')
            self.log('recording_started',firmware=status['firmware'],mode='ESP32_SENSOR_ONLY',image_format='cached JPEG, not raw RGB565',fps=fps)
            worker=threading.Thread(target=self.heartbeat,name='record-signal-heartbeat',daemon=True);worker.start()
            print('RECORDING',self.folder.resolve(),flush=True)
            end=time.monotonic()+duration
            while not self.stopping.is_set() and time.monotonic()<end:
                began=time.monotonic()
                if began>=next_check:self.ensure_stopped();next_check=began+.5
                try:
                    before=self.latest.get('sequence');pixels=self.request(self.esp+'/capture',timeout=1.2)
                    if not pixels.startswith(b'\xff\xd8') or not pixels.endswith(b'\xff\xd9'):raise ValueError('Invalid JPEG')
                    name='frames/%06d.jpg'%count;(self.folder/name).write_bytes(pixels)
                    self.log('frame',file=name,request_started=round(began-self.started,6),sequence_before=before,sequence_after=self.latest.get('sequence'))
                    count+=1
                except Exception as exc:self.log('frame_error',error=type(exc).__name__)
                if (self.folder/'stop.request').exists():break
                if self.confirmed_at is not None and time.monotonic()-self.confirmed_at>=2:break
                self.stopping.wait(max(0,1/fps-(time.monotonic()-began)))
        except Exception as exc:
            self.failure=type(exc).__name__;self.log("recording_error",error=self.failure);raise
        finally:
            self.stopping.set()
            if worker:worker.join(timeout=2)
            if self.boot:
                try:self.command('cancel')
                except Exception:self.log('cancel_deferred_to_lease')
            summary=dict(frames=count,confirmed=self.confirmed_at is not None,last_phase=self.phase,error=self.failure,duration_s=round(time.monotonic()-self.started,2),mode='ESP32_SENSOR_ONLY',motor_command_sent=False)
            (self.folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
            self.events.close();print('SUMMARY',json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--pair-file',type=Path,default=Path('.maixpy/start_signal.local.json'))
    parser.add_argument('--esp-url',default='http://10.73.124.214')
    parser.add_argument('--maix-url',default='http://10.73.124.209:8000')
    parser.add_argument('--duration',type=float,default=90)
    args=parser.parse_args()
    if not 1<=args.duration<=180:parser.error('duration must be 1..180 seconds')
    folder=args.output or Path('.maixpy/lamp-recordings')/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    recorder=Recorder(folder,args.pair_file,esp_url=args.esp_url,maix_url=args.maix_url)
    try:recorder.run(args.duration)
    except KeyboardInterrupt:pass


if __name__=='__main__':main()
