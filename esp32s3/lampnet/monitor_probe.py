"""Read-only USB probe: M=model, S=camera status, Q=network. Never ARM."""
import argparse,json,re,time
from pathlib import Path
import serial

def first_json(text):
    # Two ESP32 tasks can finish Serial.print calls on the same UART line.
    # Decode the first complete object; preserve the raw line in serial.log.
    return json.JSONDecoder().raw_decode(text)[0]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seconds',type=float,default=90);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    s=serial.Serial();s.port=a.port;s.baudrate=115200;s.timeout=.2;s.dtr=False;s.rts=False;s.open()
    models=[];golden=[];cameras=[];networks=[];errors=[];began=time.monotonic();due=began
    try:
        with (a.out/'serial.log').open('w') as log:
            while time.monotonic()-began<a.seconds:
                now=time.monotonic()
                if now>=due:s.write(b'M\nS\nQ\n');due=now+2
                line=s.readline().decode(errors='replace').strip()
                if not line:continue
                log.write(line+'\n');log.flush()
                if line.startswith('LAMP_MODEL {'):
                    try:d=first_json(line[len('LAMP_MODEL '):])
                    except json.JSONDecodeError:continue
                    d['host_t']=now-began;models.append(d)
                    (a.out/'latest-model.json').write_text(json.dumps(d,indent=2)+'\n')
                elif line.startswith('LAMP_GOLDEN {'):
                    try:golden.append(first_json(line[len('LAMP_GOLDEN '):]))
                    except json.JSONDecodeError:continue
                elif line.startswith('WIFI '):
                    try:networks.append(first_json(line[5:]))
                    except json.JSONDecodeError:continue
                elif line.startswith('STATUS '):cameras.append(dict(t=now-began,**{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}))
                elif 'ERROR' in line or 'Guru Meditation' in line or 'task_wdt' in line:errors.append(line)
    finally:s.close()
    unique={m['inferences']:m for m in models if m.get('inferences',0)>0}
    report=dict(models=list(unique.values()),golden=(models[-1].get("golden",golden) if models else golden),last_model=models[-1] if models else None,camera_samples=cameras,network=networks[-1] if networks else None,
                errors=errors,duration_s=time.monotonic()-began,controls_start=False,motor_command_sent=False,physical_start_tested=False,
                camera_advanced=len(cameras)>1 and cameras[-1]['frames']>cameras[0]['frames'],
                selftest_ok=any(m.get('selftest_ok') for m in models))
    (a.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['models','camera_samples','golden']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
