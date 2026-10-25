"""Bounded manual START/STOP lane test. Default: stopped until START.
A dedicated test may set TARGET_AUTO_SECONDS (maximum 3 seconds).
"""
import os,time,json,math
import cv2
import numpy as np
from tracker import TargetTracker,target_command,motion_from_pwm,render_target
from drive_link import DriveLink
from controller import CommandRamp
SESSION_SECONDS=55.
from preview_transport import LiveStream

def main():
    from maix import app,camera,display,image,touchscreen
    calibration_path='/maixapp/apps/robot_duckietown_test/calibration.json'
    with open(calibration_path) as f:cal=json.load(f)
    if not cal.get('geometry_confirmed',False):raise RuntimeError('Calibration not confirmed')
    cv2.setNumThreads(1)
    auto_seconds=max(0,min(3,float(os.environ.get('TARGET_AUTO_SECONDS','0'))))
    duration=float(os.environ.get('TARGET_TEST_SECONDS','0'))
    preview_only=os.environ.get('TARGET_PREVIEW_ONLY')=='1'
    link=DriveLink();stream=None;cam=None;active=False;pwm=(0,0)
    ramp=CommandRamp();consecutive=0;lost_at=None;auto_used=False;was_pressed=False
    arm_started=next_control=next_log=next_status=next_stream=0
    began=time.monotonic();frames=0;saved=set();reason='WAIT START';stats={};observed_frames=0
    try:
        link_ok=False
        try:
            print('STOP_ACK',link.send(0),flush=True);link_ok=True
        except RuntimeError as exc:
            reason='UNO NOT CONNECTED';print(reason,str(exc),flush=True)
        vision=TargetTracker(cal);cam=camera.Camera(320,240,image.Format.FMT_BGR888)
        screen=display.Display();touch=touchscreen.TouchScreen();touch.clear()
        stream=LiveStream(8000)
        scale=min(screen.width()/320,screen.height()/240)
        ox=(screen.width()-320*scale)/2;oy=(screen.height()-240*scale)/2
        print('TARGET FOLLOW READY | manual START | max session 55 seconds',flush=True)
        began=time.monotonic() # Exclude camera startup from test/debug elapsed time.
        while not app.need_exit():
            now=time.monotonic()
            if duration and now-began>=duration:break
            frame=cam.read();bgr=image.image2cv(frame,ensure_bgr=True,copy=True)
            captured=time.monotonic()
            v,omega=motion_from_pwm(pwm) if active else (0.,0.)
            result,details=vision.update(bgr,captured,v,omega);frames+=1
            stats[result['mode']]=stats.get(result['mode'],0)+1
            observed_frames+=int(result['observed'])
            command=target_command(result)
            detection_ms=(time.monotonic()-captured)*1000
            if detection_ms>200:command=None
            consecutive=consecutive+1 if command is not None and result['observed'] and result['confidence']>=.65 else 0
            x,y,pressed=touch.read();tx=(x-ox)/scale;ty=(y-oy)/scale
            action=stream.control.take();touch_action=None
            if pressed and not was_pressed and 209<=ty<240:
                touch_action='start' if 0<=tx<106 else 'stop' if 106<=tx<213 else 'exit' if 213<=tx<320 else None
            was_pressed=pressed
            if touch_action=='exit':break
            if 'stop' in (action,touch_action):
                link.send(0);active=False;pwm=(0,0);reason='STOPPED';stream.control.stopped()
                auto_used=True
            else:
                wants_start='start' in (action,touch_action)
                if auto_seconds and not auto_used and link_ok and consecutive>=8 and now-began>2:
                    wants_start=True;auto_used=True
                if wants_start and not active and not preview_only:
                    if link_ok and consecutive>=5 and command is not None:
                        link.send(1);active=True;arm_started=time.monotonic();next_control=0
                        lost_at=None;ramp.reset();reason=result['mode']
                        print('START',json.dumps({'pose':result,'auto':bool(auto_seconds)}),flush=True)
                    else:reason='ROAD NOT READY' if link_ok else 'UNO NOT CONNECTED'
            now=time.monotonic()
            if active:
                if command is None:
                    if lost_at is None:lost_at=now
                    pwm=(0,0);ramp.reset();reason='VISION PAUSE'
                else:lost_at=None
                limit=auto_seconds if auto_seconds else SESSION_SECONDS
                if now-arm_started>=limit or (lost_at is not None and now-lost_at>=.35):
                    reason='TIME LIMIT - STOP' if now-arm_started>=limit else ('VISION TOO SLOW' if detection_ms>200 else result.get('invalid_reason') or 'TARGET LOST')
                    link.send(0);active=False;pwm=(0,0);stream.control.stopped()
                    print('STOP EVENT',reason,flush=True)
                    if auto_seconds:break
                elif now>=next_control:
                    pwm=ramp.apply(command,now) if command is not None else (0,0)
                    link.send(2,*pwm);next_control=time.monotonic()+.05
                    if command is not None:reason=result['mode']
            else:
                pwm=(0,0);ramp.reset()
                if now>=next_status:
                    try:
                        ack=link.send(3 if link_ok else 0)
                        if ack[1:4]!=[0,0,0]:
                            link.stop();raise RuntimeError('Unexpected armed state')
                        if not link_ok:reason='WAIT START'
                        link_ok=True
                    except RuntimeError as exc:
                        link_ok=False;reason='UNO NOT CONNECTED'
                        print('UNO STATUS',str(exc),flush=True)
                    next_status=time.monotonic()+1
                if reason in ('WAIT START','ROAD NOT READY'):reason='WAIT START' if consecutive>=5 else 'ROAD NOT READY'
            ready=link_ok and consecutive>=5 and command is not None and not active and not preview_only
            stream.control.update(active,ready,reason,pwm)
            out,bird=render_target(bgr,result,details,cal)
            cv2.rectangle(out,(0,0),(319,37),(0,0,0),-1)
            cv2.putText(out,reason,(4,14),0,.43,(0,255,255),1)
            cv2.putText(out,'L=%d R=%d frame=%d'%(*pwm,frames),(4,30),0,.39,(255,255,255),1)
            for a,b,label,color in [(0,105,'START',(0,100,0) if ready else (55,55,55)),(106,212,'STOP',(0,0,170)),(213,319,'EXIT',(90,50,30))]:
                cv2.rectangle(out,(a,209),(b,239),color,-1);cv2.putText(out,label,(a+14,231),0,.48,(255,255,255),1)
            screen.show(image.cv2image(out,bgr=True,copy=True))
            now=time.monotonic()
            if now>=next_stream:stream.write(np.hstack((bgr,out,bird)));next_stream=now+.2
            if now>=next_log:
                print('CONTROL',json.dumps({'active':active,'reason':reason,'target':{k:v for k,v in result.items() if k!='path'},'detection_ms':round(detection_ms,1),'loop_processing_ms':round((time.monotonic()-captured)*1000,1),'pwm':pwm,'frame':frames,'uno_connected':link_ok}),flush=True);next_log=now+.5
            if os.environ.get('MAIXPY_DEBUG_IMAGES')=='1':
                for second in (2,4,6):
                    if now-began>=second and second not in saved:
                        folder=os.path.join(os.environ['MAIXPY_SKILL_RUN_DIR'],'artifacts');os.makedirs(folder,exist_ok=True)
                        for label,picture in [('raw',bgr),('control',out)]:
                            path=os.path.join(folder,'%s-%d.jpg'%(label,second));cv2.imwrite(path,picture);print('MAIXPY_SKILL_IMAGE',path,flush=True)
                        saved.add(second)
    except Exception as exc:
        print('CONTROL FAULT',repr(exc),flush=True);raise
    finally:
        print('TARGET_SUMMARY',json.dumps(dict(frames=frames,observed=observed_frames,modes=stats)),flush=True)
        print('FINAL_STOP_ACK',link.stop(),flush=True)
        if stream:stream.stop()
        link.port.close()
        print('TARGET FOLLOW EXIT',flush=True)
if __name__=='__main__':main()
