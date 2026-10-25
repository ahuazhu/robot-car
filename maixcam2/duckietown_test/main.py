"""Duckietown on-device perception test. NO UART or motor commands.
Run through build.py to produce standalone.py for MaixVision/helper.
"""
import os
import time
import json
import cv2
import numpy as np
from vision import LaneVision, annotate, bird_view
from preview_transport import LiveStream


def main():
    from maix import app, camera, display, image, touchscreen
    duration=float(os.environ.get('DUCK_TEST_SECONDS','0'))
    config_path=os.environ.get('DUCK_CALIBRATION','/maixapp/apps/robot_duckietown_test/calibration.json')
    calibration=None
    if os.path.exists(config_path):
        with open(config_path) as f:loaded_calibration=json.load(f)
        if loaded_calibration.get('geometry_confirmed',True):
            calibration=loaded_calibration
        else:
            print('CALIBRATION SUSPENDED: measured rectangular reference required',flush=True)
    cv2.setNumThreads(1)
    vision=LaneVision(calibration)
    cam=camera.Camera(320,240,image.Format.FMT_BGR888)
    screen=display.Display()
    touch=touchscreen.TouchScreen();touch.clear()
    stream=None; saved=set(); timings=[]; frames=0
    pose_valid=0; pose_sum=np.zeros(2);pose_sum_sq=np.zeros(2)
    started=time.monotonic();next_log=next_stream=0
    scale=min(screen.width()/320,screen.height()/240)
    ox=(screen.width()-320*scale)/2;oy=(screen.height()-240*scale)/2
    try:
        stream=LiveStream(8000)
        print('DUCKIETOWN upstream c5ae3a2 | NO UART | NO MOTOR COMMANDS',flush=True)
        while not app.need_exit():
            if duration and time.monotonic()-started>=duration:break
            frame=cam.read()
            bgr=image.image2cv(frame,ensure_bgr=True,copy=True)
            begin=time.monotonic();result=vision.detect(bgr)
            timings.append((time.monotonic()-begin)*1000);timings=timings[-120:]
            frames+=1
            pose=result.get('pose')
            if pose and pose['valid']:
                values=np.array([pose['d_m'],pose['phi_rad']])
                pose_valid+=1;pose_sum+=values;pose_sum_sq+=values*values
            out=annotate(bgr,result)
            cv2.putText(out,'frame %d'%frames,(4,205),0,.36,(255,255,0),1)
            cv2.rectangle(out,(230,209),(319,239),(80,50,20),-1)
            cv2.putText(out,'EXIT',(248,231),0,.5,(255,255,255),1)
            x,y,pressed=touch.read()
            if pressed and (x-ox)/scale>=230 and (y-oy)/scale>=209:
                app.set_exit_flag(True);break
            screen.show(image.cv2image(out,bgr=True,copy=True))
            now=time.monotonic()
            stream.control.update(False,False,result['status'],(0,0))
            stream.control.take() # START disabled; STOP remains a no-op in perception-only mode.
            if now>=next_stream:
                stream.write(np.hstack((bgr,out,bird_view(result))));next_stream=now+.2
            if now>=next_log:
                print('DUCK',json.dumps(dict(frame=frames,counts=result['counts'],
                      detector_ms=round(float(np.median(timings)),2),
                      fps=round(frames/max(.001,now-started),1),
                      calibrated=result['calibrated'],pose=result.get('pose'),motors=False)),flush=True)
                next_log=now+1
            if os.environ.get('MAIXPY_DEBUG_IMAGES')=='1':
                for second in (2,6):
                    if now-started>=second and second not in saved:
                        directory=os.path.join(os.environ['MAIXPY_SKILL_RUN_DIR'],'artifacts')
                        os.makedirs(directory,exist_ok=True)
                        for label,im in [('raw',bgr),('segments',out),('ground',bird_view(result))]:
                            path=os.path.join(directory,'%s-%d.jpg'%(label,second))
                            cv2.imwrite(path,im);print('MAIXPY_SKILL_IMAGE',path,flush=True)
                        with open(os.path.join(directory,'segments-%d.json'%second),'w') as f:json.dump(result,f)
                        saved.add(second)
    finally:
        if stream is not None:stream.stop()
        mean=pose_sum/max(1,pose_valid)
        std=np.sqrt(np.maximum(0,pose_sum_sq/max(1,pose_valid)-mean*mean))
        print('POSE_SUMMARY',json.dumps(dict(frames=frames,valid=pose_valid,valid_fraction=pose_valid/max(1,frames),mean_d_m=float(mean[0]),std_d_m=float(std[0]),mean_phi_rad=float(mean[1]),std_phi_rad=float(std[1]))),flush=True)
        print('DUCK TEST EXIT | NO MOTION SENT',flush=True)

if __name__=='__main__':main()
