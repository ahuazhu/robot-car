"""Parametric curve following. Preview is the default; current-pose calibration required."""
import os,time,json
from pathlib import Path
import cv2
import numpy as np
from tracker import TargetTracker,render_target
from drive_link import DriveLink
from motor_channel import MotorChannel
from controller import (CommandRamp,can_enable_drive,compensate_pwm,RoadSupervisor,
                        motion_from_pwm,assess_vision_command,startup_blockers,MAX_MOTOR_PWM)
from preview_transport import LiveStream
from steering_control import SteeringControl
from road_policy import RoadPolicy
from road_recovery import RoadSearch
from start_light import StartLight
from lap_counter import LapMarkerDetector,LapCounter
from native_geometry import native_fit_status
from crosswalk import CrosswalkPolicy,CrosswalkDetector,CrosswalkTask,draw_crosswalk_overlay


class PreviewLink:
    """No serial port is opened during geometry preview."""
    @property
    def port(self):return self
    def send(self,op,left=0,right=0):
        if op not in (0,3) or left or right:raise RuntimeError('Preview cannot drive')
        return [0,0,0,0,0]
    def stop(self):return 'PREVIEW: no UART opened'
    def close(self):pass

def main():
    from maix import app,camera,display,image,touchscreen
    calibration_path=Path(__file__).with_name('calibration.json')
    with open(calibration_path) as f:cal=json.load(f)
    runtime_path=Path(__file__).with_name('runtime.json')
    runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    road_cal=dict(cal,road_model=runtime.get('road_model',{}),control_frame=runtime.get('control_frame',{}))
    policy=RoadPolicy.from_config(road_cal)
    control_frame=runtime.get('control_frame',{})
    crosswalk_policy=CrosswalkPolicy.from_config(runtime.get('crosswalk',{}))
    crosswalk_task=CrosswalkTask(crosswalk_policy)
    start_light=StartLight(runtime.get('start_light'))
    lap_config=runtime.get('race',{})
    laps=LapCounter(lap_config,float(crosswalk_policy.body_front_m or .02),float(crosswalk_policy.body_rear_m or .16),float(crosswalk_policy.body_half_width_m or .085))
    if os.environ.get('START_LIGHT_DISARMED')=='1':start_light.cancel()
    calibrated=bool(cal.get('geometry_confirmed') and cal.get('camera_pose_verified'))
    pwm_boost=int(np.clip(runtime.get('motor_peak_boost_pwm',0),0,45))
    vision_budget_ms=float(np.clip(runtime.get('vision_budget_ms',200),80,280))
    turn_lever=float(np.clip(runtime.get('turn_lever_m',.15),.08,.25))
    steering_gain=float(np.clip(runtime.get('steering_gain',1.),.5,1.5))
    # Leave room for the shared actuator boost; final wheel duty stays capped.
    cruise_ceiling=float(MAX_MOTOR_PWM-pwm_boost)
    cruise_pwm=float(np.clip(runtime.get('cruise_pwm',60),1,cruise_ceiling))
    straight_cruise_pwm=float(np.clip(runtime.get('straight_cruise_pwm',cruise_pwm),cruise_pwm,cruise_ceiling))
    steering=SteeringControl(policy,control_frame.get('origin_m',[0.,0.]),control_frame.get('body_m'),
                             float(cal['road_width_m']),turn_lever,steering_gain,cruise_pwm,
                             float(crosswalk_policy.brake_deceleration_m_s2 or .08),
                             crosswalk_policy.speed_per_pwm_m_s,calibration=cal,response_s=crosswalk_policy.speed_response_s,straight_cruise=straight_cruise_pwm)
    # Forward cruise changes must not increase in-place recovery effort.
    search_cruise_pwm=float(np.clip(runtime.get('search_cruise_pwm',60),1,cruise_ceiling))
    search=RoadSearch(policy,steering.origin,steering.body,steering.width,search_cruise_pwm,crosswalk_policy.speed_response_s)
    backend=native_fit_status();print('GEOMETRY_BACKEND',json.dumps(backend),flush=True)
    cv2.setNumThreads(1)
    duration=float(os.environ.get('TARGET_TEST_SECONDS','0'))
    preview_only=not can_enable_drive(cal,runtime,os.environ.get('TARGET_PREVIEW_ONLY')=='1')
    link=PreviewLink() if preview_only else MotorChannel(DriveLink());stream=None;cam=None;vision=None;active=False;pwm=(0,0)
    ramp=CommandRamp();recovery=RoadSupervisor(policy.missing_command_s);consecutive=0;was_pressed=False
    next_control=next_log=next_status=next_stream=0;last_tracking_state=None;previous_captured=None
    began=time.monotonic();frames=0;saved=set();reason=('CALIBRATION REQUIRED' if not calibrated else 'PREVIEW ONLY') if preview_only else 'WAIT START';stats={};observed_frames=0
    try:
        link_ok=False
        try:
            print('PREVIEW_NO_UART' if preview_only else 'STOP_ACK',link.send(0),flush=True);link_ok=True
        except RuntimeError as exc:
            reason='UNO NOT CONNECTED';print(reason,str(exc),flush=True)
        vision=TargetTracker(road_cal)
        crosswalk_detector=CrosswalkDetector(vision.pixels,crosswalk_policy)
        lap_detector=LapMarkerDetector(vision.pixels,lap_config)
        cam=camera.Camera(320,240,image.Format.FMT_BGR888)
        screen=display.Display();touch=touchscreen.TouchScreen();touch.clear()
        stream=LiveStream(8000)
        scale=min(screen.width()/320,screen.height()/240)
        ox=(screen.width()-320*scale)/2;oy=(screen.height()-240*scale)/2
        print('CURVE FOLLOW v0.6.34 | gantry lamp start | preview_only=%s | calibrated=%s | no session time limit'%(preview_only,calibrated),flush=True)
        began=time.monotonic() # Exclude camera startup from test/debug elapsed time.
        while not app.need_exit():
            now=time.monotonic()
            if duration and now-began>=duration:break
            read_began=time.monotonic()
            frame=cam.read();camera_read_at=time.monotonic()
            bgr=image.image2cv(frame,ensure_bgr=True,copy=True)
            captured=time.monotonic()
            profile_ms=dict(camera_read=round((camera_read_at-read_began)*1000,2),
                            image_conversion=round((captured-camera_read_at)*1000,2),
                            frame_period=None if previous_captured is None else round((captured-previous_captured)*1000,2))
            previous_captured=captured
            v,omega=motion_from_pwm(pwm,turn_lever) if active else (0.,0.)
            # Arming uses a trusted current component; driving keeps bounded
            # historical support. Both queries share the same persistent model.
            result,details=vision.update(bgr,captured,v,omega,require_current=not active);frames+=1
            lap_tick=time.perf_counter()
            lap_observation=lap_detector.detect(bgr,details,details.get('measured_road') or result,captured,motion=details.get('ground_motion'))
            lap_finished=laps.update(lap_observation,details.get('ground_motion'),captured,active)
            if laps.event:print('LAP EVENT',json.dumps(laps.snapshot()),flush=True)
            profile_ms['lap_counter']=round((time.perf_counter()-lap_tick)*1000,2)
            crosswalk_tick=time.perf_counter()
            crossing=crosswalk_detector.detect(details,details.get('measured_road') or result,captured)
            motor_before=None if preview_only else link.snapshot()
            actual_ack=motor_before.get('ack',[]) if motor_before else []
            zero_ack=bool(motor_before and motor_before.get('ack_age_ms',1000)<200 and len(actual_ack)>=4 and actual_ack[1:4]==[1,0,0])
            lamp_on=bool(motor_before and motor_before.get('brake_lamp_on') and motor_before.get('brake_lamp_supported'))
            crosswalk_state=crosswalk_task.update(crossing,details.get('ground_motion'),captured,active,zero_ack,lamp_on,
                                                  bool(result.get('valid') and result.get('observed') and result.get('state')=='OBSERVED'))
            crosswalk_ms=(time.perf_counter()-crosswalk_tick)*1000
            stats[result['mode']]=stats.get(result['mode'],0)+1
            observed_frames+=int(result['observed'])
            steering_tick=time.monotonic()
            steering.observe(details['ground_motion'],captured,allow_feedback=active)
            command=steering.command(result,age_s=time.monotonic()-captured)
            if command is None and (steering.status.get('envelope') or {}).get('first_unsafe_m')==0:
                result['invalid_reason']='BODY ENVELOPE BLOCKED'
            if command is not None:
                # Tracker's nearest point remains an association/validity anchor;
                # public target fields describe the actual selected pursuit goal.
                result=dict(result,nearest_reference_m=result['control_target_m'],
                            control_target_m=steering.status['receive_target_m'],
                            target_m=(np.asarray(steering.status['receive_target_m'])+steering.origin).tolist(),
                            target_selection=steering.status['target_selection'],
                            target_distance_m=steering.status['target_distance_m'],
                            target_speed_scale=steering.status['target_speed_scale'],
                            target_inferred=not steering.status['target_observed'],
                            control_target_source='OBSERVED' if steering.status['target_observed'] else 'HISTORY PREDICTION')
            steering_done=time.monotonic()
            search_tick=steering_done
            search.observe(result,details['ground_motion'],captured)
            follow_command=command
            search_allowed=crosswalk_task.track is None or (crosswalk_task.phase=='PASSING' and crosswalk_task.pending_track is None)
            command=search.command(result,command,details['ground_motion'],captured,active=active,allowed=search_allowed)
            profile_ms['road_search']=round((time.monotonic()-search_tick)*1000,2)
            profile_ms.update(result['timings_ms'])
            profile_ms['crosswalk']=round(crosswalk_ms,2)
            profile_ms['steering']=round((steering_done-steering_tick)*1000,2)
            lamp_tick=time.monotonic()
            lamp_observation=start_light.detect(bgr) if start_light.armed else start_light.observation
            profile_ms['start_light']=round((time.monotonic()-lamp_tick)*1000,2)
            detection_ms=(time.monotonic()-captured)*1000
            lamp_start=start_light.update(lamp_observation,captured,active=active,timely=detection_ms<=vision_budget_ms)
            profile_ms['control_pipeline']=round(detection_ms,2)
            supervisor_tick=time.monotonic()
            command,consecutive=assess_vision_command(
                result,command,detection_ms,vision_budget_ms,consecutive,
                policy.measurement_quality)
            road_ready=consecutive>=policy.startup_observations
            x,y,pressed=touch.read();tx=(x-ox)/scale;ty=(y-oy)/scale
            action=stream.control.take();touch_action=None
            if pressed and not was_pressed and 209<=ty<240:
                touch_action='start_now' if 0<=tx<106 else 'stop' if 106<=tx<213 else 'exit' if 213<=tx<320 else None
            was_pressed=pressed
            if touch_action=='exit':break
            if 'stop' in (action,touch_action):
                link.stop();active=False;pwm=(0,0);recovery.reset();crosswalk_task.reset();steering.reset();search.reset(clear_map=True);start_light.cancel();laps.stop();reason='STOPPED';stream.control.stopped()
            else:
                arm_request='start' in (action,touch_action)
                direct_start='start_now' in (action,touch_action)
                manual_start=arm_request or direct_start
                if arm_request and start_light.enabled and not active and not preview_only:
                    start_light.arm();lamp_start=False;reason='WAIT START LIGHT'
                    print('WAIT_START_LIGHT',json.dumps(start_light.snapshot()),flush=True)
                wants_start=direct_start or (lamp_start if start_light.enabled else arm_request)
                if wants_start and not active and not preview_only:
                    task_blockers=crosswalk_policy.blockers() if crosswalk_policy.mode=='yield' else []
                    if crosswalk_policy.mode=='yield' and not (motor_before and motor_before.get('brake_lamp_supported')):task_blockers.append('BRAKE_LAMP_UNAVAILABLE')
                    if link_ok and road_ready and command is not None and not task_blockers and not crosswalk_task.failure:
                        try:
                            link.send(1);active=True;next_control=0
                            recovery.reset();ramp.reset();steering.reset();search.reset(clear_map=True);reason=result['mode']
                            start_light.started()
                            laps.begin(lap_observation,captured)
                            print('START',json.dumps({'pose':result,'auto':start_light.enabled and not direct_start,'source':'MANUAL' if direct_start or not start_light.enabled else 'GANTRY_LIGHT','start_light':start_light.snapshot()}),flush=True)
                        except RuntimeError as exc:
                            start_light.cancel()
                            link.stop();link_ok=False;reason='UNO NOT CONNECTED'
                            stream.control.stopped();print('START REFUSED',str(exc),flush=True)
                    else:
                        reason=crosswalk_task.failure or ('CROSSWALK SETUP REQUIRED' if task_blockers else 'ROAD NOT READY' if link_ok else 'UNO NOT CONNECTED')
                        if manual_start:
                            print('START REFUSED',json.dumps(dict(reason=reason,observed_streak=consecutive,
                                  detection_ms=round(detection_ms,1),command_available=command is not None,
                                  link_ok=link_ok)),flush=True)
            if lap_finished:
                link.stop();active=False;pwm=(0,0);ramp.reset();search.reset(clear_map=True);start_light.cancel();reason='LAPS COMPLETE';stream.control.stopped()
                print('RACE COMPLETE',json.dumps(laps.snapshot()),flush=True)
            now=time.monotonic()
            if active:
                motion=details.get('ground_motion') or {}
                measured_rest=bool(motion.get('valid') and motion.get('speed_m_s') is not None and
                                   abs(motion['speed_m_s'])<=crosswalk_policy.stationary_speed_m_s and
                                   abs(motion['yaw_rad_s'])<=crosswalk_policy.stationary_yaw_rad_s)
                failure=recovery.update(result,crosswalk_task.supervision_command(command,detection_ms,vision_budget_ms),
                                        now,detection_ms,vision_budget_ms,stop_verified=zero_ack and measured_rest,road_ready=road_ready,search_pending=search.status.get('search_pending',False))
                if command is None:
                    rolling=bool(motion.get('valid') and float(motion.get('speed_m_s') or 0.)>crosswalk_policy.stationary_speed_m_s
                                 and float(motion.get('speed_m_s') or 0.)*float(motion.get('interval_s') or 0.)>2*float(motion.get('sigma_m') or 0.))
                    pwm=ramp.pause(now,rolling=rolling);reason={'VISION_WAIT':'VISION WAIT','ROAD_SEARCH':'ROAD SEARCH'}.get(recovery.phase,'REACQUIRING ROAD')
                task_command=crosswalk_task.limit_command(command,crosswalk_state,details.get('ground_motion'))
                if crosswalk_task.failure:failure=crosswalk_task.failure
                if failure and failure!='ROAD NOT FOUND':
                    reason=failure
                    link.stop();active=False;pwm=(0,0);stream.control.stopped()
                    print('STOP EVENT',reason,json.dumps(dict(recovery=recovery.snapshot(),frame=frames,target=dict({k:v for k,v in result.items() if k!='path'},
                                            road_path_m=[[round(x,4),round(y,4)] for x,y in result.get('path',[])]),steering=steering.status)),flush=True)
                elif now>=next_control:
                    if failure=='ROAD NOT FOUND':
                        task_command=(0,0);reason='ROAD NOT FOUND'
                    pwm=task_command if search.last_action and task_command is not None else ramp.apply(task_command,now) if task_command is not None else (0,0)
                    if search.last_action:ramp.reset()
                    try:
                        lease=None
                        if any(pwm):
                            lease=search.status.get('lease_s') if search.last_action else (steering.status.get('handoff') or {}).get('lease_s') if result.get('transition_pending') else None
                        link.send(2,*compensate_pwm(pwm,pwm_boost),lease_s=lease);steering.applied(time.monotonic(),pwm);next_control=time.monotonic()+.05
                        if command is not None and failure is None:reason=('ROAD SEARCH '+search.status['phase'] if follow_command is None and search.status.get('search_pending') else 'CROSSWALK '+crosswalk_task.phase if crosswalk_policy.mode=='yield' and crosswalk_task.track else 'CURVE HANDOFF' if result.get('transition_pending') else result['mode'])
                    except RuntimeError as exc:
                        link.stop();active=False;pwm=(0,0);link_ok=False
                        reason='MOTOR LINK STOP';stream.control.stopped()
                        print('STOP EVENT',reason,str(exc),flush=True)
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
                if start_light.armed and link_ok:reason='WAIT START LIGHT'
                elif reason in ('WAIT START','ROAD NOT READY'):reason='WAIT START' if road_ready else 'ROAD NOT READY'
            if preview_only:reason='CALIBRATION REQUIRED' if not calibrated else 'PREVIEW ONLY'
            task_blockers=crosswalk_policy.blockers() if crosswalk_policy.mode=='yield' else []
            if crosswalk_policy.mode=='yield' and not (motor_before and motor_before.get('brake_lamp_supported')):task_blockers.append('BRAKE_LAMP_UNAVAILABLE')
            ready=link_ok and road_ready and command is not None and not active and not preview_only and not task_blockers and not crosswalk_task.failure
            direct_ready=ready
            if start_light.enabled and start_light.armed:ready=False
            applied_pwm=compensate_pwm(pwm,pwm_boost)
            total_pwm=abs(applied_pwm[0])+abs(applied_pwm[1])
            steering.status['published_curvature_proxy_m_inv']=(applied_pwm[1]-applied_pwm[0])/(turn_lever*total_pwm) if total_pwm else 0.
            readiness=dict(frame=frames,observed_streak=consecutive,required_observations=policy.startup_observations,
                           current_command_timely=detection_ms<=vision_budget_ms,
                           processing_ms=round(detection_ms,1),vision_budget_ms=vision_budget_ms,
                           confidence=result['confidence'],source=result['mode'],
                           blocked_by=startup_blockers(result,command,consecutive,detection_ms,
                               vision_budget_ms,link_ok,preview_only,policy.measurement_quality,required=policy.startup_observations))
            readiness['recovery']=dict(recovery.snapshot(),search=search.status,preview_error=stream.preview_error)
            readiness['laps']=dict(laps.snapshot(),detector=lap_detector.status)
            readiness['direct_start_ready']=bool(direct_ready)
            # ARM starts observation only; both triggers use the same movement gates above.
            readiness['arm_start_light_ready']=(bool(not active and not preview_only and not start_light.armed) if start_light.enabled else bool(direct_ready))
            readiness['minimum_road_quality']=policy.measurement_quality
            readiness['start_light']=start_light.snapshot()
            readiness['steering']=steering.status
            readiness['crosswalk']=dict(crosswalk_task.snapshot(captured),detection=crosswalk_detector.status)
            readiness['blocked_by'].extend(task_blockers)
            if crosswalk_task.failure:readiness['blocked_by'].append(crosswalk_task.failure)
            stream.control.update(active,ready,reason,applied_pwm,readiness=readiness)
            stream.metrics=dict(frame=frames,app_version='0.6.34-lamp-start',geometry_backend=backend,
                                frame_received_at=captured,read_ms=round((captured-read_began)*1000,1),calibrated=calibrated,preview_only=preview_only,
                                laps=dict(laps.snapshot(),detector=lap_detector.status),start_light=start_light.snapshot(),motor=None if preview_only else link.snapshot(),
                                detection_ms=round(detection_ms,1),recovery=dict(recovery.snapshot(),search=search.status),preview_error=stream.preview_error,
                                readiness=readiness,crosswalk=dict(crosswalk_task.snapshot(captured),detection=crosswalk_detector.status),crosswalk_ms=round(crosswalk_ms,2),
                                target=dict({k:v for k,v in result.items() if k!='path'},
                                            road_path_m=[[round(x,4),round(y,4)] for x,y in result.get('path',[])]),steering=steering.status)
            profile_ms['supervision_motor_telemetry']=round((time.monotonic()-supervisor_tick)*1000,2)
            overlay_tick=time.monotonic()
            # The red marker must show the actual pursuit goal in the received
            # image's coordinate frame, not the nearest geometric anchor.
            display_result=dict(result,target_m=None)
            if command is not None and steering.status.get('receive_target_m') is not None:
                display_result['target_m']=(np.asarray(steering.status['receive_target_m'])+steering.origin).tolist()
            out,bird=render_target(bgr,display_result,details,cal)
            draw_crosswalk_overlay(out,crosswalk_task.snapshot(captured),cal)
            lap_detector.draw(out)
            if laps.enabled:cv2.putText(out,'LAPS %d/%d'%(laps.count,laps.total),(4,69),0,.45,(0,255,255),1)
            if start_light.enabled and not active:
                if start_light.roi is not None:
                    lx,ly,lw,lh=start_light.roi
                    cv2.rectangle(out,(lx,ly),(lx+lw,ly+lh),(0,255,255),1)
                cv2.putText(out,'FULL FRAME LAMP '+start_light.observation.get('state','UNKNOWN'),(4,52),0,.35,(0,255,255),1)
            if not calibrated:
                cv2.putText(out,'REFERENCE GEOMETRY ONLY',(4,200),0,.40,(0,0,255),1)
            screen_out=out.copy()
            cv2.rectangle(screen_out,(0,0),(319,37),(0,0,0),-1)
            cv2.putText(screen_out,reason,(4,14),0,.43,(0,255,255),1)
            cv2.putText(screen_out,'L=%d R=%d frame=%d'%(*applied_pwm,frames),(4,30),0,.39,(255,255,255),1)
            for a,b,label,color in [(0,105,'START',(0,100,0) if direct_ready else (55,55,55)),(106,212,'STOP',(0,0,170)),(213,319,'EXIT',(90,50,30))]:
                cv2.rectangle(screen_out,(a,209),(b,239),color,-1);cv2.putText(screen_out,label,(a+14,231),0,.48,(255,255,255),1)
            profile_ms['overlay']=round((time.monotonic()-overlay_tick)*1000,2)
            display_tick=time.monotonic()
            screen_image=image.cv2image(screen_out,bgr=True,copy=True)
            display_ready=time.monotonic();screen.show(screen_image)
            profile_ms['display_conversion']=round((display_ready-display_tick)*1000,2)
            profile_ms['display_show']=round((time.monotonic()-display_ready)*1000,2)
            stream_tick=time.monotonic()
            now=time.monotonic()
            if now>=next_stream:stream.write(np.hstack((bgr,out,bird)));next_stream=now+.2
            profile_ms['stream_encode']=round((time.monotonic()-stream_tick)*1000,2)
            log_tick=time.monotonic()
            if now>=next_log or result['state']=='CONFIRMING' or result['state']!=last_tracking_state:
                print('CONTROL',json.dumps({'active':active,'reason':reason,'recovery':dict(recovery.snapshot(),search=search.status),'target':{k:v for k,v in result.items() if k!='path'},'detection_ms':round(detection_ms,1),'loop_processing_ms':round((time.monotonic()-captured)*1000,1),'pwm':applied_pwm,'logical_pwm':pwm,'frame':frames,'frame_received_at':captured,'read_ms':round((captured-read_began)*1000,1),'uno_connected':None if preview_only else link_ok,'calibrated':calibrated,'preview_only':preview_only,'crosswalk':dict(crosswalk_task.snapshot(captured),detection=crosswalk_detector.status),'start_light':start_light.snapshot(),'laps':laps.snapshot(),'steering':steering.status,'profile_ms':profile_ms}),flush=True);next_log=now+.5
            profile_ms['logging']=round((time.monotonic()-log_tick)*1000,2)
            profile_ms['frame_processing']=round((time.monotonic()-captured)*1000,2)
            stream.metrics['profile_ms']=profile_ms
            readiness['profile_ms']=profile_ms
            last_tracking_state=result['state']
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
        if cam:cam.close()
        link.port.close()
        print('TARGET FOLLOW EXIT',flush=True)
if __name__=='__main__':main()
