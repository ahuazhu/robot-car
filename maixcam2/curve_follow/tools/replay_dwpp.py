"""Open-loop controller comparison on CONTROL logs; never access any device.

Both controllers see the same recorded road, motion, target horizon and task
permission. Proposed commands are passed through the production ramp, but the
next pose remains the recorded pose. This checks decisions and constraints; it
cannot predict lap time, closed-loop tracking error or physical line contacts.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

APP_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(APP_ROOT))
sys.path.insert(0,str(APP_ROOT.parents[1]))
from robot.config import RuntimeSettings
from robot.control.controller import CommandRamp, compensate_pwm, MAX_MOTOR_PWM
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.steering_control import SteeringControl
from maixcam2.curve_follow.tools.replay_speed_profile import load_frames, distribution


def controller(settings,mode):
    speed=ProfileSpeedControl.from_settings(replace(settings,runtime=dict(settings.runtime,
        speed_profile=dict(settings.runtime['speed_profile'],controller=mode))))
    frame=settings.runtime['control_frame']
    return SteeringControl(settings.road_policy,frame['origin_m'],frame['body_m'],
        settings.calibration['road_width_m'],lever=settings.turn_lever,gain=settings.steering_gain,
        cruise=settings.cruise_pwm,confirmed_cruise=settings.confirmed_cruise_pwm,
        straight_cruise=settings.straight_cruise_pwm,
        deceleration=settings.crosswalk_policy.brake_deceleration_m_s2,
        calibration=settings.calibration,response_s=settings.crosswalk_policy.speed_response_s,
        speed_control=speed)


def compare(paths):
    settings=RuntimeSettings.load(APP_ROOT,{})
    totals={mode:dict(counts=Counter(),reasons=Counter(),overrides=Counter(),
                     compute_ms=[],target_speed_m_s=[],tracking_error_m_inv=[])
            for mode in ('PURE_PURSUIT','DWPP')}
    decisions=[];excluded_tasks=0
    for path in paths:
        controls={mode:controller(settings,mode) for mode in totals}
        ramps={mode:CommandRamp(settings.motor_rise_pwm_s) for mode in totals}
        for row in load_frames(path):
            if not row.get('active'):
                for control in controls.values():control.reset()
                for ramp in ramps.values():ramp.reset()
                continue
            phase=(row.get('crosswalk') or {}).get('phase')
            if phase not in ('SEARCH','PASSING') or (row.get('crosswalk') or {}).get('pending_observation'):
                excluded_tasks+=1
                for control in controls.values():control.reset()
                for ramp in ramps.values():ramp.reset()
                continue
            s=row.get('steering') or {};dt=s.get('motion_interval_s')
            v=s.get('body_speed_m_s');w=s.get('body_yaw_rad_s')
            if not dt or v is None or w is None:continue
            now=row['frame_received_at']
            age=(s.get('latency_prediction') or {}).get('age_s',0.)
            theta=w*dt;c,sin=math.cos(theta),math.sin(theta)
            rotation=np.array([[c,sin],[-sin,c]])
            origin=np.asarray(settings.runtime['control_frame']['origin_m'])
            # Reconstruct the transform that observe() maps to recorded body
            # forward displacement and yaw; lateral displacement is unavailable.
            translation=-np.array([v*dt,0.])@rotation.T-origin@rotation.T+origin
            motion=dict(valid=s.get('motion_measured',False),interval_s=dt,rotation=rotation,
                translation=translation,sigma_m=s['motion_sigma_m'],source='RECORDED_BODY_TWIST')
            details=dict(source=str(path),frame=row['frame'],crosswalk=phase)
            for mode,control in controls.items():
                stats=totals[mode];control.trim=float(s.get('trim_m_inv',0.))
                control.observe(motion,now,False)
                started=time.perf_counter()
                command=control.command(row['target'],age,lookahead_m=s.get('lookahead_m'),
                                        speed_profile_allowed=True)
                stats['compute_ms'].append((time.perf_counter()-started)*1000)
                plan=control.speed_control.status
                selected=bool(plan.get('selected'))
                stats['counts']['evaluated']+=1
                stats['counts']['selected' if selected else 'fallback']+=1
                stats['reasons'][plan.get('reason','NO_PURSUIT')]+=1
                if selected:
                    reference=plan['target_speed_m_s']
                    assert reference<=plan['ceiling_m_s']+1e-8,plan
                    stats['target_speed_m_s'].append(reference)
                    if mode=='DWPP':
                        dwpp=plan['dwpp'];stats['overrides'].update(dwpp['overrides'])
                        stats['counts']['exact_tracking' if dwpp['intersection'] else 'limited_tracking']+=1
                        stats['tracking_error_m_inv'].append(abs(
                            dwpp['selected_curvature_m_inv']-dwpp['requested_curvature_m_inv']))
                if command is None:command=(0,0)
                applied=ramps[mode].apply(command,now+age)
                control.applied(now+age,applied)
                assert max(abs(x) for x in compensate_pwm(applied))<=MAX_MOTOR_PWM
                details[mode]=dict(selected=selected,reason=plan.get('reason'),
                    reference_m_s=plan.get('target_speed_m_s'),wheels=list(applied),
                    dwpp=plan.get('dwpp'))
            decisions.append(details)
    summary={}
    for mode,stats in totals.items():
        summary[mode]={k:distribution(v) if isinstance(v,list) else dict(v)
                       for k,v in stats.items()}
    return dict(scope='OPEN_LOOP_RECORDED_BODY_TWIST',summary=summary,decisions=decisions,
        excluded_task_snapshots=excluded_tasks,
        limitations=['Recorded poses do not respond to counterfactual commands.',
            'Telemetry is event sampled; counts are not time shares.',
            'Lateral body displacement is unavailable; recorded target horizon is reused.',
            'Task-owned speed phases are excluded; task state machines are not replayed.',
            'Dynamic window bounds nominal commands, not measured actuator response.',
            'No lap-time or line-contact improvement can be inferred.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();report=compare(args.inputs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['summary'],ensure_ascii=False))


if __name__=='__main__':main()
