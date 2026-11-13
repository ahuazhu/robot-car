"""Supported-path speed envelope, optional DWPP, and forward-speed feedback.

This scalar regulator cannot authorize a path or bypass mission supervision.
Steering supplies an already accepted pursuit and body envelope; actual applied
effort is fed back after task limits and the motor ramp to prevent windup.
"""

import math

from robot.core.geometry import arc, describe_curve
from robot.navigation.supported_path import supported_segment
from robot.navigation.speed_profile import SpeedLimits, plan_speed_profile
from robot.core.route_confidence import confidence_speed_factor
from robot.navigation.dwpp import plan_twist
from robot.navigation.bend_response import BendResponse
from robot.control.controller import CommandRamp


class ProfileSpeedControl:
    def __init__(self, limits, maximum_effort, response_s, maximum_gap_s=.30,
                 *, high_confidence_gain=1., controller='PURE_PURSUIT',
                 angular_acceleration=2., angular_deceleration=2.,
                 bend_compensation=False, pwm_boost=37, motor_rise_pwm_s=180.,
                 bend_curvature_gain=1.):
        self.limits = limits
        self.maximum_effort = float(maximum_effort)
        self.response_s = float(response_s)
        self.maximum_gap_s = float(maximum_gap_s)
        if (isinstance(high_confidence_gain, bool)
                or not isinstance(high_confidence_gain, (int, float))
                or not math.isfinite(high_confidence_gain) or not 1 <= high_confidence_gain <= 2):
            raise ValueError('High-confidence tracking gain must be within 1..2')
        self.high_confidence_gain = float(high_confidence_gain)
        if controller not in ('PURE_PURSUIT','DWPP'):
            raise ValueError('Unknown path controller')
        if any(isinstance(v,bool) or not isinstance(v,(int,float))
               or not math.isfinite(v) or v<=0
               for v in (angular_acceleration,angular_deceleration)):
            raise ValueError('Angular acceleration limits must be positive and finite')
        self.controller=controller
        self.angular_acceleration=float(angular_acceleration)
        self.angular_deceleration=float(angular_deceleration)
        self.motor_rise_pwm_s=CommandRamp(motor_rise_pwm_s).rise_pwm_s
        if (isinstance(bend_curvature_gain,bool)
                or not isinstance(bend_curvature_gain,(int,float))
                or not math.isfinite(bend_curvature_gain) or bend_curvature_gain<=0):
            raise ValueError('Bend curvature gain must be positive and finite')
        self.bend_curvature_gain=float(bend_curvature_gain)
        if type(bend_compensation) is not bool:
            raise ValueError('bend_compensation must be boolean')
        self.bend_response=(BendResponse(limits.turn_lever_m,pwm_boost)
                            if bend_compensation and controller=='DWPP' else None)
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.maximum_effort, self.response_s, self.maximum_gap_s)):
            raise ValueError('Invalid speed feedback settings')
        self.reset()

    @classmethod
    def from_settings(cls, settings):
        config = settings.runtime.get('speed_profile', {})
        if not isinstance(config, dict):
            raise ValueError('speed_profile must be an object')
        unknown = set(config) - {'mode', 'maximum_m_s', 'lateral_acceleration_m_s2',
                                 'acceleration_m_s2', 'high_confidence_gain','controller',
                                 'angular_acceleration_rad_s2','angular_deceleration_rad_s2',
                                 'bend_compensation','bend_curvature_gain'}
        if unknown or config.get('mode', 'LEGACY') not in ('LEGACY', 'ACTIVE'):
            raise ValueError('Invalid speed_profile configuration')
        if config.get('mode', 'LEGACY') == 'LEGACY':
            return None
        maximum = config.get('maximum_m_s', .30)
        for name in ('maximum_m_s', 'lateral_acceleration_m_s2', 'acceleration_m_s2'):
            value = config.get(name, dict(maximum_m_s=.30,
                lateral_acceleration_m_s2=.15, acceleration_m_s2=.30)[name])
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError('Invalid speed_profile ' + name)
        if maximum > .30:
            raise ValueError('Active speed must be within the measured 0.30m/s ceiling')
        p = settings.crosswalk_policy
        return cls(SpeedLimits(
            maximum, config.get('lateral_acceleration_m_s2', .15),
            config.get('acceleration_m_s2', .30),
            float(p.brake_deceleration_m_s2 or .08), p.command_latency_s,
            settings.turn_lever,
        ), 200 - settings.pwm_boost, p.speed_response_s,
            high_confidence_gain=config.get('high_confidence_gain', 1.),
            controller=config.get('controller','PURE_PURSUIT'),
            angular_acceleration=config.get('angular_acceleration_rad_s2',2.),
            angular_deceleration=config.get('angular_deceleration_rad_s2',2.),
            bend_compensation=config.get('bend_compensation',False),pwm_boost=settings.pwm_boost,
            motor_rise_pwm_s=settings.motor_rise_pwm_s,
            bend_curvature_gain=config.get('bend_curvature_gain',1.))

    def reset(self):
        self.applied_at = None
        self.applied_wheels = (0.,0.)
        self.applied_mean = 0.
        self.reference = None
        self.updated_at = None
        self.status = dict(mode='ACTIVE',controller=self.controller,selected=False,reason='NOT_EVALUATED')
        self.confidence_status = {}
        self.dwpp_state = None
        self.dwpp_proposal = None
        self.bend_proposal = None
        self._gap_hold_proposal = None
        self._gap_hold = None
        self._bend_gap_proposal = None
        self._bend_gap_source = None
        self._pair_request = None

    def applied(self, now, wheels, *, delivered_wheels=None):
        hold_source,self._gap_hold_proposal=self._gap_hold_proposal,None
        self._gap_hold=None
        bend_source,self._bend_gap_proposal=self._bend_gap_proposal,None
        self._bend_gap_source=None
        self.applied_at = float(now)
        self.applied_wheels = tuple(float(value) for value in wheels)
        self.status.pop('applied_curvature_m_inv',None)
        # Keep feedback effort in the pre-boost PWM domain. The final integer
        # wire command supplies its direction/ratio only: a +37 starting boost
        # is not 37 additional units of this regulator's integrated effort.
        self.applied_mean = max(0., sum(wheels) / 2.)
        bend,self.bend_proposal=self.bend_proposal,None
        if bend is not None:
            # Allocation effort is not extra regulator feedback, including
            # when the authorized proposal came from the PWM fallback.
            requested=bend['wheels']
            norm=sum(value*value for value in requested)
            scale=sum(a*b for a,b in zip(wheels,requested))/norm if norm else 0.
            if (0<=scale<=1.02
                    and max(abs(a-scale*b) for a,b in zip(wheels,requested))<=1.5):
                self.applied_mean=bend['base_mean']*min(1.,scale)
            delivered=None if delivered_wheels is None else list(delivered_wheels)
            matches=None if delivered is None else delivered==list(bend['wire'])
            motion=(None if delivered is None or self.bend_response is None
                    else self.bend_response.predict(delivered))
            allocation=self.status.setdefault('bend_allocation',{})
            allocation.update(proposed_wire=list(bend['wire']),
                delivery_matches=matches,delivered_wire=delivered,
                delivered_model=(None if motion is None else dict(
                    speed_m_s=motion[0],yaw_rad_s=motion[1],
                    curvature_m_inv=motion[1]/motion[0] if motion[0]>0 else None)),
                active=matches is True,
                delivery_reason=('MATCHED_ALLOCATION' if matches else
                    'DELIVERY_UNKNOWN' if matches is None else 'MODIFIED_AFTER_ALLOCATION'))
        if self.controller=='DWPP':
            proposal=self.dwpp_proposal
            self.dwpp_proposal=None
            self.dwpp_state=None
            if proposal is not None and any(wheels):
                requested,v,w=proposal
                norm=sum(x*x for x in requested)
                scale=sum(a*b for a,b in zip(wheels,requested))/norm if norm else 0.
                if (0<scale<=1.02 and max(abs(a-scale*b) for a,b in zip(wheels,requested))<=1.5):
                    # Command-domain estimate after the shared motor ramp or
                    # task scaling. It is not measured wheel/body velocity.
                    delivered = wheels if delivered_wheels is None else delivered_wheels
                    total = sum(delivered)
                    if total > 0 and min(delivered) >= 0:
                        curvature = ((delivered[1]-delivered[0]) /
                                     (self.limits.turn_lever_m*total))
                        if bend is not None:
                            predicted=self.bend_response.curvature(delivered)
                            if predicted is not None:curvature=predicted
                            else:self.status['actuator_model_reason']='RAMP_OUTSIDE_MODEL_DOMAIN'
                        linear = v*min(1.,scale)
                        self.dwpp_state=(float(now),linear,linear*curvature)
                        self.status['applied_curvature_m_inv']=curvature
                        # Only an unchanged, actually published measured plan
                        # may bridge one later gap. A held plan cannot rearm it.
                        if (hold_source is not None
                                and max(abs(a-b) for a,b in zip(wheels,requested))<1e-6
                                and (bend is None or list(delivered)==bend['wire'])):
                            saved=dict(self.status)
                            saved['dwpp']=dict(saved['dwpp'],overrides=list(saved['dwpp']['overrides']))
                            saved.update(effective_curvature_m_inv=curvature,
                                reference_angular_rad_s=linear*curvature)
                            self._gap_hold=dict(source_at=hold_source,wheels=tuple(wheels),
                                state=saved,bend=None if bend is None else dict(bend))
                        if (bend_source is not None and min(delivered)>0
                                and max(abs(a-b) for a,b in zip(wheels,requested))<1e-6
                                and (bend is None or list(delivered)==bend['wire'])):
                            # Only a delivered plan with real motion feedback
                            # can seed one later replan. Predicted frames cannot
                            # rearm this source or extend its capture timestamp.
                            self._bend_gap_source=dict(bend_source,wheels=tuple(wheels),
                                reference=linear,speed_cap=min(linear,
                                    bend['estimated_speed_m_s'] if bend is not None else linear))
        if not any(wheels):
            self.reference = self.updated_at = None

    def fallback(self, reason):
        self._gap_hold_proposal=self._gap_hold=None
        self._bend_gap_proposal=self._bend_gap_source=None
        self.bend_proposal=None
        self._pair_request=None
        self.reference = self.updated_at = None
        self.dwpp_proposal=self.dwpp_state=None
        self.status = dict(mode='ACTIVE',controller=self.controller,selected=False,
                           reason=reason,**self.confidence_status)
        return None

    def allocate_command(self, wheels, curvature, *, reference_speed=None,
                         ceiling=None, effort_limits=None, source='DWPP',allow_warmup=True):
        """One calibrated actuator mapping for both accepted steering paths.

        A missing measured speed is not a missing steering intent. PWM control
        uses its requested effort as the preference when it has no speed
        reference. Unmeasured startup/low-speed regions remain explicit;
        neither this adapter nor its caller invents a calibrated response.
        """
        self._pair_request = dict(wheels=tuple(wheels), curvature=curvature,
            reference_speed=reference_speed, ceiling=ceiling,
            effort_limits=effort_limits, source=source,allow_warmup=allow_warmup)
        self.bend_proposal = None
        if self.bend_response is None:
            return tuple(wheels)
        bend = self.bend_response.allocate(wheels, curvature*self.bend_curvature_gain,
            allowed=True, ceiling=ceiling, effort_limits=effort_limits,
            lateral_acceleration=self.limits.lateral_acceleration_m_s2,
            allow_inner_reduction=True, reference_speed=reference_speed,
            motion_requested=any(wheels) and min(wheels)>=0,allow_warmup=allow_warmup)
        self.status.update(actuator_model_source=source,
            actuator_model_reason=self.bend_response.last_reason,
            actuator_base_wire=list(self.bend_response.last_base_wire),
            bend_allocation=dict(active=False, reason=self.bend_response.last_reason,
                                 source=source, requested_curvature_m_inv=curvature))
        if bend is None:
            return tuple(wheels)
        bend.update(curvature_gain=self.bend_curvature_gain,
                    requested_curvature_m_inv=curvature, controller_source=source)
        self.bend_proposal=bend
        self.status['bend_allocation']=dict(active=True, **{
            k:v for k,v in bend.items() if k not in ('wheels','base_wheels','base_mean')})
        self.status['effective_curvature_m_inv']=bend['estimated_curvature_m_inv']
        return tuple(bend['wheels'])

    def map_fallback(self, command, curvature, now, *, effort_ceiling=None, source='PWM',
                     allow_warmup=False):
        """Keep the current pursuit through a DWPP-to-PWM transition."""
        if command is None or not any(command) or self.bend_response is None:
            return command
        ceiling=self.confidence_status.get('confidence_ceiling_m_s')
        clearance=self.confidence_status.get('clearance_ceiling_m_s')
        if clearance is not None:
            ceiling=clearance if ceiling is None else min(ceiling,clearance)
        budget=self._bend_effort_limits(now,None,include_startup=True)
        if effort_ceiling is not None:
            budget={k:min(v,effort_ceiling[k]) for k,v in budget.items()}
        return self.allocate_command(command,curvature,ceiling=ceiling,
            effort_limits=budget,source=source,allow_warmup=allow_warmup)

    def stop_allocation(self, reason):
        self.bend_proposal=self._pair_request=None
        self._bend_gap_proposal=self._bend_gap_source=None
        self.reference=0.
        if self.dwpp_proposal is not None:self.dwpp_proposal=((0.,0.),0.,0.)
        self.status.update(actuator_model_reason=reason,
            bend_allocation=dict(active=False,reason=reason),
            requested_wheels=[0.,0.],target_speed_m_s=0.)
        return (0.,0.)

    def cancel_motion_gap_hold(self):
        self._gap_hold_proposal=self._gap_hold=None

    def cancel_bend_gap(self):
        self._bend_gap_proposal=self._bend_gap_source=None

    def replan_motion_gap(self, target, path, now, curvature, sigma, front_margin,
                          maximum_age_s, *, captured, clearance_limit_m_s=None):
        """Use current road/turn intent with one delivered measured-speed seed.

        This does not freeze PWM or curvature, integrate an estimated speed
        error, or renew the original motion observation's validity window.
        """
        source,self._bend_gap_source=self._bend_gap_source,None
        details=dict(active=False,reason='NO_APPLIED_MEASURED_PLAN',source_age_s=None,
                     decision_age_s=None,maximum_age_s=maximum_age_s)
        if source is None:return None,details
        age=captured-source['source_at']
        details.update(source_age_s=age,decision_age_s=now-source['source_at'])
        if not 0<age<maximum_age_s:
            details['reason']='MEASURED_PLAN_EXPIRED'
            return None,details
        if (self.dwpp_state is None or self.applied_at is None
                or not 0<=now-self.applied_at<self.maximum_gap_s
                or self.applied_wheels!=source['wheels']):
            details['reason']='APPLIED_PLAN_CHANGED'
            return None,details
        wheels=self.command(target,path,now,None,curvature,sigma,front_margin,
            eligible=True,clearance_limit_m_s=clearance_limit_m_s,
            bend_allowed=True,bend_inner_reduction=True,motion_gap=source)
        details.update(active=wheels is not None,
            reason='CURRENT_TURN_REPLANNED' if wheels is not None else self.status['reason'])
        self.status['motion_gap_replan']=details
        return wheels,details

    def hold_motion_gap(self, target, path, now, curvature, sigma, front_margin,
                        maximum_age_s, *, captured):
        """Recheck one already delivered pair without speed feedback or allocation.

        The age gate selects an old input; it does not create a new motor lease.
        Steering still checks this pair's actual arc against the current road.
        """
        saved,self._gap_hold=self._gap_hold,None
        self._gap_hold_proposal=None
        details=dict(active=False,reason='NO_APPLIED_MEASURED_PLAN',
                     source_age_s=None,decision_age_s=None,maximum_age_s=maximum_age_s)
        if saved is None:return None,details
        # GroundMotion's existing prediction window is checked at capture.
        # Command processing freshness and applied-plan age remain separate.
        age=captured-saved['source_at']
        details.update(source_age_s=age,decision_age_s=now-saved['source_at'])
        if not 0<age<maximum_age_s:
            details['reason']='MEASURED_PLAN_EXPIRED'
            return None,details
        state=saved['state']
        held_curvature=state['effective_curvature_m_inv']
        if (curvature*held_curvature<=0
                and max(abs(curvature),abs(held_curvature))>1e-8):
            details['reason']='TURN_INTENT_CHANGED'
            return None,details
        if (self.dwpp_state is None or self.applied_at is None
                or not 0<=now-self.applied_at<self.maximum_gap_s
                or self.applied_wheels!=saved['wheels']):
            details['reason']='APPLIED_PLAN_CHANGED'
            return None,details
        segment,reason=supported_segment(dict(target,control_path_m=path),
                                         skip_inferred_prefix=True,include_history=True)
        curve=None if segment is None else describe_curve(segment)
        if curve is None:
            details['reason']=reason or 'INVALID_SUPPORTED_CURVE'
            return None,details
        positions=arc(curve['path'])
        try:
            confidence=float(target.get('control_confidence',target.get('confidence',1.)))
            length=(target.get('road_model') or {}).get('length_m')
            confidence*=min(1.,max(0.,float(arc(path)[-1] if length is None else length)/.30))
            factor=confidence_speed_factor(confidence)
            reference_sigma=target['reference_sigma_m']
            if not math.isfinite(reference_sigma) or reference_sigma<0:raise ValueError()
            sigma=max(sigma,reference_sigma)
            plan=plan_speed_profile(positions,curve['curvature'],self.limits,
                distance_margin_m=front_margin+2*sigma,
                centerline_confidence=confidence,exact_curvature_required=False)
        except (KeyError,TypeError,ValueError):
            details['reason']='INVALID_CURRENT_ENVELOPE'
            return None,details
        ceiling=min(plan.entry_limit_m_s,
                    self.limits.maximum_m_s/(1+abs(self.limits.turn_lever_m*curvature)))
        if state['target_speed_m_s']>ceiling+1e-9:
            details['reason']='CURRENT_SPEED_LIMIT'
            return None,details
        bend=saved['bend']
        if bend is not None:
            local=sorted(float(k) for s,k in zip(positions,curve['curvature'])
                         if s<=front_margin+self.limits.turn_lever_m)
            middle=len(local)//2
            road_curvature=(local[middle]+local[~middle])/2 if local else 0.
            if abs(road_curvature)<1. or road_curvature*held_curvature<=0:
                details['reason']='SUPPORTED_BEND_CHANGED'
                return None,details
            physical=plan_speed_profile(positions,curve['curvature'],self.limits,
                distance_margin_m=front_margin+2*sigma,
                centerline_confidence=confidence,exact_curvature_required=False,
                nominal_wheel_limit=False)
            if bend['estimated_speed_m_s']>physical.entry_limit_m_s+1e-9:
                details['reason']='CURRENT_PHYSICAL_SPEED_LIMIT'
                return None,details
        wheels=saved['wheels']
        self.status=dict(state,reason='MOTION_GAP_HOLD',measured_speed_m_s=None,
            motion_feedback=False,speed_error_m_s=None,initial_overspeed=None,
            high_confidence_boost=False,tracking_gain=1.,reference_acceleration_m_s2=0.,
            requested_wheels=list(wheels),
            ceiling_m_s=ceiling,centerline_confidence=confidence,
            confidence_speed_factor=factor,confidence_ceiling_m_s=self.limits.maximum_m_s*factor)
        self.status['dwpp']=dict(state['dwpp'],source='HELD_APPLIED_PLAN',
                                overrides=list(state['dwpp']['overrides']))
        self.dwpp_proposal=(wheels,state['target_speed_m_s'],state['reference_angular_rad_s'])
        self.bend_proposal=bend
        details.update(active=True,reason='HELD_APPLIED_PLAN')
        self.status['motion_gap_hold']=details
        return wheels,details

    def constrain_clearance(self, command, ceiling):
        """Brake a chosen arc after its actual curvature's body sweep is checked.

        A common scale keeps that checked arc unchanged. As with any emergency
        speed reduction, braking can override the nominal acceleration window;
        the exception is explicit in telemetry and cannot authorize motion.
        """
        if self._pair_request is not None and self.bend_proposal is not None:
            old=self.bend_proposal
            if old['estimated_speed_m_s']<=ceiling+1e-9:
                return command
            request=dict(self._pair_request)
            request['ceiling']=min(ceiling,request['ceiling']) if request['ceiling'] is not None else ceiling
            reference=request['reference_speed']
            if reference is not None:request['reference_speed']=min(reference,ceiling)
            # Re-solve the current turn at the tighter physical speed cap.
            # If the table has no such slow pair, retain the original braking
            # behavior explicitly, not a false successful calibrated command.
            scale=max(0.,min(1.,ceiling/max(old['estimated_speed_m_s'],reference or 0.)))
            request['wheels']=tuple(x*scale for x in old['base_wheels'])
            wheels=self.allocate_command(**request)
            if self.bend_proposal is None:
                self.status['bend_allocation'].update(
                    reason='UNCALIBRATED_BRAKING',model_reason=self.bend_response.last_reason)
                self.status['actuator_model_reason']='UNCALIBRATED_BRAKING'
            v=min(ceiling,reference) if reference is not None else ceiling
            k=(self.bend_proposal['estimated_curvature_m_inv'] if self.bend_proposal is not None
               else self.status.get('dwpp',{}).get('selected_curvature_m_inv',request['curvature']))
            self.status.update(reason='BODY_CLEARANCE',ceiling_m_s=ceiling,
                target_speed_m_s=v,reference_angular_rad_s=v*k,
                effective_curvature_m_inv=k,requested_wheels=list(wheels))
            if self.dwpp_proposal is not None:
                self.dwpp_proposal=(wheels,v,v*k)
                self.reference=v
            return wheels
        if self.controller!='DWPP' or self.dwpp_proposal is None:return command
        requested,v,w=self.dwpp_proposal
        predicted=(self.bend_proposal['estimated_speed_m_s']
                   if self.bend_proposal is not None else v)
        if max(v,predicted)<=ceiling:return command
        scale=max(0.,min(1.,ceiling/v))
        if self.bend_proposal is not None:
            # Braking withdraws the calibrated allocation. Resume the original
            # regulator's pair and its selected arc; never solve a new pair or
            # impose an inside-wheel floor. Steering checks this restored arc.
            command=self.withdraw_bend(command,'BODY_CLEARANCE_BASELINE')
            _,v,w=self.dwpp_proposal
        wheels=tuple(x*scale for x in command)
        self.reference=v*scale
        self.dwpp_proposal=(wheels,v*scale,w*scale)
        self.status.update(reason='BODY_CLEARANCE',ceiling_m_s=ceiling,
                           target_speed_m_s=v*scale,reference_angular_rad_s=w*scale,
                           requested_wheels=list(wheels),effort_mean_pwm=sum(wheels)/2.)
        if 'BODY_SWEEP_BRAKING' not in self.status['dwpp']['overrides']:
            self.status['dwpp']['overrides'].append('BODY_SWEEP_BRAKING')
        return wheels

    def withdraw_bend(self, command, reason):
        """Restore the original pair when the proposed arc cannot be verified."""
        if self.bend_proposal is None:return command
        command=self.bend_proposal['base_wheels']
        self.bend_proposal=None
        self._pair_request=None
        _,v,_=self.dwpp_proposal
        curvature=self.status['dwpp']['selected_curvature_m_inv']
        self.dwpp_proposal=(command,v,v*curvature)
        self.status.update(effective_curvature_m_inv=curvature,
            reference_angular_rad_s=v*curvature,requested_wheels=list(command),
            effort_mean_pwm=sum(command)/2,actuator_model_reason=reason,
            bend_allocation=dict(active=False,reason=reason))
        return command

    def _bend_effort_limits(self, now, dt, *, include_startup=False):
        """Reserve only effort the unchanged downstream common ramp can pass."""
        elapsed=None if self.applied_at is None else now-self.applied_at
        fresh=elapsed is not None and 0<=elapsed<CommandRamp.MAX_CONTINUOUS_INTERVAL_S
        if include_startup:
            # Same cold-start allowance as CommandRamp, without exempting the
            # allocator from the downstream output ramp.
            interval=elapsed if fresh else .05
            if dt is not None:interval=min(interval,dt)
            previous=tuple(abs(v) for v in self.applied_wheels) if fresh else (0.,0.)
            increase=self.motor_rise_pwm_s*interval
            return dict(peak=max(previous)+increase,sum=sum(previous)+2*increase)
        if not fresh or min(self.applied_wheels)<=0:
            return None
        # applied_at is recorded after the ramp/send, and planning precedes the
        # next ramp. This elapsed interval therefore grants no future time.
        increase=self.motor_rise_pwm_s*min(dt,elapsed)
        previous=tuple(abs(value) for value in self.applied_wheels)
        return dict(peak=max(previous)+increase,sum=sum(previous)+2*increase)

    def limit_fallback(self, command):
        """Use the same normalized envelope when measured-speed feedback is unavailable."""
        if not self.confidence_status.get('confidence_valid',False):
            return (0,0)
        peak=max(abs(v) for v in command)
        ceiling=self.maximum_effort*self.confidence_status['envelope_speed_factor']
        scale=min(1.,ceiling/peak) if peak else 0.
        return tuple(v*scale for v in command)

    def command(self, target, path, now, measured_speed, curvature, sigma, front_margin,
                eligible, high_confidence=False, *, clearance_limit_m_s=None,
                measured_yaw_rate=None,bend_allowed=False,hold_source_at=None,
                bend_inner_reduction=False,bend_source_at=None,motion_gap=None):
        self._gap_hold_proposal=self._gap_hold=None
        self._bend_gap_proposal=self._bend_gap_source=None
        self.bend_proposal=None
        self._pair_request=None
        feedback=motion_gap is None
        planning_speed=measured_speed if feedback else motion_gap['speed']
        planning_yaw=measured_yaw_rate if feedback else motion_gap['yaw']
        try:
            confidence=float(target.get('control_confidence',target.get('confidence',1.)))
            length=(target.get('road_model') or {}).get('length_m')
            if length is None:length=arc(path)[-1]
            length=float(length)
            confidence*=min(1.,max(0.,length/.30))
            factor=confidence_speed_factor(confidence)
        except (ValueError,TypeError,IndexError):
            self.confidence_status={}
            return self.fallback('INVALID_CENTERLINE_CONFIDENCE')
        self.confidence_status=dict(centerline_confidence=confidence,confidence_speed_factor=factor,
                                   confidence_ceiling_m_s=self.limits.maximum_m_s*factor,confidence_valid=True)
        if clearance_limit_m_s is not None:
            if (isinstance(clearance_limit_m_s,bool)
                    or not isinstance(clearance_limit_m_s,(int,float))
                    or not math.isfinite(clearance_limit_m_s) or clearance_limit_m_s<0):
                self.confidence_status={}
                return self.fallback('INVALID_CLEARANCE_SPEED_LIMIT')
            if (target.get('road_model') or {}).get('persistent_history'):
                # Finish-priority recovery retains a crawl instead of waiting
                # indefinitely when the chassis already intersects a boundary.
                clearance_limit_m_s=max(clearance_limit_m_s,
                    self.limits.maximum_m_s*confidence_speed_factor(0.))
        envelope_factor=min(factor,1. if clearance_limit_m_s is None else
                            clearance_limit_m_s/self.limits.maximum_m_s)
        self.confidence_status.update(clearance_ceiling_m_s=clearance_limit_m_s,
                                      envelope_speed_factor=envelope_factor)
        if not eligible:
            return self.fallback('EXISTING_PATH_OR_BODY_GATE')
        if (planning_speed is None or not math.isfinite(planning_speed)
                or planning_speed < -.006):
            return self.fallback('MEASURED_FORWARD_SPEED_REQUIRED')
        if not all(math.isfinite(v) for v in (now, curvature, sigma, front_margin)) or sigma < 0:
            return self.fallback('INVALID_GEOMETRY')
        reference_sigma = target.get('reference_sigma_m')
        if reference_sigma is None or not math.isfinite(reference_sigma) or reference_sigma < 0:
            return self.fallback('MEASURED_PATH_UNCERTAINTY_REQUIRED')
        sigma = max(sigma, reference_sigma)
        segment, reason = supported_segment(dict(target, control_path_m=path),
                                            skip_inferred_prefix=True,include_history=True)
        if segment is None:
            return self.fallback(reason)
        curve = describe_curve(segment)
        if curve is None:
            return self.fallback('INVALID_SUPPORTED_CURVE')
        positions = arc(curve['path'])
        plan = plan_speed_profile(
            positions, curve['curvature'], self.limits,
            distance_margin_m=front_margin + 2 * sigma,
            initial_speed_m_s=max(0., planning_speed),
            centerline_confidence=confidence,
            clearance_limit_m_s=clearance_limit_m_s,
            exact_curvature_required=self.controller!='DWPP',
        )
        ratio = self.limits.turn_lever_m * curvature
        if ((self.controller!='DWPP' and abs(ratio) >= 1)
                or plan.entry_limit_m_s <= 0):
            return self.fallback('FORWARD_PURSUIT_NOT_FEASIBLE')
        # Use the supported-segment entry envelope as an upper bound now. The
        # unseen distance to its first point grants NO extra braking distance;
        # path authorization still belongs to steering's original body checks.
        ceiling = min(plan.entry_limit_m_s,
                      self.limits.maximum_m_s / (1 + abs(ratio)))
        if not feedback:
            ceiling=min(ceiling,motion_gap['speed_cap'])
        if self.controller=='DWPP':
            if planning_yaw is None or not math.isfinite(planning_yaw):
                return self.fallback('DWPP_MEASURED_YAW_REQUIRED')
            # Reacquisition after PWM fallback still has a real control period.
            # A fixed 50ms seed window repeatedly suppresses turn corrections
            # on this camera's 150..220ms cycle. Unsent proposals own no time.
            previous_at=self.applied_at if self.dwpp_state is None else self.dwpp_state[0]
            dt=.05 if previous_at is None else now-previous_at
        else:
            dt = .05 if self.updated_at is None else now - self.updated_at
        if not 0 < dt < self.maximum_gap_s:
            return self.fallback('CONTROL_GAP')
        fast_path = bool(feedback and high_confidence and confidence >= .8 and positions[-1] >= .40)
        reference_acceleration = self.limits.acceleration_m_s2 * (
            self.high_confidence_gain if fast_path else 1.)
        previous_reference = max(0., planning_speed) if self.reference is None else self.reference
        dwpp_status=None
        if self.controller=='DWPP':
            source=('PREDICTED_APPLIED_REFERENCE' if not feedback else
                    'APPLIED_REFERENCE' if self.dwpp_state is not None else 'MEASURED_SEED')
            current=(self.dwpp_state[1:] if self.dwpp_state is not None else
                     (min(self.limits.maximum_m_s,max(0.,planning_speed)),
                      max(-self.limits.maximum_m_s/self.limits.turn_lever_m,
                          min(self.limits.maximum_m_s/self.limits.turn_lever_m,planning_yaw))))
            solution=plan_twist(*current,curvature,ceiling,dt,
                maximum=self.limits.maximum_m_s,lever=self.limits.turn_lever_m,
                acceleration=reference_acceleration,deceleration=self.limits.deceleration_m_s2,
                angular_acceleration=self.angular_acceleration,angular_deceleration=self.angular_deceleration)
            reference=solution.linear
            if reference<=1e-10:
                return self.fallback('DWPP_NO_FORWARD_REFERENCE')
            ratio=self.limits.turn_lever_m*solution.angular/reference
            # Reference/effort conversion preserves the selected wheel ratio.
            ratio=max(-1.,min(1.,ratio))
            dwpp_status=dict(source=source,current_linear_m_s=current[0],current_angular_rad_s=current[1],
                dt_s=dt,window=dict(v_min=solution.window.v_min,v_max=solution.window.v_max,
                    w_min=solution.window.w_min,w_max=solution.window.w_max),
                intersection=solution.intersection,overrides=list(solution.overrides),
                requested_curvature_m_inv=curvature,selected_curvature_m_inv=solution.curvature)
        else:
            reference = min(ceiling, previous_reference + reference_acceleration * dt)
        actual = (self.applied_mean if self.applied_at is not None
                  and 0 <= now - self.applied_at < self.maximum_gap_s else 0.)
        # Extra effort closes a positive speed error on an independently
        # confirmed gentle path. It changes neither the reference envelope nor
        # the response to overspeed; both wheel limits remain authoritative.
        tracking_gain = (self.high_confidence_gain if fast_path and planning_speed < reference else 1.)
        gain = self.maximum_effort / (self.limits.maximum_m_s * (self.response_s + dt))
        gain *= tracking_gain
        if feedback:
            mean = max(0., min(self.maximum_effort / (1 + abs(ratio)),
                              actual + gain * (reference - planning_speed) * dt))
            if planning_speed > ceiling and actual > 0:
                mean = min(mean, actual * ceiling / planning_speed)
        else:
            # No synthetic speed-error feedback. Current geometry may require
            # braking, but a missing measurement cannot increase the reference.
            mean=max(0.,min(self.maximum_effort/(1+abs(ratio)),
                actual*min(1.,reference/motion_gap['reference'])))
        wheels = (mean * (1 - ratio), mean * (1 + ratio))
        bend=None
        bend_ceiling=None
        bend_speed_reason=None
        if self.bend_response is not None and dwpp_status is not None:
            # Outside-only augmentation was limited to an aligned nearby bend.
            # Direct inverse mapping instead executes the current pursuit
            # intent, including lateral corrections on locally straight road.
            # Steering still verifies the resulting physical arc independently.
            span=front_margin+self.limits.turn_lever_m
            local=sorted(float(k) for s,k in zip(positions,curve['curvature'])
                         if s<=span)
            middle=len(local)//2
            road_curvature=((local[middle]+local[~middle])/2 if local else 0.)
            effort_limits=self._bend_effort_limits(now,dt,include_startup=bend_inner_reduction)
            bend_reason=(None if bend_inner_reduction else
                         'BASELINE_GATE' if not bend_allowed else
                         'STARTUP_OR_LOW_SPEED' if planning_speed<.10 else
                         'NO_SUPPORTED_BEND' if not bend_inner_reduction and abs(road_curvature)<1. else
                         'COUNTER_STEER' if not bend_inner_reduction and road_curvature*solution.curvature<=0 else
                         'NO_FRESH_RAMP_BUDGET' if effort_limits is None else None)
            if bend_reason is None:
                # Keep the original DWPP reference above. A measured body
                # response must not also satisfy its ideal wheel-speed proxy.
                # The original arc's clearance cap cannot authorize/reject a
                # different arc: steering checks the proposed physical arc
                # before publication and withdraws it if its own cap fails.
                bend_plan=plan_speed_profile(
                    positions,curve['curvature'],self.limits,
                    distance_margin_m=front_margin+2*sigma,
                    centerline_confidence=confidence,
                    exact_curvature_required=False,nominal_wheel_limit=False)
                bend_ceiling=bend_plan.entry_limit_m_s
                if not feedback:bend_ceiling=min(bend_ceiling,motion_gap['speed_cap'])
                bend_speed_reason=bend_plan.entry_reason
            # This tuning margin changes only the calibrated inversion target.
            # The original DWPP speed reference stays intact;
            # steering still checks the resulting physical arc.
            if bend_inner_reduction:
                wheels=self.allocate_command(wheels,solution.curvature,
                    reference_speed=reference,ceiling=bend_ceiling,
                    effort_limits=effort_limits,source='DWPP',allow_warmup=planning_speed>.006)
                bend=self.bend_proposal
            else:
                bend=self.bend_response.allocate(wheels,solution.curvature*self.bend_curvature_gain,
                    allowed=bend_reason is None,ceiling=bend_ceiling,effort_limits=effort_limits,
                    lateral_acceleration=self.limits.lateral_acceleration_m_s2)
            if bend_reason is not None:self.bend_response.last_reason=bend_reason
            if bend is not None:
                bend['curvature_gain']=self.bend_curvature_gain
                bend['dwpp_curvature_m_inv']=solution.curvature
                wheels=bend['wheels']
                self.bend_proposal=bend
        self.reference, self.updated_at = reference, now
        self.status = dict(
            mode='ACTIVE',controller=self.controller,selected=True,reason=plan.entry_reason,
            target_speed_m_s=reference, ceiling_m_s=ceiling,
            measured_speed_m_s=float(planning_speed) if feedback else None,
            initial_overspeed=plan.initial_overspeed if feedback else None,
            supported_length_m=float(positions[-1]),
            usable_length_m=plan.stations_m[-1], limiting_station_m=plan.limiting_station_m,
            first_supported_point_m=segment[0].tolist(),
            effort_mean_pwm=sum(wheels)/2, applied_mean_pwm=actual,
            regulator_effort_mean_pwm=mean,
            requested_wheels=list(wheels),
            tracking_gain=tracking_gain, high_confidence_boost=tracking_gain > 1.,
            reference_acceleration_m_s2=reference_acceleration,
            **self.confidence_status,
        )
        if dwpp_status is not None:
            self.status.update(dwpp=dwpp_status,reference_angular_rad_s=solution.angular,
                               effective_curvature_m_inv=solution.curvature)
            self.dwpp_proposal=(wheels,solution.linear,solution.angular)
            if self.bend_response is not None:
                self.status['actuator_model_reason']=self.bend_response.last_reason
                self.status['bend_speed_ceiling_m_s']=bend_ceiling
                self.status['bend_speed_limit_reason']=bend_speed_reason
                self.status['bend_effort_limits']=effort_limits
                self.status['actuator_base_wire']=list(self.bend_response.last_base_wire)
                if bend_inner_reduction:
                    self.status['actuator_model_source']='DWPP'
                    self.status['bend_allocation']=dict(active=False,
                        reason=self.bend_response.last_reason,source='DWPP',
                        requested_curvature_m_inv=solution.curvature)
                base_motion=self.bend_response.last_base_motion
                if base_motion is not None:
                    self.status['actuator_base_speed_m_s']=base_motion[0]
                    self.status['actuator_base_curvature_m_inv']=base_motion[1]/base_motion[0]
            if bend is not None:
                self.status['bend_allocation']=dict(active=True,**{
                    key:value for key,value in bend.items()
                    if key not in ('wheels','base_wheels','base_mean')})
                self.status['effective_curvature_m_inv']=bend['estimated_curvature_m_inv']
        if hold_source_at is not None and self.controller=='DWPP':
            self._gap_hold_proposal=float(hold_source_at)
        if not feedback:
            self.status.update(motion_feedback=False,planning_speed_m_s=planning_speed,
                               speed_error_m_s=None)
        elif (bend_source_at is not None and bend_inner_reduction
                and self.controller=='DWPP' and planning_speed>=.10):
            self._bend_gap_proposal=dict(source_at=float(bend_source_at),
                                        speed=float(planning_speed),yaw=float(planning_yaw))
        return wheels
