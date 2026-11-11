"""Constant-size Dynamic Window Pure Pursuit in body velocity space.

Python adaptation of the DWPP selection in Nav2:
Copyright (c) 2025 Fumiya Ohnishi, Apache-2.0.
https://github.com/ros-navigation/navigation2/blob/main/nav2_regulated_pure_pursuit_controller/include/nav2_regulated_pure_pursuit_controller/dynamic_window_pure_pursuit_functions.hpp

The robot-car adaptation clips the window by forward-only wheel-speed bounds.
Hard speed/braking constraints take priority when the acceleration window is
already infeasible; overrides are returned explicitly, never silently hidden.
This constrains reference commands, not independently measured motor dynamics.
"""
from dataclasses import dataclass
import math

EPS = 1e-10


@dataclass(frozen=True)
class Window:
    v_min: float
    v_max: float
    w_min: float
    w_max: float


@dataclass(frozen=True)
class Solution:
    linear: float
    angular: float
    window: Window
    intersection: bool
    overrides: tuple
    vertices: tuple

    @property
    def curvature(self):
        return self.angular / self.linear if self.linear > EPS else 0.


def _clip(points, a, b, bound):
    """Intersect a convex polygon with a*v + b*w <= bound."""
    result=[]
    for start,end in zip(points,points[1:]+points[:1]):
        ds=a*start[0]+b*start[1]-bound
        de=a*end[0]+b*end[1]-bound
        inside_start,inside_end=ds<=EPS,de<=EPS
        if inside_start:
            result.append(start)
        if inside_start != inside_end:
            fraction=ds/(ds-de)
            result.append((start[0]+fraction*(end[0]-start[0]),
                           start[1]+fraction*(end[1]-start[1])))
    unique=[]
    for point in result:
        if not unique or math.hypot(point[0]-unique[-1][0],point[1]-unique[-1][1])>EPS:
            unique.append(point)
    if len(unique)>1 and math.hypot(unique[0][0]-unique[-1][0],unique[0][1]-unique[-1][1])<=EPS:
        unique.pop()
    return unique


def feasible_polygon(window, wheel_maximum, lever):
    points=[(window.v_min,window.w_min),(window.v_max,window.w_min),
            (window.v_max,window.w_max),(window.v_min,window.w_max)]
    for a,b,bound in ((1.,lever,wheel_maximum),(1.,-lever,wheel_maximum),
                      (-1.,lever,0.),(-1.,-lever,0.)):
        points=_clip(points,a,b,bound)
        if not points:break
    return points


def closest_tracking_velocity(points, curvature):
    """Minimize distance to w=k*v; break ties by greatest forward progress."""
    if not points:
        raise ValueError('Empty DWPP velocity region')
    intersections=[]
    for start,end in zip(points,points[1:]+points[:1]):
        ds=start[1]-curvature*start[0]
        de=end[1]-curvature*end[0]
        if abs(ds)<=EPS:intersections.append(start)
        if ds*de<0.:
            fraction=ds/(ds-de)
            intersections.append((start[0]+fraction*(end[0]-start[0]),
                                  start[1]+fraction*(end[1]-start[1])))
    if intersections:
        return (*max(intersections,key=lambda p:p[0]),True)
    errors=[abs(w-curvature*v) for v,w in points]
    best=min(errors)
    selected=max((p for p,error in zip(points,errors) if error<=best+EPS),key=lambda p:p[0])
    return (*selected,False)


def plan_twist(linear, angular, curvature, ceiling, dt, *, maximum, lever,
               acceleration, deceleration, angular_acceleration, angular_deceleration):
    values=(linear,angular,curvature,ceiling,dt,maximum,lever,acceleration,
            deceleration,angular_acceleration,angular_deceleration)
    if (not all(math.isfinite(v) for v in values) or min(linear,ceiling)<0
            or min(dt,maximum,lever,acceleration,deceleration,
                   angular_acceleration,angular_deceleration)<=0):
        raise ValueError('Invalid DWPP inputs')
    if linear>maximum+EPS or abs(angular)>maximum/lever+EPS:
        raise ValueError('DWPP state exceeds static velocity bounds')
    v_min=max(0.,linear-deceleration*dt)
    v_max=min(maximum,ceiling,linear+acceleration*dt)
    changes=(angular_deceleration,angular_acceleration) if angular>EPS else (
        (angular_acceleration,angular_deceleration) if angular < -EPS else
        (angular_acceleration,angular_acceleration))
    w_min=max(-maximum/lever,angular-changes[0]*dt)
    w_max=min(maximum/lever,angular+changes[1]*dt)
    overrides=[]
    if v_min>v_max:
        # A newly reduced safety envelope cannot be negotiated upward merely
        # because the previous command cannot decelerate to it in one period.
        v_min=v_max
        overrides.append('LINEAR_BRAKING')
    window=Window(v_min,v_max,w_min,w_max)
    points=feasible_polygon(window,maximum,lever)
    if not points:
        # At very low forward speed the angular acceleration interval can lie
        # wholly outside the forward-wheel cone. Brake to its nearest edge.
        bound=min(maximum/2,v_max,maximum-v_min)/lever
        closest=max(-bound,min(bound,max(w_min,min(0.,w_max))))
        window=Window(v_min,v_max,closest,closest)
        points=feasible_polygon(window,maximum,lever)
        overrides.append('ANGULAR_BRAKING')
    v,w,intersection=closest_tracking_velocity(points,curvature)
    return Solution(max(0.,v),w,window,intersection,tuple(overrides),tuple(points))
