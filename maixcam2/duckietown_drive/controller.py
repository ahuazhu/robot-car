"""Conservative initial PWM steering using camera-frame lane pose.
PWM is not measured wheel velocity; camera-to-axle offset is not calibrated.
"""
import math

SESSION_SECONDS=5.0

def drive_command(pose):
    if not pose or not pose.get('valid'):return None
    d=pose.get('d_m');phi=pose.get('phi_rad')
    if d is None or phi is None or not all(map(math.isfinite,(d,phi))):return None
    if pose.get('mode')=='CURVE':
        if abs(pose.get('near_center_m',1))>.12 or abs(phi)>.85:return None
    elif abs(d)>.10 or abs(phi)>.60:return None
    if not .235<=pose.get('width_m',.30)<=.365:return None
    # Approximate target 35 cm ahead in camera-ground coordinates.
    # Positive target y means turn left: left wheels slower than right.
    if pose.get('mode')=='CURVE':
        x,y=pose.get('target_m',(0,0))
        if not all(map(math.isfinite,(x,y))) or not .30<=x<=.65:return None
    else:
        x=.35;y=(-d-math.sin(phi)*x)/math.cos(phi)
    turn=max(-6.,min(6.,4.*2*y/(x*x+y*y)))
    if abs(turn)<1.0:turn=0.
    base=(54. if pose.get('single_side') else 62.)-8.*abs(turn)/6.
    return round(base-turn),round(base+turn)

class CommandRamp:
    def __init__(self):self.reset()
    def reset(self):self.base=0.;self.turn=0.;self.last=None
    def apply(self,command,now):
        if command is None or command==(0,0):self.reset();return (0,0)
        dt=.05 if self.last is None else max(0.,min(.1,now-self.last));self.last=now
        base=sum(command)/2.;turn=(command[1]-command[0])/2.
        self.base=min(base,self.base+180*dt)
        turn *= self.base/base if base>0 else 0.
        if turn*self.turn<=0:self.turn=0.
        step=60*dt
        self.turn+=max(-step,min(step,turn-self.turn))
        self.turn=max(-self.base,min(self.base,self.turn))
        return round(self.base-self.turn),round(self.base+self.turn)


class SingleSideGuard:
    """Permit only a short, continuous transition from a measured two-side lane."""
    def __init__(self):self.paired_at=None;self.target_y=None
    def accept(self,pose,now):
        if not pose or not pose.get('valid'):return False
        if not pose.get('single_side'):
            self.paired_at=now
            self.target_y=pose.get('target_m',(0,0))[1]
            return True
        y=pose['target_m'][1]
        if self.paired_at is None or now-self.paired_at>1.5:return False
        if self.target_y is None or abs(y-self.target_y)>.05:return False
        self.target_y=y
        return True
