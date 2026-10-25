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
