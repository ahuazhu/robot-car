"""Native replay of the same bounded C++ locator used on ESP32."""
import ctypes,subprocess,tempfile
from pathlib import Path
import numpy as np

class NativeLocator:
    def __init__(self,view=None):
        self.folder=tempfile.TemporaryDirectory(prefix='lamp-face-')
        root=Path(self.folder.name);header=Path(__file__).resolve().parents[1]/'video_preview/lamp_face_locator.h'
        source=root/'bridge.cpp'
        source.write_text('#include "'+str(header)+'"\n'+'''
static lamp_face::Locator locator;
extern "C" void view(int x,int y,int w,int h){locator.view(x,y,w,h);}
extern "C" void locate(const uint8_t *p,const uint8_t *rgb,int *out,uint8_t *mask){
  auto r=locator.locate(p,rgb);
  int data[]={int(r.valid),r.box.x,r.box.y,r.box.w,r.box.h,int(r.coarseValid),r.coarseBox.x,r.coarseBox.y,r.coarseBox.w,r.coarseBox.h,int(r.area),int(r.method),int(r.threshold),int(r.confidence*1000000+.5f)};
  memcpy(out,data,sizeof(data));memcpy(mask,locator.mask,lamp_face::PIXELS);
}
''')
        library=root/'locator.so';subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC',str(source),'-o',str(library)],check=True,capture_output=True)
        self.lib=ctypes.CDLL(str(library));self.lib.locate.argtypes=[ctypes.c_void_p]*4;self.lib.locate.restype=None
        self.lib.view.argtypes=[ctypes.c_int]*4
        if view is not None:self.lib.view(*view)
    def locate(self,probability,rgb):
        # Keep array ownership until the native call completes.
        p=np.ascontiguousarray(probability,dtype=np.uint8);c=np.ascontiguousarray(rgb,dtype=np.uint8)
        if p.shape!=(40,40) or c.shape!=(40,40,3):raise ValueError('Expected 40x40 probabilities and RGB')
        out=np.zeros(14,np.int32);mask=np.zeros((40,40),np.uint8);self.lib.locate(p.ctypes.data,c.ctypes.data,out.ctypes.data,mask.ctypes.data)
        return dict(valid=bool(out[0]),box=out[1:5].tolist(),coarse_valid=bool(out[5]),coarse_box=out[6:10].tolist(),area=int(out[10]),method=int(out[11]),threshold=int(out[12]),confidence=out[13]/1e6),mask
    def close(self):self.folder.cleanup()
