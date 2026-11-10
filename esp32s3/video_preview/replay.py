"""Compile and replay the exact ESP32 detector on RGB565 frames (no motor I/O)."""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import numpy as np


class State(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ("armed", "confirmed", "baseline", "sequence", "hits", "elapsed", "count", "locked", "phase", "target_x", "target_y", "target_w", "target_h", "contrast", "baseline_contrast", "event_age")]


class Diagnostic(ctypes.Structure):
    _fields_ = [(name,ctypes.c_float) for name in ("background_change","background_residual","darkened_fraction","required_drop","noise","exposure_gain","gray_drop","value_drop","saturation_change")]+[("mask_pixels",ctypes.c_uint32),("reason",ctypes.c_char_p)]


class RGBDiagnostic(ctypes.Structure):
    _fields_ = [(name,ctypes.c_uint32) for name in ("workspace_available","reference_ready","luma_off","rgb_off")]+[("channel",ctypes.c_int32),("joint_fraction",ctypes.c_float),("dark_fraction",ctypes.c_float),("background_samples",ctypes.c_uint32),("background_span",ctypes.c_float),("background_fitted",ctypes.c_uint32),("reason",ctypes.c_char_p)]


class NativeDetector:
    def __init__(self,full_frame=False,header=None):
        self.folder = tempfile.TemporaryDirectory(prefix="esp-start-replay-")
        root = Path(self.folder.name)
        header = Path(header).resolve() if header is not None else Path(__file__).with_name("start_signal.h")
        source = root / "replay.cpp"
        source.write_text('#include "'+str(header)+'"\n' + '''
extern "C" {
struct State {unsigned armed,confirmed,baseline,sequence,hits,elapsed,count,locked,phase,target_x,target_y,target_w,target_h,contrast,baseline_contrast,event_age;};
struct Diagnostic {float background_change,background_residual,darkened_fraction,required_drop,noise,exposure_gain,gray_drop,value_drop,saturation_change;unsigned mask_pixels;const char *reason;};
Diagnostic diagnostic(void *p){auto &d=*(start_signal::Detector*)p;return {d.backgroundChangeL,d.backgroundResidualL,d.darkenedFraction,d.requiredDrop,d.noise,d.exposureGain,d.grayDrop,d.valueDrop,d.saturationChange,d.baseline?d.maskPixels:0,d.rejectionReason};}
struct RGBDiagnostic {unsigned workspace_available,reference_ready,luma_off,rgb_off;int channel;float joint_fraction,dark_fraction;unsigned background_samples;float background_span;unsigned background_fitted;const char *reason;};
RGBDiagnostic rgbDiagnostic(void *p){
#ifdef ROBOT_SIGNAL_RGB_OR
 auto &d=*(start_signal::Detector*)p;
 return {d.rgb_workspace_available,d.rgb_reference_ready,d.luma_off,d.rgb_off,d.rgb_channel,d.rgb_joint_fraction,d.rgb_dark_fraction,d.rgb_background_samples,d.rgb_background_span,d.rgb_background_fitted,d.rgb_rejection_reason};
#else
 return {0,0,0,0,-1,0,0,0,0,0,"UNSUPPORTED"};
#endif
}
unsigned rgbReferenceBytes(){
#ifdef ROBOT_SIGNAL_RGB_OR
 return start_signal::Detector::RGB_REFERENCE_BYTES;
#else
 return 0;
#endif
}
int rgbReferenceBuffer(void *p,unsigned char *buffer,unsigned size){
#ifdef ROBOT_SIGNAL_RGB_OR
 return ((start_signal::Detector*)p)->setRGBReferenceBuffer(buffer,size);
#else
 return 0;
#endif
}
void *create(){return new start_signal::Detector();}
void destroy(void *p){delete (start_signal::Detector*)p;}
void mask(void *p,unsigned char *out){((start_signal::Detector*)p)->copyMask(out);}
void arm(void *p){((start_signal::Detector*)p)->arm();}
void modelHint(void *p,int x,int y,int w,int h,int valid,int required){auto &d=*(start_signal::Detector*)p;d.requireModelLocator(required);d.setModelHint(x,y,w,h,valid);}
int manual(void *p,int x,int y,int w,int h,int enabled){return ((start_signal::Detector*)p)->setManualROI(x,y,w,h,enabled);}
void cycle(void *p,int enabled){((start_signal::Detector*)p)->requireOffOnOff(enabled);}
void cancel(void *p){((start_signal::Detector*)p)->cancel();}
unsigned geometry(unsigned i){const unsigned a[]={start_signal::Detector::X,start_signal::Detector::Y,start_signal::Detector::RW,start_signal::Detector::RH,start_signal::Detector::STEP};return i<5?a[i]:0;}
unsigned replayNow=0;
void process(void *p,const unsigned char *pixels,unsigned now,int y){replayNow=now;((start_signal::Detector*)p)->process(pixels,now,y<0?start_signal::Detector::Y:y);}
State state(void *p){auto &d=*(start_signal::Detector*)p;return {d.armed,d.confirmed,d.baseline,d.sequence,d.hits,d.elapsed,(unsigned)d.region.count,d.baseline,(unsigned)d.phase,(unsigned)(start_signal::Detector::X+d.target.x*start_signal::Detector::STEP),(unsigned)(start_signal::Detector::Y+d.target.y*start_signal::Detector::STEP),(unsigned)(d.target.w*start_signal::Detector::STEP),(unsigned)(d.target.h*start_signal::Detector::STEP),(unsigned)(d.contrast>0?d.contrast:0),(unsigned)(d.baselineContrast>0?d.baselineContrast:0),d.eventAge(replayNow)};}
}
''')
        library = root / "detector.so"
        subprocess.run(["c++", "-std=c++11", "-O2", "-shared", "-fPIC", *(["-DROBOT_LAMPNET_FULL_FRAME=1"] if full_frame else []), str(source), "-o", str(library)],check=True,capture_output=True)
        self.lib = ctypes.CDLL(str(library))
        self.lib.create.restype=ctypes.c_void_p
        self.lib.geometry.argtypes=[ctypes.c_uint];self.lib.geometry.restype=ctypes.c_uint
        self.roi=[int(self.lib.geometry(i)) for i in range(4)];self.step=int(self.lib.geometry(4))
        for method in ("destroy","arm","cancel"):
            getattr(self.lib,method).argtypes=[ctypes.c_void_p]
        self.lib.diagnostic.argtypes=[ctypes.c_void_p];self.lib.diagnostic.restype=Diagnostic
        self.lib.rgbDiagnostic.argtypes=[ctypes.c_void_p];self.lib.rgbDiagnostic.restype=RGBDiagnostic
        self.lib.rgbReferenceBytes.restype=ctypes.c_uint
        self.lib.rgbReferenceBuffer.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint];self.lib.rgbReferenceBuffer.restype=ctypes.c_int
        self.lib.mask.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        self.lib.process.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int]
        self.lib.state.argtypes=[ctypes.c_void_p];self.lib.state.restype=State
        self.lib.modelHint.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*6
        self.lib.manual.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*5;self.lib.manual.restype=ctypes.c_int
        self.lib.cycle.argtypes=[ctypes.c_void_p,ctypes.c_int]
        self.handle=self.lib.create()
        self._rgb_reference=None
        size=int(self.lib.rgbReferenceBytes())
        if size:
            self._rgb_reference=ctypes.create_string_buffer(size)
            if not self.lib.rgbReferenceBuffer(self.handle,self._rgb_reference,size):
                self.close()
                raise RuntimeError('Cannot attach RGB reference workspace')

    def hint(self,box,required=True):
        x,y,w,h=box if box is not None else [0,0,0,0]
        self.lib.modelHint(self.handle,x,y,w,h,int(box is not None),int(required))
    def manual(self,box):
        return bool(self.lib.manual(self.handle,*(box if box else [0,0,0,0]),int(box is not None)))
    def cycle(self,enabled=True):self.lib.cycle(self.handle,int(enabled))
    def arm(self): self.lib.arm(self.handle)
    def cancel(self): self.lib.cancel(self.handle)
    def process(self, bgr, now_ms, roi_y=-1):
        if bgr.shape != (240,320,3):raise ValueError('Replay requires an unscaled 320x240 BGR frame')
        b,g,r=[bgr[:,:,i].astype(np.uint16) for i in range(3)]
        packed=((r>>3)<<11)|((g>>2)<<5)|(b>>3)
        data=packed.astype('>u2').tobytes()
        self.lib.process(self.handle,ctypes.c_char_p(data),now_ms,roi_y)
        return self.state()
    def state(self):
        s=self.lib.state(self.handle)
        return {name:int(getattr(s,name)) for name,_ in State._fields_}
    def diagnostics(self):
        d=self.lib.diagnostic(self.handle)
        return {name:(d.reason.decode() if name=="reason" else int(getattr(d,name)) if name=="mask_pixels" else float(getattr(d,name))) for name,_ in Diagnostic._fields_}
    def rgb_diagnostics(self):
        d=self.lib.rgbDiagnostic(self.handle)
        return {name:(d.reason.decode() if name=="reason" else float(getattr(d,name)) if kind is ctypes.c_float else int(getattr(d,name))) for name,kind in RGBDiagnostic._fields_}
    def mask(self):
        # One pixel represents STEP x STEP native camera pixels; no bbox fill.
        out=np.empty((self.roi[3]//self.step,self.roi[2]//self.step),dtype=np.uint8)
        self.lib.mask(self.handle,out.ctypes.data)
        return out
    def close(self):
        if self.handle:self.lib.destroy(self.handle);self.handle=None
        self._rgb_reference=None
        self.folder.cleanup()


def replay_history(root):
    import cv2,json
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    detector=NativeDetector();results=[]
    def load(path,dy):
        image=cv2.imread(str(path))[:,:320]
        if not dy:return image
        # Move the entire view by an integer offset. Unknown pixels are never
        # accepted as evidence: target plus background margin must be original.
        out=np.zeros_like(image)
        out[max(0,dy):min(240,240+dy)] = image[max(0,-dy):min(240,240-dy)]
        return out
    try:
        for source in manifest['sources']:
            dy=64-(source['roi'][1]+source['roi'][3]//2)
            for framing,shift in (('original',0),('whole_view_aligned',dy)):
                for before,after in (('on','off'),('off','on')):
                    detector.arm()
                    initial=[detector.process(load(root/(source['name']+'-'+before+'-%d.png'%i),shift),i*100) for i in range(4)]
                    states=[detector.process(load(root/(source['name']+'-'+after+'-%d.png'%i),shift),400+i*210) for i in range(4)]
                    state=initial[-1]
                    supported=bool(state['locked'] and state['target_y']-12>=max(0,shift) and state['target_y']+state['target_h']+12<=min(240,240+shift))
                    confirmed=bool(states[-1]['confirmed'])
                    results.append(dict(sample=source['name'],framing=framing,vertical_shift_px=shift,transition=before+'->'+after,
                                        locked_before_change=bool(state['locked']),target_roi=[state[k] for k in ('target_x','target_y','target_w','target_h')],
                                        original_context_available=supported,confirmed=confirmed,usable_confirmation=confirmed and supported,
                                        confirmation_ms=states[-1]['elapsed'],final_phase=states[-1]['phase']))
        jitter=sorted((root.parent/'start_signal_jitter_20261031').glob('*.png'))
        detector.arm();events=[]
        for i,p in enumerate(jitter):
            if detector.process(load(p,0),i*210)['confirmed']:events.append(p.name)
        return dict(method='Exact firmware C++ kernel, RGB565 quantization. Four distinct ON and four distinct OFF images, assigned 100ms baseline / 210ms confirmation intervals; source timestamps unavailable. Original views and integer translation of the WHOLE view are reported separately. Padding cannot qualify as lamp/background evidence.',
                    limitation='Old Maix camera, changed framing/people between ON and OFF, two short samples only. Not an ESP32 accuracy or latency measurement. Unlocked, uncertain and unavailable-context outcomes are retained.',
                    results=results,jitter_frames=len(jitter),jitter_false_triggers=len(events),jitter_events=events)
    finally:detector.close()


def replay_recording(root):
    """Replay captured JPEGs in recorded monotonic time; never reads credentials."""
    import cv2,json
    root=Path(root);detector=NativeDetector();rows=[];transitions=[];previous=None
    try:
        detector.arm()
        for line in (root/'events.jsonl').read_text().splitlines():
            frame=json.loads(line)
            if frame['kind']!='frame':continue
            state=detector.process(cv2.imread(str(root/frame['file'])),int(frame['t']*1000))
            diagnostic=detector.diagnostics()
            row=dict(t=frame['t'],file=frame['file'],state=state,diagnostic=diagnostic);rows.append(row)
            signature=(state['phase'],diagnostic['reason'])
            if signature!=previous:
                transitions.append(row);previous=signature
        locked=next((r for r in rows if r['state']['locked']),None)
        off=next((r for r in rows if r['state']['hits']),None)
        fired=next((r for r in rows if r['state']['confirmed']),None)
        return dict(method='Exact firmware C++ kernel; recorded JPEGs re-quantized to RGB565, using capture timestamps. No motor I/O.',
                    limitation='JPEGs are not original sensor RGB565 frames; sample timing does not measure physical lamp latency or on-track accuracy.',
                    frames=len(rows),first_lock_s=locked['t'] if locked else None,
                    first_off_evidence_s=off['t'] if off else None,confirmation_s=fired['t'] if fired else None,
                    confirmation_span_ms=fired['state']['elapsed'] if fired else None,
                    mask_pixels_at_confirmation=fired['diagnostic']['mask_pixels'] if fired else None,
                    transitions=transitions)
    finally:detector.close()


if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures',type=Path,default=Path('tests/fixtures/esp_start_signal_history'))
    parser.add_argument('--recording',type=Path,help='Recorder directory with events.jsonl and frames/')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=replay_recording(args.recording) if args.recording else replay_history(args.fixtures)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
