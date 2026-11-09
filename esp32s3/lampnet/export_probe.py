"""Export opt-in shadow or model-locator ESP32 firmware and numeric checks."""
from pathlib import Path
import argparse,json,shutil,hashlib
import cv2,numpy as np,tensorflow as tf

def array(name,data,signed=False):
    values=np.frombuffer(data,dtype=np.int8 if signed else np.uint8)
    lines=[','.join(str(int(v)) if signed else '0x%02x'%v for v in values[i:i+20]) for i in range(0,len(values),20)]
    return 'alignas(16) const '+('int8_t' if signed else 'unsigned char')+' '+name+'[] = {\n'+',\n'.join(lines)+'\n};\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--full-frame',action='store_true');ap.add_argument('--model-locator',action='store_true');ap.add_argument('--dual-core',action='store_true');ap.add_argument('--golden-image',type=Path,action='append',default=[]);a=ap.parse_args()
    sketch=a.out/'sketch/video_preview';sketch.mkdir(parents=True,exist_ok=True)
    src=a.repo/'esp32s3/video_preview'
    for name in ['video_preview.ino','start_signal.h','signal_service.h','lampnet_shadow.h','lampnet_esp_nn.h','lamp_face_locator.h']:shutil.copy2(src/name,sketch/name)
    if a.dual_core:
        path=sketch/'video_preview.ino';path.write_text('#define ROBOT_LAMPNET_DUAL_CORE 1\n'+path.read_text())
    if a.model_locator:
        path=sketch/'video_preview.ino';path.write_text('#define ROBOT_LAMPNET_FACE 1\n'+path.read_text())
    if a.full_frame:
        if not a.model_locator:raise SystemExit('--full-frame requires --model-locator')
        path=sketch/'video_preview.ino';path.write_text('#define ROBOT_LAMPNET_FULL_FRAME 1\n'+path.read_text())
    model_path=a.repo/'android/start_signal/model/lamp_model.int8.tflite';blob=model_path.read_bytes()
    interpreter=tf.lite.Interpreter(model_content=blob,experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,num_threads=2);interpreter.allocate_tensors()
    ti,to=interpreter.get_input_details()[0],interpreter.get_output_details()[0]
    scale,zero=ti['quantization']
    assets=a.repo/'android/start_signal/app/src/main/assets'
    samples=[('REAL_OFF','diagnostic/real-off.png',[0,0,105,76]),('DIGITS_NEGATIVE','diagnostic/real-off.png',[88,0,232,76])]
    for index,path in enumerate(a.golden_image):samples.append(('REPLAY_%d'%index,str(path.resolve()),[0,0,320,240] if a.full_frame else [80,0,160,128]))
    if len(samples)>8:raise SystemExit('At most eight numerical reference cases')
    header='''#pragma once
#include <stdint.h>
struct LampGoldenCase {const char *name;const int8_t *input;const int8_t *expected;};
extern const unsigned char kLampModel[];
extern const unsigned int kLampModelLen;
extern const LampGoldenCase kLampGolden[];
extern const unsigned int kLampGoldenCount;
'''
    source='#include "lampnet_model_data.h"\n'+array('kLampModel',blob)+'const unsigned int kLampModelLen = '+str(len(blob))+';\n'
    expected=[]
    for index,(name,path,rect) in enumerate(samples):
        x,y,w,h=rect;im=cv2.imread(str(assets/path))[y:y+h,x:x+w]
        im=cv2.cvtColor(cv2.resize(im,(160,160)),cv2.COLOR_BGR2RGB)
        data=np.clip(np.rint((im.astype(np.float32)/127.5-1)/scale+zero),-128,127).astype(np.int8)[None]
        interpreter.set_tensor(ti['index'],data);interpreter.invoke();result=interpreter.get_tensor(to['index'])
        source+=array('kInput%d'%index,data.tobytes(),True)+array('kOutput%d'%index,result.tobytes(),True)
        expected.append('{"'+name+'",kInput%d,kOutput%d}'%(index,index))
    source+='const LampGoldenCase kLampGolden[] = {'+','.join(expected)+'};\nconst unsigned int kLampGoldenCount='+str(len(samples))+';\n'
    (sketch/'lampnet_model_data.h').write_text(header);(sketch/'lampnet_model_data.cpp').write_text(source)
    report=dict(model=str(model_path),model_bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),input_shape=[int(n) for n in ti['shape']],output_shape=[int(n) for n in to['shape']],golden_cases=[s[0] for s in samples],golden_backend='TFLite BUILTIN_REF (matches portable Micro kernels)',mode='MODEL_LOCATOR' if a.model_locator else 'SHADOW_ONLY',controls_start=a.model_locator,search_roi=[0,0,320,240] if a.full_frame else [80,0,160,128],dual_core_requested=a.dual_core,source_sketch=str(src))
    (a.out/'bundle.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
