"""Export an INT8 research artifact; this does not install ESP32 firmware."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ.setdefault('TF_USE_LEGACY_KERAS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import cv2
import numpy as np
import tensorflow as tf
from train import load_sources,samples,read,crop

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--model-dir',type=Path,required=True);a=ap.parse_args()
    model=tf.keras.models.load_model(a.model_dir/'lamp_model.keras',compile=False)
    rows=load_sources(a.repo);x,_=samples(a.repo,rows,96)
    def representative():
        for frame in x:yield [frame[None].astype(np.float32)]
    converter=tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations=[tf.lite.Optimize.DEFAULT];converter.representative_dataset=representative
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
    blob=converter.convert();(a.model_dir/'lamp_model.int8.tflite').write_bytes(blob)
    runtime=tf.lite.Interpreter(model_content=blob,num_threads=4);runtime.allocate_tensors()
    inp,out=runtime.get_input_details()[0],runtime.get_output_details()[0]
    scale,zero=inp['quantization'];oscale,ozero=out['quantization'];results=[]
    for row in rows:
        image,mask=read(a.repo,row);im,truth=crop(image,mask,row['crop'])
        data=np.clip(np.round((im.astype(np.float32)/127.5-1)/scale+zero),-128,127).astype(np.int8)[None]
        runtime.set_tensor(inp['index'],data);runtime.invoke();pred=(runtime.get_tensor(out['index'])[0,:,:,0].astype(np.float32)-ozero)*oscale
        t=truth>=128;p=pred>=.55;union=np.logical_or(t,p).sum()
        results.append(dict(path=row['path'],split=row['split'],group=row['group'],iou=float(np.logical_and(t,p).sum()/union) if union else None,predicted_cells=int(p.sum())))
    report=dict(bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),input_quantization=inp['quantization'],output_quantization=out['quantization'],operators=sorted(set(op['op_name'] for op in runtime._get_ops_details())),
                largest_single_tensor_bytes=max(int(np.prod(t['shape']))*np.dtype(t['dtype']).itemsize for t in runtime.get_tensor_details()),
                limitation='INT8 CPU replay only. Largest tensor is not peak RAM. No ESP32 build, arena sizing, operator compatibility or hardware latency verification has been performed.',results=results)
    (a.model_dir/'int8-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
if __name__=='__main__':main()
