"""Train a compact lamp segmentation model from local, manually annotated history.

The independent Maix 'large' recording is excluded from training. JPEG frames
from one recording remain correlated: all reported in-session scores are only
diagnostics. No network uploads, camera control, serial or motor I/O occur here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import cv2
import numpy as np
import tensorflow as tf

SIZE, GRID, SEED = 160, 40, 20261109
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.keras.utils.set_random_seed(SEED)


def load_sources(repo):
    rows = []
    recording = repo / '.maixpy/lamp-recordings/lamp-trial-20261107-a'
    frames = [json.loads(s) for s in (recording/'events.jsonl').read_text().splitlines()]
    frames = [r for r in frames if r['kind'] == 'frame']
    # Visually reviewed keyframe polygons around the emitting panel, excluding
    # the cardboard housing and the bright horizontal beam behind it.
    keyframes = [
        (0, [[124,27],[201,34],[198,103],[113,96]]),
        (42, [[123,27],[202,32],[198,104],[112,97]]),
        (51, [[124,24],[201,29],[195,103],[112,99]]),
        (62, [[123,22],[201,28],[197,103],[113,96]]),
        (72, [[124,22],[200,27],[195,99],[113,96]]),
        (91, [[124,24],[201,29],[197,102],[113,99]]),
    ]
    for i, r in enumerate(frames):
        if i % 8: continue
        t = r['t']
        for (t0,p0),(t1,p1) in zip(keyframes,keyframes[1:]):
            if t0 <= t <= t1:
                poly = np.array(p0)+(np.array(p1)-p0)*(t-t0)/(t1-t0)
                break
        rows.append(dict(path=str((recording/r['file']).relative_to(repo)),
                         polygon=poly.tolist(), group='esp32-room-20261107',
                         split='train', t=t, crop=[80,0,160,128]))
    history = repo/'tests/fixtures/esp_start_signal_history'
    for name,poly,crop,split in [
        ('small',[[139,20],[154,20],[154,33],[139,33]],[98,0,100,80],'train'),
        ('large',[[135,4],[153,4],[154,27],[135,27]],[95,0,100,80],'validation'),
    ]:
        for p in sorted(history.glob(name+'-*.png')):
            rows.append(dict(path=str(p.relative_to(repo)),polygon=poly,
                             group='maix-'+name+'-20261031',split=split,crop=crop))
    for p in sorted((history.parent/'start_signal_jitter_20261031').glob('*.png')):
        rows.append(dict(path=str(p.relative_to(repo)),polygon=[],
                         group='maix-jitter-20261031',split='validation',crop=[80,0,160,72]))
    real=repo/'.maixpy/android-start-20261109/reference/real-lamp-off.png'
    if real.exists():
        rows.append(dict(path=str(real.relative_to(repo)),polygon=[[1020,348],[1183,349],[1183,503],[1020,503]],
                         preprocess_crop=[970,322,870,206],crop=[0,0,105,76],
                         negative_crops=[[88,0,232,76],[88,3,83,70],[165,3,150,70]],
                         group='real-gantry-off-user-reference',split='train',
                         provenance='User-provided 2388x1332 screenshot, left square OFF panel. Right green/red digits are hard negatives. No fabricated ON state.'))
    return rows


def read(repo,row):
    image = cv2.cvtColor(cv2.imread(str(repo/row['path'])),cv2.COLOR_BGR2RGB)
    if 'preprocess_crop' not in row:image=image[:,:320]
    mask = np.zeros(image.shape[:2],np.uint8)
    if row['polygon']:
        cv2.fillPoly(mask,[np.round(row['polygon']).astype(np.int32)],255)
    if 'preprocess_crop' in row:
        x,y,w,h=row['preprocess_crop'];image=image[y:y+h,x:x+w];mask=mask[y:y+h,x:x+w]
        image=cv2.resize(image,(320,round(h*320/w)));mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
    return image,mask


def crop(image, mask, rect):
    x,y,w,h=rect
    matrix=np.array([[SIZE/w,0,-x*SIZE/w],[0,SIZE/h,-y*SIZE/h]],np.float32)
    return (cv2.warpAffine(image,matrix,(SIZE,SIZE),borderMode=cv2.BORDER_REPLICATE),
            cv2.resize(cv2.warpAffine(mask,matrix,(SIZE,SIZE)),(GRID,GRID),interpolation=cv2.INTER_AREA))


def samples(repo,rows,count):
    rng=np.random.default_rng(SEED)
    loaded=[(r,*read(repo,r)) for r in rows if r['split']=='train']
    xx,yy=[],[]
    groups={g:[v for v in loaded if v[0]['group']==g] for g in set(v[0]['group'] for v in loaded)}
    group_names=sorted(groups)
    for i in range(count):
        group=groups[group_names[i%len(group_names)]]
        row,im,mask=group[int(rng.integers(len(group)))]
        x,y,w,h=row['crop']
        scale=np.exp(rng.uniform(np.log(.55),np.log(3.2)))
        w,h=w*scale,h*scale
        cx=x+row['crop'][2]/2+rng.uniform(-.24,.24)*w
        cy=y+row['crop'][3]/2+rng.uniform(-.2,.3)*h
        rect=[cx-w/2,cy-h/2,w,h]
        # Real background-only crops create negatives without fabricating lamps
        # or copying target pixels onto another scene.
        if i%4==0:
            rect=([0,120,130,110] if i%8==0 else [210,0,110,120])
            if row.get('negative_crops'):rect=row['negative_crops'][int(rng.integers(len(row['negative_crops']))) ]
        xim,ymask=crop(im,mask,rect)
        angle=rng.uniform(-7,7)
        mat=cv2.getRotationMatrix2D((SIZE/2,SIZE/2),angle,1)
        xim=cv2.warpAffine(xim,mat,(SIZE,SIZE),borderMode=cv2.BORDER_REFLECT_101)
        mat2=cv2.getRotationMatrix2D((GRID/2,GRID/2),angle,1)
        ymask=cv2.warpAffine(ymask,mat2,(GRID,GRID))
        if rng.random()<.5: xim,ymask=xim[:,::-1],ymask[:,::-1]
        gain=rng.uniform(.65,1.3); gamma=rng.uniform(.75,1.3)
        xim=np.clip(((xim.astype(np.float32)/255)**gamma)*gain*255+rng.uniform(-12,12),0,255)
        if rng.random()<.25:xim=cv2.GaussianBlur(xim,(3,3),.6)
        xx.append(xim.astype(np.float32)/127.5-1)
        yy.append((ymask.astype(np.float32)/255)[...,None])
    return np.stack(xx),np.stack(yy)


def loss(y,p):
    bce=tf.keras.backend.binary_crossentropy(y,p)
    weighted=tf.reduce_mean(bce*(1+2*y))
    dice=(2*tf.reduce_sum(y*p,axis=[1,2,3])+1)/(tf.reduce_sum(y+p,axis=[1,2,3])+1)
    return weighted+.5*tf.reduce_mean(1-dice)


def build():
    base=tf.keras.applications.MobileNetV2(input_shape=(SIZE,SIZE,3),alpha=.35,
                                         include_top=False,weights='imagenet')
    encoder=tf.keras.Model(base.input,[base.get_layer(n).output for n in
                           ['block_3_expand_relu','block_6_expand_relu','block_13_expand_relu']])
    encoder.trainable=False
    inputs=[tf.keras.Input(shape=t.shape[1:]) for t in encoder.outputs]
    hi,mid,low=inputs
    def conv(t,n):
        return tf.keras.layers.Conv2D(n,3,padding='same',activation='relu')(t)
    z=conv(low,24)
    z=tf.keras.layers.UpSampling2D(2,interpolation='bilinear')(z)
    z=conv(tf.keras.layers.Concatenate()([z,mid]),24)
    z=tf.keras.layers.UpSampling2D(2,interpolation='bilinear')(z)
    z=conv(tf.keras.layers.Concatenate()([z,hi]),16)
    mask=tf.keras.layers.Conv2D(1,1,activation='sigmoid',name='lamp_probability')(z)
    head=tf.keras.Model(inputs,mask,name='lamp_mask_head')
    inp=tf.keras.Input((SIZE,SIZE,3),name='rgb_minus1_plus1')
    model=tf.keras.Model(inp,head(encoder(inp)),name='LampNet160')
    return encoder,head,model


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epochs',type=int,default=28)
    ap.add_argument('--samples',type=int,default=480);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    rows=load_sources(a.repo)
    (a.output/'annotations.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print('sources',len(rows),'train',sum(r['split']=='train' for r in rows),flush=True)
    x,y=samples(a.repo,rows,a.samples)
    encoder,head,model=build()
    print('extracting frozen ImageNet features',flush=True)
    features=encoder.predict(x,batch_size=16,verbose=0)
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=loss)
    hist=head.fit(features,y,batch_size=16,epochs=a.epochs,verbose=2)
    model.save(a.output/'lamp_model.keras')
    # Default optimization compresses constant weights; inputs/outputs stay float.
    converter=tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types=[tf.float16]
    blob=converter.convert();(a.output/'lamp_model.tflite').write_bytes(blob)
    interpreter=tf.lite.Interpreter(model_content=blob,num_threads=4);interpreter.allocate_tensors()
    tin,tout=interpreter.get_input_details()[0],interpreter.get_output_details()[0]
    results=[];tiles=[]
    for row in rows:
        im,mask=read(a.repo,row);xim,ym=crop(im,mask,row['crop'])
        inp=xim.astype(np.float32)[None]/127.5-1
        interpreter.set_tensor(tin['index'],inp);interpreter.invoke()
        pred=interpreter.get_tensor(tout['index'])[0,:,:,0]
        expected=ym>=128;actual=pred>=.55
        union=np.logical_or(actual,expected).sum()
        iou=float(np.logical_and(actual,expected).sum()/union) if union else None
        results.append(dict(path=row['path'],group=row['group'],split=row['split'],
                            iou=iou,predicted_pixels=int(actual.sum()),label_pixels=int(expected.sum())))
        if row['split']!='train' or row['group']=='real-gantry-off-user-reference' or len(tiles)<8:
            show=xim.copy();heat=cv2.resize((pred*255).astype(np.uint8),(SIZE,SIZE))
            show[heat>=140]=(show[heat>=140]*.4+np.array([0,255,0])*.6).astype(np.uint8)
            cv2.putText(show,('val ' if row['split']!='train' else 'train ')+('%.2f'%iou if iou is not None else 'negative'),(2,155),0,.35,(255,80,80),1)
            tiles.append(cv2.cvtColor(show,cv2.COLOR_RGB2BGR))
    while len(tiles)%6:tiles.append(np.zeros((SIZE,SIZE,3),np.uint8))
    cv2.imwrite(str(a.output/'evaluation.jpg'),cv2.vconcat([cv2.hconcat(tiles[i:i+6]) for i in range(0,len(tiles),6)]))
    report=dict(name='LampNet160 MobileNetV2 alpha0.35 frozen encoder + learned segmentation decoder',
                version='0.1.0-prototype',seed=SEED,input=[1,SIZE,SIZE,3],output=list(tout['shape'].astype(int)),
                bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),parameters=model.count_params(),
                augmented_training_images=len(x),epochs=a.epochs,final_loss=float(hist.history['loss'][-1]),
                training_groups=sorted(set(r['group'] for r in rows if r['split']=='train')),
                heldout_groups=sorted(set(r['group'] for r in rows if r['split']!='train')),
                limitation='Prototype: two positive training sessions plus at most one user-provided real gantry OFF screenshot; no Xiaomi camera training data or real gantry ON samples. Historical source-heldout groups were inspected during model tuning and are validation, not an untouched test set. Frame-level IoU is not start-event accuracy. Independent complete ON/OFF and negative-sequence tests remain required.',
                results=results)
    # Convert numpy scalar shapes to plain integers for strict JSON compatibility.
    report['output']=[int(n) for n in report['output']]
    (a.output/'training-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
