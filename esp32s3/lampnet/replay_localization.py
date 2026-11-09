"""Replay the deployed INT8 locator against manually annotated recorded frames.

This reproduces lampnet_shadow.h candidate selection. It does not train a model,
open hardware, ARM the car, or interpret a rejected/missing candidate as OFF.
"""
import argparse,html,itertools,json,hashlib
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

def locate(prob):
    labels_count,labels,stats,_=cv2.connectedComponentsWithStats((prob>=.55).astype(np.uint8),connectivity=4)
    ranked=sorted(range(1,labels_count),key=lambda k:(-int(stats[k,cv2.CC_STAT_AREA]),k))
    if not ranked:return dict(valid=False,box=[80,0,0,0],confidence=0,area=0),np.zeros((40,40),np.uint8)
    best=ranked[0];x,y,w,h,area=map(int,stats[best]);mask=(labels==best).astype(np.uint8)
    second=int(stats[ranked[1],cv2.CC_STAT_AREA]) if len(ranked)>1 else 0
    confidence=float(prob[mask.astype(bool)].mean());aspect=w*1.25/h;fill=area/(w*h)
    valid=area>=6 and area<1040 and confidence>=.65 and .5<=aspect<=1.9 and fill>=.45 and not(second>=6 and second>area*.65)
    return dict(valid=valid,box=[80+x*4,y*128//40,w*4,h*128//40],confidence=confidence,area=area,second_area=second,aspect=aspect,fill=fill),mask

def overlap(a,b):
    x,y,w,h=a;u,v,p,q=b
    intersection=max(0,min(x+w,u+p)-max(x,u))*max(0,min(y+h,v+q)-max(y,v))
    return intersection/(w*h+p*q-intersection) if w*h+p*q>intersection else 0,intersection

def sensitivity(box,reference,tolerance):
    x,y,w,h=reference;values=[]
    for delta in itertools.product([-tolerance,tolerance],repeat=4):
        x0,y0,x1,y1=x+delta[0],y+delta[1],x+w+delta[2],y+h+delta[3]
        if x1>x0 and y1>y0:values.append(overlap(box,[x0,y0,x1-x0,y1-y0])[0])
    return [min(values),max(values)]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['model','trial','annotations','out']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    annotation=json.loads(a.annotations.read_text());runtime=tf.lite.Interpreter(model_path=str(a.model),experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,num_threads=2);runtime.allocate_tensors()
    ti,to=runtime.get_input_details()[0],runtime.get_output_details()[0]
    assert list(ti['shape'])==[1,160,160,3] and list(to['shape'])==[1,40,40,1]
    ins,inz=ti['quantization'];outs,outz=to['quantization']
    results=[];pictures=[]
    for i,entry in enumerate(annotation['frames']):
        original=cv2.imread(str(a.trial/entry['file']));assert original is not None and original.shape[:2]==(240,320)
        crop=cv2.cvtColor(cv2.resize(original[:128,80:240],(160,160)),cv2.COLOR_BGR2RGB)
        q=np.clip(np.rint((crop.astype(np.float32)/127.5-1)/ins+inz),-128,127).astype(np.int8)[None]
        runtime.set_tensor(ti['index'],q);runtime.invoke();prob=(runtime.get_tensor(to['index'])[0,:,:,0].astype(np.float32)-outz)*outs
        prediction,mask=locate(prob);reference=entry['lamp_face_box'];box=prediction['box'];iou,intersection=overlap(box,reference)
        tolerance=sensitivity(box,reference,annotation['edge_uncertainty_px'])
        result=dict(**entry,prediction=prediction,box_iou=iou,iou_edge_uncertainty_range=tolerance,reference_coverage=intersection/(reference[2]*reference[3]),box_outside_reference_fraction=1-intersection/(box[2]*box[3]) if box[2]*box[3] else None,width_ratio=box[2]/reference[2])
        results.append(result)
        show=cv2.resize(original,(640,480));roi=show[:256,160:480];large=cv2.resize(mask,(320,256),interpolation=cv2.INTER_NEAREST).astype(bool)
        roi[large]=(roi[large]*.78+np.array([20,20,255])*.22).astype(np.uint8)
        for b,color in [(box,(0,0,255)),(reference,(0,255,255))]:
            x,y,w,h=b;cv2.rectangle(show,(x*2,y*2),((x+w)*2,(y+h)*2),color,2)
        title='%s  %s  IoU %.2f  gate %s'%(entry['phase'],Path(entry['file']).stem,iou,'PASS' if prediction['valid'] else 'REJECT')
        cv2.rectangle(show,(0,440),(640,480),(15,23,42),-1);cv2.putText(show,title,(10,465),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
        filename='frame-%02d.jpg'%i;cv2.imwrite(str(a.out/filename),show);result['annotated_image']=filename;pictures.append(show)
    contact=cv2.vconcat([cv2.hconcat(pictures[i:i+3]) for i in range(0,len(pictures),3)])
    cv2.imwrite(str(a.out/'all-frames.jpg'),contact)
    cv2.imwrite(str(a.out/'comparison.jpg'),cv2.hconcat([pictures[1],pictures[4],pictures[7]]))
    phases=[]
    for phase in dict.fromkeys(x['phase'] for x in results):
        rows=[r for r in results if r['phase']==phase]
        phases.append(dict(phase=phase,n=len(rows),gate_passed=sum(x['prediction']['valid'] for x in rows),median_iou=float(np.median([x['box_iou'] for x in rows])),min_iou=min(x['box_iou'] for x in rows),max_iou=max(x['box_iou'] for x in rows),median_width_ratio=float(np.median([x['width_ratio'] for x in rows])),median_outside_fraction=float(np.median([x['box_outside_reference_fraction'] for x in rows])),iou_at_least_50=sum(x['box_iou']>=.5 for x in rows),iou_at_least_70=sum(x['box_iou']>=.7 for x in rows)))
    tight_count=sum(x['box_iou']>=.7 and x['prediction']['valid'] for x in results)
    accurate=tight_count==len(results) and bool(results)
    report=dict(result='LOCALIZATION_ACCURATE_ON_REPLAY' if accurate else 'LOCALIZATION_NOT_ACCURATE',acceptance_criterion='All annotated frames must pass the existing geometry gate and reach box IoU >= 0.70 for tight localization',tight_localization_count=tight_count,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),firmware='0.4.8-lampnet-espnn',method='Desktop BUILTIN_REF replay of the unchanged deployed INT8 model on JPEGs, reproducing firmware connected-component selection and geometry gate. Not a new live ESP32 recording.',controls_start=False,training_changed=False,annotation_note=annotation['note'],edge_uncertainty_px=annotation['edge_uncertainty_px'],n=len(results),phases=phases,median_iou=float(np.median([x['box_iou'] for x in results])),iou_at_least_50=sum(x['box_iou']>=.5 for x in results),iou_at_least_70=sum(x['box_iou']>=.7 for x in results),all_iou_below_70_even_with_edge_uncertainty=all(x['iou_edge_uncertainty_range'][1]<.7 for x in results),frames=results,limitations=['Nine correlated samples from one camera position; not a general detection accuracy/recall estimate.','Reference boxes are approximate manual annotations, with +/-3px sensitivity analysis.','Candidate boxes are shown even if the firmware geometry gate would hide them.','Device/host bit-exact agreement was separately verified on six reference inputs, not all frames of this replay.'])
    (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    table=''.join('<tr><td>%s</td><td>%d / %d</td><td>%.2f</td><td>%.2f倍</td><td>%.0f%%</td></tr>'%(html.escape(p['phase']),p['gate_passed'],p['n'],p['median_iou'],p['median_width_ratio'],p['median_outside_fraction']*100) for p in phases)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>信号灯定位回测</title><style>body{margin:22px auto;max-width:1200px;padding:0 20px;background:#0f172a;color:#e2e8f0;font:17px system-ui;line-height:1.7}h1{font-size:27px;margin-bottom:10px}img{width:100%;border-radius:8px}p{margin:10px 0}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #475569;padding:7px}small{color:#94a3b8}a{color:#38bdf8}</style><h1>回测结论：仍不能准确框住灯面</h1><p>当前加速版使用同一个INT8模型。已回放9张亮灯、灭灯、再次亮灯画面，未重新训练、未修改检测阈值、未启动小车。</p><p><b>黄框：人工灯面参考；红框与淡红色掩膜：模型候选。</b>几何检查拒绝的候选也显示出来，以便看清拒绝原因。</p><img src="comparison.jpg" alt="亮灯、灭灯、再次亮灯的定位对照"><table><tr><th>状态</th><th>通过原几何检查</th><th>框重合度中位数</th><th>框宽/灯宽</th><th>框内背景占比</th></tr>'''+table+'''</table><p>模型框覆盖灯面外的大量背景、光晕与横梁；即使候选通过原几何检查，也不等于框准了。硬件提速没有改变这些分割结果。</p><p><small>框重合度指IoU，越接近1越贴合。本次人工参考框边缘存在约±3像素误差；已做敏感性检查。同一摆位的9张相关画面不能代表泛化准确率。这里是电脑端JPEG回放，不是新录制的板上视频。</small></p><p><a href="all-frames.jpg">查看全部9张对照</a> · <a href="report.json">逐帧结果</a></p><img src="all-frames.jpg" alt="全部9张标注对照"></html>'''
    if accurate:
        page=page.replace('回测结论：仍不能准确框住灯面','回测结论：本组画面定位贴合').replace('模型框覆盖灯面外的大量背景、光晕与横梁；即使候选通过原几何检查，也不等于框准了。硬件提速没有改变这些分割结果。','本组画面的候选均通过几何检查且框重合度达到0.70，仍需使用不同摆位和背景验证泛化。')
    (a.out/'index.html').write_text(page)
    print(json.dumps({k:v for k,v in report.items() if k not in ['frames','limitations']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
