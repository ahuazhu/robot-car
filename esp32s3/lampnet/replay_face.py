"""Evaluate cached model outputs using the exact ESP32 C++ face locator.

Cache is produced separately by the unchanged INT8 model; no training occurs.
"""
import argparse,html,json
from pathlib import Path
import cv2
import numpy as np
from esp32s3.lampnet.native_locator import NativeLocator

def iou(a,b):
    x,y,w,h=a;u,v,p,q=b
    n=max(0,min(x+w,u+p)-max(x,u))*max(0,min(y+h,v+q)-max(y,v))
    return n/(w*h+p*q-n) if w*h+p*q>n else 0

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','samples','out']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);cache=np.load(a.cache);samples=json.loads(a.samples.read_text());rows=[];locator=NativeLocator()
    try:
        for i,source in enumerate(samples):
            result,_=locator.locate(cache['probability'][i],cache['rgb'][i]);reference=source.get('box');r=dict(source=source,**result)
            r['old_iou']=iou(result['coarse_box'],reference) if reference else None;r['new_iou']=iou(result['box'],reference) if reference else None;rows.append(r)
    finally:locator.close()
    groups=[]
    for name in dict.fromkeys(x['source']['group'] for x in rows):
        items=[r for r in rows if r['source']['group']==name]
        groups.append(dict(group=name,n=len(items),old_valid=sum(r['coarse_valid'] for r in items),new_valid=sum(r['valid'] for r in items),lost=sum(r['coarse_valid'] and not r['valid'] for r in items),old_median_iou=float(np.median([r['old_iou'] for r in items if r['old_iou'] is not None])) if any(r['old_iou'] is not None for r in items) else None,new_median_iou=float(np.median([r['new_iou'] for r in items if r['new_iou'] is not None])) if any(r['new_iou'] is not None for r in items) else None))
    pictures=[]
    for n,r in enumerate(x for x in rows if x['source']['group']=='actual_tuning'):
        im=cv2.resize(cv2.imread(r['source']['path']),(640,480))
        for b,color in [(r['coarse_box'],(60,60,255)),(r['box'],(70,255,70)),(r['source']['box'],(0,230,255))]:
            x,y,w,h=[round(v*2) for v in b];cv2.rectangle(im,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(im,(0,440),(640,480),(15,23,42),-1);cv2.putText(im,'%s  IoU %.2f -> %.2f'%(r['source']['phase'],r['old_iou'],r['new_iou']),(12,465),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(a.out/('frame-%02d.jpg'%n)),im);pictures.append(im)
    cv2.imwrite(str(a.out/'comparison.jpg'),cv2.hconcat([pictures[0],pictures[3],pictures[6]]));cv2.imwrite(str(a.out/'all-frames.jpg'),cv2.vconcat([cv2.hconcat(pictures[i:i+3]) for i in range(0,len(pictures),3)]))
    report=dict(method='Identical firmware C++ locator on cached unchanged INT8 outputs, RGB block means decoded from quantized inputs.',model_changed=False,groups=groups,frames=rows,limitations=['Actual tuning 9 and additional 28 frames share one pose and were inspected during development; neither is an independent accuracy estimate.','Historical frames overlap model training data; results are regression checks only.','Manual reference boxes have about +/-3px uncertainty.','Candidate false positives are separate from confirmed ON->OFF events.','Historical large Maix lamps remain poorly localized; background candidates remain.'])
    (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    table=''.join('<tr><td>%s</td><td>%d</td><td>%d → %d</td><td>%s</td></tr>'%(html.escape(g['group']),g['n'],g['old_valid'],g['new_valid'],'—' if g['old_median_iou'] is None else '%.2f → %.2f'%(g['old_median_iou'],g['new_median_iou'])) for g in groups)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>灯面细化回测</title><style>body{max-width:1200px;margin:24px auto;padding:0 18px;background:#0f172a;color:#e2e8f0;font:17px system-ui;line-height:1.7}h1{font-size:28px}img{width:100%;border-radius:10px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #475569;padding:8px;text-align:left}a{color:#38bdf8}small{color:#94a3b8}</style><h1>模型粗定位 → 灯面细化 → 锁定后判断熄灯</h1><p>模型权重未变。高置信核心、亮度/颜色与连通域共同收紧灯面；弱灯保留原候选作为回退。稳定亮灯后冻结像素区域，熄灯阶段不再重新找灯。</p><p>红框：原模型候选　绿框：细化候选　黄框：人工参考</p><img src="comparison.jpg"><table><tr><th>样本组</th><th>图片</th><th>有效候选</th><th>IoU中位数</th></tr>'''+table+'''</table><p>9张调参图片及另外28张图片来自同一摆位，均已在开发中检查，不能当作独立准确率。历史图也与训练数据重叠，仅用于回归检查。</p><p>仍有局限：历史大灯定位较差；17张干扰图中的10个原有误候选仍存在。候选不等于发车，必须经过稳定亮灯及熄灯确认。新距离/摆位的实物测试另行记录。</p><p><a href="report.json">逐帧JSON</a> · <a href="all-frames.jpg">全部9张对照</a></p><img src="all-frames.jpg"></html>'''
    (a.out/'index.html').write_text(page);print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
