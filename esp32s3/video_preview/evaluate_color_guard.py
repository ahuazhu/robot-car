"""Offline labeled-camera replay; no network, serial, camera or motor I/O."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from .replay import NativeDetector
except ImportError:  # Running the file directly from its directory.
    from replay import NativeDetector



def read_frames(spec, data_root):
    root = data_root/spec['root']
    records = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines() if line.strip()]
    frames = [r for r in records if r.get('kind') == 'frame' and spec['start'] <= r['t'] <= spec['end']]
    out = []
    for row in frames:
        path = root/row['file']; image = cv2.imread(str(path))
        if image is None or image.shape != (240,320,3):
            raise ValueError('Not an original 320x240 ESP frame: '+str(path))
        out.append(dict(t=float(row['t']),file=path.relative_to(data_root).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),image=image))
    if not out:raise ValueError('No original frames in the declared time window: '+str(root))
    return out


def replay(detector, frames, roi, cycle, paced=False):
    detector.cancel()
    if not detector.manual(roi):raise ValueError('Invalid manual ROI: '+repr(roi))
    detector.cycle(cycle); detector.arm()
    first_lock = first_hit = first_confirm = None
    phases = Counter(); reasons = Counter(); transitions = []; previous = None
    for i, row in enumerate(frames):
        # Paced experiments deliberately assign protocol time to DISTINCT images;
        # they do not fill missing frames or claim source-camera timing/accuracy.
        now = i*210 if paced else round((row['t']-frames[0]['t'])*1000)
        state = detector.process(row['image'], now); diag = detector.diagnostics()
        rgb_diag = detector.rgb_diagnostics()
        phases[str(state['phase'])] += 1; reasons[diag['reason']] += 1
        point = dict(file=row['file'], source_t=row['t'], protocol_ms=now)
        if state['locked'] and first_lock is None: first_lock = point
        if state['hits'] and first_hit is None: first_hit = dict(point,rgb_diagnostic=rgb_diag)
        if state['confirmed'] and first_confirm is None: first_confirm = dict(point,confirmation_span_ms=state['elapsed'],rgb_diagnostic=rgb_diag)
        signature = (state['phase'],diag['reason'])
        if signature != previous:
            transitions.append(dict(point, phase=state['phase'], reason=diag['reason'], diagnostic=diag,rgb_diagnostic=rgb_diag))
            previous = signature
    return dict(frames=len(frames),initial_lamp_state='OFF' if cycle else 'ON',first_lock=first_lock,
                first_off_evidence=first_hit,first_confirmation=first_confirm,reasons=dict(reasons),phases=dict(phases),transitions=transitions)


def photometry(frames, box):
    x,y,w,h=box; rows=[]
    for row in frames:
        image=row['image'][y:y+h,x:x+w]; hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV); lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        b,g,r=[image[:,:,c].astype(float) for c in range(3)]
        rows.append(dict(L=float(lab[:,:,0].mean()),V=float(hsv[:,:,2].mean()),S=float(hsv[:,:,1].mean()/255),
                         red_excess=float(np.maximum(0,r-np.maximum(g,b)).mean()),green_excess=float(np.maximum(0,g-np.maximum(r,b)).mean()),
                         saturation_ge_0_25_fraction=float((hsv[:,:,1]>=64).mean()),overexposed_fraction=float((hsv[:,:,2]>=250).mean())))
    return dict(frames=len(rows),median={k:round(float(np.median([r[k] for r in rows])),4) for k in rows[0]}) if rows else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True,help='JSON input with datasets/ROI/time windows; contains no images')
    parser.add_argument('--data-root',type=Path,required=True,help='Private recordings directory; never modified')
    parser.add_argument('--header',type=Path,default=Path(__file__).with_name('start_signal.h'),help='Exact standalone detector header to compile')
    parser.add_argument('--output',type=Path,required=True,help='Replay JSON output; a sibling frame/SHA manifest is also written')
    args=parser.parse_args();inputs=json.loads(args.manifest.read_text());specs=inputs['datasets']
    result=dict(method='Actual C++ NativeDetector, full_frame=True, operator/core manual ROI, RGB565 re-quantized from original ESP JPEGs. Both OFF-cycle and already-ON modes.',
                header=str(args.header.resolve()),header_sha256=hashlib.sha256(args.header.read_bytes()).hexdigest(),
                input_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                limitations=['JPEGs differ from original camera RGB565; no physical switching latency can be measured.',
                  'No verified real red/green ON/OFF recordings were found. Available physical lamps are white/warm white, often saturated.',
                  'No independently labeled occlusion, ambient-exposure-only or white-balance-only negative recordings were found. People/glare in images do not supply event truth.',
                  'Golden/self-test images and old Maix-camera fixtures are excluded from primary real-ESP evaluation.',
                  'Frames within a recording are correlated; frame counts are not independent trials or an accuracy denominator.',
                  'Paced probes are photometric/protocol sensitivity checks only; they assign 210ms to distinct snapshots and may reorder stable windows, never source-camera accuracy or latency.'],datasets=[])
    manifest=[];detector=NativeDetector(full_frame=True,header=args.header)
    try:
        for spec in specs:
            frames=read_frames(spec,args.data_root); gaps=np.diff([r['t'] for r in frames]);windows={k:[r for r in frames if bounds and bounds[0]<=r['t']<=bounds[1]] for k,bounds in spec['windows'].items()}
            entry={k:v for k,v in spec.items()};entry['color_class']='white_or_warm_white_visual_spotcheck'
            entry['timing']=dict(frames=len(frames),median_gap_s=float(np.median(gaps)),max_gap_s=float(max(gaps)),gaps_over_750ms=int((gaps>.75).sum()),all_intervals_over_750ms=bool((gaps>.75).all()))
            entry['frame_photometry']={k:photometry(v,spec['roi']) for k,v in windows.items()}
            entry['strict_recorded_time']={mode:replay(detector,[r for r in frames if mode=='ON' or r['t']>=spec['off_start']],spec['roi'],mode=='OFF') for mode in ['OFF','ON']}
            probes={};on=windows['ON1'];off=windows['OFF'];on2=windows['ON2'];off2=windows.get('OFF2',[])
            if len(on)>=4 and len(off)>=8:
                probes['ON_to_OFF_distinct_snapshots']=dict(expected='confirm',result=replay(detector,on[:4]+off[:4],spec['roi'],False,True))
                probes['OFF_ON_OFF_steady_snapshots']=dict(expected='confirm',chronological_source_windows=bool(off2),result=replay(detector,off[:4]+(on2[:4] if on2 else on[:4])+(off2[:4] if off2 else off[4:8]),spec['roi'],True,True))
                probes['OFF_remains_OFF']=dict(expected='no_confirmation',result=replay(detector,off[:8],spec['roi'],True,True))
                probes['OFF_to_ON_without_second_OFF']=dict(expected='no_confirmation',result=replay(detector,off[:4]+(on2[:4] if on2 else on[:4]),spec['roi'],True,True))
                probes['ON_remains_ON']=dict(expected='no_confirmation',result=replay(detector,on[:8],spec['roi'],False,True))
            if len(on)>=4 and len(on2)>=4:
                probes['ON1_to_ON2_distinct_snapshots']=dict(expected='no_confirmation',result=replay(detector,on[:4]+on2[:4],spec['roi'],False,True))
            entry['paced_sensitivity_probes']=probes;result['datasets'].append(entry)
            manifest.extend(dict(dataset=spec['name'],t=r['t'],file=r['file'],sha256=r['sha256']) for r in frames)
            print(json.dumps({'dataset':spec['name'],'timing':entry['timing'],'strict':{k:{'lock':v['first_lock'],'confirmation':v['first_confirmation']} for k,v in entry['strict_recorded_time'].items()},'paced':{k:bool(v['result']['first_confirmation']) for k,v in probes.items()}},ensure_ascii=False))
        jitter=inputs.get('jitter')
        if jitter:
            samples=[]
            for i,path in enumerate(sorted((args.data_root/jitter['root']).glob('*.png'))):
                image=cv2.imread(str(path))
                if image is None:raise ValueError('Unreadable jitter frame: '+str(path))
                if jitter.get('crop_first_320_columns'):image=image[:,:320]
                if image.shape!=(240,320,3):raise ValueError('Invalid jitter frame: '+str(path))
                samples.append(dict(t=i*.21,file=path.relative_to(args.data_root).as_posix(),image=image))
            if not samples:raise ValueError('Missing jitter inputs: '+str(args.data_root/jitter['root']))
            result['supplemental_jitter']=dict(camera=jitter['camera'],source_timestamps_available=False,assigned_interval_ms=210,frames=len(samples),
                limitation='Distinct old-camera images at assigned protocol time, not original-time ESP evaluation.',
                outcomes={mode:replay(detector,samples,jitter['roi'],mode=='OFF',True) for mode in ['OFF','ON']})
    finally:detector.close()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    args.output.with_name(args.output.stem+'-frames.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('REPORT',args.output)

if __name__=='__main__':main()
