"""Replay images; counts are diagnostics, NOT a lane-following success metric."""
import argparse,json,sys,time
from pathlib import Path
import cv2
import numpy as np
from vision import LaneVision,annotate,bird_view
p=argparse.ArgumentParser();p.add_argument('images',nargs='+');p.add_argument('--output',required=True);p.add_argument('--calibration')
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
cal=json.loads(Path(a.calibration).read_text()) if a.calibration else None
v=LaneVision(cal);report=[]
for i,name in enumerate(a.images):
 im=cv2.imread(name)
 if im is None:raise ValueError('Unreadable image: '+name)
 start=time.perf_counter();r=v.detect(im);ms=(time.perf_counter()-start)*1000
 destination=out/('%03d-%s.jpg'%(i,Path(name).stem))
 cv2.imwrite(str(destination),np.hstack((im,annotate(im,r),bird_view(r))))
 report.append(dict(image=name,counts=r['counts'],milliseconds=round(ms,2),calibrated=r['calibrated'],motor_enabled=False,preview=str(destination)))
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
