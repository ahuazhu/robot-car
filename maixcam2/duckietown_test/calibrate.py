"""Record measured four-corner ground-plane calibration for preview.
Example: --points '[[xNL,yNL],[xNR,yNR],[xFR,yFR],[xFL,yFL]]'
Do not use example or guessed points as physical calibration.
"""
import argparse,json
from pathlib import Path
from vision import make_calibration
p=argparse.ArgumentParser();p.add_argument('--points',required=True);p.add_argument('--near',type=float,required=True);p.add_argument('--far',type=float,required=True);p.add_argument('--width',type=float,default=.30);p.add_argument('--image',required=True);p.add_argument('--output',required=True)
a=p.parse_args();image=Path(a.image)
if not image.is_file():p.error('Reference image missing')
cal=make_calibration(json.loads(a.points),a.near,a.far,a.width)
cal['source_image']=str(image.resolve())
Path(a.output).write_text(json.dumps(cal,indent=2));print(a.output)
