"""Bundle existing preview transport and upstream detector for single-file runner."""
from pathlib import Path
root=Path(__file__).resolve().parent
old=(root.parent/'lane_follow/main.py').read_text()
transport=old[old.index('LIVE_HTML ='):old.index('\ndef main():')]
transport='LIVE_HTML = '+repr((root/'preview.html').read_text())+'\n\n'+transport[transport.index('class WebControl:'):]
(root/'preview_transport.py').write_text('import time, json\n'+transport)
vision=(root/'vision.py').read_text().replace('from duckietown_core import LineDetector, ColorRange, GroundProjector, NormalizedImagePoint','')
vision=vision.replace('from lane_pose import estimate_lane_pose','')
main=(root/'main.py').read_text().replace('from vision import LaneVision, annotate, bird_view','').replace('from preview_transport import LiveStream','')
(root/'standalone.py').write_text((root/'duckietown_core.py').read_text()+'\n'+(root/'lane_pose.py').read_text()+'\n'+vision+'\nimport time,json\n'+transport+'\n'+main)
print('Built',root/'standalone.py')
