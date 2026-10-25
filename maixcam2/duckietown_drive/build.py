from pathlib import Path
root=Path(__file__).resolve().parent
perception=root.parent/'duckietown_test'
old=(root.parent/'lane_follow/main.py').read_text()
html=old[old.index('LIVE_HTML = ')+len('LIVE_HTML = '):old.index('\nclass WebControl:')].strip()
import ast
html=ast.literal_eval(html)
html=html.replace('每轮最多运行12秒','每轮最多运行5秒').replace('左侧为摄像头原始画面；右侧为道路识别、运行状态和左右轮指令。','左侧为原图，中间为道路识别与轮速指令，右侧为地面投影。首次使用低速测试。')
html=html.replace('max-width:1280px','max-width:1440px').replace('.labels span{width:50%', '.labels span{width:33.33%').replace('<span>巡线识别</span>','<span>巡线识别</span><span>地面投影</span>')
(root/'preview.html').write_text(html)
transport='import time,json\nLIVE_HTML = '+repr(html)+'\n\n'+old[old.index('class WebControl:'):old.index('\ndef main():')]
(root/'preview_transport.py').write_text(transport)
vision=(perception/'vision.py').read_text().replace('from lane_pose import estimate_lane_pose','').replace('from duckietown_core import LineDetector, ColorRange, GroundProjector, NormalizedImagePoint','')
main=(root/'main.py').read_text()
for line in ['from vision import LaneVision,annotate,bird_view','from drive_link import DriveLink','from controller import drive_command,CommandRamp,SESSION_SECONDS,SingleSideGuard','from preview_transport import LiveStream']:main=main.replace(line,'')
parts=[(perception/'duckietown_core.py').read_text(),(perception/'lane_pose.py').read_text(),vision,(perception/'drive_link.py').read_text(),(root/'controller.py').read_text(),transport,main]
(root/'standalone.py').write_text('\n'.join(parts))
print('Built Duckietown drive test')
