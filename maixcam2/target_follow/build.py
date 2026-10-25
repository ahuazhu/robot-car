from pathlib import Path
root=Path(__file__).resolve().parent
old=root.parent/'duckietown_drive'
transport=(old/'preview_transport.py').read_text().replace('每轮最多运行5秒','每轮最长55秒；单边可靠时持续跟踪，双边丢失仅短时预测').replace('首次使用低速测试。','转弯自动缩短前视并减速；红十字是目标，OFF VIEW 三角仅标方向。ONE 为单侧估计，PREDICT 为短时记忆。')
transport=transport.replace('小车实时画面','前视点巡线 v0.3')
(root/'preview_transport.py').write_text(transport)
main=(root/'main.py').read_text()
for line in ['from tracker import TargetTracker,target_command,motion_from_pwm,render_target','from drive_link import DriveLink','from controller import CommandRamp','from preview_transport import LiveStream']:main=main.replace(line,'')
parts=[(root/'tracker.py').read_text(),(root/'controller.py').read_text(),(root.parent/'duckietown_test/drive_link.py').read_text(),transport,main]
(root/'standalone.py').write_text("import os\nos.environ['OPENBLAS_NUM_THREADS']='1'\nos.environ['OMP_NUM_THREADS']='1'\n"+'\n'.join(parts))
print('Built target_follow')
