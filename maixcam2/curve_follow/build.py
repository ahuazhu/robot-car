"""Bundle the curve app and generate its preview-default deployment script."""
from pathlib import Path
import base64
import argparse
import json
import ast
parser=argparse.ArgumentParser()
parser.add_argument("--drive",action="store_true",help="Enable driving using configured start mode; default is forced preview")
args=parser.parse_args()
root=Path(__file__).resolve().parent
project=root.parents[1]
lamp_config=json.loads((root/'runtime.json').read_text()).get('start_light',{})
lamp_assets=[lamp_config['template_file']]+[entry['template_file'] for entry in lamp_config.get('extra_templates',[])]
if any(Path(name).name!=name for name in lamp_assets):raise ValueError('Invalid lamp asset path')
modules=('native_geometry','road_policy','geometry','ground_motion','road_model','road_recovery','tracker','crosswalk','controller','steering_control','drive_link','motor_channel','preview_transport','start_light','lap_counter','main')
parts=[]
for name in modules:
    source=(root/(name+'.py')).read_text()
    lines=source.splitlines(keepends=True)
    # Local imports resolve to bundled definitions. AST line spans handle
    # multi-line imports without depending on their formatting.
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.ImportFrom) and node.module in modules:
            lines[node.lineno-1:node.end_lineno]=['\n']*(node.end_lineno-node.lineno+1)
    parts.append(''.join(lines))
app="import os\nos.environ['OPENBLAS_NUM_THREADS']='1'\nos.environ['OMP_NUM_THREADS']='1'\n"+'\n'.join(parts)
compile(app,'standalone.py','exec')
(root/'standalone.py').write_text(app)
setup="from pathlib import Path\nimport os,base64,subprocess,time,hashlib\nroot=Path('/maixapp/apps/robot_curve_follow')\nroot.mkdir(parents=True,exist_ok=True)\n"
setup+='''backup=root/'backups'/time.strftime('%Y%m%d-%H%M%S',time.gmtime())
backup.mkdir(parents=True,exist_ok=True)
for name in ('main.py','calibration.json','runtime.json','curve_fit_kernel.c','curve_fit_kernel.so','app.yaml','start_light_template.png'):
    previous=root/name
    if previous.exists():(backup/name).write_bytes(previous.read_bytes())
print('DEVICE_BACKUP',str(backup),flush=True)
'''
for filename,data in [('app.yaml',(root/'app.yaml').read_bytes()),('main.py',app.encode()),('calibration.json',(root/'calibration.json').read_bytes()),('runtime.json',(root/'runtime.json').read_bytes()),('curve_fit_kernel.c',(root/'curve_fit_kernel.c').read_bytes()),*[(name,(root/name).read_bytes()) for name in lamp_assets]]:
    setup+="temporary=root/%r\ntemporary.write_bytes(base64.b64decode(%r))\n\n"%(filename+'.next',base64.b64encode(data).decode())
setup+='''compile_result=subprocess.run(['cc','-x','c','-std=c99','-O3','-fPIC','-shared',str(root/'curve_fit_kernel.c.next'),'-o',str(root/'curve_fit_kernel.so.next'),'-lm'],capture_output=True,text=True)
if compile_result.returncode:
    print(compile_result.stdout,compile_result.stderr,flush=True)
    raise RuntimeError('Native quadratic kernel compilation failed; old files retained')
import ctypes
compiled=ctypes.CDLL(str(root/'curve_fit_kernel.so.next'))
compiled.curve_fit_local
compiled.curve_agree_local
compiled.curve_fit_chain
compiled.curve_project_polyline
compiled.curve_boundary_chains
compiled.curve_select_pursuit
compiled.curve_agree_paths
for name in ('curve_fit_kernel.c','curve_fit_kernel.so','main.py','calibration.json','runtime.json','app.yaml','start_light_template.png'):
    (root/(name+'.next')).replace(root/name)
print('NATIVE_KERNEL_READY',flush=True)
# Register the same app that remote deployment runs. The old icon pointed to
# an obsolete program; preserve it on disk but remove its launcher manifest.
registry=root.parent/'app.info'
if registry.exists():(backup/'app.info').write_bytes(registry.read_bytes())
legacy=root.parent/'robot_lane_follow'
legacy_manifest=legacy/'app.yaml'
if legacy_manifest.exists():
    (backup/'legacy-app.yaml').write_bytes(legacy_manifest.read_bytes())
    if (legacy/'main.py').exists():
        (backup/'legacy-main.py').write_bytes((legacy/'main.py').read_bytes())
    legacy_manifest.replace(legacy/'app.yaml.disabled')
try:
    subprocess.run(['/usr/bin/python3',str(root.parent/'gen_app_info.py'),str(root.parent)],check=True)
    import configparser
    registered=configparser.ConfigParser(interpolation=None)
    registered.read(registry)
    if registered.get('robot_curve_follow','exec')!='main.py' or registered.has_section('robot_lane_follow'):
        raise RuntimeError('Launcher registration does not select the current app')
except Exception:
    if (backup/'legacy-app.yaml').exists():
        legacy_manifest.write_bytes((backup/'legacy-app.yaml').read_bytes())
    if (backup/'app.info').exists():registry.write_bytes((backup/'app.info').read_bytes())
    raise
# Preserve the user's launcher placement after regenerating the ID-sorted registry.
import re
text=registry.read_text()
blocks=re.split(r'(?m)(?=^\\[)',text)
target=next(b for b in blocks if b.startswith('[robot_curve_follow]\\n'))
remaining=[b for b in blocks if b is not target]
index=next(i for i,b in enumerate(remaining) if b.startswith('[basic]\\n'))+1
remaining.insert(index,target)
registry.write_text(''.join(remaining))
print('LAUNCHER_REGISTERED robot_curve_follow',flush=True)
env=dict(os.environ,MAIXPY_DEBUG_IMAGES='0',TARGET_TEST_SECONDS='0',TARGET_PREVIEW_ONLY='1',START_LIGHT_DISARMED='1')
with open('/tmp/robot_curve_follow.log','w') as log:
    p=subprocess.Popen(['/usr/bin/python3','-u',str(root/'main.py')],cwd=str(root),stdout=log,stderr=log,env=env,start_new_session=True)
import json,urllib.request
received=set();deadline=time.monotonic()+30
while time.monotonic()<deadline and p.poll() is None:
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/vision',timeout=1) as response:health=json.load(response)
        if health.get('frame',0)>0:received.add(health['frame'])
        if len(received)>=3:break
    except (OSError,ValueError):pass
    time.sleep(.25)
if len(received)<3:
    print(Path('/tmp/robot_curve_follow.log').read_text()[-5000:],flush=True)
    raise RuntimeError('App did not produce three JSON telemetry frames')
with urllib.request.urlopen('http://127.0.0.1:8000/status',timeout=2) as response:status=json.load(response)
if status['active'] or status['left'] or status['right']:raise RuntimeError('Unexpected drive state after deployment')
print('FIRST_FRAMES_READY',health['app_version'],sorted(received),'active',status['active'],flush=True)
print('PID',p.pid,'running',p.poll() is None,'SHA256',hashlib.sha256((root/'main.py').read_bytes()).hexdigest(),flush=True)
print(Path('/tmp/robot_curve_follow.log').read_text()[-3000:],flush=True)
if p.poll() is not None:raise RuntimeError('Curve app exited')
'''
setup=setup.replace("'start_light_template.png'",','.join(repr(name) for name in lamp_assets))
if args.drive:
    setup=setup.replace("TARGET_PREVIEW_ONLY='1'", "TARGET_PREVIEW_ONLY='0'")
name='install_curve_follow_drive.py' if args.drive else 'install_curve_follow.py'
(project/'.maixpy'/name).write_text(setup)
print('Built curve_follow and .maixpy/'+name+(' (configured start enabled)' if args.drive else ' (forced preview)'))
