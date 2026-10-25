import importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'maixcam2/curve_follow'))
spec=importlib.util.spec_from_file_location('curve_actual_tracker',ROOT/'maixcam2/curve_follow/tracker.py')
tracker=importlib.util.module_from_spec(spec);spec.loader.exec_module(tracker)
from controller import VisionRecovery,target_command
from geometry import choose_target

class ActualMemoryTraceTests(unittest.TestCase):
    def test_road_recovery_on_fixed_historical_commands_is_preserved(self):
        data=json.loads((ROOT/'tests/fixtures/curve_memory_actual.json').read_text())
        t=tracker.TargetTracker(data['calibration']);initial=data['initial']
        t.path=np.asarray(initial['path']);t.last_time=initial['last_time'];t.last_seen=initial['last_seen'];t.confidence=initial['confidence'];t.mode=initial['mode']
        # Fixed historical sensor/command trace: road-model association remains
        # responsible for confirmation; there is no second target state machine.
        guard=VisionRecovery();seen={}
        for row in data['rows']:
            m=row['measurement']
            m=dict(m,path=np.asarray(m['path'])) if m is not None else None
            t.pixels.measure=lambda *args,m=m:(m,{})
            result,_=t.update(None,row['captured'],row['v'],row['omega'])
            command=target_command(result,.15,1.2)
            if row['detection_ms']>280:command=None
            self.assertIsNone(guard.update(result,command,row['decision_at'],row['detection_ms'],280,stop_verified=True),(row['frame'],result))
            seen[row['frame']]=result
            if result['state']=='CONFIRMING':self.assertIsNone(command)
        self.assertTrue(seen[868]['observed'])
        self.assertIsNotNone(command)

    def test_white_road_markings_do_not_break_the_observed_boundary(self):
        import cv2
        raw=cv2.imread(str(ROOT/'tests/fixtures/curve_white_markings.png'))
        self.assertIsNotNone(raw)
        data=json.loads((ROOT/'tests/fixtures/curve_memory_actual.json').read_text())
        pixels=tracker.CurvePixels(data['calibration'])
        measured,details=pixels.measure(raw)
        self.assertIsNotNone(measured)
        self.assertGreater(details['fits']['left']['length'],.60)
        self.assertLess(np.linalg.norm(measured['path'],axis=1).min(),.20)

if __name__=='__main__':unittest.main()
