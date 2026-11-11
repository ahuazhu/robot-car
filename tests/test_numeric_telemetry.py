"""Live JSON must preserve numerical flags instead of killing the driving loop."""
from tests.support.driving import road
from robot.core.telemetry import dumps
from robot.navigation.profile_speed_control import ProfileSpeedControl
from robot.navigation.speed_profile import SpeedLimits
from robot.interface.preview_transport import LiveStream
import json
import unittest
from urllib.request import urlopen
import numpy as np


class NumericTelemetryTests(unittest.TestCase):
    def test_numpy_values_preserve_json_boolean_number_and_array_types(self):
        values=dict(flag=np.bool_(True),count=np.int64(7),speed=np.float32(.125),
                    points=np.array([[1.,2.]]),mask=np.array([True,False]))
        decoded=json.loads(dumps(values,allow_nan=False))
        self.assertIs(decoded['flag'],True)
        self.assertIs(type(decoded['count']),int)
        self.assertEqual(decoded['speed'],.125)
        self.assertEqual(decoded['points'],[[1.,2.]])
        self.assertEqual(decoded['mask'],[True,False])
        with self.assertRaises(TypeError):dumps(dict(unknown=object()))
        with self.assertRaises(ValueError):dumps(dict(value=np.float32('nan')),allow_nan=False)

    def test_clearance_limit_and_applied_numpy_effort_reproduce_field_type(self):
        p=ProfileSpeedControl(SpeedLimits(.3,.15,.3,.2,.35,.15),218,.4,controller='DWPP')
        t=road(reference_sigma_m=.005,road_model=dict(persistent_history=True))
        stream=LiveStream(0)
        try:
            base='http://127.0.0.1:%d'%stream.server.server_address[1]
            for i in range(15):
                now=1.+i*.18
                p.applied(now-.1,(np.float64(60),np.float64(70)))
                speed=(.1,.0,.02)[i%3]
                command=p.command(t,t['control_path_m'],now,speed,2.,.005,.075,True,
                    measured_yaw_rate=0.,clearance_limit_m_s=np.float64(.05))
                self.assertIsNotNone(command)
                self.assertIs(type(p.status['initial_overspeed']),bool)
                self.assertEqual(p.status['initial_overspeed'],speed>.05)
                # The producer is JSON-safe even without the boundary adapter.
                json.dumps(p.status,allow_nan=False)
                # Future numeric leaves must be converted at BOTH real HTTP
                # endpoints and the shared logger, not by default=str.
                status=dict(p.status,numeric_flag=np.bool_(i%2),numeric_count=np.int32(i))
                json.loads(dumps(dict(steering=status),allow_nan=False))
                stream.metrics=dict(steering=status)
                stream.control.update(False,True,'TEST',(0,0),readiness=dict(steering=status))
                for endpoint in ('/status','/vision'):
                    with urlopen(base+endpoint,timeout=2) as r:
                        self.assertEqual(r.status,200)
                        data=json.load(r)
                    plan=(data.get('readiness',data))['steering']
                    self.assertIs(type(plan['initial_overspeed']),bool)
                    self.assertIs(type(plan['numeric_flag']),bool)
                    self.assertIs(type(plan['numeric_count']),int)
        finally:
            stream.stop()

if __name__=='__main__':unittest.main()
