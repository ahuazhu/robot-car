import importlib.util,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('curve_web_start',Path(__file__).resolve().parents[1]/'maixcam2/curve_follow/preview_transport.py')
web=importlib.util.module_from_spec(spec);spec.loader.exec_module(web)

class WebStartTests(unittest.TestCase):
 def ready(self):
  c=web.WebControl();c.update(False,True,'WAIT START',(0,0));return c
 def test_request_survives_processing_delay_within_status_freshness(self):
  with patch.object(web.time,'monotonic',return_value=10.) as clock:
   c=self.ready();self.assertTrue(c.submit('start',0)[0]);clock.return_value=10.65
   self.assertEqual(c.take(),'start')
 def test_request_expires_at_same_freshness_boundary_as_status(self):
  with patch.object(web.time,'monotonic',return_value=10.) as clock:
   c=self.ready();c.submit('start',0);clock.return_value=11.
   self.assertIsNone(c.take())
 def test_stop_cancels_pending_start_and_invalidates_its_generation(self):
  with patch.object(web.time,'monotonic',return_value=10.):
   c=self.ready();c.submit('start',0);self.assertTrue(c.submit('stop')[0]);self.assertEqual(c.take(),'stop')
   self.assertFalse(c.submit('start',0)[0])
 def test_stale_status_cannot_enqueue_new_start(self):
  with patch.object(web.time,'monotonic',return_value=10.) as clock:
   c=self.ready();clock.return_value=11.;self.assertFalse(c.submit('start',0)[0])
 def test_not_ready_or_active_cannot_enqueue_start(self):
  with patch.object(web.time,'monotonic',return_value=10.):
   for active,ready in ((False,False),(True,True)):
    c=self.ready();c.update(active,ready,'TEST',(0,0));self.assertFalse(c.submit('start',0)[0])

class DirectStartTests(unittest.TestCase):
 def waiting(self,ready=True):
  c=web.WebControl();c.update(False,False,'WAIT START LIGHT',(0,0),readiness={'direct_start_ready':ready});return c
 def test_direct_start_can_queue_while_lamp_waiting_button_is_disabled(self):
  with patch.object(web.time,'monotonic',return_value=10.):
   c=self.waiting();self.assertFalse(c.submit('start',0)[0]);self.assertTrue(c.submit('start_now',0)[0]);self.assertEqual(c.take(),'start_now')
 def test_missing_road_or_unready_drive_still_blocks_direct_start(self):
  with patch.object(web.time,'monotonic',return_value=10.):
   c=self.waiting(False);self.assertFalse(c.submit('start_now',0)[0]);self.assertIsNone(c.take())
 def test_stop_cancels_direct_start_and_stale_generations(self):
  with patch.object(web.time,'monotonic',return_value=10.):
   c=self.waiting();c.submit('start_now',0);c.submit('stop');self.assertEqual(c.take(),'stop');self.assertIsNone(c.take());self.assertFalse(c.submit('start_now',0)[0])
 def test_direct_start_expires_when_queue_or_status_is_stale(self):
  with patch.object(web.time,'monotonic',return_value=10.) as clock:
   c=self.waiting();c.submit('start_now',0);clock.return_value=11.;self.assertIsNone(c.take());self.assertFalse(c.submit('start_now',0)[0])
 def test_active_mission_cannot_receive_a_second_direct_start(self):
  with patch.object(web.time,'monotonic',return_value=10.):
   c=self.waiting();c.update(True,False,'RUNNING',(60,60),readiness={'direct_start_ready':True});self.assertFalse(c.submit('start_now',0)[0])

class LampArmTests(unittest.TestCase):
 def control(self,arm=True,active=False):
  c=web.WebControl();c.update(active,False,'ROAD NOT READY',(0,0),readiness={'direct_start_ready':False,'arm_start_light_ready':arm,'blocked_by':['ROAD_NO_TARGET','UNO_DISCONNECTED']});return c
 def test_can_arm_with_no_road_or_uno_without_allowing_direct_start(self):
  with patch.object(web.time,'monotonic',return_value=10.):
   c=self.control();self.assertFalse(c.submit('start_now',0)[0]);self.assertTrue(c.submit('start',0)[0]);self.assertEqual(c.take(),'start')
 def test_active_already_armed_and_stale_states_cannot_arm(self):
  with patch.object(web.time,'monotonic',return_value=10.) as clock:
   for c in (self.control(arm=False),self.control(active=True)):
    self.assertFalse(c.submit('start',0)[0])
   c=self.control();clock.return_value=11.;self.assertFalse(c.submit('start',0)[0])
 def test_stop_cancels_arm_and_obsolete_generation(self):
  with patch.object(web.time,'monotonic',return_value=10.):
   c=self.control();self.assertTrue(c.submit('start',0)[0]);c.submit('stop');self.assertEqual(c.take(),'stop');self.assertFalse(c.submit('start',0)[0])

if __name__=='__main__':unittest.main()
