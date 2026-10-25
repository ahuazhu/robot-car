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
if __name__=='__main__':unittest.main()
