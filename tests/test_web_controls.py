import importlib.util,json,urllib.request,urllib.error,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).resolve().parents[1]/'maixcam2/lane_follow/main.py'
spec=importlib.util.spec_from_file_location('lane',p);lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
class ControlTests(unittest.TestCase):
 def setUp(self):self.c=lane.WebControl();self.c.update(False,True,'WAIT START',(0,0))
 def test_stop_cancels_pending_start_and_rejects_delayed_start(self):
  g=self.c.status()['generation'];self.assertTrue(self.c.submit('start',g)[0]);self.assertTrue(self.c.submit('stop')[0]);self.assertEqual(self.c.take(),'stop');self.assertFalse(self.c.submit('start',g)[0]);self.assertIsNone(self.c.take())
 def test_start_cannot_queue_until_ready(self):
  self.c.update(False,False,'ROAD NOT READY',(0,0));self.assertFalse(self.c.submit('start',0)[0]);self.c.update(False,True,'WAIT START',(0,0));self.assertIsNone(self.c.take())
 def test_stale_pending_start_is_discarded(self):
  self.c.submit('start',0)
  with patch.object(lane.time,'monotonic',return_value=self.c.updated+2):self.assertIsNone(self.c.take())
 def test_stop_is_allowed_even_with_stale_status(self):
  with patch.object(lane.time,'monotonic',return_value=self.c.updated+10):
   self.assertFalse(self.c.submit('start',0)[0]);self.assertTrue(self.c.submit('stop')[0]);self.assertEqual(self.c.take(),'stop')
 def test_http_reads_cannot_start_and_post_requires_page_token(self):
  stream=lane.LiveStream(port=0);url='http://127.0.0.1:'+str(stream.server.server_port)
  try:
   with urllib.request.urlopen(url) as r:self.assertIn('START 开始',r.read().decode())
   with urllib.request.urlopen(url+'/status') as r:self.assertFalse(json.load(r)['active'])
   self.assertIsNone(stream.control.take())
   req=urllib.request.Request(url+'/control',data=b'{"action":"start","generation":0}',headers={'Content-Type':'application/json'})
   with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(req)
   self.assertEqual(e.exception.code,403);self.assertIsNone(stream.control.take())
   req.add_header('X-Control-Token',stream.token)
   with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(req)
   self.assertEqual(e.exception.code,409)
   req=urllib.request.Request(url+'/control',data=b'{"action":"stop"}',headers={'X-Control-Token':stream.token,'Content-Type':'application/json'})
   with urllib.request.urlopen(req) as r:self.assertEqual(r.status,202)
   self.assertEqual(stream.control.take(),'stop')
  finally:stream.stop()
if __name__=='__main__':unittest.main()
