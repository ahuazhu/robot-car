import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/curve_follow'))
from preview_transport import LiveStream
class PreviewFaultTests(unittest.TestCase):
 def test_jpeg_failure_preserves_control_and_recovers_next_frame(self):
  stream=LiveStream(0)
  try:
   with patch('preview_transport.cv2.imencode',return_value=(False,None)):
    self.assertFalse(stream.write(np.zeros((8,8,3),np.uint8)))
   self.assertIsNotNone(stream.preview_error);self.assertEqual(stream.sequence,0)
   self.assertTrue(stream.write(np.zeros((8,8,3),np.uint8)))
   self.assertIsNone(stream.preview_error);self.assertEqual(stream.sequence,1)
  finally:stream.stop()
 def test_preview_bind_failure_leaves_local_controls_usable(self):
  with patch('http.server.ThreadingHTTPServer',side_effect=OSError('port unavailable')):
   stream=LiveStream(0)
  self.assertIsNotNone(stream.preview_error);stream.control.update(False,True,'WAIT START',(0,0))
  self.assertTrue(stream.control.submit('start',0)[0]);self.assertEqual(stream.control.take(),'start')
  self.assertFalse(stream.write(None));stream.stop()
if __name__=='__main__':unittest.main()
