import contextlib,io,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'maixcam2/curve_follow'))
import drive_link

class Clock:
    def __init__(self):self.now=0.
    def monotonic(self):self.now+=.001;return self.now
    def sleep(self,t):self.now+=t

class Port:
    def __init__(self,reply):self.reply=reply;self.tx=None
    def read(self,timeout=0):
        if self.tx is None:return b''
        reply,self.reply=self.reply,b'';return reply
    def write_str(self,text):self.tx=text

def link(reply):
    obj=drive_link.DriveLink.__new__(drive_link.DriveLink)
    obj.seq=0;obj.port=Port(reply);return obj

def ack(sequence=1):
    body=('%d,0,0,0,0'%sequence).encode()
    return b'!'+body+b'*%02X\n'%drive_link.crc8(body)

class RawReplyDiagnosticsTests(unittest.TestCase):
    def test_brake_status_extension_and_legacy_status_both_work(self):
        for flags in (4,5,7):
            body=('1,0,0,0,0,%d'%flags).encode()
            obj=link(b'!'+body+b'*%02X\n'%drive_link.crc8(body))
            with patch.object(drive_link,'time',Clock()):self.assertEqual(obj.send(3)[5],flags)
    def test_contradictory_lamp_status_is_rejected(self):
        for flags in (0,2,6,8):
            body=('1,0,0,0,0,%d'%flags).encode()
            obj=link(b'!'+body+b'*%02X\n'%drive_link.crc8(body))
            with patch.object(drive_link,'time',Clock()):
                with self.assertRaisesRegex(RuntimeError,'brake lamp'):obj.send(3)
    def test_valid_ack_after_noise_is_accepted(self):
        obj=link(b'noise\n'+ack())
        with patch.object(drive_link,'time',Clock()):
            self.assertEqual(obj.send(3),[1,0,0,0,0])
        self.assertIsNone(obj.last_error_diagnostics)
    def test_missing_ack_records_zero_bytes_without_accepting_it(self):
        obj=link(b'')
        with patch.object(drive_link,'time',Clock()),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'acknowledgement timeout'):obj.send(3)
        self.assertEqual(obj.last_error_diagnostics['rx_bytes'],0)
        self.assertEqual(obj.last_error_diagnostics['operation'],3)
    def test_bad_crc_and_wrong_sequence_remain_rejected_and_recorded(self):
        for received in (b'!1,0,0,0,0*00\n',ack(2)):
            obj=link(received)
            with patch.object(drive_link,'time',Clock()),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'acknowledgement timeout'):obj.send(3)
            self.assertEqual(obj.last_error_diagnostics['rx_hex'],received.hex())

if __name__=='__main__':unittest.main()
