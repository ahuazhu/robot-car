from tests.support import ROOT
from tests.support.uart import Clock, Port, link, ack
import contextlib, io, unittest
from unittest.mock import patch

from robot.control import drive_link


class RawReplyDiagnosticsTests(unittest.TestCase):
    def test_brake_status_extension_and_legacy_status_both_work(self):
        for flags in (4, 5, 7):
            body = ("1,0,0,0,0,%d" % flags).encode()
            obj = link(b"!" + body + b"*%02X\n" % drive_link.crc8(body))
            with patch.object(drive_link, "time", Clock()):
                self.assertEqual(obj.send(3)[5], flags)

    def test_contradictory_lamp_status_is_rejected(self):
        for flags in (0, 2, 6, 8):
            body = ("1,0,0,0,0,%d" % flags).encode()
            obj = link(b"!" + body + b"*%02X\n" % drive_link.crc8(body))
            with patch.object(drive_link, "time", Clock()):
                with self.assertRaisesRegex(RuntimeError, "brake lamp"):
                    obj.send(3)

    def test_valid_ack_after_noise_is_accepted(self):
        obj = link(b"noise\n" + ack())
        with patch.object(drive_link, "time", Clock()):
            self.assertEqual(obj.send(3), [1, 0, 0, 0, 0])
        self.assertIsNone(obj.last_error_diagnostics)

    def test_missing_ack_records_zero_bytes_without_accepting_it(self):
        obj = link(b"")
        with patch.object(drive_link, "time", Clock()), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "acknowledgement timeout"):
                obj.send(3)
        self.assertEqual(obj.last_error_diagnostics["rx_bytes"], 0)
        self.assertEqual(obj.last_error_diagnostics["operation"], 3)

    def test_bad_crc_and_wrong_sequence_remain_rejected_and_recorded(self):
        for received in (b"!1,0,0,0,0*00\n", ack(2)):
            obj = link(received)
            with (
                patch.object(drive_link, "time", Clock()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "acknowledgement timeout"):
                    obj.send(3)
            self.assertEqual(obj.last_error_diagnostics["rx_hex"], received.hex())


if __name__ == "__main__":
    unittest.main()
