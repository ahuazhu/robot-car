from tests.support import ROOT
import unittest
from unittest.mock import patch
from robot.interface import preview_transport as web
from robot.control.requests import ControlRequest


class WebStartTests(unittest.TestCase):
    def test_bottom_stop_uses_same_stop_action_and_remains_enabled(self):
        from html.parser import HTMLParser

        class Buttons(HTMLParser):
            def __init__(self):
                super().__init__()
                self.buttons = {}

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "button" and "id" in attrs:
                    self.buttons[attrs["id"]] = attrs

        page = Buttons()
        page.feed(web.LIVE_HTML)
        top = page.buttons["stop"]
        bottom = page.buttons["stop-bottom"]
        self.assertEqual(bottom["onclick"], top["onclick"])
        self.assertEqual(bottom["onclick"], "control('stop')")
        self.assertNotIn("disabled", bottom)

    def ready(self):
        c = web.WebControl()
        c.update(False, True, "WAIT START", (0, 0))
        return c

    def test_request_survives_processing_delay_within_status_freshness(self):
        with patch.object(web.time, "monotonic", return_value=10.0) as clock:
            c = self.ready()
            self.assertTrue(c.submit("start", 0)[0])
            clock.return_value = 10.65
            self.assertEqual(c.take(), ControlRequest("start"))

    def test_request_expires_at_same_freshness_boundary_as_status(self):
        with patch.object(web.time, "monotonic", return_value=10.0) as clock:
            c = self.ready()
            c.submit("start", 0)
            clock.return_value = 11.0
            self.assertIsNone(c.take())

    def test_stop_cancels_pending_start_and_invalidates_its_generation(self):
        with patch.object(web.time, "monotonic", return_value=10.0):
            c = self.ready()
            c.submit("start", 0)
            self.assertTrue(c.submit("stop")[0])
            self.assertEqual(c.take(), ControlRequest("stop"))
            self.assertFalse(c.submit("start", 0)[0])

    def test_stale_status_cannot_enqueue_new_start(self):
        with patch.object(web.time, "monotonic", return_value=10.0) as clock:
            c = self.ready()
            clock.return_value = 11.0
            self.assertFalse(c.submit("start", 0)[0])

    def test_not_ready_or_active_cannot_enqueue_start(self):
        with patch.object(web.time, "monotonic", return_value=10.0):
            for active, ready in ((False, False), (True, True)):
                c = self.ready()
                c.update(active, ready, "TEST", (0, 0))
                self.assertFalse(c.submit("start", 0)[0])


class DirectStartTests(unittest.TestCase):
    def waiting(self, ready=True):
        c = web.WebControl()
        c.update(False, False, "WAIT START LIGHT", (0, 0), readiness={"direct_start_ready": ready})
        return c

    def test_direct_start_can_queue_while_lamp_waiting_button_is_disabled(self):
        with patch.object(web.time, "monotonic", return_value=10.0):
            c = self.waiting()
            self.assertFalse(c.submit("start", 0)[0])
            self.assertTrue(c.submit("start_now", 0)[0])
            self.assertEqual(c.take(), ControlRequest("start_now"))

    def test_missing_road_or_unready_drive_still_blocks_direct_start(self):
        with patch.object(web.time, "monotonic", return_value=10.0):
            c = self.waiting(False)
            self.assertFalse(c.submit("start_now", 0)[0])
            self.assertIsNone(c.take())

    def test_stop_cancels_direct_start_and_stale_generations(self):
        with patch.object(web.time, "monotonic", return_value=10.0):
            c = self.waiting()
            c.submit("start_now", 0)
            c.submit("stop")
            self.assertEqual(c.take(), ControlRequest("stop"))
            self.assertIsNone(c.take())
            self.assertFalse(c.submit("start_now", 0)[0])

    def test_direct_start_expires_when_queue_or_status_is_stale(self):
        with patch.object(web.time, "monotonic", return_value=10.0) as clock:
            c = self.waiting()
            c.submit("start_now", 0)
            clock.return_value = 11.0
            self.assertIsNone(c.take())
            self.assertFalse(c.submit("start_now", 0)[0])

    def test_active_mission_cannot_receive_a_second_direct_start(self):
        with patch.object(web.time, "monotonic", return_value=10.0):
            c = self.waiting()
            c.update(True, False, "RUNNING", (60, 60), readiness={"direct_start_ready": True})
            self.assertFalse(c.submit("start_now", 0)[0])


class LampArmTests(unittest.TestCase):
    def control(self, arm=True, active=False):
        c = web.WebControl()
        c.update(
            active,
            False,
            "ROAD NOT READY",
            (0, 0),
            readiness={
                "direct_start_ready": False,
                "arm_start_light_ready": arm,
                "blocked_by": ["ROAD_NO_TARGET", "UNO_DISCONNECTED"],
            },
        )
        return c

    def test_can_arm_with_no_road_or_uno_without_allowing_direct_start(self):
        with patch.object(web.time, "monotonic", return_value=10.0):
            c = self.control()
            self.assertFalse(c.submit("start_now", 0)[0])
            self.assertTrue(c.submit("start", 0)[0])
            self.assertEqual(c.take(), ControlRequest("start"))

    def test_active_already_armed_and_stale_states_cannot_arm(self):
        with patch.object(web.time, "monotonic", return_value=10.0) as clock:
            for c in (self.control(arm=False), self.control(active=True)):
                self.assertFalse(c.submit("start", 0)[0])
            c = self.control()
            clock.return_value = 11.0
            self.assertFalse(c.submit("start", 0)[0])

    def test_stop_cancels_arm_and_obsolete_generation(self):
        with patch.object(web.time, "monotonic", return_value=10.0):
            c = self.control()
            self.assertTrue(c.submit("start", 0)[0])
            c.submit("stop")
            self.assertEqual(c.take(), ControlRequest("stop"))
            self.assertFalse(c.submit("start", 0)[0])


class RaceTargetTests(unittest.TestCase):
    def control(self, *, armed=False):
        c = web.WebControl()
        c.update(
            False,
            True,
            "WAIT START LIGHT" if armed else "WAIT START",
            (0, 0),
            readiness={
                "direct_start_ready": True,
                "arm_start_light_ready": not armed,
                "start_light": {"armed": armed},
                "laps": {"total_laps": 4},
            },
        )
        return c

    def test_both_start_actions_capture_integer_boundaries_and_optional_target(self):
        with patch.object(web.time, "monotonic", return_value=10.0):
            for action in ("start", "start_now"):
                for target in (1, 100, None):
                    with self.subTest(action=action, target=target):
                        c = self.control()
                        self.assertTrue(c.submit(action, 0, laps=target)[0])
                        self.assertEqual(c.take(), ControlRequest(action, target))
                        self.assertIsNone(c.take())

    def test_invalid_targets_cannot_enqueue_either_start_action(self):
        invalid = (True, False, 0, -1, 101, 1.0, 1.5, "3", "", [], {}, float("nan"), float("inf"))
        with patch.object(web.time, "monotonic", return_value=10.0):
            for action in ("start", "start_now"):
                for target in invalid:
                    with self.subTest(action=action, target=target):
                        c = self.control()
                        self.assertFalse(c.submit(action, 0, laps=target)[0])
                        self.assertIsNone(c.take())

    def test_queued_target_survives_later_status_and_other_submissions(self):
        with patch.object(web.time, "monotonic", return_value=10.0):
            c = self.control()
            self.assertTrue(c.submit("start", 0, laps=7)[0])
            c.update(False, True, "WAIT START", (0, 0), readiness={"laps": {"total_laps": 2}})
            self.assertFalse(c.submit("start", 0, laps=9)[0])
            self.assertFalse(c.submit("start", 0, laps="bad")[0])
            self.assertEqual(c.take(), ControlRequest("start", 7))

    def test_stop_ignores_invalid_target_and_stale_generation_and_never_expires(self):
        with patch.object(web.time, "monotonic", return_value=10.0) as clock:
            c = self.control()
            self.assertTrue(c.submit("start_now", 0, laps=7)[0])
            self.assertTrue(c.submit("stop", -1, laps={"invalid": True})[0])
            clock.return_value = 20.0
            self.assertEqual(c.take(), ControlRequest("stop"))
            self.assertIsNone(c.take())
            c.update(False, True, "WAIT START", (0, 0))
            self.assertFalse(c.submit("start_now", 0, laps=7)[0])

    def test_armed_direct_start_accepts_only_matching_or_unspecified_target(self):
        with patch.object(web.time, "monotonic", return_value=10.0):
            for target in (4, None):
                with self.subTest(target=target):
                    c = self.control(armed=True)
                    self.assertFalse(c.submit("start_now", 0, laps=5)[0])
                    self.assertIsNone(c.take())
                    self.assertTrue(c.submit("start_now", 0, laps=target)[0])
                    self.assertEqual(c.take(), ControlRequest("start_now", target))


if __name__ == "__main__":
    unittest.main()
