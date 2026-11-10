"""Display-only RGB evidence cannot grant or revoke the Maix start permit."""
import json
import unittest

from tests import test_esp_start_signal as peer_tests


class RGBSignalTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.case = peer_tests.SignalPeerTests()
        self.case.setUp()
        self.addCleanup(self.case.tearDown)
        self.client, self.remote = self.case.client, self.case.remote

    def test_old_firmware_without_optional_fields_still_confirms(self):
        self.case.arm(); self.case.confirm()
        self.assertTrue(self.client.permission)
        self.assertFalse(any(key.startswith('rgb_') for key in self.client.snapshot()))

    def test_fields_forward_from_frozen_confirmation_even_after_peer_loss(self):
        diagnostics = dict(rgb_workspace_available=True, rgb_reference_ready=True,
                           luma_off=False, rgb_off=True, rgb_channel=1,
                           rgb_joint_fraction=.91, rgb_dark_fraction=.96,
                           rgb_background_samples=502, rgb_background_span=89.,
                           rgb_background_fitted=True, rgb_rejection_reason='RGB_STRONG_OFF',
                           rgb_required_drop=48., rgb_required_relative_drop=.4,
                           rgb_dark_max=96, rgb_dark_chroma_max=12, rgb_black_max=32,
                           rgb_required_fraction=.6)
        self.remote.update(diagnostics)
        self.case.arm(); self.case.confirm()
        self.client.transport = lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
        self.client._step()
        state = self.client.snapshot()
        self.assertTrue(state['permission'])
        for key, value in diagnostics.items(): self.assertEqual(state[key], value)

    def test_bad_diagnostics_cannot_reject_a_valid_event_or_break_json(self):
        diagnostics = dict(rgb_workspace_available=1, rgb_reference_ready='true',
                           luma_off=[], rgb_off='false', rgb_channel=True,
                           rgb_joint_fraction=float('nan'), rgb_dark_fraction=1.1,
                           rgb_background_samples=-1, rgb_background_span=float('inf'),
                           rgb_background_fitted=1, rgb_rejection_reason={'bad':'shape'},
                           rgb_required_drop=True, rgb_required_relative_drop=-.1,
                           rgb_dark_max=256, rgb_dark_chroma_max=12.0, rgb_black_max=False,
                           rgb_required_fraction='0.6')
        self.remote.update(diagnostics)
        self.case.arm(); self.case.confirm()
        state = self.client.snapshot()
        self.assertTrue(state['permission'])
        for key in diagnostics: self.assertIsNone(state[key], key)
        json.dumps(state, allow_nan=False)

    def test_rgb_true_without_formal_confirmation_cannot_grant_permission(self):
        self.case.arm()
        self.remote.update(phase='SIGNAL_READY', rgb_off=True, rgb_channel=0,
                           rgb_joint_fraction=1., rgb_background_fitted=True,
                           rgb_rejection_reason='RGB_STRONG_OFF', confirmed=False)
        self.client._step()
        self.assertFalse(self.client.permission)
        self.assertFalse(self.client.consume())
        self.assertTrue(self.client.snapshot()['rgb_off'])

