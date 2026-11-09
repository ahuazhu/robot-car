"""Run the shipped browser script with deterministic fetch, image and timer ports."""
from pathlib import Path
import shutil
import subprocess
import unittest
from tests.support import ROOT, APP_ROOT


class PreviewConnectionTests(unittest.TestCase):
    def test_field_controls_connection_and_roi_draft(self):
        node=shutil.which('node')
        if not node:
            bundled=Path('/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node')
            if bundled.exists():node=str(bundled)
        if not node:self.skipTest('Node.js required for browser-script regression')
        result=subprocess.run([node,str(ROOT/'tests/js/preview_connection.test.js'),str(APP_ROOT/'robot/interface/preview.html')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
