"""Run the shipped browser script with deterministic fetch, image and timer ports."""
from pathlib import Path
import shutil
import subprocess
import unittest
from tests.support import ROOT, APP_ROOT


class PreviewConnectionTests(unittest.TestCase):
    def test_disconnect_hung_image_background_resume_and_freshness(self):
        node=shutil.which('node')
        if not node:
            bundled=Path('/Applications/ChatGPT.app/Contents/Resources/cua_node/bin/node')
            if bundled.exists():node=str(bundled)
        if not node:self.skipTest('Node.js required for browser-script regression')
        subprocess.run([node,str(ROOT/'tests/js/preview_connection.test.js'),str(APP_ROOT/'robot/interface/preview.html')],check=True,capture_output=True,text=True)
