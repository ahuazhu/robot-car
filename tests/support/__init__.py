"""Shared test paths; the application is imported through its real package."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = ROOT / "maixcam2" / "curve_follow"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
