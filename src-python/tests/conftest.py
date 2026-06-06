"""Pytest bootstrap: put the sidecar root (src-python/) on sys.path so tests
use the same flat imports the app uses (e.g. `from stt_providers...`)."""
import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parent.parent
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))
