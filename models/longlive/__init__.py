import sys
from pathlib import Path

_LONG_LIVE_ROOT = Path(__file__).resolve().parent
if str(_LONG_LIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_LONG_LIVE_ROOT))
