"""Canonical memflow RL entrypoint.

This keeps the per-model training-script convention while preserving the
historical `train_kvpo.py` command.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.memflow.train_kvpo import main


if __name__ == "__main__":
    main()
