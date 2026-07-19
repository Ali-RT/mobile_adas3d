"""Backward-compatible entry point for the canonical KITTI split installer."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_kitti_chen_split import main


if __name__ == "__main__":
    main()
