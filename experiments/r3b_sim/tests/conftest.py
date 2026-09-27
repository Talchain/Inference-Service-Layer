"""Make ``sim`` importable when pytest runs from the ISL repo root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
