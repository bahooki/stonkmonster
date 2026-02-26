#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit.web.cli as stcli


if __name__ == "__main__":
    sys.argv = ["streamlit", "run", str(ROOT / "src/stonkmodel/ui/dashboard.py")]
    raise SystemExit(stcli.main())
