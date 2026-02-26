#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn


if __name__ == "__main__":
    uvicorn.run("stonkmodel.api.main:app", host="0.0.0.0", port=8000, reload=True)
