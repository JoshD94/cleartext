#!/usr/bin/env python3
"""Explicit entry point; no effect on the v7 demo or defaults."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.structure.experiment import main

if __name__ == "__main__":
    main()
