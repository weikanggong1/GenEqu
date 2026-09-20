"""Compute observed-measurement risk with the bundled fixed rare-endpoint model."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    core = Path(__file__).resolve().parent / "core"
    sys.path.insert(0, str(core))
    runpy.run_path(str(core / "predict.py"), run_name="__main__")
