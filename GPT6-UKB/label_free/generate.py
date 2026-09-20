"""Generate the label-free 201-measurement population with its fixed dictionary."""
from pathlib import Path
import runpy
import sys

CORE = Path(__file__).resolve().parent / "core"

if __name__ == "__main__":
    if not any(arg == "--schema" or arg.startswith("--schema=") for arg in sys.argv[1:]):
        sys.argv.extend(["--schema", str(CORE / "measurement_dictionary.csv")])
    runpy.run_path(str(CORE / "generate.py"), run_name="__main__")
