"""Generate the selected rare-endpoint populations with bundled fixed equations."""
from pathlib import Path
import runpy
import sys

CORE = Path(__file__).resolve().parent / "core"

if __name__ == "__main__":
    sys.path.insert(0, str(CORE))
    from model import RISKS
    for option, filename in [("--schema", "measurement_dictionary.csv"),
                             ("--disease-identities", "disease_identities.csv")]:
        if not any(arg == option or arg.startswith(option + "=") for arg in sys.argv[1:]):
            sys.argv.extend([option, str(CORE / filename)])
    if not any(arg == "--diseases" or arg.startswith("--diseases=") for arg in sys.argv[1:]):
        sys.argv.extend(["--diseases", *RISKS])
    runpy.run_path(str(CORE / "generate.py"), run_name="__main__")
