"""Run the frozen generator with its bundled schema and endpoint definitions."""
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parent
if __name__ == "__main__":
    for option, filename in [("--schema", "allowed_schema.csv"), ("--disease-identities", "disease_identities.csv")]:
        if not any(arg == option or arg.startswith(option + "=") for arg in sys.argv[1:]):
            sys.argv.extend([option, str(ROOT / filename)])
    runpy.run_path(str(ROOT / "generate_gpt6_rare_v1.py"), run_name="__main__")
