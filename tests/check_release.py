"""Verify the complete public inventory, source hashes, English text and paths."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={".git","__pycache__",".venv","venv","generated"}

def packaged_files():
    return {str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
            if p.is_file() and not any(part in EXCLUDED for part in p.relative_to(ROOT).parts)
            and p.suffix != ".pyc" and p.name != "release_manifest.json"}

def main():
    manifest=json.loads((ROOT/"release_manifest.json").read_text())
    assert packaged_files()==set(manifest["files"]), "Release inventory differs from manifest"
    for name,digest in manifest["files"].items():
        path=ROOT/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
        if path.suffix in {".py",".md",".csv",".json",".txt"}:
            text=path.read_text(encoding="utf-8-sig")
            if path.suffix==".json":text=json.dumps(json.loads(text),ensure_ascii=False)
            assert not re.search(r"[\u3400-\u9fff\uac00-\ud7af]",text),name
            private_path = "/"+"mnt/[a-z]/(?:Users|users)/[A-Za-z0-9]|/"+"public/home/[A-Za-z0-9]|C:"+"\\\\Users\\\\"
            assert not re.search(private_path,text,re.I),name
    sources=json.loads((ROOT/"release_source_hashes.json").read_text())
    for name,digest in sources["files"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    print(f"Verified {len(manifest['files'])} public files and {len(sources['files'])} scientific source hashes.")

if __name__=="__main__":
    main()
