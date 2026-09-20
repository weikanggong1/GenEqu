"""Verify packaged file hashes and reject untranslated text or private paths."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'release_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        path=ROOT/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, name
        if path.suffix in {'.py','.md','.csv','.json','.txt'}:
            text=path.read_text(encoding='utf-8-sig')
            if path.suffix=='.json':text=json.dumps(json.loads(text),ensure_ascii=False)
            assert not re.search(r'[\u3400-\u9fff\uac00-\ud7af]',text), name
            assert not re.search(r'/mnt/[a-z]/Users/[A-Za-z0-9]|/public/home/[A-Za-z0-9]|C:\\\\Users\\\\',text,re.I), name
    print(f"Verified {len(manifest['files'])} file hashes and English text checks.")

if __name__=='__main__':
    main()
