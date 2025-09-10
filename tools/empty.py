# tools/find_empty_texts.py
import json
import re
from pathlib import Path
import sys

def load_rows(path: Path):
    text = path.read_text(encoding="utf-8-sig")
    s = text.lstrip()
    if s.startswith("["):  # JSON array
        return json.loads(text)
    else:  # JSONL
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[ERROR] line {i}: {e}")
        return rows

def word_count(t: str) -> int:
    t = re.sub(r"\s+", " ", t or "").strip()
    return len(t.split()) if t else 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/find_empty_texts.py your_file.json[l]")
        sys.exit(1)

    path = Path(sys.argv[1])
    rows = load_rows(path)

    empties = []
    for r in rows:
        text_val = r.get("text", "")
        wc = word_count(text_val)
        if wc == 0:
            empties.append(r)

    print(f"Total records: {len(rows)}")
    print(f"Empty text count: {len(empties)}")

    # Show first 10 empties for inspection
    for e in empties[:10]:
        print(json.dumps(e, ensure_ascii=False, indent=2))