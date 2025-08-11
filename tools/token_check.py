# tools/token_check.py
import json
from pathlib import Path
from transformers import AutoTokenizer

FILE = Path("processed_data_new/crossnews_gold_processed.json")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def load_rows(path: Path):
    # Open with utf-8-sig to strip BOM if present
    text = path.read_text(encoding="utf-8-sig")

    # Detect JSON array vs JSONL by first non-space char
    first = next((ch for ch in text.lstrip()[:1]), "")
    if first == "[":  # JSON array
        data = json.loads(text)
        for obj in data:
            yield obj
    else:  # JSONL
        for lineno, line in enumerate(text.splitlines(), start=1):
            s = line.strip()
            if not s:
                continue  # skip blanks
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Bad JSONL at line {lineno}: {e.msg} (pos {e.pos})\nLine: {line[:200]}"
                ) from e

def main():
    rows = list(load_rows(FILE))

    # Pull both sides
    texts = []
    for r in rows:
        if "text0" in r and "text1" in r:
            texts.append(r["text0"])
            texts.append(r["text1"])
        else:
            # fallback if your schema differs
            for k in ("text",):
                if k in r:
                    texts.append(r[k])

    # Token stats
    token_lengths = []
    for t in texts:
        token_ids = tokenizer(t, add_special_tokens=True, truncation=False)["input_ids"]
        token_lengths.append(len(token_ids))

    if not token_lengths:
        print("No texts found. Check keys (expected 'text0'/'text1').")
        return

    total = len(token_lengths)
    avg_len = sum(token_lengths) / total
    min_len = min(token_lengths)
    max_len = max(token_lengths)

    print(f"Total texts: {total}")
    print(f"Avg tokens:  {avg_len:.2f}")
    print(f"Min tokens:  {min_len}")
    print(f"Max tokens:  {max_len}")

if __name__ == "__main__":
    main()