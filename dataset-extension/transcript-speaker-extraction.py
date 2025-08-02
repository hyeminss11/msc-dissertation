from pathlib import Path

# file path

BASE_DIR = Path(__file__).parent
file_path = BASE_DIR / "john_hooper_1.txt"
output_path = BASE_DIR / "jh_text_only.txt"

# result list
jh_lines = []

# extract
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("JH:"):
            text = line[3:].strip()
            # delete " “ ”
            text = text.replace('"', "").replace("“", "").replace("”", "")
            jh_lines.append(text)

# combine
jh_text = " ".join(jh_lines)

# preview
print(jh_text[:500])

# save
with open(output_path, "w", encoding="utf-8") as f:
    f.write(jh_text)