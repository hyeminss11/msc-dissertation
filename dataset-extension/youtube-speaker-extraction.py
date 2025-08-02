from pathlib import Path
import re
from difflib import SequenceMatcher

# Path to the VTT subtitle file
vtt_file = Path("Sign Up - Into Football ｜ Lucy Ward's top broadcasting moments & life as a female pundit [MNDEhSPEJuo].en.vtt")

lines = []
last_clean_line = ""
skip_tokens = {"[Music]", "[Applause]", "♪", ""}

def clean_line(line):
    # Remove <timestamp><c> tags
    line = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}><c>", "", line)
    line = line.replace("</c>", "")
    # Remove curly or straight quotation marks
    line = line.replace("“", "").replace("”", "").replace('"', "")
    return line.strip()

with open(vtt_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # Skip metadata and timestamps
        if (
            not line
            or line.startswith("WEBVTT")
            or "-->" in line
            or line in skip_tokens
        ):
            continue

        # Clean the line
        cleaned = clean_line(line)

        # Skip exact duplicates
        if cleaned == last_clean_line:
            continue

        # Skip near-duplicates based on similarity ratio
        ratio = SequenceMatcher(None, cleaned, last_clean_line).ratio()
        if ratio > 0.95:
            continue

        lines.append(cleaned)
        last_clean_line = cleaned

# Join all lines into a single string
transcript = " ".join(lines)

# Save the cleaned transcript to a file
output_path = Path("dataset-extension/drafts/lucywardsing_1.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(transcript)

print(f"Transcript saved to {output_path}. Total lines used: {len(lines)}")