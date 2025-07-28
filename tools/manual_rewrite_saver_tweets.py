import json
import os

# 1. Put all rewritten entries here
entries = [
    {
        "author": "mlevenson",
        "original": "To many, it looked like the type of watchdog reporting that many news organizations consider the hallmark of responsible journalism. But Gov. Mike Parson of Missouri had a different view.\n\nhttps://t.co/YmsEJOusiM",
        "rewritten": "Governor Mike Parson of Missouri held a different perspective on what many considered to be the hallmark of responsible journalism: watchdog reporting. You can read more here: https://t.co/YmsEJOusiM"
    },
    {
        "author": "jillfilipovic",
        "original": "@caro @BriannaWu Obviously the problem was you, the person without noise canceling headphones.",
        "rewritten": "@caro @BriannaWu Clearly, the issue lay with you, as you were the one without noise-canceling headphones."
    },
    {
        "author": "npwcnn",
        "original": "@davidschneider @lukemcgee In the brotherhood of flags.",
        "rewritten": "@davidschneider @lukemcgee In the brotherhood of flags."
    },
    {
        "author": "juliemtoi",
        "original": "https://t.co/nrae5Fp0rD",
        "rewritten": "You can find the information at: https://t.co/nrae5Fp0rD By the way, to unlock the full functionality of all Apps, enable Gemini Apps Activity."
    },    
    {
        "author": "greenslader",
        "original": "\"Local ownership has to be the key to the survival of local publications, with their roots firmly embedded in the communities they serve\" --- Steve Egginton, owner, Mendip Times, former director of the Society of Editors, taking us back to basics. Break the chains, go local! https://t.co/6FHIEfTBCU",
        "rewritten": """For local publications to survive, local ownership is crucial, with their roots deeply embedded in the communities they serve," states Steve Egginton, owner of Mendip Times and former director of the Society of Editors, emphasizing a return to fundamentals. He advocates for breaking free from traditional constraints and embracing a local focus. Learn more here: https://t.co/6FHIEfTBCU"""
    }]

# 2. json file directory
output_path = "experiments/llm_rewriting_trial1/output/tweets_rewritten_manual.json"

# 3. load existing data
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = []

# 4. add new entries and save
data.extend(entries)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Appended {len(entries)} entries to {output_path}")