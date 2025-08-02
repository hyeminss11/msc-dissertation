import json
from pathlib import Path

# New Item
new_item = {
    "author": "PippaCrerar",
    "source": "https://www.youtube.com/watch?v=dw-SCQqmMEE",
    "text": "I'm Pippa Crerar and I've just won journalist of the year so a lot of this is about political journalism in general but in particularly particularly um to do with the partygate story which I first heard about at a time when we were all living under these really strict rules which were imposed by the government and most people I think were sticking to them and I subsequently found out that people at the heart of government hadn't actually been sticking to them in fact they've been flighting them there was a series of social events in the heart of government and we exposed them um I guess the rest is history there was a it wasn't just the first the first exclusive there was six months of stories by other journalists as well I mean ITV's Paul Brand in particular deserves a call out because he did huge amounts of work on this as well but it was a subsequent month it really put Royce Johnson in a very difficult position ultimately it was about his integrity at the heart of it and he lost the faith of his story in peace there were other factors too but ultimately led to a stand form it's a huge amount I mean I'm actually I feel very um I feel a bit shaky about it I'm completely honest um it was absolutely lovely to get the recognition of your peers in this way I mean in some ways right to do the partygate story was a pretty lonely path um we particularly when you're faced with sort of denials and obfuscation that we met from number 10 from government departments and from the Prime Minister himself who replied on because we were sure that we were we were onto something we've trusted our sources and uh you know and we felt that it was worth it was worth pursuing so to have sort of public recognition of like this is incredibly special so thank you",
    "type": "interview_transcript",
    "note": "Youtube Interview, automatic subtitle generation + manual"
}

# path
json_path = Path("dataset-extension/other-media.json")

# load json
if json_path.exists():
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = []

# append
data.append(new_item)

# save
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Added new interview. Total interviews: {len(data)}")