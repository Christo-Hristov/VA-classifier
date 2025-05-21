"""
A few points.  
1- pip install --upgrade openai # in terminal
2- Change the path to the api_key (1)
3- Update model and temp at position (4)
    - "gpt-4.1-2025-04-14"
    - "gpt-4o-mini"

gpt4_VA classifier does the following:

Input 
    - Text input

Process:
    - Prompting

Output
    - Valence (negative vs. positive)
    - Arousal (calm vs. excited)
    - Each of those take numeric values from [-1, 1] # we scaled this parameter for simplicity
"""
import json
import argparse
from pathlib import Path
from openai import OpenAI

# 1. Get API key for gpt4.0
key_path = Path.home() / "Desktop" / "openai_key.txt" # update the path as you go
api_key = key_path.read_text().strip()           
client = OpenAI(api_key=api_key)

# 2. System Prompt
SYSTEM_PROMPT = (
    "You are an affective-computing expert. For any text return "
    'JSON {"valence":float[-1,1],"arousal":float[-1,1]}. '
    "Think step-by-step but reveal only the JSON."
)

# 3. Few Shot section.

# Raw data from "https://aclanthology.org/W16-0404.pdf"
# 3.1. The valence space
VALENCE = [
    ("!",.251), (":)",.237), ("Birthday",.212), ("Happy",.197),
    ("Thank",.196), ("Great",.195), ("Love",.195), ("Thanks",.179),
    ("Wishes",.170), ("Wonderful",.159),
    ("Hate",-0.163), (":(", -0.159), ("?",-0.117), ("Sick",-0.112),
    ("Why",-0.102), (":'(", -0.094), ("Not",-0.093), ("Bored",-0.092),
    ("Stupid",-0.089), ("...",-0.087)
]
# 3.2. The arousal space
AROUSAL = [
    ("!", .773), ("Birthday", .097), ("Happy", .081), ("Its", .079),
    ("Wishes", .076), ("Soooo", .074), ("Thanks", .073),
    ("Christmas", .071), ("Sunday", .069), ("Yay", .064),
    ("[..]*", -0.206), (".", -0.164), ("Status", -0.064),
    ("Life", -0.064), ("People", -0.060), ("Bored", -0.059),
    (":/", -0.056), ("Of", -0.056), ("Deal", -0.056), ("Every", -0.054)
]

merged: dict[str, dict[str,float]] = {}
for w, r in VALENCE:
    merged.setdefault(w, {})["valence"] = round(r, 3)
for w, r in AROUSAL:
    merged.setdefault(w, {})["arousal"] = round(r, 3)

def sent(w: str) -> str:                 # minimal context sentence
    return f"{w}"

FEWSHOT = []
for w, dims in merged.items():
    v = dims.get("valence", 0.0)         # fill missing with 0
    a = dims.get("arousal", 0.0)
    FEWSHOT.extend([
        {"role": "user", "content": sent(w)},
        {"role": "assistant",
         "content": json.dumps({"valence": v, "arousal": a})}
    ])

extra_rows = [
    # (valence, arousal, text) from EmoBank normalized between [-1, 1]
    (0.000,  0.000, "Remember what she said in my last letter?"),
    (0.280,  0.140, "Sherry learned through our Future Works class that she could rise out of the mire of the welfare system and support her family."),
    (0.100,  0.100, "Goodwill prepares people for life-long employment."),
    (-0.715,  0.500, "My girlfriend has disappeared, I don't even know where to start looking, and I need help!"),
]

for v, a, txt in extra_rows:
    FEWSHOT.extend([
        {"role": "user", "content": txt},
        {"role": "assistant", "content": json.dumps({"valence": v, "arousal": a})}
    ])

# 4. Core model
def classify(text:str, model:str="gpt-4o-mini", temp:float=0.0): # update model and temp
    msgs = [{"role":"system","content":SYSTEM_PROMPT}, *FEWSHOT,
            {"role":"user","content":text}]
    for _ in range(3):
        r = client.chat.completions.create(model=model, messages=msgs,
                                           temperature=temp)
        try:
            return json.loads(r.choices[0].message.content)
        except json.JSONDecodeError:
            msgs.append({"role":"system","content":"Respond with ONLY JSON."})
    raise RuntimeError("Model never returned valid JSON")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Valence-Arousal classifier")
    p.add_argument("text", nargs="+")
    p.add_argument("-m","--model",default="gpt-4o-mini")
    args = p.parse_args()
    print(json.dumps(classify(" ".join(args.text), args.model), indent=2))