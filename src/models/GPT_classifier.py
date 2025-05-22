"""
A few points.  
1- pip install --upgrade openai # in terminal
2- Change the path to the api_key (1)

Input 
    - Text input
Output
    - Valence (negative vs. positive)
    - Arousal (calm vs. excited)
    - Each of those take numeric values from [-1, 1] # we scaled this parameter for simplicity
"""
import json, argparse, re
from pathlib import Path
from openai import OpenAI

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  

#1- Get API key using Miko's API key. 
key_path = Path.home() / "Desktop" / "openai_key.txt" 
api_key = key_path.read_text().strip()           
client = OpenAI(api_key=api_key)

#2- Tenacity backoff wrapper: https://cookbook.openai.com/examples/how_to_handle_rate_limits
"""
Rate limit: 500 RPM, 200,000 TPM, 2,000,000 TPD
"""
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

# 3. Fixed system prompt. Using few-shot, instruction prompting sampling (+, +) amd (-, -) from EmoBank train/dev. 
SYSTEM_PROMPT = """
You are an affective-computing expert.

For each input, estimate:

- Valence: -1 (very negative) to 1 (very positive)
- Arousal: -1 (inactive) to 1 (excited)

Output ONLY JSON:
{"valence": float, "arousal": float}
(round both to two decimals)

Examples:
Input: Wonderful Simply Superb!
Output: {"valence": 0.8, "arousal": 0.65}
Input: migrate to 900,
Output: {"valence": -0.07, "arousal": -0.355}

Do NOT add any extra text.

Note: Happiness = positive valence, moderate arousal; Excitement = positive valence, high arousal; Sadness = negative valence, moderate arousal; Anger = negative valence, high arousal.
""".strip()

def classify(text: str, model: str = "o4-mini", temp: float = 1.0) -> dict:
    """
    Return {'valence': v, 'arousal': a} floats in [-1, 1].
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}]
    for _ in range(3):
        resp = completion_with_backoff(
            model=model, messages=messages, temperature=temp
        )
        content = resp.choices[0].message.content.strip()
        content = re.sub(r'^\s*JSON\s+', '', content, flags=re.I)
        try:
            obj = json.loads(content)
            return {
                "valence": round(float(obj["valence"]), 2),
                "arousal": round(float(obj["arousal"]), 2)
            }
        except Exception:
            messages.append({"role": "system",
                             "content": "Reply only with valid JSON."})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Few-shot VA classifier")
    ap.add_argument("text", nargs="+", help="Input sentence in quotes")
    ap.add_argument("-m", "--model", default="o4-mini",
                    help="OpenAI model name")
    ap.add_argument("-t", "--temp", type=float, default=1,
                    help="Sampling temperature (0-2)")
    args = ap.parse_args()

    result = classify(" ".join(args.text), model=args.model, temp=args.temp)
    print(json.dumps(result, indent=2))