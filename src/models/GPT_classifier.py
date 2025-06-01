"""
GPT_classifier.py
------------------------------------------------

Prompting of a GPT model to calculate a valence/arousal score given an inputted string

Additional point:
- Tenacity backoff wrapper to handle rate limits (because of Tier 1 on OpenAI)
"""
import json, re, os, threading, concurrent.futures as cf
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

#1- Get API key using Miko's API key. 
key_path = Path.home() / "Desktop" / "openai_key.txt" 
api_key = key_path.read_text().strip()           
client = OpenAI(api_key=api_key)

#2- Fixed system prompt.
SYSTEM_PROMPT = """
You are an affective-computing expert.
For each input, estimate:
- Valence: -1 (very negative) to 1 (very positive)
- Arousal: -1 (inactive) to 1 (excited)
For each input sentence, respond with exactly:
{"valence": <float>, "arousal": <float>}
No extra words, no explanation, no apologies—only a JSON object, nothing else.
""".strip()

#3- Tenacity backoff wrapper: https://cookbook.openai.com/examples/how_to_handle_rate_limits
"""
Rate limit: 500 RPM, 200,000 TPM, 2,000,000 TPD for o4-mini
"""
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    before_sleep=lambda retry_state: 
        print(f"[VaClassifier] Rate limit hit; retrying (attempt {retry_state.attempt_number})...")
)
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)
_completion = completion_with_backoff

_MAX_PAR = int(os.getenv("GPT_VA_MAX_WORKERS", "10"))
_sema = threading.Semaphore(_MAX_PAR)

def _call_openai(text: str, model: str, temperature: float):
    with _sema:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ]
        for _ in range(3):
            resp = _completion(model=model, messages=msgs, temperature=temperature)

            # 1) grab the raw content
            raw = resp.choices[0].message.content

            # 2) ensure it's a string before stripping
            if not isinstance(raw, str):
                raw = "" if raw is None else str(raw)

            content = raw.strip()
            content = re.sub(r"^\s*JSON\s+", "", content, flags=re.I)

            try:
                d = json.loads(content)
                # make sure valence/arousal are numeric
                v = float(d.get("valence", None))
                a = float(d.get("arousal", None))
                return {"valence": round(v, 2), "arousal": round(a, 2)}
            except Exception:
                # if parsing failed, explicitly ask for JSON-only and retry
                msgs.append({"role": "system", "content": "JSON only."})
        raise RuntimeError("Failed to return valid JSON for: " + text[:60])


def classify(text: str, model: str = "o4-mini", temperature: float = 1.0):
    #Return valence/arousal for a single string.
    return _call_openai(text, model, temperature)

def classify_many(texts, *, model: str = "o4-mini", temperature: float = 1.0,
                  max_workers: int | None = None):
    #Parallel VA scoring for an iterable of texts
    max_workers = max_workers or _MAX_PAR
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_call_openai, t, model, temperature) for t in texts]
        return [f.result() for f in futs]