import os
import glob
import json
import pandas as pd
import random
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
import re

API_KEY_PATH = Path("PATH")
MODEL = "gpt-4o-mini"
MAX_RETRIES = 3

client = OpenAI(api_key=API_KEY_PATH.read_text().strip())

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(MAX_RETRIES),
    before_sleep=lambda rs: print(f"[retry] attempt {rs.attempt_number} failed…")
)
def chat_with_backoff(messages):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
    )

# Manual MAE to avoid heavy scipy import
def mean_absolute_error_manual(y_true, y_pred):
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0

def summarize_text(text: str) -> str:
    prompt = (
        "Please provide a concise one-paragraph summary of the following conversation transcript:\n\n"
        f"{text}\n\nSummary:"
    )
    msgs = [
        {"role": "system",  "content": "You are an expert in text summarization."},
        {"role": "user",    "content": prompt},
    ]
    resp = chat_with_backoff(msgs)
    return resp.choices[0].message.content.strip()


def build_assessment_prompt(summary: str, va: dict | None = None) -> str:
    p = (
        f"You are a clinical assessor. Based on the following conversation summary:\n\n'''\n{summary}\n'''\n\n"
    )
    if va is not None:
        p += f"Average Valence: {va['valence']:.2f}, Average Arousal: {va['arousal']:.2f}\n\n"
    p += (
        "What is your estimate of the participant's PHQ score on a 0–27 scale?\n\n"
        "Answer exactly in JSON form:\n"
        '{"phq_estimate":float}\n'
    )
    return p


def extract_json(content: str) -> dict:
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("Could not parse JSON", content, 0)


def assess(summary: str, va: dict | None = None) -> float:
    prompt = build_assessment_prompt(summary, va)
    msgs = [
        {"role": "system", "content": "You are an expert clinical psychologist."},
        {"role": "user",   "content": prompt},
    ]
    resp = chat_with_backoff(msgs)
    content = resp.choices[0].message.content.strip()
    try:
        result = extract_json(content)
        return float(result.get("phq_estimate", 0.0))
    except json.JSONDecodeError:
        print(f"[Warning] Invalid JSON response: {content}")
        return 0.0

if not os.path.exists("test_split.csv"):
    raise FileNotFoundError("test_split.csv not found.")
labels_df = pd.read_csv("test_split.csv", index_col="Participant_ID")

va_agg = {}
for fn in tqdm(glob.glob("*_Transcript.csv"), desc="Computing true VA"):
    pid = Path(fn).stem.split("_")[0]
    df = pd.read_csv(fn)
    va_agg[pid] = {"valence": df["valence"].mean(), "arousal": df["arousal"].mean()}

summary_dir = Path("PATH")
summaries = {}
for pid in tqdm(labels_df.index.astype(str), desc="Summaries"):
    path = summary_dir / f"{pid}.txt"
    if path.exists():
        summaries[pid] = path.read_text(encoding="utf-8").strip()
    else:
        transcript_file = f"{pid}_Transcript.csv"
        if not os.path.exists(transcript_file):
            continue
        df = pd.read_csv(transcript_file)
        summaries[pid] = summarize_text(" ".join(df["Text"].tolist()))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summaries[pid], encoding="utf-8")

results = {}
modes = ["summary", "summary+va", "summary+random_va"]
for mode in tqdm(modes, desc="Running experiments"):
    y_true, y_pred = [], []
    for pid, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"Assessing {mode}"):
        pid_str = str(pid)
        summary = summaries.get(pid_str, "")
        if mode == "summary+va":
            va = va_agg.get(pid_str)
        elif mode == "summary+random_va":
            va = {"valence": random.uniform(-1, 1), "arousal": random.uniform(-1, 1)}
        else:
            va = None
        estimate = assess(summary, va)
        y_true.append(row["PHQ_Score"])
        y_pred.append(estimate)
    results[mode] = {"mae": mean_absolute_error_manual(y_true, y_pred)}

df = pd.DataFrame(results).T
print(df)
with open("results.txt", "w", encoding="utf-8") as f:
    for mode, m in results.items():
        f.write(f"{mode}: MAE = {m['mae']:.4f}\n")
print("Summaries loaded/created, results saved to results.txt")
