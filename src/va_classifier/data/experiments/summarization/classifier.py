"""
Transcript Summarization Experiment for PHQ-8 Prediction (§4 Cross-cutting Experiments)

Purpose: Tests the impact of transcript summarization on PHQ-8 depression prediction accuracy
         by comparing full transcripts vs condensed summaries with/without VA context.
         This experiment addresses prompt length optimization for clinical AI applications.

Paper Section: §4 Cross-cutting Experiments - Summarization vs Full Transcript Analysis
Research Question: Can summarized transcripts maintain diagnostic accuracy while reducing prompt length?

This script implements three experimental conditions mentioned in the paper:

1. Summary Only (summary):
   - Uses GPT-4o-mini to generate concise one-paragraph summaries
   - Tests whether condensed content preserves diagnostic information
   - Reduces prompt length for efficiency and cost optimization
   - Baseline for summarization effectiveness

2. Summary + True VA (summary+va):
   - Combines transcript summaries with computed valence-arousal scores
   - VA scores calculated from full transcripts using RoBERTa regressor
   - Tests whether emotional context enhances summarized content
   - Optimal combination of efficiency and diagnostic information

3. Summary + Random VA (summary+random_va):
   - Uses summaries with randomly generated VA scores (-1 to 1)
   - Control condition to test VA score significance
   - Validates that true emotional context provides meaningful signal
   - Distinguishes VA benefit from random noise

Experimental Design:
- Systematic comparison across all test participants
- Consistent summarization prompts for reproducibility
- JSON-structured responses for reliable parsing
- MAE evaluation metric for performance comparison

Clinical Rationale:
- Long transcripts may overwhelm few-shot learning context windows
- Summaries could distill essential diagnostic information
- VA scores provide complementary emotional signals
- Efficiency gains important for clinical deployment scalability

Technical Implementation:
- Robust API handling with exponential backoff retry logic
- Automatic summary generation and caching for efficiency
- Multiple experimental modes with controlled comparisons
- Error handling for malformed JSON responses

Processing Pipeline:
1. Load test participant IDs and ground truth PHQ-8 scores
2. Compute average VA scores from full transcripts
3. Generate or load cached summaries for each participant
4. Run three experimental conditions:
   - Assess summaries alone
   - Assess summaries with true VA context
   - Assess summaries with random VA (control)
5. Calculate MAE for each condition and compare results

Output Metrics:
- MAE (Mean Absolute Error) for each experimental condition
- Comparative analysis of summarization impact
- VA context contribution assessment
- Results saved to results.txt for further analysis

Research Applications:
- Informs prompt optimization strategies for clinical AI
- Tests trade-offs between efficiency and diagnostic accuracy
- Validates VA enhancement value in condensed content
- Supports development of scalable clinical assessment tools

Related Files:
- va_classifier.phq8.autocot.o4_mini — Uses summarization in AutoCoT experiments
- va_classifier.ediac.experiments.length_truncation — Alternative content reduction
- results/experiments/summarization/ — Experimental outputs and summaries

Usage:
    python classifier.py
    # Requires: test_split.csv, *_Transcript.csv files with VA scores
    # Outputs: results.txt with MAE comparison across conditions

Note: Summaries are cached to avoid redundant API calls during repeated experiments.
      Random VA control validates that true emotional context provides meaningful signal.
"""

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
