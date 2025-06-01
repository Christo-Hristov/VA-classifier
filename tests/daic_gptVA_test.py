"""
daic_gptVA_test.py
------------------------------------------------

For every DAIC-WOZ participant in src/data/test_split.csv:
- Load transcript from Google Drive
- Compute Valence-Arousal (VA) scores for each line/row in text column
- Then feed the annotated transcript to o4-mini and calculate a single PHQ-8 score
- outputs results
"""

import os, argparse, math, re
from pathlib import Path

import pandas as pd
from openai import OpenAI
from src.models.GPT_classifier import classify_many   # batch VA scorer

#1- Get Directorey and API key
TRANSCRIPT_DIR = (
    "/Users/michaelrimer/Library/CloudStorage/GoogleDrive-mikor@stanford.edu/"
    ".shortcut-targets-by-id/1ZsoGK8SvUwFMzu_xhgN5lWRtBV9Rdfq_/cs277 project/"
    "edaic_transcripts"
)
TEST_SPLIT_PATH = "src/data/test_split.csv"
API_KEY_PATH = os.path.expanduser("~/Desktop/openai_key.txt")

client = OpenAI(api_key=open(API_KEY_PATH).read().strip())

#2- For each line in the transcript, compute Valence and Arousal score
def add_va_scores(df: pd.DataFrame, model: str) -> pd.DataFrame:
    # Compute VA for each line in parallel.
    scores = classify_many(
        df["Text"].tolist(),
        model=model,
        temperature=1.0,      
        max_workers=10         
    )
    df[["valence", "arousal"]] = pd.DataFrame(scores)
    return df

#3- System prompt for the LLM to estimate PHQ-8 score
SYSTEM_PROMPT = """
You are a clincial psychiatrist, trained on identifying depression.
Given a conversation where every line has Valence and
Arousal scores (scaled from -1 to 1), estimate the participant's PHQ-8 total score
(0 to 24). Respond only with a single float.
""".strip()

def phq8_from_annotated(df: pd.DataFrame, model: str, temp: float = 1.0) -> float:
    rows = [f"{v:.2f}\t{a:.2f}\t{t}" for v, a, t in df[["valence", "arousal", "Text"]].values]
    prompt = "\n".join(rows)

    resp = client.chat.completions.create(
        model=model,
        temperature=temp,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    raw = resp.choices[0].message.content
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)

    txt = raw.strip()

    # Try to extract the first numeric token (integer or decimal)
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", txt)
    if not m:
        raise ValueError(f"LLM did not return a parseable number: {repr(txt)}")
    return float(m.group(1))


# Main function to process each participant's transcript
def main() -> None:
    p = argparse.ArgumentParser(description="DAIC PHQ-8 batch evaluator")
    p.add_argument("--model", default="o4-mini", help="OpenAI model name")
    p.add_argument("--limit", type=int, help="Only first N participants")
    args = p.parse_args()

    split = pd.read_csv(TEST_SPLIT_PATH)
    if args.limit:
        split = split.head(args.limit)

    gold, pred = [], []

    for _, row in split.iterrows():
        pid = row["Participant_ID"]
        t_path = os.path.join(TRANSCRIPT_DIR, f"{pid}_Transcript.csv")
        if not os.path.exists(t_path):
            continue

        df = pd.read_csv(t_path)
        df = add_va_scores(df, model=args.model)
        pred_phq = phq8_from_annotated(df, model=args.model)

        gold.append(float(row["PHQ_Score"]))
        pred.append(pred_phq)

    n = len(gold)
    mae  = sum(abs(g - p) for g, p in zip(gold, pred)) / n
    rmse = math.sqrt(sum((g - p) ** 2 for g, p in zip(gold, pred)) / n)

    print(f"Participants evaluated : {n}")
    print(f"MAE                   : {mae:.3f}")
    print(f"RMSE                  : {rmse:.3f}")

if __name__ == "__main__":
    main()