"""
daic_gpt_test.py
------------------------------------------------
For every DAIC-WOZ participant in src/data/test_split.csv:
- Load transcript from Google Drive
- Then feed the annotated transcript to o4-mini and calculate a single PHQ-8 score
    - NO VA Inserts! 
- outputs results
"""

import os, argparse, math
from pathlib import Path

import pandas as pd
from openai import OpenAI

#1- Get Directorey and API key
TRANSCRIPT_DIR = (
    "/Users/michaelrimer/Library/CloudStorage/GoogleDrive-mikor@stanford.edu/"
    ".shortcut-targets-by-id/1ZsoGK8SvUwFMzu_xhgN5lWRtBV9Rdfq_/cs277 project/"
    "edaic_transcripts"
)
TEST_SPLIT_PATH = "src/data/test_split.csv"
API_KEY_PATH = os.path.expanduser("~/Desktop/openai_key.txt")

client = OpenAI(api_key=open(API_KEY_PATH).read().strip())

#2- System prompt for the LLM to estimate PHQ-8 score
SYSTEM_PROMPT = """
You are a clincial psychiatrist, trained on identifying depression.
Given a conversation estimate the participant's PHQ-8 total score
(0 to 24). Respond only with a single float.
""".strip()

def phq8_from_text(text: str, model: str, temp: float = 1.0) -> float:
    # Return PHQ‑8 float from the text.
    resp = client.chat.completions.create(
        model=model,
        temperature=temp,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
    )
    return float(resp.choices[0].message.content.strip())

# Main function to process each participant's transcript
def main() -> None:
    ap = argparse.ArgumentParser(description="DAIC PHQ-8 evaluator (no VA)")
    ap.add_argument("--model", default="o4-mini", help="OpenAI model name")
    ap.add_argument("--limit", type=int, help="Process only first N participants")
    args = ap.parse_args()

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
        transcript_text = "\n".join(df["Text"].tolist())
        pred_phq = phq8_from_text(transcript_text, model=args.model)

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