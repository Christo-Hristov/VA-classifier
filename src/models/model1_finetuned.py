import os, re, time, math, argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

from src.utils.util import get_va_scores

API_KEY_PATH = os.path.expanduser("~/Desktop/openai_key.txt")
api_key = Path(API_KEY_PATH).read_text().strip() if os.path.exists(API_KEY_PATH) else os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("No OpenAI API key found.")

client = OpenAI(api_key=api_key)

TRANSCRIPT_DIR = (
    "/Users/michaelrimer/Library/CloudStorage/GoogleDrive-mikor@stanford.edu/"
    ".shortcut-targets-by-id/1ZsoGK8SvUwFMzu_xhgN5lWRtBV9Rdfq_/cs277 project/"
    "edaic_transcripts"
)
TRAIN_SPLIT_PATH = "src/data/train_split.csv"
OUT_JSONL_PATH   = "phq8_gpt_train.jsonl"

def add_va_scores(df: pd.DataFrame, model_path=None, device=None) -> pd.DataFrame:
    texts = df["Text"].fillna("").tolist()
    scores = get_va_scores(texts, model_path=model_path, device=device)
    df[["valence", "arousal"]] = pd.DataFrame(scores)
    return df

def format_prompt(df: pd.DataFrame) -> str:
    rows = [f"{v:+.2f}\t{a:+.2f}\t{t}" for v, a, t in df[["valence", "arousal", "Text"]].values]
    return "\n".join(rows)

def main():
    train_split = pd.read_csv(TRAIN_SPLIT_PATH)
    dataset = []

    for _, row in tqdm(train_split.iterrows(), total=len(train_split), desc="Preparing examples"):
        pid = row["Participant_ID"]
        phq_score = int(row["PHQ_Score"])
        transcript_path = os.path.join(TRANSCRIPT_DIR, f"{pid}_Transcript.csv")

        if not os.path.exists(transcript_path):
            print(f"[WARN] Missing transcript for {pid} — skipped")
            continue

        df = pd.read_csv(transcript_path)
        df = add_va_scores(df)

        prompt = format_prompt(df)
        example = {
            "messages": [
                {"role": "system", "content": "You are a clinical psychiatrist. Estimate the PHQ-8 total from the annotated dialogue."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": str(phq_score)}
            ]
        }
        dataset.append(example)

    # Save as JSONL
    import json
    with open(OUT_JSONL_PATH, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"\n[INFO] Training data saved to {OUT_JSONL_PATH}")
    
    # Upload and fine-tune
    uploaded_file = client.files.create(
        file=open(OUT_JSONL_PATH, "rb"),
        purpose="fine-tune"
    )
    
    fine_tune_job = client.fine_tuning.jobs.create(
        training_file=uploaded_file.id,
        model="gpt-3.5-turbo",  # or "gpt-4" if available
        suffix="phq8-predictor"
    )

    print(f"[INFO] Fine-tuning started. Job ID: {fine_tune_job.id}")

if __name__ == "__main__":
    main()
