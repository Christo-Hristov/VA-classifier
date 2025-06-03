# Prepare JSON objects for transcripts with no VA

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

TRANSCRIPT_DIR = "/content/drive/MyDrive/edaic_transcripts"

TRAIN_SPLIT_PATH = "src/data/dev_split.csv"
OUT_JSONL_PATH   = "/content/drive/MyDrive/phq8_gpt_dev_noVA.jsonl"

def add_va_scores(df: pd.DataFrame, model_path=None, device=None) -> pd.DataFrame:
    texts = df["Text"].fillna("").tolist()
    scores = get_va_scores(texts, model_path=model_path, device=device)
    df[["valence", "arousal"]] = pd.DataFrame(scores)
    return df

def format_prompt(df: pd.DataFrame) -> str:
    return "\n".join(df["Text"].fillna("").tolist())

def main():
    ap = argparse.ArgumentParser(description="Prepare PHQ-8 fine-tuning data")
    ap.add_argument("--va_model", default=None,
                help="Path to RoBERTa VA checkpoint (.pt)")
    ap.add_argument("--transcripts", default=None,
                help="Folder containing <PID>_Transcript.csv files")
    args = ap.parse_args()
    
    train_split = pd.read_csv(TRAIN_SPLIT_PATH)
    dataset = []
    SYSTEM_PROMPT = """
    You are a clinical psychiatrist.
    Every line below has Valence and Arousal (-1 to 1).
    Estimate the participant's PHQ-8 total (0-24) and reply with **only** a number.
    """.strip()

    for _, row in tqdm(train_split.iterrows(), total=len(train_split), desc="Preparing examples"):
        pid = row["Participant_ID"]
        phq_score = int(row["PHQ_Score"])
        transcript_path = os.path.join(TRANSCRIPT_DIR, f"{pid}_Transcript.csv")

        if not os.path.exists(transcript_path):
            print(f"[WARN] Missing transcript for {pid} — skipped")
            continue

        df = pd.read_csv(transcript_path)
        #df = add_va_scores(df, model_path=args.va_model)
        prompt = format_prompt(df)
        print(f"The prompt is: {prompt}")
        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
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

if __name__ == "__main__":
    main()
