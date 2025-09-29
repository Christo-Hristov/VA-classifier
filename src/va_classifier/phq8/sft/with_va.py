"""
PHQ-8 Supervised Fine-Tuning Data Preparation with VA Context (§3.2.3)
======================================================================

Purpose:
    Prepare training data for supervised fine-tuning (SFT) of GPT models on PHQ-8
    depression prediction using VA-annotated clinical transcripts. This script
    augments each utterance with Valence/Arousal (-1..1) from a VA regressor and
    formats JSONL examples for fine-tuning.

Paper Section:
    §3.2.3 Supervised Fine-Tuning of GPT — VA-enhanced setting

Comparison / Key Finding:
    Baseline (text-only) → VA-enhanced SFT yields a 12.2% MAE reduction on PHQ-8
    (see Results tables). This script produces the VA-enhanced training set.

Pipeline (what this script does):
    1) Load dev/train split with Participant_ID and PHQ-8 labels
    2) Read each participant’s transcript CSV
    3) Infer per-line VA scores via the provided VA model (get_va_scores)
    4) Build a VA-aware prompt: "<valence>\t<arousal>\t<text>" per line
    5) Emit JSONL examples for OpenAI SFT (system/user/assistant format)

Key Features:
    - Per-utterance VA inference (valence, arousal ∈ [-1, 1])
    - Compact, line-wise tab-separated prompt format for clarity
    - Clinical system prompt (psychiatrist role; numeric-only target)
    - Robust handling of missing transcripts (warn and skip)

Training Example Format:
    System:
        "You are a clinical psychiatrist.
         Every line below has Valence and Arousal (-1 to 1).
         Estimate the participant's PHQ-8 total (0-24) and reply with **only** a number."
    User:
        "+0.12\t-0.05\tI’ve been sleeping more than usual
         -0.34\t+0.10\tIt’s hard to focus at work
         +0.05\t-0.02\tI still enjoy calling my sister"
    Assistant:
        "7"

Research Purpose:
    - Tests whether continuous affect (VA) improves PHQ-8 prediction in SFT
    - Provides reproducible, VA-enhanced training data for the PHQ-8 task
    - Aligns code artifacts with paper claims (12.2% MAE reduction)

Comparison Context:
    - Baseline generator: phq8_sft_without_va.py (text-only)
    - This file: VA-enhanced generator for direct A/B comparison

Related Files / Modules:
    - va_classifier.va_regressor.roberta_direct        # VA inference model
    - va_classifier.phq8.sft.evaluate_finetuned        # Evaluate fine-tuned models
    - phq8_sft_without_va.py                           # Text-only data generator (baseline)

Inputs (defaults in code):
    - Transcript CSV directory: TRANSCRIPT_DIR (e.g., "/content/.../edaic_transcripts")
    - Train/dev split CSV:      TRAIN_SPLIT_PATH  (e.g., "src/data/dev_split.csv")
    - VA model checkpoint:      --va_model (/path/to/roberta_va.ckpt or .pt)
    - OpenAI API key:           ~/Desktop/openai_key.txt or OPENAI_API_KEY env var

Outputs:
    - JSONL: OUT_JSONL_PATH (e.g., "/content/.../phq8_gpt_dev.jsonl")
      Each line is a fine-tuning example with VA-aware prompt and gold PHQ-8.

Usage:
    python phq8_sft_with_va.py \
        --va_model /path/to/roberta_va_checkpoint.pt \
        --transcripts /path/to/transcript_csv_folder

Notes:
    - This script is optimized for *data preparation* and inspection, not runtime speed.
    - Datasets are not included in the repo; see paper for access details and ethics.
"""

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
OUT_JSONL_PATH   = "/content/drive/MyDrive/phq8_gpt_dev.jsonl"

def add_va_scores(df: pd.DataFrame, model_path=None, device=None) -> pd.DataFrame:
    texts = df["Text"].fillna("").tolist()
    scores = get_va_scores(texts, model_path=model_path, device=device)
    df[["valence", "arousal"]] = pd.DataFrame(scores)
    return df

def format_prompt(df: pd.DataFrame) -> str:
    rows = [f"{v:+.2f}\t{a:+.2f}\t{t}" for v, a, t in df[["valence", "arousal", "Text"]].values]
    return "\n".join(rows)

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
        df = add_va_scores(df, model_path=args.va_model)
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
