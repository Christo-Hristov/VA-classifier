"""
PHQ-8 Supervised Fine-Tuning Data Preparation without VA Context (§3.2.3)
==========================================================================

Purpose: Prepares training data for supervised fine-tuning of GPT models on PHQ-8 
         depression prediction using text-only clinical transcripts. This creates 
         the baseline comparison for evaluating VA enhancement benefits.

Paper Section: §3.2.3 Supervised Fine-Tuning of GPT (baseline comparison)
Comparison: Baseline for measuring VA context improvement (12.2% improvement with VA)

This script implements the text-only data preparation pipeline for SFT:
1. Load Training Split: Read development/training participant IDs and PHQ-8 labels
2. Text-Only Formatting: Create simple prompts with raw transcript text
3. JSONL Creation: Format examples for OpenAI fine-tuning API
4. Training Data Export: Save formatted dataset for supervised fine-tuning
5. Baseline Establishment: Enable comparison against VA-enhanced approach

Key Features:
- Pure text-based training (no emotional context)
- Simple prompt format: Raw transcript lines concatenated
- Clinical system prompt for psychiatrist role-playing
- Concise output format (number-only responses for efficiency)
- Robust error handling for missing transcripts

Text-Only Training Format:
System: "You are a clinical psychiatrist. Every line below has Valence and Arousal (-1 to 1). 
         Estimate the participant's PHQ-8 total (0-24) and reply with **only** a number."

User: "I've been feeling okay lately
       But sometimes I get really down
       It's hard to concentrate on anything"

Research Purpose:
- Establishes baseline performance for text-only fine-tuning
- Enables measurement of VA context value (12.2% improvement)
- Tests whether emotional signals are necessary for clinical prediction
- Provides control condition for multimodal enhancement evaluation

Comparison Context:
- Baseline for va_classifier.phq8.sft.with_va (VA-enhanced version)
- Tests core research hypothesis about emotional context value
- Enables rigorous evaluation of multimodal approach benefits

Related Files:
- va_classifier.phq8.sft.with_va — VA-enhanced version (best performance)
- va_classifier.phq8.sft.evaluate_finetuned — Evaluation of fine-tuned models
- va_classifier.va_regressor.roberta_direct — VA score generation (unused here)

Usage:
    python without_va.py --transcripts /path/to/transcripts
    # Generates text-only training data for baseline comparison

Output:
    - Training data: phq8_gpt_dev_noVA.jsonl (text-only format)
    - Used for establishing baseline fine-tuning performance
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
