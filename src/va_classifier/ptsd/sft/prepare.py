"""
PTSD Supervised Fine-Tuning Data Preparation with VA Context (§3.3.3)

Purpose: Prepares training data for supervised fine-tuning of GPT models on PTSD prediction
         using VA-annotated clinical transcripts from E-DAIC dataset. This script creates
         JSONL training examples for both PCL-5 severity scoring and binary PTSD classification.

Paper Section: §3.3.3 Supervised Fine-Tuning for PTSD Prediction
Task: Dual-output PTSD assessment - PCL-5 severity (0-80) + binary classification (0/1)

This script implements the data preparation pipeline for PTSD fine-tuning:

1. Transcript Processing:
   - Loads clinical interview transcripts from E-DAIC dataset
   - Supports multiple transcript variants (full, VA-pruned, length-pruned)
   - Handles missing transcripts gracefully with skip warnings

2. VA Score Integration:
   - Optionally computes VA scores using RoBERTa direct regressor
   - Integrates valence-arousal context into training prompts
   - Supports text-only mode (--no_va) for baseline comparison

3. PTSD-Specific Formatting:
   - Creates structured prompts with VA scores and transcript text
   - Formats ground truth as dual outputs: PCL-5 + binary PTSD
   - Uses clinical psychiatrist role for domain-appropriate responses

4. JSONL Generation:
   - Produces OpenAI-compatible fine-tuning format
   - System/user/assistant message structure
   - Saves training examples for supervised learning

Clinical Context:
- PTSD assessment requires analyzing trauma-related linguistic patterns
- PCL-5 (PTSD Checklist for DSM-5) provides severity scoring (0-80)
- Binary classification identifies PTSD presence/absence
- VA context may capture emotional numbing, hypervigilance patterns

Training Format Example:
System: "You are a highly experienced psychiatrist specializing in trauma..."
User:   "Valence | Arousal | Text
         +0.12   | -0.05   | I have trouble sleeping
         -0.34   | +0.10   | Loud noises make me jump"
Assistant: "PCL-5 Score: 45
           PTSD Binary: 1"

Key Features:
- Dual-output prediction (severity + classification)
- VA-enhanced prompts for emotional pattern recognition
- Support for multiple transcript preprocessing variants
- Robust error handling for missing data
- Clinical domain-specific system prompting

Configuration Options:
- --no_va: Text-only training (baseline)
- --compute_va: Generate VA scores on-the-fly
- --transcripts: Custom transcript directory
- --va_model: Specify RoBERTa VA model path

Research Applications:
- Tests VA context impact on PTSD prediction accuracy
- Enables comparison with PHQ-8 depression fine-tuning
- Supports analysis of trauma-specific linguistic patterns
- Foundation for clinical PTSD screening tools

Related Files:
- va_classifier.ptsd.sft.evaluate — Fine-tuned model evaluation
- va_classifier.va_regressor.roberta_direct — VA score generation
- va_classifier.phq8.sft.prepare — PHQ-8 equivalent for comparison

Usage:
    # VA-enhanced training data
    python prepare.py --compute_va
    
    # Text-only baseline
    python prepare.py --no_va
    
    # Use pre-computed VA scores
    python prepare.py --transcripts /path/to/va_annotated/

Output:
    PTSD_train_no_VA.jsonl — Training data for OpenAI fine-tuning API
    Contains dual-output examples for PCL-5 severity + binary classification

Note: PTSD prediction requires specialized clinical expertise and should be used
      only as a screening tool, not for definitive diagnosis.
"""

import os, argparse, math, re, time
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ───────────────  LOCAL VA REGRESSOR  ─────────────── #
# util.py lives in src/models/; adjust if located elsewhere
from src.utils.util import get_va_scores

# ───────────────  CONSTANT PATHS  ─────────────── #

TRANSCRIPT_DIR = "/content/drive/MyDrive/model1_outputted_va_scores"

TEST_SPLIT_PATH = "/content/drive/My Drive/train_split.csv"
API_KEY_PATH = os.path.expanduser("~/Desktop/openai_key.txt")
VA_MODEL_PATH = "/content/drive/MyDrive/VA-classifier/src/models/roberta_direct_emobank/3_hidden_256_128_patience_5_finetune_1e-05.pt"

if os.path.exists(API_KEY_PATH):
    api_key = Path(API_KEY_PATH).read_text().strip()
else:                                # ← fallback to env-var
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No openai_key.txt file and OPENAI_API_KEY not set."
        )

client = OpenAI(api_key=api_key)


# ───────────────  VA SCORING  ─────────────── #
def add_va_scores(
    df: pd.DataFrame,
    model_path: str | None = None,
    device: str | None     = None
) -> pd.DataFrame:
    """
    Run RoBERTa VA regressor on every row in df["Text"] and
    append the predictions as two new columns.
    """
    texts   = df["Text"].fillna("").tolist()
    scores  = get_va_scores(texts, model_path=model_path, device=device)
    df[["valence", "arousal"]] = pd.DataFrame(scores)
    return df

# ───────────────  PTSD PROMPT  ─────────────── #
SYSTEM_PROMPT = """
You are a highly experienced psychiatrist specializing in trauma and mental health disorders.

Your task is to analyze patient transcripts—containing only the patient's speech—and classify whether the
patient has PTSD or not. PTSD is a psychiatric condition that arises after exposure to traumatic events, characterized by symptoms such as hypervigilance, emotional numbing, intrusive thoughts, and avoidance.

Output:

- First line: Estimate the participant's total score of PCL-5 (0-80) and reply with 'PCL-5 Score: [0-80]' 
- Second line: Output 0 if there is no indication of PTSD and 1 if PTSD is present. Reply with 'PTSD Binary: [0 or 1]'.

Overall, output should be 2 lines with a single number on each line.
""".strip()

# Format prompt


def format_prompt(df: pd.DataFrame, no_va: bool = False) -> str:
    header = f"{'Valence':>8} | {'Arousal':>8} | Text"
    separator = "-" * 60

    if no_va:
        rows = [f"{'':>8} | {'':>8} | {t}" for t in df["Text"].fillna("")]
    else:
        rows = [
            f"{v:+8.2f} | {a:+8.2f} | {t}"
            for v, a, t in df[["valence", "arousal", "Text"]].values
        ]

    return "\n".join([header, separator] + rows)



# Prepare JSONL for SFT
def prepare_supervised_jsonl(split_df: pd.DataFrame, transcript_dir: str, va_model: str, output_path: str, no_va: bool = False, compute_va: bool = False):
    examples = []
    if compute_va:
        transcript_dir = "/content/drive/MyDrive/edaic_transcripts"

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Preparing fine-tune data"):
        pid = row["Participant_ID"]

        # Choose file variant
        name = f"{pid}_Transcript.csv"
        if "va_pruned" in transcript_dir:
            name = f"{pid}_va_pruned_transcript.csv"
        if "length_pruned" in transcript_dir:
            name = f"{pid}_lengthpruned_transcript.csv"
        transcript_path = os.path.join(transcript_dir, name)

        if not os.path.exists(transcript_path):
            print(f"[SKIP] {transcript_path} missing.")
            continue

        df = pd.read_csv(transcript_path)
        if compute_va:
            df = add_va_scores(df,
                            model_path=va_model)
            df.to_csv(f"/content/drive/MyDrive/model1_outputted_va_scores/{name}")
        if not all(col in df.columns for col in ["Text"]):
            continue
        if not no_va and not all(col in df.columns for col in ["valence", "arousal"]):
            continue

        prompt = format_prompt(df, no_va=no_va)
        print(prompt)
        completion = f"PCL-5 Score: {int(row['PTSD Severity'])}\nPTSD Binary: {int(row['PCL-C (PTSD)'])}"

        entry = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
        }
        examples.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    print(f"✅ Saved {len(examples)} fine-tune examples to: {output_path}")




# ───────────────  MAIN  ─────────────── #
def main() -> None:
    ap = argparse.ArgumentParser(description="DAIC-WOZ PCL evaluator")
    ap.add_argument("--model_id",        default="gpt-4o-mini-2024-07-18",
                    help="OpenAI model for PCL-5 estimation")
    ap.add_argument("--temperature",  type=float, default=1.0,
                    help="Sampling temperature for the PCL step")
    ap.add_argument("--va_model",     default=None,
                    help="Path to RoBERTa VA checkpoint (.pt). "
                         "If omitted, util.py uses its default.")
    ap.add_argument("--device",       default=None,
                    help="'cpu' or 'cuda'; util.py auto-detects if None")
    ap.add_argument("--transcripts",  default=None,
               help="Folder that contains the <PID>_Transcript.csv files")
    ap.add_argument("--limit",        type=int,
                    help="Only first N participants (debug)")
    ap.add_argument("--no_va", action="store_true",
                help="Use only text (no valence/arousal) in prompt")
    ap.add_argument("--few_shot", action="store_true")
    ap.add_argument("--compute_va", action="store_true")



    args = ap.parse_args()

    transcript_dir = TRANSCRIPT_DIR
    if args.transcripts:
        transcript_dir = args.transcripts

    train_split = pd.read_csv("/content/drive/My Drive/train_split.csv")
    prepare_supervised_jsonl(
        split_df=train_split,
        transcript_dir=transcript_dir,
        va_model=VA_MODEL_PATH,
        output_path="/content/drive/MyDrive/PTSD_results/PTSD_train_no_VA.jsonl",
        no_va=args.no_va,
        compute_va=args.compute_va
    )
    

if __name__ == "__main__":
    main()