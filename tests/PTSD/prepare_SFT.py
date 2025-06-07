"""
model1_daic.py  ―  DAIC-WOZ batch evaluator
------------------------------------------------
1. Load each transcript listed in  src/data/test_split.csv
2. Use LOCAL RoBERTa (get_va_scores) to predict Valence & Arousal per line
3. Prompt o4-mini with the annotated dialogue → single PHQ-8 score
4. Report MAE / RMSE
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

TEST_SPLIT_PATH = "/content/drive/My Drive/test_split.csv"
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
            df.to_csv("/content/drive/MyDrive/model1_outputted_va_scores/name")
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
        output_path="/content/drive/MyDrive/PTSD_results/PTSD_with_VA.jsonl",
        no_va=args.no_va,
        compute_va=args.compute_va
    )
    

if __name__ == "__main__":
    main()