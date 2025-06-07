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

if os.path.exists(API_KEY_PATH):
    api_key = Path(API_KEY_PATH).read_text().strip()
else:                                # ← fallback to env-var
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No openai_key.txt file and OPENAI_API_KEY not set."
        )

client = OpenAI(api_key=api_key)

# ───────────────  PTSD PROMPT  ─────────────── #
SYSTEM_PROMPT = """
You are a clinical psychiatrist and want to identify PTSD from clinical transcripts.
Every sentence below has a Valence and Arousal (-1 to 1) score.
Estimate the participant's total score of PCL-5 (0-80) and reply with **only** a number.
Estimate whether a participant has PTSD (0 if negative, 1 if positive) and reply with **only** a number in the next line.
Overall, output should be 2 lines with a single number on each line.
""".strip()


def pcl5_from_annotated(
    df: pd.DataFrame,
    model: str,
    temperature: float = 1.0
) -> float:

    rows = [f"{v:+.2f}\t{a:+.2f}\t{t}" for v, a, t in df[["valence", "arousal", "Text"]].values]
    prompt = "\n".join(rows)

    print(prompt)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    txt = str(resp.choices[0].message.content).strip()
    print(txt)
    lines = response_text.splitlines()
    severity = float(lines[0].strip())
    binary = float(lines[1].strip())

    print(f"severity: {severity}")
    print(f"binary: {binary}")

    return severity, binary

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

    args = ap.parse_args()


    split = pd.read_csv(TEST_SPLIT_PATH)
    if args.limit:
        split = split.head(args.limit)

    gold_severity, pred_severity = [], []
    gold_binary, pred_binary = [], []

    for _, row in tqdm(split.iterrows(), total=len(split), desc="Participants"):
        pid   = row["Participant_ID"]
        print(row)
        print(f"\n[{time.strftime('%H:%M:%S')}] → PID {pid}")
        csv_p = os.path.join(TRANSCRIPT_DIR, f"{pid}_Transcript.csv")
        if not os.path.exists(csv_p):
            print("  [WARN] transcript missing - skipped")
            continue

        df = pd.read_csv(csv_p)

        try:
            severity, binary = pcl5_from_annotated(df,
                                      model=args.model_id,
                                      temperature=args.temperature)
        except Exception as e:
            print(f"  [ERROR] PHQ failed → {e}")
            phq = float("nan")

        gold_severity.append(float(row["PTSD_Severity"]))
        pred_severity.append(severity)

        gold_binary.append(float(row["PTSD_Binary"]))
        pred_binary.append(binary)

    # ───── Metrics (skip NaNs) ─────

    # MAE and RMSE
    pairs = [(g, p) for g, p in zip(gold, pred) if not math.isnan(p)]
    n     = len(pairs)
    mae   = (sum(abs(g-p) for g, p in pairs) / n) if n else float("nan")
    rmse  = (math.sqrt(sum((g-p)**2 for g, p in pairs) / n)
             if n else float("nan"))
    
    # Accuracy and precision/recall

    gold_binary_int = [int(g) for g in gold_binary]
    pred_binary_int = [int(p) for p in pred_binary]

    accuracy  = accuracy_score(gold_binary_int, pred_binary_int)
    precision = precision_score(gold_binary_int, pred_binary_int)
    recall    = recall_score(gold_binary_int, pred_binary_int)


    print(f"\nParticipants evaluated : {n}/{len(split)}")
    print(f"MAE                   : {mae:.3f}")
    print(f"RMSE                  : {rmse:.3f}")
    print(f"Accuracy                  : {accuracy:.3f}")
    print(f"Precision                  : {precision:.3f}")
    print(f"Recall                  : {precision:.3f}")



if __name__ == "__main__":
    main()
