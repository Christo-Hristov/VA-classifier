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

# ───────────────  LOCAL VA REGRESSOR  ─────────────── #
# util.py lives in src/models/; adjust if located elsewhere
from src.utils.util import get_va_scores

# ───────────────  CONSTANT PATHS  ─────────────── #

TRANSCRIPT_DIR = "/content/drive/MyDrive/edaic_transcripts"

MODEL_ID = "ft:gpt-4.1-mini-2025-04-14:personal:cs277-project:Be8KypW4"
TEST_SPLIT_PATH = "src/data/test_split.csv"
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

# ───────────────  PHQ-8 PROMPT  ─────────────── #
SYSTEM_PROMPT = """
You are a clinical psychiatrist.
Every line below has Valence and Arousal (-1 to 1).
Estimate the participant's PHQ-8 total (0-24) and reply with **only** a number.
""".strip()


def phq8_from_annotated(
    df: pd.DataFrame,
    model: str,
    temperature: float = 1.0
) -> float:
    rows   = [f"{v:+.2f}\t{a:+.2f}\t{t}"
              for v, a, t in df[["valence", "arousal", "Text"]].values]
    prompt = "\n".join(rows)

    resp = client.chat.completions.create(
        model=MODEL_ID,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )

    txt = str(resp.choices[0].message.content).strip()
    m   = re.search(r"([0-9]+(?:\.[0-9]+)?)", txt)
    print(txt, m)
    if not m:
        raise ValueError(f"PHQ parse error: {repr(txt)}")
    return float(m.group(1))

# ───────────────  MAIN  ─────────────── #
def main() -> None:
    ap = argparse.ArgumentParser(description="DAIC-WOZ PHQ-8 evaluator")
    ap.add_argument("--model",        default="o4-mini",
                    help="OpenAI model for PHQ estimation")
    ap.add_argument("--temperature",  type=float, default=1.0,
                    help="Sampling temperature for the PHQ step")
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

    gold, pred = [], []

    for _, row in tqdm(split.iterrows(), total=len(split), desc="Participants"):
        pid   = row["Participant_ID"]
        print(f"\n[{time.strftime('%H:%M:%S')}] → PID {pid}")
        csv_p = os.path.join(TRANSCRIPT_DIR, f"{pid}_Transcript.csv")
        if not os.path.exists(csv_p):
            print("  [WARN] transcript missing - skipped")
            continue

        df = pd.read_csv(csv_p)
        df = add_va_scores(df,
                           model_path=args.va_model,
                           device=args.device)

        try:
            phq = phq8_from_annotated(df,
                                      model=args.model,
                                      temperature=args.temperature)
        except Exception as e:
            print(f"  [ERROR] PHQ failed → {e}")
            phq = float("nan")

        gold.append(float(row["PHQ_Score"]))
        pred.append(phq)

    # ───── Metrics (skip NaNs) ─────
    pairs = [(g, p) for g, p in zip(gold, pred) if not math.isnan(p)]
    n     = len(pairs)
    mae   = (sum(abs(g-p) for g, p in pairs) / n) if n else float("nan")
    rmse  = (math.sqrt(sum((g-p)**2 for g, p in pairs) / n)
             if n else float("nan"))

    print(f"\nParticipants evaluated : {n}/{len(split)}")
    print(f"MAE                   : {mae:.3f}")
    print(f"RMSE                  : {rmse:.3f}")

if __name__ == "__main__":
    main()
