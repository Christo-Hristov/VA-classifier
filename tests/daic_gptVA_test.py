"""
daic_gptVA_test.py
------------------------------------------------

For every DAIC-WOZ participant in src/data/test_split.csv:
- Load transcript from Google Drive
- Compute Valence-Arousal (VA) scores for each line/row in text column
- Then feed the annotated transcript to o4-mini and calculate a single PHQ-8 score
- outputs results
"""

import os, argparse, math, re, time
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from openai import OpenAI
from src.models.GPT_classifier import classify, classify_many

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
def add_va_scores(df: pd.DataFrame, model: str, temperature: float) -> pd.DataFrame:
    texts = df["Text"].fillna("").tolist()

    try:
        # fast, multi-threaded call
        scores = classify_many(
            texts,
            model=model,
            temperature=temperature,   # o4-mini ignores this but keep API pure
            max_workers=10
        )
    except Exception as e:
        # if *any* line blows up, drop to slow but safe per-line loop
        print(f"[WARNING] classify_many crashed → {e}\n-- falling back one-by-one")
        scores = []
        for t in tqdm(texts, desc=" VA scoring (fallback)", leave=False):
            try:
                scores.append(classify(t, model=model, temperature=temperature))
            except Exception as ie:
                print(f"[WARN] VA failed for “{t[:30]}…” → {ie}")
                scores.append({"valence": 0.0, "arousal": 0.0})

    df[["valence", "arousal"]] = pd.DataFrame(scores)
    return df

#3- System prompt for the LLM to estimate PHQ-8 score
SYSTEM_PROMPT = """
You are a clincial psychiatrist, trained on identifying depression.
Given a conversation where every line has Valence and
Arousal scores (scaled from -1 to 1), estimate the participant's PHQ-8 total score
(0 to 24). Respond only with a single float.
""".strip()

def phq8_from_annotated(df: pd.DataFrame, model: str, temperature: float = 1.0) -> float:
    rows = [f"{v:.2f}\t{a:.2f}\t{t}" for v, a, t in df[["valence", "arousal", "Text"]].values]
    prompt = "\n".join(rows)

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    raw = resp.choices[0].message.content
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)

    txt = raw.strip()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", txt)
    if not m:
        raise ValueError(f"LLM did not return a parseable number: {repr(txt)}")
    return float(m.group(1))


# Main function to process each participant's transcript
def main() -> None:
    p = argparse.ArgumentParser(description="DAIC PHQ-8 batch evaluator")
    p.add_argument("--model", default="o4-mini", help="OpenAI model name")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature for both VA and PHQ calls")
    p.add_argument("--limit", type=int, help="Only first N participants")
    args = p.parse_args()

    split = pd.read_csv(TEST_SPLIT_PATH)
    if args.limit:
        split = split.head(args.limit)

    gold, pred = [], []

    # 6) Wrap the participant loop in tqdm
    for _, row in tqdm(split.iterrows(), total=len(split), desc="Participants"):
        pid = row["Participant_ID"]
        print(f"\n[{time.strftime('%H:%M:%S')}] → Processing PID = {pid}")
        t_path = os.path.join(TRANSCRIPT_DIR, f"{pid}_Transcript.csv")
        if not os.path.exists(t_path):
            print(f"  [WARNING] transcript missing, skipping {pid}")
            continue

        df = pd.read_csv(t_path)

        # 7) Pass args.temperature into add_va_scores
        df = add_va_scores(df, model=args.model, temperature=args.temperature)
        print(f"  [INFO] VA done for {pid}, now PHQ estimation…")

        try:
            # 8) Call with `temperature=` to match the updated signature
            pred_phq = phq8_from_annotated(df, model=args.model, temperature=args.temperature)
        except Exception as e:
            print(f"  [ERROR] PHQ failed for {pid}: {e}")
            pred_phq = float("nan")

        gold.append(float(row["PHQ_Score"]))
        pred.append(pred_phq)

    # 9) Compute metrics (skip NaNs if any)
    valid_pairs = [(g, p) for g, p in zip(gold, pred) if not math.isnan(p)]
    n = len(valid_pairs)
    if n > 0:
        mae  = sum(abs(g - p) for g, p in valid_pairs) / n
        rmse = math.sqrt(sum((g - p) ** 2 for g, p in valid_pairs) / n)
    else:
        mae = rmse = float("nan")

    print(f"\nParticipants evaluated : {n}/{len(split)}")
    print(f"MAE                   : {mae:.3f}")
    print(f"RMSE                  : {rmse:.3f}")

if __name__ == "__main__":
    main()