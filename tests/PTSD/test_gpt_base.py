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
import random

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
You are a highly experienced psychiatrist specializing in trauma and mental health disorders.

Your task is to analyze patient transcripts—containing only the patient's speech—and classify whether the
patient has PTSD or not. PTSD is a psychiatric condition that arises after exposure to traumatic events, characterized by symptoms such as hypervigilance, emotional numbing, intrusive thoughts, and avoidance.

Each line of the transcript contains:
- A valence score, reflecting the positivity or negativity of the patient’s emotion.
- An arousal score, reflecting the intensity of the emotion.
- The patient's spoken text.

Valence ranges from -1.0 (strongly negative) to +1.0 (strongly positive).
Arousal ranges from -1.0 (very calm) to +1.0 (very activated).


Output:

- First line: Estimate the participant's total score of PCL-5 (0-80) and reply with 'PCL-5 Score: [0-80]' 
- Second line: Output 0 if there is no indication of PTSD and 1 if PTSD is present. Reply with 'PTSD Binary: [0 or 1]'.

Overall, output should be 2 lines with a single number on each line.
""".strip()

### Generate 3 few shots


def make_few_shot_prompt(df: pd.DataFrame, severity: float, binary: int, no_va: bool = False) -> str:
    if no_va:
        header = f"{'Valence':>8} | {'Arousal':>8} | Text"
        separator = "-" * 60
        rows = [f"{'':>8} | {'':>8} | {t}" for t in df["Text"].fillna("")]
    else:
        header = f"{'Valence':>8} | {'Arousal':>8} | Text"
        separator = "-" * 60
        rows = [
            f"{v:+8.2f} | {a:+8.2f} | {t}"
            for v, a, t in df[["valence", "arousal", "Text"]].values
        ]

    transcript_block = "\n".join([header, separator] + rows)
    return f"{transcript_block}\nPCL-5 Score: {severity:.1f}\nPTSD Binary: {binary}\n"


# Format prompt


def format_prompt(df: pd.DataFrame, no_va: bool = False, random_va: bool = False) -> str:
    header = f"{'Valence':>8} | {'Arousal':>8} | Text"
    separator = "-" * 60

    if no_va:
        rows = [f"{'':>8} | {'':>8} | {t}" for t in df["Text"].fillna("")]
    elif random_va:
        rows = [
            f"{random.uniform(-0.5, 0.5):+8.2f} | {random.uniform(-0.5, 0.5):+8.2f} | {t}"
            for t in df["Text"].fillna("")
        ]
    else:
        rows = [
            f"{v:+8.2f} | {a:+8.2f} | {t}"
            for v, a, t in df[["valence", "arousal", "Text"]].values
        ]

    return "\n".join([header, separator] + rows)



def pcl5_from_annotated(
    df: pd.DataFrame,
    model: str,
    system_prompt: str,
    temperature: float = 1.0,
    no_va: bool = False,
    random_va: bool = False
) -> float:

    prompt = format_prompt(df, no_va=no_va, random_va=random_va)

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
    )
    txt = str(resp.choices[0].message.content).strip()
    
    severity_match = re.search(r"PCL-5 Score:\s*(\d+)", txt)
    if severity_match:
        severity = int(severity_match.group(1))
     
    binary_match = re.search(r"PTSD Binary:\s*(\d+)", txt)
    if binary_match:
        binary = int(binary_match.group(1))


    #severity = float(lines[0].strip())
    #binary = float(lines[1].strip())

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
    ap.add_argument("--no_va", action="store_true",
                help="Use only text (no valence/arousal) in prompt")
    ap.add_argument("--few_shot", action="store_true")
    ap.add_argument("--random_va", action="store_true")


    args = ap.parse_args()
    print(args.model_id)

    transcript_dir = TRANSCRIPT_DIR
    if args.transcripts:
        transcript_dir = args.transcripts

    split = pd.read_csv(TEST_SPLIT_PATH)
    if args.limit:
        split = split.head(args.limit)

    gold_severity, pred_severity = [], []
    gold_binary, pred_binary = [], []
    pids = []

    # Make few shot examples

    few_shot_contexts = []
    few_shot_limit = 3
    used_pids = set()
    #train_split = pd.read_csv("/content/drive/My Drive/train_split.csv")

    for _, row in split.iterrows():
        pid = row["Participant_ID"]
        if pid in used_pids:
            continue
        name = f"{pid}_Transcript.csv"
        if "va_pruned" in transcript_dir:
            name = f"{pid}_va_pruned_transcript.csv"
        if "length_pruned" in transcript_dir:
            name = f"{pid}_lengthpruned_transcript.csv"
        transcript_path = os.path.join(transcript_dir, name)
        if not os.path.exists(transcript_path):
            print(f"{transcript_path} does not exist")
            continue

        df = pd.read_csv(transcript_path)
        if not all(col in df.columns for col in ["Text"]):
            continue

        if not args.no_va and not all(col in df.columns for col in ["valence", "arousal"]):
            continue

        severity = float(row["PTSD_Severity"])
        binary = int(row["PTSD_Binary"])
        few_shot_contexts.append(make_few_shot_prompt(df, severity, binary, no_va=args.no_va))
        used_pids.add(pid)

        if len(few_shot_contexts) == few_shot_limit:
            break

    FEW_SHOT_CONTEXT = "\n---\n".join(few_shot_contexts).strip()


    if args.no_va:
        print("Without VA")
    else:
        print("With VA")

    if args.few_shot:
        final_system_prompt = SYSTEM_PROMPT + "\n\n--- FEW-SHOT EXAMPLES ---\n\n" + FEW_SHOT_CONTEXT
    else:
        final_system_prompt = SYSTEM_PROMPT
    
    print(final_system_prompt)

    for _, row in tqdm(split.iterrows(), total=len(split), desc="Participants"):
        pid   = row["Participant_ID"]
        pids.append(pid)
        print(f"\n[{time.strftime('%H:%M:%S')}] → PID {pid}")
        name = f"{pid}_Transcript.csv"
        if "va_pruned" in transcript_dir:
            name = f"{pid}_va_pruned_transcript.csv"
        if "length_pruned" in transcript_dir:
            name = f"{pid}_lengthpruned_transcript.csv"
        csv_p = os.path.join(transcript_dir, name)
        if not os.path.exists(csv_p):
            print("  [WARN] transcript missing - skipped")
            continue

        df = pd.read_csv(csv_p)

        try:
            severity, binary = pcl5_from_annotated(df,
                                      model=args.model_id,
                                      system_prompt=final_system_prompt,
                                      temperature=args.temperature,
                                      no_va=args.no_va, 
                                      random_va=args.random_va)
        except Exception as e:
            print(f"  [ERROR] PHQ failed → {e}")
            phq = float("nan")

        gold_severity.append(float(row["PTSD_Severity"]))
        pred_severity.append(severity)

        gold_binary.append(float(row["PTSD_Binary"]))
        pred_binary.append(binary)

    # Save results
    df = pd.DataFrame({"Paritcipant_ID" : pid, 
                    "GT Severity" : gold_severity,
                    "Predicted Severity" : pred_severity,
                    "GT Binary" : gold_binary,
                    "Predicted Binary" : pred_binary
                    })
    df.to_csv("/content/drive/MyDrive/PTSD_results/base_gpt.csv")


    # ───── Metrics (skip NaNs) ─────

    # MAE and RMSE
    pairs = [(g, p) for g, p in zip(gold_severity, pred_severity)
         if not math.isnan(g) and not math.isnan(p)]    
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
