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
You are a highly experienced psychiatrist specializing in trauma and mental health disorders.

Your task is to analyze patient transcripts—containing only the patient's speech—and classify whether the
patient has PTSD or not. PTSD is a psychiatric condition that arises after exposure to traumatic events, characterized by symptoms such as hypervigilance, emotional numbing, intrusive thoughts, and avoidance.

Every sentence in the transcript has a Valence and Arousal (-1 to 1) score. Valence and arousal are two key dimensions of emotional experience often used in affective computing and psychology to quantify the emotional tone of text or speech. Valence measures how positive or negative an emotion is (e.g., happiness vs. sadness), while arousal measures the intensity or activation level of that emotion (e.g., calm vs. panicked). In individuals with PTSD, emotional responses are often dysregulated: they may show frequent negative valence (e.g., fear, guilt, sadness) and high arousal (e.g., anxiety, hyperalertness), even in neutral situations.

According to the DSM-5 diagnostic criteria, PTSD is characterized by: a. Intrusion Symptoms: At least one
symptom such as recurrent, involuntary, and intrusive distressing memories of the traumatic event(s);
recurrent distressing dreams related to the event(s); dissociative reactions (e.g., flashbacks) in which the
event seems to recur; intense or prolonged psychological distress at exposure to internal or external cues that
symbolize or resemble the traumatic event(s); or marked physiological reactions to such cues. b. Avoidance:
Persistent avoidance of stimuli associated with the traumatic event(s), evidenced by efforts to avoid
distressing memories, thoughts, or feelings about or closely associated with the event(s) and/or avoidance of
external reminders (people, places, conversations, activities, objects, or situations) that trigger these
memories. c. Negative Alterations in Cognitions and Mood: Two or more symptoms such as inability to
remember an important aspect of the traumatic event(s) (typically due to dissociative amnesia); persistent and
exaggerated negative beliefs or expectations about oneself, others, or the world; persistent, distorted
cognitions about the cause or consequences of the traumatic event(s) leading to self-blame or blaming others;
persistent negative emotional state (e.g., fear, horror, anger, guilt, or shame); markedly diminished interest in
significant activities; feelings of detachment or estrangement from others; or a persistent inability to
experience positive emotions. d. Alterations in Arousal and Reactivity: Two or more symptoms such as
irritable behavior and angry outbursts (with little or no provocation); reckless or self-destructive behavior;
hypervigilance; exaggerated startle response; problems with concentration; or sleep disturbances.


Output:

- First line: Estimate the participant's total score of PCL-5 (0-80) and reply with **only** a number.
- Second line: Output 0 if there is no indication of PTSD and 1 if PTSD is present. Reply with **only** a number.

Overall, output should be 2 lines with a single number on each line.
""".strip()


def format_prompt(df: pd.DataFrame, no_va: bool = False) -> str:
    if no_va:
        return "\n".join(df["Text"].dropna().tolist())
    
    header = f"{'Valence':>8} | {'Arousal':>8} | Text"
    separator = "-" * 60
    rows = [
        f"{v:+8.2f} | {a:+8.2f} | {t}"
        for v, a, t in df[["valence", "arousal", "Text"]].values
    ]
    return "\n".join([header, separator] + rows)



def pcl5_from_annotated(
    df: pd.DataFrame,
    model: str,
    temperature: float = 1.0,
    no_va: bool = False
) -> float:

    prompt = format_prompt(df, no_va=no_va)

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    txt = str(resp.choices[0].message.content).strip()
    #print(txt)
    lines = txt.splitlines()
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
    ap.add_argument("--no_va", action="store_true",
                help="Use only text (no valence/arousal) in prompt")

    args = ap.parse_args()


    split = pd.read_csv(TEST_SPLIT_PATH)
    if args.limit:
        split = split.head(args.limit)

    gold_severity, pred_severity = [], []
    gold_binary, pred_binary = [], []
    pids = []

    for _, row in tqdm(split.iterrows(), total=len(split), desc="Participants"):
        pid   = row["Participant_ID"]
        pids.append(pid)
        print(f"\n[{time.strftime('%H:%M:%S')}] → PID {pid}")
        csv_p = os.path.join(TRANSCRIPT_DIR, f"{pid}_Transcript.csv")
        if not os.path.exists(csv_p):
            print("  [WARN] transcript missing - skipped")
            continue

        df = pd.read_csv(csv_p)

        try:
            severity, binary = pcl5_from_annotated(df,
                                      model=args.model_id,
                                      temperature=args.temperature,
                                      no_va=args.no_va)
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
