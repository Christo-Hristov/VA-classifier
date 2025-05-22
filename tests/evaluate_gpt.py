"""
evaluate_emo.py
Evaluate testopen classify function on the EmoBank test split.

Returns the following metrics:
- MAE (mean absolute error) in 2D (valence + arousal)
- Pearson correlation (r) in 1D (valence accuracy and arousal accuracy)
"""
import argparse, tempfile, urllib.request, time, concurrent.futures as cf
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr
import json, re
from GPT_classifer import classify

RAW_URL = ("https://raw.githubusercontent.com/"
           "JULIELab/EmoBank/master/corpus/emobank.csv")

to_unit = lambda x: (x - 3.0) / 2.0  # convert the EmoBank 1-5 scale to [-1, 1]

def get_local_csv(path_or_url: str) -> Path:
    if path_or_url.startswith("http"):
        print("⬇️  Downloading EmoBank …")
        tmp = Path(tempfile.mkstemp(suffix=".csv")[1])
        urllib.request.urlretrieve(path_or_url, tmp)
        return tmp
    return Path(path_or_url).expanduser()

def va_from_any(raw): # convert classify() output to valence/arousal tupl
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        v, a = raw
    elif isinstance(raw, dict):
        v, a = raw.get("valence"), raw.get("arousal")
    elif isinstance(raw, str):
        s = re.sub(r'^\s*json\s+', '', raw, flags=re.I).strip().rstrip('.')
        obj = json.loads(s)
        v, a = obj.get("valence"), obj.get("arousal")
    else:
        raise ValueError(f"Unrecognized classify() output: {raw!r}")
    return float(v), float(a)

def mae_2d(gold_v, gold_a, pred_v, pred_a): # calc MAE score
    total = sum(abs(gv - pv) + abs(ga - pa)
                for gv, pv, ga, pa in zip(gold_v, pred_v, gold_a, pred_a))
    return total / (len(gold_v) * 2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=RAW_URL)
    p.add_argument("-m", "--model", default="o4-mini")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent API calls (default: 1 for safety)")
    args = p.parse_args()

    df = pd.read_csv(get_local_csv(args.csv), index_col=0)
    test = df[df["split"] == "test"]
    if args.limit:
        test = test.head(args.limit)
    gold_V = [to_unit(v) for v in test["V"]]
    gold_A = [to_unit(a) for a in test["A"]]

    def call(text):
        return va_from_any(classify(text, model=args.model, temp=args.temp))

    t0 = time.time()
    pred_V, pred_A = [None]*len(test), [None]*len(test)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(call, txt): idx for idx, txt in enumerate(test["text"])}
        for j, fut in enumerate(cf.as_completed(futures), 1):
            i = futures[fut]
            v, a = fut.result()
            pred_V[i], pred_A[i] = v, a
            if j % 25 == 0:
                print(f"{j}/{len(test)} processed…", flush=True)

    # Metrics
    mae = mae_2d(gold_V, gold_A, pred_V, pred_A)
    r_v, _ = pearsonr(gold_V, pred_V) if len(set(pred_V)) > 1 else (float('nan'), None)
    r_a, _ = pearsonr(gold_A, pred_A) if len(set(pred_A)) > 1 else (float('nan'), None)

    print("\n──────── Results ────────")
    print(f"2D MAE        : {mae:.3f}")
    print(f"Valence r     : {r_v:.3f}")
    print(f"Arousal r     : {r_a:.3f}")
    print(f"Evaluated {len(test)} sentences in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()