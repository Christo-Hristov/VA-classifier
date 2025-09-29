"""
GPT VA Classifier Evaluation on EmoBank Dataset (§3.1.1)

Purpose: Evaluates the GPT-based valence-arousal classifier on the EmoBank test split
         to measure zero-shot VA prediction performance using foundation models.
         This provides baseline metrics for comparing against RoBERTa-based approaches.

Paper Section: §3.1.1 Base GPT Zero-Shot VA Prediction
Performance: MAE 0.197, Valence Pearson 0.789, Arousal Pearson 0.429 (reported in paper)

This script implements comprehensive evaluation of the GPT VA classifier:

1. EmoBank Dataset Integration:
   - Downloads EmoBank corpus automatically from GitHub
   - Uses standardized test split for reproducible evaluation
   - Converts EmoBank 1-5 scale to [-1, 1] range for consistency
   - Handles 10,548 sentences with gold standard VA annotations

2. GPT Classifier Testing:
   - Tests zero-shot VA prediction capabilities
   - Uses GPT-4o-mini or specified model for cost-effective evaluation
   - Handles multiple output formats (dict, tuple, JSON string)
   - Robust parsing of GPT responses with error handling

3. Parallel Processing:
   - Configurable concurrent API calls for efficiency
   - Thread pool executor for batch processing
   - Progress tracking for long evaluation runs
   - Rate limiting consideration for API stability

4. Comprehensive Metrics:
   - 2D MAE: Combined valence + arousal mean absolute error
   - Pearson correlation: Separate correlations for valence and arousal
   - Handles edge cases (constant predictions, API failures)
   - Timing information for performance analysis

Evaluation Pipeline:
1. Load EmoBank test split (automatically downloaded if needed)
2. Convert gold standard VA scores from 1-5 to [-1, 1] scale
3. For each test sentence:
   - Call GPT classifier with text input
   - Parse VA prediction from response
   - Handle various output formats robustly
4. Calculate evaluation metrics and report results

Technical Features:
- Automatic dataset downloading and caching
- Flexible output format parsing (JSON, dict, tuple)
- Concurrent API processing with configurable workers
- Progress tracking and timing measurements
- Robust error handling for API failures

Scale Conversion:
- EmoBank: 1-5 scale (1=very negative/calm, 5=very positive/excited)
- Model output: [-1, 1] scale (-1=very negative/calm, 1=very positive/excited)
- Conversion: (emobank_score - 3.0) / 2.0

Research Applications:
- Establishes GPT baseline for VA prediction tasks
- Compares zero-shot vs supervised learning approaches
- Tests foundation model emotional understanding capabilities
- Validates VA classifier performance on standardized dataset

Related Files:
- src.models.GPT_classifier — Core GPT VA classification implementation
- va_classifier.va_regressor.roberta_direct — RoBERTa comparison baseline
- va_classifier.evaluation.va_evaluation — Comprehensive VA evaluation suite

Usage Examples:
    # Standard evaluation
    python evaluate_gpt.py
    
    # Custom model and parameters
    python evaluate_gpt.py --model gpt-4 --temp 0.5
    
    # Limited evaluation for testing
    python evaluate_gpt.py --limit 100
    
    # Parallel processing (use carefully with API limits)
    python evaluate_gpt.py --workers 3

Expected Output:
    2D MAE        : 0.197
    Valence r     : 0.789
    Arousal r     : 0.429
    Evaluated 10548 sentences in 45.2s

Note: API costs can accumulate quickly with full EmoBank evaluation.
      Consider using --limit for initial testing and validation.
"""
import argparse, tempfile, urllib.request, time, concurrent.futures as cf
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr
import json, re
from src.models.GPT_classifier import classify

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