"""
Base GPT VA Classifier (§3.1.1)

Purpose: GPT-o4-mini zero-shot valence-arousal prediction baseline
         Tests if foundation models can inherently predict continuous affective signals.

Paper Section: §3.1.1 Base GPT
Performance: MAE 0.197, Valence Pearson 0.789, Arousal Pearson 0.429

This module implements a robust GPT-4 based classifier for valence-arousal (VA) scoring
of text inputs. It includes rate limiting, error handling, and parallel processing
capabilities for efficient batch processing.

Key Features:
- Valence scoring: -1 (very negative) to 1 (very positive)
- Arousal scoring: -1 (inactive/calm) to 1 (excited/active)
- Automatic retry with exponential backoff for rate limits
- Parallel processing for batch classification
- JSON-only response format for reliable parsing

Inputs: Raw text strings
Outputs: {"valence": float, "arousal": float} in [-1, 1] range

Usage:
    # Single text classification
    result = classify("I feel great today!")
    # Returns: {"valence": 0.8, "arousal": 0.6}
    
    # Batch classification
    texts = ["Happy text", "Sad text", "Neutral text"]
    results = classify_many(texts)

Related Files:
- va_classifier.prompts — System prompts for VA prediction
- va_classifier.evaluation.va_evaluation — Performance evaluation
- va_classifier.data_prep.emobank — Evaluation dataset
    
Author: Research Team
Date: 2024
"""
import json, re, os, threading, concurrent.futures as cf
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

#1- Get API key using Miko's API key. 
key_path = Path.home() / "Desktop" / "openai_key.txt" 
api_key = key_path.read_text().strip()           
client = OpenAI(api_key=api_key)

#2- Fixed system prompt.
SYSTEM_PROMPT = """
You are an affective-computing expert.
For each input, estimate:
- Valence: -1 (very negative) to 1 (very positive)
- Arousal: -1 (inactive) to 1 (excited)
For each input, return exactly:
{"valence": float, "arousal": float}
No extra words, no explanation—return only a JSON object.
""".strip()

#3- Tenacity backoff wrapper: https://cookbook.openai.com/examples/how_to_handle_rate_limits
"""
Rate limit: 500 RPM, 200,000 TPM, 2,000,000 TPD for o4-mini
"""
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    before_sleep=lambda retry_state: print(
        f"[VA-retry] attempt {retry_state.attempt_number} failed, backing off…"
    )
)
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)
_completion = completion_with_backoff

_MAX_PAR = int(os.getenv("GPT_VA_MAX_WORKERS", "10"))
_sema = threading.Semaphore(_MAX_PAR)

def _call_openai(text: str, model: str, temperature: float):
    with _sema:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ]
        for _ in range(3):
            resp = _completion(
                model=model,
                messages=msgs,
                temperature=temperature,
                timeout=30,
                response_format={"type": "json_object"}   # ← NEW
            )

            # 1) grab the raw content
            raw = resp.choices[0].message.content

            # 2) ensure it's a string before stripping
            if not isinstance(raw, str):
                raw = "" if raw is None else str(raw)

            content = raw.strip()

            try:
                d = json.loads(content)
                # make sure valence/arousal are numeric
                v = float(d.get("valence", None))
                a = float(d.get("arousal", None))
                return {"valence": round(v, 2), "arousal": round(a, 2)}
            except Exception:
                # if parsing failed, explicitly ask for JSON-only and retry
                msgs.append({"role": "system", "content": "JSON only."})
        raise RuntimeError("Failed to return valid JSON for: " + text[:60])


def classify(text: str, model: str = "o4-mini", temperature: float = 1.0):
    """
    Classify a single text string for valence and arousal scores.
    
    Args:
        text (str): Input text to classify
        model (str): OpenAI model to use (default: "o4-mini")
        temperature (float): Sampling temperature (default: 1.0)
        
    Returns:
        dict: Dictionary with 'valence' and 'arousal' keys, values in [-1, 1]
        
    Example:
        >>> classify("I'm feeling excited about this project!")
        {"valence": 0.75, "arousal": 0.82}
    """
    return _call_openai(text, model, temperature)

def classify_many(texts, *, model: str = "o4-mini", temperature: float = 1.0,
                  max_workers: int | None = None):
    """
    Parallel VA scoring for multiple texts using ThreadPoolExecutor.
    
    Args:
        texts: Iterable of text strings to classify
        model (str): OpenAI model to use (default: "o4-mini")
        temperature (float): Sampling temperature (default: 1.0)
        max_workers (int, optional): Number of parallel workers
        
    Returns:
        list: List of dictionaries with 'valence' and 'arousal' scores
        
    Example:
        >>> texts = ["Happy text", "Sad text"]
        >>> classify_many(texts)
        [{"valence": 0.8, "arousal": 0.5}, {"valence": -0.6, "arousal": -0.2}]
    """
    max_workers = max_workers or _MAX_PAR
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_call_openai, t, model, temperature) for t in texts]
        return [f.result() for f in futs]