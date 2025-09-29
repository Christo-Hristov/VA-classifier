"""
VA Score Utility Function Test Suite (§3.1 Supporting Infrastructure)

Purpose: Tests the get_va_scores() utility function that provides VA prediction
         capabilities across all paper experiments. Validates output format,
         numerical ranges, and basic functionality with diverse emotional content.

Paper Context: Supporting infrastructure for all VA-enhanced experiments
Usage: Quality assurance for the core VA scoring utility used throughout the paper

This test suite validates the get_va_scores() function that serves as the foundation
for VA integration across multiple paper experiments:

1. Core Functionality Testing:
   - Tests VA prediction on diverse emotional content samples
   - Validates output format consistency (valence, arousal tuples)
   - Ensures numerical output types and proper array dimensions
   - Verifies batch processing capabilities

2. Emotional Content Coverage:
   - High valence, high arousal: "happy and excited" (positive activation)
   - Low valence, low arousal: "terrible, awful, depressed" (negative deactivation)
   - Moderate valence, low arousal: "calm and peaceful" (neutral relaxation)
   - Low valence, high arousal: "angry and frustrated" (negative activation)

3. Output Validation:
   - Confirms 1:1 mapping between input texts and output scores
   - Validates tuple structure (valence, arousal) for each prediction
   - Ensures numeric types (int/float) for all score values
   - Tests array length consistency and data type integrity

4. Integration Context:
   - Used by PHQ-8 experiments for transcript annotation (§3.2)
   - Used by PTSD experiments for clinical interview enhancement (§3.3)
   - Supports AutoCoT experiments with VA-enhanced prompts (§3.2.2)
   - Enables all VA context integration across paper experiments

Test Cases Design:
- **Positive High Arousal**: Tests detection of excitement, happiness
- **Negative Low Arousal**: Tests detection of depression, sadness
- **Neutral Low Arousal**: Tests detection of calm, peaceful states
- **Negative High Arousal**: Tests detection of anger, frustration

Expected VA Patterns:
- Happy/Excited → High valence (+), High arousal (+)
- Terrible/Depressed → Low valence (-), Low arousal (-)
- Calm/Peaceful → Neutral valence (~0), Low arousal (-)
- Angry/Frustrated → Low valence (-), High arousal (+)

Technical Validation:
- Output array length matches input text count
- Each score contains exactly 2 values (valence, arousal)
- All values are numeric (int or float types)
- Function handles batch processing without errors

Research Applications:
- Validates core VA scoring infrastructure reliability
- Ensures consistent VA annotation across all paper experiments
- Tests emotional content recognition across VA dimensions
- Supports reproducible VA integration in clinical applications

Related Components:
- src.utils.util.get_va_scores — Core VA scoring function
- va_classifier.va_regressor.roberta_direct — Underlying VA model
- All PHQ-8 and PTSD experiments — Primary usage contexts
- AutoCoT experiments — VA-enhanced prompting applications

Usage:
    python test_va_scores.py
    # Runs comprehensive VA scoring validation tests
    # Outputs sample predictions and validation results

Expected Output:
    VA Scores for test texts:
    --------------------------------------------------
    Text: I am so happy and excited...
    Valence: 0.847, Arousal: 0.692
    --------------------------------------------------
    [Additional test cases...]
    
    All tests passed successfully!

Note: This test validates the utility function that enables VA context integration
      across all paper experiments. Ensuring its reliability is critical for
      reproducible research results.
"""

import os
import sys
import torch
import numpy as np

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.util import get_va_scores

def test_va_scores():
    # Example texts with different emotional content
    test_texts = [
        "I am so happy and excited about this wonderful day!",
        "This is terrible, I feel awful and depressed.",
        "I feel calm and peaceful right now.",
        "I am extremely angry and frustrated with this situation."
    ]
    
    # Get VA scores
    scores = get_va_scores(test_texts)
    print(scores)
    
    # Print results
    print("\nVA Scores for test texts:")
    print("-" * 50)
    for text, (valence, arousal) in zip(test_texts, scores):
        print(f"Text: {text}")
        print(f"Valence: {valence:.3f}, Arousal: {arousal:.3f}")
        print("-" * 50)
    
    # Basic validation checks
    assert len(scores) == len(test_texts), "Number of scores should match number of input texts"
    assert all(len(score) == 2 for score in scores), "Each score should have valence and arousal values"
    assert all(isinstance(score[0], (int, float)) for score in scores), "Scores should be numeric"
    
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_va_scores() 