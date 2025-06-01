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