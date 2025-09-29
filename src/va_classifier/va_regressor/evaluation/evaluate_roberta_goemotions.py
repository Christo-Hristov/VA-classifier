"""
RoBERTa GoEmotions Emotion Classification Evaluation (§3.1.3)

Purpose: Evaluates the RoBERTa model trained on GoEmotions dataset for multi-label
         emotion classification. This model serves as the base component for the
         hybrid VA regressor that combines emotion features with VA prediction.

Paper Section: §3.1.3 RoBERTa with GoEmotions Context (emotion classification component)
Task: Multi-label emotion classification on 28 GoEmotions emotion categories

This script evaluates the emotion classification foundation for the hybrid VA approach:

1. GoEmotions Emotion Classification:
   - 28 emotion categories from GoEmotions dataset
   - Multi-label classification (sentences can have multiple emotions)
   - Binary classification threshold at 0.5 for each emotion
   - Exact match accuracy (all emotion labels must be correct)

2. Hybrid VA Context:
   - This model provides emotion features for VA regression
   - Trained emotion representations transferred to VA prediction
   - Tests whether emotion classification features improve VA accuracy
   - Foundation component for the two-stage hybrid approach

3. Evaluation Methodology:
   - Validation set: Development and hyperparameter tuning
   - Test set: Final performance assessment for transfer learning
   - Exact match accuracy: Strict evaluation requiring all labels correct
   - Batch processing: Efficient GPU-accelerated evaluation

4. Technical Implementation:
   - PyTorch model with multi-label sigmoid output
   - Threshold-based prediction (>0.5 = positive emotion)
   - GPU/CPU compatible with automatic device detection
   - Progress tracking for evaluation monitoring

Model Architecture:
- Base: RoBERTa-base (768-dimensional embeddings)
- Classification Head: Linear layer (768 → 28 emotions)
- Output: Sigmoid activation for multi-label prediction
- Training: Multi-label binary cross-entropy loss

GoEmotions Dataset Context:
- 28 emotion categories: admiration, amusement, anger, annoyance, approval, etc.
- Multi-label: Text can express multiple emotions simultaneously
- Large-scale: ~58k Reddit comments with emotion annotations
- Preprocessing: Tokenized with RoBERTa tokenizer, multi-hot encoded labels

Evaluation Pipeline:
1. Load trained GoEmotions RoBERTa model
2. Load processed GoEmotions dataset (validation or test split)
3. Configure evaluation mode and batch processing
4. For each batch:
   - Forward pass through emotion classification model
   - Apply 0.5 threshold to get binary predictions
   - Calculate exact match accuracy (all 28 labels correct)
5. Report final accuracy and sample statistics

Research Context:
- This emotion model feeds into the hybrid VA regressor (§3.1.3)
- Tests hypothesis: emotion classification features → better VA prediction
- Compares against direct VA regression (§3.1.2) without emotion context
- Establishes emotion classification baseline for transfer learning

Key Features:
- Multi-label emotion classification evaluation
- Exact match accuracy (strict evaluation metric)
- GPU acceleration with CPU fallback
- Validation and test set evaluation modes
- Progress monitoring for long evaluations

Usage Examples:
    # Validate emotion classification performance
    python evaluate_roberta_goemotions.py validate
    
    # Test final emotion classification accuracy
    python evaluate_roberta_goemotions.py test

Expected Output Format:
    Validation Accuracy: 0.4567
    Total Correct: 1234, Total Samples: 2700
    
Note: Accuracy appears low due to strict exact match requirement.
      Partial matches (some emotions correct) are not counted.

Research Applications:
- Validates emotion classification component of hybrid VA approach
- Tests transfer learning foundation for VA prediction enhancement
- Establishes emotion feature quality for downstream VA tasks
- Supports analysis of emotion context impact on VA regression

Related Files:
- src.models.roberta_goemotions — GoEmotions classification model
- va_classifier.va_regressor.roberta_goemotions.hybrid_model — VA hybrid approach
- tests.completed_VA_tests.evaluate_roberta_emobank — VA evaluation comparison
- data/processed/goemotions — Preprocessed emotion dataset

Note: This evaluation focuses on emotion classification accuracy. The hybrid VA
      regressor evaluation (which uses these emotion features) is in evaluate_roberta_emobank.py.
"""

# Load model directly
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.roberta_goemotions import RoBERTaModel

def load_model():
    model = RoBERTaModel(num_labels=28)
    model.load_state_dict(torch.load("models/roberta_goemotions/best_model.pt", map_location=torch.device('cpu')))
    return model

def evaluate_model(model, test=False):
    # Load processed GoEmotions dataset
    dataset = load_from_disk("data/processed/goemotions")
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "multi_hot_labels"]
    )

    # Create validation dataloader
    val_loader = DataLoader(
        dataset["validation"] if not test else dataset["test"],
        batch_size=16,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Initialize metrics
    correct_val = 0
    total_val = 0

    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["multi_hot_labels"].to(device).float()

            # Get model predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (outputs > 0.5).float()

            # Calculate accuracy
            correct = (preds == labels).all(dim=1).sum().item()
            correct_val += correct
            total_val += labels.size(0)

    # Calculate final metrics
    accuracy = correct_val / total_val
    print(f"Total Correct: {correct_val}, Total Samples: {total_val}")
    print(f"{'Test' if test else 'Validation'} Accuracy: {accuracy:.4f}")


if __name__ == "__main__":

    # take arguments if validation or test
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    if args and args[0] == "test":
        print("Running on test set...")
        model = load_model()
        evaluate_model(model)
    elif args and args[0] == "validate":
        print("Running on validation set...")
        model = load_model()
        evaluate_model(model)
    else:
        print("Please specify 'test' or 'validate' as an argument.")

