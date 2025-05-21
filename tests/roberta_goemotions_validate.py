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
    model.load_state_dict(torch.load("models/roberta_goemotions/best_model_goemotions.pt", map_location=torch.device('cpu')))
    return model

def evaluate_model(model):
    # Load processed GoEmotions dataset
    dataset = load_from_disk("data/processed/goemotions")
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "multi_hot_labels"]
    )

    # Create validation dataloader
    val_loader = DataLoader(
        dataset["validation"],
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
    print(f"Validation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    model = load_model()
    evaluate_model(model)

