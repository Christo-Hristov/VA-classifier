# Load model directly
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import numpy as np

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.roberta_goemotions_emobank import VARegressor, RoBERTaModel

def load_model():
    # First load the base emotion model
    base_model = RoBERTaModel(num_labels=28)
    # Create VA regressor
    model = VARegressor(base_model, freeze=False)
    # Load the best VA model state dict
    model.load_state_dict(torch.load("models/roberta_emobank/best_va_model.pt", map_location=torch.device('cpu')))
    return model

def evaluate_model(model, test=False):
    # Load processed EmoBank dataset
    dataset = load_from_disk("data/processed/emobank")
    # Set the format for the dataset
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # Create validation/test dataloader
    val_loader = DataLoader(
        dataset["validation" if not test else "test"],
        batch_size=16,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Initialize metrics for VA regression
    all_preds = []
    all_labels = []

    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float()  # VA values [valence, arousal]

            # Get model predictions
            preds = model(input_ids=input_ids, attention_mask=attention_mask)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics for VA regression
    mae = torch.mean(torch.abs(all_preds - all_labels)).item()
    mse = torch.mean((all_preds - all_labels) ** 2).item()
    rmse = np.sqrt(mse)
    
    # Calculate separate metrics for valence and arousal
    valence_mae = torch.mean(torch.abs(all_preds[:, 0] - all_labels[:, 0])).item()
    arousal_mae = torch.mean(torch.abs(all_preds[:, 1] - all_labels[:, 1])).item()
    
    # Print results
    print(f"\n{'Test' if test else 'Validation'} Results:")
    print(f"Overall MAE: {mae:.4f}")
    print(f"Overall MSE: {mse:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Valence MAE: {valence_mae:.4f}")
    print(f"Arousal MAE: {arousal_mae:.4f}")
    
    # Print sample predictions
    print("\nSample predictions (first 5):")
    for i in range(min(5, len(all_preds))):
        print(f"Pred: [V={all_preds[i, 0]:.2f}, A={all_preds[i, 1]:.2f}] | True: [V={all_labels[i, 0]:.2f}, A={all_labels[i, 1]:.2f}]")


if __name__ == "__main__":
    # take arguments if validation or test
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    if args and args[0] == "test":
        print("Running on test set...")
        model = load_model()
        evaluate_model(model, test=True)
    elif args and args[0] == "validate":
        print("Running on validation set...")
        model = load_model()
        evaluate_model(model, test=False)
    else:
        print("Please specify 'test' or 'validate' as an argument.")