# Load model directly
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
from scipy.stats import pearsonr

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.roberta_goemotions_emobank import VARegressorGoEmotions, RoBERTaModel
from src.models.roberta_direct_to_VA_edits import VARegressor

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(model_name="roberta_goemotions_emobank"):
    # Get the project root directory
    # Load the appropriate model based on the model name
    if model_name == "roberta_goemotions_emobank":
        base_model = RoBERTaModel(num_labels=28)
        model = VARegressorGoEmotions(base_model, freeze=False)
        model_path = os.path.join(project_root, "models/roberta_goemotions_emobank/best_va_model.pt")
    elif model_name == "roberta_emobank":
        model = VARegressor()       
        model_path = os.path.join(project_root, "models/roberta_emobank/best_model.pt")
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are: roberta_goemotions_emobank, roberta_emobank")
    
    # Load the model state dict
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

def evaluate_model(model, test=False):
    # Load processed EmoBank dataset
    dataset = load_from_disk(os.path.join(project_root, "data/processed/emobank"))
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
    
    # Convert to numpy for scipy stats
    preds_np = all_preds.numpy()
    labels_np = all_labels.numpy()
    
    # Calculate overall MAE
    mae = torch.mean(torch.abs(all_preds - all_labels)).item()
    
    # Calculate Pearson correlation for each dimension
    valence_pearson = pearsonr(preds_np[:, 0], labels_np[:, 0])[0]
    arousal_pearson = pearsonr(preds_np[:, 1], labels_np[:, 1])[0]
    
    # Print results
    print(f"\n{'Test' if test else 'Validation'} Results:")
    print(f"Overall MAE: {mae:.4f}")
    print(f"Valence Pearson: {valence_pearson:.4f}")
    print(f"Arousal Pearson: {arousal_pearson:.4f}")
    
    # Print sample predictions
    print("\nSample predictions (first 5):")
    for i in range(min(5, len(all_preds))):
        print(f"Pred: [V={all_preds[i, 0]:.2f}, A={all_preds[i, 1]:.2f}] | True: [V={all_labels[i, 0]:.2f}, A={all_labels[i, 1]:.2f}]")


if __name__ == "__main__":
    # take arguments for validation/test and model name
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    if len(args) < 2:
        print("Please provide both the evaluation mode ('test' or 'validate') and the model name.")
        print("Example: python evaluate_roberta_emobank.py test roberta_goemotions_emobank")
        sys.exit(1)
        
    eval_mode = args[0]
    model_name = args[1]
    
    if eval_mode == "test":
        print(f"Running on test set with model: {model_name}...")
        model = load_model(model_name)
        evaluate_model(model, test=True)
    elif eval_mode == "validate":
        print(f"Running on validation set with model: {model_name}...")
        model = load_model(model_name)
        evaluate_model(model, test=False)
    else:
        print("Please specify 'test' or 'validate' as the first argument.")
        print("Example: python evaluate_roberta_emobank.py test roberta_goemotions_emobank")