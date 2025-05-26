import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from copy import deepcopy
from tqdm import tqdm
from datasets import load_from_disk
import sys
import os

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.models.roberta_goemotions import RoBERTaModel

class VARegressor(nn.Module):
    def __init__(self, pretrained_model: nn.Module, freeze: bool = True):
        super(VARegressor, self).__init__()

        # Copy components from the pretrained RoBERTa emotion model
        self.transformer = pretrained_model.transformer
        self.fc1 = pretrained_model.fc1

        # Freeze the RoBERTa model parameters during initial training
        if freeze:
            for param in self.fc1.parameters():
                param.requires_grad = False
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # New head for VA regression
        self.fc2 = nn.Linear(256, 128)  # New intermediate layer
        self.fc3 = nn.Linear(128, 2)  
        self.tanh = nn.Tanh()  # Final output: [valence, arousal]

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token

        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        out = self.fc3(x)         
        out = self.tanh(out)
        return out

# === MAE Metric ===
def mean_absolute_error(preds, labels):
    return torch.mean(torch.abs(preds - labels))

# === Training Script ===
def train_va_regressor(model, train_dataset, val_dataset, device):
    batch_size = 16
    initial_lr = 3e-5
    fine_tune_lr = 1e-5
    epochs = 20
    clip_norm = 1.0
    patience = 3  # Epochs to wait before unfreezing
    mae_history = []
    best_val_mae = float('inf')  # Initialize with infinity since we want to minimize MAE
    frozen = True

    # Load and preprocess EmoBank dataset
    dataset = load_from_disk("data/processed/emobank")
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float()

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, labels)
            loss.backward()

            if not frozen:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # === Validation ===
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device).float()

                preds = model(input_ids, attention_mask)
                loss = criterion(preds, labels)
                total_val_loss += loss.item()

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        val_mae = mean_absolute_error(all_preds, all_labels)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae:.4f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "models/roberta_emobank/best_va_model.pt")
            print("✅ Saved new best model.")

        mae_history.append(val_mae)

        # Check for plateau and unfreeze
        if frozen and len(mae_history) > patience:
            recent = mae_history[-(patience + 1):]
            plateaued = max(recent) - min(recent) < 0.01
            getting_worse = all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))
            if plateaued or getting_worse:
                print("🔓 MAE plateaued. Unfreezing encoder and fc1, reducing learning rate.")
                for p in model.transformer.parameters():
                    p.requires_grad = True
                for p in model.fc1.parameters():
                    p.requires_grad = True
                optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=fine_tune_lr)
                frozen = False


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create and load the emotion model
    emotion_model = RoBERTaModel(num_labels=28)
    
    continue_training = False

    if continue_training:
        # Load the pretrained model weights
        state_dict = torch.load("models/roberta_goemotions/best_model.pt", 
                                map_location=device)
        emotion_model.load_state_dict(state_dict)
    
    # Create and train the VA regressor
    va_model = VARegressor(emotion_model)
    train_va_regressor(va_model, "data/processed/emobank", "data/processed/emobank", device)

