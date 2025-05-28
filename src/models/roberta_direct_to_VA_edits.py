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
import argparse


# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

class VARegressor(nn.Module):
    def __init__(self, backbone_name="roberta-base", n_layers=3, hidden_fc1=256,
                  hidden_fc2=128, hidden_fc3=64, dropout=0.3, freeze=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        h = self.encoder.config.hidden_size
        self.n_layers = n_layers
        if self.n_layers == 2:
            self.fc1 = nn.Linear(h, hidden_fc1)  # One layer after RoBERTa
            self.fc2 = nn.Linear(hidden_fc1, 2)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.tanh = nn.Tanh()
        if self.n_layers == 3:
            self.fc1 = nn.Linear(h, hidden_fc1)  # 2 layers after RoBERTa
            self.fc2 = nn.Linear(hidden_fc1, hidden_fc2)  
            self.fc3 = nn.Linear(hidden_fc2, 2) 
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.tanh = nn.Tanh()
        if self.n_layers == 4:
            self.fc1 = nn.Linear(h, hidden_fc1)  # 3 layers after RoBERTa
            self.fc2 = nn.Linear(hidden_fc1, hidden_fc2)  
            self.fc3 = nn.Linear(hidden_fc2, hidden_fc3)
            self.fc4 = nn.Linear(hidden_fc3, 2)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.tanh = nn.Tanh()


        # Freeze the RoBERTa model parameters during initial training
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token

        
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        if self.n_layers == 2:
            return self.tanh(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)        
        if self.n_layers == 3:
            return self.tanh(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        return self.tanh(x)

# === MAE Metric ===
def mean_absolute_error(preds, labels):
    return torch.mean(torch.abs(preds - labels))

# === Training Script ===
def train(hidden_fc1=256,
          hidden_fc2=128,
          hidden_fc3=64,
          n_layers=3,
          batch_size=16, 
          initial_lr=3e-5,
          fine_tune_lr=1e-5,
          epochs=20,
          clip_norm=1.0,
          patience=3,
          mae_history=[],
          best_val_mae=float('inf'),
          frozen=True,
          gradual_unfreeze=False):

    # Load and preprocess EmoBank dataset
    print("Training model with following parameters: \n")

    print(f"n_layers = {n_layers}")
    print(f"hidden_fc1 = {hidden_fc1}")
    if n_layers >= 3:
        print(f"hidden_fc2 = {hidden_fc2}")
    if n_layers >= 4:
        print(f"hidden_fc3 = {hidden_fc3}")
    print(f"batch_size = {batch_size}")
    print(f"initial_lr = {initial_lr}")
    print(f"fine_tune_lr = {fine_tune_lr}")
    print(f"epochs = {epochs}")
    print(f"frozen = {frozen}")
    print(f"patience = {patience}")
    print(f"gradual_unfreezing = {gradual_unfreeze} \n")

    dataset = load_from_disk("data/processed/emobank")
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["val"], batch_size=batch_size, shuffle=False)
    
    model = VARegressor(n_layers=n_layers, hidden_fc1=hidden_fc1, hidden_fc2=hidden_fc2, hidden_fc3=hidden_fc3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # overall 2‑D MAE
        val_mae = mean_absolute_error(all_preds, all_labels)
        # component‑wise MAE
        valence_mae = torch.mean(torch.abs(all_preds[:, 0] - all_labels[:, 0]))
        arousal_mae = torch.mean(torch.abs(all_preds[:, 1] - all_labels[:, 1]))
        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val MAE (2D): {val_mae:.4f} | "
            f"Val MAE (valence): {valence_mae:.4f} | "
            f"Val MAE (arousal): {arousal_mae:.4f}"
        )

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), f"src/models/roberta_direct_emobank/{n_layers}_hidden_{hidden_fc1}_{hidden_fc2}_patience_{patience}_finetune_{fine_tune_lr}.pt")
            print("✅ Saved new best model.")
            

        mae_history.append(val_mae)

        # Check for plateau and unfreeze
        if frozen and len(mae_history) > patience:
            recent = mae_history[-(patience + 1):]
            plateaued = max(recent) - min(recent) < 0.01
            getting_worse = all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))
            if plateaued or getting_worse:
                print("🔓 MAE plateaued. Unfreezing encoder, reducing learning rate.")
                for p in model.encoder.parameters():
                    p.requires_grad = True
                optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=fine_tune_lr)
                frozen = False



if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--hidden_fc1",    type=int,   default=256)
        parser.add_argument("--hidden_fc2",    type=int,   default=128)
        parser.add_argument("--hidden_fc3",    type=int,   default=64)
        parser.add_argument("--n_layers",      type=int,   default=3)
        parser.add_argument("--batch_size",    type=int,   default=16)
        parser.add_argument("--initial_lr",    type=float, default=3e-5)
        parser.add_argument("--fine_tune_lr",  type=float, default=1e-5)
        parser.add_argument("--epochs",        type=int,   default=20)
        parser.add_argument("--clip_norm",     type=float, default=1.0)
        parser.add_argument("--patience",      type=int,   default=3)
        parser.add_argument("--frozen",        action="store_true")
        parser.add_argument("--gradual_unfreeze", action="store_true")
        args = parser.parse_args()

        train(
            hidden_fc1       = args.hidden_fc1,
            hidden_fc2       = args.hidden_fc2,
            hidden_fc3       = args.hidden_fc3,
            n_layers         = args.n_layers,
            batch_size       = args.batch_size,
            initial_lr       = args.initial_lr,
            fine_tune_lr     = args.fine_tune_lr,
            epochs           = args.epochs,
            clip_norm        = args.clip_norm,
            patience         = args.patience,
            frozen           = args.frozen,
            gradual_unfreeze = args.gradual_unfreeze,
        )