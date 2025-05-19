"""
RoBERTa model fine-tuned on GoEmotions dataset for emotion classification.
This model will be used to project emotions to valence-arousal space.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

class RoBERTaModel(nn.Module):
    def __init__(self, num_labels):
        super(RoBERTaModel, self).__init__()
        self.transformer = AutoModel.from_pretrained("roberta-base")
        self.fc1 = nn.Linear(self.transformer.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return self.sigmoid(logits)


def train():
    # Load and preprocess GoEmotions dataset
    dataset = load_from_disk("data/processed/goemotions")
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "multi_hot_labels"])

    train_loader = DataLoader(
        dataset["train"], batch_size=16,
        shuffle=True)  # expiriment with different batch sizes

    val_loader = DataLoader(dataset["validation"],
                            batch_size=16,
                            shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoBERTaModel(num_labels=28).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * epochs)

    best_val_accuracy = 0.0  # Track best validation accuracy

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # === TRAINING ===
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["multi_hot_labels"].to(device).float()

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct_train += (preds == labels).all(dim=1).sum().item()
            total_train += labels.size(0)

            progress_bar.set_postfix(loss=total_loss / len(progress_bar))

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(
            f"Epoch {epoch + 1}: Training Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}"
        )

        # === VALIDATION ===
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["multi_hot_labels"].to(device).float()

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = (outputs > 0.5).float()
                correct_val += (preds == labels).all(dim=1).sum().item()
                total_val += labels.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}: Validation Loss = {val_loss:.4f}, Accuracy = {val_accuracy:.4f}"
        )

        # === SAVE BEST MODEL ===
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(),
                       "models/roberta_goemotions/best_model.pt")
            print(f"✅ Saved new best model (Accuracy: {val_accuracy:.4f})")

if __name__ == "__main__":
    train()