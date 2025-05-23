# This script preprocesses the GoEmotions dataset by creating multi-hot labels,
# tokenizing the text, and saving the processed dataset for further use in machine learning models.

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import os

NUM_CLASSES = 28  # 27 emotions + neutral

# Function to create a multi-hot encoded label for a given list of labels.
def create_multi_hot_label(labels):
    multi_hot = [0] * NUM_CLASSES
    for label in labels:
        multi_hot[label] = 1
    return multi_hot

# Function to preprocess the GoEmotions dataset by loading, creating multi-hot labels,
# tokenizing, and saving the dataset.
def preprocess_goemotions():
    # Load all dataset splits
    dataset = DatasetDict({
        "train": load_dataset("go_emotions", "simplified", split="train"),
        "validation": load_dataset("go_emotions", "simplified", split="validation"),
        "test": load_dataset("go_emotions", "simplified", split="test")
    })

    # Sanity check: Print dataset sizes
    print("\nDataset Summary:")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size: {len(dataset['test'])}")

    # Print a sample from the train set
    print("\nSample from train set:")
    print(dataset['train'][0])

    # Add multi-hot labels to each split
    dataset = dataset.map(lambda x: {"multi_hot_labels": create_multi_hot_label(x["labels"])})

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    # Tokenize all splits
    dataset = dataset.map(tokenize_function, batched=True)

    # Save the processed dataset
    output_dir = "data/processed/goemotions"
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"✅ Full GoEmotions DatasetDict saved to {output_dir}")

if __name__ == "__main__":
    preprocess_goemotions()
