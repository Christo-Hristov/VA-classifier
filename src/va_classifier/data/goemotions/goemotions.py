"""
preprocess_goemotions.py - GoEmotions Dataset Preprocessing
==========================================================

This script preprocesses the GoEmotions dataset for emotion classification training.
GoEmotions contains Reddit comments labeled with 28 emotion categories.

Key preprocessing steps:
1. Load train/validation/test splits from HuggingFace datasets
2. Convert emotion labels to multi-hot encoding (28-dimensional binary vectors)
3. Tokenize text using RoBERTa tokenizer with padding and truncation
4. Save processed dataset to disk for training

Dataset: https://huggingface.co/datasets/go_emotions
Paper: "GoEmotions: A Dataset of Fine-Grained Emotions" (Demszky et al., 2020)

Usage:
    python preprocess_goemotions.py
"""

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import os

NUM_CLASSES = 28  # 27 emotions + neutral

def create_multi_hot_label(labels):
    """
    Convert list of emotion label indices to multi-hot encoded vector.
    
    Args:
        labels (list): List of emotion indices (0-27)
        
    Returns:
        list: Binary vector of length 28 with 1s at label positions
    """
    multi_hot = [0] * NUM_CLASSES
    for label in labels:
        multi_hot[label] = 1
    return multi_hot

def preprocess_goemotions():
    """
    Main preprocessing function for GoEmotions dataset.
    
    Loads the dataset, applies multi-hot encoding, tokenizes text,
    and saves the processed dataset to data/processed/goemotions/
    """
    # Load all dataset splits
    dataset = DatasetDict({
        "train": load_dataset("go_emotions", "simplified", split="train"),
        "validation": load_dataset("go_emotions", "simplified", split="validation"),
        "test": load_dataset("go_emotions", "simplified", split="test")
    })

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
