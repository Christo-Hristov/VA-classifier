# This script preprocesses the EmoBank dataset by cleaning text, converting scores, splitting the data,
# tokenizing the text, and saving the processed dataset for further use in machine learning models.

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import pandas as pd
import os

# Function to clean text by removing non-string values and stripping quotation marks
# from the beginning and end of the text.
def clean_text(text):
    # Handle non-string values
    if not isinstance(text, str):
        return ""
    
    # Remove quotation marks from beginning and end if present
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text

# Function to preprocess the EmoBank dataset by loading, cleaning, splitting, tokenizing,
# and saving the dataset.
def preprocess_emobank():
    # Load the raw dataset
    df = pd.read_csv("data/raw/emobank/emobank.csv")
    
    # Sanity check: Print min and max values for V, A, and D columns
    for d in ['V', 'A', 'D']:
        print(f"Min {d}: {df.loc[df[d].idxmin()]}")
        print()
        print(f"Max {d}: {df.loc[df[d].idxmax()]}")
        print()
        print()

    # Convert V, A scores from [1,5] to [-1,1] range
    df['V'] = (df['V'] - 3) / 2
    df['A'] = (df['A'] - 3) / 2
    
    # Clean text by removing quotation marks
    df['text'] = df['text'].apply(clean_text)
    
    # Create train/test splits based on the 'split' column
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    val_df = df[df['split'] == 'dev']
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    val_dataset = Dataset.from_pandas(val_df)
    # Create DatasetDict
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "val": val_dataset
    })
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
    
    # Tokenize all splits
    dataset = dataset.map(tokenize_function, batched=True)

    dataset = dataset.map(lambda x: {
    "label": [x["V"], x["A"]]
    })

    
    # Remove the D column from all splits
    dataset = dataset.remove_columns(['D'])
    dataset = dataset.remove_columns(["V", "A"])
    dataset = dataset.remove_columns(["split"])
    dataset = dataset.remove_columns(["id"])
    
    # Save the processed dataset
    output_dir = "data/processed/emobank"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print dataset info for sanity check
    print("\nDataset Summary:")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")
    print(f"Val size: {len(dataset['val'])}")
    print("\nSample from train set:")
    print(dataset['train'][0])
    
    dataset.save_to_disk(output_dir)
    print(f"✅ Full EmoBank DatasetDict saved to {output_dir}")

if __name__ == "__main__":
    preprocess_emobank()
