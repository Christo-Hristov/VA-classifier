# VA-Classifier: Valence-Arousal Classification Project

This project implements and compares different approaches for classifying text into valence-arousal space.

## Project Structure (UPDATE FULLY LATER)

```
VA-classifier/
├── data/                      # Data directory
│   ├── raw/                   # Raw datasets
│   │   ├── goemotions/        # GoEmotions dataset
│   │   └── emobank/          # EmoBank dataset
│   └── processed/             # Processed datasets
├── models/                    # Model implementations
│   ├── roberta_goemotions/    # RoBERTa model fine-tuned on GoEmotions
│   ├── roberta_emobank/      # RoBERTa model fine-tuned on EmoBank
│   └── gpt4_classifier/       # GPT-4 based classifier
├── src/                       # Source code
│   ├── data/                  # Data processing scripts
│   │   ├── download.py        # Dataset download scripts
│   │   └── preprocess.py      # Data preprocessing scripts
│   ├── models/                # Model training and evaluation
│   │   ├── roberta_goemotions.py
│   │   ├── roberta_emobank.py
│   │   └── GPT_classifier.py  # o4-mini default VA classifier
│   └── utils/                 # Utility functions
│       ├── metrics.py         # Evaluation metrics
│       └── visualization.py   # Visualization tools
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Unit tests
    └── evaluation_gpt.py      # VA evaluation for model/GPT_classifier.py
├── requirements.txt           # Project dependencies
└── config/                    # Configuration files
    └── model_configs.yaml     # Model configurations
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets:
```bash
python src/data/preprocess_goemotions.py
```

```bash
python src/data/preprocess_emobank.py
```

## Project Components

### 1. RoBERTa + GoEmotions
- Fine-tunes RoBERTa on GoEmotions dataset
- Projects emotion outputs to valence-arousal space
- Implementation in `models/roberta_goemotions/`

### 2. RoBERTa + EmoBank
- Uses fine-tuned RoBERTa with custom classifier head
- Trained on EmoBank dataset
- Implementation in `models/roberta_emobank/`

### 3. GPT-4 Classifier
- Implements GPT-4 based classification
- Includes prompt engineering and validation pipeline
- Implementation in `models/gpt4_classifier/`

## Dataset Information

### EmoBank Dataset
- Source: https://github.com/JULIELab/EmoBank
- Contains train/dev/test splits
- Valence-Arousal scores scaled from [0,5] to [-1,1]

### GoEmotions Dataset
- Used for initial RoBERTa fine-tuning
- Provides emotion labels for projection to VA space

### EDAIC-WOZ
- https://dcapswoz.ict.usc.edu/wwwedaic/

## License

[Add appropriate license information]
