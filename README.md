# VA-classifier
# Valence-Arousal Classifier in Determining PHQ-8 Scores

## Project Overview

This project aims to determine the most effective approach for classifying text into valence-arousal space, which can later be used to estimate PHQ-8 depression scores from language. The project is broken into three major modeling strategies using different approaches to valence-arousal (VA) estimation, followed by a shared dataset pipeline.

---

## Part 1: Determining Best Valence-Arousal Classifier

### 1. RoBERTa Model Fine-Tuned on GoEmotions Dataset

- **Objective**: Fine-tune a pre-trained RoBERTa model on the GoEmotions dataset and project its outputs to valence-arousal space.
- **Tasks**:
  - Use the model from the original GitHub repo: [Mental Health Chatbot using RoBERTa](https://github.com/kashyaparun25/Mental-Health-Chatbot-using-RoBERTa-and-Gemini/blob/main/README.md)
  - Fine-tune RoBERTa on the GoEmotions dataset.
  - Create a projection function to map emotion outputs to valence-arousal space.

---

### 2. RoBERTa Model with Custom Classifier Head Fine-Tuned on EmoBank

- **Objective**: Adapt a fine-tuned RoBERTa model for direct valence-arousal regression using EmoBank.
- **Tasks**:
  - Remove the original classifier head from RoBERTa.
  - Design and add a custom regression head to predict continuous valence and arousal values.
  - Define a loss function suitable for VA regression (e.g., MSE or CCC).
  - Fine-tune the model using the EmoBank train split.

---

### 3. Prompted ChatGPT-4 for Zero-Shot VA Regression

- **Objective**: Use ChatGPT-4 with crafted prompts to predict valence-arousal scores directly from text.
- **Tasks**:
  - Engineer effective prompts to maximize ChatGPT-4's predictive accuracy in the valence-arousal space.
  - Develop a pipeline for automatic prompting, prediction, and performance validation.
  - Ensure model outputs only valence-arousal scores to allow evaluation against ground truth.

---

## Dataset Tasks

- **Source**: [EmoBank GitHub Repository](https://github.com/JULIELab/EmoBank/tree/master)
- **Structure**:
  - Dataset is split into train, dev, and test sets.
- **Preprocessing**:
  - Convert valence-arousal scores from `[0, 5]` to `[-1, 1]`.
  - Load and preprocess EmoBank data into the repository for use across all models.

---

## Future Work

After evaluating the best-performing valence-arousal predictor, we plan to investigate its usefulness in downstream clinical applications such as predicting PHQ-8
