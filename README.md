# Text and Valence--Arousal: A Foundational Approach for Mental Health Prediction

This repository accompanies the Stanford CS277 / BIODS 271 research
project:\
**"Text and Valence--Arousal: A Two-Dimensional Foundational Approach
for Mental Health Prediction"** ([final paper in
`/docs`](./docs/CS_277_Final.pdf)).

We introduce a unified framework that learns **Valence--Arousal (VA)**
from text and uses these signals to improve downstream prediction of
**depression (PHQ-8)** and **PTSD (PCL-5)**.

------------------------------------------------------------------------

## 🔑 Key Contributions

-   **VA Regressor (Part A, §3.1)**
    -   GPT-4o-mini baseline (zero-shot)\
    -   RoBERTa Direct VA regression (best performance: MAE 0.0759)\
    -   RoBERTa + GoEmotions emotion context
-   **PHQ-8 Depression Prediction (Part B, §3.2)**
    -   Zero-shot GPT\
    -   AutoCoT (few-shot chain-of-thought)\
    -   Supervised Fine-Tuning (text-only vs VA-enhanced, 12.2% MAE
        reduction with VA)
-   **PTSD Prediction (Part C, §3.3)**
    -   Benchmarks with zero-shot, AutoCoT, and SFT using VA-augmented
        transcripts\
    -   **Note:** Implementation scripts are missing here, as PTSD
        experiments were run in shared Google Colab notebooks (see paper
        for full details).
-   **Cross-Cutting Experiments (§4)**
    -   Prompt-length sensitivity (summarization, pruning)\
    -   VA trajectory visualizations across demographics

------------------------------------------------------------------------

## 📂 Repository Structure

    src/va_classifier/
      va_regressor/      # Part A: VA models (GPT, RoBERTa, GoEmotions)
      phq8/              # Part B: PHQ-8 experiments (zero-shot, AutoCoT, SFT)
      ptsd/              # Part C: PTSD experiments (partial; Colab-based code missing)
      data/              # Preprocessing for EmoBank, GoEmotions, E-DAIC
      visualizations/    # VA trajectory & experimental plots
    results/             # Processed outputs (redacted, see note below)
    figures/             # High-level performance plots (MAE/RMSE comparisons)
    docs/                # Final report & presentation

------------------------------------------------------------------------

## ⚠️ Important Notes

-   **PTSD Code:** PTSD implementations were conducted in Google Colab
    notebooks and are **not included** here. See [final
    paper](./docs/CS_277_Final.pdf) for methodology and results.\
-   **Results Files:** Some experimental outputs are omitted because
    they contain **sensitive transcript-derived data** (E-DAIC). Only
    high-level aggregate plots are included.\
-   **Datasets:** EmoBank, GoEmotions, and E-DAIC must be obtained
    separately. Licensing and ethics restrictions prevent redistribution
    here.

------------------------------------------------------------------------

## 📑 How to Navigate

-   Interested in **core ML modeling?** → see
    `src/va_classifier/va_regressor/`\
-   Want to see **depression prediction experiments?** → see
    `src/va_classifier/phq8/`\
-   Curious about **visualizations of VA dynamics?** → see
    `src/va_classifier/visualizations/`\
-   For full methodology, results, and context → read the [final
    paper](./docs/CS_277_Final.pdf).

------------------------------------------------------------------------

## 👥 Contributions

-   Christo Hristov --- Developed VA regression models (GoEmotions vs
    EmoBank pre-training)\
-   Kevina Wang --- Prompt engineering, transcript length/content
    sensitivity studies\
-   Miko Rimer --- Psychiatric framing, PHQ-8 prediction experiments\
-   Yalcin Tur --- VA trajectory visualizations\
-   Ion Martinis --- PTSD benchmark implementation
