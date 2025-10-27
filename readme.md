# CardioRisk — Early Heart Disease Risk Prediction

> A lightweight, recall-focused AI model for early heart disease screening built with PyTorch.

---

## Overview

This project predicts the risk of **heart disease or heart attack** using simple survey-style health indicators (like smoking, activity level, blood pressure, and age).  

It’s designed with **real-world clinical deployment** in mind:
- Focuses on **recall/sensitivity** — catching as many at-risk patients as possible.
- Adds **interpretability** via an **attention mechanism**.
- Handles **class imbalance** using **focal loss**.
- Evaluated on **clinically meaningful metrics**, not just accuracy.

---

## Dataset

- **Source:** [CDC BRFSS 2015 Heart Disease Health Indicators](https://www.cdc.gov/brfss/)
- **Samples:** ~253K  
- **Features:** 21 tabular health indicators  
- **Target:** `HeartDiseaseorAttack` (binary 0/1)  
- **Class Imbalance:** ~9% positives

---

## Models

### 1. Baseline MLP
A simple fully-connected neural net (ReLU + Dropout + BatchNorm) trained with cross-entropy loss.

### 2. ClinicalAttentionNet
Improved model featuring:
- **Attention Block** — learns per-feature importance for interpretability  
- **Focal Loss** — focuses on hard examples and minority class  
- **Cosine Annealing LR Schedule** — improves generalization  
- **AdamW Optimizer** — stable training with weight decay  

---

## Metrics

- **Recall / Sensitivity** — priority metric for clinical screening  
- **ROC-AUC & F1 Score** — overall model quality  
- **Attention Weights** — interpret which features drive predictions  

---

## How to Run

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
jupyter notebook cardio.ipynb

---

## Dependencies

torch
pandas
numpy
matplotlib
scikit-learn

---

## Results
- The attention model identifies more true positive cases than the baseline.
- Improves recall while maintaining interpretability.
- Top contributing features: General Health, Age, Difficulty Walking, High BP, Physical Activity.

---

## Disclaimer

- This project is for research and educational use only.
- It is not a diagnostic or clinical decision-making tool.

---

## TL;DR

Built a recall-focused, interpretable neural net for early heart disease risk detection — using only survey-based features and class-imbalance-aware training.