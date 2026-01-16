# NLP Stacking Ensemble – 20 Newsgroups

This project implements an end-to-end NLP pipeline on the 20 Newsgroups dataset.

## What it does
- Loads ~18,000 documents across 20 categories
- Splits data into train/test (80/20)
- Preprocesses text and vectorizes using TF-IDF (2000 features)
- Visualizes document clusters using PCA + t-SNE
- Trains:
  - Naive Bayes
  - Logistic Regression
  - Neural Network (MLP)
- Builds a Stacking Ensemble using NB + LR with MLP as meta-learner
- Evaluates using:
  - Accuracy (~82.6%)
  - Classification report
  - Confusion matrix (raw + normalized)
  - ROC-AUC
- Visualizes:
  - Class separation (t-SNE)
  - Model comparison
  - Feature importance (Logistic Regression coefficients)

## Key Learning Outcomes
- NLP preprocessing and feature engineering
- Ensemble learning with stacking
- Model evaluation and error analysis
- Interpretability in text classification

This notebook demonstrates an evaluation-driven, end-to-end NLP workflow.
