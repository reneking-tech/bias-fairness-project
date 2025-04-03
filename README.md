# AI Fairness Project

## Overview
This project demonstrates an end-to-end pipeline for addressing bias in AI systems using the UCI Adult dataset. The project covers:
- Data cleaning and leakage removal
- Bias detection using fairness metrics
- Bias mitigation (using techniques such as reweighing and disparate impact remover)
- Training models on both the original and mitigated datasets
- Evaluating model performance and fairness

The goal is to build a fair and responsible AI system and showcase the trade-offs between predictive performance and fairness.

## Folder Structure

AI_Fairness_Project/
├── data/
│   ├── raw/                     # Original data files (e.g., adult.data, adult.test, adult.names)
│   ├── processed/               # Intermediate processed data (optional)
│   ├── train.csv                # Your original training data (or downloaded data)
│   ├── train_cleaned.csv        # Data after cleaning and leakage removal
│   └── weighted_train.csv       # Data after applying bias mitigation (e.g., reweighing)
├── models/                      # Saved trained model files (e.g., pickle/joblib files)
├── notebooks/                   # Jupyter notebooks for exploration, analysis, and visualization
│   ├── data_exploration.ipynb
│   ├── bias_detection.ipynb
│   └── evaluation.ipynb
├── src/                         # Python scripts (the "engine" of your project)
│   ├── data_cleaning.py         # Loads raw data, removes leakage, and saves cleaned data
│   ├── bias_detection.py        # Loads cleaned data, computes baseline fairness metrics
│   ├── bias_mitigation.py       # Applies a bias mitigation technique (e.g., reweighing) and saves weighted data
│   ├── train_model.py           # Loads weighted data, trains a model, and saves the trained model
│   └── evaluation.py            # Evaluates model performance and fairness
├── venv_tf/                     # Your Python virtual environment folder (using Python 3.9/3.10)
├── README.md                    # Project description and instructions
└── requirements.txt             # List of required Python packagesR
