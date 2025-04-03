import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# --- Setting Up ---
# I'm starting by determining the project root so that I can build absolute paths reliably.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cleaned_csv_path = os.path.join(project_root, "data", "train_cleaned.csv")
print("Loading cleaned data from:", cleaned_csv_path)

# If the cleaned dataset isn't present, then we need to stop and prompt the user.
if not os.path.exists(cleaned_csv_path):
    sys.exit("Error: 'train_cleaned.csv' not found. Please run the data_cleaning.py script first.")

# --- Data Loading ---
# Load our cleaned dataset. This file should have been prepared in an earlier step.
df = pd.read_csv(cleaned_csv_path)
print("Cleaned data loaded. Shape:", df.shape)

# --- Data Preparation ---
# I'm assuming our target column is called 'income_binary'.
if 'income_binary' not in df.columns:
    sys.exit("Error: Expected target column 'income_binary' not found.")

# Separate features from the target.
X = df.drop(['income_binary'], axis=1)
y = df['income_binary']

# Split the data into training and test sets (using a 70-30 split).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Data split into training and testing sets.")

# --- Feature Scaling ---
# Standardizing features can help our model perform better.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Baseline Model Training ---
# I'm using Logistic Regression as our baseline classifier.
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)
pred_baseline = model.predict(X_test_scaled)

# Evaluate the model's performance.
acc = accuracy_score(y_test, pred_baseline)
print("Baseline Model Accuracy:", acc)
print("Baseline Classification Report:")
print(classification_report(y_test, pred_baseline))

# --- Fairness Evaluation ---
# We need to ensure our protected attribute, 'sex', is included in our feature set.
if 'sex' not in X.columns:
    sys.exit("Error: Protected attribute 'sex' not found in data.")

# Build a DataFrame for fairness evaluation that includes true and predicted labels.
X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df['true_label'] = y_test.values
X_test_df['pred_label'] = pred_baseline

# Convert the test data into an AIF360 BinaryLabelDataset format.
dataset_true = BinaryLabelDataset(df=X_test_df, label_names=['true_label'], protected_attribute_names=['sex'])
dataset_pred = dataset_true.copy()
dataset_pred.labels = X_test_df['pred_label'].values.reshape(-1, 1)

# Compute fairness metrics: Statistical Parity Difference and Disparate Impact.
fair_metric = ClassificationMetric(
    dataset_true,
    dataset_pred,
    privileged_groups=[{'sex': 1}],    # Assuming 'sex' = 1 is privileged (e.g., Male)
    unprivileged_groups=[{'sex': 0}]   # 'sex' = 0 is unprivileged (e.g., Female)
)
print("Baseline Fairness Metrics:")
print("Statistical Parity Difference:", fair_metric.statistical_parity_difference())
print("Disparate Impact:", fair_metric.disparate_impact())
