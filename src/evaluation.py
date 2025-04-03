import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# Step 1: Load Weighted Data

# Construct the absolute path to the weighted data file
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
weighted_csv_path = os.path.join(project_root, "data", "weighted_train.csv")
print("Looking for weighted data at:", weighted_csv_path)

# Check if the file exists; if not, exit with an error message.
if not os.path.exists(weighted_csv_path):
    sys.exit(f"Error: Weighted data file not found at {weighted_csv_path}.\n"
             "Please run 'python3 src/bias_mitigation.py' to generate weighted_train.csv first.")

# Load the weighted data.
df = pd.read_csv(weighted_csv_path)
print("Weighted data loaded. Shape:", df.shape)

# Step 2: Prepare Data for Evaluation

# Check for the target column 'income_binary'
if 'income_binary' not in df.columns:
    sys.exit("Error: Expected target column 'income_binary' not found in weighted data.")

# Separate features and target.
X = df.drop(['income_binary'], axis=1)
y = df['income_binary']

# Split the data (70% train, 30% test).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Data split into training and testing sets.")
print("Training set shape:", X_train.shape, "Test set shape:", X_test.shape)

# Scale features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train a Model on Weighted Data

# Train a Logistic Regression model.
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)
print("Model trained on weighted data.")

# Predict on test set.
predictions = model.predict(X_test_scaled)

# Evaluate performance.
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, predictions))

# Step 4: Evaluate Fairness Metrics

# Check that the protected attribute 'sex' is in the features.
if 'sex' not in X.columns:
    sys.exit("Error: Protected attribute 'sex' not found in weighted data.")

# Prepare DataFrame for fairness evaluation.
X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df['true_label'] = y_test.values
X_test_df['pred_label'] = predictions

# Convert to AIF360 BinaryLabelDataset.
dataset_true = BinaryLabelDataset(df=X_test_df, label_names=['true_label'], protected_attribute_names=['sex'])
dataset_pred = dataset_true.copy()
dataset_pred.labels = X_test_df['pred_label'].values.reshape(-1, 1)

# Compute fairness metrics.
fair_metric = ClassificationMetric(
    dataset_true,
    dataset_pred,
    privileged_groups=[{'sex': 1}],   # Assuming 'sex' = 1 indicates the privileged group (e.g., Male)
    unprivileged_groups=[{'sex': 0}]  # 'sex' = 0 indicates the unprivileged group (e.g., Female)
)
print("\nFairness Metrics on Weighted Model Predictions:")
print("Statistical Parity Difference:", fair_metric.statistical_parity_difference())
print("Disparate Impact:", fair_metric.disparate_impact())

# Step 5: Visualize Group-Level Positive Prediction Rates

# Calculate average positive prediction rate by protected group.
grouped = X_test_df.groupby('sex')['pred_label'].mean()
print("\nAverage positive prediction rate by group:")
print(grouped)

plt.figure(figsize=(8, 6))
plt.bar(grouped.index.astype(str), grouped.values, color=['skyblue', 'orange'])
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Positive Prediction Rate')
plt.title('Positive Prediction Rate by Protected Group')
plt.grid(True)
plt.show()
