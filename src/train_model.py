import os
import sys
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Construct the absolute path to weighted_train.csv
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
weighted_csv_path = os.path.join(project_root, "data", "weighted_train.csv")
print("Looking for weighted data at:", weighted_csv_path)

# Check if the weighted data file exists
if not os.path.exists(weighted_csv_path):
    sys.exit(f"Error: Weighted data file not found at {weighted_csv_path}.\n"
             "Please run 'python3 src/bias_mitigation.py' to generate weighted_train.csv before running this script.")

# Load the weighted data
weighted_df = pd.read_csv(weighted_csv_path)
print("Weighted data loaded. Shape:", weighted_df.shape)

# Assume the target column is 'income_binary'
if 'income_binary' not in weighted_df.columns:
    sys.exit("Error: Expected target column 'income_binary' not found in weighted data.")

# Separate features and target
X = weighted_df.drop(['income_binary'], axis=1)
y = weighted_df['income_binary']

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Weighted data split into training and testing sets.")

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model on the weighted data
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)
print("Model trained on weighted data.")

# Evaluate model performance on the test set
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print("Weighted Model Accuracy:", accuracy)
print("Weighted Model Classification Report:")
print(classification_report(y_test, predictions))

# Save the trained model using pickle in the "models" folder.
models_dir = os.path.join(project_root, "models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
model_path = os.path.join(models_dir, "logistic_regression_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print("Model saved as:", model_path)
