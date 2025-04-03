import os
import sys
import pandas as pd
from aif360.datasets import AdultDataset

# Print the current working directory (for debugging purposes)
project_root = os.getcwd()
print("Project root directory:", project_root)

# Construct the absolute path to the raw data file.
# Since this script is in the src/ folder, the raw file is expected at ../data/raw/adult/adult.data.
raw_data_file = os.path.join(project_root, "data", "raw", "adult", "adult.data")
print("Looking for raw data file at:", raw_data_file)

# Define file paths for the CSV files we will create.
train_csv = os.path.join(project_root, "data", "train.csv")
cleaned_csv = os.path.join(project_root, "data", "train_cleaned.csv")

# Check if train.csv already exists; if not, create it from the raw data.
if not os.path.exists(train_csv):
    print("train.csv not found. Creating train.csv from raw Adult dataset...")
    try:
        # Attempt to load using AIF360's AdultDataset (if data is already numeric).
        dataset = AdultDataset(protected_attribute_names=['sex'], features_to_drop=['fnlwgt'])
        data_df, label_names, protected_attribute_names = dataset.convert_to_dataframe()
    except Exception as e:
        print("Error using AdultDataset:", e)
        print("Falling back to manual loading and encoding...")
        # Define column names based on the Adult dataset documentation.
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income-per-year'
        ]
        # Load the raw data with pandas. The raw file has no header, so we supply the column names.
        data_df = pd.read_csv(raw_data_file, header=None, names=columns, na_values='?')
        # Drop rows with missing data because missing values can lead to errors or unreliable model performance.
        data_df = data_df.dropna()
        # Encode categorical variables to numeric codes:
        # This is necessary because many machine learning models, and AIF360, require numerical input.
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country', 'income-per-year']
        for col in categorical_cols:
            data_df[col] = pd.Categorical(data_df[col]).codes
        # Create a binary target column.
        # Here, we assume that a higher categorical code corresponds to '>50K' income.
        data_df['income_binary'] = data_df['income-per-year'].apply(lambda x: 1 if x > 0 else 0)
        # Remove the original income column as it's now redundant.
        data_df = data_df.drop('income-per-year', axis=1)
        label_names = ['income_binary']
        protected_attribute_names = ['sex']
    # Save the resulting DataFrame as train.csv.
    data_df.to_csv(train_csv, index=False)
    print("train.csv created and saved at:", train_csv)
else:
    print("train.csv already exists.")

# Load train.csv for further processing.
data_df = pd.read_csv(train_csv)
print("Loaded train.csv with shape:", data_df.shape)

# Remove the leakage feature if it exists (e.g., "14_ <=50K").
if "14_ <=50K" in data_df.columns:
    data_df = data_df.drop("14_ <=50K", axis=1)
    print("Leakage feature '14_ <=50K' removed.")
else:
    print("No leakage feature '14_ <=50K' found.")

# Choose the target column ('income_binary' if it exists, otherwise '14_ >50K').
target_column = 'income_binary' if 'income_binary' in data_df.columns else '14_ >50K'
data_df[target_column] = data_df[target_column].astype(int)

# Ensure the protected attribute 'sex' is numeric.
if 'sex' in data_df.columns:
    data_df['sex'] = data_df['sex'].astype(int)

# Save the final cleaned dataset.
data_df.to_csv(cleaned_csv, index=False)
print("Cleaned dataset saved at:", cleaned_csv)
print("Data sample:")
print(data_df.head())
