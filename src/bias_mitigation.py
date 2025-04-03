import os
import sys
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

# Define the absolute path for the cleaned data file.
cleaned_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "train_cleaned.csv"))
print("Looking for cleaned data at:", cleaned_csv_path)

# Check if the cleaned data file exists.
if not os.path.exists(cleaned_csv_path):
    sys.exit(f"Error: Cleaned data file not found at {cleaned_csv_path}.\n"
             "Please run 'python3 src/data_cleaning.py' to generate train_cleaned.csv before running this script.")

# Load the cleaned data.
df = pd.read_csv(cleaned_csv_path)
print("Loaded cleaned data with shape:", df.shape)

# Convert the DataFrame to an AIF360 BinaryLabelDataset.
# Here, the target column is 'income_binary' and the protected attribute is 'sex'.
dataset = BinaryLabelDataset(df=df, label_names=['income_binary'], protected_attribute_names=['sex'])

# Apply the Reweighing algorithm.
rw = Reweighing(unprivileged_groups=[{'sex': 0}], privileged_groups=[{'sex': 1}])
dataset_reweighted = rw.fit_transform(dataset)
print("Reweighing applied to dataset.")

# Convert the reweighted dataset back to a pandas DataFrame.
weighted_df, _ = dataset_reweighted.convert_to_dataframe()

# Define the path to save the weighted data.
weighted_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "weighted_train.csv"))

# Save the weighted data.
weighted_df.to_csv(weighted_csv_path, index=False)
print("Weighted data saved as:", weighted_csv_path)
