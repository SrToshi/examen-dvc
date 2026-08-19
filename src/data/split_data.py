import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

# Test_size and random_state are read from params.yaml to keep the split reproducible
# and versioned as part of the DVC pipeline (see the 'split' stage in dvc.yaml).
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["split"]

# Splits raw.csv into train/test sets for features (X) and target (y).
df = pd.read_csv("data/raw_data/raw.csv")

# The 'date' column is dropped (not a predictive feature, not covered by the assignment spec).
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["date", "silica_concentrate"], axis=1),
    df["silica_concentrate"],
    test_size=params["test_size"],
    random_state=params["random_state"]
)

# We persist the train/test splits in the processed_data folder for use in the next step of the pipeline (normalization).
X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train.to_csv("data/processed_data/y_train.csv", index=False)
y_test.to_csv("data/processed_data/y_test.csv", index=False)
