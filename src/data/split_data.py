import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["split"]

df = pd.read_csv("data/raw_data/raw.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(["date", "silica_concentrate"], axis=1),
    df["silica_concentrate"],
    test_size=params["test_size"],
    random_state=params["random_state"]
)

X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train.to_csv("data/processed_data/y_train.csv", index=False)
y_test.to_csv("data/processed_data/y_test.csv", index=False)
