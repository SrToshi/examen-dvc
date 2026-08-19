import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Scales X_train/X_test with StandardScaler, fit only on X_train to avoid data leakage.

# set up the scaler
scaler = StandardScaler()

# we will read the train and test data that we created in the previous step, form the preprocessed_data folder
X_train = pd.read_csv("data/processed_data/X_train.csv")
X_test = pd.read_csv("data/processed_data/X_test.csv")

# we adjust with the train data and transform both the train and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# we persist the scaled data in the processed_data folder 
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv("data/processed_data/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("data/processed_data/X_test_scaled.csv", index=False)

# we also persist the scaler object in the models folder
joblib.dump(scaler, "models/scaler.pkl")

# Note: Scaling has no real effect on GBR's splits (tree-based, scale-invariant) — this
# step exists mainly for pipeline completeness / portability to scale-sensitive models.