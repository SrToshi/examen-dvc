import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib

best_params = joblib.load("models/best_params.pkl")

model = GradientBoostingRegressor(random_state = 0, **best_params)

X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")["silica_concentrate"]  # Assuming the target column is named "silica concentrate"

model.fit(X_train_scaled, y_train)

joblib.dump(model, "models/gbr_model.pkl")

