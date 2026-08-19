import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib


# Trains the final GradientBoostingRegressor using the best hyperparameters
# found by grid_search.py (loaded from models/best_params.pkl).
best_params = joblib.load("models/best_params.pkl")

model = GradientBoostingRegressor(random_state = 0, **best_params)

# Here we obtain the scaled training data and the target variable from the processed_data folder
X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")["silica_concentrate"]  # Assuming the target column is named "silica concentrate"

# Kept as a separate stage from gridsearch to decouple hyperparameter search
# (expensive, exploratory) from model fitting (cheap, reproducible).
model.fit(X_train_scaled, y_train)

# Here we persist the trained model in the models folder for use in the next step of the pipeline (evaluation).
joblib.dump(model, "models/gbr_model.pkl")


