import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Evaluates the trained model on the held-out test set.

model = joblib.load("models/gbr_model.pkl")

# here we obtain the scaled test data and the target variable from the processed_data folder
X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")["silica_concentrate"]  # Assuming the target column is named "silica concentrate"

# here we make predictions on the test set and compute evaluation metrics (MSE, R2).
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)   

# this dictionary will be persisted in metrics/scores.json for use in the next step of the pipeline (reporting).
model_metrics= {
    "mean_squared_error": float(mse),
    "r2_score": float(r2)
}

# Outputs: metrics/scores.json (MSE, R2).
with open("metrics/scores.json", "w") as f:
    json.dump(model_metrics, f)

# Outputs: data/prediction.csv (raw predictions)
pd.DataFrame(y_pred, columns=["predicted_silica_concentrate"]).to_csv("data/prediction.csv", index=False)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")   

