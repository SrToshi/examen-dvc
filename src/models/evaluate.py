import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

model = joblib.load("models/gbr_model.pkl")

X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")["silica_concentrate"]  # Assuming the target column is named "silica concentrate"

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)   

model_metrics= {
    "mean_squared_error": float(mse),
    "r2_score": float(r2)
}

with open("metrics/scores.json", "w") as f:
    json.dump(model_metrics, f)

pd.DataFrame(y_pred, columns=["predicted_silica_concentrate"]).to_csv("data/prediction.csv", index=False)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")   