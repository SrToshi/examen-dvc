import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["gridsearch"]
    cv = params.pop("cv")  # Extract cv from params and remove it from the dictionary

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")["silica_concentrate"]  # Assuming the target column is named "silica concentrate"  

grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state = 0),
                           param_grid = params,
                           cv=cv,
                           n_jobs = -1,
                           verbose = 2
                           )

grid_search.fit(X_train, y_train)

joblib.dump(grid_search.best_params_, "models/best_params.pkl")
