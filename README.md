# DVC and DagsHub Exam

This is my submission for the DVC/DagsHub module exam. The task: build a DVC-tracked pipeline that predicts `silica_concentrate` in a mineral flotation process, from 8 operational features, using a `GradientBoostingRegressor`.

## What's in here

Five stages, chained through `dvc.yaml`:
raw.csv → split → normalize → gridsearch → training → evaluate


- **split** — train/test split (80/20), target is `silica_concentrate`. The `date` column is dropped; it's a timestamp, not a predictive feature, and wasn't part of the assignment's feature list.

- **normalize** — `StandardScaler`, fit on train only, applied to both. Worth noting: since the final model is tree-based, scaling has essentially no effect on GBR's splits. It's here mostly for pipeline completeness — if this were swapped for a scale-sensitive model later, the step is already in place.

- **gridsearch** — searches over `n_estimators`, `learning_rate`, `max_depth`,
  `min_samples_leaf`, `subsample` (grid defined in `params.yaml`). Only the best hyperparameter combination is saved here (`best_params.pkl`), not the fitted model itself — training is a separate stage on purpose, to keep the (expensive) search  decoupled from the (cheap, deterministic) final fit.

- **training** — fits the final model, found in the previous step, although persisted only via the parameters of the GBR.

- **evaluate** — predictions on the test set, plus `scores.json` (MSE, R²).

## Results

MSE: 0.754, R²: 0.247

R² is fairly modest, but consistent between cross-validation during grid search
(~0.22) and the final test evaluation (~0.25) — so this isn't overfitting, it looks more like a real ceiling on how much these 8 features can explain. A few of the best hyperparameters also landed at the edge of the grid I searched (highest `n_estimators`, lowest `learning_rate`), which suggests there might be a bit more to squeeze out with a wider search — didn't chase that further given time constraints, but it's a natural next step.

## Reproducing

```bash
git clone <this repo>
cd examen-dvc
pip install -r requirements.txt
dvc pull
dvc repro
```

## A note on "models tab"

The assignment asks for *"a .pkl file in the __models__ tab of DagsHub"*. DagsHub has two things called "Models": a top-level MLflow-backed Model Registry, and a "Models" filter inside the Files view that just lists DVC-tracked files under `models/`. Nothing in the assignment mentions MLflow, and the same bold/italic-emphasis pattern is used for *"the __data__ tab"* — which is unambiguously the Files/Data filter (verified: CSVs render fine there). Reading both consistently, `models/` here refers to the Files/Models filter, where `best_params.pkl`, `gbr_model.pkl` and `scaler.pkl` are all visible with their DVC label. Didn't wire up MLflow registration on top of an already-working pipeline for what reads as an already-satisfied, ambiguous requirement.

## On algorithm coupling

`GradientBoostingRegressor` is currently hardcoded into `grid_search.py` and
`training.py` rather than parametrized and that means taht switching models would mean touching code in two places, not just `params.yaml`. Fine for what this exam asks for (the diagram specifies `gbr_model.pkl` explicitly, I took it as a clue), but worth flagging as the natural next step if this pipeline were to compare multiple model families.