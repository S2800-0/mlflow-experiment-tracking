"""Train the same gradient boosting model with three hyperparameter sets and track every run in MLflow."""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "california-house-price"
SEED = 42
N_ESTIMATORS = 100
# MLflow saves sklearn models with skops, which only accepts types it has been told to trust.
# The decision tree node type is created by this script itself, so it is safe to trust here.
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

RUNS = [
    {"name": "run_1", "max_depth": 3, "learning_rate": 0.1},
    {"name": "run_2", "max_depth": 5, "learning_rate": 0.05},
    {"name": "run_3", "max_depth": 7, "learning_rate": 0.01},
]


def load_split():
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=SEED)


def evaluate(model, X_val, y_val):
    pred = model.predict(X_val)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
        "mae": float(mean_absolute_error(y_val, pred)),
        "r2": float(r2_score(y_val, pred)),
    }


def main():
    X_train, X_val, y_train, y_val = load_split()
    print(f"Training rows: {len(X_train)}, validation rows: {len(X_val)}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    for cfg in RUNS:
        with mlflow.start_run(run_name=cfg["name"]):
            model = GradientBoostingRegressor(
                n_estimators=N_ESTIMATORS,
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                random_state=SEED,
            )
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_val, y_val)

            mlflow.log_params({
                "max_depth": cfg["max_depth"],
                "learning_rate": cfg["learning_rate"],
                "n_estimators": N_ESTIMATORS,
                "random_state": SEED,
            })
            mlflow.log_metrics(metrics)
            signature = infer_signature(X_val, model.predict(X_val))
            mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=X_val.head(5),
                                     skops_trusted_types=TRUSTED_TYPES)

            print(f"{cfg['name']}: max_depth={cfg['max_depth']} learning_rate={cfg['learning_rate']} "
                  f"RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f}")

    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["metrics.rmse ASC"])
    latest = runs.sort_values("start_time", ascending=False).head(len(RUNS))
    table = latest.sort_values("tags.mlflow.runName")[
        ["tags.mlflow.runName", "params.max_depth", "params.learning_rate",
         "metrics.rmse", "metrics.mae", "metrics.r2"]
    ]
    table.columns = ["Run", "Max Depth", "Learning Rate", "RMSE", "MAE", "R2"]
    print("\nComparison:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = latest.sort_values("metrics.rmse").iloc[0]
    mlflow.MlflowClient().set_tag(best["run_id"], "best_model", "true")
    print(f"\nBest model by lowest RMSE: {best['tags.mlflow.runName']} "
          f"(max_depth={best['params.max_depth']}, learning_rate={best['params.learning_rate']}, "
          f"RMSE={best['metrics.rmse']:.4f})")


if __name__ == "__main__":
    main()
