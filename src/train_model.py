"""Train a Random Forest model with MLflow tracking and upload the best model to HF."""

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "ProdTaken"
TRAIN_FILENAME = "processed/train.csv"
TEST_FILENAME = "processed/test.csv"


def resolve_processed_file(
    project_root: Path,
    dataset_repo_id: str | None,
    hf_token: str | None,
    filename: str,
) -> Path:
    local_file = project_root / "data" / filename

    if dataset_repo_id:
        try:
            downloaded_file = hf_hub_download(
                repo_id=dataset_repo_id,
                filename=filename,
                repo_type="dataset",
                token=hf_token,
            )
            print(f"Downloaded {filename} from Hugging Face: {downloaded_file}")
            return Path(downloaded_file)
        except Exception as exc:
            print(
                f"Unable to download {filename} from Hugging Face. "
                f"Falling back to local file: {exc}"
            )

    if not local_file.exists():
        raise FileNotFoundError(f"Processed data not found locally: {local_file}")

    print(f"Using local processed file: {local_file}")
    return local_file


def load_data(project_root: Path, dataset_repo_id: str | None, hf_token: str | None):
    train_path = resolve_processed_file(project_root, dataset_repo_id, hf_token, TRAIN_FILENAME)
    test_path = resolve_processed_file(project_root, dataset_repo_id, hf_token, TEST_FILENAME)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_pipeline(X_train: pd.DataFrame):
    categorical_features = X_train.select_dtypes(include="object").columns.tolist()
    numeric_features = X_train.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    param_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__class_weight": [None, "balanced"],
    }

    return pipeline, param_grid


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return metrics, report


def upload_model_to_hf(artifacts_dir: Path, model_repo_id: str, hf_token: str | None) -> None:
    if not hf_token:
        print("HF_TOKEN not found. Skipping model upload.")
        return

    api = HfApi(token=hf_token)

    create_repo(
        repo_id=model_repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
        token=hf_token,
    )

    api.upload_folder(
        folder_path=str(artifacts_dir),
        repo_id=model_repo_id,
        repo_type="model",
    )

    print(f"Best model uploaded to Hugging Face Model Hub: {model_repo_id}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    hf_token = os.getenv("HF_TOKEN")
    dataset_repo_id = os.getenv("DATASET_REPO_ID")
    hf_username = os.getenv("HF_USERNAME")
    model_repo_id = os.getenv("MODEL_REPO_ID") or f"{hf_username}/visit-with-us-random-forest"

    if not dataset_repo_id and not (project_root / "data" / TRAIN_FILENAME).exists():
        raise ValueError("DATASET_REPO_ID is missing and no local processed train/test files were found.")

    mlflow.set_tracking_uri(f"file://{(project_root / 'mlruns').resolve()}")
    mlflow.set_experiment("visit-with-us-model-training")

    train_df, test_df = load_data(project_root, dataset_repo_id, hf_token)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    pipeline, param_grid = build_pipeline(X_train)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    artifacts_dir = project_root / "artifacts" / "random_forest_model"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="random_forest_grid_search"):
        mlflow.log_param("model_name", "RandomForestClassifier")
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("scoring", "roc_auc")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_dict(param_grid, "param_grid.json")

        search.fit(X_train, y_train)

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.to_csv(artifacts_dir / "cv_results.csv", index=False)

        for idx, row in cv_results.iterrows():
            with mlflow.start_run(run_name=f"trial_{idx}", nested=True):
                mlflow.log_params(row["params"])
                mlflow.log_metric("mean_test_score", float(row["mean_test_score"]))
                mlflow.log_metric("std_test_score", float(row["std_test_score"]))
                mlflow.log_metric("rank_test_score", int(row["rank_test_score"]))

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_roc_auc", float(best_cv_score))

        metrics, report = evaluate_model(best_model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(report, "classification_report.json")
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        joblib.dump(best_model, artifacts_dir / "model.joblib")

        with open(artifacts_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(artifacts_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        model_card = f"""---
license: mit
library_name: scikit-learn
tags:
- mlflow
- scikit-learn
- random-forest
- binary-classification
---

# Visit With Us Random Forest Model

## Dataset
- Dataset repo: `{dataset_repo_id}`
- Train file: `processed/train.csv`
- Test file: `processed/test.csv`

## Best Parameters
{json.dumps(best_params, indent=2)}

## Test Metrics
{json.dumps(metrics, indent=2)}
"""
        (artifacts_dir / "README.md").write_text(model_card)

        print("Best Parameters:", best_params)
        print("Best CV ROC-AUC:", round(best_cv_score, 4))
        print("Test Metrics:", metrics)

    upload_model_to_hf(artifacts_dir, model_repo_id, hf_token)


if __name__ == "__main__":
    main()
