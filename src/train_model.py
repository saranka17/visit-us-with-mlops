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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "ProdTaken"
TRAIN_FILENAME = "processed/train.csv"
TEST_FILENAME = "processed/test.csv"
FEATURE_SCHEMA_FILENAME = "processed/feature_schema.json"
DATA_METADATA_FILENAME = "processed/data_metadata.json"


def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_n_jobs() -> int:
    # Default to a single process locally because some restricted environments block process-based parallelism. CI can override this to `-1`.
    raw_value = os.getenv("SKLEARN_N_JOBS", "1").strip()
    try:
        return int(raw_value)
    except ValueError:
        print(f"Invalid SKLEARN_N_JOBS value '{raw_value}'. Falling back to 1.")
        return 1


def resolve_processed_file(
    project_root: Path,
    dataset_repo_id: str | None,
    hf_token: str | None,
    filename: str,
) -> Path:
    local_file = project_root / "data" / filename

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
        if allow_local_fallback and local_file.exists():
            print(
                f"Unable to download {filename} from Hugging Face. "
                "Using the local file because ALLOW_LOCAL_FALLBACK is enabled: "
                f"{exc}"
            )
            return local_file
        raise RuntimeError(
            f"Unable to download {filename} from Hugging Face Dataset Hub."
        ) from exc


def load_data(project_root: Path, dataset_repo_id: str | None, hf_token: str | None):
    train_path = resolve_processed_file(project_root, dataset_repo_id, hf_token, TRAIN_FILENAME)
    test_path = resolve_processed_file(project_root, dataset_repo_id, hf_token, TEST_FILENAME)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def load_optional_json(
    project_root: Path,
    dataset_repo_id: str | None,
    hf_token: str | None,
    filename: str,
) -> dict:
    if dataset_repo_id and hf_token:
        try:
            downloaded_file = hf_hub_download(
                repo_id=dataset_repo_id,
                filename=filename,
                repo_type="dataset",
                token=hf_token,
            )
            return json.loads(Path(downloaded_file).read_text())
        except Exception:
            pass

    return {}


def build_pipeline(X_train: pd.DataFrame, n_jobs: int):
    categorical_features = X_train.select_dtypes(include="object").columns.tolist()
    numeric_features = X_train.select_dtypes(exclude="object").columns.tolist()

    # Encode categoricals and pass through numeric columns in one pipeline so the exact same preprocessing is reused during inference.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42, n_jobs=n_jobs)),
        ]
    )

    param_grid = {
        "model__n_estimators": [150, 250, 350],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__class_weight": [None, "balanced"],
    }

    return pipeline, param_grid, categorical_features, numeric_features


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    # Keep a probability-based metric alongside threshold-based metrics because the ranking quality matters for lead prioritization use cases.
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
        raise ValueError("HF_TOKEN must be set to upload the best model to Hugging Face.")

    try:
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
    except Exception as exc:
        raise RuntimeError(
            "Unable to upload the best model to Hugging Face Model Hub."
        ) from exc


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    dataset_repo_id = os.getenv("DATASET_REPO_ID") or (
        f"{hf_username}/visit-with-us-tourism" if hf_username else None
    )
    model_repo_id = os.getenv("MODEL_REPO_ID") or (
        f"{hf_username}/visit-with-us-random-forest" if hf_username else "visit-with-us-random-forest"
    )

    if not dataset_repo_id and not env_flag("ALLOW_LOCAL_FALLBACK", default=False):
        raise ValueError(
            "DATASET_REPO_ID or HF_USERNAME must be set to load train and test data from Hugging Face."
        )

    mlflow.set_tracking_uri(f"file://{(project_root / 'mlruns').resolve()}")
    mlflow.set_experiment("visit-with-us-model-training")
    n_jobs = resolve_n_jobs()

    # Training reads back the processed data from the Hub when available so the model step is decoupled from whichever machine created the split.
    train_df, test_df = load_data(project_root, dataset_repo_id, hf_token)
    feature_schema = load_optional_json(project_root, dataset_repo_id, hf_token, FEATURE_SCHEMA_FILENAME)
    data_metadata = load_optional_json(project_root, dataset_repo_id, hf_token, DATA_METADATA_FILENAME)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    pipeline, param_grid, categorical_features, numeric_features = build_pipeline(X_train, n_jobs)
    # Shuffle before each fold split so the CV estimate is less sensitive to any accidental ordering in the saved training data.
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_strategy,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
    )

    artifacts_dir = project_root / "artifacts" / "random_forest_model"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="random_forest_grid_search"):
        # Log run-level context once so each experiment can be understood later without reopening the notebook.
        mlflow.log_param("model_name", "RandomForestClassifier")
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("scoring", "roc_auc")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("n_jobs", n_jobs)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("categorical_feature_count", len(categorical_features))
        mlflow.log_param("numeric_feature_count", len(numeric_features))
        mlflow.log_dict(param_grid, "param_grid.json")

        search.fit(X_train, y_train)

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.to_csv(artifacts_dir / "cv_results.csv", index=False)

        # Nested runs make it easier to inspect every parameter combination in MLflow while still keeping one parent run for the full experiment.
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

        # Save the final trained pipeline locally because the deployment app loads the exact serialized estimator from the model hub.
        joblib.dump(best_model, artifacts_dir / "model.joblib")

        with open(artifacts_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(artifacts_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)

        (artifacts_dir / "feature_schema.json").write_text(
            json.dumps(feature_schema or {}, indent=2)
        )

        # Persist model metadata separately from raw metrics so deployment and reporting layers can read one compact summary file.
        model_metadata = {
            "model_name": "RandomForestClassifier",
            "dataset_repo_id": dataset_repo_id,
            "model_repo_id": model_repo_id,
            "target_column": TARGET_COLUMN,
            "feature_columns": X_train.columns.tolist(),
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "best_params": best_params,
            "best_cv_roc_auc": float(best_cv_score),
            "test_metrics": metrics,
            "data_metadata": data_metadata,
        }
        (artifacts_dir / "model_metadata.json").write_text(
            json.dumps(model_metadata, indent=2)
        )

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

        mlflow.log_artifact(str(artifacts_dir / "cv_results.csv"))
        mlflow.log_artifact(str(artifacts_dir / "metrics.json"))
        mlflow.log_artifact(str(artifacts_dir / "best_params.json"))
        mlflow.log_artifact(str(artifacts_dir / "feature_schema.json"))
        mlflow.log_artifact(str(artifacts_dir / "model_metadata.json"))

        print("Best Parameters:", best_params)
        print("Best CV ROC-AUC:", round(best_cv_score, 4))
        print("Test Metrics:", metrics)

    upload_model_to_hf(artifacts_dir, model_repo_id, hf_token)


if __name__ == "__main__":
    main()
