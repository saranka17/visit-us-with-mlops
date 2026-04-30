"""Prepare tourism data from the Hugging Face Dataset Hub."""

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "ProdTaken"
RAW_FILENAME = "tourism.csv"
PROCESSED_FILENAMES = {
    "cleaned": "processed/cleaned.csv",
    "train": "processed/train.csv",
    "test": "processed/test.csv",
    "schema": "processed/feature_schema.json",
    "metadata": "processed/data_metadata.json",
}


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names first.
    df.columns = [str(column).strip() for column in df.columns]

    # Remove non-predictive identifier columns.
    unnamed_columns = [
        column for column in df.columns
        if not column or column.startswith("Unnamed:")
    ]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Trim whitespace in text columns.
    object_columns = df.select_dtypes(include="object").columns
    for column in object_columns:
        df[column] = df[column].astype(str).str.strip()

    # Standardize inconsistent categorical labels observed in the raw data.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({
            "Fe Male": "Female",
        })

    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({
            "Unmarried": "Single",
        })

    # Remove only exact duplicate rows from the raw data.
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def build_feature_schema(df: pd.DataFrame) -> dict:
    """Create deployment-friendly metadata for each model feature."""
    schema: dict[str, dict] = {}

    for column in df.columns:
        if column == TARGET_COLUMN:
            continue

        series = df[column].dropna()

        if pd.api.types.is_object_dtype(df[column]):
            # Store the allowed values for categorical features so the
            schema[column] = {
                "type": "categorical",
                "categories": sorted(series.astype(str).unique().tolist()),
                "default": series.mode().iat[0] if not series.empty else "",
            }
            continue

        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        # Preserve integer-like fields as integers in the UI even if pandas loaded them as floats because of missing values in the raw data.
        is_integer_like = (
            pd.api.types.is_integer_dtype(df[column])
            or ((numeric_series % 1) == 0).all()
        )
        default_value = numeric_series.median() if not numeric_series.empty else 0

        schema[column] = {
            "type": "integer" if is_integer_like else "float",
            "min": float(numeric_series.min()) if not numeric_series.empty else 0.0,
            "max": float(numeric_series.max()) if not numeric_series.empty else 0.0,
            "default": int(default_value) if is_integer_like else float(default_value),
        }

    return schema


def build_data_metadata(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
    # Save compact summary metadata so the model card and deployment layer can explain how the training data was derived.
    return {
        "target_column": TARGET_COLUMN,
        "raw_shape": list(raw_df.shape),
        "cleaned_shape": list(cleaned_df.shape),
        "removed_rows": int(len(raw_df) - len(cleaned_df)),
        "removed_columns": sorted(set(raw_df.columns) - set(cleaned_df.columns)),
        "feature_columns": [column for column in cleaned_df.columns if column != TARGET_COLUMN],
        "class_distribution": cleaned_df[TARGET_COLUMN].value_counts().sort_index().to_dict(),
    }



def resolve_raw_data_path(
    data_dir: Path,
    dataset_repo_id: str | None,
    hf_token: str | None,
) -> Path:
    local_file = data_dir / RAW_FILENAME

    if dataset_repo_id:
        try:
            downloaded_file = hf_hub_download(
                repo_id=dataset_repo_id,
                filename=RAW_FILENAME,
                repo_type="dataset",
                token=hf_token,
            )
            print(f"Downloaded raw dataset from Hugging Face: {downloaded_file}")
            return Path(downloaded_file)
        except Exception as exc:
            print(
                "Unable to download the raw dataset from Hugging Face. "
                f"Falling back to the local file: {exc}"
            )

    # Local fallback keeps the notebook runnable even when network access or credentials are unavailable.
    if not local_file.exists():
        raise FileNotFoundError(f"Raw data not found locally: {local_file}")

    print(f"Using local raw dataset: {local_file}")
    return local_file


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv(project_root / ".env")

    hf_username = os.getenv("HF_USERNAME")
    dataset_repo_id = os.getenv("DATASET_REPO_ID") or (
        f"{hf_username}/visit-with-us-tourism" if hf_username else None
    )
    hf_token = os.getenv("HF_TOKEN")

    raw_data_path = resolve_raw_data_path(data_dir, dataset_repo_id, hf_token)
    df = pd.read_csv(raw_data_path)
    cleaned_df = clean_dataset(df)

    if TARGET_COLUMN not in cleaned_df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    cleaned_path = processed_dir / "cleaned.csv"
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned full dataset to: {cleaned_path}")

    # Stratified splitting preserves the class ratio of the target in both train and test sets, which is important for consistent evaluation.
    train_df, test_df = train_test_split(
        cleaned_df,
        test_size=0.2,
        random_state=42,
        stratify=cleaned_df[TARGET_COLUMN],
    )

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    feature_schema = build_feature_schema(cleaned_df)
    feature_schema_path = processed_dir / "feature_schema.json"
    feature_schema_path.write_text(json.dumps(feature_schema, indent=2))

    data_metadata = build_data_metadata(df, cleaned_df)
    metadata_path = processed_dir / "data_metadata.json"
    metadata_path.write_text(json.dumps(data_metadata, indent=2))

    print(f"Saved cleaned training data to: {train_path}")
    print(f"Saved cleaned testing data to: {test_path}")
    print(f"Saved feature schema to: {feature_schema_path}")
    print(f"Saved data metadata to: {metadata_path}")

    if dataset_repo_id and hf_token:
        api = HfApi(token=hf_token)
        try:
            # Upload both the data splits and the metadata used by the deployment layer so the entire workflow stays reproducible.
            uploads = {
                cleaned_path: PROCESSED_FILENAMES["cleaned"],
                train_path: PROCESSED_FILENAMES["train"],
                test_path: PROCESSED_FILENAMES["test"],
                feature_schema_path: PROCESSED_FILENAMES["schema"],
                metadata_path: PROCESSED_FILENAMES["metadata"],
            }
            for local_path, repo_path in uploads.items():
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=repo_path,
                    repo_id=dataset_repo_id,
                    repo_type="dataset",
                )
            print(
                "Uploaded cleaned data, train/test splits, and metadata to: "
                f"{dataset_repo_id}"
            )
        except Exception as exc:
            print(
                "Unable to upload processed datasets to Hugging Face. "
                f"Keeping the local files only: {exc}"
            )
    else:
        print("HF_TOKEN or DATASET_REPO_ID not found. Skipping Hugging Face upload.")



if __name__ == "__main__":
    main()
