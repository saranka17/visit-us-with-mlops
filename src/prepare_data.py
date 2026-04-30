"""Prepare tourism data from the Hugging Face Dataset Hub."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "ProdTaken"
RAW_FILENAME = "tourism.csv"


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

    print(f"Saved cleaned training data to: {train_path}")
    print(f"Saved cleaned testing data to: {test_path}")

    if dataset_repo_id and hf_token:
        api = HfApi(token=hf_token)
        try:
            api.upload_file(
                path_or_fileobj=str(cleaned_path),
                path_in_repo="processed/cleaned.csv",
                repo_id=dataset_repo_id,
                repo_type="dataset",
            )
            api.upload_file(
                path_or_fileobj=str(train_path),
                path_in_repo="processed/train.csv",
                repo_id=dataset_repo_id,
                repo_type="dataset",
            )
            api.upload_file(
                path_or_fileobj=str(test_path),
                path_in_repo="processed/test.csv",
                repo_id=dataset_repo_id,
                repo_type="dataset",
            )
            print(f"Uploaded cleaned, train, and test datasets to: {dataset_repo_id}")
        except Exception as exc:
            print(
                "Unable to upload processed datasets to Hugging Face. "
                f"Keeping the local files only: {exc}"
            )
    else:
        print("HF_TOKEN or DATASET_REPO_ID not found. Skipping Hugging Face upload.")



if __name__ == "__main__":
    main()
