"""Register raw tourism data to Hugging Face Dataset Hub."""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise ValueError(f"{name} must be set to register data on Hugging Face.")


def main():
    # Resolve all project-relative paths from the script location so the code
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "tourism.csv"
    load_dotenv(project_root / ".env")

    hf_username = os.getenv("HF_USERNAME")
    dataset_repo_id = os.getenv("DATASET_REPO_ID") or (
        f"{hf_username}/visit-with-us-tourism" if hf_username else None
    )
    repo_type = "dataset"
    hf_token = require_env("HF_TOKEN")

    if not data_path.exists():
        raise FileNotFoundError(f"Raw data not found: {data_path}")

    if not dataset_repo_id:
        raise ValueError(
            "DATASET_REPO_ID or HF_USERNAME must be set to register data on Hugging Face."
        )

    # Use the Hub API directly instead of a git-based push so the same logic
    api = HfApi(token=hf_token)

    try:
        try:
            api.repo_info(repo_id=dataset_repo_id, repo_type=repo_type)
            print(f"Dataset repo '{dataset_repo_id}' already exists. Using it.")
        except RepositoryNotFoundError:
            print(f"Dataset repo '{dataset_repo_id}' not found. Creating it...")
            create_repo(
                repo_id=dataset_repo_id,
                repo_type=repo_type,
                private=False,
                exist_ok=True,
                token=hf_token,
            )
            print(f"Dataset repo '{dataset_repo_id}' created.")

        # Keep the filename stable in the dataset repo so downstream steps know exactly where to download the raw dataset from.
        api.upload_file(
            path_or_fileobj=str(data_path),
            path_in_repo=data_path.name,
            repo_id=dataset_repo_id,
            repo_type=repo_type,
        )
        print(f"Raw dataset uploaded to Hugging Face Dataset Hub: {dataset_repo_id}")
    except Exception as exc:
        raise RuntimeError(
            "Unable to upload the raw dataset to Hugging Face Dataset Hub."
        ) from exc


if __name__ == "__main__":
    main()
