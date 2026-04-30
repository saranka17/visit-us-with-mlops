"""Push deployment files to a Hugging Face Space."""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    deployment_dir = project_root / "deployment"

    load_dotenv(project_root / ".env")

    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    space_repo_id = os.getenv("SPACE_REPO_ID") or f"{hf_username}/visit-with-us-tourism-space"

    if not hf_token:
        raise ValueError("HF_TOKEN must be set in .env")

    if not deployment_dir.exists():
        raise FileNotFoundError(f"Deployment directory not found: {deployment_dir}")

    create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=hf_token,
    )

    api = HfApi(token=hf_token)
    api.upload_folder(
        folder_path=str(deployment_dir),
        repo_id=space_repo_id,
        repo_type="space",
    )

    print(f"Deployment files uploaded to Hugging Face Space: {space_repo_id}")


if __name__ == "__main__":
    main()
