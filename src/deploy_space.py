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
    space_repo_id = os.getenv("SPACE_REPO_ID") or (
        f"{hf_username}/visit-with-us-tourism-space" if hf_username else None
    )

    if not hf_token:
        raise ValueError("HF_TOKEN must be set in .env")
    if not space_repo_id:
        raise ValueError("SPACE_REPO_ID or HF_USERNAME must be set in .env")

    if not deployment_dir.exists():
        raise FileNotFoundError(f"Deployment directory not found: {deployment_dir}")

    # Validate the deployment bundle before pushing so missing files fail fast locally instead of surfacing later as a broken Hugging Face Space.
    required_files = ["Dockerfile", "README.md", "app.py", "requirements.txt"]
    missing_files = [
        filename for filename in required_files
        if not (deployment_dir / filename).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing deployment files: {', '.join(sorted(missing_files))}"
        )

    create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=hf_token,
    )

    api = HfApi(token=hf_token)
    # Upload the whole folder so the Space always receives a self-contained deployment package with matching app, config, and dependency files.
    api.upload_folder(
        folder_path=str(deployment_dir),
        repo_id=space_repo_id,
        repo_type="space",
    )

    print(f"Deployment files uploaded to Hugging Face Space: {space_repo_id}")


if __name__ == "__main__":
    main()
