---
title: Visit With Us Tourism Predictor
sdk: docker
app_port: 8501
---

# Visit With Us Tourism Predictor

Streamlit app for predicting whether a customer will purchase the wellness tourism package.

## Runtime Notes

- The Space uses the `Dockerfile` in this folder.
- The app downloads `model.joblib` and metadata artifacts from the Hugging Face Model Hub.
- Optional environment variables: `MODEL_REPO_ID` and `HF_TOKEN`.
