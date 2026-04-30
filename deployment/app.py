import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


DEFAULT_MODEL_REPO_ID = "saranka85/visit-with-us-tourism-random-forest"
ARTIFACT_FILENAMES = [
    "model.joblib",
    "metrics.json",
    "best_params.json",
    "feature_schema.json",
    "model_metadata.json",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def resolve_artifact(filename: str, model_repo_id: str, hf_token: str | None) -> Path:
    local_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "random_forest_model"
        / filename
    )

    try:
        downloaded = hf_hub_download(
            repo_id=model_repo_id,
            filename=filename,
            repo_type="model",
            token=hf_token,
        )
        return Path(downloaded)
    except Exception:
        # Fallback to local artifacts so the app can still run during local development even when Hub access is unavailable.
        if local_path.exists():
            return local_path
        raise FileNotFoundError(f"Unable to load artifact: {filename}")


@st.cache_resource
def load_model_bundle():
    load_dotenv()
    model_repo_id = os.getenv("MODEL_REPO_ID", DEFAULT_MODEL_REPO_ID)
    hf_token = os.getenv("HF_TOKEN")

    # Download all deployment artifacts together so the app uses one coherent model package rather than mixing old local files with new remote ones.
    artifact_paths = {
        filename: resolve_artifact(filename, model_repo_id, hf_token)
        for filename in ARTIFACT_FILENAMES
    }

    model = joblib.load(artifact_paths["model.joblib"])
    metrics = load_json(artifact_paths["metrics.json"])
    params = load_json(artifact_paths["best_params.json"])
    feature_schema = load_json(artifact_paths["feature_schema.json"])
    model_metadata = load_json(artifact_paths["model_metadata.json"])

    return model, metrics, params, feature_schema, model_metadata, model_repo_id


def render_numeric_input(column: str, spec: dict):
    minimum = spec.get("min", 0)
    maximum = spec.get("max", minimum + 1)
    default = spec.get("default", minimum)

    # Integer-like fields should stay as integers in the UI because they represent counts, binary flags, or category codes.
    if spec.get("type") == "integer":
        return st.number_input(
            column,
            min_value=int(minimum),
            max_value=int(maximum),
            value=int(default),
            step=1,
        )

    return st.number_input(
        column,
        min_value=float(minimum),
        max_value=float(maximum),
        value=float(default),
    )


def build_input_dataframe(feature_schema: dict, feature_order: list[str]) -> tuple[bool, pd.DataFrame]:
    inputs: dict[str, object] = {}

    with st.form("prediction_form"):
        for column in feature_order:
            spec = feature_schema.get(column, {})
            field_type = spec.get("type")

            if field_type == "categorical":
                # The UI is generated from saved training metadata so it stays aligned with the model even if the feature list changes later.
                options = spec.get("categories", [])
                default_value = spec.get("default")
                default_index = options.index(default_value) if default_value in options else 0
                inputs[column] = st.selectbox(column, options, index=default_index)
            else:
                inputs[column] = render_numeric_input(column, spec)

        submitted = st.form_submit_button("Predict")

    input_df = pd.DataFrame([inputs], columns=feature_order)
    return submitted, input_df


st.set_page_config(page_title="Visit With Us Predictor", layout="centered")
st.title("Visit With Us Tourism Predictor")
st.write("Predict whether a customer will purchase the wellness tourism package.")

model, metrics, params, feature_schema, model_metadata, model_repo_id = load_model_bundle()
feature_order = model_metadata.get("feature_columns") or list(feature_schema.keys())

with st.expander("Model Details"):
    st.write(f"Model Hub Repo: `{model_repo_id}`")
    if model_metadata:
        # Show the compact metadata payload first because it includes metrics, feature order, and dataset lineage in one place.
        st.json(model_metadata)
    elif metrics or params:
        st.write("Metrics")
        st.json(metrics)
        st.write("Best Parameters")
        st.json(params)

submitted, input_df = build_input_dataframe(feature_schema, feature_order)
st.session_state["latest_input_df"] = input_df.copy()

if submitted:
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    st.subheader("Input Data")
    st.dataframe(input_df)

    st.subheader("Prediction")
    st.write("ProdTaken:", prediction)
    st.write("Purchase Probability:", round(probability, 4))

    if prediction == 1:
        st.success("This customer is likely to purchase the package.")
    else:
        st.warning("This customer is less likely to purchase the package.")
