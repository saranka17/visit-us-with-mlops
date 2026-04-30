import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


DEFAULT_MODEL_REPO_ID = "saranka85/visit-with-us-tourism-random-forest"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def resolve_artifact(filename: str, model_repo_id: str, hf_token: str | None) -> Path:
    local_path = Path(__file__).resolve().parents[1] / "artifacts" / "random_forest_model" / filename

    try:
        downloaded = hf_hub_download(
            repo_id=model_repo_id,
            filename=filename,
            repo_type="model",
            token=hf_token,
        )
        return Path(downloaded)
    except Exception:
        if local_path.exists():
            return local_path
        raise FileNotFoundError(f"Unable to load artifact: {filename}")


@st.cache_resource
def load_model_bundle():
    load_dotenv()
    model_repo_id = os.getenv("MODEL_REPO_ID", DEFAULT_MODEL_REPO_ID)
    hf_token = os.getenv("HF_TOKEN")

    model_path = resolve_artifact("model.joblib", model_repo_id, hf_token)
    metrics_path = resolve_artifact("metrics.json", model_repo_id, hf_token)
    params_path = resolve_artifact("best_params.json", model_repo_id, hf_token)

    model = joblib.load(model_path)
    metrics = load_json(metrics_path)
    params = load_json(params_path)

    return model, metrics, params, model_repo_id


def build_input_dataframe() -> pd.DataFrame:
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=80, value=35)
        typeof_contact = st.selectbox("TypeofContact", ["Company Invited", "Self Enquiry"])
        city_tier = st.selectbox("CityTier", [1, 2, 3])
        duration_of_pitch = st.number_input("DurationOfPitch", min_value=1, max_value=150, value=15)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        number_of_person_visiting = st.selectbox("NumberOfPersonVisiting", [1, 2, 3, 4, 5])
        number_of_followups = st.selectbox("NumberOfFollowups", [1, 2, 3, 4, 5, 6])
        product_pitched = st.selectbox("ProductPitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        preferred_property_star = st.selectbox("PreferredPropertyStar", [3, 4, 5])
        marital_status = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
        number_of_trips = st.number_input("NumberOfTrips", min_value=1, max_value=25, value=3)
        passport = st.selectbox("Passport", [0, 1])
        pitch_satisfaction_score = st.selectbox("PitchSatisfactionScore", [1, 2, 3, 4, 5])
        own_car = st.selectbox("OwnCar", [0, 1])
        number_of_children_visiting = st.selectbox("NumberOfChildrenVisiting", [0, 1, 2, 3])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("MonthlyIncome", min_value=1000, max_value=100000, value=22000)

        submitted = st.form_submit_button("Predict")

    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    return submitted, input_df


st.set_page_config(page_title="Visit With Us Predictor", layout="centered")
st.title("Visit With Us Tourism Predictor")
st.write("Predict whether a customer will purchase the wellness tourism package.")

model, metrics, params, model_repo_id = load_model_bundle()

with st.expander("Model Details"):
    st.write(f"Model Hub Repo: `{model_repo_id}`")
    if metrics:
        st.json(metrics)
    if params:
        st.json(params)

submitted, input_df = build_input_dataframe()

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
