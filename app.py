import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🏥")

@st.cache_resource
def load():
    model     = joblib.load("models/best_model.pkl")
    le_sex    = joblib.load("models/le_sex.pkl")
    le_smoker = joblib.load("models/le_smoker.pkl")
    scaler    = joblib.load("models/scaler.pkl")
    return model, le_sex, le_smoker, scaler

model, le_sex, le_smoker, scaler = load()

st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter patient details to estimate annual insurance charges.")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    age      = st.slider("Age", 18, 64, 30)
    sex      = st.selectbox("Sex", ["male", "female"])
    bmi      = st.number_input("BMI", 10.0, 60.0, 28.0, step=0.1)
with col2:
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
    smoker   = st.selectbox("Smoker", ["no", "yes"])
    region   = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict Insurance Cost", type="primary", use_container_width=True):
    d = pd.DataFrame([{
        "age": float(age), "sex": sex, "bmi": float(bmi),
        "children": float(children), "smoker": smoker, "region": region
    }])
    d["sex"]    = le_sex.transform(d["sex"])
    d["smoker"] = le_smoker.transform(d["smoker"])
    d = pd.get_dummies(d, columns=["region"], drop_first=True)
    for col in ["region_northwest", "region_southeast", "region_southwest"]:
        if col not in d.columns:
            d[col] = 0
    d[["age", "bmi", "children"]] = scaler.transform(d[["age", "bmi", "children"]])
    X = d[["age", "sex", "bmi", "children", "smoker",
           "region_northwest", "region_southeast", "region_southwest"]].astype(float)

    pred = np.expm1(model.predict(X)[0])

    st.markdown("---")
    st.metric("Estimated Annual Insurance Cost", f"${pred:,.2f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Smoker",       "Yes 🔴" if smoker == "yes" else "No 🟢")
    c2.metric("BMI Category", "Obese 🔴" if bmi >= 30 else "Normal 🟢")
    c3.metric("Age",          f"{age} yrs")

st.caption("Medical Insurance Cost Predictor | Supervised ML Mini Project")
