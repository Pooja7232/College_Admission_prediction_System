import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="College Admission Prediction", layout="centered")

st.title("🎓 College Admission Prediction System")
st.write("Enter student details to predict admission probability.")

# ----------------------------
# Load Model and Scaler Safely
# ----------------------------

@st.cache_resource
def load_files():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler
    except:
        return None, None

model, scaler = load_files()

if model is None or scaler is None:
    st.error("❌ Model or Scaler file not found. Please check deployment files.")
    st.stop()

# ----------------------------
# User Inputs
# ----------------------------

gre = st.number_input("GRE Score", min_value=260, max_value=340, value=300)
toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
rating = st.slider("University Rating", 1, 5, 3)
sop = st.slider("SOP Strength", 1.0, 5.0, 3.0)
lor = st.slider("LOR Strength", 1.0, 5.0, 3.0)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0)
research = st.selectbox("Research Experience", [0, 1])

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Admission Chance"):
    try:
        input_data = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        percentage = round(prediction * 100, 2)

        if percentage >= 75:
            st.success(f"🎉 High Chance of Admission: {percentage}%")
        elif percentage >= 50:
            st.warning(f"🙂 Moderate Chance of Admission: {percentage}%")
        else:
            st.error(f"⚠ Low Chance of Admission: {percentage}%")

    except Exception as e:
        st.error("Prediction failed. Please check model compatibility.")
