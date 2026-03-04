import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🎓 College Admission Prediction System")

# User Inputs
gre = st.number_input("GRE Score", 260, 340)
toefl = st.number_input("TOEFL Score", 0, 120)
rating = st.slider("University Rating", 1, 5)
sop = st.slider("SOP Strength", 1.0, 5.0)
lor = st.slider("LOR Strength", 1.0, 5.0)
cgpa = st.number_input("CGPA", 0.0, 10.0)
research = st.selectbox("Research Experience", [0, 1])

# Prediction
if st.button("Predict Admission Chance"):
    input_data = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Chance of Admission: {round(prediction[0]*100, 2)}%")
