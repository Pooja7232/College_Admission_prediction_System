import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

st.title("🎓 College Admission Prediction System")

st.write("Enter student details to predict admission chances")

# Input Sliders
gre = st.slider("GRE Score", 260, 340, 300)
toefl = st.slider("TOEFL Score", 80, 120, 100)
rating = st.slider("University Rating", 1, 5, 3)
sop = st.slider("Statement of Purpose (SOP)", 1.0, 5.0, 3.0)
lor = st.slider("Letter of Recommendation (LOR)", 1.0, 5.0, 3.0)
cgpa = st.slider("CGPA", 6.0, 10.0, 8.0)
research = st.selectbox("Research Experience", [0, 1])

if st.button("Predict Admission Chance"):

    data = pd.DataFrame({
        "GRE Score":[gre],
        "TOEFL Score":[toefl],
        "University Rating":[rating],
        "SOP":[sop],
        "LOR":[lor],
        "CGPA":[cgpa],
        "Research":[research]
    })

    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(data)

    probability = float(result[0])

    st.subheader("Prediction Result")

    # Progress bar
    st.progress(probability)

    # Percentage output
    st.success(f"Chance of Admission: {probability*100:.2f}%")