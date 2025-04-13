import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="UCLA Admission Predictor", layout="centered")

# --- Header ---
st.markdown("<h2 style='text-align: center; color: #2c3e50;'>🎓 UCLA Admission Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict your chance of graduate admission at UCLA based on academic factors.</p>", unsafe_allow_html=True)
st.divider()

# --- Input Form ---
with st.form("admission_form"):
    st.markdown("### 📋 Applicant Details")
    col1, col2 = st.columns(2)

    with col1:
        gre = st.number_input("GRE Score", min_value=260, max_value=340, value=320, help="Enter your GRE score (out of 340)")
        toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110, help="Enter your TOEFL score (out of 120)")
        sop = st.slider("SOP Strength", min_value=1.0, max_value=5.0, value=3.5, step=0.5, help="Statement of Purpose strength (1 to 5)")
        lor = st.slider("LOR Strength", min_value=1.0, max_value=5.0, value=3.5, step=0.5, help="Letter of Recommendation strength (1 to 5)")
        
    with col2:
        univ_rating = st.selectbox("University Rating", options=[1, 2, 3, 4, 5], index=2, help="How prestigious is your target university?")
        cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.5, help="Your GPA on a 10-point scale")
        research = st.radio("Research Experience", options=["No", "Yes"], index=1, help="Have you done research?")

    submitted = st.form_submit_button("🎯 Predict Admission Chance")

# --- Prediction Section ---
if submitted:
    research_val = 1 if research == "Yes" else 0
    input_data = pd.DataFrame([{
        "GRE_Score": gre,
        "TOEFL_Score": toefl,
        "University_Rating": univ_rating,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": research_val
    }])

    # Load training data for scaling
    base_df = pd.read_csv("data/Admission.csv").drop(columns=["Serial_No", "Admit_Chance"])
    scaler = StandardScaler()
    scaler.fit(base_df)
    input_scaled = scaler.transform(input_data)

    # Load pre-trained model and predict
    model = load_model("models/admission_model.h5")
    prediction = model.predict(input_scaled)[0][0]

    # Display result
    st.divider()
    st.markdown("### 🧾 Prediction Result")
    if prediction > 0.7:
        st.success("🎉 You are **likely** to be admitted!")
    else:
        st.error("⚠️ You are **unlikely** to be admitted.")
    st.markdown(f"<h4 style='text-align:center;'>Predicted Admission Chance: <span style='color:#27ae60;'>{prediction:.2f}</span></h4>", unsafe_allow_html=True)
    st.progress(min(int(prediction * 100), 100))

    # Explanation
    st.divider()
    st.markdown("### 🧠 Model Insights")
    st.markdown("We used a trained neural network model to predict your chance of admission based on historical data.")
    st.image("feature_importance_plot.png", caption="Feature Importance Chart", use_container_width=True)
