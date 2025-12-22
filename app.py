import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏥 Diabetes Prediction Assistant")
st.write("Enter the patient's diagnostic information below to check the risk of diabetes.")


col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)


if st.button("Predict Diagnosis"):
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                       'SkinThickness', 'Insulin', 'BMI',
                                       'DiabetesPedigreeFunction', 'Age'])

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("Results:")
    if prediction[0] == 1:
        st.error(f"High Risk: The patient is likely diabetic. (Confidence: {probability:.2%})")
    else:
        st.success(f"Low Risk: The patient is likely healthy. (Confidence: {1 - probability:.2%})")