import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title(" Diabetes Prediction App")
st.write("Enter the patient details below:")

preg = st.number_input("Pregnancies", 0)
glucose = st.number_input("Glucose Level", 0)
bp = st.number_input("Blood Pressure", 0)
skin = st.number_input("Skin Thickness", 0)
insulin = st.number_input("Insulin", 0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 0)

if st.button("Predict"):
    features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    features = scaler.transform(features)
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error(" The patient is likely Diabetic")
    else:
        st.success("The patient is Not Diabetic")
