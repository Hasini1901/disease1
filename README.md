# disease1
diabetes-predict-app/
│
├── app.py                   # Main Streamlit app
├── diabetes_model.pkl       # Trained machine learning model (upload this)
└── requirements.txt         # Dependencies

import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("Transforming Healthcare with AI")
st.subheader("AI-powered Disease Prediction: Diabetes Risk")
st.markdown("### Enter Patient Data:")

# Input fields
preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 0, 120)

if st.button("Predict"):
    features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(features)
    st.success("Positive for Diabetes" if prediction[0] == 1 else "Not Diabetic")

streamlit
numpy
scikit-learn
