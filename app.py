{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0e851551-8143-4f58-9948-4c9e368f81a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "model = joblib.load('diabetes_model.joblib')\n",
    "scaler = joblib.load('scaler.joblib')\n",
    "\n",
    "st.title(\"Diabetes Prediction App\")\n",
    "\n",
    "st.write(\"Enter the patient details below:\")\n",
    "\n",
    "preg = st.number_input(\"Pregnancies\", 0)\n",
    "glucose = st.number_input(\"Glucose Level\", 0)\n",
    "bp = st.number_input(\"Blood Pressure\", 0)\n",
    "skin = st.number_input(\"Skin Thickness\", 0)\n",
    "insulin = st.number_input(\"Insulin\", 0)\n",
    "bmi = st.number_input(\"BMI\", 0.0)\n",
    "dpf = st.number_input(\"Diabetes Pedigree Function\", 0.0)\n",
    "age = st.number_input(\"Age\", 0)\n",
    "\n",
    "if st.button(\"Predict\"):\n",
    "    features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])\n",
    "    features = scaler.transform(features)\n",
    "    prediction = model.predict(features)[0]\n",
    "    \n",
    "    if prediction == 1:\n",
    "        st.error(\" The patient is likely **Diabetic**\")\n",
    "    else:\n",
    "        st.success(\" The patient is **Not Diabetic**\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
