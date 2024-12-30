import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

st.title("Insurance Prediction")
age = st.slider('Age', 1, 100)
sex = st.selectbox('Gender (male: 1,female: 0)',[0,1])
bmi = st.number_input('BMI')
children = st.number_input('Children')
smoker = st.selectbox('Smoker (Yes: 1 , No: 0)',[0,1])

input_data = [age, sex, bmi, children, smoker]


result=""
if st.button('Result'):
    input_data = np.asarray(input_data).reshape(1,-1)
    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)
    result = prediction[0]

st.success(result)
