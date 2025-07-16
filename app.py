import streamlit as st
import numpy as np
import pandas as pd
import joblib

# load out saved components

model=joblib.load('gradient_boosting_model.pkl')
label_encoder=joblib.load('label_encoder.pkl')
scaler=joblib.load('scaler.pkl')


st.title('Glass Type Prediction App')
st.write(' This model predict the glass type according to the input features below .Please enter your value')

# Features you outline here should math the features used in the training of the model

#Input Features
RI=st.slider('Refractive index:',1.5,1.8)
Na = st.slider('Sodium:',10.0,18.0)
Mg = st.slider('Magnesium:',0.0,4.0)
Al = st.slider('Aluminium:',0.0,4.0)
Si = st.slider('Silicon:',70.0,80.0)
K = st.slider('Potassium:',0.0,0.5)
Ca = st.slider('Calcium:',5.0,10.0)
Ba = st.slider('Barium:',0.0,5.0)
Fe = st.slider('Iron:',0.0,5.0)

#Preparing Imput Features for model 
features=np.array([[RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]])
scaled_features=scaler.transform(features)

#Prediction
if st.button('Predict Glass Type'):
    prediction_encoded=model.predict(scaled_features)
    prediction_label=label_encoder.inverse_transform(prediction_encoded)[0]
    st.success(f'Predict glass type :{prediction_label}')



