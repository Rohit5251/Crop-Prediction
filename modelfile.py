import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

st.title('Crop Prediction Based on Soil and Weather Conditions')

# Brief description
st.markdown("""
    This application allows you to input agricultural data (soil and weather conditions),
    and predicts the best crop for the given conditions based on a machine learning model.
    """)

dt=pickle.load(open("crop_dt_model.pkl","rb"))

soil=['Black', 'Dark Brown', 'Light Brown', 'Medium Brown', 'Red', 'Red ', 'Reddish Brown']
district= ['Kolhapur', 'Pune' ,'Sangli', 'Satara', 'Solapur']
crops= ['Cotton' , 'Ginger' ,'Gram', 'Grapes', 'Groundnut', 'Jowar', 'Maize' ,'Masoor' ,'Moong', 'Rice', 'Soybean' ,'Sugarcane', 'Tur' ,'Turmeric', 'Urad', 'Wheat']
# Form for user input
with st.form(key='crop_form'):
    district_name = st.selectbox('District Name', ['Kolhapur' ,'Pune', 'Sangli', 'Satara', 'Solapur'])
    soil_color = st.selectbox('Soil Color', ['Black', 'Dark Brown', 'Light Brown', 'Medium Brown', 'Red', 'Red ', 'Reddish Brown'])
    nitrogen = st.slider('Nitrogen Content (ppm)', 0, 100, 10)
    phosphorus = st.slider('Phosphorus Content (ppm)', 0, 100, 10)
    potassium = st.slider('Potassium Content (ppm)', 0, 100, 10)
    ph = st.slider('pH Level', 0.0, 14.0, 7.0)
    rainfall = st.slider('Rainfall (mm)', 0, 2000, 500)
    temperature = st.slider('Temperature (°C)', -10, 50, 25)

    # Submit button
    submit_button = st.form_submit_button(label='Predict Crop')

if submit_button:
    st.success("Data uploaded Successfully")
    # Display the input data
    st.write(f'### District: {district_name}')
    st.write(f'**Soil Color:** {soil_color}')
    st.write(f'**Nitrogen Content:** {nitrogen} ppm')
    st.write(f'**Phosphorus Content:** {phosphorus} ppm')
    st.write(f'**Potassium Content:** {potassium} ppm')
    st.write(f'**pH Level:** {ph}')
    st.write(f'**Rainfall:** {rainfall} mm')
    st.write(f'**Temperature:** {temperature} °C')

    res=dt.predict([[district.index(district_name), soil.index(soil_color),nitrogen,phosphorus,potassium,ph,rainfall,temperature]])
    st.markdown(f"**PREDICTION:** {crops[res[0]]}")






