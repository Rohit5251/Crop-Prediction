import streamlit as st
import numpy as np
import pandas as pd
import base64
import pickle
from sklearn.tree import DecisionTreeClassifier
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Function to load and encode the image in base64
def get_base64_of_bin_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Provide the path to your local image
image_file_path = "cropimg.jpg" 
base64_img = get_base64_of_bin_file(image_file_path)

# CSS for setting the background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{base64_img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""

st.sidebar.header("About Krishi Disha")
st.sidebar.write("Krishi Disha is a user-friendly platform designed to help farmers make better decisions about which crops to grow. By considering factors like soil type, weather conditions, and nutrient levels, Krishi Disha suggests the most suitable crops for each situation. This helps farmers increase their productivity and grow healthier crops, ensuring a better yield and improved farming practices.")
st.markdown(page_bg_img, unsafe_allow_html=True)
# Set up Gemini API key
GEMINI_API_KEY = "Add your API key"
st.title('Krishi Disha : Crop Prediction Based on Soil and Weather Conditions')

# Brief description
st.markdown("""
    This application allows you to input agricultural data (soil and weather conditions),
    and predicts the best crop for the given conditions based on a machine learning model.
""")

# Load trained Decision Tree model
dt = pickle.load(open("crop_dt_model.pkl", "rb"))

soil = ['Black', 'Dark Brown', 'Light Brown', 'Medium Brown', 'Red', 'Red ', 'Reddish Brown']
district = ['Kolhapur', 'Pune', 'Sangli', 'Satara', 'Solapur']
crops = ['Cotton', 'Ginger', 'Gram', 'Grapes', 'Groundnut', 'Jowar', 'Maize', 'Masoor', 
         'Moong', 'Rice', 'Soybean', 'Sugarcane', 'Tur', 'Turmeric', 'Urad', 'Wheat']

# Form for user input
with st.form(key='crop_form'):
    district_name = st.selectbox('District Name', district)
    soil_color = st.selectbox('Soil Color', soil)
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
    
    # Display input data
    st.write(f'### Input Data:')
    st.write(f'**District:** {district_name}')
    st.write(f'**Soil Color:** {soil_color}')
    st.write(f'**Nitrogen Content:** {nitrogen} ppm')
    st.write(f'**Phosphorus Content:** {phosphorus} ppm')
    st.write(f'**Potassium Content:** {potassium} ppm')
    st.write(f'**pH Level:** {ph}')
    st.write(f'**Rainfall:** {rainfall} mm')
    st.write(f'**Temperature:** {temperature} °C')

    # Model Prediction
    res = dt.predict([[district.index(district_name), soil.index(soil_color),
                      nitrogen, phosphorus, potassium, ph, rainfall, temperature]])
    predicted_crop = crops[res[0]]
    
    st.markdown(f"### PREDICTION: 🌱 {predicted_crop}")

    # --- LangChain Explanation using Gemini ---
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

    prompt_template = PromptTemplate(
        input_variables=["district", "soil", "nitrogen", "phosphorus", "potassium", "ph", "rainfall", "temperature", "crop"],
        template="""
        Based on the given soil and weather conditions, summarize why {crop} is the best-suited crop.

        **Soil and Climate Conditions:**
        - **District:** {district}
        - **Soil Color:** {soil}
        - **Nitrogen:** {nitrogen} ppm
        - **Phosphorus:** {phosphorus} ppm
        - **Potassium:** {potassium} ppm
        - **pH Level:** {ph}
        - **Rainfall:** {rainfall} mm
        - **Temperature:** {temperature} °C

        Explain in a concise and informative manner.
        """
    )

    query = prompt_template.format(
        district=district_name, soil=soil_color, nitrogen=nitrogen,
        phosphorus=phosphorus, potassium=potassium, ph=ph,
        rainfall=rainfall, temperature=temperature, crop=predicted_crop
    )

    explanation = llm.predict(query)

    with st.expander("See the Explanation :", expanded=False):  
        st.markdown("### 🌾 Why this Crop?")
        st.write(explanation)
