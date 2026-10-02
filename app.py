import streamlit as st
import joblib
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import base64
import subprocess

# Function to convert a file to base64
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-position: center;
        background-size: cover;
        font-family: "Times New Roman", serif;
    }}
    h1, h2, h3, p {{
        font-family: "Times New Roman", serif;
        color: black;  /* Set text color to white */
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image and text color
set_background('Background/2.webp')

# Load the pre-trained Random Forest model
loaded_rf_model = joblib.load('random_forest_model.pkl')

# Define the feature names (same as in your training data)
feature_names = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity',
                 'Previous_cgpa', 'Access_to_Resources_Encoded', 'Motivation_Level_Encoded', 'Family_Income_Encoded',
                 'Teacher_Quality_Encoded', 'School_Type_Encoded', 'Peer_Influence_Encoded',
                 'Parental_Education_Level_Encoded', 'Distance_from_Home_Encoded', 'Gender_Encoded',
                 'Parental_Involvement_Encoded', 'Extracurricular_Activities_Encoded', 'Internet_Access_Encoded',
                 'Learning_Disabilities_Encoded']

# Streamlit UI components for user input
st.title('Predicting University Student Graduation Using Academic Performance and Machine Learning A Systematic Literature Review')
st.write("Enter the features of the student to predict their CGPA category:")

# Collect inputs from the user
hours_studied = st.slider('Hours Studied', 0, 24, 5)
attendance = st.slider('Attendance (%)', 0, 100, 24)
sleep_hours = st.slider('Sleep Hours', 0, 24, 10)
tutoring_sessions = st.selectbox('Tutoring Sessions', [0, 1])
physical_activity = st.selectbox('Physical Activity (0 = No, 1 = Yes)', [0, 1])
previous_cgpa = st.slider('Previous CGPA (0-4)', 0.0, 4.0, 1.0)
access_to_resources = st.selectbox('Access to Resources', [0, 1])
motivation_level = st.selectbox('Motivation Level (0 = Low, 1 = High)', [0, 1])
family_income = st.selectbox('Family Income', [0, 1])
teacher_quality = st.selectbox('Teacher Quality', [0, 1])
school_type = st.selectbox('School Type', [0, 1])
peer_influence = st.selectbox('Peer Influence', [0, 1])
parental_education_level = st.selectbox('Parental Education Level', [0, 1])
distance_from_home = st.selectbox('Distance from Home', [0, 1])
gender = st.selectbox('Gender (0 = Male, 1 = Female)', [0, 1])
parental_involvement = st.selectbox('Parental Involvement', [0, 1])
extracurricular_activities = st.selectbox('Extracurricular Activities', [0, 1])
internet_access = st.selectbox('Internet Access', [0, 1])
learning_disabilities = st.selectbox('Learning Disabilities', [0, 1])

# Store user input in a DataFrame
input_data = pd.DataFrame([[hours_studied, attendance, sleep_hours, tutoring_sessions, physical_activity, previous_cgpa,
                            access_to_resources, motivation_level, family_income, teacher_quality, school_type, peer_influence,
                            parental_education_level, distance_from_home, gender, parental_involvement, extracurricular_activities,
                            internet_access, learning_disabilities]], columns=feature_names)

# Predict the CGPA category using the loaded model
prediction = loaded_rf_model.predict(input_data)

# Display the result
if prediction[0] == 4:
    prediction_text = "Excellent"
elif prediction[0] == 3:
    prediction_text = "Good"
elif prediction[0] == 2:
    prediction_text = "Average"
elif prediction[0] == 1:
    prediction_text = "Below Average"
else:
    prediction_text = "Poor"

st.write(f"Predicted CGPA Category: {prediction_text}")
