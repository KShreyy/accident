<<<<<<< HEAD
import pandas as pd
import streamlit as st
import pickle
import numpy as np

# Load the trained model and pre-processing objects
with open('best_model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

with open('label_encoder_company.pkl', 'rb') as le_company_file:
    label_encoder_company = pickle.load(le_company_file)

with open('label_encoder_month.pkl', 'rb') as le_month_file:
    label_encoder_month = pickle.load(le_month_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title for the Streamlit app
st.title('Accident Prediction App')

# Description
st.write('Predict the number of accidents for a company in a given month.')

# Load and clean the dataset for UI
file_path = 'Industry_accident_dataset.csv'  # Replace with your file path
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
unique_companies = df['Company'].unique()
unique_months = df['Month'].unique()

# Select a company from the dropdown
selected_company = st.selectbox('Select Company:', unique_companies)
encoded_company = label_encoder_company.transform([selected_company])[0]

# Select a month from the dropdown
selected_month = st.selectbox('Select Month:', unique_months)
encoded_month = label_encoder_month.transform([selected_month])[0]

# Preprocess the input data
encoded_features = scaler.transform([[encoded_company, encoded_month]])

# Predict the number of accidents
predicted_accidents = best_model.predict(encoded_features)[0]

# Display the prediction
st.write(f'Predicted number of accidents for **{selected_company}** in **{selected_month}**: **{predicted_accidents:.2f}**')

# Plot actual vs predicted values
# Load the dataset for visualization
df['Accident Count'] = 1
df_grouped = df.groupby(['Company', 'Month']).size().reset_index(name='Accident Count')

df_grouped['Company'] = label_encoder_company.transform(df_grouped['Company'])
df_grouped['Month'] = label_encoder_month.transform(df_grouped['Month'])
df_grouped[['Company', 'Month']] = scaler.transform(df_grouped[['Company', 'Month']])

X = df_grouped[['Company', 'Month']]
y = df_grouped['Accident Count']

y_pred = best_model.predict(X)

# Create a DataFrame for actual vs predicted values
actual_vs_predicted = pd.DataFrame({'Actual': y, 'Predicted': y_pred})

# Display the line chart
st.write('Actual vs Predicted Number of Accidents')
st.line_chart(actual_vs_predicted)

# add an input form for predictions
st.header('Predict Accidents for a Custom Input')
custom_company = st.text_input('Enter Company Name:')
custom_month = st.text_input('Enter Month:')
if custom_company and custom_month:
    if custom_company in unique_companies and custom_month in unique_months:
        custom_encoded_company = label_encoder_company.transform([custom_company])[0]
        custom_encoded_month = label_encoder_month.transform([custom_month])[0]
        custom_encoded_features = scaler.transform([[custom_encoded_company, custom_encoded_month]])
        custom_predicted_accidents = best_model.predict(custom_encoded_features)[0]
        st.write(f'Predicted number of accidents for **{custom_company}** in **{custom_month}**: **{custom_predicted_accidents:.2f}**')
    else:
        st.write('Please enter valid Company and Month names.')

# moe info
st.sidebar.header('Additional Information')
st.sidebar.write('This app uses a machine learning model to predict the number of accidents for a given company and month based on historical data.')

st.sidebar.header('Files')
st.sidebar.write('- best_model.pkl: Trained machine learning model.')
st.sidebar.write('- label_encoder_company.pkl: Label encoder for company names.')
st.sidebar.write('- label_encoder_month.pkl: Label encoder for months.')
st.sidebar.write('- scaler.pkl: Scaler for feature normalization.')
=======
import pandas as pd
import streamlit as st
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error  # Import MSE function

# Load the trained model and pre-processing objects
with open('best_model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

with open('label_encoder_company.pkl', 'rb') as le_company_file:
    label_encoder_company = pickle.load(le_company_file)

with open('label_encoder_month.pkl', 'rb') as le_month_file:
    label_encoder_month = pickle.load(le_month_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title for the Streamlit app
st.title('Accident Prediction App')

# Description
st.write('Predict the number of accidents for a company in a given month.')

# Load and clean the dataset for UI
file_path = 'Industry_accident_dataset.csv'  # Replace with your file path
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
unique_companies = df['Company'].unique()
unique_months = df['Month'].unique()

# Select a company from the dropdown
selected_company = st.selectbox('Select Company:', unique_companies)
encoded_company = label_encoder_company.transform([selected_company])[0]

# Select a month from the dropdown
selected_month = st.selectbox('Select Month:', unique_months)
encoded_month = label_encoder_month.transform([selected_month])[0]

# Preprocess the input data
encoded_features = scaler.transform([[encoded_company, encoded_month]])

# Predict the number of accidents
predicted_accidents = best_model.predict(encoded_features)[0]

# Display the prediction
st.write(f'Predicted number of accidents for **{selected_company}** in **{selected_month}**: **{predicted_accidents:.2f}**')

# Plot actual vs predicted values
# Load the dataset for visualization
df['Accident Count'] = 1
df_grouped = df.groupby(['Company', 'Month']).size().reset_index(name='Accident Count')

df_grouped['Company'] = label_encoder_company.transform(df_grouped['Company'])
df_grouped['Month'] = label_encoder_month.transform(df_grouped['Month'])
df_grouped[['Company', 'Month']] = scaler.transform(df_grouped[['Company', 'Month']])

X = df_grouped[['Company', 'Month']]
y = df_grouped['Accident Count']

y_pred = best_model.predict(X)

# Create a DataFrame for actual vs predicted values
actual_vs_predicted = pd.DataFrame({'Actual': y, 'Predicted': y_pred})

# Calculate MSE
mse = mean_squared_error(y, y_pred)

# Create a DataFrame for actual vs predicted values
actual_vs_predicted = pd.DataFrame({'Actual': y, 'Predicted': y_pred})

# Display the MSE
st.write(f'Mean Squared Error (MSE): **{mse:.2f}**')

# Display the line chart
st.write('Actual vs Predicted Number of Accidents')
st.line_chart(actual_vs_predicted)

# add an input form for predictions
st.header('Predict Accidents for a Custom Input')
custom_company = st.text_input('Enter Company Name:')
custom_month = st.text_input('Enter Month:')
if custom_company and custom_month:
    if custom_company in unique_companies and custom_month in unique_months:
        custom_encoded_company = label_encoder_company.transform([custom_company])[0]
        custom_encoded_month = label_encoder_month.transform([custom_month])[0]
        custom_encoded_features = scaler.transform([[custom_encoded_company, custom_encoded_month]])
        custom_predicted_accidents = best_model.predict(custom_encoded_features)[0]
        st.write(f'Predicted number of accidents for **{custom_company}** in **{custom_month}**: **{custom_predicted_accidents:.2f}**')
    else:
        st.write('Please enter valid Company and Month names.')

# moe info
st.sidebar.header('Additional Information')
st.sidebar.write('This app uses a machine learning model to predict the number of accidents for a given company and month based on historical data.')

st.sidebar.header('Files')
st.sidebar.write('- best_model.pkl: Trained machine learning model.')
st.sidebar.write('- label_encoder_company.pkl: Label encoder for company names.')
st.sidebar.write('- label_encoder_month.pkl: Label encoder for months.')
st.sidebar.write('- scaler.pkl: Scaler for feature normalization.')
>>>>>>> 0d6257e (Initial commit)
