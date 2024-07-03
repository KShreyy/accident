#!/bin/bash

# Step 1: Download the file
echo "Downloading file..."
curl -o your_file.pdf https://github.com/KShreyy/Accident_pred

# Step 2: Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Step 3: Run Streamlit application
echo "Running Streamlit application..."
streamlit run your_app.py

chmod +x run_pipeline.sh
