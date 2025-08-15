#!/bin/bash

echo "Starting Medical Insurance Cost Prediction App..."
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/lib/python*/site-packages/streamlit" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Train models if they don't exist
if [ ! -f "models/linear_regression.pkl" ]; then
    echo "Training models..."
    python train_models.py
fi

# Start the Streamlit app
echo
echo "Starting Streamlit app..."
echo "The app will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the app"
echo

streamlit run app/main.py --server.port 8501