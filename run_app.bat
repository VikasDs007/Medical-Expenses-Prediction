@echo off
echo Starting Medical Insurance Cost Prediction App...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install requirements if needed
if not exist "venv\Lib\site-packages\streamlit" (
    echo Installing requirements...
    pip install -r requirements.txt
)

REM Train models if they don't exist
if not exist "models\linear_regression.pkl" (
    echo Training models...
    python train_models.py
)

REM Start the Streamlit app
echo.
echo Starting Streamlit app...
echo The app will be available at: http://localhost:8501
echo Press Ctrl+C to stop the app
echo.
streamlit run app/main.py --server.port 8501

pause