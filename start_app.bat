@echo off
echo Starting Oil Spill Detection System...
echo ======================================

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Attempting to run with global python...
)

echo Starting Streamlit App...
streamlit run app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error encountered. Please check the output above.
    pause
)
