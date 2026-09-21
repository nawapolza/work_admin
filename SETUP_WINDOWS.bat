@echo off
setlocal
cd /d "%~dp0"
echo [1/4] Checking Python...
python --version || goto :python_error
echo [2/4] Creating project venv...
python -m venv venv || goto :error
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo [3/4] Installing Flask + TensorFlow/Keras...
pip install -r backend\requirements.txt || goto :error
echo [4/4] Installing React frontend...
cd frontend
call npm install || goto :error
cd ..
if not exist .env copy .env.example .env
echo.
echo Setup complete. Put both .h5 models in backend\models then run RUN_WINDOWS.bat
pause
exit /b 0
:python_error
echo Python not found. Install Python 3.10-3.12 and select Add Python to PATH.
pause
exit /b 1
:error
echo Setup failed. Read README.md for troubleshooting.
pause
exit /b 1

