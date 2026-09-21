@echo off
setlocal
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
  echo venv not found. Run SETUP_WINDOWS.bat first.
  pause
  exit /b 1
)
call venv\Scripts\activate.bat
start "Mango AI Backend" cmd /k "python backend\app.py"
timeout /t 3 /nobreak >nul
start "Mango AI Frontend" cmd /k "cd frontend && npm run dev"
echo.
echo Starting Mango AI Assistant...
echo Look at the "Mango AI Frontend" window and open the Local URL shown by Vite.
echo Default: http://localhost:5173
echo If 5173 is busy, Vite will automatically use 5174, 5175, and so on.
echo Do not run RUN_WINDOWS.bat twice. You can reuse the browser URL already open.
pause
