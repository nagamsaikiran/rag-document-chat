@echo off
REM ============================================================
REM  DocChat RAG - One-time setup
REM  Double-click this once from the project root.
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo === Checking prerequisites ===
where python >nul 2>nul || (echo [ERROR] Python not found. Install from python.org ^(check "Add to PATH"^). & pause & exit /b 1)
where node   >nul 2>nul || (echo [ERROR] Node not found. Install from nodejs.org. & pause & exit /b 1)
python --version
node --version

echo.
echo === Backend: creating virtual environment ===
cd backend
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat

echo === Backend: installing Python dependencies ===
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist .env (
    copy .env.example .env >nul
    echo.
    echo [ACTION NEEDED] Created backend\.env
    echo    Open it and paste your free GEMINI_API_KEY, then save.
    echo    Get one at https://aistudio.google.com/apikey
)
cd ..

echo.
echo === Frontend: installing Node dependencies ===
cd frontend
if not exist .env.local (
    copy .env.local.example .env.local >nul
)
call npm install
cd ..

echo.
echo ============================================================
echo  Setup complete!
echo  1. Open backend\.env and add your GEMINI_API_KEY (if not done)
echo  2. Double-click 2-start-backend.bat
echo  3. Double-click 3-start-frontend.bat
echo  4. Open http://localhost:3000 in your browser
echo ============================================================
echo.
pause
