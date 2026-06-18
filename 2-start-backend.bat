@echo off
REM ============================================================
REM  DocChat RAG - Start the backend (FastAPI)
REM  Keep this window open while testing. Press Ctrl+C to stop.
REM ============================================================
cd /d "%~dp0\backend"
call .venv\Scripts\activate.bat
echo Starting backend on http://localhost:8000 ...
echo (Leave this window open. Press Ctrl+C to stop.)
echo.
uvicorn app.main:app --reload --port 8000
pause
