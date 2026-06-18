@echo off
REM ============================================================
REM  DocChat RAG - Start the frontend (Next.js)
REM  Keep this window open while testing. Press Ctrl+C to stop.
REM  Then open http://localhost:3000 in your browser.
REM ============================================================
cd /d "%~dp0\frontend"
echo Starting frontend on http://localhost:3000 ...
echo (Leave this window open. Press Ctrl+C to stop.)
echo.
call npm run dev
pause
