@echo off
REM ═══════════════════════════════════════════════════════════════════════════
REM  HR AI Assistant — One-Command Startup Script (Windows)
REM ═══════════════════════════════════════════════════════════════════════════

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║            HR AI Assistant — AlNoor Technologies            ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if Ollama is running
echo [1/4] Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo       ⚠ Ollama is not running. Starting Ollama...
    start /B ollama serve >nul 2>&1
    timeout /t 3 /nobreak >nul
)
echo       ✓ Ollama is ready

REM Check if model is available
echo [2/4] Checking LLM model...
ollama list 2>nul | findstr /i "llama3.1:8b" >nul
if %errorlevel% neq 0 (
    echo       ⚠ Model not found. Pulling llama3.1:8b (this may take a while)...
    ollama pull llama3.1:8b
)
echo       ✓ Model ready

REM Install Python dependencies
echo [3/4] Installing Python dependencies...
if not exist ".venv" (
    python -m uv venv .venv
)
python -m uv pip install -r requirements.txt -q
echo       ✓ Dependencies installed

REM Start the server
echo [4/4] Starting the HR AI Assistant...
echo.
echo  ┌────────────────────────────────────────────────────────────┐
echo  │  API:      http://localhost:8000                          │
echo  │  Docs:     http://localhost:8000/docs                     │
echo  │  Health:   http://localhost:8000/health                   │
echo  │  Frontend: cd frontend ^&^& npm run dev                    │
echo  └────────────────────────────────────────────────────────────┘
echo.

.venv\Scripts\python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
