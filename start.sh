#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  HR AI Assistant — One-Command Startup Script (Linux/Mac)
# ═══════════════════════════════════════════════════════════════════════════

set -e

echo ""
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║            HR AI Assistant — AlNoor Technologies            ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if Ollama is running
echo "[1/4] Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "      ⚠ Ollama is not running. Please start it with: ollama serve"
    exit 1
fi
echo "      ✓ Ollama is ready"

# Check if model is available
echo "[2/4] Checking LLM model..."
if ! ollama list 2>/dev/null | grep -qi "llama3.1:8b"; then
    echo "      ⚠ Model not found. Pulling llama3.1:8b (this may take a while)..."
    ollama pull llama3.1:8b
fi
echo "      ✓ Model ready"

# Install Python dependencies
echo "[3/4] Installing Python dependencies..."
if [ ! -d ".venv" ]; then
    python3 -m uv venv .venv
fi
python3 -m uv pip install -r requirements.txt -q
echo "      ✓ Dependencies installed"

# Start the server
echo "[4/4] Starting the HR AI Assistant..."
echo ""
echo "  ┌────────────────────────────────────────────────────────────┐"
echo "  │  API:      http://localhost:8000                          │"
echo "  │  Docs:     http://localhost:8000/docs                     │"
echo "  │  Health:   http://localhost:8000/health                   │"
echo "  │  Frontend: cd frontend && npm run dev                     │"
echo "  └────────────────────────────────────────────────────────────┘"
echo ""

.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
