"""
Configuration and LLM factory for the HR AI Assistant.

Supports two LLM backends:
  1. Google Gemini free tier (requires GEMINI_API_KEY env var)
  2. Ollama local models (requires Ollama running locally)

The system auto-detects which backend to use based on available config.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
POLICIES_DIR = DATA_DIR / "policies"
EMPLOYEES_CSV = DATA_DIR / "employees.csv"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ── Embedding model (local, free) ────────────────────────────────────────────
# E5 uses instruction-prefixed embeddings ("query: ..." / "passage: ...")
# which significantly improves retrieval quality over plain models like MiniLM.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
CHROMA_COLLECTION_NAME = "hr_policies"

# ── RAG settings ─────────────────────────────────────────────────────────────
RAG_TOP_K = 3

# ── LLM settings ────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")


def get_llm():
    """
    Factory function that returns the best available LLM.

    Priority:
      1. Gemini (if GEMINI_API_KEY is set)
      2. Ollama (fallback, must be running locally)
    """
    if GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
            convert_system_message_to_human=False,
        )
    else:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )


def get_llm_info() -> str:
    """Return a human-readable string describing which LLM backend is active."""
    if GEMINI_API_KEY:
        return f"Google Gemini ({GEMINI_MODEL})"
    return f"Ollama ({OLLAMA_MODEL} at {OLLAMA_BASE_URL})"
