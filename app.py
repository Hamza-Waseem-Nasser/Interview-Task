"""
FastAPI application for the HR AI Assistant.

Endpoints:
  POST /ask     — Ask the HR assistant a question
  GET  /health  — Health check
  GET  /        — Serves the frontend (if built)

Error handling uses structured Pydantic models and proper HTTP status codes.
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import get_llm_info
from src.ingest import ingest_policies
from src.agent import ask_agent

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan: startup & shutdown ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before the app starts serving requests."""
    logger.info("=" * 60)
    logger.info("HR AI Assistant — Starting up")
    logger.info(f"LLM Backend: {get_llm_info()}")
    logger.info("=" * 60)

    # Ingest policy documents into ChromaDB (idempotent)
    logger.info("Ingesting policy documents...")
    ingest_policies()
    logger.info("Policy documents ready.")

    yield  # App is now running

    logger.info("HR AI Assistant — Shutting down")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="HR AI Assistant",
    description="Internal HR AI assistant for AlNoor Technologies. "
                "Answers employee questions from policy documents and structured HR data.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (allow frontend dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────
class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    employee_id: str = Field(
        ...,
        description="Employee ID, e.g. 'EMP001'",
        examples=["EMP001"],
    )
    question: str = Field(
        ...,
        description="The HR-related question to ask",
        examples=["How many leave days do I have left?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for conversation memory. "
                    "If not provided, a new session is created.",
    )


class Reference(BaseModel):
    """A source reference from the RAG retrieval."""

    policy_name: str = Field(description="Name of the policy document")
    section_number: str = Field(description="Section number within the policy")
    section_title: str = Field(description="Title of the referenced section")
    source_file: str = Field(description="Original PDF filename")
    relevance_score: float = Field(description="Cosine similarity score (0-1)")


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    answer: str = Field(description="The assistant's response")
    source: str = Field(
        description="Source of the answer: 'rag', 'structured_data', 'both', or 'unknown'"
    )
    references: list[Reference] = Field(
        default=[],
        description="Policy document sections referenced in the answer",
    )
    session_id: str = Field(description="Session ID for follow-up questions")


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str = "healthy"
    llm_backend: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post(
    "/ask",
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Ask the HR Assistant",
    description="Submit an HR-related question. The agent routes to the "
                "appropriate tool (policy RAG or employee data) and returns "
                "an answer with source attribution.",
)
async def ask_endpoint(request: AskRequest):
    """Main endpoint: processes an employee's HR question."""
    # Validate input
    if not request.employee_id.strip():
        raise HTTPException(
            status_code=400,
            detail="employee_id cannot be empty",
        )
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="question cannot be empty",
        )

    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = await ask_agent(
            employee_id=request.employee_id.strip(),
            question=request.question.strip(),
            session_id=session_id,
        )
        return AskResponse(
            answer=result["answer"],
            source=result["source"],
            references=result.get("references", []),
            session_id=session_id,
        )
    except Exception as e:
        logger.exception(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process your question: {str(e)}",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the health status and configuration of the assistant.",
)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(llm_backend=get_llm_info())
