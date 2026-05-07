"""
In-memory session memory for the HR AI Assistant.

Stores conversation history per session_id so the agent can maintain
context across multiple turns within a conversation.
"""

import logging
from collections import defaultdict

from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# ── In-memory store ──────────────────────────────────────────────────────────
# Maps session_id → list of messages
_sessions: dict[str, list[BaseMessage]] = defaultdict(list)

# Maximum messages to keep per session (to prevent context overflow)
MAX_MESSAGES_PER_SESSION = 20


def get_session_history(session_id: str) -> list[BaseMessage]:
    """Retrieve the conversation history for a session."""
    return list(_sessions[session_id])


def add_to_session(session_id: str, messages: list[BaseMessage]) -> None:
    """Add messages to a session's history, pruning old ones if needed."""
    history = _sessions[session_id]
    history.extend(messages)

    # Keep only the most recent messages if we exceed the limit
    if len(history) > MAX_MESSAGES_PER_SESSION:
        _sessions[session_id] = history[-MAX_MESSAGES_PER_SESSION:]
        logger.info(
            f"Session '{session_id}' pruned to {MAX_MESSAGES_PER_SESSION} messages"
        )


def clear_session(session_id: str) -> None:
    """Clear a session's history."""
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Session '{session_id}' cleared")


def list_sessions() -> list[str]:
    """List all active session IDs."""
    return list(_sessions.keys())
