"""
RAG retrieval tool: searches the ChromaDB vector store for policy-related
content and returns relevant chunks with detailed source references.

Uses E5 instruction-prefixed embeddings:
  - Queries are prefixed with "query: " for better retrieval
  - Documents were stored with "passage: " prefix during ingestion

Returns structured references that the agent can cite in its response,
and that the API can return to the frontend for display.
"""

import logging

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import get_llm, CHROMA_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME, RAG_TOP_K

logger = logging.getLogger(__name__)

# ── Module-level singletons (initialized on first use) ───────────────────────
_embedding_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None

# Store the last retrieved references for the API to access
_last_references: list[dict] = []


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_collection() -> chromadb.Collection:
    """Lazy-load the ChromaDB collection."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    return _collection


def get_last_references() -> list[dict]:
    """
    Get the references from the last RAG retrieval.

    Returns a list of dicts with keys:
      - policy_name: e.g. "Annual Leave Policy"
      - section_number: e.g. "5"
      - section_title: e.g. "Carry-Over"
      - source_file: e.g. "policy_01_annual_leave.pdf"
      - relevance_score: cosine similarity (0-1, higher = more relevant)
    """
    return list(_last_references)


def _rewrite_query(query: str) -> str:
    """Rewrite a conversational query into a dense keyword search query."""
    llm = get_llm()
    sys_msg = SystemMessage(
        content=(
            "You are an expert HR assistant. Rewrite the following user query into a clean, "
            "concise, keyword-rich search query suitable for a vector database search over HR policy documents. "
            "Remove any conversational filler, greetings, or personal pronouns. "
            "If the query is in Arabic, rewrite it in Arabic. If in English, in English. "
            "Output ONLY the rewritten search query, nothing else."
        )
    )
    hum_msg = HumanMessage(content=query)
    try:
        response = llm.invoke([sys_msg, hum_msg])
        rewritten = response.content.strip()
        # Clean up if the model outputted thinking blocks or quotes
        import re
        rewritten = re.sub(r"<think>.*?</think>", "", rewritten, flags=re.DOTALL).strip()
        rewritten = rewritten.strip('"\'')
        if rewritten:
            logger.info(f"Query Rewritten: '{query}' -> '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}. Using original query.")
    
    return query


@tool
def search_policies(query: str) -> str:
    """Search company policy documents to answer questions about HR policies,
    rules, procedures, leave entitlements, remote work rules, performance
    reviews, code of conduct, and training/development programs.

    Use this tool when the question is about company rules, policies,
    procedures, or general HR guidelines — NOT about a specific employee's
    personal data.

    Args:
        query: The policy-related question to search for.

    Returns:
        Relevant policy excerpts with source citations. Each excerpt includes
        the policy name, section number, and section title for reference.
    """
    global _last_references

    model = _get_embedding_model()
    collection = _get_collection()

    # Rewrite the query for better vector search
    optimized_query = _rewrite_query(query)
    print(f"\n✨ [RAG MAGIC] Raw Query: '{query}'")
    print(f"✨ [RAG MAGIC] Rewritten Optimized Query: '{optimized_query}'\n")

    # E5 embedding format: prefix queries with "query: "
    # This tells the model this is a search query, not a document
    prefixed_query = f"query: {optimized_query}"
    query_embedding = model.encode(prefixed_query).tolist()

    # Retrieve top-k results
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=RAG_TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        _last_references = []
        return "No relevant policy information found."

    # Build structured references and formatted output
    references = []
    formatted_chunks = []

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance → similarity (1 - distance)
        similarity = round(1 - distance, 3)

        ref = {
            "policy_name": meta["policy_name"],
            "section_number": meta["section_number"],
            "section_title": meta["section_title"],
            "source_file": meta["source_file"],
            "relevance_score": similarity,
        }
        references.append(ref)

        # Format for the LLM with clear citation markers
        citation = (
            f"[Reference: {meta['policy_name']}, "
            f"Section {meta['section_number']}: {meta['section_title']} "
            f"(relevance: {similarity:.0%})]"
        )
        formatted_chunks.append(f"{citation}\n{doc}")

    # Store references for the API to access
    _last_references = references

    logger.info(
        f"RAG retrieved {len(references)} chunks: "
        + ", ".join(f"{r['policy_name']} §{r['section_number']}" for r in references)
    )

    return "\n\n---\n\n".join(formatted_chunks)
