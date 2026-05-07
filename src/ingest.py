"""
PDF ingestion pipeline: reads policy PDFs, splits them into section-based
chunks, generates embeddings, and stores them in ChromaDB.

Chunking Strategy:
  - Each PDF is split by numbered section headers (e.g., "1. Purpose", "2. Scope")
  - Each section becomes one chunk (~100-400 tokens)
  - Tables and lists within a section are kept together
  - Metadata includes source file, section number, section title, and policy name

Why section-based chunking?
  - The policy documents are small (1-2 pages) and well-structured
  - Each section is a self-contained semantic unit
  - Fixed-size chunking would split tables and related content arbitrarily
  - This approach yields ~35-40 chunks total, which is very manageable

Embedding Strategy:
  - Uses E5 instruction-prefixed embeddings ("passage: ..." for documents)
  - At query time, the query is prefixed with "query: ..."
  - This asymmetric approach improves retrieval quality significantly

Production Scalability Notes:
  - Each chunk carries rich metadata (file hash, policy name, section, ingestion date)
  - The ingestion is idempotent: re-running skips already-indexed documents
  - For 100s of files: add file-level hashing to detect changes and re-index only modified docs
  - For 1000s of files: switch to batch ingestion with progress tracking
"""

import re
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from PyPDF2 import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    POLICIES_DIR,
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)

# ── Policy name mapping ──────────────────────────────────────────────────────
POLICY_NAMES = {
    "policy_01_annual_leave.pdf": "Annual Leave Policy",
    "policy_02_remote_work.pdf": "Remote Work Policy",
    "policy_03_performance_review.pdf": "Performance Review Policy",
    "policy_04_code_of_conduct.pdf": "Code of Conduct Policy",
    "policy_05_training_development.pdf": "Training and Development Policy",
}


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for change detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def split_into_sections(text: str, source_file: str, file_hash: str) -> list[dict]:
    """
    Split document text into section-based chunks.

    Each chunk is a dict with keys:
      - content: the section text (prefixed with "passage: " for E5)
      - raw_content: the original section text without prefix
      - section_number: e.g. "1"
      - section_title: e.g. "Purpose"
      - source_file: filename
      - policy_name: human-readable policy name
      - file_hash: SHA-256 hash for change detection
      - ingested_at: ISO timestamp
    """
    policy_name = POLICY_NAMES.get(source_file, source_file)
    ingested_at = datetime.now(timezone.utc).isoformat()

    # Match numbered section headers like "1. Purpose" or "10. Appendix"
    section_pattern = re.compile(r"^(\d+)\.\s+(.+)$", re.MULTILINE)
    matches = list(section_pattern.finditer(text))

    chunks = []

    if not matches:
        # No sections found — treat the entire document as one chunk
        raw = text.strip()
        chunks.append(
            {
                "content": f"passage: {policy_name}\n\n{raw}",
                "raw_content": raw,
                "section_number": "0",
                "section_title": "Full Document",
                "source_file": source_file,
                "policy_name": policy_name,
                "file_hash": file_hash,
                "ingested_at": ingested_at,
            }
        )
        return chunks

    for i, match in enumerate(matches):
        section_num = match.group(1)
        section_title = match.group(2).strip()

        # Section content runs from this header to the next header (or end of text)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_content = text[start:end].strip()

        # Build the raw content with policy name context
        raw = f"{policy_name}\n\n{section_content}"

        # E5 embedding format: prefix documents with "passage: "
        # This tells the model this is a document to be retrieved, not a query
        prefixed = f"passage: {raw}"

        chunks.append(
            {
                "content": prefixed,
                "raw_content": raw,
                "section_number": section_num,
                "section_title": section_title,
                "source_file": source_file,
                "policy_name": policy_name,
                "file_hash": file_hash,
                "ingested_at": ingested_at,
            }
        )

    return chunks


def ingest_policies() -> chromadb.Collection:
    """
    Ingest all policy PDFs into ChromaDB.

    This function is idempotent: if the collection already exists and has
    documents, it skips re-ingestion.

    Returns the ChromaDB collection.
    """
    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Check if collection already exists with data
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        if collection.count() > 0:
            logger.info(
                f"Collection '{CHROMA_COLLECTION_NAME}' already has "
                f"{collection.count()} documents. Skipping ingestion."
            )
            return collection
    except Exception:
        pass  # Collection doesn't exist yet

    # Create or get the collection
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Process each PDF
    all_chunks = []
    pdf_files = sorted(POLICIES_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {POLICIES_DIR}")
        return collection

    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        file_hash = _compute_file_hash(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        chunks = split_into_sections(text, pdf_path.name, file_hash)
        all_chunks.extend(chunks)
        logger.info(f"  → {len(chunks)} sections extracted (hash: {file_hash})")

    logger.info(f"Total chunks: {len(all_chunks)}")

    # Generate embeddings using the E5-prefixed content
    texts = [chunk["content"] for chunk in all_chunks]
    logger.info("Generating embeddings with E5 (passage-prefixed)...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Store in ChromaDB — we store raw_content as the document (without E5 prefix)
    # so the LLM sees clean text, while embeddings use the prefixed version
    ids = [f"{chunk['source_file']}__section_{chunk['section_number']}" for chunk in all_chunks]
    documents = [chunk["raw_content"] for chunk in all_chunks]
    metadatas = [
        {
            "source_file": chunk["source_file"],
            "section_number": chunk["section_number"],
            "section_title": chunk["section_title"],
            "policy_name": chunk["policy_name"],
            "file_hash": chunk["file_hash"],
            "ingested_at": chunk["ingested_at"],
        }
        for chunk in all_chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(f"Successfully ingested {len(all_chunks)} chunks into ChromaDB.")
    return collection


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collection = ingest_policies()
    print(f"\nIngestion complete. Collection has {collection.count()} documents.")
