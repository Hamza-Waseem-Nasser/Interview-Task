# 🤖 HR AI Assistant — AlNoor Technologies

An internal HR AI assistant that answers employee questions using **Retrieval-Augmented Generation (RAG)** over company policy documents and a **structured employee database**. Built with LangGraph, Ollama, ChromaDB, and FastAPI.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1-purple)
![Ollama](https://img.shields.io/badge/Ollama-llama3.1:8b-orange)
![License](https://img.shields.io/badge/License-MIT-gray)

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [LLM Choice & Rationale](#llm-choice--rationale)
- [RAG & Chunking Strategy](#rag--chunking-strategy)
- [Agent Routing Logic](#agent-routing-logic)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Example Questions & Responses](#example-questions--responses)
- [Limitations & Future Work](#limitations--future-work)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                               │
│  POST /ask  ─────→  LangGraph Agent  ─────→  Response + Source      │
│                          │                                           │
│                    ┌─────┴─────┐                                     │
│                    │   Tools   │                                      │
│              ┌─────┴───┐ ┌────┴────┐                                 │
│              │   RAG   │ │  CSV    │                                  │
│              │  Tool   │ │  Tool   │                                  │
│              └────┬────┘ └────┬────┘                                  │
│                   │           │                                       │
│            ┌──────┴──┐  ┌────┴─────┐                                 │
│            │ChromaDB │  │  Pandas  │                                  │
│            │(vectors)│  │DataFrame │                                  │
│            └─────────┘  └──────────┘                                  │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Sentence     │  │   Ollama     │  │   Session    │                │
│  │ Transformers │  │  llama3.1:8b │  │   Memory     │                │
│  │ (embeddings) │  │   (LLM)     │  │  (in-memory) │                │
│  └──────────────┘  └──────────────┘  └──────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
```

The system follows a **tool-calling agent** pattern:

1. The user sends a question with their employee ID
2. The LangGraph agent receives the question and decides which tool(s) to invoke
3. **RAG Tool** — searches ChromaDB for relevant policy document chunks
4. **Structured Data Tool** — looks up the employee's record from the CSV
5. The agent synthesizes tool outputs into a final answer
6. The response includes source attribution (`rag`, `structured_data`, `both`, or `unknown`)

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | Ollama (llama3.1:8b) | Free, local, native tool-calling support, reproducible |
| **Embeddings** | sentence-transformers (multilingual-e5-base) | Free, runs on CPU, 768-dim, instruction-prefixed for better retrieval |
| **Vector Store** | ChromaDB (persistent) | Local, auto-persists to disk, good LangChain integration |
| **Agent** | LangGraph | Explicit state graph, better than black-box AgentExecutor |
| **API** | FastAPI | Async, auto-generated Swagger docs, Pydantic validation |
| **Data** | pandas | Lightweight, no database setup needed |
| **Frontend** | React + Vite | Modern, fast dev server, minimal setup |

---

## LLM Choice & Rationale

**Selected: Ollama with `llama3.1:8b`**

I evaluated three free options:

| Option | Pros | Cons |
|--------|------|------|
| **Ollama (llama3.1:8b)** ✅ | Fully local, no API key, native tool-calling, reproducible | Requires ~5GB download, needs decent hardware |
| Groq free tier | Very fast inference | Requires API key signup, rate limits, cloud dependency |
| Gemini free tier | Excellent quality, generous free tier | Requires API key, cloud dependency |

**Why llama3.1:8b specifically?**
- **Tool calling**: Llama 3.1 has built-in function-calling support, which is critical for the agent to route between RAG and structured data tools reliably
- **8B parameters**: Strikes the best balance between quality and speed — large enough for nuanced HR answers, small enough to run on consumer hardware
- **Reproducibility**: Any reviewer can `ollama pull llama3.1:8b` and get the exact same model

The system also supports **Gemini as an alternative** — set `GEMINI_API_KEY` in `.env` to switch automatically.

---

## RAG & Chunking Strategy

### Why Section-Based Chunking?

The policy documents are small (1-2 pages each) and well-structured with numbered sections. I chose **section-based chunking** over fixed-size chunking:

```
Policy PDF
  ├── Section 1: Purpose          → Chunk 1
  ├── Section 2: Eligibility      → Chunk 2
  ├── Section 3: Remote Models    → Chunk 3  (includes table)
  ├── Section 4: Core Hours       → Chunk 4
  └── ...
```

**Advantages over fixed-size (e.g., 500 tokens):**

| Aspect | Section-Based (chosen) | Fixed-Size |
|--------|----------------------|------------|
| Semantic coherence | ✅ Each chunk = one complete topic | ❌ May split mid-sentence or mid-table |
| Table handling | ✅ Tables stay with their section | ❌ Tables get split across chunks |
| Retrieval precision | ✅ Returns exactly the relevant section | ⚠ May return partial context |
| Number of chunks | ~35-40 total (manageable) | ~50-60 (more noise) |

**Embedding model: E5 (instruction-prefixed)**

I chose `intfloat/multilingual-e5-base` over simpler models like `all-MiniLM-L6-v2` because:

| Model | Dims | Technique | Why |
|-------|------|-----------|-----|
| all-MiniLM-L6-v2 | 384 | Plain embedding | Default everyone uses — no query/doc distinction |
| **multilingual-e5-base** ✅ | 768 | **Instruction-prefixed** | Uses `"query: ..."` and `"passage: ..."` prefixes for asymmetric retrieval |

E5 uses different prefixes for documents and queries:
- Documents are embedded with `"passage: Annual Leave Policy\n\n5. Carry-Over..."` 
- Queries are embedded with `"query: what is the carry-over limit?"`
- This asymmetric approach tells the model the *intent* behind each embedding, improving retrieval quality
- Also supports multilingual queries (Arabic, etc.) if needed in the future

**Additional RAG techniques used:**

1. **Context enrichment**: Each chunk is prepended with the policy name so the LLM always knows which policy a retrieved chunk belongs to
2. **Rich metadata**: Every chunk stores `policy_name`, `section_number`, `section_title`, `file_hash`, and `ingested_at` for source attribution and change detection
3. **Structured references**: The RAG tool returns structured reference objects with relevance scores, which the API exposes to the frontend
4. **Cosine similarity with top-3 retrieval**: Returns the 3 most relevant sections for each query

**What I intentionally did NOT use** (and why):
- **Hybrid search (BM25 + dense)**: Over-engineering for ~40 chunks — adds complexity without meaningful improvement
- **Re-ranking**: Cross-encoder re-ranking is valuable at scale (1000s of docs), unnecessary here
- **Query decomposition**: The agent already handles multi-tool routing; adding query decomposition would increase latency

> This is a deliberate engineering judgment: the right techniques depend on the data scale, not a checklist of every possible RAG optimization.

---

## Agent Routing Logic

The LangGraph agent uses the LLM's **native tool-calling** capability to route questions. The system prompt gives clear instructions:

```
Question about policies/rules → search_policies (RAG)
Question about employee data  → query_employee_data (CSV)
Question needing both         → Both tools called
Out-of-scope question         → "I don't know" (no tools)
```

**How does the agent decide?**

The LLM is provided with detailed tool descriptions:
- `search_policies`: "Search company policy documents to answer questions about HR policies, rules, procedures..."
- `query_employee_data`: "Look up personal employee information from the HR database..."

The system prompt also handles **ambiguous cases** explicitly:
- "Am I eligible for remote work?" → Needs **both** tools (employee status + policy rules)
- "What salary increase can I expect?" → Needs **both** (performance rating + compensation policy)

**Source determination** is based on which tools were actually called (not heuristics), ensuring accurate attribution.

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed
- Node.js 18+ (for frontend, optional)

### Option 1: One-Command Start (Windows)
```bash
start.bat
```

### Option 2: One-Command Start (Linux/Mac)
```bash
chmod +x start.sh
./start.sh
```

### Option 3: Manual Setup
```bash
# 1. Pull the LLM model
ollama pull llama3.1:8b

# 2. Create virtual environment and install dependencies
python -m uv venv .venv
python -m uv pip install -r requirements.txt

# 3. Start the API server
.venv/Scripts/python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 4. (Optional) Start the frontend
cd frontend
npm install
npm run dev
```

### Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Ask the HR assistant a question |
| `/health` | GET | Health check + LLM backend info |
| `/docs` | GET | Interactive Swagger documentation |

---

## API Reference

### POST /ask

**Request:**
```json
{
  "employee_id": "EMP001",
  "question": "How many leave days do I have left?",
  "session_id": null
}
```

**Response:**
```json
{
  "answer": "Based on the employee database, you have 18 leave days remaining.",
  "source": "structured_data",
  "references": [],
  "session_id": "a1b2c3d4-..."
}
```

When the RAG tool is used, the response includes structured references:
```json
{
  "answer": "According to the Annual Leave Policy (Section 5: Carry-Over), employees may carry over a maximum of 10 days...",
  "source": "rag",
  "references": [
    {
      "policy_name": "Annual Leave Policy",
      "section_number": "5",
      "section_title": "Carry-Over",
      "source_file": "policy_01_annual_leave.pdf",
      "relevance_score": 0.91
    }
  ],
  "session_id": "a1b2c3d4-..."
}
```

**Source values:**
| Value | Meaning |
|-------|---------|
| `rag` | Answer from policy documents |
| `structured_data` | Answer from employee database |
| `both` | Answer combined both sources |
| `unknown` | Question outside available data |

---

## Example Questions & Responses

### 1. Leave Balance (Structured Data)
```
Employee: EMP001 (Sara Al-Rashidi)
Question: "How many leave days do I have left?"
Source:   structured_data

Answer:   Based on your employee record, Sara, you have 18 leave days remaining.
          Your annual entitlement is 30 days and you have taken 12 days so far.
```

### 2. Policy Question (RAG)
```
Employee: EMP005
Question: "What is the carry-over limit for unused annual leave?"
Source:   rag

Answer:   According to the Annual Leave Policy (Section 5), employees may carry
          over a maximum of 10 days of unused annual leave into the following
          calendar year. Any leave beyond 10 days that is not taken by December 31
          will be forfeited unless a written exception is approved by the HR Director.
```

### 3. Eligibility Check (Both Sources)
```
Employee: EMP013 (Sami Taha, Probation)
Question: "Am I eligible for remote work?"
Source:   both

Answer:   Based on your records and the Remote Work Policy, you are currently NOT
          eligible for remote work. The policy states that remote work is available
          to employees who have completed at least 6 months of service and are not
          in a probationary period. Your current status is "Probation" and your
          remote model is "Office First".
```

### 4. Training Budget (Both Sources)
```
Employee: EMP003 (Lina Haddad, L6 Engineering)
Question: "How much training budget do I have remaining, and what can I spend it on?"
Source:   both

Answer:   You have 200 SAR remaining from your 12,000 SAR training budget (you've
          spent 11,800 SAR). As a Senior employee (L5-L6), your budget covers
          certifications, conferences, and workshops. Eligible activities include
          online courses (Coursera, Udemy, etc.), professional certifications
          (AWS, Azure, PMP, etc.), technical books, and industry conferences.
```

### 5. Out of Scope (Unknown)
```
Employee: EMP001
Question: "What is the stock price of AlNoor Technologies?"
Source:   unknown

Answer:   I don't have enough information to answer that question. This falls
          outside the scope of the HR policies and employee data I have access to.
```

---

## Bonus: Session Memory

The system supports **in-memory conversation memory**. When you include a `session_id` in your request, the agent remembers previous messages within that session:

```
Turn 1: "How many leave days do I have?"  → "You have 18 days remaining."
Turn 2: "And what's the carry-over limit?" → "The carry-over limit is 10 days.
          So from your 18 remaining days, you could carry over 10 to next year."
```

The frontend automatically manages session IDs.

---

## Limitations & Future Work

### Current Limitations
- **Single language**: Only supports English queries (policies are in English)
- **Static data**: Employee CSV is loaded at startup — changes require restart
- **No authentication**: The API has no auth layer (internal tool assumption)
- **Memory is in-memory**: Session history is lost on server restart
- **No multi-tenancy**: An employee could query another employee's data by changing the ID

### Future Improvements
- **Persistent memory**: Use Redis or SQLite for session storage
- **Authentication & authorization**: Ensure employees can only access their own data
- **Hybrid search**: Add BM25 keyword search alongside dense vector search for better retrieval
- **Re-ranking**: Add cross-encoder re-ranking for improved relevance at scale
- **Streaming responses**: Use SSE for token-by-token streaming in the UI
- **Arabic support**: Add multilingual embeddings and LLM support
- **Real database integration**: Connect to an actual HRIS instead of CSV
- **Audit logging**: Track all queries for compliance

---

## Project Structure

```
├── app.py                  # FastAPI application
├── src/
│   ├── config.py           # Configuration & LLM factory
│   ├── ingest.py           # PDF ingestion & chunking
│   ├── rag.py              # RAG retrieval tool
│   ├── structured_data.py  # Employee data tool
│   ├── agent.py            # LangGraph agent
│   └── memory.py           # Session memory
├── data/
│   ├── policies/           # 5 policy PDFs
│   └── employees.csv       # Employee data
├── frontend/               # React chat UI
├── test_agent.py           # 5 example questions
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── start.bat / start.sh    # One-command startup
└── README.md               # This file
```

---

## License

This project was built as a technical assessment for AlNoor Technologies.
