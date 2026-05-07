# HR AI Assistant — Technical Architecture & Process Report

## 1. Executive Summary
This project delivers a production-grade internal HR AI Assistant for AlNoor Technologies. It uses a **LangGraph-based Agentic architecture** to answer employee questions in both Arabic and English by intelligently routing queries between two distinct data sources: a Vector Database (ChromaDB) for unstructured policy documents, and a structured CSV database for personal employee records.

---

## 2. LLM Evaluation & Evolution Process

Our development process involved testing multiple LLMs to find the optimal balance of privacy, tool-calling reliability, and bilingual proficiency.

### Phase 1: Local Deployment with Ollama (Llama 3.1:8B)
- **The Goal:** Start with a fully local, privacy-first model using Ollama to ensure sensitive HR data never leaves the corporate network.
- **The Results:** While Llama 3.1:8B performed well in English, we observed several critical limitations:
  1. **Tool-Calling Instability:** The model often struggled with native tool calling. Instead of returning a structured JSON payload to trigger a tool, it would occasionally "leak" the raw JSON as plain text in the chat, forcing us to build complex Regex cleanup and fallback synthesis prompts (`_clean_response`).
  2. **Bilingual Leakage:** When interacting in Arabic, the model struggled to maintain strict language boundaries, often injecting English meta-commentary into Arabic responses (e.g., replying with *"Since your question is about a personal preference... لا يمكنني الإجابة"*).

### Phase 2: Migration to Google Gemini (2.5 Flash)
- **The Goal:** Meet the project requirements by utilizing the Gemini API for superior bilingual reasoning and reliable agentic tool orchestration.
- **The Results:** Gemini immediately resolved the routing instabilities. It natively executed LangGraph tool calls flawlessly, correctly handled LangChain's list-based message blocks, and maintained perfect, native-sounding Arabic conversational flows without any English leakage.
- **Current State:** The project is configured to use Gemini via the `GEMINI_API_KEY` in the `.env` file, but maintains full backwards compatibility with Ollama for teams that require strict local execution.

---

## 3. Data Architecture: The "Two-Tool" Strategy

A major architectural decision was avoiding the common pitfall of dumping all data into a single Vector Database. Vector Databases (which use cosine similarity) excel at semantic text matching but fail catastrophically at exact tabular lookups (e.g., retrieving a specific employee's remaining leave balance).

**The Solution: Agentic Routing**
We provided the LLM with two dedicated tools and strict routing rules in the `SYSTEM_PROMPT`:
1. **`search_policies` (ChromaDB / RAG):** Called for general rules, procedures, and policy questions.
2. **`query_employee_data` (Pandas CSV):** Called for exact numerical lookups regarding a specific user's personal data.

**Why we avoided Tabular Metadata Filtering in ChromaDB:**
Instead of embedding the CSV rows and storing the exact data in ChromaDB metadata, we kept the CSV entirely separate. Relying on an Embedding Neural Network to perform an exact row lookup is computationally wasteful, prone to "semantic leakage" (hallucinating wrong numbers), and creates an enterprise security risk where an employee might accidentally retrieve a colleague's salary data. Our separation guarantees exact, O(1) lookups for personal data.

---

## 4. Advanced RAG Implementation Details

For the unstructured policy documents, we implemented a highly optimized RAG pipeline:

### Query Optimization (Rewriting)
Instead of passing raw, conversational user input directly into the similarity search, the backend intercepts the query and uses the LLM to rewrite it. 
* *Example User Input:* "Hey, what is the policy on taking time off to travel to Lebanon?"
* *Rewritten Query:* "Lebanon travel leave HR policy vacation time request approval process"
This dense search string drastically improves Vector DB accuracy.

### Asymmetric Embeddings (`intfloat/multilingual-e5-base`)
We selected the E5 embedding model over the standard MiniLM. E5 supports asymmetric retrieval, meaning we actively prefix the ingested PDF chunks with `passage: ` and our rewritten search queries with `query: `. This forces the model to understand the *intent* of the embedding, resulting in much sharper retrieval.

### Smart Chunking & Dynamic Citations
We utilized **Section-Based Chunking** instead of blind fixed-size token splitting. This ensures that entire tables and coherent policy sections stay together. Furthermore, we implemented a "metadata smuggler" (`get_last_references()`) to extract the exact source filenames and section numbers from ChromaDB and pass them through the LangGraph tool interface so the React frontend can render dynamic, clickable citation badges.

### Techniques Intentionally Omitted
- **Hybrid Search (BM25 + Dense) & Cross-Encoder Re-ranking:** While powerful for massive datasets (10,000+ documents), implementing these for a small corpus of ~40 highly-structured HR policy chunks introduces unnecessary latency and complexity without a meaningful boost in accuracy.

---

## 5. Security & Context Management (Hidden Inputs)

To enforce strict Row-Level Security (RLS) and prevent employees from querying their colleagues' data:
1. The React UI sends the logged-in `employee_id` (e.g., `EMP001`) to the backend.
2. Before the LLM sees the user's question, we invisibly prepend `[Employee ID: EMP001]` to the chat prompt.
3. The LLM is strictly instructed that its privileges are bound to this ID, ensuring it only passes this specific ID to the `query_employee_data` tool and gracefully refuses requests for other employees' data (e.g., *"اعطيني معلومات رشاد"*).

---

## 6. How to Run the Project

### Prerequisites
- Python 3.11+
- Node.js 18+

### Step 1: Clone & Configure
1. Clone the repository.
2. Copy `.env.example` to `.env`.
3. To use Gemini (Recommended), add your API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```
4. (Optional) To use local Ollama, ensure Ollama is running and uncomment the Ollama variables in `.env`.

### Step 2: Start the Backend (FastAPI)
Using `uv` (or standard `pip`):
```bash
python -m uv venv .venv
# Activate the virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

python -m uv pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Start the Frontend (React)
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:5173`. You can log in using any Employee ID from the `data/employees.csv` file (e.g., `EMP001`, `EMP005`).
