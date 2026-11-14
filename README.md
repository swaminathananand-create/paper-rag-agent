# 📄 Academic Paper RAG Agent

A production-quality **Retrieval-Augmented Generation (RAG)** agent that ingests academic papers (local PDFs, `.txt` files, or remote URLs), retrieves targeted evidence passages, and generates structured summary reports — all in a consistent, machine-readable format.

Also implements the **stretch goal**: auto-generate a thematic literature review outline from multiple papers, grouped by research theme.

---

## 🧠 What is RAG?

> **Retrieval-Augmented Generation (RAG)** is a technique where an AI agent retrieves relevant source material *before* generating a response, rather than relying solely on the LLM's training data.

Instead of asking Gemini to summarize a paper from memory:

```
User paper → chunk into passages → index (vector/BM25) → query with specific aspect questions
→ retrieve top-k evidence passages → feed to Gemini → structured summary grounded in real text
```

This eliminates hallucination, enables citations, and works with *any* document the LLM has never seen.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Paper RAG Pipeline                               │
│                                                                           │
│  Source                  Chunk               Retrieve           Synthesize│
│  (PDF/URL/txt)           & Index             (Targeted)         (Gemini)  │
│                                                                           │
│  PaperLoader ──► PaperSplitter ──► PaperRetriever ──► PaperSummarizer   │
│      │                │                  │                     │          │
│  PyPDFLoader      RecursiveChar      BM25 (offline)     ChatGoogleGenAI   │
│  HTML scraper     TextSplitter   or Gemini Embeddings   gemini-3.8-flash  │
│  URL download     chunk_size=1000   InMemoryVectorStore  JSON + Markdown  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
paper_rag_agent/
├── cli.py                        # Rich CLI entrypoint
├── requirements.txt              # Dependencies
├── pytest.ini                    # Test configuration
├── .env.example                  # API key template
│
├── src/paper_rag/
│   ├── config.py                 # Settings (API key, models, chunking)
│   ├── schemas.py                # Pydantic data models
│   ├── loader.py                 # Document ingestion (PDF, URL, text)
│   ├── splitter.py               # RecursiveCharacterTextSplitter
│   ├── retriever.py              # BM25 + Gemini embedding retriever
│   ├── summarizer.py             # RAG summarization engine (Gemini / mock)
│   └── literature_review.py     # Multi-paper thematic review agent
│
├── sample_papers/
│   ├── attention_is_all_you_need.txt
│   ├── retrieval_augmented_generation.txt
│   ├── react_reasoning_acting.txt
│   ├── chain_of_thought_prompting.txt
│   ├── self_rag.txt
│   └── paper_urls.txt            # 5 arXiv URLs for download testing
│
└── tests/
    ├── test_loader.py
    ├── test_splitter.py
    ├── test_retriever.py
    ├── test_summarizer.py
    └── test_literature_review.py
```

---

## ⚡ Quick Start

### 1. Set Up Environment

```powershell
# Navigate to the project directory
cd C:\Users\Asus\.gemini\antigravity\scratch\paper_rag_agent

# Install dependencies (uv is fastest)
uv pip install --system-certs -r requirements.txt

# OR standard venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key (Optional)

The agent runs in **offline BM25 extractive mode** without an API key. To unlock Gemini-powered structured summaries:

```powershell
# Copy the template
copy .env.example .env

# Edit .env and add your key:
#   GEMINI_API_KEY=your_key_here
# Get a free key at: https://aistudio.google.com/
```

---

## 🚀 Usage

### Summarize a Single Paper

```powershell
# From a local text file
.venv\Scripts\python cli.py summarize sample_papers\attention_is_all_you_need.txt

# From a local PDF
.venv\Scripts\python cli.py summarize path\to\your\paper.pdf

# From an arXiv URL
.venv\Scripts\python cli.py summarize https://arxiv.org/abs/1706.03762
```

**Output** (structured, consistent format for every paper):
```
┌─────────────────────────────────────────────────────────────┐
│ Structured Paper Summary: Attention Is All You Need          │
├──────────────────────────────────────────────────────────────┤
│ 1. Abstract & Context      │ 2. Main Argument & Thesis       │
│ 3. Methodology             │ 4. Key Findings & Results       │
│ 5. Implications            │ 🔍 RAG Retrieved Evidence       │
└──────────────────────────────────────────────────────────────┘
```

Saved to `output/<paper_slug>_summary.md` + `output/<paper_slug>_summary.json`

---

### Batch Summarize Multiple Papers

```powershell
# Using the pre-configured URL list
.venv\Scripts\python cli.py batch sample_papers\paper_urls.txt

# Or create your own list (one URL/path per line, # = comment)
.venv\Scripts\python cli.py batch my_papers.txt
```

---

### Generate a Literature Review (Stretch Goal 🎓)

```powershell
# Review all 5 sample papers
.venv\Scripts\python cli.py review sample_papers\attention_is_all_you_need.txt sample_papers\retrieval_augmented_generation.txt sample_papers\react_reasoning_acting.txt sample_papers\chain_of_thought_prompting.txt sample_papers\self_rag.txt

# With a custom title
.venv\Scripts\python cli.py review sample_papers\*.txt --title "Survey of Foundational LLM Architectures"
```

**Output**: A thematic synthesis outline with:
- Executive summary across all papers
- 2–4 research theme clusters with per-theme comparative analysis
- Consensus points and active debates across authors
- Identified research gaps and future directions

Saved to `output/literature_review_outline.md` + `.json`

---

## 📊 Output Format

Every paper produces a **consistent** structured report:

| Section | Content |
|---|---|
| **Abstract** | 2–4 sentence synopsis of problem, context, objective |
| **Main Argument** | Central thesis and primary hypotheses |
| **Methodology** | Dataset, architecture, experimental design |
| **Key Findings** | Bulleted empirical or theoretical results |
| **Implications** | Broader significance and future directions |
| **RAG Evidence** | Cited page-referenced excerpts from the actual document |

---

## 🧪 Run Tests

```powershell
# All tests (offline mode — no API key needed)
.venv\Scripts\pytest tests\ -v

# Single test module
.venv\Scripts\pytest tests\test_summarizer.py -v

# Specific test
.venv\Scripts\pytest tests\test_retriever.py::TestBM25Retriever::test_methodology_query_retrieves_relevant_content -v
```

---

## 🔑 API Key Modes

| Mode | When | Behavior |
|---|---|---|
| **Offline BM25** | No `GEMINI_API_KEY` set | Deterministic extractive summarization via BM25 keyword retrieval. Fully functional, instant, no cost. |
| **Live Gemini** | `GEMINI_API_KEY` configured | Semantic vector embeddings + Gemini LLM synthesis. Produces richer, abstractive summaries with JSON reasoning chains. |

The agent **automatically detects** which mode to use and gracefully falls back to BM25 if an API call fails at runtime.

---

## 🧩 Tech Stack

| Component | Technology |
|---|---|
| LLM | Gemini (`gemini-3.8-flash`) via `langchain-google-genai` |
| Embeddings | `gemini-embedding-001` (with BM25 fallback) |
| Vector Store | `InMemoryVectorStore` (LangChain Core) |
| PDF Ingestion | `PyPDFLoader` (LangChain Community) |
| Chunking | `RecursiveCharacterTextSplitter` |
| Offline Retrieval | Custom BM25 (built-in, no dependencies) |
| Data Validation | Pydantic v2 |
| CLI | Rich |
| Testing | pytest |
