# IT Support RAG Assistant

A Retrieval-Augmented Generation (RAG) application built with **LangChain**, **Streamlit**, **Google Gemini**, and **Chroma** vector store. It answers questions from your uploaded documents using a hybrid search approach (BM25 + Vector embeddings).

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How to Delete Old DB Cache](#how-to-delete-old-db-cache)
- [How It Works](#how-it-works)
- [Debugging & Troubleshooting](#debugging--troubleshooting)
- [What Was Fixed](#what-was-fixed)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Multi-format document support**: PDF, TXT, CSV, DOCX, PPTX, XLSX, MD
- **Hybrid search**: Combines BM25 (keyword) + Dense Vector (semantic) retrieval for better results
- **Auto-rebuild on file change**: Detects when documents are added, removed, or modified and automatically rebuilds the index
- **Custom Ensemble Retriever**: Built-in hybrid retriever that deduplicates and ranks results from both search methods
- **Unique DB per run**: Avoids file-locking issues by creating a new Chroma DB directory on each rebuild
- **Streamlit sidebar with debug panel**: See loaded files, chunk count, errors, and raw retrieved chunks
- **Force Rebuild button**: Manually clear cache and rebuild the index at any time
- **Source attribution**: Every answer shows which document and page the information came from
- **Clear, grounded answers**: Prompt strictly requires the LLM to answer only from provided context
- **No hallucination guard**: If the answer is not in the documents, it replies: *"I could not find that information in the uploaded documents."*

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI Framework | Streamlit |
| LLM | Google Gemini 1.5 Flash (`gemini-1.5-flash`) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | ChromaDB |
| Hybrid Retrieval | BM25 + Chroma Vector Search |
| Text Splitting | Recursive Character Text Splitter (800 char chunks, 100 overlap) |
| Document Loaders | PyPDF, Unstructured (Word, PPT, Excel), CSV, Text, Markdown |
| API Key Management | `python-dotenv` |

---

## Project Structure

```
Ragwithgeminiapi/
│
├── .env                          # Google API key (DO NOT COMMIT THIS)
├── .venv/                        # Python virtual environment
├── .idea/                        # PyCharm IDE config
│
├── raggemini.py                  # Main application (Streamlit app)
├── main.py                       # Placeholder / sample script
│
├── db_runs/                      # Auto-generated Chroma DB directories
│   ├── db_xxxxxxxxx/
│   └── db_yyyyyyyy/
│
├── README.md                     # This file
│
└── (no requirements.txt)         # Install dependencies manually (see below)
```

> **Note**: The `db_runs/` folder is auto-created. You can safely delete it at any time — it will be rebuilt.

---

## Setup & Installation

### 1. Clone or create the project

```bash
cd "C:\Users\USER\PycharmProjects\Ragwithgeminiapi"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit langchain-google-genai langchain-community langchain-huggingface langchain-text-splitters chromadb pypdf python-dotenv unstructured openpyxl
```

> **Note**: `unstructured` may need additional system dependencies (e.g., `libreoffice` for PPTX/DOCX on Linux). On Windows, it usually works out of the box for most formats.

### 4. Set up your Google API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

Get your API key from: [Google AI Studio](https://aistudio.google.com/app/apikey)

### 5. Upload your documents

Place your documents in:

```
C:\Users\USER\Documents\Data\document
```

Supported files:
- `.pdf` — PDF documents
- `.txt` — Plain text files
- `.csv` — Comma-separated values
- `.docx` — Microsoft Word
- `.pptx` — Microsoft PowerPoint
- `.xlsx` — Microsoft Excel
- `.md` — Markdown files

---

## Configuration

All important settings are near the top of `raggemini.py`:

```python
# ── CONFIG ──────────────────────────────────────────────
FOLDER_PATH = r"C:\Users\USER\Documents\Data\document"
BASE_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_runs")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

| Setting | Description | Change If... |
|---------|-------------|--------------|
| `FOLDER_PATH` | Where your documents live | You want to load documents from a different folder |
| `BASE_DB_DIR` | Where Chroma DBs are stored | You want to change the DB storage location |
| `EMBEDDING_MODEL` | Sentence-transformer model for embeddings | You need a different embedding model |
| `chunk_size=800` | Text chunk size for splitting | Documents are too short or too long |
| `chunk_overlap=100` | Overlap between chunks | You want more/less context continuity |
| `k=10` | Number of chunks retrieved | You want more or fewer sources per answer |

---

## Usage

### Start the app

```bash
streamlit run raggemini.py
```

### Ask a question

1. Type your question in the text box (e.g., "What is the Blue Screen of Death?")
2. The app will:
   - Retrieve the top 10 most relevant chunks (BM25 + Vector)
   - Pass them to Gemini 1.5 Flash
   - Display the answer + sources
3. Expand **"🔍 Raw retrieved chunks"** to see exactly what text was retrieved

### Force rebuild the index

Click the **🔄 Force Rebuild Index** button in the left sidebar to:
- Clear the Streamlit cache
- Delete old Chroma DB directories
- Reload and re-index all documents from scratch

This is useful when:
- You added/removed documents but the auto-detector didn't catch it
- You're seeing stale or wrong answers
- The DB got corrupted or locked

---

## How to Delete Old DB Cache

If you need to manually delete the old `./db` or `./db_runs` folder:

### Windows PowerShell

```powershell
# Navigate to project folder
cd "C:\Users\USER\PycharmProjects\Ragwithgeminiapi"

# Delete the old ./db folder (if it exists from previous code)
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue db

# Delete the new ./db_runs folder
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue db_runs
```

### Windows Command Prompt (cmd.exe)

```cmd
cd C:\Users\USER\PycharmProjects\Ragwithgeminiapi
rmdir /s /q db 2>nul
rmdir /s /q db_runs 2>nul
```

### Why delete the DB?

- Chroma DB files can become **locked** if another process is using them
- Old cached data can contain **stale documents** that were deleted from the source folder
- Corrupted DB files can cause **runtime errors**

After deleting, simply restart the app with `streamlit run raggemini.py` — it will rebuild automatically.

---

## How It Works

### 1. Document Loading

```
C:\Users\USER\Documents\Data\document
         │
         ├── Desktopsupport-document.pdf.pdf
         └── IT_Infrastructure_Kubernetes_RAG_Test_Document.pdf
         │
         ▼
    PyPDFLoader / TextLoader / etc.
         │
         ▼
    List of Document objects (with metadata: source, page)
```

### 2. Text Splitting

```
Document pages
    │
    ▼
RecursiveCharacterTextSplitter
    │ chunk_size=800, chunk_overlap=100
    ▼
List of text chunks (splits)
```

### 3. Embedding & Vector Store

```
Text chunks
    │
    ▼
HuggingFaceEmbeddings (all-MiniLM-L6-v2)
    │
    ▼
Chroma Vector Store (saved to db_runs/db_xxxxxxx/)
```

### 4. Hybrid Retrieval

```
User Query
    │
    ├───► BM25 Retriever (keyword matching) ─┐
    │                                          ├──► SimpleEnsembleRetriever
    ├───► Chroma Vector Retriever (semantic) ─┘         │
                                                          ▼
                                              Deduplicated + ranked chunks
                                                          │
                                                          ▼
                                              Top 10 chunks passed to LLM
```

### 5. Answer Generation

```
Retrieved chunks (context) + User question
    │
    ▼
ChatPromptTemplate
    │
    ▼
Google Gemini 1.5 Flash
    │
    ▼
StrOutputParser
    │
    ▼
Final Answer (with source attribution)
```

---

## Debugging & Troubleshooting

### Symptom: "No documents were loaded"

**Cause**: The document folder is empty or contains unsupported file types.

**Fix**:
1. Check `FOLDER_PATH` is correct
2. Verify files are in the folder
3. Check the **Errors** section in the sidebar for specific file loading errors

### Symptom: Answer is about old documents that don't exist anymore

**Cause**: Streamlit cache is holding old data.

**Fix**: Click **🔄 Force Rebuild Index** in the sidebar, or stop the app and delete the DB folder (see [How to Delete Old DB Cache](#how-to-delete-old-db-cache)).

### Symptom: "I could not find that information in the uploaded documents."

**Cause**: The retriever didn't find relevant chunks for your query.

**Fix**:
1. Check **"🔍 Raw retrieved chunks"** to see what was retrieved
2. Try rephrasing your question with keywords from the document
3. Check that the document actually contains the answer

### Symptom: Chroma DB errors or "file locked" messages

**Cause**: Another process (or a crashed Streamlit instance) is holding the DB files open.

**Fix**:
1. Stop all Streamlit processes
2. Delete the `db_runs/` folder (see [How to Delete Old DB Cache](#how-to-delete-old-db-cache))
3. Restart the app

### Symptom: Import error for `EnsembleRetriever`

**Cause**: `langchain.retrievers` module doesn't have `EnsembleRetriever` in your installed version.

**Fix**: Already fixed in the current code by using a custom `SimpleEnsembleRetriever` class. If you see this error, make sure you're running the latest `raggemini.py`.

---

## What Was Fixed

| Original Bug | Impact | Fix Applied |
|-------------|--------|-------------|
| `from langchain_classic.retrievers import EnsembleRetriever` | Module doesn't exist — import error | Replaced with `from langchain.retrievers` then with a custom `SimpleEnsembleRetriever` class |
| `from langchain.retrievers import EnsembleRetriever` (red underline) | Import fails in newer langchain versions | Built a custom `SimpleEnsembleRetriever` that doesn't depend on `langchain.retrievers` module |
| Fixed `./db` path used for Chroma | If run from different directory, DB path is wrong | Changed to absolute path using `os.path.dirname(os.path.abspath(__file__))` |
| Single `./db` directory locked by other processes | `shutil.rmtree` silently fails, DB not created | Now uses **unique DB directory per run** (`db_runs/db_xxxxxxx`) with UUID |
| `source` variable undefined in source display | `NameError` crash when showing sources | Fixed to use `doc.metadata.get("source", "Unknown")` |
| `@st.cache_resource` with no invalidation key | Old documents stayed cached forever even after deletion | Added `files_hash` (MD5 of filenames + sizes + mtimes) as cache key |
| System prompt was domain-gated to Kubernetes/AWS | BSOD questions rejected even if document had the answer | Relaxed prompt to answer ANY topic found in the documents |
| PowerShell `rmdir /s /q db 2>nul` failed | PowerShell treats `2>nul` as file redirect | Updated instructions to use `Remove-Item -Recurse -Force -ErrorAction SilentlyContinue` |
| Stale `my_document.pdf` sources still appearing | Old cached index returning deleted documents | Auto-cleanup of old DB directories + hash-based cache invalidation |
| No visibility into what was loaded or retrieved | Hard to debug why answers are wrong | Added Streamlit sidebar with: loaded files list, chunk count, errors, DB path, raw chunks |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

MIT License — feel free to use, modify, and distribute.

---

## Author

Built for IT Support document Q&A using Google Gemini and LangChain.

For questions or issues, check the **Debug Info** sidebar in the app or review the troubleshooting section above.
