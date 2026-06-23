from dotenv import load_dotenv
import os
import hashlib
import shutil
import uuid

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────
FOLDER_PATH = r"C:\Users\USER\Documents\Data\raggeminifiles"
BASE_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_runs")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── PROMPT ───────────────────────────────────────────────
template = """
You are a document Q&A assistant. Answer ONLY from the context below.

STRICT RULES:
1. Use ONLY the context provided. Do not use any outside knowledge.
2. Do NOT mention Kubernetes, AWS, or any domain unless the context does.
3. If the answer is not in the context, reply EXACTLY with this phrase and nothing else:
   "I could not find that information in the uploaded documents."
4. Never explain what you do or don't know. Never suggest alternatives.

Context:
{context}

Question:
{question}

Answer:"""

# ── ENSEMBLE RETRIEVER (with fallback) ─────────────────

# Try to import EnsembleRetriever from available packages
try:
    from langchain.retrievers import EnsembleRetriever as _EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever as _EnsembleRetriever
    except ImportError:
        _EnsembleRetriever = None


class SimpleEnsembleRetriever:
    """Fallback ensemble retriever if langchain's EnsembleRetriever is not available."""
    def __init__(self, retrievers, weights):
        self.retrievers = retrievers
        self.weights = weights

    def invoke(self, query):
        all_docs = []
        for retriever in self.retrievers:
            try:
                docs = retriever.invoke(query)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Retriever error: {e}")
        # Deduplicate by source + page + content prefix
        seen = set()
        unique_docs = []
        for doc in all_docs:
            key = f"{doc.metadata.get('source', '')}:{doc.metadata.get('page', '')}:{doc.page_content[:100]}"
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        return unique_docs[:10]

    def get_relevant_documents(self, query):
        return self.invoke(query)


def make_ensemble_retriever(bm25_retriever, vector_retriever, weights=(0.4, 0.6)):
    """Create an ensemble retriever, using langchain's if available, else custom."""
    if _EnsembleRetriever is not None:
        return _EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=list(weights)
        )
    return SimpleEnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=list(weights)
    )


# ── HELPERS ──────────────────────────────────────────────

def get_files_hash(folder_path):
    """Compute a hash of file names + sizes + modification times to detect changes."""
    if not os.path.exists(folder_path):
        return ""
    files = sorted(
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    )
    hasher = hashlib.md5()
    for f in files:
        fp = os.path.join(folder_path, f)
        hasher.update(f.encode())
        hasher.update(str(os.path.getsize(fp)).encode())
        hasher.update(str(int(os.path.getmtime(fp))).encode())
    return hasher.hexdigest()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_documents(folder_path):
    """Load all supported documents from the given folder."""
    docs = []
    loaded_files = []
    errors = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if not os.path.isfile(file_path):
            continue

        try:
            if file.lower().endswith(".pdf"):
                loaded = PyPDFLoader(file_path).load()
                docs.extend(loaded)
                loaded_files.append(f"{file} ({len(loaded)} pages)")
            elif file.lower().endswith(".txt"):
                loaded = TextLoader(file_path, encoding="utf-8").load()
                docs.extend(loaded)
                loaded_files.append(f"{file} ({len(loaded)} pages)")
            elif file.lower().endswith(".csv"):
                loaded = CSVLoader(file_path).load()
                docs.extend(loaded)
                loaded_files.append(f"{file} ({len(loaded)} rows)")
            elif file.lower().endswith(".docx"):
                loaded = UnstructuredWordDocumentLoader(file_path).load()
                docs.extend(loaded)
                loaded_files.append(f"{file} ({len(loaded)} pages)")
            elif file.lower().endswith(".pptx"):
                loaded = UnstructuredPowerPointLoader(file_path).load()
                docs.extend(loaded)
                loaded_files.append(f"{file} ({len(loaded)} pages)")
            elif file.lower().endswith(".xlsx"):
                loaded = UnstructuredExcelLoader(file_path).load()
                docs.extend(loaded)
                loaded_files.append(f"{file} ({len(loaded)} pages)")
            elif file.lower().endswith(".md"):
                loaded = UnstructuredMarkdownLoader(file_path).load()
                docs.extend(loaded)
                loaded_files.append(f"{file} ({len(loaded)} pages)")
        except Exception as e:
            errors.append(f"{file}: {e}")

    return docs, loaded_files, errors


@st.cache_resource(show_spinner="Building RAG index...")
def build_rag(files_hash):
    """Build the RAG chain and retriever from documents in FOLDER_PATH."""
    print("Loading documents...")
    docs, loaded_files, errors = load_documents(FOLDER_PATH)

    print(f"Loaded {len(docs)} pages from {len(loaded_files)} files")

    if not docs:
        return None, None, loaded_files, errors, 0, ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Create a unique DB directory to avoid file locking issues
    db_id = str(uuid.uuid4())[:8]
    db_path = os.path.join(BASE_DB_DIR, f"db_{db_id}")
    os.makedirs(db_path, exist_ok=True)

    # Clean up old DB directories (keep only the last 3)
    try:
        if os.path.exists(BASE_DB_DIR):
            all_dbs = sorted(
                [d for d in os.listdir(BASE_DB_DIR) if d.startswith("db_")],
                key=lambda x: os.path.getctime(os.path.join(BASE_DB_DIR, x))
            )
            for old_db in all_dbs[:-3]:
                old_path = os.path.join(BASE_DB_DIR, old_db)
                try:
                    shutil.rmtree(old_path)
                except Exception:
                    pass
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_path
    )

    vector = vectorstore.as_retriever(search_kwargs={"k": 10})
    bm25 = BM25Retriever.from_documents(splits)
    bm25.k = 10

    retriever = make_ensemble_retriever(bm25, vector, weights=(0.4, 0.6))

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    prompt = ChatPromptTemplate.from_template(template)

    setup = RunnableParallel(
        context=retriever | RunnableLambda(format_docs),
        question=RunnablePassthrough()
    )

    rag_chain = (
        setup
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, loaded_files, errors, len(splits), db_path


# ── UI ───────────────────────────────────────────────────
st.title("IT Support RAG Assistant")

# ── Sidebar: Debug & Controls ────────────────────────────
with st.sidebar:
    st.header("Controls")

    if st.button("🔄 Force Rebuild Index", help="Clears cache and rebuilds from documents"):
        st.cache_resource.clear()
        # Also nuke any leftover DB folders
        try:
            if os.path.exists(BASE_DB_DIR):
                shutil.rmtree(BASE_DB_DIR)
        except Exception:
            pass
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.header("Debug Info")
    st.write(f"**Document folder:** `{FOLDER_PATH}`")
    st.write(f"**DB base dir:** `{BASE_DB_DIR}`")

    files_hash = get_files_hash(FOLDER_PATH)
    st.write(f"**Files hash:** `{files_hash}`")

    st.write("**Files in folder:**")
    if os.path.exists(FOLDER_PATH):
        for f in os.listdir(FOLDER_PATH):
            st.write(f"  - {f}")
    else:
        st.error(f"Folder does not exist: {FOLDER_PATH}")

    if "last_hash" not in st.session_state:
        st.session_state.last_hash = None

    if "rag_chain" not in st.session_state or st.session_state.last_hash != files_hash:
        with st.spinner("Indexing..."):
            result = build_rag(files_hash)
            st.session_state.rag_chain = result[0]
            st.session_state.retriever = result[1]
            st.session_state.loaded_files = result[2]
            st.session_state.errors = result[3]
            st.session_state.chunk_count = result[4]
            st.session_state.db_path = result[5]
            st.session_state.last_hash = files_hash

    st.write(f"**Files loaded:** {len(st.session_state.get('loaded_files', []))}")
    st.write(f"**Chunks created:** {st.session_state.get('chunk_count', 0)}")
    st.write(f"**DB path:** `{st.session_state.get('db_path', 'N/A')}`")

    with st.expander("Loaded files"):
        for f in st.session_state.get("loaded_files", []):
            st.write(f"- {f}")

    with st.expander("Errors"):
        errors = st.session_state.get("errors", [])
        if errors:
            for e in errors:
                st.error(e)
        else:
            st.write("No errors.")

rag_chain = st.session_state.rag_chain
retriever = st.session_state.retriever

# ── Main query area ──────────────────────────────────────
query = st.text_input("Ask a question about your documents")

if query:
    if rag_chain is None:
        st.error("No documents were loaded. Please check the document folder and click 'Force Rebuild Index'.")
    else:
        with st.spinner("Searching documents..."):
            docs = retriever.invoke(query)
            answer = rag_chain.invoke(query)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            st.write(f"📄 **{os.path.basename(source)}** | Page {page}")
            st.code(doc.page_content[:500])

        # Debug: show all retrieved chunks
        with st.expander("🔍 Raw retrieved chunks"):
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                st.write(f"**Chunk {i}** — {os.path.basename(source)} | Page {page}")
                st.code(doc.page_content)
