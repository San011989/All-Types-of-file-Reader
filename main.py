import os
import shutil
import tempfile
import re
import difflib
import pandas as pd
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import pdfplumber
import pytesseract
from PIL import Image

from loaders import (SUPPORTED_EXTS, TABULAR_EXTS, TEXT_EXTS, get_ext, load_tabular, load_tables,
                     load_text, save_tables, delete_tables, tables_to_markdown)

if not shutil.which("tesseract"):
    default_tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(default_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = default_tesseract_path

load_dotenv()

# On Render this points at the persistent disk (see render.yaml).
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
# New collection name: uses COSINE distance (the old default was L2, which made
# the relevance threshold meaningless). Old data in "langchain" is simply ignored.
COLLECTION_NAME = "doc_qa_cosine"
# Cleaned copies of CSV/Excel tables (used for "show me the data").
TABLES_DIR = os.getenv("TABLES_DIR", "./tables")

st.set_page_config(page_title="Document Question with RAG", page_icon="📖", layout="wide")

st.markdown(
    """
    <style>
    .main-title {text-align:center; font-size:40px; font-weight:bold; margin-bottom:5px;}
    .subtitle {text-align:center; font-size:18px; color:gray; margin-bottom:30px;}
    </style>
    <div class="main-title">📖 Document Question with RAG</div>
    <div class="subtitle">Ask questions strictly based on your uploaded PDF, CSV, Excel or text files.</div>
    """,
    unsafe_allow_html=True,
)

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not configured.")
    st.info("Locally: create a .env file with OPENAI_API_KEY=... \n\n"
            "On Render: add it under Environment.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# VECTOR STORE / LLM
# =========================================================

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


try:
    vectorstore = load_vectorstore()
except Exception as e:
    st.error(f"Failed to load Chroma vector database at '{CHROMA_DIR}'.")
    st.exception(e)
    st.stop()


def get_llm(temp: float):
    return ChatOpenAI(model="gpt-4o-mini", temperature=temp)


def count_chunks() -> int:
    try:
        return vectorstore._collection.count()
    except Exception:
        return 0


def list_indexed_sources() -> dict:
    """{source_name: number_of_chunks}"""
    try:
        metas = vectorstore._collection.get(include=["metadatas"])["metadatas"]
    except Exception:
        return {}
    sources = {}
    for m in metas:
        name = (m or {}).get("source", "Unknown")
        sources[name] = sources.get(name, 0) + 1
    return sources


def delete_source(name: str):
    vectorstore._collection.delete(where={"source": name})
    delete_tables(TABLES_DIR, name)


# =========================================================
# PROMPTS
# =========================================================

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a careful QA assistant. Answer using ONLY the context below.

Rules:
1. Use facts from the context. Quote numbers, names and dates exactly as written.
2. The context may contain table rows written as "Column: value | Column: value".
   Each line starting with "Row N" is one record. Read columns carefully.
3. A "TABLE OVERVIEW AND SCHEMA" block contains column names, row counts and
   statistics for the WHOLE table; the row chunks only show some rows.
   For questions about totals/averages/counts, use the schema statistics when
   they apply. If the answer would need rows that are not in the context,
   say that only part of the data was retrieved instead of guessing.
4. If the document mentions the topic but gives no full definition, explain
   what it does say.
5. If the context has nothing relevant, reply exactly:
   "I cannot answer this question based on the provided document."
6. Mention the source file (and page / sheet) when it helps.
7. The user may write commands instead of questions ("show", "list", "summarize"). Treat them as requests: if the context holds the requested data, present it (rows as a Markdown table) instead of refusing.

Context:
{context}"""),
    ("human", "{question}"),
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's last message as ONE standalone search query, using the "
     "chat history to resolve pronouns and references. Do not answer it. "
     "If it is already standalone, return it unchanged.\n\nChat history:\n{history}"),
    ("human", "{question}"),
])


# =========================================================
# FILE EXTRACTION
# =========================================================

def extract_pdf_documents(file_path, source_name):
    documents = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(page_content=text,
                                          metadata={"source": source_name, "page": i}))
    if documents:
        return documents

    # scanned PDF -> OCR
    try:
        with fitz.open(file_path) as pdf:
            for i, page in enumerate(pdf):
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                text = pytesseract.image_to_string(Image.open(BytesIO(pixmap.tobytes("png"))))
                if text and text.strip():
                    documents.append(Document(page_content=text.strip(),
                                              metadata={"source": source_name, "page": i, "ocr": True}))
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError("This PDF looks scanned and the Tesseract OCR engine is not installed.") from error
    return documents


def chunk_prose(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for n, chunk in enumerate(chunks):
        meta = chunk.metadata
        label = f"[Source: {meta.get('source')}"
        if meta.get("page") is not None:
            label += f", Page {meta['page'] + 1}"
        label += "]"
        # context header makes each chunk self-explanatory for embedding + LLM
        chunk.page_content = f"{label}\n{chunk.page_content}"
        meta["type"] = "text"
        meta["chunk_index"] = n
    return chunks


def build_overview_document(documents, source_name):
    sample = "\n\n".join(d.page_content for d in documents[:5])[:8000]
    prompt = ("Provide a comprehensive overview of what this document contains, including "
              "its main topics, subject and key themes, based on these initial pages:\n\n" + sample)
    text = get_llm(0.0).invoke(prompt).content
    return Document(
        page_content=f"DOCUMENT OVERVIEW AND SUMMARY:\nFile Name: {source_name}\n\n{text}",
        metadata={"source": source_name, "type": "overview"},
    )


def process_file(name: str, data: bytes, chunk_size: int, chunk_overlap: int, progress=None):
    ext = get_ext(name)

    if ext in TABULAR_EXTS:
        chunks = load_tabular(data, name)          # already row-aware chunks + schema
        tables = load_tables(data, name)
    elif ext in TEXT_EXTS:
        docs = load_text(data, name)
        chunks = chunk_prose(docs, chunk_size, chunk_overlap) if docs else []
        if chunks:
            chunks.append(build_overview_document(docs, name))
    elif ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            docs = extract_pdf_documents(tmp_path, name)
        finally:
            os.remove(tmp_path)
        chunks = chunk_prose(docs, chunk_size, chunk_overlap) if docs else []
        if chunks:
            chunks.append(build_overview_document(docs, name))
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not chunks:
        raise ValueError("No readable content found in this file.")

    # replace the previous version of the same file, keep other files
    delete_source(name)
    if ext in TABULAR_EXTS:
        save_tables(TABLES_DIR, name, tables)

    batch = 100
    for start in range(0, len(chunks), batch):
        vectorstore.add_documents(chunks[start:start + batch])
        if progress:
            progress.progress(min(1.0, (start + batch) / len(chunks)))
    return len(chunks)


# =========================================================
# RETRIEVAL
# =========================================================

def format_documents(docs):
    out = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata or {}
        where = f"Source: {m.get('source', 'Unknown')}"
        if m.get("page") is not None:
            where += f", Page: {m['page'] + 1}"
        if m.get("sheet"):
            where += f", Sheet: {m['sheet']}"
        if m.get("row_start"):
            where += f", Rows: {m['row_start']}-{m['row_end']}"
        out.append(f"Document {i}\n{where}\nContent:\n{doc.page_content}")
    return "\n\n".join(out)


def is_overview_query(question: str) -> bool:
    keys = ["summarize", "summary", "overview", "what is in", "what is inside",
            "what is the document about", "what is this file", "what does the file",
            "about the document", "about this document", "describe the file",
            "what columns", "which columns", "how many rows", "how many columns",
            "show me the data", "show the data", "show data", "show me data",
            "what data", "preview", "list all", "what is in the file"]
    q = question.lower()
    return any(k in q for k in keys)


def is_show_data_query(question: str) -> bool:
    q = question.lower().strip(" ?.!")
    words = q.split()
    verbs = ("show", "display", "print", "view", "see", "preview")
    nouns = ("data", "table", "rows", "file", "sheet", "content", "contents", "records")
    return (len(words) <= 6 and any(v in words for v in verbs)
            and any(n in words for n in nouns))


def get_overview_docs():
    try:
        res = vectorstore.get(where={"type": "overview"})
        return [Document(page_content=t, metadata=(res["metadatas"][i] or {}))
                for i, t in enumerate(res.get("documents") or [])]
    except Exception:
        return []


def retrieve(question, k, threshold):
    results = vectorstore.similarity_search_with_score(question, k=k)
    docs = [d for d, s in results if s <= threshold]
    if not docs:
        # Nothing was close enough (vague question such as "show me the data").
        # Hand the 3 nearest chunks to the LLM; its prompt still makes it answer
        # "cannot answer" if they are irrelevant.
        docs = [d for d, _ in results[:3]]
    return docs, [round(float(s), 3) for _, s in results]


def condense_question(question, temp):
    history = [m for m in st.session_state.messages[:-1]][-6:]
    if not history:
        return question
    text = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history)
    try:
        chain = CONDENSE_PROMPT | get_llm(0.0) | StrOutputParser()
        return chain.invoke({"history": text, "question": question}).strip() or question
    except Exception:
        return question




def extract_participant_query(question: str):
    """
    Extract participant names only when a specific person is requested.
    Generic queries such as 'show me the data' must not be treated
    as participant-specific requests.
    """

    q = " ".join(question.strip().split())

    patterns = [
        r"^(?:please\s+)?(?:show|display|give|find|list|filter|view)\s+"
        r"(?:me\s+)?only\s+(.+?)\s+"
        r"(?:data|records|rows|scores|marks)[?.!]*$",

        r"^(?:please\s+)?(?:show|display|give|find|list|filter|view)\s+"
        r"(?:me\s+)?(.+?)\s+"
        r"(?:data|records|rows|scores|marks)\s+only[?.!]*$",

        r"^(?:please\s+)?(?:show|display|give|find|list|filter|view)\s+"
        r"(?:me\s+)?(.+?)\s+"
        r"(?:data|records|rows|scores|marks)[?.!]*$",
    ]

    for pattern in patterns:
        match = re.match(pattern, q, flags=re.IGNORECASE)

        if match:
            name = match.group(1).strip(" '\"")

            # Remove generic words that are not participant names.
            generic_words = {
                "the", "all", "complete", "entire",
                "full", "overall", "this", "these"
            }

            if name.casefold() in generic_words:
                return None

            name = re.sub(
                r"^only\s+", "", name,
                flags=re.IGNORECASE
            ).strip()

            return name if name else None

    return None



def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _find_participant_column(columns, requested_name):
    """Match a participant header, allowing minor spelling differences."""
    requested = _normalise_name(requested_name)
    if not requested:
        return None

    best_col, best_score = None, 0.0
    requested_tokens = set(requested.split())

    for col in columns:
        header = str(col)
        # Ignore student ID in parentheses when comparing names.
        name_only = re.sub(r"\s*\([^)]*\)\s*$", "", header)
        candidate = _normalise_name(name_only)
        candidate_tokens = set(candidate.split())

        ratio = difflib.SequenceMatcher(None, requested, candidate).ratio()
        if requested_tokens and candidate_tokens:
            overlap = len(requested_tokens & candidate_tokens) / len(requested_tokens)
            score = max(ratio, overlap)
        else:
            score = ratio

        if score > best_score:
            best_col, best_score = col, score

    # 0.70 tolerates common variants such as Bharat/Bharath,
    # while still requiring a reasonably close participant name.
    return best_col if best_score >= 0.70 else None



def _markdown_table(df):
    """
    Convert DataFrame into a Markdown table.
    Compatible with newer Pandas versions.
    """

    safe = df.fillna("").astype(str).copy()

    # Escape special Markdown characters.
    for col in safe.columns:
        safe[col] = safe[col].str.replace(
            "|", r"\|", regex=False
        ).str.replace(
            "\n", " ", regex=False
        )

    headers = list(safe.columns)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]

    for row in safe.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(row) + " |"
        )

    return "\n".join(lines)



def get_participant_data(requested_name: str):
    """Read stored table CSVs and return only the topic column + matched participant."""
    if not os.path.isdir(TABLES_DIR):
        return "", None

    sections = []
    matched_header = None

    for filename in sorted(os.listdir(TABLES_DIR)):
        if not filename.lower().endswith(".csv"):
            continue

        filepath = os.path.join(TABLES_DIR, filename)
        try:
            df = pd.read_csv(
                filepath, dtype=str, keep_default_na=False, encoding="utf-8-sig"
            )
        except Exception:
            continue

        if df.empty or len(df.columns) < 2:
            continue

        participant_col = _find_participant_column(df.columns[1:], requested_name)
        if participant_col is None:
            continue

        # Keep the assessment-topic column and exactly one participant column.
        filtered = df[[df.columns[0], participant_col]]
        matched_header = str(participant_col)

        source, _, sheet = filename[:-4].partition("__")
        sections.append(
            f"**{source} — sheet {sheet}**\n\n"
            f"Participant: **{participant_col}**\n\n"
            f"{_markdown_table(filtered)}"
        )

    return "\n\n".join(sections), matched_header


def generate_answer(question, k, threshold, temp):
    # STEP 1: Handle participant-specific Excel/CSV requests BEFORE
    # the generic "show data" preview logic.
    participant_name = extract_participant_query(question)

    if participant_name:
        filtered_data, matched_header = get_participant_data(participant_name)

        if filtered_data:
            return {
                "answer": filtered_data,
                "source_type": f"📊 Filtered participant data — {matched_header}",
                "documents": [],
                "scores": [],
                "query": question,
            }

        return {
            "answer": (
                f"No matching participant column was found for **{participant_name}**. "
                "Please check the spelling. You can also use the exact participant "
                "name shown in the Excel column heading."
            ),
            "source_type": "⚠️ Participant not found",
            "documents": [],
            "scores": [],
            "query": question,
        }

    # STEP 2: Show the complete table only for a general data-preview request.
    if is_show_data_query(question):
        preview = tables_to_markdown(TABLES_DIR, limit=50)
        if preview:
            return {
                "answer": preview,
                "source_type": "📊 Table preview (stored file)",
                "documents": [],
                "scores": [],
                "query": question,
            }

    # STEP 3: Normal RAG retrieval for other questions.
    search_query = condense_question(question, temp)
    docs, scores = retrieve(search_query, k, threshold)

    if is_overview_query(question):
        overview = get_overview_docs()
        seen = {d.page_content for d in overview}
        docs = overview + [d for d in docs if d.page_content not in seen]

    if not docs:
        return {
            "answer": "I cannot answer this question based on the provided document.",
            "source_type": "⚠️ No Relevant Context Found",
            "documents": [],
            "scores": scores,
            "query": search_query,
        }

    chain = RAG_PROMPT | get_llm(temp) | StrOutputParser()
    answer = chain.invoke({
        "context": format_documents(docs),
        "question": question,
    })
    return {
        "answer": answer,
        "source_type": "📚 RAG — Uploaded Documents",
        "documents": docs,
        "scores": scores,
        "query": search_query,
    }

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("📄 Upload & Database")

    uploaded_files = st.file_uploader(
        "Upload PDF, CSV, Excel or text files",
        type=SUPPORTED_EXTS,
        accept_multiple_files=True,
    )

    with st.expander("Chunking settings (PDF / text only)"):
        chunk_size = st.slider("Chunk size (characters)", 500, 2500, 1200, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)
        st.caption("CSV/Excel are always split by whole rows, with the column header repeated.")

    if uploaded_files and st.button("Process and Add to Database", use_container_width=True):
        for f in uploaded_files:
            bar = st.progress(0.0, text=f"Processing {f.name}...")
            try:
                n = process_file(f.name, f.getvalue(), chunk_size, min(chunk_overlap, chunk_size // 2), bar)
                st.success(f"{f.name}: stored {n} chunks")
            except Exception as e:
                st.error(f"{f.name}: {e}")
            finally:
                bar.empty()
        st.session_state.messages = []
        st.rerun()

    st.subheader("Indexed files")
    sources = list_indexed_sources()
    if not sources:
        st.info("Nothing indexed yet.")
    for src, n in sources.items():
        c1, c2 = st.columns([4, 1])
        c1.write(f"**{src}** — {n} chunks")
        if c2.button("🗑️", key=f"del_{src}", help=f"Remove {src}"):
            delete_source(src)
            st.rerun()

    if st.button("🗑️ Clean Database", use_container_width=True):
        try:
            vectorstore.delete_collection()
            shutil.rmtree(TABLES_DIR, ignore_errors=True)
            st.cache_resource.clear()
            st.session_state.messages = []
            st.rerun()
        except Exception as e:
            st.error(f"Error cleaning database: {e}")

    st.divider()
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.header("📚 RAG Settings")
    top_k = st.slider("Chunks to retrieve", 1, 20, 8, 1)
    relevance_threshold = st.slider(
        "Max cosine distance", 0.1, 1.5, 0.8, 0.05,
        help="Chunks farther than this are ignored. Lower = stricter. "
             "Raise it if you get 'cannot answer' too often.")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# CHAT
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "source_type" in message:
            st.caption(message["source_type"])

ready = count_chunks() > 0
if not ready:
    st.info("💡 Upload a file and click 'Process and Add to Database' to enable chat.")

query = st.chat_input(
    "Ask a question about your documents..." if ready else "🔒 Process a file first...",
    disabled=not ready,
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔎 Searching documents..."):
            try:
                result = generate_answer(query, top_k, relevance_threshold, temperature)
                st.caption(result["source_type"])
                st.markdown(result["answer"])

                with st.expander("🛠️ Debug Info & Retrieved Chunks"):
                    st.write(f"**Search query used:** {result['query']}")
                    st.write(f"**Distances of top results:** {result['scores']}")
                    st.write(f"**Threshold:** {relevance_threshold} — "
                             f"**{len(result['documents'])}** chunk(s) used")
                    for i, doc in enumerate(result["documents"], start=1):
                        m = doc.metadata or {}
                        st.markdown(f"**Chunk {i}** — {m.get('source')} "
                                    f"{('· sheet ' + m['sheet']) if m.get('sheet') else ''} "
                                    f"{('· page ' + str(m['page'] + 1)) if m.get('page') is not None else ''}")
                        st.text(doc.page_content)

                st.session_state.messages.append({
                    "role": "assistant", "content": result["answer"],
                    "source_type": result["source_type"]})
            except Exception as e:
                st.error("An error occurred while generating the answer.")
                st.exception(e)
