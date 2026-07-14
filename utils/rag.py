"""
Core RAG pipeline:
  chunks -> embeddings -> ChromaDB -> retrieval -> LLM answer + sources
"""

import re
import time

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from utils.embedding import get_embeddings, get_llm

DB_DIR = "db"
COLLECTION = "pdf_chat"
TOP_K = 6  # number of chunks to retrieve per question (higher helps recall across multiple PDFs)
INDEX_BATCH_SIZE = 50  # chunks embedded per API call, so a rate-limit retry only redoes a small batch
INDEX_MAX_RETRIES = 5

_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s")

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a warm, articulate assistant that answers questions using ONLY
the provided context from uploaded PDF documents. The documents may be in
English, Nepali (नेपाली), or a mix of both.and its crazy

Rules:
- Answer in the same language the question was asked in. If the question is
  in Nepali (Devanagari script), answer fully and fluently in Nepali — do not
  transliterate or switch to English. If the question mixes English and
  Nepali, answer in that same mixed style. If it's in English, answer in
  English, even when the source context is in Nepali (translate naturally).
- Write like a knowledgeable person explaining it to someone, not like a
  search result: clear, well-organized, and natural — use short paragraphs
  or bullet points when that helps readability.
- Be concise but complete. Don't pad the answer with filler.
- If the context does not contain the answer, say so plainly (in the
  question's language) and don't make anything up.
- Never mention "the context" or "the documents provided" — just answer
  naturally, as if you already knew this.

Context:
{context}

Question: {question}

Answer:"""
)


@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """Open (or create) the persistent Chroma collection."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=DB_DIR,
    )


def index_documents(chunks):
    """Embed chunks and store them in ChromaDB. Returns number indexed.

    Embeds in small batches with retry/backoff so a Gemini free-tier
    rate limit (429 RESOURCE_EXHAUSTED) pauses and resumes instead of
    crashing the whole indexing run.
    """
    if not chunks:
        return 0
    vs = get_vectorstore()

    for i in range(0, len(chunks), INDEX_BATCH_SIZE):
        batch = chunks[i : i + INDEX_BATCH_SIZE]
        for attempt in range(INDEX_MAX_RETRIES):
            try:
                vs.add_documents(batch)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" not in str(e) or attempt == INDEX_MAX_RETRIES - 1:
                    raise
                match = _RETRY_DELAY_RE.search(str(e))
                delay = float(match.group(1)) + 1 if match else min(60, 5 * (2**attempt))
                st.toast(
                    f"Gemini embedding quota hit — waiting {delay:.0f}s before retrying...",
                    icon="⏳",
                )
                time.sleep(delay)

    return len(chunks)


def clear_index():
    """Delete everything in the collection (fresh start)."""
    vs = get_vectorstore()
    ids = vs.get().get("ids", [])
    if ids:
        vs.delete(ids=ids)


def delete_source(source_name):
    """Delete every chunk that came from a single file, by its source name."""
    vs = get_vectorstore()
    ids = vs.get(where={"source": source_name}).get("ids", [])
    if ids:
        vs.delete(ids=ids)


def get_source_counts():
    """Return {source_filename: chunk_count} for everything currently indexed."""
    vs = get_vectorstore()
    metadatas = vs.get().get("metadatas", []) or []
    counts = {}
    for meta in metadatas:
        src = meta.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts


def retrieve(question, k=TOP_K):
    """Vector similarity search over indexed chunks."""
    vs = get_vectorstore()
    return vs.similarity_search(question, k=k)


def format_context(docs):
    """Join retrieved chunks into a single context block with source labels."""
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{src} — page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _extract_text(content):
    """
    Gemini 3+ can return content as a list of blocks (text + internal
    'thought signature' metadata) instead of a plain string.
    This pulls out only the actual answer text.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()

    return str(content)


def answer_question(question):
    """
    Full RAG step: retrieve -> prompt -> LLM.
    Returns {"answer": str, "sources": list of Documents}
    """
    docs = retrieve(question)

    if not docs:
        return {
            "answer": "No documents are indexed yet. Upload a PDF first.",
            "sources": [],
        }

    llm = get_llm()
    chain = RAG_PROMPT | llm
    response = chain.invoke(
        {"context": format_context(docs), "question": question}
    )

    return {"answer": _extract_text(response.content), "sources": docs}
