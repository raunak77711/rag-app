"""
PDF Chat (RAG) — Streamlit UI
Run with:  streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from utils.loader import load_pdfs
from utils.splitter import split_documents
from utils.rag import (
    answer_question,
    clear_index,
    delete_source,
    get_source_counts,
    index_documents,
)

st.set_page_config(page_title="Chat with your PDFs", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C63FF, #00B4A6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: -0.6rem;
    }
    [data-testid="stChatMessage"] {
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources"}]
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def render_sources(sources):
    with st.expander(f"📎 Sources ({len(sources)})"):
        for doc in sources:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            with st.container(border=True):
                st.markdown(f"**{src}** · page {page}")
                st.caption(doc.page_content[:400] + "...")


# sidebar
with st.sidebar:
    st.markdown("### 📚 pdf chat")
    st.caption("Upload PDFs, then ask questions — in English or नेपाली.")

    uploaded = st.file_uploader(
        "Upload PDF(s)",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if st.button("Index documents", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("Upload at least one PDF first.")
        else:
            already_indexed = set(get_source_counts().keys())
            new_files = [f for f in uploaded if f.name not in already_indexed]
            if not new_files:
                st.info("Those files are already indexed.")
            else:
                with st.spinner("Extracting → chunking → embedding..."):
                    tmp_paths = []
                    for f in new_files:
                        tmp = tempfile.NamedTemporaryFile(
                            delete=False, suffix=f"__{f.name}"
                        )
                        tmp.write(f.getvalue())
                        tmp.close()
                        tmp_paths.append((tmp.name, f.name))

                    docs = []
                    for tmp_path, original_name in tmp_paths:
                        pages = load_pdfs([tmp_path])
                        for p in pages:
                            p.metadata["source"] = original_name
                        docs.extend(pages)
                        Path(tmp_path).unlink(missing_ok=True)

                    chunks = split_documents(docs)
                    n = index_documents(chunks)

                st.success(f"Indexed {n} chunks from {len(new_files)} file(s).")
                st.rerun()

    source_counts = get_source_counts()

    if source_counts:
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Files", len(source_counts))
        c2.metric("Chunks", sum(source_counts.values()))

        st.subheader("Indexed files")
        for name, count in source_counts.items():
            with st.container(border=True):
                fc1, fc2 = st.columns([5, 1])
                fc1.markdown(f"📄 **{name}**")
                fc1.caption(f"{count} chunks")
                if fc2.button("🗑", key=f"remove_{name}", help=f"Remove {name}"):
                    delete_source(name)
                    st.rerun()

        if st.button("Clear all", use_container_width=True):
            clear_index()
            st.session_state.messages = []
            st.session_state.uploader_key += 1
            st.rerun()

# main chat area
st.markdown('<p class="app-title">Chat with your documents</p>', unsafe_allow_html=True)
st.caption("Ask in English or नेपाली — answers come back in the same language.")

source_counts = get_source_counts()

if not source_counts and not st.session_state.messages:
    st.info(
        "👈 Upload one or more PDFs in the sidebar and click **Index documents** "
        "to get started. PDFs in English and Nepali (नेपाली) are both supported."
    )

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "📚"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

question = None

if source_counts and not st.session_state.messages:
    st.caption("Try asking:")
    suggestions = [
        "Summarize these documents",
        "यी कागजातहरूको मुख्य बुँदा के हो?",
        "What are the key takeaways?",
    ]
    cols = st.columns(len(suggestions))
    for col, suggestion in zip(cols, suggestions):
        if col.button(suggestion, use_container_width=True):
            question = suggestion

typed_question = st.chat_input("Ask something about your PDFs...")
if typed_question:
    question = typed_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Thinking..."):
            result = answer_question(question)

        st.markdown(result["answer"])
        if result["sources"]:
            render_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
