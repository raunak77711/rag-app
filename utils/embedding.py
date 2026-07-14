"""
Embedding + LLM setup, using Gemini.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Return the embedding model — turns text into vectors for search."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


@st.cache_resource(show_spinner=False)
def get_llm(temperature=0.2):
    """Return the chat model — this is what generates the actual answers."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=temperature)