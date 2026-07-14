"""loads the pdf and extract text with page metadata"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path):
    """
    Load a single PDF and return one Document per page.
    Each Document keeps metadata: {"source": <filename>, "page": <page number>}
    """
    file_path = Path(file_path)
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = file_path.name
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1

    return docs

def load_pdfs(file_paths):
    """Load multiple PDFs into a single list of page-level Documents."""
    all_docs = []
    for path in file_paths:
        all_docs.extend(load_pdf(path))
    return all_docs
    