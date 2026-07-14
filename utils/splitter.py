"""Split page-level documents into overlapping chunks for retrieval."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 200    # overlap so answers don't get cut mid-sentence


def split_documents(docs):
    """
    Split documents into chunks. Page/source metadata is preserved
    on every chunk, which is what lets us cite sources later.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # "।" and "॥" are Devanagari sentence-ending marks (used in Nepali/Hindi)
        # — without them, Nepali text only splits on paragraphs/spaces and chunks
        # end up cutting sentences in half.
        separators=["\n\n", "\n", "।", "॥", ". ", " ", ""],
    )
    return splitter.split_documents(docs)