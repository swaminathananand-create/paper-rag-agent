"""Academic text splitting and chunking using RecursiveCharacterTextSplitter."""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from paper_rag.config import Config


class PaperSplitter:
    """Chunks academic documents preserving paragraph semantics and source metadata."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

        # Academic document friendly separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n\n",
                "\n\n## ",
                "\n\n# ",
                "\n\n",
                "\n",
                ". ",
                "; ",
                " ",
                "",
            ],
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of documents/pages into overlapping chunks.

        Enriches each chunk with chunk_index and preserves original page metadata.
        """
        chunks = self.splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            # Ensure snippet doesn't contain orphan null bytes or excessive whitespace
            chunk.page_content = chunk.page_content.replace("\x00", "").strip()

        # Filter out empty or whitespace-only chunks
        valid_chunks = [c for c in chunks if len(c.page_content) > 30]
        return valid_chunks
