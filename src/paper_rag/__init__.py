"""Academic Paper RAG Agent.

Retrieval-Augmented Generation (RAG) agent for academic paper ingestion,
structured summarization, and multi-paper thematic literature reviews.
"""

from paper_rag.schemas import PaperSummary, LiteratureReviewOutline, PaperMetadata
from paper_rag.loader import PaperLoader
from paper_rag.splitter import PaperSplitter
from paper_rag.retriever import PaperRetriever
from paper_rag.summarizer import PaperSummarizer
from paper_rag.literature_review import LiteratureReviewAgent

__all__ = [
    "PaperSummary",
    "LiteratureReviewOutline",
    "PaperMetadata",
    "PaperLoader",
    "PaperSplitter",
    "PaperRetriever",
    "PaperSummarizer",
    "LiteratureReviewAgent",
]
