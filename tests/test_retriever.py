"""Tests for the BM25 retriever module (offline mode)."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from paper_rag.loader import PaperLoader
from paper_rag.splitter import PaperSplitter
from paper_rag.retriever import BM25Retriever, PaperRetriever

SAMPLE_DIR = Path(__file__).parent.parent / "sample_papers"
ATTENTION_TXT = SAMPLE_DIR / "attention_is_all_you_need.txt"
RAG_TXT = SAMPLE_DIR / "retrieval_augmented_generation.txt"


def get_chunks(path, chunk_size=500):
    loader = PaperLoader()
    splitter = PaperSplitter(chunk_size=chunk_size, chunk_overlap=80)
    docs, _ = loader.load(str(path))
    return splitter.split_documents(docs)


class TestBM25Retriever:
    def setup_method(self):
        self.chunks = get_chunks(ATTENTION_TXT)
        self.retriever = BM25Retriever(self.chunks)

    def test_returns_documents(self):
        results = self.retriever.similarity_search("self-attention mechanism transformer", k=3)
        assert len(results) >= 1

    def test_k_limits_results(self):
        results = self.retriever.similarity_search("methodology architecture", k=2)
        assert len(results) <= 2

    def test_methodology_query_retrieves_relevant_content(self):
        results = self.retriever.similarity_search("encoder decoder layers multi-head attention", k=3)
        combined_text = " ".join(d.page_content.lower() for d in results)
        # At least one result should mention attention or encoder
        assert "attention" in combined_text or "encoder" in combined_text

    def test_returns_fallback_when_no_matches(self):
        # Extremely rare query that likely has no term match
        results = self.retriever.similarity_search("xylophone quantum crystallography astro", k=2)
        # Should fall back to returning first N documents
        assert len(results) >= 1


class TestPaperRetriever:
    def setup_method(self):
        self.chunks = get_chunks(ATTENTION_TXT)
        # Force offline BM25 mode (no API)
        self.retriever = PaperRetriever(self.chunks, use_api=False)

    def test_retrieve_returns_documents(self):
        results = self.retriever.retrieve("self-attention mechanism")
        assert len(results) >= 1

    def test_retrieve_aspects_returns_all_five_aspects(self):
        aspects = self.retriever.retrieve_aspects()
        assert "abstract" in aspects
        assert "main_argument" in aspects
        assert "methodology" in aspects
        assert "key_findings" in aspects
        assert "implications" in aspects

    def test_all_aspect_results_have_content(self):
        aspects = self.retriever.retrieve_aspects()
        for aspect, docs in aspects.items():
            assert len(docs) >= 1, f"No chunks retrieved for aspect: {aspect}"
            for doc in docs:
                assert len(doc.page_content) > 10

    def test_engine_name_is_bm25_offline(self):
        assert "BM25" in self.retriever.engine_name
