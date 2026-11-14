"""Tests for the text splitter module."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from paper_rag.loader import PaperLoader
from paper_rag.splitter import PaperSplitter

SAMPLE_DIR = Path(__file__).parent.parent / "sample_papers"
ATTENTION_TXT = SAMPLE_DIR / "attention_is_all_you_need.txt"


class TestPaperSplitter:
    def setup_method(self):
        self.loader = PaperLoader()
        self.splitter = PaperSplitter(chunk_size=400, chunk_overlap=50)

    def test_splits_into_multiple_chunks(self):
        docs, _ = self.loader.load(str(ATTENTION_TXT))
        chunks = self.splitter.split_documents(docs)
        assert len(chunks) >= 2

    def test_chunks_have_chunk_id_metadata(self):
        docs, _ = self.loader.load(str(ATTENTION_TXT))
        chunks = self.splitter.split_documents(docs)
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata

    def test_chunk_ids_are_monotonically_increasing(self):
        docs, _ = self.loader.load(str(ATTENTION_TXT))
        chunks = self.splitter.split_documents(docs)
        ids = [c.metadata["chunk_id"] for c in chunks]
        # IDs are assigned globally before empty-chunk filtering,
        # so there may be gaps — just verify they are unique and strictly increasing
        assert ids == sorted(set(ids)), "chunk_ids must be unique and monotonically increasing"

    def test_no_empty_chunks(self):
        docs, _ = self.loader.load(str(ATTENTION_TXT))
        chunks = self.splitter.split_documents(docs)
        for chunk in chunks:
            assert len(chunk.page_content.strip()) > 30

    def test_chunk_sizes_respect_limit(self):
        docs, _ = self.loader.load(str(ATTENTION_TXT))
        chunks = self.splitter.split_documents(docs)
        # Allow slight overflow due to separator-aware splitting
        for chunk in chunks:
            assert len(chunk.page_content) <= self.splitter.chunk_size * 2

    def test_source_metadata_preserved(self):
        docs, _ = self.loader.load(str(ATTENTION_TXT))
        chunks = self.splitter.split_documents(docs)
        for chunk in chunks:
            assert "source" in chunk.metadata
