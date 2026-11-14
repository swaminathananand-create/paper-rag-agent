"""Tests for the document loader module."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from paper_rag.loader import PaperLoader


SAMPLE_DIR = Path(__file__).parent.parent / "sample_papers"
ATTENTION_TXT = SAMPLE_DIR / "attention_is_all_you_need.txt"
RAG_TXT = SAMPLE_DIR / "retrieval_augmented_generation.txt"


class TestPaperLoaderLocalFile:
    def setup_method(self):
        self.loader = PaperLoader()

    def test_loads_text_file_returns_documents(self):
        docs, title = self.loader.load(str(ATTENTION_TXT))
        assert len(docs) >= 1
        assert all(len(d.page_content) > 0 for d in docs)

    def test_loads_text_file_has_source_metadata(self):
        docs, title = self.loader.load(str(ATTENTION_TXT))
        for doc in docs:
            assert "source" in doc.metadata

    def test_loads_text_infers_title(self):
        docs, title = self.loader.load(str(ATTENTION_TXT))
        assert isinstance(title, str)
        assert len(title) > 5

    def test_multiple_samples_load_correctly(self):
        for sample_file in SAMPLE_DIR.glob("*.txt"):
            docs, title = self.loader.load(str(sample_file))
            assert len(docs) >= 1, f"No docs loaded for {sample_file.name}"

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            self.loader.load("/nonexistent/path/to/paper.txt")
