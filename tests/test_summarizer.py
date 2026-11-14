"""Tests for the end-to-end RAG summarizer pipeline (offline mock mode)."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from paper_rag.summarizer import PaperSummarizer
from paper_rag.schemas import PaperSummary

SAMPLE_DIR = Path(__file__).parent.parent / "sample_papers"
ATTENTION_TXT = SAMPLE_DIR / "attention_is_all_you_need.txt"
RAG_TXT = SAMPLE_DIR / "retrieval_augmented_generation.txt"


@pytest.fixture
def summarizer(tmp_path):
    return PaperSummarizer(use_api=False, output_dir=tmp_path)


class TestPaperSummarizerOffline:
    def test_summarize_returns_paper_summary(self, summarizer):
        summary = summarizer.summarize(str(ATTENTION_TXT))
        assert isinstance(summary, PaperSummary)

    def test_summary_has_title(self, summarizer):
        summary = summarizer.summarize(str(ATTENTION_TXT))
        assert len(summary.metadata.title) > 5

    def test_summary_has_non_empty_sections(self, summarizer):
        summary = summarizer.summarize(str(ATTENTION_TXT))
        assert len(summary.abstract) > 20
        assert len(summary.main_argument) > 20
        assert len(summary.methodology) > 20
        assert len(summary.implications) > 20

    def test_summary_has_key_findings(self, summarizer):
        summary = summarizer.summarize(str(ATTENTION_TXT))
        assert isinstance(summary.key_findings, list)
        assert len(summary.key_findings) >= 1

    def test_summary_has_evidence(self, summarizer):
        summary = summarizer.summarize(str(ATTENTION_TXT))
        assert isinstance(summary.evidence, list)
        assert len(summary.evidence) >= 1

    def test_summary_paper_id_is_slug(self, summarizer):
        summary = summarizer.summarize(str(ATTENTION_TXT))
        # paper_id should only have safe slug characters
        import re
        assert re.match(r"^[a-zA-Z0-9_\-]+$", summary.paper_id)

    def test_summary_saves_markdown_file(self, summarizer):
        summary = summarizer.summarize(str(ATTENTION_TXT))
        md_file = summarizer.output_dir / f"{summary.paper_id}_summary.md"
        assert md_file.exists()
        content = md_file.read_text(encoding="utf-8")
        assert "## 1. Abstract" in content

    def test_summary_saves_json_file(self, summarizer):
        summary = summarizer.summarize(str(ATTENTION_TXT))
        json_file = summarizer.output_dir / f"{summary.paper_id}_summary.json"
        assert json_file.exists()
        import json
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert "metadata" in data

    def test_to_markdown_contains_all_sections(self, summarizer):
        summary = summarizer.summarize(str(ATTENTION_TXT))
        md = summary.to_markdown()
        assert "## 1. Abstract" in md
        assert "## 2. Main Argument" in md
        assert "## 3. Methodology" in md
        assert "## 4. Key Findings" in md
        assert "## 5. Implications" in md

    def test_two_different_papers_produce_different_output(self, summarizer):
        s1 = summarizer.summarize(str(ATTENTION_TXT))
        s2 = summarizer.summarize(str(RAG_TXT))
        assert s1.abstract != s2.abstract
