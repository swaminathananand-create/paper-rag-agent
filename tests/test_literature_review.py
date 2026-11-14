"""Tests for the multi-paper literature review agent (offline mode)."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from paper_rag.literature_review import LiteratureReviewAgent
from paper_rag.schemas import LiteratureReviewOutline

SAMPLE_DIR = Path(__file__).parent.parent / "sample_papers"
ALL_TXT_PAPERS = list(SAMPLE_DIR.glob("*.txt"))


@pytest.fixture
def agent(tmp_path):
    return LiteratureReviewAgent(use_api=False, output_dir=tmp_path)


class TestLiteratureReviewAgent:
    def test_review_returns_outline(self, agent):
        sources = [str(p) for p in ALL_TXT_PAPERS[:3]]
        review = agent.generate_review(sources)
        assert isinstance(review, LiteratureReviewOutline)

    def test_review_total_papers_matches_input(self, agent):
        sources = [str(p) for p in ALL_TXT_PAPERS[:3]]
        review = agent.generate_review(sources)
        assert review.total_papers == 3

    def test_review_has_executive_summary(self, agent):
        sources = [str(p) for p in ALL_TXT_PAPERS[:2]]
        review = agent.generate_review(sources)
        assert len(review.executive_summary) > 30

    def test_review_has_themes(self, agent):
        sources = [str(p) for p in ALL_TXT_PAPERS[:3]]
        review = agent.generate_review(sources)
        assert len(review.themes) >= 1
        for theme in review.themes:
            assert len(theme.theme_name) > 0
            assert len(theme.synthesized_insights) > 10

    def test_review_has_research_gaps(self, agent):
        sources = [str(p) for p in ALL_TXT_PAPERS[:3]]
        review = agent.generate_review(sources)
        assert len(review.identified_research_gaps) >= 1

    def test_review_has_paper_references(self, agent):
        sources = [str(p) for p in ALL_TXT_PAPERS[:3]]
        review = agent.generate_review(sources)
        assert len(review.paper_references) == 3

    def test_review_saves_markdown(self, agent):
        sources = [str(p) for p in ALL_TXT_PAPERS[:2]]
        agent.generate_review(sources)
        md_path = agent.output_dir / "literature_review_outline.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "## Executive Summary" in content

    def test_review_saves_json(self, agent):
        sources = [str(p) for p in ALL_TXT_PAPERS[:2]]
        agent.generate_review(sources)
        json_path = agent.output_dir / "literature_review_outline.json"
        assert json_path.exists()
        import json
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "themes" in data

    def test_to_markdown_contains_required_sections(self, agent):
        sources = [str(p) for p in ALL_TXT_PAPERS[:2]]
        review = agent.generate_review(sources)
        md = review.to_markdown()
        assert "Executive Summary" in md
        assert "References" in md
        assert "Research Gaps" in md

    def test_five_papers_review(self, agent):
        sources = [str(p) for p in ALL_TXT_PAPERS[:5]]
        review = agent.generate_review(sources)
        assert review.total_papers == len(sources)
