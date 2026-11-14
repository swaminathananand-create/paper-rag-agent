"""Multi-paper literature review agent: groups papers by theme and synthesizes cross-paper outlines."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from paper_rag.config import Config
from paper_rag.schemas import (
    PaperSummary,
    ThematicCluster,
    LiteratureReviewOutline,
)
from paper_rag.summarizer import PaperSummarizer


LIT_REVIEW_PROMPT = """You are a senior academic research scholar writing an exhaustive, publishable literature review outline synthesizing multiple research papers.

Below are the structured summaries of {paper_count} academic papers:

{papers_data}

Your task is to synthesize these papers into a comprehensive, high-level literature review outline.
Strict requirements:
1. Group the papers by 2 to 4 overarching research themes (e.g., "Model Architectures & Attention Mechanisms", "Retrieval & Grounding Paradigms", "Reasoning & Agentic Workflows").
2. Compare and contrast their methodologies, highlight points of consensus, and identify debates or contradictions.
3. Formulate key unexplored research gaps and future directions.
4. Output STRICT JSON conforming to the following schema:

{{
  "title": "A Synthesis of Recent Advances: [Domain/Topic]",
  "executive_summary": "Comprehensive 3-5 sentence synthesis of the current state of literature across all included papers.",
  "themes": [
    {{
      "theme_name": "Name of Theme 1",
      "description": "Scope and theoretical context of this theme.",
      "contributing_papers": ["Exact Title of Paper A", "Exact Title of Paper B"],
      "synthesized_insights": "Cross-paper synthesis explaining how these papers relate, build on, or complement each other.",
      "methodological_patterns": "Analysis of shared experimental setups, architectures, or divergent techniques.",
      "consensus_points": ["Consensus point 1", "Consensus point 2"],
      "divergences_or_debates": ["Divergence or debate point 1"]
    }}
  ],
  "comparative_analysis": "In-depth cross-thematic comparison discussing historical progression, paradigm shifts, and architectural trade-offs.",
  "identified_research_gaps": [
    "Unexplored gap or limitation 1",
    "Unexplored gap or limitation 2",
    "Promising future research vector 3"
  ]
}}

JSON response:
"""


class LiteratureReviewAgent:
    """Agent that ingests multiple papers and generates a cross-paper thematic literature review."""

    def __init__(self, use_api: Optional[bool] = None, output_dir: Path = None):
        self.use_api = Config.has_api_key() if use_api is None else use_api
        self.output_dir = output_dir or Config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summarizer = PaperSummarizer(use_api=self.use_api, output_dir=self.output_dir)

    def generate_review(
        self,
        sources_or_summaries: List[Any],
        review_title: Optional[str] = None
    ) -> LiteratureReviewOutline:
        """Generates a thematic literature review outline from paper sources or summaries.

        Args:
            sources_or_summaries: List of paper URLs/paths OR pre-computed PaperSummary objects.
            review_title: Optional title override.

        Returns:
            LiteratureReviewOutline object.
        """
        # Ensure all items are PaperSummary objects
        summaries: List[PaperSummary] = []
        for item in sources_or_summaries:
            if isinstance(item, PaperSummary):
                summaries.append(item)
            elif isinstance(item, str):
                summary = self.summarizer.summarize(item)
                summaries.append(summary)
            else:
                raise ValueError(f"Unsupported item type in review: {type(item)}")

        if not summaries:
            raise ValueError("At least one paper is required to generate a literature review.")

        # Build paper references table
        references = []
        for s in summaries:
            references.append({
                "title": s.metadata.title,
                "authors": s.metadata.authors,
                "source": s.metadata.source,
                "year": s.metadata.year,
            })

        # Synthesize via LLM or deterministic fallback
        if self.use_api:
            review = self._synthesize_with_gemini(summaries, references, review_title)
        else:
            review = self._synthesize_mock_fallback(summaries, references, review_title)

        self.save_review(review)
        return review

    def _synthesize_with_gemini(
        self,
        summaries: List[PaperSummary],
        references: List[Dict[str, Any]],
        review_title: Optional[str] = None,
    ) -> LiteratureReviewOutline:
        """Uses Gemini LLM to synthesize the multi-paper thematic review."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        papers_payload = []
        for i, s in enumerate(summaries, 1):
            papers_payload.append(f"""### Paper {i}: {s.metadata.title}
- **Authors**: {', '.join(s.metadata.authors)}
- **Abstract**: {s.abstract}
- **Main Argument**: {s.main_argument}
- **Methodology**: {s.methodology}
- **Key Findings**: {'; '.join(s.key_findings)}
- **Implications**: {s.implications}
""")
        papers_text = "\n\n".join(papers_payload)

        prompt = LIT_REVIEW_PROMPT.format(
            paper_count=len(summaries),
            papers_data=papers_text,
        )

        try:
            llm = ChatGoogleGenerativeAI(
                model=Config.DEFAULT_MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=0.3,
            )
            response = llm.invoke(prompt)
            content = response.content
        except Exception:
            llm = ChatGoogleGenerativeAI(
                model=Config.FALLBACK_MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=0.3,
            )
            response = llm.invoke(prompt)
            content = response.content

        parsed = self._clean_and_parse_json(content)

        themes = []
        for t in parsed.get("themes", []):
            themes.append(
                ThematicCluster(
                    theme_name=t.get("theme_name", "General Research Theme"),
                    description=t.get("description", ""),
                    contributing_papers=t.get("contributing_papers", []),
                    synthesized_insights=t.get("synthesized_insights", ""),
                    methodological_patterns=t.get("methodological_patterns", ""),
                    consensus_points=t.get("consensus_points", []),
                    divergences_or_debates=t.get("divergences_or_debates", []),
                )
            )

        return LiteratureReviewOutline(
            title=review_title or parsed.get("title", f"Thematic Literature Review ({len(summaries)} Papers)"),
            total_papers=len(summaries),
            executive_summary=parsed.get("executive_summary", "Synthesis of selected academic papers."),
            themes=themes,
            comparative_analysis=parsed.get("comparative_analysis", ""),
            identified_research_gaps=parsed.get("identified_research_gaps", []),
            paper_references=references,
        )

    def _synthesize_mock_fallback(
        self,
        summaries: List[PaperSummary],
        references: List[Dict[str, Any]],
        review_title: Optional[str] = None,
    ) -> LiteratureReviewOutline:
        """Deterministic thematic clustering fallback when running offline or testing."""
        # Split papers into two thematic groups based on keyword heuristics
        theme1_papers = []
        theme2_papers = []

        for s in summaries:
            text = (s.metadata.title + " " + s.main_argument + " " + s.methodology).lower()
            if any(k in text for k in ["retrieval", "rag", "knowledge", "memory", "database", "search"]):
                theme1_papers.append(s.metadata.title)
            else:
                theme2_papers.append(s.metadata.title)

        if not theme1_papers:
            theme1_papers = [s.metadata.title for s in summaries[:len(summaries)//2 or 1]]
            theme2_papers = [s.metadata.title for s in summaries[len(summaries)//2 or 1:]]
        elif not theme2_papers:
            theme2_papers = theme1_papers[len(theme1_papers)//2:]
            theme1_papers = theme1_papers[:len(theme1_papers)//2]

        cluster1 = ThematicCluster(
            theme_name="Retrieval-Augmented Architecture & External Grounding",
            description="Focuses on integrating non-parametric memory, dense retrieval vectors, and dynamic source context into language generation.",
            contributing_papers=theme1_papers or [s.metadata.title for s in summaries[:2]],
            synthesized_insights="The literature highlights that grounding generative models with dynamic retrieval substantially mitigates hallucinations, improves factual precision, and allows knowledge updates without re-training.",
            methodological_patterns="Widespread adoption of dual-encoder architectures, dense vector indexes (FAISS/Chroma), and iterative query expansion.",
            consensus_points=[
                "External knowledge retrieval outperforms purely parametric memorization on knowledge-intensive tasks.",
                "Chunking granularity and retrieval precision directly bound downstream generation fidelity."
            ],
            divergences_or_debates=[
                "Debate persists between single-step retrieval versus multi-hop agentic reasoning chains."
            ]
        )

        cluster2 = ThematicCluster(
            theme_name="Reasoning, Sequence Modeling & Attention Foundations",
            description="Examines core sequence representations, attention mechanisms, and agentic reasoning loops that form the backbone of modern LLM architectures.",
            contributing_papers=theme2_papers or [s.metadata.title for s in summaries[-2:]],
            synthesized_insights="Investigates inductive biases, self-attention scalability, and synergetic reasoning frameworks that empower autonomous agents to plan and execute multi-step tasks.",
            methodological_patterns="Transformer self-attention layers, cross-entropy sequence optimization, and thought-action-observation prompting loops.",
            consensus_points=[
                "Self-attention provides superior parallelization and long-range dependency capture compared to recurrence.",
                "Interleaving reasoning steps with external tool execution significantly boosts problem-solving accuracy."
            ],
            divergences_or_debates=[
                "Trade-off between quadratic attention compute cost versus linear approximations in long-context domains."
            ]
        )

        return LiteratureReviewOutline(
            title=review_title or f"Thematic Literature Review: Foundations of Modern Agentic & RAG Systems ({len(summaries)} Papers)",
            total_papers=len(summaries),
            executive_summary=(
                f"This literature review outlines a multi-dimensional synthesis across {len(summaries)} foundational research contributions. "
                "The survey bridges architectural innovations in self-attention mechanisms with retrieval-augmented generation paradigms, "
                "charting how modern systems transition from static parameter memory to dynamic, grounded agentic reasoning."
            ),
            themes=[cluster1, cluster2],
            comparative_analysis=(
                "Across the examined papers, a distinct paradigm evolution emerges: early foundational work addressed architectural throughput and representation "
                "(e.g., self-attention mechanisms), while subsequent generations addressed the epistemic boundaries of language models through retrieval augmentation "
                "and agentic tool interaction. Rather than competing, these approaches serve as complementary layers in modern AI engineering."
            ),
            identified_research_gaps=[
                "Real-time indexing latency when scaling retrieval-augmented workflows to millions of dynamic documents.",
                "Robust hallucination verification mechanisms for complex multi-hop academic reasoning.",
                "Unified benchmarks assessing agentic tool usage in highly specialized scientific domains."
            ],
            paper_references=references,
        )

    def _clean_and_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Strips markdown code blocks and safely parses JSON."""
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)
        clean = clean.strip()
        try:
            return json.loads(clean)
        except Exception:
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return {}

    def save_review(self, review: LiteratureReviewOutline) -> Tuple[Path, Path]:
        """Saves literature review outline as Markdown and JSON."""
        md_path = self.output_dir / "literature_review_outline.md"
        json_path = self.output_dir / "literature_review_outline.json"

        md_path.write_text(review.to_markdown(), encoding="utf-8")
        json_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
        return md_path, json_path
