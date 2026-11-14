"""Data models and structured schemas for paper summaries and literature reviews."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata extracted or associated with an academic paper."""
    title: str = Field(..., description="Title of the paper")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    source: str = Field(..., description="File path or URL of the paper")
    year: Optional[int] = Field(default=None, description="Publication year if identifiable")
    page_count: int = Field(default=1, description="Total number of pages processed")


class RetrievedEvidence(BaseModel):
    """Specific excerpt retrieved from the paper supporting the summary."""
    aspect: str = Field(..., description="Section aspect: Abstract, Main Argument, Methodology, Implications")
    page: Optional[int] = Field(default=None, description="Source page number")
    snippet: str = Field(..., description="Relevant text excerpt from the document")


class PaperSummary(BaseModel):
    """Consistent structured summary report for an academic paper."""
    paper_id: str = Field(..., description="Unique slug or identifier for the paper")
    metadata: PaperMetadata = Field(..., description="Paper metadata (title, authors, source)")
    abstract: str = Field(..., description="Concise synopsis of the paper context, problem, and objective")
    main_argument: str = Field(..., description="Core thesis, central hypotheses, and foundational arguments")
    methodology: str = Field(..., description="Research design, datasets, models, experiments, and techniques used")
    key_findings: List[str] = Field(default_factory=list, description="Primary empirical or theoretical results")
    implications: str = Field(..., description="Broader significance, theoretical contributions, and practical impact")
    limitations: Optional[str] = Field(default=None, description="Limitations or caveats identified by the authors")
    evidence: List[RetrievedEvidence] = Field(default_factory=list, description="Retrieved citation snippets from RAG")

    def to_markdown(self) -> str:
        """Format the summary into a publication-ready Markdown report."""
        authors_str = ", ".join(self.metadata.authors) if self.metadata.authors else "Unknown Authors"
        year_str = f" ({self.metadata.year})" if self.metadata.year else ""
        findings_md = "\n".join(f"- {f}" for f in self.key_findings)

        evidence_md = ""
        if self.evidence:
            evidence_md = "\n\n### 🔍 Grounded RAG Evidence (Retrieved Chunks)\n"
            for ev in self.evidence[:4]:
                page_info = f" (Page {ev.page})" if ev.page else ""
                snippet_clean = ev.snippet.strip().replace("\n", " ")
                if len(snippet_clean) > 200:
                    snippet_clean = snippet_clean[:200] + "..."
                evidence_md += f"- **[{ev.aspect}{page_info}]**: *\"{snippet_clean}\"*\n"

        limitations_sec = ""
        if self.limitations:
            limitations_sec = f"\n\n### Limitations & Boundary Conditions\n{self.limitations}"

        return f"""# Structured Paper Summary: {self.metadata.title}

**Authors**: {authors_str}{year_str}  
**Source**: `{self.metadata.source}`  
**Pages**: {self.metadata.page_count}

---

## 1. Abstract & Context
{self.abstract}

---

## 2. Main Argument & Central Thesis
{self.main_argument}

---

## 3. Methodology & Technical Approach
{self.methodology}

---

## 4. Key Findings & Results
{findings_md}

---

## 5. Implications & Significance
{self.implications}{limitations_sec}{evidence_md}
"""


class ThematicCluster(BaseModel):
    """A thematic group in a cross-paper literature review."""
    theme_name: str = Field(..., description="Name of the thematic cluster")
    description: str = Field(..., description="Description of the thematic scope and relevance")
    contributing_papers: List[str] = Field(default_factory=list, description="Titles of contributing papers")
    synthesized_insights: str = Field(..., description="Cross-paper synthesis for this specific theme")
    methodological_patterns: str = Field(..., description="Common methodologies or divergent approaches observed")
    consensus_points: List[str] = Field(default_factory=list, description="Points of strong agreement across papers")
    divergences_or_debates: List[str] = Field(default_factory=list, description="Points of debate, friction, or difference")


class LiteratureReviewOutline(BaseModel):
    """Structured literature review synthesizing multiple papers grouped by theme."""
    title: str = Field(default="Literature Review: Thematic Synthesis", description="Title of the review")
    total_papers: int = Field(..., description="Number of papers included")
    executive_summary: str = Field(..., description="High-level overview of the literature landscape")
    themes: List[ThematicCluster] = Field(default_factory=list, description="Papers grouped by theme")
    comparative_analysis: str = Field(..., description="Cross-theme comparative analysis and overarching trends")
    identified_research_gaps: List[str] = Field(default_factory=list, description="Emerging gaps and open research questions")
    paper_references: List[Dict[str, Any]] = Field(default_factory=list, description="List of paper metadata references")

    def to_markdown(self) -> str:
        """Format the literature review into a structured Markdown document."""
        themes_md = []
        for i, theme in enumerate(self.themes, 1):
            papers_list = "\n".join(f"  - *{p}*" for p in theme.contributing_papers)
            consensus_list = "\n".join(f"- {c}" for c in theme.consensus_points)
            debates_list = "\n".join(f"- {d}" for d in theme.divergences_or_debates)

            themes_md.append(f"""### Theme {i}: {theme.theme_name}

**Scope**: {theme.description}

**Contributing Papers**:
{papers_list}

**Thematic Synthesis**:
{theme.synthesized_insights}

**Methodological Patterns**:
{theme.methodological_patterns}

**Consensus Points**:
{consensus_list}

**Debates & Divergences**:
{debates_list}
""")

        all_themes_str = "\n---\n\n".join(themes_md)
        gaps_md = "\n".join(f"- {g}" for g in self.identified_research_gaps)

        ref_md = []
        for ref in self.paper_references:
            authors = ", ".join(ref.get("authors", [])) or "Authors"
            title = ref.get("title", "Untitled")
            source = ref.get("source", "")
            ref_md.append(f"- **{title}** - {authors}. Source: `{source}`")
        references_str = "\n".join(ref_md)

        return f"""# {self.title}

**Synthesized Papers**: {self.total_papers}  
**Generated by**: Academic Paper RAG Agent

---

## Executive Summary
{self.executive_summary}

---

## Thematic Groupings & Cross-Paper Analysis

{all_themes_str}

---

## Comparative Synthesis & Overarching Trends
{self.comparative_analysis}

---

## Identified Research Gaps & Future Directions
{gaps_md}

---

## References
{references_str}
"""
