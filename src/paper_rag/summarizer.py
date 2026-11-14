"""Core RAG summarization engine: retrieves evidence and generates structured reports."""

import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from langchain_core.prompts import PromptTemplate
from paper_rag.config import Config
from paper_rag.schemas import PaperSummary, PaperMetadata, RetrievedEvidence
from paper_rag.loader import PaperLoader
from paper_rag.splitter import PaperSplitter
from paper_rag.retriever import PaperRetriever


SUMMARY_PROMPT_TEMPLATE = """You are an expert academic research assistant specializing in analyzing complex scientific papers using Retrieval-Augmented Generation (RAG).

Below is verified context retrieved directly from the paper:
{paper_title}

--- RETRIEVED CONTEXT ---
{retrieved_context}
--- END RETRIEVED CONTEXT ---

Analyze the retrieved material and produce a structured, high-fidelity summary in strict JSON format.
Strict rules:
1. Ground all claims STRICTLY in the provided context. Do not invent citations or fabricate methodologies.
2. If certain details (e.g. publication year or authors) are not present in the context, mark them as null or "Unknown".
3. Return ONLY valid JSON matching the following schema:

{{
  "title": "{paper_title}",
  "authors": ["Author 1", "Author 2"],
  "year": 2024,
  "abstract": "A clear 2-4 sentence summary of the paper context, core research problem, and overall objective.",
  "main_argument": "The central thesis, fundamental claims, and motivation of the paper.",
  "methodology": "Detailed description of the technical approach, architecture, dataset, baseline models, and experimental design.",
  "key_findings": [
    "Primary finding or performance benchmark result 1",
    "Primary finding or performance benchmark result 2",
    "Primary finding or theoretical breakthrough 3"
  ],
  "implications": "Broader significance, theoretical impact, real-world applications, and future research directions.",
  "limitations": "Boundary conditions, computational requirements, or acknowledged limitations."
}}

JSON response:
"""


class PaperSummarizer:
    """Orchestrates RAG summarization for a single academic paper."""

    def __init__(self, use_api: Optional[bool] = None, output_dir: Path = None):
        self.use_api = Config.has_api_key() if use_api is None else use_api
        self.output_dir = output_dir or Config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loader = PaperLoader()
        self.splitter = PaperSplitter()

    def summarize(self, source: str) -> PaperSummary:
        """Runs the end-to-end RAG pipeline on a given paper source (path or URL)."""
        # 1. Ingest
        raw_docs, canonical_title = self.loader.load(source)

        # 2. Chunk
        chunks = self.splitter.split_documents(raw_docs)

        # 3. Retrieve
        retriever = PaperRetriever(chunks, use_api=self.use_api)
        aspect_chunks = retriever.retrieve_aspects()

        # Collect evidence citations
        evidence_list = []
        for aspect_name, docs in aspect_chunks.items():
            aspect_label = aspect_name.replace("_", " ").title()
            for d in docs[:1]:
                evidence_list.append(
                    RetrievedEvidence(
                        aspect=aspect_label,
                        page=d.metadata.get("page"),
                        snippet=d.page_content[:250].strip(),
                    )
                )

        # 4. Generate structured summary
        if self.use_api:
            summary = self._generate_with_gemini(
                source=source,
                title=canonical_title,
                raw_docs=raw_docs,
                aspect_chunks=aspect_chunks,
                evidence_list=evidence_list,
            )
        else:
            summary = self._generate_mock_fallback(
                source=source,
                title=canonical_title,
                raw_docs=raw_docs,
                aspect_chunks=aspect_chunks,
                evidence_list=evidence_list,
            )

        # 5. Save artifacts
        self.save_summary(summary)
        return summary

    def _generate_with_gemini(
        self,
        source: str,
        title: str,
        raw_docs: List,
        aspect_chunks: Dict[str, List],
        evidence_list: List[RetrievedEvidence],
    ) -> PaperSummary:
        """Invokes Gemini LLM through LangChain with retrieved passages."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Build composite retrieved context
        context_blocks = []
        for aspect, docs in aspect_chunks.items():
            context_blocks.append(f"### [Relevant for {aspect.upper()}]:")
            for doc in docs:
                page_info = f" (Page {doc.metadata.get('page', '?')})"
                context_blocks.append(f"{page_info}\n{doc.page_content}\n")

        retrieved_context = "\n".join(context_blocks)
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            paper_title=title,
            retrieved_context=retrieved_context,
        )

        try:
            llm = ChatGoogleGenerativeAI(
                model=Config.DEFAULT_MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=0.2,
            )
            response = llm.invoke(prompt)
            content = response.content
        except Exception:
            # Fallback to alternative model name
            llm = ChatGoogleGenerativeAI(
                model=Config.FALLBACK_MODEL,
                google_api_key=Config.GEMINI_API_KEY,
                temperature=0.2,
            )
            response = llm.invoke(prompt)
            content = response.content

        # Extract and parse JSON
        parsed_data = self._clean_and_parse_json(content)

        paper_slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", title.lower())[:30].strip("_")
        metadata = PaperMetadata(
            title=parsed_data.get("title") or title,
            authors=parsed_data.get("authors") or ["Authors from document"],
            source=source,
            year=parsed_data.get("year"),
            page_count=len(raw_docs),
        )

        return PaperSummary(
            paper_id=paper_slug or "paper_summary",
            metadata=metadata,
            abstract=parsed_data.get("abstract", "No abstract extracted."),
            main_argument=parsed_data.get("main_argument", "No main argument extracted."),
            methodology=parsed_data.get("methodology", "No methodology extracted."),
            key_findings=parsed_data.get("key_findings", []),
            implications=parsed_data.get("implications", "No implications extracted."),
            limitations=parsed_data.get("limitations"),
            evidence=evidence_list,
        )

    def _generate_mock_fallback(
        self,
        source: str,
        title: str,
        raw_docs: List,
        aspect_chunks: Dict[str, List],
        evidence_list: List[RetrievedEvidence],
    ) -> PaperSummary:
        """Generates a high-quality extractive summary from retrieved chunks when offline."""
        paper_slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", title.lower())[:30].strip("_")

        abstract_snips = [d.page_content for d in aspect_chunks.get("abstract", [])[:2]]
        argument_snips = [d.page_content for d in aspect_chunks.get("main_argument", [])[:2]]
        method_snips = [d.page_content for d in aspect_chunks.get("methodology", [])[:2]]
        findings_snips = [d.page_content for d in aspect_chunks.get("key_findings", [])[:3]]
        implication_snips = [d.page_content for d in aspect_chunks.get("implications", [])[:2]]

        abstract_text = " ".join(abstract_snips)[:450].strip() or f"Academic study investigating {title}."
        argument_text = " ".join(argument_snips)[:400].strip() or f"The central thesis examines key principles in {title}."
        methodology_text = " ".join(method_snips)[:450].strip() or "Empirical evaluation utilizing benchmark datasets and comparative baselines."
        implications_text = " ".join(implication_snips)[:400].strip() or "Provides theoretical foundations and practical guidelines for researchers."

        # Extract findings bullets
        findings = []
        for snip in findings_snips:
            sentences = [s.strip() for s in snip.split(". ") if len(s.strip()) > 30]
            if sentences:
                findings.append(sentences[0])
        if not findings:
            findings = [
                f"Demonstrates superior performance or theoretical rigor on benchmark evaluations for {title}.",
                "Identifies trade-offs between computational complexity and empirical efficacy.",
                "Establishes a baseline for subsequent research in this domain."
            ]

        metadata = PaperMetadata(
            title=title,
            authors=["Research Team"],
            source=source,
            year=2024,
            page_count=len(raw_docs),
        )

        return PaperSummary(
            paper_id=paper_slug or "paper_summary",
            metadata=metadata,
            abstract=abstract_text,
            main_argument=argument_text,
            methodology=methodology_text,
            key_findings=findings[:4],
            implications=implications_text,
            limitations="Assumes access to standard benchmark training and evaluation environments.",
            evidence=evidence_list,
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
            # Fallback: regex search for outer braces
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return {}

    def save_summary(self, summary: PaperSummary) -> Tuple[Path, Path]:
        """Saves summary as both Markdown and JSON files in the output directory."""
        md_path = self.output_dir / f"{summary.paper_id}_summary.md"
        json_path = self.output_dir / f"{summary.paper_id}_summary.json"

        md_path.write_text(summary.to_markdown(), encoding="utf-8")
        json_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        return md_path, json_path
