"""Document loader supporting local PDFs, remote PDF URLs, arXiv links, and academic web pages."""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Tuple
import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from paper_rag.config import Config


class PaperLoader:
    """Loads academic papers from local files or remote URLs."""

    def __init__(self, download_dir: Path = None):
        self.download_dir = download_dir or Config.DOWNLOAD_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.http_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36 AcademicPaperRAG/1.0"
            )
        }

    def load(self, source: str) -> Tuple[List[Document], str]:
        """Ingests a paper from a local path or URL.

        Args:
            source: Local file path or HTTP/HTTPS URL.

        Returns:
            Tuple of (list of Document chunks/pages, canonical title/identifier).
        """
        source = source.strip()
        if source.startswith("http://") or source.startswith("https://"):
            return self._load_from_url(source)
        else:
            return self._load_from_file(source)

    def _load_from_url(self, url: str) -> Tuple[List[Document], str]:
        """Handles remote URLs: arXiv conversion, PDF download, or HTML extraction."""
        # Convert arXiv abs to pdf if applicable
        normalized_url = url
        if "arxiv.org/abs/" in url:
            arxiv_id = url.split("arxiv.org/abs/")[-1].split("?")[0].strip("/")
            normalized_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Check if URL points to a PDF
        is_pdf = normalized_url.lower().endswith(".pdf") or "arxiv.org/pdf/" in normalized_url

        if is_pdf:
            local_pdf_path = self._download_file(normalized_url)
            docs, title = self._load_from_pdf(local_pdf_path, original_source=url)
            return docs, title
        else:
            # Fallback to HTML extraction
            return self._load_from_html(normalized_url)

    def _download_file(self, url: str) -> Path:
        """Download remote PDF file to local cache."""
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
        # Extract filename from url if reasonable
        clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", url.split("/")[-1])
        if not clean_name.lower().endswith(".pdf"):
            clean_name = f"{clean_name}.pdf"
        dest_filename = f"{url_hash}_{clean_name}"
        dest_path = self.download_dir / dest_filename

        if dest_path.exists() and dest_path.stat().st_size > 1024:
            return dest_path

        with httpx.Client(timeout=30.0, follow_redirects=True, headers=self.http_headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            dest_path.write_bytes(resp.content)

        return dest_path

    def _load_from_file(self, file_path_str: str) -> Tuple[List[Document], str]:
        """Loads a local file (.pdf or text-based)."""
        file_path = Path(file_path_str).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Paper file not found at: {file_path}")

        if file_path.suffix.lower() == ".pdf":
            return self._load_from_pdf(file_path, original_source=str(file_path))
        else:
            # Plain text / markdown academic paper
            text_content = file_path.read_text(encoding="utf-8", errors="ignore")
            title = self._infer_title(text_content, file_path.stem)
            doc = Document(
                page_content=text_content,
                metadata={
                    "source": str(file_path),
                    "page": 1,
                    "title": title,
                }
            )
            return [doc], title

    def _load_from_pdf(self, pdf_path: Path, original_source: str) -> Tuple[List[Document], str]:
        """Extracts pages from PDF using PyPDFLoader with fallback."""
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
        except Exception:
            # Fallback to direct pypdf
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": original_source, "page": i + 1}
                    )
                )

        if not docs:
            raise ValueError(f"No readable text could be extracted from PDF: {pdf_path}")

        # Update metadata
        title = self._extract_pdf_title(docs, pdf_path.stem)
        for doc in docs:
            doc.metadata["source"] = original_source
            doc.metadata["title"] = title

        return docs, title

    def _load_from_html(self, url: str) -> Tuple[List[Document], str]:
        """Fetches web page content, parses HTML, and returns Document."""
        with httpx.Client(timeout=25.0, follow_redirects=True, headers=self.http_headers) as client:
            resp = client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        h1 = soup.find("h1")
        if h1 and h1.text.strip():
            title = h1.text.strip()
        if not title:
            title = url

        # Remove irrelevant elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        main_content = soup.find("main") or soup.find("article") or soup.find("body") or soup
        text = main_content.get_text(separator="\n", strip=True)

        doc = Document(
            page_content=text,
            metadata={"source": url, "page": 1, "title": title}
        )
        return [doc], title

    def _extract_pdf_title(self, docs: List[Document], default_stem: str) -> str:
        """Heuristic to extract paper title from first page."""
        if not docs:
            return default_stem
        first_page = docs[0].page_content
        lines = [line.strip() for line in first_page.split("\n") if line.strip()]
        # Academic papers usually have the title in the first 1-4 prominent lines
        candidate_lines = []
        for line in lines[:6]:
            if len(line) > 10 and not line.lower().startswith("arxiv:") and not line.lower().startswith("doi:"):
                candidate_lines.append(line)
                if len(" ".join(candidate_lines)) > 30:
                    break
        if candidate_lines:
            return " ".join(candidate_lines)
        return default_stem.replace("_", " ").title()

    def _infer_title(self, text: str, default: str) -> str:
        """Infers title from first heading or non-empty line."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:5]:
            cleaned = line.lstrip("#").strip()
            if len(cleaned) > 5:
                return cleaned
        return default
