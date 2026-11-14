"""Configuration management for Paper RAG Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if available
project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(project_root / ".env")


class Config:
    """Project configuration settings."""

    # API Keys
    GEMINI_API_KEY: str = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    )

    # Models
    # Using current Gemini recommended model gemini-3.8-flash with fallback
    DEFAULT_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    FALLBACK_MODEL: str = "gemini-flash-latest"
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

    # Document Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # RAG Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # Paths
    PROJECT_ROOT: Path = project_root
    DOWNLOAD_DIR: Path = project_root / "downloads"
    OUTPUT_DIR: Path = project_root / "output"
    SAMPLE_DIR: Path = project_root / "sample_papers"

    @classmethod
    def ensure_directories(cls):
        """Ensure necessary output and cache directories exist."""
        cls.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def has_api_key(cls) -> bool:
        """Check if Gemini API key is configured."""
        return bool(cls.GEMINI_API_KEY and not cls.GEMINI_API_KEY.startswith("your_"))
