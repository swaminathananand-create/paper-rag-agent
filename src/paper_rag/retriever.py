"""RAG retrieval layer: Vector indexing with Gemini embeddings and BM25 fallback."""

import math
import re
from collections import Counter
from typing import List, Dict, Optional
from langchain_core.documents import Document
from paper_rag.config import Config


class BM25Retriever:
    """A lightweight in-memory BM25 lexical retriever for offline/fallback RAG."""

    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_lens = [len(self._tokenize(doc.page_content)) for doc in documents]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.corpus_size = len(documents)

        # Build term frequencies & document frequencies
        self.doc_freqs: Dict[str, int] = Counter()
        self.term_freqs: List[Dict[str, int]] = []
        for doc in documents:
            tokens = self._tokenize(doc.page_content)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf.keys():
                self.doc_freqs[term] += 1

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text)]

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Calculates BM25 score and returns top-k documents."""
        query_terms = self._tokenize(query)
        scores = []

        for idx, tf in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_lens[idx]
            for term in query_terms:
                if term not in tf:
                    continue
                df = self.doc_freqs.get(term, 0)
                # BM25 IDF
                idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
                freq = tf[term]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avgdl, 1)))
                score += idf * (numerator / denominator)
            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_docs = [self.documents[idx] for score, idx in scores[:k] if score > 0]
        # If no positive matches, return first k documents as baseline context
        if not top_docs and self.documents:
            top_docs = self.documents[:k]
        return top_docs


class PaperRetriever:
    """Manages document indexing and targeted RAG retrieval for academic papers."""

    def __init__(self, chunks: List[Document], use_api: Optional[bool] = None):
        self.chunks = chunks
        self.use_api = Config.has_api_key() if use_api is None else use_api
        self.vector_store = None
        self.bm25_retriever = BM25Retriever(chunks)
        self.engine_name = "BM25 (Offline/Deterministic)"

        if self.use_api:
            try:
                self._initialize_vector_store()
                self.engine_name = f"Gemini Embeddings ({Config.EMBEDDING_MODEL}) + InMemoryVectorStore"
            except Exception as e:
                # Fall back gracefully if embeddings call fails (e.g. quota or offline)
                self.vector_store = None
                self.engine_name = f"BM25 Fallback (API initialization error: {e})"

    def _initialize_vector_store(self):
        """Initializes LangChain InMemoryVectorStore with GoogleGenerativeAIEmbeddings."""
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore

        embeddings = GoogleGenerativeAIEmbeddings(
            model=f"models/{Config.EMBEDDING_MODEL}",
            google_api_key=Config.GEMINI_API_KEY,
        )
        self.vector_store = InMemoryVectorStore.from_documents(
            documents=self.chunks,
            embedding=embeddings,
        )

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """Retrieves top-k most relevant document chunks for a query."""
        k = top_k or Config.TOP_K
        if self.vector_store is not None:
            try:
                return self.vector_store.similarity_search(query, k=k)
            except Exception:
                # In case of API runtime failure, fall back to BM25
                return self.bm25_retriever.similarity_search(query, k=k)
        return self.bm25_retriever.similarity_search(query, k=k)

    def retrieve_aspects(self) -> Dict[str, List[Document]]:
        """Executes targeted RAG queries for all required structured summary sections."""
        queries = {
            "abstract": "abstract problem statement research objective context background",
            "main_argument": "main thesis central argument hypothesis primary claim motivation",
            "methodology": "methodology architecture experimental design dataset model baseline pipeline",
            "key_findings": "results empirical findings evaluation benchmarks performance metrics",
            "implications": "broader implications conclusion theoretical impact practical future directions limitations",
        }

        aspect_results = {}
        for aspect, query in queries.items():
            aspect_results[aspect] = self.retrieve(query, top_k=3)
        return aspect_results
