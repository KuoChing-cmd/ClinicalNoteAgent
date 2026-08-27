"""
RAG Retriever for Experience Rules and Case Base.

Uses sentence-transformers (all-MiniLM-L6-v2, CPU) + FAISS for semantic
similarity search, replacing the previous approach of injecting all rules
into the LLM prompt.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    """Manages FAISS indices for experience rules and case base entries."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"📦 Loading embedding model: {model_name} (CPU)...")
        self.model = SentenceTransformer(model_name, device='cpu')
        self.embed_dim = self.model.get_embedding_dimension()

        # Experience rules index
        self._exp_index = None
        self._exp_texts = []

        # Case base index
        self._case_index = None
        self._case_texts = []   # stringified summaries for embedding
        self._case_objects = []  # original case dicts for retrieval

    # ------------------------------------------------------------------
    # Experience Rules
    # ------------------------------------------------------------------

    def build_experience_index(self, rules):
        """Build FAISS index from a list of experience rule strings."""
        self._exp_texts = list(rules)
        if not self._exp_texts:
            self._exp_index = faiss.IndexFlatIP(self.embed_dim)
            return
        embeddings = self._encode(self._exp_texts)
        self._exp_index = faiss.IndexFlatIP(self.embed_dim)
        self._exp_index.add(embeddings)
        print(f"   ✅ Experience index built: {self._exp_index.ntotal} rules")

    def add_experience(self, rule):
        """Incrementally add a single rule to the experience index."""
        self._exp_texts.append(rule)
        emb = self._encode([rule])
        if self._exp_index is None:
            self._exp_index = faiss.IndexFlatIP(self.embed_dim)
        self._exp_index.add(emb)

    def search_experience(self, query, top_k=5):
        """Return the top_k most relevant experience rules for a query."""
        if self._exp_index is None or self._exp_index.ntotal == 0:
            return []
        top_k = min(top_k, self._exp_index.ntotal)
        q_emb = self._encode([query])
        _, indices = self._exp_index.search(q_emb, top_k)
        return [self._exp_texts[i] for i in indices[0] if 0 <= i < len(self._exp_texts)]

    # ------------------------------------------------------------------
    # Case Base
    # ------------------------------------------------------------------

    @staticmethod
    def _case_to_text(case):
        """Convert a case dict to a flat text string for embedding."""
        parts = []
        for cat, summ in case.get('category_summaries', {}).items():
            parts.append(f"{cat}: {summ}")
        meta = case.get('meta_summary', {})
        if isinstance(meta, dict):
            parts.append(f"mortality_risk={meta.get('mortality_risk_score', '?')} "
                         f"readmission_risk={meta.get('icu_readmission_risk_score', '?')}")
            if meta.get('final_summary'):
                parts.append(meta['final_summary'])
        return " | ".join(parts)

    def build_case_index(self, cases):
        """Build FAISS index from a list of case base dicts."""
        self._case_objects = list(cases)
        self._case_texts = [self._case_to_text(c) for c in cases]
        if not self._case_texts:
            self._case_index = faiss.IndexFlatIP(self.embed_dim)
            return
        embeddings = self._encode(self._case_texts)
        self._case_index = faiss.IndexFlatIP(self.embed_dim)
        self._case_index.add(embeddings)
        print(f"   ✅ Case base index built: {self._case_index.ntotal} cases")

    def add_case(self, case):
        """Incrementally add a single case to the case index."""
        self._case_objects.append(case)
        text = self._case_to_text(case)
        self._case_texts.append(text)
        emb = self._encode([text])
        if self._case_index is None:
            self._case_index = faiss.IndexFlatIP(self.embed_dim)
        self._case_index.add(emb)

    def search_cases(self, query, top_k=1):
        """Return the top_k most similar case dicts for a query."""
        if self._case_index is None or self._case_index.ntotal == 0:
            return []
        top_k = min(top_k, self._case_index.ntotal)
        q_emb = self._encode([query])
        _, indices = self._case_index.search(q_emb, top_k)
        return [self._case_objects[i] for i in indices[0] if 0 <= i < len(self._case_objects)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode(self, texts):
        """Encode texts to L2-normalized embeddings (for cosine via inner product)."""
        embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        return embeddings.astype(np.float32)
