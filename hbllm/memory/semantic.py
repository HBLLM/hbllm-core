"""
Semantic Memory (RAG Vector Database).

Embeds text using Sentence-Transformers and stores them for cosine-similarity 
semantic search. This allows the agent to recall long-term context that falls
out of the immediate rolling episodic window.
"""

import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

class SemanticMemory:
    """
    A lightweight in-memory vector database for the Modular Brain.
    Uses sentence-transformers to compute dense embeddings and NumPy
    for exact k-Nearest Neighbors cosine similarity search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.documents = []
        self.vectors = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model {self.model_name}...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

    def store(self, content: str, metadata: dict[str, Any] = None):
        """Embed and store a document."""
        self._load_model()
        
        # Embed the content string
        embedding = self.model.encode([content])[0]
        
        doc = {
            "content": content,
            "metadata": metadata or {}
        }
        self.documents.append(doc)
        
        if self.vectors is None:
            self.vectors = np.array([embedding])
        else:
            self.vectors = np.vstack((self.vectors, embedding))
            
        logger.debug(f"Stored semantic document: {content[:50]}...")
        return len(self.documents) - 1

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search for the most semantically similar documents to the query."""
        if not self.documents or self.vectors is None:
            return []
            
        self._load_model()
        query_vec = self.model.encode([query])[0]
        
        # Compute cosine similarity
        similarities = np.dot(self.vectors, query_vec) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec) + 1e-9
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # Optionally filter out low-confidence hits
            if similarities[idx] < 0.2:
                continue
                
            res = self.documents[idx].copy()
            res["score"] = float(similarities[idx])
            results.append(res)
            
        return results
