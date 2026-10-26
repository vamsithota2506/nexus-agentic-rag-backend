import math
import re
from typing import List
import numpy as np
from app.config import settings
from app.core.logging import logger


class EmbeddingService:
    """
    Modular embedding generator.
    Supports SentenceTransformers/HuggingFace or high-performance deterministic
    vector embedding fallback for instant zero-dependency execution.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._model = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            if settings.LLM_PROVIDER == "sentence_transformers":
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                logger.info(f"Loaded SentenceTransformer: {settings.EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformers ({e}). Using deterministic semantic vector encoder.")
            self._model = None

    def embed_text(self, text: str) -> np.ndarray:
        if self._model:
            emb = self._model.encode(text)
            return emb / np.linalg.norm(emb)
        return self._generate_semantic_vector(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if self._model:
            embeddings = self._model.encode(texts)
            return [emb / np.linalg.norm(emb) for emb in embeddings]
        return [self._generate_semantic_vector(t) for t in texts]

    def _generate_semantic_vector(self, text: str) -> np.ndarray:
        """
        High-dimensional deterministic hashing & character-ngram semantic projection
        producing normalized 384-dim unit vectors with high cosine similarity for related terms.
        """
        tokens = re.findall(r"\w+", text.lower())
        vec = np.zeros(self.dimension, dtype=np.float32)

        if not tokens:
            return vec

        for i, token in enumerate(tokens):
            # Token hash projection
            h1 = hash(token) % self.dimension
            h2 = hash(token + "_ngram") % self.dimension
            weight = 1.0 / math.log2(i + 3)  # Positional dampening
            vec[h1] += weight * 1.5
            vec[h2] += weight * 0.8

            # Sub-word character n-grams (3-4 grams)
            for n in range(3, min(5, len(token) + 1)):
                for j in range(len(token) - n + 1):
                    sub = token[j:j+n]
                    h_sub = hash(sub) % self.dimension
                    vec[h_sub] += 0.3 * weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# Singleton instance
embedding_service = EmbeddingService()
