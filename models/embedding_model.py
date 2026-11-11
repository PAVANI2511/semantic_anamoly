"""
Module 3: Semantic Representation.
Provides configurable Sentence Transformers and TF-IDF embedding backends
with caching and device handling.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
import os

from utils.logging_config import logger
from config.settings import config


class BaseEmbeddingModel(ABC):
    """Abstract interface for all embedding backends."""

    @abstractmethod
    def embed_sentences(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of sentences into normalized numpy embeddings [N, D]."""
        pass

    def embed_document(self, sentence_embeddings: np.ndarray) -> np.ndarray:
        """
        Computes the global document representation by mean-pooling sentence embeddings.
        Returns a normalized 1D vector [D].
        """
        if len(sentence_embeddings) == 0:
            raise ValueError("Cannot embed empty document.")
        mean_vec = np.mean(sentence_embeddings, axis=0, keepdims=True)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-12:
            mean_vec = mean_vec / norm
        return mean_vec.squeeze(0)

    @abstractmethod
    def get_dimension(self) -> int:
        """Returns embedding dimensionality."""
        pass


class TransformerEmbeddingModel(BaseEmbeddingModel):
    """
    Sentence Transformers embedding backend.
    Caches model weights to avoid re-loading on each inference call.
    """

    _model_cache: Dict[str, Any] = {}

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        self.model_name = model_name or config.model.primary_embedding_model
        self.device = device or config.model.device
        self.batch_size = batch_size
        self._model = self._load_model(self.model_name, self.device)
        if hasattr(self._model, "get_embedding_dimension"):
            self._dim = self._model.get_embedding_dimension()
        else:
            self._dim = self._model.get_sentence_embedding_dimension()

    @classmethod
    def _load_model(cls, model_name: str, device: str):
        cache_key = f"{model_name}_{device}"
        if cache_key in cls._model_cache:
            return cls._model_cache[cache_key]

        from sentence_transformers import SentenceTransformer
        import torch

        # Device resolution
        target_device = "cpu"
        if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            target_device = "cuda"

        import os
        # Set offline flag if offline or fallback
        try:
            model = SentenceTransformer(model_name, device=target_device)
        except Exception as e:
            logger.info("Attempting offline load...")
            os.environ["HF_HUB_OFFLINE"] = "1"
            model = SentenceTransformer(model_name, device=target_device)

        cls._model_cache[cache_key] = model
        return model

    def embed_sentences(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def get_dimension(self) -> int:
        return self._dim


class TFIDFEmbeddingModel(BaseEmbeddingModel):
    """
    TF-IDF Vectorizer baseline embedding backend.
    Useful as an ultra-fast baseline and fallback.
    """

    def __init__(self, max_features: int = 512):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._is_fitted = False
        self._dim = max_features

    def embed_sentences(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        if not self._is_fitted:
            matrix = self.vectorizer.fit_transform(texts).toarray()
            self._is_fitted = True
            self._dim = matrix.shape[1]
        else:
            matrix = self.vectorizer.transform(texts).toarray()

        # L2 normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        matrix = matrix / norms
        return matrix.astype(np.float32)

    def get_dimension(self) -> int:
        return self._dim


class EmbeddingFactory:
    """Factory to instantiate and retrieve embedding backends."""

    @staticmethod
    def get_embedding_model(backend: str = "transformer", model_name: Optional[str] = None) -> BaseEmbeddingModel:
        if backend == "tfidf":
            return TFIDFEmbeddingModel()
        return TransformerEmbeddingModel(model_name=model_name)
