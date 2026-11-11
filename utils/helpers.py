"""
Utility functions and mathematical helpers for Semantic Sentinel.
"""

from typing import List, Tuple, Any, Dict
import numpy as np


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between two 2D numpy arrays.
    Shapes: a (N, D), b (M, D) -> returns (N, M).
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
        
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    
    # Avoid zero division
    norm_a = np.where(norm_a == 0, 1e-12, norm_a)
    norm_b = np.where(norm_b == 0, 1e-12, norm_b)
    
    return np.dot(a / norm_a, (b / norm_b).T)


def min_max_normalize(scores: np.ndarray | List[float], eps: float = 1e-8) -> np.ndarray:
    """
    Normalizes scores to [0.0, 1.0] interval.
    If all scores are equal, returns zeros array with matching shape.
    """
    arr = np.asarray(scores, dtype=np.float32)
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if np.isclose(max_val, min_val, atol=eps):
        return np.zeros_like(arr)
        
    return (arr - min_val) / (max_val - min_val)


def get_sliding_window_neighbors(
    index: int,
    total_sentences: int,
    window_size: int = 2
) -> List[int]:
    """
    Returns list of neighbor indices within the window size, excluding current index.
    """
    start = max(0, index - window_size)
    end = min(total_sentences, index + window_size + 1)
    return [i for i in range(start, end) if i != index]


def truncate_text(text: str, max_length: int = 80) -> str:
    """Truncates text with ellipsis if length exceeds max_length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
