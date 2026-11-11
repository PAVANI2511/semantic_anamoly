"""
Module 4: Global Semantic Context Analysis.
Measures deviation of individual sentences from the global document context.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from utils.logging_config import logger
from utils.helpers import cosine_similarity_matrix, min_max_normalize
from preprocessing.sentence_segmenter import SentenceItem


class GlobalContextAnalyzer:
    """
    Analyzes document-wide semantic topic representation and computes
    global deviation score G(i) for each sentence.
    """

    def __init__(self, smoothing_window: int = 1):
        self.smoothing_window = smoothing_window

    def compute_global_deviations(
        self,
        sentence_embeddings: np.ndarray,
        sentences: List[SentenceItem]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Computes the global semantic deviation score G(i) for each sentence.
        
        Returns:
            - normalized_deviations: np.ndarray of shape [N] in [0, 1]
            - raw_deviations: np.ndarray of shape [N]
            - metadata: dictionary containing global document centroid and stats
        """
        num_sentences = len(sentence_embeddings)
        if num_sentences == 0:
            return np.array([]), np.array([]), {}

        # 1. Compute global document centroid vector
        centroid = np.mean(sentence_embeddings, axis=0, keepdims=True)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-12:
            centroid = centroid / centroid_norm

        # 2. Compute cosine similarity of each sentence to global centroid
        similarities = cosine_similarity_matrix(sentence_embeddings, centroid).squeeze(axis=1)

        # 3. Raw global deviation: 1 - cosine_similarity
        raw_deviations = 1.0 - similarities

        # 4. Contextual Transition Smoothing
        # Prevent penalizing smooth topical transitions by incorporating
        # paragraph-level context if available
        adjusted_deviations = raw_deviations.copy()
        if num_sentences >= 4 and self.smoothing_window > 0:
            # Dampen deviation if neighboring sentences in the same paragraph share similar topic
            smoothed = np.zeros_like(raw_deviations)
            for i in range(num_sentences):
                p_id = sentences[i].paragraph_id if i < len(sentences) else 0
                # Find peers in same paragraph
                peers = [
                    j for j in range(max(0, i - self.smoothing_window), min(num_sentences, i + self.smoothing_window + 1))
                    if (j < len(sentences) and sentences[j].paragraph_id == p_id)
                ]
                smoothed[i] = np.mean(raw_deviations[peers]) if peers else raw_deviations[i]
            
            # Weighted mix of point deviation and paragraph context
            adjusted_deviations = 0.75 * raw_deviations + 0.25 * smoothed

        # 5. Calibrate global deviation to [0, 1] without artificial relative stretching
        normalized_deviations = np.clip(adjusted_deviations, 0.0, 1.0).astype(np.float32)

        metadata = {
            "mean_similarity_to_doc": float(np.mean(similarities)),
            "std_similarity_to_doc": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "num_sentences": num_sentences
        }

        return normalized_deviations, raw_deviations, metadata
