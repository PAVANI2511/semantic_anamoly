"""
Module 4: Global Semantic Context Analysis.
Measures deviation of individual sentences from the global document context
with multi-topic clustering, paragraph smoothing, and edge-case handling.
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from utils.logging_config import logger
from utils.helpers import cosine_similarity_matrix
from preprocessing.sentence_segmenter import SentenceItem


class GlobalContextAnalyzer:
    """
    Analyzes document-wide semantic topic representations and computes
    global deviation score G(i) for each sentence.
    """

    def __init__(self, smoothing_window: int = 1, multi_topic_threshold: int = 8):
        self.smoothing_window = smoothing_window
        self.multi_topic_threshold = multi_topic_threshold

    def compute_global_deviations(
        self,
        sentence_embeddings: np.ndarray,
        sentences: List[SentenceItem]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Computes the global semantic deviation score G(i) for each sentence.
        
        Handles:
            - Single sentence documents (returns 0.0)
            - Short documents (1-3 sentences)
            - Multi-topic documents (uses multi-centroid maximum similarity)
            - Repeated/identical sentences (returns 0.0)
        """
        num_sentences = len(sentence_embeddings)
        if num_sentences == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), {}

        # Edge case: Single sentence document cannot be anomalous relative to itself
        if num_sentences == 1:
            return (
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                {"mean_similarity_to_doc": 1.0, "is_single_sentence": True}
            )

        # 1. Compute global document centroid vector
        centroid = np.mean(sentence_embeddings, axis=0, keepdims=True)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-12:
            centroid = centroid / centroid_norm

        # 2. Compute similarity to global centroid
        similarities = cosine_similarity_matrix(sentence_embeddings, centroid).squeeze(axis=1)

        # Edge case: Identical or near-identical sentences throughout document
        if np.std(similarities) < 1e-6 and np.mean(similarities) > 0.95:
            return (
                np.zeros(num_sentences, dtype=np.float32),
                np.zeros(num_sentences, dtype=np.float32),
                {"mean_similarity_to_doc": 1.0, "is_homogeneous": True}
            )

        # 3. Multi-Topic Document Handling:
        # In multi-paragraph or longer documents, cluster paragraphs to avoid centroid collapse
        # between legitimate distinct sections (e.g. Methods vs Results)
        if num_sentences >= self.multi_topic_threshold:
            paragraph_ids = sorted(list({s.paragraph_id for s in sentences if hasattr(s, "paragraph_id")}))
            if len(paragraph_ids) >= 2:
                # Compute paragraph sub-centroids
                p_centroids = []
                for p_id in paragraph_ids:
                    p_indices = [idx for idx, s in enumerate(sentences) if s.paragraph_id == p_id]
                    if p_indices:
                        p_vec = np.mean(sentence_embeddings[p_indices], axis=0, keepdims=True)
                        p_norm = np.linalg.norm(p_vec)
                        if p_norm > 1e-12:
                            p_centroids.append(p_vec / p_norm)

                if p_centroids:
                    p_mat = np.vstack(p_centroids)  # [P, D]
                    # Similarity of each sentence to the nearest valid paragraph topic cluster
                    p_sims = cosine_similarity_matrix(sentence_embeddings, p_mat)  # [N, P]
                    max_p_sims = np.max(p_sims, axis=1)
                    # Blend global centroid similarity with nearest paragraph cluster similarity
                    similarities = 0.60 * similarities + 0.40 * max_p_sims

        # 4. Raw global deviation: 1 - similarity
        raw_deviations = 1.0 - similarities

        # 5. Short Document Safeguard:
        # For documents with <= 3 sentences, dampen deviation so normal variance is not over-penalized
        if num_sentences <= 3:
            raw_deviations = raw_deviations * 0.70

        # 6. Contextual Transition Smoothing
        adjusted_deviations = raw_deviations.copy()
        if num_sentences >= 4 and self.smoothing_window > 0:
            smoothed = np.zeros_like(raw_deviations)
            for i in range(num_sentences):
                p_id = sentences[i].paragraph_id if i < len(sentences) else 0
                peers = [
                    j for j in range(max(0, i - self.smoothing_window), min(num_sentences, i + self.smoothing_window + 1))
                    if (j < len(sentences) and sentences[j].paragraph_id == p_id)
                ]
                smoothed[i] = np.mean(raw_deviations[peers]) if peers else raw_deviations[i]

            adjusted_deviations = 0.80 * raw_deviations + 0.20 * smoothed

        # 7. Bound calibrated deviation in [0, 1]
        normalized_deviations = np.clip(adjusted_deviations, 0.0, 1.0).astype(np.float32)

        metadata = {
            "mean_similarity_to_doc": float(np.mean(similarities)),
            "std_similarity_to_doc": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "num_sentences": num_sentences
        }

        return normalized_deviations, raw_deviations.astype(np.float32), metadata
