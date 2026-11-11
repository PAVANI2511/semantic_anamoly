"""
Module 5: Local Semantic Relationship Analysis.
Measures local contextual inconsistency L(i) and tracks entity/attribute coherence.
"""

from typing import List, Dict, Any, Tuple, Set
import numpy as np
import re

from utils.logging_config import logger
from utils.helpers import cosine_similarity_matrix, min_max_normalize, get_sliding_window_neighbors
from preprocessing.sentence_segmenter import SentenceItem


class LocalContextAnalyzer:
    """
    Analyzes pairwise sentence coherence within a sliding window and identifies
    sudden local topic dislocations, entity shifts, and contextual fractures.
    """

    def __init__(self, window_size: int = 2):
        self.window_size = window_size
        self._entity_pattern = re.compile(r'\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b')

    def compute_local_inconsistencies(
        self,
        sentence_embeddings: np.ndarray,
        sentences: List[SentenceItem]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Computes the local contextual inconsistency score L(i) for each sentence.

        Returns:
            - normalized_local_scores: np.ndarray [N] in [0, 1]
            - raw_local_scores: np.ndarray [N]
            - local_details: list of dicts with neighbor similarities and bridge anomaly indicators
        """
        num_sentences = len(sentence_embeddings)
        if num_sentences == 0:
            return np.array([]), np.array([]), []

        if num_sentences == 1:
            return np.array([0.0]), np.array([0.0]), [{"neighbor_sims": [], "is_bridge_anomaly": False}]

        # Compute full pairwise similarity matrix
        sim_matrix = cosine_similarity_matrix(sentence_embeddings, sentence_embeddings)

        raw_scores = np.zeros(num_sentences, dtype=np.float32)
        local_details = []

        # Extract simple entities/proper nouns across sentences for entity consistency tracking
        sentence_entities = [self._extract_entities(s.text) for s in sentences]
        global_entity_pool: Set[str] = set()
        for ents in sentence_entities:
            global_entity_pool.update(ents)

        for i in range(num_sentences):
            neighbor_indices = get_sliding_window_neighbors(i, num_sentences, self.window_size)
            if not neighbor_indices:
                raw_scores[i] = 0.0
                local_details.append({"neighbor_sims": [], "is_bridge_anomaly": False, "entity_novelty": 0.0})
                continue

            neighbor_sims = sim_matrix[i, neighbor_indices]
            mean_neighbor_sim = float(np.mean(neighbor_sims))
            base_inconsistency = 1.0 - mean_neighbor_sim

            # Bridge Anomaly Detection:
            # If sentence i has a predecessor (i-1) and successor (i+1),
            # and (i-1) and (i+1) are similar to EACH OTHER, but i is dissimilar to BOTH,
            # this is a classic intrusive anomaly!
            is_bridge_anomaly = False
            bridge_penalty = 0.0
            if i > 0 and i < num_sentences - 1:
                prev_sim = sim_matrix[i, i - 1]
                next_sim = sim_matrix[i, i + 1]
                surrounding_sim = sim_matrix[i - 1, i + 1]

                if surrounding_sim > 0.45 and prev_sim < 0.35 and next_sim < 0.35:
                    is_bridge_anomaly = True
                    bridge_penalty = float(surrounding_sim - max(prev_sim, next_sim))

            # Entity Novelty & Inconsistency check
            current_ents = sentence_entities[i]
            neighbor_ents = set()
            for n_idx in neighbor_indices:
                neighbor_ents.update(sentence_entities[n_idx])

            # Measure if new unknown entities suddenly appear that never appear anywhere else
            entity_novelty = 0.0
            if current_ents and len(sentences) >= 3:
                unshared = [e for e in current_ents if e not in neighbor_ents]
                entity_novelty = len(unshared) / max(len(current_ents), 1)

            # Combined raw local inconsistency
            raw_score = base_inconsistency + 0.3 * bridge_penalty + 0.15 * entity_novelty
            raw_scores[i] = raw_score

            local_details.append({
                "neighbor_indices": neighbor_indices,
                "neighbor_sims": [float(s) for s in neighbor_sims],
                "mean_neighbor_sim": mean_neighbor_sim,
                "is_bridge_anomaly": is_bridge_anomaly,
                "entity_novelty": float(entity_novelty),
                "entities": list(current_ents)
            })

        normalized_scores = np.clip(raw_scores, 0.0, 1.0).astype(np.float32)
        return normalized_scores, raw_scores, local_details

    def _extract_entities(self, text: str) -> Set[str]:
        """Simple entity and proper-noun extractor using capitalization patterns."""
        tokens = re.findall(r'\b[A-Za-z0-9_-]+\b', text)
        if not tokens:
            return set()

        entities = set()
        common_words = {"The", "A", "An", "In", "On", "At", "However", "Therefore", "Moreover", "Furthermore", "This", "That", "These", "Those", "It", "Twelve", "High", "Emergency", "Geological", "Telemetry"}

        # Subsequent capitalized tokens (ignoring sentence-initial capital)
        for token in tokens[1:]:
            if token[0].isupper() and token not in common_words and len(token) > 2:
                entities.add(token)

        # Multi-word proper names anywhere (e.g. "Detective Miller", "Jezero Crater", "Fortune 500")
        multi_matches = self._entity_pattern.findall(text)
        for m in multi_matches:
            parts = m.split()
            if len(parts) >= 2 and parts[0] not in common_words:
                entities.add(m.strip())

        return entities
