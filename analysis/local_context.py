"""
Module 5: Local Semantic Relationship Analysis.
Measures local contextual inconsistency L(i), bridge anomalies,
and tracks entity, numerical, and attribute coherence.
"""

from typing import List, Dict, Any, Tuple, Set
import numpy as np
import re

from utils.logging_config import logger
from utils.helpers import cosine_similarity_matrix, get_sliding_window_neighbors
from preprocessing.sentence_segmenter import SentenceItem


class LocalContextAnalyzer:
    """
    Analyzes pairwise sentence coherence within a sliding window and identifies
    sudden local topic dislocations, entity shifts, and contextual fractures.
    """

    def __init__(self, window_size: int = 2):
        self.window_size = window_size
        self._entity_pattern = re.compile(r'\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b')
        self._numeric_pattern = re.compile(r'\b\d+(?:\.\d+)?%?\b')
        self._date_pattern = re.compile(r'\b(?:19|20)\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b')

    def compute_local_inconsistencies(
        self,
        sentence_embeddings: np.ndarray,
        sentences: List[SentenceItem],
        window_size: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Computes the local contextual inconsistency score L(i) for each sentence.

        Returns:
            - normalized_local_scores: np.ndarray [N] in [0, 1]
            - raw_local_scores: np.ndarray [N]
            - local_details: list of dicts with neighbor similarities, bridge markers, and entities
        """
        num_sentences = len(sentence_embeddings)
        effective_window = window_size or self.window_size

        if num_sentences == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), []

        if num_sentences == 1:
            return (
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                [{"neighbor_sims": [], "is_bridge_anomaly": False, "entity_novelty": 0.0}]
            )

        # Pairwise similarity matrix
        sim_matrix = cosine_similarity_matrix(sentence_embeddings, sentence_embeddings)

        raw_scores = np.zeros(num_sentences, dtype=np.float32)
        local_details = []

        # Extract entities, numbers, and dates
        sentence_entities = [self._extract_entities(s.text) for s in sentences]
        sentence_numbers = [self._numeric_pattern.findall(s.text) for s in sentences]
        sentence_dates = [self._date_pattern.findall(s.text) for s in sentences]

        for i in range(num_sentences):
            neighbor_indices = get_sliding_window_neighbors(i, num_sentences, effective_window)
            if not neighbor_indices:
                raw_scores[i] = 0.0
                local_details.append({"neighbor_sims": [], "is_bridge_anomaly": False, "entity_novelty": 0.0})
                continue

            neighbor_sims = sim_matrix[i, neighbor_indices]
            mean_neighbor_sim = float(np.mean(neighbor_sims))
            base_inconsistency = max(0.0, 1.0 - mean_neighbor_sim)

            # 1. Bridge Anomaly Detection:
            # If sentence i has predecessor (i-1) and successor (i+1),
            # and (i-1) and (i+1) are strongly coherent with EACH OTHER,
            # but sentence i is alienated from BOTH, this indicates an intrusive anomaly.
            is_bridge_anomaly = False
            bridge_penalty = 0.0
            if 0 < i < num_sentences - 1:
                prev_sim = sim_matrix[i, i - 1]
                next_sim = sim_matrix[i, i + 1]
                surrounding_sim = sim_matrix[i - 1, i + 1]

                # Legitimate transition check:
                # If i is similar to i+1 (initiating a new topic followed by i+1), it is NOT a bridge anomaly
                if surrounding_sim > 0.40 and prev_sim < 0.35 and next_sim < 0.35:
                    is_bridge_anomaly = True
                    bridge_penalty = float(surrounding_sim - max(prev_sim, next_sim))

            # 2. Entity Novelty & Inconsistency check
            current_ents = sentence_entities[i]
            neighbor_ents = set()
            for n_idx in neighbor_indices:
                neighbor_ents.update(sentence_entities[n_idx])

            entity_novelty = 0.0
            unshared_entities = []
            if current_ents and num_sentences >= 3:
                unshared = [e for e in current_ents if e not in neighbor_ents]
                # Only penalize if unshared entity is also alien to the whole document except this sentence
                alien_ents = []
                for ent in unshared:
                    doc_occurrences = sum(1 for ents in sentence_entities if ent in ents)
                    if doc_occurrences == 1:
                        alien_ents.append(ent)
                if alien_ents:
                    entity_novelty = len(alien_ents) / max(len(current_ents), 1)
                    unshared_entities = alien_ents

            # 3. Numeric & Date conflict indicator
            has_numeric_anomaly = False
            if sentence_numbers[i] and 0 < i:
                prev_nums = set(sentence_numbers[i - 1])
                curr_nums = set(sentence_numbers[i])
                # If sentences share strong semantic vocabulary but have conflicting numbers/percentages
                shared_words = set(sentences[i].tokens) & set(sentences[i - 1].tokens)
                if len(shared_words) >= 3 and curr_nums and prev_nums and not (curr_nums & prev_nums):
                    has_numeric_anomaly = True

            # Combined local inconsistency
            raw_score = (
                base_inconsistency +
                0.25 * bridge_penalty +
                0.15 * entity_novelty +
                (0.10 if has_numeric_anomaly else 0.0)
            )
            raw_scores[i] = raw_score

            local_details.append({
                "neighbor_indices": neighbor_indices,
                "neighbor_sims": [float(s) for s in neighbor_sims],
                "mean_neighbor_sim": mean_neighbor_sim,
                "is_bridge_anomaly": is_bridge_anomaly,
                "entity_novelty": float(entity_novelty),
                "unshared_entities": unshared_entities,
                "entities": list(current_ents),
                "numbers": sentence_numbers[i],
                "dates": sentence_dates[i]
            })

        normalized_scores = np.clip(raw_scores, 0.0, 1.0).astype(np.float32)
        return normalized_scores, raw_scores.astype(np.float32), local_details

    def _extract_entities(self, text: str) -> Set[str]:
        """Extracts named entities and proper nouns, ignoring sentence-initial common words."""
        tokens = re.findall(r'\b[A-Za-z0-9_-]+\b', text)
        if not tokens:
            return set()

        entities = set()
        common_words = {
            "The", "A", "An", "In", "On", "At", "However", "Therefore", "Moreover",
            "Furthermore", "This", "That", "These", "Those", "It", "Twelve", "High",
            "Emergency", "Geological", "Telemetry", "Average", "Scientific", "Global",
            "Comprehensive", "Mitigating", "Future", "Unlike", "Advanced", "Operating"
        }

        # Subsequent capitalized tokens (ignoring sentence-initial capital unless multi-word)
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
