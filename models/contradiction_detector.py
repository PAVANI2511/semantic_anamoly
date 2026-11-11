"""
Contradiction and Semantic Inconsistency Detection.
Identifies logical contradictions, negation flips, and conflicting claims
between a candidate sentence and prior contextual sentences.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import re

from utils.logging_config import logger
from utils.helpers import cosine_similarity_matrix, min_max_normalize
from preprocessing.sentence_segmenter import SentenceItem


class ContradictionDetector:
    """
    Detects semantic contradictions using lexical/polarity heuristics
    and optional transformer cross-encoder NLI.
    """

    # Antonym and polarity contradiction indicator pairs
    CONTRADICTING_PAIRS = [
        ("increas", "decreas"), ("ris", "fall"), ("rose", "fell"), ("gain", "loss"),
        ("profit", "loss"), ("success", "fail"), ("succeed", "fail"), ("successful", "unsuccessful"),
        ("grow", "shrink"), ("growth", "decline"), ("surged", "plunged"),
        ("true", "false"), ("correct", "incorrect"), ("valid", "invalid"),
        ("positive", "negative"), ("enable", "disable"), ("safe", "danger"),
        ("accept", "reject"), ("approv", "deni"), ("always", "never"),
        ("present", "absent"), ("highest", "lowest"), ("high", "low")
    ]

    NEGATION_TERMS = {"not", "never", "no", "neither", "nor", "hardly", "scarcely", "barely", "none", "cannot", "n't"}

    def __init__(self, use_nli_model: bool = False, model_name: str = "cross-encoder/nli-deberta-v3-xsmall"):
        self.use_nli_model = use_nli_model
        self.model_name = model_name
        self._nli_model = None

    def compute_contradiction_scores(
        self,
        sentences: List[SentenceItem],
        sentence_embeddings: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Computes contradiction score C(i) for each sentence against surrounding sentences.

        Returns:
            - normalized_scores: np.ndarray [N] in [0, 1]
            - raw_scores: np.ndarray [N]
            - contradiction_details: List of metadata dicts with conflict pairs and explanations
        """
        num_sentences = len(sentences)
        if num_sentences == 0:
            return np.array([]), np.array([]), []

        raw_scores = np.zeros(num_sentences, dtype=np.float32)
        details: List[Dict[str, Any]] = []

        # Pairwise embeddings similarity if provided
        sim_matrix = None
        if sentence_embeddings is not None and len(sentence_embeddings) == num_sentences:
            sim_matrix = cosine_similarity_matrix(sentence_embeddings, sentence_embeddings)

        for i in range(num_sentences):
            cand_sent = sentences[i]
            max_contradiction = 0.0
            conflict_info = None

            # Compare sentence i against previous context (up to 5 preceding sentences)
            search_start = max(0, i - 5)
            for j in range(search_start, i):
                prev_sent = sentences[j]
                
                # Check semantic overlap: contradictions happen between sentences talking about the same subject
                lexical_overlap = self._compute_subject_overlap(cand_sent.text, prev_sent.text)
                emb_sim = sim_matrix[i, j] if sim_matrix is not None else 0.5

                if lexical_overlap > 0.15 or emb_sim > 0.40:
                    contra_prob, reasons = self._detect_contradiction_between(cand_sent.text, prev_sent.text)
                    if contra_prob > max_contradiction:
                        max_contradiction = contra_prob
                        conflict_info = {
                            "conflicting_sentence_idx": j,
                            "conflicting_sentence_text": prev_sent.text,
                            "reasons": reasons,
                            "score": float(contra_prob)
                        }

            raw_scores[i] = max_contradiction
            details.append(conflict_info or {"score": 0.0, "reasons": []})

        normalized_scores = min_max_normalize(raw_scores)
        return normalized_scores, raw_scores, details

    def _detect_contradiction_between(self, text_a: str, text_b: str) -> Tuple[float, List[str]]:
        """Evaluates whether text_a directly contradicts text_b."""
        reasons = []
        contradiction_score = 0.0

        words_a = set(re.findall(r'\b\w+\b', text_a.lower()))
        words_b = set(re.findall(r'\b\w+\b', text_b.lower()))

        # 1. Antonym opposition test on shared topic
        for pair_x, pair_y in self.CONTRADICTING_PAIRS:
            match_x_a = any(w.startswith(pair_x) or pair_x in w for w in words_a)
            match_y_b = any(w.startswith(pair_y) or pair_y in w for w in words_b)
            match_y_a = any(w.startswith(pair_y) or pair_y in w for w in words_a)
            match_x_b = any(w.startswith(pair_x) or pair_x in w for w in words_b)

            if (match_x_a and match_y_b) or (match_y_a and match_x_b):
                contradiction_score += 0.70
                reasons.append(f"Opposing semantic terms detected: '{pair_x}' vs '{pair_y}'")
                break

        # 2. Negation asymmetry check
        neg_in_a = bool(self.NEGATION_TERMS.intersection(words_a))
        neg_in_b = bool(self.NEGATION_TERMS.intersection(words_b))
        
        # High shared vocabulary but one is negated and one is affirmative
        shared_content_words = [w for w in (words_a & words_b) if len(w) > 3 and w not in self.NEGATION_TERMS]
        if neg_in_a != neg_in_b and len(shared_content_words) >= 2:
            contradiction_score += 0.50
            reasons.append("Negation polarity inversion on shared subject matter")

        # Cap score at 1.0
        final_score = min(1.0, contradiction_score)
        return final_score, reasons

    def _compute_subject_overlap(self, text_a: str, text_b: str) -> float:
        """Computes Jaccard overlap of non-stopword tokens."""
        words_a = set(re.findall(r'\b\w{3,}\b', text_a.lower()))
        words_b = set(re.findall(r'\b\w{3,}\b', text_b.lower()))
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)
