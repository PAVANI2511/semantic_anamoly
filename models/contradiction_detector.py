"""
Module: Contradiction and Semantic Inconsistency Detection.
Combines Pretrained Natural Language Inference (CrossEncoder NLI) with
lexical polarity and negation fallback detection.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import re
import os

from utils.logging_config import logger
from utils.helpers import cosine_similarity_matrix
from preprocessing.sentence_segmenter import SentenceItem
from config.settings import config


class ContradictionDetector:
    """
    Dual-mode contradiction detector:
    1. Pretrained NLI Transformer (CrossEncoder) evaluating [Contradiction, Entailment, Neutral]
    2. Lexical / Polarity / Negation heuristic analyzer (Fallback & fast mode)
    """

    CONTRADICTING_PAIRS = [
        ("increas", "decreas"), ("ris", "fall"), ("rose", "fell"), ("gain", "loss"),
        ("rais", "reduc"), ("raised", "reduced"), ("surge", "plunge"), ("surged", "plunged"),
        ("profit", "loss"), ("success", "fail"), ("succeed", "fail"), ("successful", "unsuccessful"),
        ("grow", "shrink"), ("growth", "decline"),
        ("true", "false"), ("correct", "incorrect"), ("valid", "invalid"),
        ("positive", "negative"), ("enable", "disable"), ("safe", "danger"),
        ("accept", "reject"), ("approv", "deni"), ("always", "never"),
        ("present", "absent"), ("highest", "lowest"), ("high", "low")
    ]

    NEGATION_TERMS = {"not", "never", "no", "neither", "nor", "hardly", "scarcely", "barely", "none", "cannot", "n't"}

    _cached_nli_model: Any = None
    _cached_model_name: Optional[str] = None

    def __init__(
        self,
        use_nli: bool = True,
        model_name: Optional[str] = None,
        max_context_pairs: int = 6
    ):
        self.use_nli = use_nli
        self.model_name = model_name or config.model.nli_model_name
        self.max_context_pairs = max_context_pairs
        self.is_nli_active = False
        self._nli_model = None

        if self.use_nli:
            self._load_nli_model()

    def _load_nli_model(self) -> None:
        """Loads and caches the pretrained CrossEncoder NLI transformer."""
        # Reuse class-level cached instance if matching
        if ContradictionDetector._cached_nli_model is not None and ContradictionDetector._cached_model_name == self.model_name:
            self._nli_model = ContradictionDetector._cached_nli_model
            self.is_nli_active = True
            return

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder NLI model: {self.model_name}...")
            model = CrossEncoder(self.model_name)
            self._nli_model = model
            self.is_nli_active = True
            ContradictionDetector._cached_nli_model = model
            ContradictionDetector._cached_model_name = self.model_name
            logger.info(f"NLI model {self.model_name} loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Failed to load NLI model '{self.model_name}' ({e}). "
                f"Falling back to rule-based lexical contradiction detector."
            )
            self._nli_model = None
            self.is_nli_active = False

    @property
    def backend(self) -> str:
        """Returns the active backend name: 'nli_transformer' or 'lexical_fallback'."""
        return "nli_transformer" if (self.use_nli and self.is_nli_active) else "lexical_fallback"

    @backend.setter
    def backend(self, val: str) -> None:
        """Dynamically switches between NLI transformer and lexical fallback backends."""
        if val in {"lexical", "lexical_fallback", "rule_based"}:
            self.use_nli = False
            self.is_nli_active = False
        else:
            self.use_nli = True
            if self._nli_model is not None:
                self.is_nli_active = True
            else:
                self._load_nli_model()

    def get_active_backend(self) -> str:
        """Returns the name of the active detection backend."""
        return self.backend

    def compute_contradiction_scores(
        self,
        sentences: List[SentenceItem],
        sentence_embeddings: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Evaluates contradiction score C(i) for each sentence against preceding context.

        Returns:
            - normalized_scores: np.ndarray [N] in [0, 1]
            - raw_scores: np.ndarray [N]
            - details: List of metadata dicts with conflict pairs and explanations
        """
        num_sentences = len(sentences)
        if num_sentences == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), []

        if num_sentences == 1:
            return (
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                [{"score": 0.0, "backend": self.get_active_backend(), "reasons": []}]
            )

        raw_scores = np.zeros(num_sentences, dtype=np.float32)
        details: List[Dict[str, Any]] = []

        # Precompute similarity matrix if embeddings available
        sim_matrix = None
        if sentence_embeddings is not None and len(sentence_embeddings) == num_sentences:
            sim_matrix = cosine_similarity_matrix(sentence_embeddings, sentence_embeddings)

        for i in range(num_sentences):
            cand_sent = sentences[i]
            max_contradiction = 0.0
            best_conflict: Dict[str, Any] = {
                "score": 0.0,
                "backend": self.get_active_backend(),
                "conflicting_sentence_idx": None,
                "reasons": []
            }

            # Select candidate antecedent pairs within window
            search_start = max(0, i - self.max_context_pairs)
            pairs_to_test = []
            pair_indices = []

            for j in range(search_start, i):
                prev_sent = sentences[j]
                # Filter candidates: test pairs that share lexical overlap OR moderate semantic similarity
                lexical_overlap = self._compute_subject_overlap(cand_sent.text, prev_sent.text)
                emb_sim = sim_matrix[i, j] if sim_matrix is not None else 0.50

                # Contradiction requires a shared topic/subject
                if lexical_overlap > 0.10 or emb_sim > 0.35:
                    pairs_to_test.append((prev_sent.text, cand_sent.text))
                    pair_indices.append(j)

            # Execution Mode A: NLI CrossEncoder Transformer
            if self.use_nli and self.is_nli_active and pairs_to_test and self._nli_model is not None:
                try:
                    # Model outputs logits for [contradiction, entailment, neutral]
                    logits = self._nli_model.predict(pairs_to_test)
                    if not isinstance(logits, np.ndarray):
                        logits = np.array(logits)

                    if logits.ndim == 1:
                        logits = logits.reshape(1, -1)

                    # Softmax probabilities
                    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

                    for p_idx, (j, prob_vec) in enumerate(zip(pair_indices, probs)):
                        contra_prob = float(prob_vec[0])
                        entail_prob = float(prob_vec[1])
                        neutral_prob = float(prob_vec[2])

                        # High contradiction probability
                        if contra_prob > max_contradiction:
                            max_contradiction = contra_prob
                            best_conflict = {
                                "score": contra_prob,
                                "backend": "nli_transformer",
                                "conflicting_sentence_idx": j,
                                "conflicting_sentence_text": sentences[j].text,
                                "entailment_prob": round(entail_prob, 4),
                                "neutral_prob": round(neutral_prob, 4),
                                "reasons": [
                                    f"NLI Cross-Encoder detected contradiction with sentence #{j+1} "
                                    f"(Probability: {contra_prob*100:.1f}%)"
                                ]
                            }
                except Exception as e:
                    logger.warning(f"NLI inference error ({e}). Using lexical fallback.")

            # Execution Mode B: Lexical Fallback / Augmenter
            # Check lexical antonyms and negation asymmetries
            for j in range(search_start, i):
                prev_sent = sentences[j]
                lex_score, lex_reasons = self._detect_lexical_contradiction(cand_sent.text, prev_sent.text)
                if lex_score > max_contradiction:
                    max_contradiction = lex_score
                    best_conflict = {
                        "score": lex_score,
                        "backend": "lexical_fallback" if not self.is_nli_active else "nli_augmented_lexical",
                        "conflicting_sentence_idx": j,
                        "conflicting_sentence_text": prev_sent.text,
                        "reasons": lex_reasons
                    }

            raw_scores[i] = max_contradiction
            details.append(best_conflict)

        # Calibrate scores to [0, 1] without artificial relative distortion
        normalized_scores = np.clip(raw_scores, 0.0, 1.0).astype(np.float32)
        return normalized_scores, raw_scores.astype(np.float32), details

    def _detect_contradiction_between(self, text_a: str, text_b: str) -> Tuple[float, List[str]]:
        """Pairwise contradiction detection between two texts using active backend."""
        lex_score, lex_reasons = self._detect_lexical_contradiction(text_a, text_b)
        if self.use_nli and self.is_nli_active and self._nli_model is not None:
            try:
                logits = self._nli_model.predict([(text_a, text_b)])
                if not isinstance(logits, np.ndarray):
                    logits = np.array(logits)
                if logits.ndim == 1:
                    logits = logits.reshape(1, -1)
                exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
                contra_prob = float(probs[0, 0])
                if contra_prob >= 0.40:
                    combined_reasons = [f"NLI Cross-Encoder detected contradiction (Probability: {contra_prob*100:.1f}%)"] + lex_reasons
                    return max(contra_prob, lex_score), combined_reasons
            except Exception:
                pass
        return lex_score, lex_reasons

    def _detect_lexical_contradiction(self, text_a: str, text_b: str) -> Tuple[float, List[str]]:
        """Evaluates whether text_a directly contradicts text_b using lexical cues."""
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
        
        shared_content_words = [w for w in (words_a & words_b) if len(w) > 3 and w not in self.NEGATION_TERMS]
        if neg_in_a != neg_in_b and len(shared_content_words) >= 2:
            contradiction_score += 0.50
            reasons.append(f"Negation polarity inversion on shared vocabulary: {', '.join(shared_content_words[:3])}")

        return min(1.0, contradiction_score), reasons

    def _compute_subject_overlap(self, text_a: str, text_b: str) -> float:
        """Computes Jaccard overlap of non-stopword tokens."""
        words_a = set(re.findall(r'\b\w{3,}\b', text_a.lower()))
        words_b = set(re.findall(r'\b\w{3,}\b', text_b.lower()))
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)
