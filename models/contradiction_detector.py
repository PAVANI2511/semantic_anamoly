"""
Module: Contradiction and Semantic Inconsistency Detection.
Combines Pretrained Natural Language Inference (CrossEncoder NLI) with
lexical polarity and negation fallback detection.
Supports safe lazy loading, dependency isolation, and automatic fallback on inference failure.
"""

import os
import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

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

    GENERIC_DOMAIN_TERMS = {
        "system", "systems", "model", "models", "analysis", "context",
        "document", "documents", "process", "processes", "approach",
        "method", "methods", "stage", "stages", "component", "components",
        "data", "application", "sentence", "sentences", "information"
    }

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
        self._nli_model: Any = None

        if self.use_nli:
            self._load_nli_model()

    @staticmethod
    def is_nli_dependency_available() -> bool:
        """Checks if sentence_transformers is installed and importable."""
        try:
            import sentence_transformers  # noqa: F401
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    @classmethod
    def _import_cross_encoder(cls):
        """
        Safely imports the CrossEncoder class from sentence_transformers.
        Isolated to allow clean unit testing and mocking independent of the optional dependency.
        """
        from sentence_transformers import CrossEncoder
        return CrossEncoder

    def _load_nli_model(self, force_reload: bool = False) -> bool:
        """
        Loads and caches the pretrained CrossEncoder NLI transformer safely.
        Returns True if the model is ready, False otherwise.
        """
        # Reuse class-level cached instance if matching
        if not force_reload and ContradictionDetector._cached_nli_model is not None and ContradictionDetector._cached_model_name == self.model_name:
            self._nli_model = ContradictionDetector._cached_nli_model
            self.is_nli_active = True
            return True

        try:
            cross_encoder_cls = self._import_cross_encoder()
            logger.info(f"Loading CrossEncoder NLI model: {self.model_name}...")
            is_offline = os.environ.get("SEMANTIC_SENTINEL_OFFLINE", "").strip().lower() in {"1", "true", "yes"} or os.environ.get("HF_HUB_OFFLINE", "") == "1"
            if is_offline:
                try:
                    model = cross_encoder_cls(self.model_name, local_files_only=True)
                except TypeError:
                    model = cross_encoder_cls(self.model_name)
            else:
                try:
                    model = cross_encoder_cls(self.model_name)
                except Exception as net_err:
                    logger.info(f"Initial remote load failed ({net_err}); attempting local cache fallback...")
                    try:
                        model = cross_encoder_cls(self.model_name, local_files_only=True)
                    except Exception:
                        raise net_err

            self._nli_model = model
            self.is_nli_active = True
            ContradictionDetector._cached_nli_model = model
            ContradictionDetector._cached_model_name = self.model_name
            logger.info(f"NLI model {self.model_name} loaded successfully.")
            return True
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(
                f"Optional NLI dependency 'sentence_transformers' unavailable ({e}). "
                f"Activating rule-based lexical contradiction detector."
            )
            self._nli_model = None
            self.is_nli_active = False
            return False
        except Exception as e:
            logger.warning(
                f"Failed to load NLI model '{self.model_name}' ({e}). "
                f"Falling back to rule-based lexical contradiction detector."
            )
            self._nli_model = None
            self.is_nli_active = False
            return False

    @property
    def backend(self) -> str:
        """Returns the active backend name: 'nli_transformer' or 'lexical_fallback'."""
        return "nli_transformer" if (self.use_nli and self.is_nli_active and self._nli_model is not None) else "lexical_fallback"

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

    def _predict_pairs_nli(self, pairs_to_test: List[Tuple[str, str]]) -> Optional[np.ndarray]:
        """
        Executes CrossEncoder inference on sentence pairs.
        Returns softmax probabilities array [K, 3] or None if inference raises an exception.
        """
        if not self._nli_model or not pairs_to_test:
            return None

        try:
            logits = self._nli_model.predict(pairs_to_test)
            if not isinstance(logits, np.ndarray):
                logits = np.array(logits)

            if logits.ndim == 1:
                logits = logits.reshape(1, -1)

            # Softmax probabilities [Contradiction, Entailment, Neutral]
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            return probs
        except Exception as e:
            logger.warning(f"NLI inference error ({e}). Seamlessly activating lexical fallback.")
            return None

    @classmethod
    def _is_evaluable_for_contradiction(cls, s: SentenceItem) -> bool:
        """
        Determines whether a sentence is evaluable for contradiction.
        Structural elements, section headings, list numbers, and short non-sentence fragments
        must never be evaluated as premises or hypotheses in contradiction detection.
        """
        if getattr(s, "is_structural", False) or getattr(s, "is_heading", False):
            return False

        text = s.cleaned_text if hasattr(s, "cleaned_text") and s.cleaned_text else s.text
        words = text.split()
        if len(words) < 3 or len(text) < 8:
            return False

        # Headings or title phrases without terminal punctuation
        if len(words) <= 3 and not text.endswith(('.', '!', '?', ';', '"', "'")):
            return False

        # Ignore markdown heading hashes
        if text.startswith(("#", "##", "###", "####")):
            return False

        # Ignore standalone list numbers or markers
        if re.match(r'^(?:\d{1,3}\.|\([0-9a-zA-Z]+\)|[a-zA-Z]\.|\*|-|\+)\s*$', text):
            return False

        return True

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

            # Guard 1: Non-evaluable sentences (headings, list numbers, fragments)
            if not self._is_evaluable_for_contradiction(cand_sent):
                raw_scores[i] = 0.0
                details.append(best_conflict)
                continue

            # Select candidate antecedent pairs within context window
            search_start = max(0, i - self.max_context_pairs)
            pairs_to_test = []
            pair_indices = []

            for j in range(search_start, i):
                prev_sent = sentences[j]
                # Guard 2: Antecedent must also be evaluable prose
                if not self._is_evaluable_for_contradiction(prev_sent):
                    continue

                lexical_overlap = self._compute_subject_overlap(cand_sent.text, prev_sent.text)
                emb_sim = sim_matrix[i, j] if sim_matrix is not None else 0.50

                # Contradiction candidates require a shared topic or moderate semantic continuity
                # Sentences discussing disjoint topics/tools are not contradictory hypotheses
                if lexical_overlap >= 0.15 or emb_sim >= 0.52:
                    pairs_to_test.append((prev_sent.text, cand_sent.text))
                    pair_indices.append(j)

            nli_success = False

            # Execution Mode A: Pretrained NLI Transformer
            if self.use_nli and self.is_nli_active and pairs_to_test:
                probs = self._predict_pairs_nli(pairs_to_test)
                if probs is not None:
                    nli_success = True
                    for p_idx, (j, prob_vec) in enumerate(zip(pair_indices, probs)):
                        contra_prob = float(prob_vec[0])
                        entail_prob = float(prob_vec[1])
                        neutral_prob = float(prob_vec[2])

                        # Genuine contradiction criteria:
                        # 1. Contradiction probability must dominate over neutral and entailment
                        # 2. If neutral probability is high (>= 0.45) or contra <= neutral, it is neutral exposition
                        if contra_prob >= 0.60 and contra_prob > neutral_prob and contra_prob > entail_prob:
                            lex_score, _ = self._detect_lexical_contradiction(cand_sent.text, sentences[j].text)
                            has_clue = lex_score > 0.30

                            # Verify that the two sentences actually discuss the same specific subject/claim
                            w1 = set(re.findall(r'\b\w{4,}\b', cand_sent.text.lower())) - self.GENERIC_DOMAIN_TERMS
                            w2 = set(re.findall(r'\b\w{4,}\b', sentences[j].text.lower())) - self.GENERIC_DOMAIN_TERMS
                            has_specific_overlap = len(w1.intersection(w2)) > 0

                            # If no lexical clue exists and the pair lacks shared specific claim vocabulary,
                            # they are descriptive parallel statements (e.g. list items), not contradictory assertions
                            if not has_clue and not has_specific_overlap:
                                continue

                            # If no lexical polarity/negation clue exists and confidence is moderate, calibrate
                            effective_prob = contra_prob
                            if not has_clue and contra_prob < 0.85 and neutral_prob > 0.25:
                                effective_prob = float(contra_prob * (1.0 - neutral_prob))

                            if effective_prob > max_contradiction:
                                max_contradiction = effective_prob
                                best_conflict = {
                                    "score": effective_prob,
                                    "backend": "nli_transformer",
                                    "conflicting_sentence_idx": j,
                                    "conflicting_sentence_text": sentences[j].text,
                                    "entailment_prob": round(entail_prob, 4),
                                    "neutral_prob": round(neutral_prob, 4),
                                    "reasons": [
                                        f"NLI Cross-Encoder detected contradiction with sentence #{j+1} "
                                        f"(Probability: {effective_prob*100:.1f}%)"
                                    ]
                                }

            # Execution Mode B: Lexical Fallback / Augmenter
            # Executed if NLI is inactive, or if NLI inference failed, or to augment lexical antonym clues
            if not nli_success or max_contradiction < 0.50:
                for j in range(search_start, i):
                    prev_sent = sentences[j]
                    if not self._is_evaluable_for_contradiction(prev_sent):
                        continue
                    lex_score, lex_reasons = self._detect_lexical_contradiction(cand_sent.text, prev_sent.text)
                    if lex_score > max_contradiction:
                        max_contradiction = lex_score
                        best_conflict = {
                            "score": lex_score,
                            "backend": "lexical_fallback" if not nli_success else "nli_augmented_lexical",
                            "conflicting_sentence_idx": j,
                            "conflicting_sentence_text": prev_sent.text,
                            "reasons": lex_reasons
                        }

            raw_scores[i] = max_contradiction
            details.append(best_conflict)

        # Calibrate output range into [0, 1]
        norm_scores = np.clip(raw_scores, 0.0, 1.0)
        return norm_scores, raw_scores, details

    def _compute_subject_overlap(self, sent1: str, sent2: str) -> float:
        """Calculates Jaccard overlap of non-stopword content tokens."""
        w1 = set(re.findall(r'\b\w{4,}\b', sent1.lower()))
        w2 = set(re.findall(r'\b\w{4,}\b', sent2.lower()))
        if not w1 or not w2:
            return 0.0
        return len(w1.intersection(w2)) / len(w1.union(w2))

    def _detect_lexical_contradiction(self, sent_curr: str, sent_prev: str) -> Tuple[float, List[str]]:
        """
        Rule-based lexical analyzer detecting polar antonyms, negation flips, and numeric discrepancies.
        """
        curr_lower = sent_curr.lower()
        prev_lower = sent_prev.lower()
        reasons = []
        score = 0.0

        # Check opposing polar semantic pairs
        for w1, w2 in self.CONTRADICTING_PAIRS:
            if (w1 in curr_lower and w2 in prev_lower) or (w2 in curr_lower and w1 in prev_lower):
                reasons.append(f"Opposing antonym stems: '{w1}' vs '{w2}'")
                score = max(score, 0.85)

        # Check asymmetric negation on shared predicate
        has_neg_curr = any(re.search(rf'\b{re.escape(neg)}\b', curr_lower) for neg in self.NEGATION_TERMS)
        has_neg_prev = any(re.search(rf'\b{re.escape(neg)}\b', prev_lower) for neg in self.NEGATION_TERMS)

        if has_neg_curr != has_neg_prev:
            overlap = self._compute_subject_overlap(sent_curr, sent_prev)
            if overlap > 0.20:
                reasons.append("Negation polarity flip across shared subject predicate")
                score = max(score, 0.75)

        # Numerical discrepancy check on identical metric terms
        nums_curr = re.findall(r'\b\d+(?:\.\d+)?%?\b', curr_lower)
        nums_prev = re.findall(r'\b\d+(?:\.\d+)?%?\b', prev_lower)
        if nums_curr and nums_prev and nums_curr != nums_prev:
            overlap = self._compute_subject_overlap(sent_curr, sent_prev)
            if overlap > 0.25:
                reasons.append(f"Conflicting numerical values: {nums_curr} vs {nums_prev}")
                score = max(score, 0.70)

        return score, reasons

    def _detect_contradiction_between(self, sent_curr: str, sent_prev: str) -> Tuple[float, List[str]]:
        """Convenience method for testing pairwise contradiction."""
        if self.use_nli and self.is_nli_active:
            probs = self._predict_pairs_nli([(sent_prev, sent_curr)])
            if probs is not None:
                prob = float(probs[0, 0])
                if prob > 0.50:
                    return prob, [f"NLI Cross-Encoder contradiction (prob={prob:.2f})"]
        return self._detect_lexical_contradiction(sent_curr, sent_prev)
