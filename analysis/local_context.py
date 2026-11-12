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
    Section-aware and technical-vocabulary-aware to prevent false positives.
    """

    TECHNICAL_AND_SECTION_WORDS = {
        # Section titles & organization
        "Introduction", "Overview", "Project", "Methodology", "Methods",
        "Results", "Experimental", "Technical", "Implementation", "Discussion",
        "Conclusion", "Conclusions", "Abstract", "Evaluation", "Background",
        "Architecture", "Analysis", "References", "Appendix", "Contradiction",
        "Detection", "Stage", "Stages", "Phase", "Phases", "Step", "Steps",
        "Context", "Global", "Local",
        # Programming languages, frameworks & NLP tools
        "Python", "Streamlit", "PyTorch", "Torch", "Hugging", "Face", "Transformers",
        "Sentence", "DeBERTa", "BERT", "CrossEncoder", "NumPy", "Pandas",
        "Scikit", "Learn", "NLTK", "Spacy", "TensorFlow", "Keras",
        "API", "REST", "JSON", "CSV", "PDF", "HTML", "SQL", "CUDA",
        "Linux", "Windows", "Git", "Docker", "Database", "Dataset",
        # Common technical/document vocabulary
        "Model", "System", "Application", "Pipeline", "Framework", "Library",
        "Algorithm", "Embedding", "Embeddings", "Artificial", "Intelligence",
        "Machine", "Learning", "Deep", "Natural", "Language", "Processing",
        "Data", "Information", "Report", "Document"
    }

    COMMON_WORDS = {
        "The", "A", "An", "In", "On", "At", "However", "Therefore", "Moreover",
        "Furthermore", "This", "That", "These", "Those", "It", "Twelve", "High",
        "Emergency", "Geological", "Telemetry", "Average", "Scientific", "Global",
        "Comprehensive", "Mitigating", "Future", "Unlike", "Advanced", "Operating"
    }

    def __init__(self, window_size: int = 2):
        self.window_size = window_size
        self._entity_pattern = re.compile(r'\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b')
        self._numeric_pattern = re.compile(r'\b\d+(?:\.\d+)?%?\b')
        self._date_pattern = re.compile(r'\b(?:19|20)\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b')

    def compute_local_inconsistencies(
        self,
        sentence_embeddings: np.ndarray,
        sentences: List[SentenceItem],
        window_size: int | None = None,
        global_scores: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Computes the local contextual inconsistency score L(i) for each sentence.
        Incorporates section boundary awareness, prose-only neighbor filtering,
        and global topic affinity damping.

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

        # Identify structural sentences to exclude from prose neighbor analysis
        is_structural_mask = [
            getattr(s, "is_structural", False) or getattr(s, "is_heading", False)
            for s in sentences
        ]

        for i in range(num_sentences):
            # Guard: Structural headings and markers receive 0.0 local inconsistency
            if is_structural_mask[i]:
                raw_scores[i] = 0.0
                local_details.append({
                    "neighbor_indices": [],
                    "neighbor_sims": [],
                    "mean_neighbor_sim": 1.0,
                    "is_bridge_anomaly": False,
                    "entity_novelty": 0.0,
                    "unshared_entities": [],
                    "entities": [],
                    "numbers": [],
                    "dates": []
                })
                continue

            raw_neighbor_indices = get_sliding_window_neighbors(i, num_sentences, effective_window)
            # Filter out structural elements from neighbors so they don't drag down prose scores
            prose_neighbors = [idx for idx in raw_neighbor_indices if not is_structural_mask[idx]]

            # If all immediate window neighbors were structural, expand window search for nearest prose sentences
            if not prose_neighbors:
                for dist in range(1, max(effective_window * 2 + 1, 6)):
                    for offset in [-dist, dist]:
                        cand_idx = i + offset
                        if 0 <= cand_idx < num_sentences and not is_structural_mask[cand_idx]:
                            if cand_idx not in prose_neighbors:
                                prose_neighbors.append(cand_idx)
                    if len(prose_neighbors) >= 2:
                        break

            if not prose_neighbors:
                raw_scores[i] = 0.0
                local_details.append({"neighbor_sims": [], "is_bridge_anomaly": False, "entity_novelty": 0.0})
                continue

            neighbor_sims = sim_matrix[i, prose_neighbors]
            mean_neighbor_sim = float(np.mean(neighbor_sims))
            max_neighbor_sim = float(np.max(neighbor_sims))

            # Effective continuity: In multi-aspect technical sections, strong connection
            # to AT LEAST ONE neighboring sentence signifies legitimate narrative progression.
            effective_sim = 0.55 * max_neighbor_sim + 0.45 * mean_neighbor_sim
            base_inconsistency = max(0.0, 1.0 - effective_sim)

            # Global Topic Affinity Damping:
            # If the sentence is demonstrably on-topic for the overall document (G(i) is low),
            # damp the local neighbor penalty.
            if global_scores is not None and len(global_scores) == num_sentences:
                g_score = float(global_scores[i])
                if g_score < 0.45:
                    topic_damping = 0.40 + 0.60 * g_score
                    base_inconsistency = base_inconsistency * topic_damping

            # 1. Bridge Anomaly Detection:
            # Sentence i alienates both predecessor and successor who share strong mutual coherence.
            # Must NOT trigger if sentence i is on-topic or an introductory sentence.
            is_bridge_anomaly = False
            bridge_penalty = 0.0
            
            # Find closest preceding and succeeding prose sentences
            prev_prose = [idx for idx in prose_neighbors if idx < i]
            next_prose = [idx for idx in prose_neighbors if idx > i]

            if prev_prose and next_prose:
                p_idx = max(prev_prose)
                n_idx = min(next_prose)
                prev_sim = sim_matrix[i, p_idx]
                next_sim = sim_matrix[i, n_idx]
                surrounding_sim = sim_matrix[p_idx, n_idx]

                # Check if flanking sentences cross section boundaries
                same_section = (getattr(sentences[p_idx], "section_id", 0) == getattr(sentences[n_idx], "section_id", 0))

                # Legitimate transition / on-topic check:
                is_on_topic = (global_scores is not None and len(global_scores) == num_sentences and global_scores[i] < 0.42)
                if same_section and not is_on_topic and surrounding_sim > 0.45 and prev_sim < 0.30 and next_sim < 0.30:
                    is_bridge_anomaly = True
                    bridge_penalty = float(surrounding_sim - max(prev_sim, next_sim))

            # 2. Entity Novelty & Inconsistency check
            current_ents = sentence_entities[i]
            neighbor_ents = set()
            for n_idx in prose_neighbors:
                neighbor_ents.update(sentence_entities[n_idx])

            entity_novelty = 0.0
            unshared_entities = []
            if current_ents and num_sentences >= 3:
                unshared = [e for e in current_ents if e not in neighbor_ents]
                # Only penalize if unshared entity is also alien to the whole document except this sentence
                # AND sentence is globally divergent (not normal on-topic technical prose)
                is_on_topic = (global_scores is not None and len(global_scores) == num_sentences and global_scores[i] < 0.42)
                if not is_on_topic:
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
            if sentence_numbers[i] and prev_prose:
                p_idx = max(prev_prose)
                prev_nums = set(sentence_numbers[p_idx])
                curr_nums = set(sentence_numbers[i])
                shared_words = set(sentences[i].tokens) & set(sentences[p_idx].tokens)
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
                "neighbor_indices": prose_neighbors,
                "neighbor_sims": [float(s) for s in neighbor_sims],
                "mean_neighbor_sim": mean_neighbor_sim,
                "effective_sim": effective_sim,
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
        """
        Extracts genuine proper named entities, ignoring section titles,
        common English words, and technical programming terminology.
        """
        tokens = re.findall(r'\b[A-Za-z0-9_-]+\b', text)
        if not tokens:
            return set()

        entities = set()
        all_excluded = self.COMMON_WORDS | self.TECHNICAL_AND_SECTION_WORDS

        # Subsequent capitalized tokens (ignoring sentence-initial capital unless multi-word)
        for token in tokens[1:]:
            if token[0].isupper() and token not in all_excluded and len(token) > 2:
                entities.add(token)

        # Multi-word proper names (e.g. "Detective Miller", "Jezero Crater")
        multi_matches = self._entity_pattern.findall(text)
        for m in multi_matches:
            parts = m.split()
            if len(parts) >= 2:
                # If all parts are technical/section words, ignore
                if not all(p in all_excluded for p in parts) and parts[0] not in self.COMMON_WORDS:
                    # Ignore title-cased section headings
                    if not any(p in self.TECHNICAL_AND_SECTION_WORDS for p in parts):
                        entities.add(m.strip())

        return entities
