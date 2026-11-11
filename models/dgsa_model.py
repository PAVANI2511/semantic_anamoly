"""
DGSA Model: Dual-Granularity Semantic Anomaly Detection Framework.
Unifies global context, local neighborhood consistency, contradiction analysis,
span localization, and explainable AI into an end-to-end pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import numpy as np

from utils.logging_config import logger
from config.settings import config, DGSAParameters
from preprocessing.sentence_segmenter import SentenceItem, SentenceSegmenter
from models.embedding_model import BaseEmbeddingModel, EmbeddingFactory
from analysis.global_context import GlobalContextAnalyzer
from analysis.local_context import LocalContextAnalyzer
from models.contradiction_detector import ContradictionDetector
from analysis.anomaly_scoring import DGSAScorer
from models.span_localizer import SpanLocalizer
from analysis.explanation_engine import ExplanationEngine, AnomalyExplanation


@dataclass
class DGSAResult:
    """Complete document analysis result produced by the DGSA framework."""
    raw_text: str
    sentences: List[SentenceItem]
    anomaly_scores: List[float]
    is_anomaly: List[bool]
    categories: List[str]
    explanations: List[AnomalyExplanation]
    global_scores: List[float]
    local_scores: List[float]
    contradiction_scores: List[float]
    
    # Metadata
    total_sentences: int
    num_anomalous: int
    num_normal: int
    anomaly_percentage: float
    threshold_used: float
    weights_used: Dict[str, float]
    inference_time_ms: float
    document_assessment: str
    embedding_model_name: str


class DGSAModel:
    """
    Dual-Granularity Semantic Anomaly (DGSA) Detector.
    
    Core Formulation:
        A(i) = alpha * G(i) + beta * L(i) + gamma * C(i)
    """

    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        embedding_backend: str = "transformer",
        params: Optional[DGSAParameters] = None
    ):
        self.params = params or config.dgsa
        self.embedding_model = embedding_model or EmbeddingFactory.get_embedding_model(backend=embedding_backend)
        
        # Modules
        self.global_analyzer = GlobalContextAnalyzer()
        self.local_analyzer = LocalContextAnalyzer(window_size=self.params.local_window_size)
        self.contradiction_detector = ContradictionDetector()
        self.scorer = DGSAScorer(
            alpha=self.params.alpha,
            beta=self.params.beta,
            gamma=self.params.gamma,
            threshold=self.params.anomaly_threshold
        )
        self.span_localizer = SpanLocalizer()
        self.segmenter = SentenceSegmenter()

    def analyze_document(
        self,
        text: str,
        paragraphs: Optional[List[str]] = None,
        custom_threshold: Optional[float] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        enable_global: bool = True,
        enable_local: bool = True,
        enable_contradiction: bool = True
    ) -> DGSAResult:
        """
        Executes end-to-end DGSA pipeline on document text.
        """
        start_time = time.perf_counter()

        # Step 1: Preprocessing & Sentence Segmentation
        sentences = self.segmenter.segment(text, paragraphs=paragraphs)
        if not sentences:
            return self._empty_result(text)

        threshold = custom_threshold if custom_threshold is not None else self.scorer.threshold
        if custom_weights:
            self.scorer.alpha = custom_weights.get("alpha", self.scorer.alpha)
            self.scorer.beta = custom_weights.get("beta", self.scorer.beta)
            self.scorer.gamma = custom_weights.get("gamma", self.scorer.gamma)

        # Step 2: Semantic Representation
        sentence_texts = [s.cleaned_text for s in sentences]
        embeddings = self.embedding_model.embed_sentences(sentence_texts)

        # Step 3: Global Semantic Context Analysis
        norm_global, raw_global, global_meta = self.global_analyzer.compute_global_deviations(
            embeddings, sentences
        )

        # Step 4: Local Semantic Context Analysis
        norm_local, raw_local, local_details = self.local_analyzer.compute_local_inconsistencies(
            embeddings, sentences
        )

        # Step 5: Contradiction Detection
        norm_contra, raw_contra, contra_details = self.contradiction_detector.compute_contradiction_scores(
            sentences, embeddings
        )

        # Step 6: Dual-Level Composite Anomaly Scoring A(i)
        composite_scores, eff_alpha, eff_beta, eff_gamma = self.scorer.compute_composite_scores(
            norm_global, norm_local, norm_contra,
            enable_global=enable_global,
            enable_local=enable_local,
            enable_contradiction=enable_contradiction
        )

        # Step 7: Classification, Category Inference & Span Localization
        num_sentences = len(sentences)
        is_anomaly: List[bool] = []
        categories: List[str] = []
        explanations: List[AnomalyExplanation] = []

        # Create global vocabulary bag for span outlier matching
        all_words = set()
        for s in sentences:
            all_words.update(w.lower() for w in s.tokens if len(w) > 3)

        for i in range(num_sentences):
            score = float(composite_scores[i])
            flagged = score >= threshold
            is_anomaly.append(flagged)

            # Infer category
            cat = self.scorer.classify_category(
                float(norm_global[i]),
                float(norm_local[i]),
                float(norm_contra[i]),
                local_details[i],
                contra_details[i]
            )
            categories.append(cat)

            # Localize spans
            loc_result = self.span_localizer.localize_spans(
                sentence=sentences[i],
                anomaly_category=cat,
                anomaly_score=score,
                global_context_tokens=all_words,
                contradiction_details=contra_details[i]
            )

            # Generate explainable AI report
            explanation = ExplanationEngine.generate_explanation(
                sentence=sentences[i],
                all_sentences=sentences,
                category=cat,
                anomaly_score=score,
                is_anomalous=flagged,
                global_score=float(norm_global[i]),
                local_score=float(norm_local[i]),
                contradiction_score=float(norm_contra[i]),
                alpha=eff_alpha,
                beta=eff_beta,
                gamma=eff_gamma,
                local_details=local_details[i],
                contradiction_details=contra_details[i],
                localized_spans=loc_result.spans
            )
            explanations.append(explanation)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        num_anomalous = sum(is_anomaly)
        anomaly_pct = (num_anomalous / num_sentences * 100) if num_sentences > 0 else 0.0

        # Assess document health
        if anomaly_pct == 0:
            doc_assessment = "Semantically Coherent: No anomalies detected."
        elif anomaly_pct < 15:
            doc_assessment = "Mild Inconsistencies: Document is mostly coherent with minor deviations."
        elif anomaly_pct < 35:
            doc_assessment = "Moderate Anomaly Density: Notable topical or contextual fractures observed."
        else:
            doc_assessment = "Severe Semantic Fragmentation: Substantial contradictions or topic disruptions detected."

        model_name = getattr(self.embedding_model, "model_name", "TF-IDF Vectorizer")

        return DGSAResult(
            raw_text=text,
            sentences=sentences,
            anomaly_scores=[float(s) for s in composite_scores],
            is_anomaly=is_anomaly,
            categories=categories,
            explanations=explanations,
            global_scores=[float(s) for s in norm_global],
            local_scores=[float(s) for s in norm_local],
            contradiction_scores=[float(s) for s in norm_contra],
            total_sentences=num_sentences,
            num_anomalous=num_anomalous,
            num_normal=num_sentences - num_anomalous,
            anomaly_percentage=round(anomaly_pct, 2),
            threshold_used=threshold,
            weights_used={"alpha": eff_alpha, "beta": eff_beta, "gamma": eff_gamma},
            inference_time_ms=round(elapsed_ms, 2),
            document_assessment=doc_assessment,
            embedding_model_name=model_name
        )

    def _empty_result(self, text: str) -> DGSAResult:
        return DGSAResult(
            raw_text=text,
            sentences=[],
            anomaly_scores=[],
            is_anomaly=[],
            categories=[],
            explanations=[],
            global_scores=[],
            local_scores=[],
            contradiction_scores=[],
            total_sentences=0,
            num_anomalous=0,
            num_normal=0,
            anomaly_percentage=0.0,
            threshold_used=self.scorer.threshold,
            weights_used={"alpha": self.params.alpha, "beta": self.params.beta, "gamma": self.params.gamma},
            inference_time_ms=0.0,
            document_assessment="Empty Document",
            embedding_model_name=config.model.primary_embedding_model
        )
