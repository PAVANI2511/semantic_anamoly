"""
Configuration settings for Semantic Sentinel (DGSA Framework).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

# Avoid network timeouts when offline if models are cached
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
if Path(os.path.expanduser("~/.cache/huggingface/hub")).exists():
    os.environ["HF_HUB_OFFLINE"] = "1"

# Base Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_documents"
DATASETS_DIR = DATA_DIR / "datasets"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
BENCHMARKS_DIR = OUTPUTS_DIR / "benchmarks"

for directory in [DATA_DIR, SAMPLE_DOCS_DIR, DATASETS_DIR, PROCESSED_DIR, OUTPUTS_DIR, REPORTS_DIR, BENCHMARKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for embedding and inference models."""
    primary_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    fast_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model_name: str = "cross-encoder/nli-deberta-v3-xsmall"
    primary_nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    use_nli: bool = True
    nli_fallback_enabled: bool = True
    device: str = "auto"  # "auto", "cuda", or "cpu"
    batch_size: int = 32
    max_seq_length: int = 256
    cache_dir: Path = BASE_DIR / ".cache" / "models"


@dataclass
class DGSAParameters:
    """
    Mathematical weights and hyperparameters for DGSA scoring:
    A(i) = alpha * G(i) + beta * L(i) + gamma * C(i)
    """
    alpha: float = 0.40  # Global context deviation weight
    beta: float = 0.35   # Local contextual inconsistency weight
    gamma: float = 0.25  # Contradiction / polarity deviation weight
    
    # Context window: number of neighboring sentences before & after to consider
    local_window_size: int = 2
    
    # Anomaly classification threshold
    anomaly_threshold: float = 0.50
    
    # Sensitivity presets adjusting threshold
    sensitivity_presets: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.65,      # High specificity, fewer false positives
        "medium": 0.50,   # Balanced precision/recall
        "high": 0.35      # High sensitivity, flags subtle deviations
    })

    def validate(self) -> None:
        """Validates that weights are non-negative and normalizes them to sum to 1.0."""
        if self.alpha < 0.0 or self.beta < 0.0 or self.gamma < 0.0:
            raise ValueError(f"DGSA weights must be non-negative: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")
        total = self.alpha + self.beta + self.gamma
        if total <= 0.0:
            raise ValueError("Sum of DGSA weights must be strictly positive.")
        self.alpha = float(self.alpha / total)
        self.beta = float(self.beta / total)
        self.gamma = float(self.gamma / total)

    def clone(self) -> "DGSAParameters":
        """Returns an independent deep copy to prevent cross-run state pollution."""
        return DGSAParameters(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            local_window_size=self.local_window_size,
            anomaly_threshold=self.anomaly_threshold,
            sensitivity_presets=dict(self.sensitivity_presets)
        )


@dataclass
class AnomalyCategories:
    """Defined semantic anomaly types."""
    TOPIC_DEVIATION = "topic_deviation"
    CONTRADICTION = "contradiction"
    CONTEXTUAL_INCONSISTENCY = "contextual_inconsistency"
    ENTITY_INCONSISTENCY = "entity_inconsistency"
    ATTRIBUTE_INCONSISTENCY = "attribute_inconsistency"
    NORMAL = "normal"

    ALL_CATEGORIES: List[str] = field(default_factory=lambda: [
        "topic_deviation",
        "contradiction",
        "contextual_inconsistency",
        "entity_inconsistency",
        "attribute_inconsistency",
    ])


@dataclass
class AppConfig:
    """Master application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dgsa: DGSAParameters = field(default_factory=DGSAParameters)
    categories: AnomalyCategories = field(default_factory=AnomalyCategories)
    supported_formats: List[str] = field(default_factory=lambda: [".txt", ".pdf", ".docx"])
    max_file_size_mb: int = 15
    random_seed: int = 42


# Default singleton instance
config = AppConfig()
