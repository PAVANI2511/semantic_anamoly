"""
Dataset Loader and Splitter for Semantic Sentinel.
Supports JSON and CSV benchmark datasets with document leakage prevention.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import csv
import pandas as pd

from utils.logging_config import logger


@dataclass
class AnnotatedSentence:
    """A sentence with ground-truth anomaly annotations."""
    sentence_id: int
    text: str
    is_anomaly: int  # 0 or 1
    category: str
    token_spans: List[Dict[str, Any]] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0


@dataclass
class AnnotatedDocument:
    """A document containing annotated sentences."""
    doc_id: str
    title: str
    domain: str
    text: str
    is_synthetic: bool
    sentences: List[AnnotatedSentence] = field(default_factory=list)


class DatasetLoader:
    """Loads, validates, and splits datasets for evaluation."""

    @classmethod
    def load_from_json(cls, json_path: str | Path) -> List[AnnotatedDocument]:
        """Loads annotated documents from a structured JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        raw_docs = data.get("documents", data) if isinstance(data, dict) else data
        for item in raw_docs:
            sentences = []
            for s_data in item.get("sentences", []):
                sentences.append(AnnotatedSentence(
                    sentence_id=s_data["sentence_id"],
                    text=s_data["text"],
                    is_anomaly=int(s_data.get("is_anomaly", 0)),
                    category=s_data.get("category", "normal"),
                    token_spans=s_data.get("token_spans", []),
                    start_char=s_data.get("start_char", 0),
                    end_char=s_data.get("end_char", len(s_data["text"]))
                ))

            docs.append(AnnotatedDocument(
                doc_id=str(item.get("doc_id", len(docs))),
                title=item.get("title", f"Doc_{len(docs)}"),
                domain=item.get("domain", "general"),
                text=item.get("text", " ".join(s.text for s in sentences)),
                is_synthetic=item.get("is_synthetic", True),
                sentences=sentences
            ))

        logger.info(f"Loaded {len(docs)} annotated documents from {path}")
        return docs

    @classmethod
    def split_train_test(
        cls,
        documents: List[AnnotatedDocument],
        test_ratio: float = 0.30,
        random_seed: int = 42
    ) -> Tuple[List[AnnotatedDocument], List[AnnotatedDocument]]:
        """
        Splits dataset by DOCUMENT ID to strictly prevent sentence leakage between train and test.
        """
        import random
        rng = random.Random(random_seed)
        shuffled = list(documents)
        rng.shuffle(shuffled)

        n_test = max(1, int(len(shuffled) * test_ratio))
        test_docs = shuffled[:n_test]
        train_docs = shuffled[n_test:]

        logger.info(f"Split documents into {len(train_docs)} train and {len(test_docs)} test (Zero document leakage)")
        return train_docs, test_docs

    @classmethod
    def flatten_sentences(cls, documents: List[AnnotatedDocument]) -> Tuple[List[str], List[int], List[str]]:
        """Extracts texts, labels (0/1), and categories from document collections."""
        texts = []
        labels = []
        categories = []
        for doc in documents:
            for s in doc.sentences:
                texts.append(s.text)
                labels.append(s.is_anomaly)
                categories.append(s.category)
        return texts, labels, categories
