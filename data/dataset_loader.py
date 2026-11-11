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
    def split_train_val_test(
        cls,
        documents: List[AnnotatedDocument],
        train_ratio: float = 0.50,
        val_ratio: float = 0.25,
        test_ratio: float = 0.25,
        random_seed: int = 42
    ) -> Tuple[List[AnnotatedDocument], List[AnnotatedDocument], List[AnnotatedDocument]]:
        """
        Splits dataset by DOCUMENT ID to strictly eliminate sentence leakage across train, validation, and test splits.
        All sentences from a single document stay exclusively in one split.
        """
        import random
        rng = random.Random(random_seed)
        shuffled = list(documents)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        
        train_docs = shuffled[:n_train]
        val_docs = shuffled[n_train:n_train + n_val]
        test_docs = shuffled[n_train + n_val:]
        if not test_docs:
            test_docs = val_docs[-1:]
            val_docs = val_docs[:-1]

        logger.info(
            f"Strict Document Split: {len(train_docs)} Train, {len(val_docs)} Validation, {len(test_docs)} Test "
            f"(Zero document or sentence leakage across partitions)"
        )
        return train_docs, val_docs, test_docs

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

    @classmethod
    def extract_document_spans(cls, documents: List[AnnotatedDocument]) -> List[Dict[str, Any]]:
        """Extracts ground-truth span annotations per document for span localization evaluation."""
        doc_spans = []
        for doc in documents:
            gold_spans = []
            char_offset = 0
            # Construct approximate document text and spans
            doc_text = " ".join(s.text for s in doc.sentences)
            for s in doc.sentences:
                start_in_doc = doc_text.find(s.text, char_offset)
                if start_in_doc == -1:
                    start_in_doc = char_offset
                char_offset = start_in_doc + len(s.text)

                for span in s.token_spans:
                    gold_spans.append({
                        "text": span.get("text", ""),
                        "start_char": start_in_doc + span.get("start_char", 0),
                        "end_char": start_in_doc + span.get("end_char", len(span.get("text", ""))),
                        "category": s.category
                    })

            doc_spans.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "domain": doc.domain,
                "doc_text": doc_text,
                "gold_spans": gold_spans
            })
        return doc_spans

