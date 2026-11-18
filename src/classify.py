"""Taxonomy-aware hybrid classification with zero-label support."""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import Config

TOKEN_RE = re.compile(r"[A-Z0-9]+")


def _clean(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def load_taxonomy(cfg: Config, taxonomy: pd.DataFrame) -> List[Tuple[str, ...]]:
    level_columns = cfg.level_columns
    paths: List[Tuple[str, ...]] = []
    seen = set()
    for _, row in taxonomy.iterrows():
        values = [_clean(row.get(column)) for column in level_columns]
        while values and values[-1] is None:
            values.pop()
        if not values:
            continue
        path = tuple(values)
        if path not in seen:
            seen.add(path)
            paths.append(path)
    if not paths:
        raise ValueError("Taxonomy contained no valid paths")
    return paths


def build_text(record: Dict[str, Any], fields: Sequence[str]) -> str:
    parts: List[str] = []
    for field in fields:
        value = record.get(field)
        if value:
            parts.append(str(value))
    return " | ".join(dict.fromkeys(parts))


def _tfidf_similarity(texts: List[str], documents: List[str]) -> np.ndarray:
    if not texts or not documents:
        return np.zeros((len(texts), len(documents)), dtype=np.float32)
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=300_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=120_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    combined = texts + documents
    x_char = char.fit_transform(combined)
    x_word = word.fit_transform(combined)
    matrix = normalize(hstack([x_char, x_word], format="csr"))
    query = matrix[: len(texts)]
    doc = matrix[len(texts) :]
    return np.asarray((query @ doc.T).todense(), dtype=np.float32)


def _format_document(path: Tuple[str, ...], level_columns: Sequence[str], template: str) -> str:
    values: Dict[str, str] = {"levels": " > ".join(path)}
    for index, value in enumerate(path):
        values[f"l{index + 1}"] = value
    for column in level_columns:
        values[column] = ""
    try:
        return template.format(**values).strip()
    except (KeyError, IndexError):
        return " > ".join(path)


def _embedding_similarity(
    texts: List[str],
    paths: List[Tuple[str, ...]],
    cfg: Config,
) -> Optional[np.ndarray]:
    emb_cfg = cfg.get("classification.zero_shot.optional_embeddings", {}) or {}
    if not emb_cfg.get("enabled"):
        return None
    model_name = emb_cfg.get("model")
    if not model_name:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu")
    except Exception as exc:  # pragma: no cover - optional dependency/model
        print(f"[classify] embeddings unavailable ({exc}); using TF-IDF only", file=sys.stderr)
        return None

    query_prefix = str(emb_cfg.get("query_prefix", ""))
    doc_prefix = str(emb_cfg.get("document_prefix", ""))
    template = str(emb_cfg.get("document_template", "{levels}"))
    level_columns = cfg.level_columns
    documents = [
        doc_prefix + _format_document(path, level_columns, template) for path in paths
    ]
    # Many spend lines repeat the same description; encode each unique text once.
    unique_texts = list(dict.fromkeys(texts))
    text_to_index = {text: index for index, text in enumerate(unique_texts)}
    text_indices = np.array([text_to_index[text] for text in texts], dtype=np.int32)
    try:
        doc_vectors = model.encode(
            documents,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        unique_vectors = model.encode(
            [query_prefix + text for text in unique_texts],
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vectors = unique_vectors[text_indices]
        return np.asarray(query_vectors @ doc_vectors.T, dtype=np.float32)
    except Exception as exc:  # pragma: no cover
        print(f"[classify] embedding encoding failed ({exc}); using TF-IDF only", file=sys.stderr)
        return None


def _node_token_sets(paths: List[Tuple[str, ...]]) -> List[List[frozenset[str]]]:
    result: List[List[frozenset[str]]] = []
    for path in paths:
        nodes = []
        for value in path:
            tokens = frozenset(token for token in TOKEN_RE.findall(value.upper()) if len(token) > 2)
            nodes.append(tokens)
        result.append(nodes)
    return result


def _mention_scores(texts: List[str], paths: List[Tuple[str, ...]], weight_cfg: Dict[str, Any]) -> np.ndarray:
    if not weight_cfg.get("enabled"):
        return np.zeros((len(texts), len(paths)), dtype=np.float32)
    node_sets = _node_token_sets(paths)
    path_weights = np.array([sum(len(node) for node in nodes) or 1 for nodes in node_sets], dtype=np.float32)
    scores = np.zeros((len(texts), len(paths)), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = frozenset(TOKEN_RE.findall(text.upper()))
        if not tokens:
            continue
        for column, nodes in enumerate(node_sets):
            matched = 0
            for node in nodes:
                if node and node.issubset(tokens):
                    matched += len(node)
            scores[row, column] = matched / path_weights[column]
    maximum = scores.max(axis=1, keepdims=True)
    return scores / maximum if maximum.size and maximum[0, 0] > 0 else scores


def _knn_smooth(scores: np.ndarray, texts: List[str], k: int, alpha: float) -> np.ndarray:
    """Smooth taxonomy scores with the mean score of similar unlabeled lines."""
    if alpha <= 0 or k <= 0 or len(texts) < 2:
        return scores
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=300_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = normalize(vectorizer.fit_transform(texts))
    output = np.zeros_like(scores)
    neighbors = min(k, len(texts) - 1)
    for start in range(0, len(texts), 256):
        similarity = (matrix[start : start + 256] @ matrix.T).toarray()
        for row in range(similarity.shape[0]):
            similarity[row, start + row] = -1e9
        indices = np.argpartition(-similarity, kth=neighbors - 1, axis=1)[:, :neighbors]
        output[start : start + 256] = (
            (1.0 - alpha) * scores[start : start + 256] + alpha * scores[indices].mean(axis=1)
        )
    return output


def _group_smooth(
    scores: np.ndarray,
    records: List[Dict[str, Any]],
    group_fields: Sequence[str],
    beta: float,
) -> np.ndarray:
    """Smooth scores within repeated material/description groups."""
    if beta <= 0:
        return scores
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = None
        for field in group_fields:
            value = record.get(field) or (record.get("raw", {}) or {}).get(field)
            if value:
                key = f"{field}:{str(value).strip().upper()}"
                break
        if key:
            groups[key].append(index)
    output = scores.copy()
    for indices in groups.values():
        if len(indices) < 2:
            continue
        mean = scores[indices].mean(axis=0)
        for index in indices:
            output[index] = (1.0 - beta) * scores[index] + beta * mean
    return output


def _label_scores(
    texts: List[str],
    records: List[Dict[str, Any]],
    paths: List[Tuple[str, ...]],
    cfg: Config,
    labels: Optional[pd.DataFrame],
) -> Optional[np.ndarray]:
    retrieval_cfg = cfg.get("classification.label_retrieval", {}) or {}
    if not retrieval_cfg.get("enabled") or labels is None:
        return None
    id_column = cfg.get("classification.labels.record_id_column", "record_id")
    category_columns = list(cfg.get("classification.labels.category_columns", []) or [])
    include_column = cfg.get("classification.labels.include_column")
    if id_column not in labels.columns or not category_columns:
        return None

    by_id: Dict[str, Dict[str, Any]] = {}
    for _, row in labels.iterrows():
        label_id = _clean(row.get(id_column))
        if not label_id:
            continue
        if include_column and str(row.get(include_column)).strip() in {"0", "False", "false", "None", ""}:
            continue
        path = tuple(value for value in (_clean(row.get(column)) for column in category_columns) if value)
        if path:
            by_id[label_id.strip()] = {"path": path}

    labeled_indices: List[int] = []
    label_texts: List[str] = []
    label_paths: List[Tuple[str, ...]] = []
    for index, record in enumerate(records):
        label = by_id.get(record["record_id"]) or by_id.get(record.get("raw_id", ""))
        if label is None:
            continue
        labeled_indices.append(index)
        label_texts.append(texts[index])
        label_paths.append(label["path"])

    if len(labeled_indices) < 2:
        return None

    path_to_index = {path: index for index, path in enumerate(paths)}
    label_path_indices = np.array(
        [path_to_index.get(path, -1) for path in label_paths], dtype=np.int32
    )

    exact_fields = [str(field) for field in (retrieval_cfg.get("exact_fields") or [])]
    exact_maps: Dict[str, Dict[str, Counter]] = {field: defaultdict(Counter) for field in exact_fields}
    for index, record in zip(labeled_indices, label_paths):
        path_index = path_to_index.get(record)
        if path_index < 0:
            continue
        canonical = records[index]
        for field in exact_fields:
            value = canonical.get(field) or (canonical.get("raw", {}) or {}).get(field)
            if value:
                key = re.sub(r"\s+", " ", str(value).strip().upper())
                exact_maps[field][key][path_index] += 1

    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=300_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=120_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    label_char = char.fit_transform(label_texts)
    label_word = word.fit_transform(label_texts)
    label_matrix = normalize(hstack([label_char, label_word], format="csr"))
    query_matrix = normalize(hstack([char.transform(texts), word.transform(texts)], format="csr"))

    top_k = min(5, len(label_texts))
    scores = np.zeros((len(texts), len(paths)), dtype=np.float32)
    batch_size = 256
    for start in range(0, len(texts), batch_size):
        batch = query_matrix[start : start + batch_size] @ label_matrix.T
        dense = batch.toarray()
        top = np.argpartition(-dense, kth=top_k - 1, axis=1)[:, :top_k]
        values = np.take_along_axis(dense, top, axis=1)
        for row in range(dense.shape[0]):
            for column, similarity in zip(top[row], values[row]):
                path_index = label_path_indices[column]
                if path_index >= 0:
                    scores[start + row, path_index] += float(similarity) + 0.01

    for row, record in enumerate(records):
        for field in exact_fields:
            value = record.get(field) or (record.get("raw", {}) or {}).get(field)
            if not value:
                continue
            key = re.sub(r"\s+", " ", str(value).strip().upper())
            candidates = exact_maps[field].get(key)
            if candidates:
                scores[row, candidates.most_common(1)[0][0]] = 1.0
    maximum = scores.max(axis=1, keepdims=True)
    return scores / maximum if maximum.size and maximum[0, 0] > 0 else scores


def classify(
    records: List[Dict[str, Any]],
    cfg: Config,
    taxonomy: pd.DataFrame,
    labels: Optional[pd.DataFrame],
    enabled: bool = True,
) -> Dict[str, Any]:
    stats = {"classified": 0, "taxonomy_paths": 0, "embeddings": False, "retrieval": False}
    level_columns = cfg.level_columns
    paths = load_taxonomy(cfg, taxonomy)
    stats["taxonomy_paths"] = len(paths)

    target_indices = [index for index, record in enumerate(records) if record.get("include", True)]
    if enabled and target_indices:
        target_records = [records[index] for index in target_indices]
        text_fields = list(cfg.get("classification.text_fields", []) or [])
        texts = [build_text(record, text_fields) for record in target_records]

        full_docs = [" > ".join(path) for path in paths]
        leaf_docs = [path[-1] for path in paths]
        parent_docs = [" > ".join(path[:-1]) if len(path) > 1 else path[-1] for path in paths]

        full_sim = _tfidf_similarity(texts, full_docs)
        leaf_sim = _tfidf_similarity(texts, leaf_docs)
        parent_sim = _tfidf_similarity(texts, parent_docs)
        zero_scores = 0.55 * full_sim + 0.30 * leaf_sim + 0.15 * parent_sim

        emb_sim = _embedding_similarity(texts, paths, cfg)
        emb_weight = 0.0
        if emb_sim is not None:
            emb_weight = float(cfg.get("classification.zero_shot.optional_embeddings.weight", 0.0))
            stats["embeddings"] = True

        base_scores = float(cfg.get("classification.zero_shot.weight", 1.0)) * zero_scores
        if emb_sim is not None:
            base_scores = base_scores + emb_weight * emb_sim

        # Smooth only the label-free score. Label retrieval is deliberately
        # excluded so supervised evidence is not diluted by transductive votes.
        smoothing_cfg = cfg.get("classification.zero_shot.smoothing", {}) or {}
        knn_cfg = smoothing_cfg.get("knn", {}) or {}
        if knn_cfg.get("enabled"):
            base_scores = _knn_smooth(
                base_scores,
                texts,
                int(knn_cfg.get("k", 15)),
                float(knn_cfg.get("alpha", 0.35)),
            )
        group_cfg = smoothing_cfg.get("group", {}) or {}
        if group_cfg.get("enabled"):
            base_scores = _group_smooth(
                base_scores,
                target_records,
                list(group_cfg.get("fields", []) or []),
                float(group_cfg.get("beta", 0.35)),
            )

        mention_weight = float(cfg.get("classification.taxonomy_mention.weight", 0.0))
        mention = _mention_scores(texts, paths, cfg.get("classification.taxonomy_mention", {}) or {})

        retrieval = _label_scores(texts, target_records, paths, cfg, labels)
        retrieval_weight = 0.0
        if retrieval is not None:
            retrieval_weight = float(cfg.get("classification.label_retrieval.weight", 0.0))
            stats["retrieval"] = True

        total = base_scores + mention_weight * mention
        if retrieval is not None:
            total = total + retrieval_weight * retrieval

        order = np.argsort(-total, axis=1)
        for row, record in enumerate(target_records):
            best = int(order[row, 0])
            second = int(order[row, 1]) if total.shape[1] > 1 else best
            path = paths[best]
            record["category_path"] = list(path)
            top_score = float(total[row, best])
            second_score = float(total[row, second])
            margin = max(0.0, top_score - second_score)
            base = float(cfg.get("classification.confidence.base", 0.45))
            scale = float(cfg.get("classification.confidence.margin_scale", 2.5))
            confidence = base + 0.35 * min(1.0, top_score) + 0.25 * (1.0 - np.exp(-scale * margin))
            record["classification_confidence"] = float(np.clip(confidence, 0.0, 1.0))
            stats["classified"] += 1
    else:
        for record in records:
            if record.get("include", True):
                record["category_path"] = []
                record["classification_confidence"] = 0.0

    return stats
