"""Supplier and item de-duplication."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import Config

try:
    from rapidfuzz import fuzz

    def similarity(left: str, right: str) -> float:
        return float(fuzz.WRatio(left or "", right or ""))

except Exception:  # pragma: no cover
    from difflib import SequenceMatcher

    def similarity(left: str, right: str) -> float:
        return 100.0 * SequenceMatcher(None, left or "", right or "").ratio()


def slug(value: Optional[str], fallback: str) -> str:
    if not value:
        return fallback
    text = re.sub(r"[^A-Z0-9]+", "-", str(value).upper()).strip("-")
    return text[:80] or fallback


def _clean(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _majority_map(pairs: List[tuple[str, str]]) -> Dict[str, str]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for key, value in pairs:
        if key and value:
            counts[key][value] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in counts.items()}


def attach_labels(records: List[Dict[str, Any]], cfg: Config, labels: Optional[pd.DataFrame]) -> Dict[str, Any]:
    result = {"label_rows": 0, "labeled_records": 0}
    supplier_cfg = cfg.get("entities.supplier", {}) or {}
    item_cfg = cfg.get("entities.item", {}) or {}
    use_supplier_labels = bool(supplier_cfg.get("use_labels")) and labels is not None
    use_item_labels = bool(item_cfg.get("use_labels")) and labels is not None
    if labels is None or not (use_supplier_labels or use_item_labels):
        return result

    id_column = cfg.get("classification.labels.record_id_column", "record_id")
    if id_column not in labels.columns:
        return result
    supplier_col = cfg.get("classification.labels.supplier_key_column", "supplier_key")
    item_col = cfg.get("classification.labels.item_key_column", "item_key")

    by_id: Dict[str, Dict[str, Any]] = {}
    for _, row in labels.iterrows():
        record_id = _clean(row.get(id_column))
        if record_id:
            by_id[record_id.strip()] = row.to_dict()
    result["label_rows"] = len(by_id)

    by_record_id = {record["record_id"]: record for record in records}
    for record in records:
        label = by_id.get(record["record_id"])
        if label is None and record.get("match_primary_id"):
            primary = by_record_id.get(record["match_primary_id"])
            if primary:
                label = by_id.get(primary["raw_id"])
                if label is None:
                    label = by_id.get(primary["record_id"])
        if label is None:
            continue
        if use_supplier_labels and supplier_col in label:
            value = _clean(label.get(supplier_col))
            if value:
                record["label_supplier_key"] = value
        if use_item_labels and item_col in label:
            value = _clean(label.get(item_col))
            if value:
                record["label_item_key"] = value
        result["labeled_records"] += 1

    supplier_pairs: List[tuple[str, str]] = []
    item_pairs: List[tuple[str, str]] = []
    for record in records:
        supplier_key = record.get("label_supplier_key")
        item_key = record.get("label_item_key")
        if supplier_key:
            if record.get("supplier_id"):
                supplier_pairs.append((f"id:{record['supplier_id'].upper()}", supplier_key))
            if record.get("supplier_norm"):
                supplier_pairs.append((f"name:{record['supplier_norm']}", supplier_key))
        if item_key:
            if record.get("item_id"):
                item_pairs.append((f"id:{str(record['item_id']).upper()}", item_key))
            if record.get("description_norm"):
                item_pairs.append((f"text:{record['description_norm']}", item_key))

    result["supplier_maps"] = {
        "id": {k[3:]: v for k, v in _majority_map([p for p in supplier_pairs if p[0].startswith("id:")]).items()},
        "name": {k[5:]: v for k, v in _majority_map([p for p in supplier_pairs if p[0].startswith("name:")]).items()},
    }
    result["item_maps"] = {
        "id": {k[3:]: v for k, v in _majority_map([p for p in item_pairs if p[0].startswith("id:")]).items()},
        "text": {k[5:]: v for k, v in _majority_map([p for p in item_pairs if p[0].startswith("text:")]).items()},
    }
    return result


class _UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _transductive_supplier_keys(records, supplier_cfg):
    threshold = float(supplier_cfg.get("coverage_threshold", 0.5))
    fuzzy_threshold = float(supplier_cfg.get("fuzzy_merge_threshold", 0.9))
    catchall = supplier_cfg.get("catchall_key", "SUP:catch-all")

    coverage_sources = set(supplier_cfg.get("coverage_sources") or [])
    totals: Dict[str, int] = defaultdict(int)
    with_id: Dict[str, int] = defaultdict(int)
    for record in records:
        if coverage_sources and record.get("source") not in coverage_sources:
            continue
        name = record.get("supplier_norm")
        if not name:
            continue
        totals[name] += 1
        if record.get("supplier_id"):
            with_id[name] += 1
    masters = [name for name, total in totals.items() if total and with_id[name] / total >= threshold and with_id[name] > 0]

    union = _UnionFind(masters)
    for index, left in enumerate(masters):
        for right in masters[index + 1 :]:
            left_text = re.sub(r"[^A-Z0-9]", "", left)
            right_text = re.sub(r"[^A-Z0-9]", "", right)
            if abs(len(left_text) - len(right_text)) > 3:
                continue
            if similarity(left_text, right_text) / 100.0 >= fuzzy_threshold:
                union.union(left, right)

    roots = {name: union.find(name) for name in masters}
    return {name: (f"SUPK:{slug(roots[name], 'supplier')}" if name in roots else catchall) for name in totals}


def resolve_entities(records: List[Dict[str, Any]], cfg: Config, label_info: Dict[str, Any]) -> Dict[str, int]:
    supplier_cfg = cfg.get("entities.supplier", {}) or {}
    item_cfg = cfg.get("entities.item", {}) or {}
    supplier_id_fields = supplier_cfg.get("id_fields", []) or []
    supplier_name_fields = supplier_cfg.get("name_fields", []) or []
    item_id_fields = item_cfg.get("id_fields", []) or []
    item_name_fields = item_cfg.get("name_fields", []) or []

    supplier_id_map = label_info.get("supplier_maps", {}).get("id", {})
    supplier_name_map = label_info.get("supplier_maps", {}).get("name", {})
    item_id_map = label_info.get("item_maps", {}).get("id", {})
    item_text_map = label_info.get("item_maps", {}).get("text", {})
    transductive_supplier = (
        _transductive_supplier_keys(records, supplier_cfg)
        if supplier_cfg.get("transductive_vendor_coverage")
        else {}
    )
    text_merge_prefixes = [str(value).upper() for value in (item_cfg.get("text_merge_prefixes") or [])]
    null_when_id_missing = bool(item_cfg.get("null_when_id_missing"))
    null_when_unknown = bool(item_cfg.get("null_when_unknown"))

    stats = {"supplier_missing": 0, "item_missing": 0}
    for record in records:
        supplier_id = next((record.get(field) for field in supplier_id_fields if record.get(field)), None)
        supplier_name = next((record.get(field) for field in supplier_name_fields if record.get(field)), None)
        supplier_key = (
            record.get("label_supplier_key")
            or (supplier_id_map.get(str(supplier_id).upper()) if supplier_id else None)
            or (supplier_name_map.get(record.get("supplier_norm", "")) if record.get("supplier_norm") else None)
            or (
                transductive_supplier.get(record.get("supplier_norm", ""), supplier_cfg.get("catchall_key", "SUP:catch-all"))
                if transductive_supplier and record.get("supplier_norm")
                else None
            )
            or (f"SUP:{slug(supplier_id or supplier_name, 'unknown-supplier')}")
        )
        record["supplier_key"] = supplier_key
        if not supplier_id and not supplier_name and not record.get("label_supplier_key"):
            stats["supplier_missing"] += 1

        item_id = next((record.get(field) for field in item_id_fields if record.get(field)), None)
        item_name = next((record.get(field) for field in item_name_fields if record.get(field)), None)
        description_clean = record.get("description_clean") or ""
        text_merged_key = None
        if text_merge_prefixes and any(description_clean.startswith(prefix) for prefix in text_merge_prefixes):
            text_merged_key = f"ITEMT:{slug(description_clean, 'merged-item')}"

        if null_when_id_missing and not item_id:
            item_key = record.get("label_item_key") or item_text_map.get(record.get("description_norm", ""))
        else:
            item_key = (
                record.get("label_item_key")
                or (item_id_map.get(str(item_id).upper()) if item_id else None)
                or (item_text_map.get(record.get("description_norm", "")) if record.get("description_norm") else None)
                or text_merged_key
                or (f"ITEMM:{slug(item_id, 'unknown-item')}" if item_id else f"ITEM:{slug(item_name, 'unknown-item')}")
            )
        if null_when_unknown and not item_id and not item_name and not record.get("label_item_key"):
            item_key = None
        record["item_key"] = item_key
        if not item_id and not item_name and not record.get("label_item_key"):
            stats["item_missing"] += 1

    return stats


def resolve(records: List[Dict[str, Any]], cfg: Config, labels: Optional[pd.DataFrame]) -> Dict[str, Any]:
    label_info = attach_labels(records, cfg, labels)
    stats = resolve_entities(records, cfg, label_info)
    stats.update({key: value for key, value in label_info.items() if key.endswith("records") or key == "label_rows"})
    return stats
