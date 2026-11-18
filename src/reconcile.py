"""Cross-system reconciliation for company A."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .config import Config

try:
    from rapidfuzz import fuzz

    def similarity(left: str, right: str) -> float:
        return float(fuzz.WRatio(left or "", right or ""))

except Exception:  # pragma: no cover - fallback for minimal environments
    from difflib import SequenceMatcher

    def similarity(left: str, right: str) -> float:
        return 100.0 * SequenceMatcher(None, left or "", right or "").ratio()


def date_score(left, right) -> float:
    if left is None or right is None:
        return 0.5
    try:
        days = abs((left - right).days)
    except Exception:
        return 0.5
    if days <= 7:
        return 1.0
    if days <= 60:
        return 1.0 - (days - 7) / 53.0
    return 0.0


def reconcile(records: List[Dict[str, Any]], cfg: Config) -> Dict[str, int]:
    stats = {"candidates": 0, "matched": 0, "rejected": 0}
    if not cfg.get("reconciliation.enabled"):
        return stats

    primary_name = cfg.get("reconciliation.primary")
    secondary_name = cfg.get("reconciliation.secondary")
    tolerance = float(cfg.get("reconciliation.amount_tolerance", 0.005))
    min_score = float(cfg.get("reconciliation.min_score", 65))
    reason = cfg.get("reconciliation.reason", "DUPLICATE")
    weights = cfg.get("reconciliation.weights", {}) or {}
    weight_sum = float(sum(weights.values())) or 1.0

    primary = [
        record
        for record in records
        if record.get("source") == primary_name
        and record.get("include", True)
        and record.get("po")
        and record.get("amount_reporting") is not None
    ]
    secondary = [
        record
        for record in records
        if record.get("source") == secondary_name
        and record.get("include", True)
        and record.get("po")
        and record.get("amount_reporting") is not None
    ]

    by_po: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in primary:
        by_po[str(record["po"])].append(record)

    candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for tool in secondary:
        amount = abs(float(tool["amount_reporting"]))
        for erp in by_po.get(str(tool["po"]), []):
            erp_amount = abs(float(erp["amount_reporting"]))
            relative = abs(amount - erp_amount) / max(amount, erp_amount, 1.0)
            if relative > tolerance:
                continue
            stats["candidates"] += 1
            amount_score = 1.0 - relative / tolerance if tolerance > 0 else 1.0

            if tool.get("po_item") and erp.get("po_item"):
                item_score = 1.0 if str(tool["po_item"]) == str(erp["po_item"]) else 0.0
            else:
                item_score = 0.5

            supplier_score = similarity(tool.get("supplier_norm", ""), erp.get("supplier_norm", ""))
            if not tool.get("supplier_norm") or not erp.get("supplier_norm"):
                supplier_score = 50.0
            supplier_score /= 100.0

            text_score = similarity(tool.get("description_norm", ""), erp.get("description_norm", ""))
            if not tool.get("description_norm") or not erp.get("description_norm"):
                text_score = 50.0
            text_score /= 100.0

            day_score = date_score(tool.get("date"), erp.get("date"))
            score = (
                weights.get("amount", 0) * amount_score
                + weights.get("po_item", 0) * item_score
                + weights.get("supplier", 0) * supplier_score
                + weights.get("text", 0) * text_score
                + weights.get("date", 0) * day_score
            ) / weight_sum * 100.0
            if score >= min_score:
                candidates.append((score, tool, erp))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_secondary: set[str] = set()
    used_primary: set[str] = set()
    for score, tool, erp in candidates:
        if tool["record_id"] in used_secondary or erp["record_id"] in used_primary:
            continue
        tool["include"] = False
        tool["exclusion_reason"] = reason
        tool["eligibility_confidence"] = min(0.99, score / 100.0)
        tool["match_primary_id"] = erp["record_id"]
        tool["match_score"] = score
        used_secondary.add(tool["record_id"])
        used_primary.add(erp["record_id"])
        stats["matched"] += 1

    stats["rejected"] = stats["candidates"] - stats["matched"]
    return stats
