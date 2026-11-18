"""Status, non-procurement and duplicate exclusion rules."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from .config import Config


def field_value(record: Dict[str, Any], field: str) -> Any:
    if field in record:
        return record.get(field)
    raw = record.get("raw", {}) or {}
    return raw.get(field)


def normalized_value(value: Any) -> str:
    if value is None:
        return "<none>"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).strip().lower()


def apply_eligibility(records: List[Dict[str, Any]], cfg: Config) -> Dict[str, int]:
    stats = defaultdict(int)
    status_cfg = cfg.get("eligibility.status", {}) or {}
    nonproc_rules = cfg.get("eligibility.non_procurement", []) or []
    duplicate_rules = cfg.get("eligibility.duplicates", []) or []

    if status_cfg.get("enabled"):
        status_field = status_cfg.get("field", "status")
        allowed = {str(value).strip().lower() for value in status_cfg.get("allowed", [])}
        for record in records:
            status = field_value(record, status_field)
            if not status:
                continue
            status_text = str(status).strip()
            status_lower = status_text.lower()
            reason = None
            for rule in status_cfg.get("exclude", []) or []:
                values = {str(value).lower() for value in rule.get("values", [])}
                if status_lower in values:
                    reason = rule.get("reason", "NOT_APPROVED")
                    break
            if reason is None and allowed and status_lower not in allowed:
                reason = "NOT_APPROVED"
            if reason:
                record["include"] = False
                record["exclusion_reason"] = reason
                record["eligibility_confidence"] = 0.99
                stats[reason] += 1
            else:
                record["eligibility_confidence"] = 0.95

    for rule in nonproc_rules:
        field = rule.get("field")
        pattern = rule.get("regex")
        if not field or not pattern:
            continue
        flags = 0
        if "ignorecase" in str(rule.get("flags", "ignorecase")).lower():
            flags |= re.IGNORECASE
        compiled = re.compile(pattern, flags)
        reason = rule.get("reason", "NON_PROCUREMENT")
        for record in records:
            if not record.get("include", True):
                continue
            value = field_value(record, field)
            if value is None:
                continue
            if compiled.search(str(value)):
                record["include"] = False
                record["exclusion_reason"] = reason
                record["eligibility_confidence"] = 0.97
                stats[reason] += 1

    for rule in duplicate_rules:
        fields = rule.get("fields", []) or []
        sources = set(rule.get("sources", []) or [])
        reason = rule.get("reason", "DUPLICATE")
        if not fields:
            continue
        groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            if not record.get("include", True):
                continue
            if sources and record.get("source") not in sources:
                continue
            key = tuple(normalized_value(field_value(record, field)) for field in fields)
            groups[key].append(record)
        for group in groups.values():
            if len(group) < 2:
                continue
            for duplicate in group[1:]:
                duplicate["include"] = False
                duplicate["exclusion_reason"] = reason
                duplicate["eligibility_confidence"] = 0.9
                stats[reason] += 1

    return dict(stats)
