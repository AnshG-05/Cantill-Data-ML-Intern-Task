"""Build, validate and write the required CSV output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config

OUTPUT_COLUMNS = [
    "record_id",
    "include",
    "exclusion_reason",
    "supplier_key",
    "item_key",
    "spend",
    "l1",
    "l2",
    "l3",
    "l4",
    "confidence",
]
VALID_REASONS = {"CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"}


def _valid_paths(cfg: Config, taxonomy: pd.DataFrame) -> set[tuple[str, ...]]:
    paths = set()
    for _, row in taxonomy.iterrows():
        values: List[str] = []
        for column in cfg.level_columns:
            value = row.get(column)
            if value is None or pd.isna(value) or not str(value).strip():
                break
            values.append(str(value).strip())
        if values:
            paths.add(tuple(values))
    return paths


def build_frame(records: List[Dict[str, Any]], cfg: Config, taxonomy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in records:
        included = bool(record.get("include", True))
        spend = record.get("amount_reporting") if included else 0.0
        if spend is None or not np.isfinite(float(spend)):
            spend = 0.0
        category = list(record.get("category_path") or []) if included else []
        categories = category + [None] * max(0, 4 - len(category))
        confidence = record.get("classification_confidence") if included else record.get("eligibility_confidence", 0.9)
        if confidence is None or not np.isfinite(float(confidence)):
            confidence = 0.0
        rows.append(
            {
                "record_id": record.get("record_id", ""),
                "include": int(included),
                "exclusion_reason": None if included else record.get("exclusion_reason", "NON_PROCUREMENT"),
                "supplier_key": record.get("supplier_key") or "SUP:unknown-supplier",
                "item_key": record.get("item_key"),
                "spend": round(float(spend), 2),
                "l1": categories[0],
                "l2": categories[1],
                "l3": categories[2],
                "l4": categories[3],
                "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 3),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def validate_frame(frame: pd.DataFrame, cfg: Config, taxonomy: pd.DataFrame, expected_rows: int) -> Dict[str, Any]:
    issues: List[str] = []
    if len(frame) != expected_rows:
        issues.append(f"row count {len(frame)} != expected {expected_rows}")
    if list(frame.columns) != OUTPUT_COLUMNS:
        issues.append("column names or order do not match the required schema")
    if not frame["include"].isin([0, 1]).all():
        issues.append("include contains values other than 0/1")
    if not np.isfinite(frame["spend"].astype(float)).all():
        issues.append("spend contains non-finite values")
    if not frame["confidence"].between(0, 1).all():
        issues.append("confidence outside [0,1]")
    excluded = frame[frame["include"] == 0]
    included = frame[frame["include"] == 1]
    if not excluded["exclusion_reason"].isin(VALID_REASONS).all():
        issues.append("excluded rows contain invalid exclusion_reason")
    if included["exclusion_reason"].notna().any():
        issues.append("included rows contain an exclusion_reason")
    if (excluded["spend"].astype(float) != 0).any():
        issues.append("excluded rows contain non-zero spend")

    valid_paths = _valid_paths(cfg, taxonomy)
    level_columns = [column for column in cfg.level_columns if column in {"l1", "l2", "l3", "l4"}]
    # cfg level columns are named category_l*, so compare by ordinal position.
    invalid = 0
    for row in included.itertuples(index=False):
        path = tuple(value for value in [row.l1, row.l2, row.l3, row.l4] if value)
        # A/B taxonomies have three levels; C may have four. The taxonomy set
        # itself determines which path lengths are valid.
        if path not in valid_paths:
            invalid += 1
    if invalid:
        issues.append(f"{invalid} included rows have a taxonomy path absent from the taxonomy file")

    stats = {
        "rows": len(frame),
        "included": int(len(included)),
        "excluded": int(len(excluded)),
        "total_spend": float(frame["spend"].sum()),
        "issues": issues,
        "valid": not issues,
    }
    return stats


def write_output(
    records: List[Dict[str, Any]],
    cfg: Config,
    taxonomy: pd.DataFrame,
    output_override: Optional[str] = None,
) -> tuple[Path, pd.DataFrame, Dict[str, Any]]:
    frame = build_frame(records, cfg, taxonomy)
    stats = validate_frame(frame, cfg, taxonomy, len(records))
    output_path = Path(output_override) if output_override else cfg.resolve(cfg.get("paths.output"))
    if output_path is None:
        raise ValueError("No output path configured")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    stats["output"] = str(output_path)
    return output_path, frame, stats
