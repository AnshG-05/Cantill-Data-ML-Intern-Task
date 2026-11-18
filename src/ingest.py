"""Read raw CSV files and build canonical input records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .config import Config

MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "<na>"}


def clean_string(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return None
    return text


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_side_files(cfg: Config) -> Dict[str, Optional[pd.DataFrame]]:
    result: Dict[str, Optional[pd.DataFrame]] = {}
    taxonomy_file = cfg.data_path("classification.taxonomy.file")
    if taxonomy_file is None or not taxonomy_file.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_file}")
    result["taxonomy"] = read_csv(taxonomy_file)

    labels_file = cfg.data_path("classification.labels.file")
    result["labels"] = read_csv(labels_file) if labels_file and labels_file.exists() else None

    fx_file = cfg.data_path("normalization.fx.rate_file")
    if fx_file is None:
        fx_file = cfg.data_path("paths.fx_rates")
    result["fx"] = read_csv(fx_file) if fx_file and fx_file.exists() else None
    return result


def ingest_records(cfg: Config, side_files: Dict[str, Optional[pd.DataFrame]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    input_order = 0
    for source in cfg.get("inputs", []):
        path = cfg.data_path("inputs.file", source["file"])
        if path is None or not path.exists():
            raise FileNotFoundError(f"Input file not found: {source.get('file')}")
        frame = read_csv(path)
        fields = source.get("fields", {}) or {}
        id_cfg = source.get("record_id", {}) or {}
        id_column = id_cfg["column"]
        prefix = id_cfg.get("prefix")
        if id_column not in frame.columns:
            raise ValueError(f"Record id column '{id_column}' missing from {path.name}")

        for _, row in frame.iterrows():
            raw = {column: clean_string(row[column]) for column in frame.columns}
            raw_id = clean_string(row[id_column]) or ""
            record_id = f"{prefix}:{raw_id}" if prefix else raw_id
            canonical: Dict[str, Any] = {
                "record_id": record_id,
                "source": source.get("name", path.stem),
                "source_file": path.name,
                "raw_id": raw_id,
                "raw": raw,
                "input_order": input_order,
                "include": True,
                "exclusion_reason": None,
                "eligibility_confidence": 0.95,
            }
            for canonical_name, column_name in fields.items():
                canonical[canonical_name] = clean_string(row[column_name]) if column_name else None
            canonical["account_description"] = canonical.get("account")
            records.append(canonical)
            input_order += 1
    return records


def ingest(cfg: Config) -> tuple[List[Dict[str, Any]], Dict[str, Optional[pd.DataFrame]]]:
    side_files = load_side_files(cfg)
    records = ingest_records(cfg, side_files)
    return records, side_files
