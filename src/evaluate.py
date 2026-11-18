"""Evaluation helpers for labeled datasets and output QA."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score

from .config import Config


def _aligned_accuracy(truth: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    truth = truth.astype(str).reset_index(drop=True)
    predicted = predicted.astype(str).reset_index(drop=True)
    mapping = (
        pd.DataFrame({"truth": truth, "pred": predicted})
        .groupby("truth")["pred"]
        .agg(lambda values: values.value_counts().idxmax())
        .to_dict()
    )
    aligned = truth.map(mapping)
    return {
        "aligned_accuracy": float((aligned == predicted).mean()) if len(predicted) else 0.0,
        "ari": float(adjusted_rand_score(truth, predicted)) if truth.nunique() > 1 and len(truth) else 0.0,
        "homogeneity": float(homogeneity_score(truth, predicted)) if truth.nunique() > 1 and len(truth) else 0.0,
        "completeness": float(completeness_score(truth, predicted)) if truth.nunique() > 1 and len(truth) else 0.0,
    }


def evaluate_frame(
    cfg: Config,
    frame: pd.DataFrame,
    labels: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if labels is None or labels.empty:
        return result

    id_column = cfg.get("classification.labels.record_id_column", "record_id")
    if id_column not in labels.columns:
        return result
    indexed = frame.set_index("record_id")
    label_ids = labels[id_column].astype(str).str.strip()
    mask = label_ids.isin(indexed.index)
    label_subset = labels[mask].copy()
    label_subset[id_column] = label_ids[mask]
    if label_subset.empty:
        return result

    aligned = indexed.loc[label_subset[id_column]].reset_index()
    result["labeled_rows"] = len(aligned)

    include_column = cfg.get("classification.labels.include_column")
    if include_column and include_column in label_subset.columns:
        truth_include = label_subset[include_column].astype(str).str.strip().isin(["1", "True", "true"]).astype(int)
        pred_include = aligned["include"].astype(int)
        result["include_accuracy"] = float((truth_include.values == pred_include.values).mean())
        exclusion_reason_col = "exclusion_reason"
        if exclusion_reason_col in label_subset.columns:
            truth_reason = label_subset[exclusion_reason_col].fillna("").astype(str).str.strip()
            pred_reason = aligned[exclusion_reason_col].fillna("").astype(str).str.strip()
            excluded_truth = truth_include.values == 0
            if excluded_truth.any():
                result["exclusion_reason_accuracy"] = float(
                    (truth_reason.values[excluded_truth] == pred_reason.values[excluded_truth]).mean()
                )

    category_columns = list(cfg.get("classification.labels.category_columns", []) or [])
    if category_columns:
        both_included = np.ones(len(aligned), dtype=bool)
        if include_column and include_column in label_subset.columns:
            both_included &= (
                label_subset[include_column].astype(str).str.strip().isin(["1", "True", "true"]).values
                & aligned["include"].values.astype(bool)
            )
        if both_included.any():
            label_categories = label_subset.loc[both_included, category_columns].reset_index(drop=True)
            pred_categories = aligned.loc[both_included, ["l1", "l2", "l3", "l4"]].reset_index(drop=True)
            for level, column in enumerate(category_columns):
                output_column = f"l{level + 1}"
                if output_column in pred_categories.columns:
                    result[f"{output_column}_accuracy"] = float(
                        (
                            label_categories[column].fillna("").astype(str).str.strip().values
                            == pred_categories[output_column].fillna("").astype(str).str.strip().values
                        ).mean()
                    )

    supplier_col = cfg.get("classification.labels.supplier_key_column")
    if supplier_col and supplier_col in label_subset.columns:
        result["supplier_grouping"] = _aligned_accuracy(
            label_subset[supplier_col].fillna("<none>"), aligned["supplier_key"].fillna("<none>")
        )
    item_col = cfg.get("classification.labels.item_key_column")
    if item_col and item_col in label_subset.columns:
        result["item_grouping"] = _aligned_accuracy(
            label_subset[item_col].fillna("<none>"), aligned["item_key"].fillna("<none>")
        )
    return result


def print_evaluation(metrics: Dict[str, Any]) -> None:
    if not metrics:
        print("[evaluate] no labels available")
        return
    print("[evaluate] " + " ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items() if not isinstance(value, dict)))
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"[evaluate] {key}: " + " ".join(f"{name}={score:.4f}" for name, score in value.items()))
