#!/usr/bin/env python
"""Command line entry point for the Cantill spend pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.classify import classify
from src.config import load_config
from src.eligibility import apply_eligibility
from src.entities import resolve
from src.evaluate import evaluate_frame, print_evaluation
from src.ingest import ingest
from src.normalize import normalize_records
from src.reconcile import reconcile
from src.write_output import write_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven spend normalization and classification")
    parser.add_argument("--config", required=True, help="Path to a company YAML config")
    parser.add_argument("--output", default=None, help="Override the configured output CSV path")
    parser.add_argument("--no-labels", action="store_true", help="Ignore label files (zero-label mode)")
    parser.add_argument("--embeddings", dest="embeddings", action="store_true", default=None, help="Enable optional local embeddings")
    parser.add_argument("--no-embeddings", dest="embeddings", action="store_false", help="Disable optional local embeddings")
    parser.add_argument("--smoothing", dest="smoothing", action="store_true", default=None, help="Enable kNN/group zero-label smoothing")
    parser.add_argument("--no-smoothing", dest="smoothing", action="store_false", help="Disable kNN/group zero-label smoothing")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip label-based evaluation")
    parser.add_argument("--no-classify", action="store_true", help="Skip classification (debugging)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    if args.embeddings is not None:
        cfg.data.setdefault("classification", {}).setdefault("zero_shot", {}).setdefault("optional_embeddings", {})["enabled"] = args.embeddings
    if args.smoothing is not None:
        smoothing = cfg.data.setdefault("classification", {}).setdefault("zero_shot", {}).setdefault("smoothing", {})
        for name in ("knn", "group"):
            smoothing.setdefault(name, {})["enabled"] = args.smoothing

    records, side_files = ingest(cfg)
    input_records = len(records)
    eval_labels = side_files.get("labels")
    working_labels = None if args.no_labels else eval_labels

    normalize_stats = normalize_records(records, cfg, side_files)
    eligibility_stats = apply_eligibility(records, cfg)
    reconcile_stats = reconcile(records, cfg)
    entity_stats = resolve(records, cfg, working_labels)
    classify_stats = classify(
        records,
        cfg,
        side_files["taxonomy"],
        working_labels,
        enabled=not args.no_classify,
    )

    output_path, frame, qa = write_output(records, cfg, side_files["taxonomy"], args.output)
    elapsed = time.perf_counter() - started

    print(f"[run] company={cfg.company} input_records={input_records} elapsed={elapsed:.1f}s")
    print(f"[run] normalization={normalize_stats}")
    print(f"[run] eligibility={eligibility_stats}")
    print(f"[run] reconciliation={reconcile_stats}")
    print(f"[run] entities={entity_stats}")
    print(f"[run] classify={classify_stats}")
    print(f"[run] output={output_path} included={qa['included']} excluded={qa['excluded']} total_spend={qa['total_spend']:.2f}")
    print(f"[run] qa_valid={qa['valid']} issues={qa['issues']}")
    if not args.no_evaluate:
        print_evaluation(evaluate_frame(cfg, frame, eval_labels))
    return 0 if qa["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
