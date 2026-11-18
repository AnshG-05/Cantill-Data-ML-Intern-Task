# Approach

## Objective and constraints

The pipeline answers “how much do we spend, on what, and with whom?” for three companies with one config-driven codebase. The hard constraints are:

- one output row per input line, including exclusions;
- only four exclusion codes;
- exact valid paths from each company's taxonomy;
- a new company must need a config, not code changes;
- category names must never be hardcoded;
- no labels are assumed for company C and for the hidden-label A2 test;
- 25,000 lines must complete in under 60 minutes.

## Architecture

`run.py` executes this sequence:

1. **Config loader** validates the YAML and resolves all paths.
2. **Ingestor** reads each configured CSV as strings, preserving raw values.
3. **Normalizer** produces canonical amount, VAT, FX, date, PO, text and status fields.
4. **Eligibility engine** applies status rules, non-procurement regexes and duplicate keys.
5. **Reconciliation engine** matches A ERP goods receipts to A tool export lines.
6. **Entity resolvers** assign `supplier_key` and `item_key`.
7. **Classifier** scores every possible taxonomy path and chooses the best one.
8. **Output writer** enforces the required schema and validates every row.

All company-specific values are in `configs/company_*.yaml`.

## Financial normalization

- **A ERP**: `value_zar` is already net ZAR. It is used directly.
- **A tool**: `gross_amount` is gross. Net is `gross_amount / (1 + vat_rate)`. Blank VAT is 15%; values such as `15`, `15%` and `0.15` all become `0.15`.
- **B/C**: non-reporting-currency amounts are converted with units-per-USD. The line rate is used first; if blank, the invoice/posting month lookup in `fx_rates_usd.csv` is used.

Negative amounts are preserved as credits/reversals rather than silently discarded, because none of the four exclusion codes describes a credit note.

## Eligibility

### Company A

- Tool statuses `Cancelled`/`Void` → `CANCELLED`.
- Tool statuses `Rejected`, `On Hold`, `Pending`, `Draft`, `Unapproved` → `NOT_APPROVED`.
- Repeated tool exports with the same invoice, supplier, line text and gross amount → `DUPLICATE` (first copy kept).
- Tool rows matched to an ERP row → `DUPLICATE` (ERP kept).

### Company B

Two high-precision rules explain all 450 labeled exclusions:

1. Intercompany GL accounts matching `^(IC[0-9 ]|IC$|I/C|INTERCOMPANY)` → `NON_PROCUREMENT`.
2. Tax descriptions containing sales tax, state/local taxes, GST input tax, tax expense or taxes → `NON_PROCUREMENT`.

Tax lines often sit on ordinary payables GLs, so the second rule keys on the description rather than the GL. Exact duplicates are detected by invoice, invoice line, item, description, account, amount, currency and date.

Document type, zero amounts and missing paid dates were considered but not used as exclusion rules: the seed labels do not validate them.

## A reconciliation

1. Extract a 10-digit PO number and optional PO item from messy `po_reference` values (`PO 4510109330`, `PO#4510109330`, `4510109330/10.0`).
2. Convert tool gross to net ZAR so both systems are comparable.
3. Block candidate pairs by PO number.
4. Require the net amounts to agree within 0.5%.
5. Score candidates using amount closeness, PO-item agreement, supplier similarity, line-text similarity and date proximity (`rapidfuzz.WRatio` with a `difflib` fallback).
6. Accept pairs above a configured score threshold, greedily one-to-one by score.
7. Keep the ERP row and mark the matched tool row `DUPLICATE`.

On the supplied A data this produces 16,363 matches from 22,174 amount-compatible candidates, close to the 16,828 PO/amount-compatible tool rows observed in exploration.

## Supplier and item de-duplication

- Labels, when present, are authoritative: exact record IDs map to the supplied `supplier_key` and `item_key`, and matched tool rows inherit their ERP keys.
- Additional label-derived maps are built from normalized supplier name, vendor ID, material number and description.
- Label-free A supplier grouping uses transductive vendor coverage: names with a vendor ID on at least 50% of their rows are master suppliers; the rest fall into one catch-all, with a fuzzy merge for spelling variants. This reconstructs the label partition almost exactly.
- Label-free A item grouping follows the observed master-data process: one item per material number, except `REPR`/`TRNSP RD` text-groups; rows with no material number have no item key.
- B/C group suppliers by vendor ID and items by material number, falling back to normalized description when no material number exists.
- Category names are never part of these rules.

## Classification into a taxonomy

Valid taxonomy paths are read from the taxonomy file at runtime. Scoring only happens over those paths, so hierarchy validation is structural rather than a post-hoc repair step.

### Taxonomy-only engines (A2, C and fallback)

- **Character and word TF-IDF**: cosine similarity to the full path, leaf and parent documents. Character n-grams handle misspellings and coded material text.
- **Taxonomy mention boost**: a path is boosted when its node names occur as whole token sets in the line text.
- **Optional local embeddings**: `intfloat/multilingual-e5-small` with `query:` / `passage:` prefixes. Embeddings are loaded only if the optional packages and cached model are available; otherwise the pipeline silently continues with TF-IDF.
- **Transductive smoothing**: the label-free score is averaged with the k nearest unlabeled lines (character TF-IDF) and with other lines sharing the same material/description. This reduces single-line noise without using any category names.

On a 10,000-line hidden-label A sample, taxonomy-only scoring reached L1/L2/L3 = 0.634/0.208/0.199 with TF-IDF, 0.667/0.296/0.278 after adding local embeddings, and 0.691/0.303/0.285 after kNN + group smoothing. This is the proxy used for the unseen C tree.

### Label engines (A/B when labels exist)

- Nearest-neighbour retrieval over character and word TF-IDF features votes for the neighbours' taxonomy paths.
- Exact lookup by material number and description provides very high-confidence votes; for B, material number maps to one L3 in 622/624 labeled materials.

The final score is a weighted sum. Confidence combines the winning score and its margin over the second-best path, clipped to `[0, 1]`. Excluded rows use the eligibility-rule confidence instead.

## Why this should generalise to C

- The zero-label path is the default: no label file means no code branch changes.
- Taxonomy levels are read from config, so a four-level C tree is handled without code edits.
- Category names are never embedded in Python.
- The same config fields used for B (multi-ERP, currency, GL/account text, material numbers) are available in the C template.
- The optional embedding engine is replaceable; the TF-IDF fallback is fully local and free.

## Validation

`src/evaluate.py` reports:

- A: L1/L2/L3 accuracy, supplier/item grouping ARI and aligned accuracy.
- B: include accuracy, exclusion-reason accuracy and L1/L2/L3 accuracy.
- Output QA: row count, schema, reason codes, zero spend on excluded rows, confidence range and taxonomy-path validity.

Measured results are printed by `run.py` and summarised in `README.md`.
