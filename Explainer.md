# Project Explainer — Cantill Data & ML Intern Task

## 1. Purpose

The task is to build a tool that answers a procurement question:

> How much do we spend, on what, and with whom?

The answer is a clean spend table where every real purchase line is counted once, has a clean supplier, a clean item, a category from the company's own tree, and a correct net spend amount in the reporting currency.

The tool must work unattended on fresh data from companies A, B and C. Company C is unseen until the blind test and has no labels.

## 2. What must be built

A **config-driven data pipeline and classifier** that:

1. reads raw files using a company config,
2. normalizes amounts, tax, currency and dates,
3. decides what counts as spend,
4. reconciles duplicate records across systems,
5. de-duplicates suppliers and items,
6. classifies every included line into the company taxonomy,
7. produces one CSV per company in the required format.

A new company must need a new config, not new code. Category names must never be hardcoded.

## 3. Data in the kit

| Company | Files | Main lesson |
|---|---|---|
| A | `erp_goods_receipts.csv`, `invoice_tool_export.csv`, `labels_A.csv`, `taxonomy_A.csv` | Same purchases appear in two systems; VAT and PO formats are messy |
| B | `invoice_lines.csv`, `fx_rates_usd.csv`, `seed_labels.csv`, `taxonomy_B.csv` | Many ERPs, currencies, blank descriptions and only vendor IDs |
| C | One invoice-line file and a tree, revealed at blind test | Unseen tree, no labels, up to four levels |

Observed sizes:

- A ERP: 37,029 rows
- A tool: 28,520 rows
- A labels: 37,029 rows
- A taxonomy: 40 leaves, 5 L1 / 24 L2 / 40 L3
- B invoices: 41,028 rows
- B seed labels: 3,000 rows
- B taxonomy: 40 leaves, 9 L1 / 25 L2 / 40 L3
- B FX rates: 683 rows

## 4. Key concepts

### Spend cube

A clean table where each purchase line is counted once with supplier, item, category and net spend.

### Config-driven ingestion

All company-specific details live in a config file: file paths, column names, formats, currencies, statuses, rules and taxonomy location.

### Reconciliation

Company A has both an ERP goods-receipt file and an accounts-payable tool export. The same purchase can appear in both. The ERP row is counted; the matching tool row is excluded as `DUPLICATE`. Tool-only purchases remain included.

### Financial normalization

A ERP amounts are already net ZAR. A tool amounts are gross and need VAT removed:

```text
net = gross / (1 + vat_rate)
```

Blank A VAT rate uses 15 percent.

B is reported in USD. For a non-USD line:

```text
usd_amount = line_amount / units_per_usd
```

If the line-level rate is blank, use `fx_rates_usd.csv` for the invoice month.

### Exclusion codes

Only four codes are allowed:

- `CANCELLED`
- `NOT_APPROVED`
- `NON_PROCUREMENT`
- `DUPLICATE`

Only approved or paid invoices count as spend.

### Supplier and item de-duplication

One `supplier_key` per real supplier and one `item_key` per real item. The grader checks grouping, not the exact text of the keys.

### Taxonomy-aware classification

Categories are read from the taxonomy file at runtime. The classifier must output an exact valid path and leave levels below the tree's leaf blank.

### Zero-label classification

Company C has no examples. The tool must classify using only category names and unlabeled text. This is tested in advance by hiding A's labels.

### Confidence

Every row needs a value from 0 to 1 reflecting confidence in the decision or classification.

## 5. Required output

File names:

```text
<yourname>__A.csv
<yourname>__B.csv
<yourname>__A2.csv
<yourname>__B2.csv
<yourname>__C.csv
```

Columns:

```text
record_id,include,exclusion_reason,supplier_key,item_key,spend,l1,l2,l3,l4,confidence
```

Rules:

- one row per input record, including excluded rows,
- excluded rows have `spend = 0` and a valid reason code,
- included rows have a blank reason,
- categories use exact taxonomy names,
- A/B use l1-l3; C may use l1-l4,
- confidence is between 0 and 1.

Record IDs:

- A ERP: `ERP:<line_id>`
- A tool: `TOOL:<tool_record_id>`
- B/C: raw `line_id`

## 6. Architecture

```text
CLI / run.py
    ↓
Config Loader
    ↓
Ingestor → Normalizer → Eligibility Engine
    ↓
Reconciliation Engine (A only)
    ↓
Supplier Resolver + Item Resolver
    ↓
Classifier → Hierarchy Validator → Confidence Calibrator
    ↓
Output Writer → QA Validator
```

### Data layers

1. **Raw layer**: untouched strings from source files.
2. **Canonical layer**: standardized internal records.
3. **Decision layer**: exclusions, matches, clusters, classifications.
4. **Output layer**: final CSV.

### Module map

| Module | Job |
|---|---|
| `config.py` | Load and validate company config |
| `ingest.py` | Read configured CSV files as strings |
| `normalize.py` | Parse amounts, VAT, dates, FX and text |
| `eligibility.py` | Apply status and non-procurement rules |
| `reconcile.py` | Match A ERP/tool records and mark duplicates |
| `entities.py` | Build supplier and item clusters |
| `classify.py` | Predict taxonomy paths and confidence |
| `write_output.py` | Write the exact required schema |
| `evaluate.py` | Measure accuracy and runtime |

## 7. Runtime workflow

1. Parse the CLI command and load config.
2. Read all configured input files.
3. Build canonical records and record IDs.
4. Parse amounts, VAT, dates and FX.
5. Apply exclusion rules.
6. If enabled, reconcile systems and mark duplicates.
7. Resolve supplier keys.
8. Resolve item keys.
9. Classify included rows into valid taxonomy paths.
10. Write output CSV.
11. Run validation checks and print a summary.

## 8. Matching workflow for A

1. Normalize PO references such as `PO 4510109330`, `PO#4510109330`, `4510109330/10.0`.
2. Convert tool gross amounts to net ZAR.
3. Block candidate pairs by PO number.
4. Score candidates using amount, PO item, supplier, text and quantity.
5. Accept high-confidence matches.
6. Keep ERP, exclude matched tool rows as `DUPLICATE`.
7. Collapse repeated tool exports.
8. Keep tool-only records as valid spend.

## 9. Classification workflow

```text
description + item code + supplier + account
        ↓
Rule engine
        ↓
Label retrieval engine (only when labels exist)
        ↓
Taxonomy zero-shot engine (always available)
        ↓
Optional LLM fallback
        ↓
Hierarchy validator
        ↓
l1..l4 + confidence
```

For C, the taxonomy-only engine is primary. For A and B, labels improve accuracy but must not leak into the zero-label test.

## 10. Development workflow

1. Create repository and config schema.
2. Implement ingestion and normalization.
3. Implement output writer and validation early.
4. Implement exclusion rules.
5. Implement A reconciliation.
6. Implement supplier/item clustering.
7. Implement label-based classification.
8. Implement taxonomy-only classification.
9. Test A with labels hidden and record accuracy.
10. Test B with seed labels.
11. Benchmark 25,000 rows.
12. Write `README.md`, `APPROACH.md` and `DECISIONS.md`.
13. Freeze the repository.

## 11. Blind-test workflow

Before build deadline:

- commit frozen code,
- commit A/B outputs,
- submit the form with the public repo link.

At blind test:

- add only the C config,
- run the frozen pipeline,
- write A2/B2/C outputs into `submission/`,
- push only the C config and three outputs,
- make no code changes.

## 12. Submission rules

The public repository must contain:

- code,
- configs,
- outputs,
- `README.md`,
- `APPROACH.md`,
- `DECISIONS.md`.

It must not contain task data files. Add the data directory to `.gitignore`.

## 13. Scoring priorities

| Dataset | Points | Main focus |
|---|---:|---|
| A2 | 30 | Matching, exclusions, total spend, supplier/item dedup, A-tree classification |
| B2 | 20 | Exclusions, USD amount, total spend, clear-cut categories |
| C | 50 | Unseen-tree classification, exclusions, supplier/item dedup |

Highest priority is therefore robust zero-shot classification into an unseen taxonomy, followed by A reconciliation.

## 14. Data confidentiality

The data comes from real companies with identifiers replaced. Use it only for this task. Because the repository is public, no raw data, extracted samples or generated debug files containing source rows should be committed.
