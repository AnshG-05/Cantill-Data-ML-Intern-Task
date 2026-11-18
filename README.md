# Cantill Data & ML Intern Task

A config-driven spend pipeline that turns raw procurement files from different companies into one clean spend table:

> one row per input line, one counted purchase, one supplier, one item, one valid taxonomy path, one net amount.

The same code runs companies A, B and the unseen company C. Adding a company requires only a new YAML config; category names are never hardcoded in code.

## Deliverables

- `run.py` — CLI entry point
- `src/` — ingest, normalize, eligibility, reconcile, entity resolution, classification, output and evaluation modules
- `configs/company_A.yaml`, `configs/company_B.yaml` — production configs
- `configs/company_C.example.yaml` — blind-test config template
- `outputs/ansh__A.csv`, `outputs/ansh__B.csv` — committed output files
- `APPROACH.md`, `DECISIONS.md` — design and rationale
- `requirements.txt`, `requirements-optional.txt` — dependencies

## Quick start

```powershell
# core dependencies
& "C:\Users\Admin\anaconda3\python.exe" -m pip install -r requirements.txt

# optional: local zero-shot embeddings (falls back to TF-IDF without this)
& "C:\Users\Admin\anaconda3\python.exe" -m pip install -r requirements-optional.txt

# company A
& "C:\Users\Admin\anaconda3\python.exe" run.py --config configs/company_A.yaml

# company B
& "C:\Users\Admin\anaconda3\python.exe" run.py --config configs/company_B.yaml
```

The environment used for development is Anaconda Python at `C:\Users\Admin\anaconda3\python.exe` (pandas 2.2.3, numpy 2.1.3, scikit-learn 1.6.1).

### Useful flags

| Flag | Effect |
|---|---|
| `--output PATH` | Write to a different CSV path (used for blind-test A2/B2/C outputs) |
| `--no-labels` | Ignore any label file; used for zero-label A2 and company C |
| `--embeddings` / `--no-embeddings` | Enable/disable optional local embeddings |
| `--smoothing` / `--no-smoothing` | Enable/disable kNN + group smoothing |
| `--no-evaluate` | Skip label-based accuracy reporting |
| `--no-classify` | Debug upstream stages only |

Example — zero-label A2 output:

```powershell
& "C:\Users\Admin\anaconda3\python.exe" run.py `
  --config configs/company_A.yaml `
  --output outputs/ansh__A2.csv `
  --no-labels --embeddings --smoothing
```

## Output contract

`outputs/ansh__A.csv` and `outputs/ansh__B.csv` use exactly these columns:

```text
record_id,include,exclusion_reason,supplier_key,item_key,spend,l1,l2,l3,l4,confidence
```

Rules enforced by `src/write_output.py`:

- one row per input record, including excluded rows;
- `include` is `1` or `0`;
- excluded rows have `spend = 0` and one of `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, `DUPLICATE`;
- included rows have a blank reason, a non-zero/positive-or-negative net amount and a valid taxonomy path;
- category names come from the taxonomy file at runtime;
- `confidence` is in `[0, 1]`.

Record IDs follow the brief:

- A ERP: `ERP:<line_id>`
- A tool: `TOOL:<tool_record_id>`
- B/C: raw `line_id`

## Adding a company (blind test C)

1. Copy `configs/company_C.example.yaml` to `configs/company_C.yaml`.
2. Point `paths.data_dir`, input filenames, taxonomy and output path at the C files.
3. Map canonical fields to C column names.
4. Set `classification.taxonomy.level_columns` to the C tree levels (up to `category_l4`).
5. Run:

```powershell
& "C:\Users\Admin\anaconda3\python.exe" run.py --config configs/company_C.yaml
```

No Python file is changed. If C has no labels (expected), leave `classification.labels: null`; the taxonomy-only engine is used.

## Pipeline

```text
raw CSVs + YAML config
        ↓
canonical records (record_id, amount, currency, VAT, dates, text, status)
        ↓
eligibility (status, non-procurement, duplicates)
        ↓
reconciliation (A ERP ↔ tool; matched tool rows become DUPLICATE)
        ↓
supplier/item resolvers (labels when available, deterministic rules otherwise)
        ↓
hybrid classifier (rules + label retrieval + taxonomy zero-shot)
        ↓
hierarchy validator + confidence
        ↓
validated CSV
```

Company-specific behaviour lives in the config: input files and column maps, date/VAT/FX rules, allowed and excluded statuses, non-procurement regexes, duplicate keys, reconciliation weights, entity rules and taxonomy/label paths.

## Classification

The classifier keeps a list of valid paths read from the taxonomy. It never invents category names.

| Engine | When used | Signal |
|---|---|---|
| Taxonomy mention | always | whole taxonomy node names found in the line text |
| Taxonomy zero-shot | always | TF-IDF char/word cosine against full path, leaf and parent documents |
| Local embeddings | optional | `intfloat/multilingual-e5-small` with `query:` / `passage:` prefixes; falls back silently if unavailable |
| Label retrieval | A/B when labels exist | nearest labelled lines plus exact material/description lookups |

For C and zero-label A2, label retrieval is inactive; taxonomy mention, TF-IDF and optional embeddings are active. Scores are combined over valid taxonomy paths, the best path is selected, and confidence comes from the top score and its margin over the runner-up.

## Results

Measured on the supplied data (classification accuracy is against withheld labels where available):

| Check | Result |
|---|---:|
| A L1 / L2 / L3 accuracy (labels available) | 0.993 / 0.983 / 0.972 |
| A supplier grouping ARI | 1.000 |
| A item grouping ARI | 0.994 |
| B include accuracy (seed labels) | 0.9997 |
| B exclusion-reason accuracy (seed labels) | 1.000 |
| B L1 / L2 / L3 accuracy (seed labels) | 0.984 / 0.981 / 0.970 |
| A hidden-label proxy (zero labels, 10k sample, L1/L2/L3) | 0.691 / 0.303 / 0.285 |
| 25k-line runtime (TF-IDF fallback) | ~5 minutes for B; ~9 minutes for A |
| Full A run with optional embeddings (65,549 lines) | ~25 minutes |
| Runtime target | under 60 minutes for 25,000 lines |

`run.py` prints the current evaluation block every time it runs.

## Data privacy

The `data/` directory is gitignored. No raw rows, extracted samples or debug files containing source data are part of the repository. Only code, configs, documentation and the required output CSVs are committed.
