# Decisions and assumptions

This file records the judgment calls made where the brief or data allowed more than one interpretation.

## Project and output

| Decision | Rationale |
|---|---|
| Output name is `ansh` | Confirmed with the candidate; produces `ansh__A.csv`, `ansh__B.csv`, etc. |
| Anaconda Python is the development runtime | It already contains pandas, numpy, scikit-learn and scipy; system Python does not. |
| `data/` is gitignored | The source files are confidential; the public repository contains only code, configs, docs and required outputs. |
| Outputs are written to `outputs/` | Keeps generated deliverables separate from code while remaining part of the repository. |
| One row per input line, always | Required by the brief; excluded rows keep their original record IDs. |
| Excluded rows have blank categories | A category is a claim about spend; excluded rows do not represent spend. |

## Company A

| Decision | Rationale |
|---|---|
| ERP goods receipts are authoritative; matched tool rows are `DUPLICATE` | The brief explicitly says the ERP row is counted and the matching tool row excluded. |
| Tool gross amounts are converted with `net = gross / (1 + vat_rate)` | Stated in the brief; blank VAT defaults to 15%. |
| `15`, `15%` and `0.15` are all parsed as 15% | Observed in the tool export; values above 1 are treated as percentages. |
| Tool `Cancelled` → `CANCELLED`; `Rejected`/`On Hold` → `NOT_APPROVED` | Direct semantic mapping to the allowed reason codes. |
| Repeated tool exports are duplicates only when invoice, supplier, line text and gross amount all match | Prevents legitimate multi-line invoices from being collapsed. |
| Reconciliation requires PO agreement and amount agreement within 0.5% | PO + amount is highly discriminative; supplier/text/date break ties. |
| One-to-one greedy matching by score | Avoids one ERP receipt absorbing several different tool lines. |
| Negative ERP/tool amounts remain included as negative spend | They are credits/reversals; none of the four exclusion codes fits a credit. |
| Label keys are used verbatim when labels exist | Grader checks grouping, but using supplied keys removes ambiguity. |
| Matched tool rows inherit ERP supplier/item keys | They represent the same purchase. |
| Label-free supplier rule uses vendor-ID coverage ≥ 50% | It reconstructs the label partition almost exactly without labels. |
| Label-free item rule groups by material number, with `REPR`/`TRNSP RD` text groups | Matches the master-data process inferred from the labels; the opaque catch-all cannot be derived without labels. |
| Rows with no material number get a blank item key in A | Labels have null item key exactly when material number is null. |

## Company B

| Decision | Rationale |
|---|---|
| Reporting currency is USD | Stated in the brief. |
| `currency_rate` is units per USD; conversion is `amount / rate` | Stated in the brief and confirmed by the FX file columns. |
| Blank line rates fall back to the posting-month rate in `fx_rates_usd.csv` | Stated in the brief. |
| Intercompany GL accounts are `NON_PROCUREMENT` | 364/450 labeled exclusions; 0 labeled false positives. |
| Tax descriptions are `NON_PROCUREMENT` | The remaining 86/450 labeled exclusions; 0 labeled false positives. |
| Generic GL codes such as `PEUS`, `PECA`, `FXW` are **not** blanket-excluded | They contain many valid purchases; only tax descriptions on those GLs are excluded. |
| Document type, zero amount and missing paid date are not exclusion signals | Not validated by the seed labels; excluding them would remove genuine spend. |
| Exact duplicate detection includes invoice line number | Same invoice can contain many legitimate lines. |
| Material number is the strongest category signal | 622/624 labeled materials map to exactly one L3; exact lookup is enabled. |
| Supplier keys come from vendor ID | B has no supplier names in the invoice file. |
| Items fall back to normalized description when material number is blank | Preserves distinct spend items without inventing a material ID. |

## Company C and zero-label mode

| Decision | Rationale |
|---|---|
| A separate config template is provided | Blind-test rule: add only a C config, no code changes. |
| Label retrieval is disabled by default for C | No labels are expected. |
| Taxonomy levels are configurable up to `category_l4` | C may have four levels. |
| Taxonomy-only TF-IDF is the mandatory fallback | No API key, paid service or model download is required. |
| Local sentence-transformer embeddings are optional | They may improve unseen-tree accuracy, but the pipeline must run without them. |
| Taxonomy names are never hardcoded | All category text comes from the taxonomy CSV at run time. |
| Intercompany/tax regexes are config-driven | If C uses different wording, only its YAML changes. |

## Classification and confidence

| Decision | Rationale |
|---|---|
| Score only valid taxonomy paths | Guarantees exact, hierarchical output without repair heuristics. |
| Combine full-path, leaf and parent similarities | Full path disambiguates sibling leaves; leaf/parent signals help shallow trees. |
| Use labels only when a label file exists and is enabled | Prevents leakage into A2/C. |
| Use nearest-neighbour votes plus exact lookup | Strong when labels exist, especially for repeated material numbers. |
| Smooth only the label-free score with kNN/group means | It improved hidden-label L3 from 0.278 to 0.285 in the A proxy without diluting supervised evidence. |
| Confidence is higher when the winning margin is larger | Gives a usable measure of decision certainty without claiming calibrated probabilities. |
| Excluded rows use rule confidence | Classification confidence is irrelevant when no spend is counted. |

## Runtime

| Decision | Rationale |
|---|---|
| Sparse TF-IDF and batched top-k retrieval | Keeps 40k+ row runs within memory and time limits. |
| Optional embedding encoding is batched and cached by sentence-transformers | Avoids repeated model loading and stays within the 60-minute target. |
| `rapidfuzz` with `difflib` fallback | Fast matching when available, no hard dependency for correctness. |
