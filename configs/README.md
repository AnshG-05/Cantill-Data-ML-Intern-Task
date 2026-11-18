# Config schema

Every company is described by one YAML file. The pipeline never branches on `company`; it only reads these keys.

## Required sections

```yaml
company: A                       # label used in logs
reporting_currency: ZAR          # target currency

paths:
  data_dir: data/company_A
  taxonomy: taxonomy_A.csv
  labels: labels_A.csv           # null for zero-label companies
  output: outputs/ansh__A.csv

inputs:                          # one entry per source file
  - name: erp                    # canonical source name
    file: erp_goods_receipts.csv
    date_format: auto            # auto | dayfirst | iso
    record_id:
      prefix: ERP                # null means use raw id
      column: line_id
    fields:
      amount: value_zar          # canonical field -> raw column
      currency: document_currency
      supplier: supplier_name
      item_id: material_number
      description: material_short_text
      # optional: description2..4, supplier_id, po, po_item,
      # quantity, status, invoice, line, account, posting_date, vat_rate
```

### Normalization

- `normalization.vat`: enable gross→net conversion, rate column, default rate.
- `normalization.fx`: enable currency conversion, line rate field, monthly fallback file.
- `normalization.po`: regexes for messy PO references.

### Eligibility

- `eligibility.status`: allowed/excluded status values and reason codes.
- `eligibility.non_procurement`: ordered regex rules over canonical fields.
- `eligibility.duplicates`: field lists that define a repeated export.

### Reconciliation

- Enable only for A.
- Define primary/secondary sources, blocking field, amount tolerance, score weights and threshold.

### Entities

- `entities.supplier` / `entities.item`: ID fields, name fields, label usage and deterministic grouping options.
- Item options: `null_when_id_missing` (A labels have no item key without a material number) and `null_when_unknown` (avoid one catch-all group when both ID and text are blank).

### Classification

```yaml
classification:
  taxonomy:
    file: taxonomy_A.csv
    level_columns: [category_l1, category_l2, category_l3]  # up to l4
  labels:
    file: labels_A.csv
    record_id_column: record_id
    category_columns: [category_l1, category_l2, category_l3]
  text_fields: [description, description2, item_id, supplier]
  zero_shot: ...
  label_retrieval: ...
  taxonomy_mention: ...
  hierarchy: strict
  confidence: ...
```

Category names are read from the taxonomy file; no category string belongs in a config or in Python source.
