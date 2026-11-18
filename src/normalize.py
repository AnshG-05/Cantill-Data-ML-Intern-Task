"""Amount, VAT, FX, date, PO and text normalization."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import Config

CURRENCY_SYMBOLS = "R$€£¥"


def parse_number(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(rf"[{re.escape(CURRENCY_SYMBOLS)}]", "", text)
    text = text.replace(",", "")
    text = text.replace("%", "")
    text = text.strip().strip("()")
    text = re.sub(r"\s+", "", text)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def parse_rate(value: Any, default: Optional[float] = None, percent: bool = True) -> Optional[float]:
    number = parse_number(value)
    if number is None:
        return default
    if percent and number > 1:
        number = number / 100.0
    return number


def parse_date(value: Any, hint: str = "auto"):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if hint == "dayfirst":
            return pd.to_datetime(text, dayfirst=True, errors="coerce")
        return pd.to_datetime(text, errors="coerce")
    except Exception:
        return None


def month_key(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    try:
        stamp = pd.Timestamp(value)
        return f"{stamp.year:04d}-{stamp.month:02d}"
    except Exception:
        return None


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compact(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", normalize_text(value).upper())


def clean_upper(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", normalize_text(value).upper())).strip()


def normalize_po(value: Any, number_regex: Optional[str], item_regex: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    text = normalize_text(value)
    if not text:
        return None, None
    number = None
    item = None
    if number_regex:
        match = re.search(number_regex, text)
        number = match.group(0) if match else None
    if number is None:
        match = re.search(r"\d{4,}", text)
        number = match.group(0) if match else text
    if item_regex:
        match = re.search(item_regex, text)
        if match:
            item = match.group(1)
    return number, item


def _fx_lookup(cfg: Config, fx_frame, currency: str, month: Optional[str]) -> Optional[float]:
    if fx_frame is None:
        return None
    columns = cfg.get("normalization.fx.rate_columns", {}) or {}
    currency_col = columns.get("currency", "currency")
    month_col = columns.get("month", "month")
    rate_col = columns.get("rate", "units_per_usd")
    subset = fx_frame
    if currency_col in subset.columns:
        subset = subset[subset[currency_col].astype(str).str.upper() == currency.upper()]
    if month and month_col in subset.columns:
        subset = subset[subset[month_col].astype(str).str[:7] == month]
    if subset.empty and currency_col in fx_frame.columns:
        subset = fx_frame[fx_frame[currency_col].astype(str).str.upper() == currency.upper()]
    if subset.empty:
        return None
    rate = parse_number(subset.iloc[0].get(rate_col))
    return rate if rate and rate > 0 else None


def normalize_records(
    records: List[Dict[str, Any]], cfg: Config, side_files: Dict[str, Any]
) -> Dict[str, Any]:
    reporting_currency = str(cfg.get("reporting_currency", "USD")).upper()
    vat_cfg = cfg.get("normalization.vat", {}) or {}
    fx_cfg = cfg.get("normalization.fx", {}) or {}
    po_cfg = cfg.get("normalization.po", {}) or {}
    date_hint = "auto"

    vat_enabled = bool(vat_cfg.get("enabled"))
    vat_sources = set(vat_cfg.get("apply_to") or [])
    vat_rate_field = vat_cfg.get("rate_field")
    vat_default = vat_cfg.get("default_rate", 0.0)

    fx_enabled = bool(fx_cfg.get("enabled"))
    fx_rate_field = fx_cfg.get("rate_field")
    fx_month_field = fx_cfg.get("month_field")
    fx_fallback = fx_cfg.get("fallback_rate")
    fx_frame = side_files.get("fx")

    number_regex = po_cfg.get("extract_number_regex")
    item_regex = po_cfg.get("extract_item_regex")
    fx_missing = 0
    amount_missing = 0

    for record in records:
        amount = parse_number(record.get("amount"))
        if amount is None:
            amount = parse_number(record.get("line_amount"))
        record["amount_local"] = amount
        record["amount_raw"] = amount
        if amount is None:
            amount_missing += 1

        vat_rate = None
        if vat_enabled and (not vat_sources or record["source"] in vat_sources):
            raw_vat = record.get(vat_rate_field)
            if raw_vat is None:
                raw_vat = record.get("vat_rate")
            vat_rate = parse_rate(raw_vat, default=vat_default, percent=True)
        record["vat_rate"] = vat_rate

        amount_net = amount
        if amount_net is not None and vat_rate is not None and vat_cfg.get("gross_to_net", True):
            divisor = 1.0 + vat_rate
            if divisor > 0:
                amount_net = amount_net / divisor
        record["amount_net_local"] = amount_net

        currency = normalize_text(record.get("currency")).upper() or reporting_currency
        record["currency"] = currency
        rate = 1.0
        if fx_enabled and currency != reporting_currency:
            rate = parse_number(record.get(fx_rate_field))
            if fx_cfg.get("rate_kind") == "units_per_usd" and (rate is None or rate <= 0):
                month_date = record.get(fx_month_field) or record.get("posting_date") or record.get("date")
                rate = _fx_lookup(cfg, fx_frame, currency, month_key(month_date))
            if rate is None or rate <= 0:
                rate = parse_number(fx_fallback) or 1.0
                fx_missing += 1
        record["fx_rate"] = rate
        record["amount_reporting"] = amount_net / rate if amount_net is not None else None

        source_cfg = next((item for item in cfg.get("inputs", []) if item.get("name") == record["source"]), None)
        date_hint = str(source_cfg.get("date_format", "auto")) if source_cfg else "auto"
        record["date"] = parse_date(record.get("date"), date_hint)
        record["posting_date"] = parse_date(record.get("posting_date"), "auto")

        po_value = record.get("po")
        po_item_value = record.get("po_item")
        if po_item_value == po_value and item_regex:
            po_item_value = None
        po, po_item = normalize_po(po_value or po_item_value, number_regex, item_regex)
        if po_item_value and po_item is None:
            _, po_item = normalize_po(po_item_value, None, item_regex)
        record["po"] = po
        record["po_item"] = po_item

        for key in ["supplier", "supplier_id", "item_id", "description", "description2", "description3", "description4", "invoice", "status", "account"]:
            value = record.get(key)
            record[key] = normalize_text(value) if value is not None else None
        record["account_description"] = record.get("account")
        record["supplier_norm"] = compact(record.get("supplier"))
        record["supplier_clean"] = clean_upper(record.get("supplier"))
        record["description_norm"] = compact(record.get("description"))
        record["description_clean"] = clean_upper(record.get("description"))
        record["quantity_value"] = parse_number(record.get("quantity"))

    return {
        "amount_missing": amount_missing,
        "fx_missing": fx_missing,
        "reporting_currency": reporting_currency,
    }
