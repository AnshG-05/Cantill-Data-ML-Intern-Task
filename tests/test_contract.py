"""Small data-free unit tests for the output contract and core helpers."""

from pathlib import Path
import unittest

import pandas as pd

from src.config import Config
from src.eligibility import apply_eligibility
from src.normalize import parse_number, parse_rate
from src.write_output import OUTPUT_COLUMNS, build_frame, validate_frame

CONFIG = {
    "company": "TEST",
    "eligibility": {
        "status": {
            "enabled": True,
            "field": "status",
            "allowed": ["Paid"],
            "exclude": [{"values": ["Cancelled"], "reason": "CANCELLED"}],
        },
        "non_procurement": [
            {"field": "account_description", "regex": "^IC", "reason": "NON_PROCUREMENT"}
        ],
        "duplicates": [
            {"sources": ["inv"], "fields": ["invoice", "description"], "reason": "DUPLICATE"}
        ],
    },
    "classification": {
        "taxonomy": {"level_columns": ["category_l1", "category_l2", "category_l3"]}
    },
}


class TestNormalize(unittest.TestCase):
    def test_parse_number(self):
        self.assertEqual(parse_number("R 1,234.56"), 1234.56)
        self.assertEqual(parse_number("(52.10)"), -52.10)
        self.assertIsNone(parse_number(""))

    def test_parse_vat_rate(self):
        self.assertAlmostEqual(parse_rate("15", default=0.15), 0.15)
        self.assertAlmostEqual(parse_rate("0.15", default=0.15), 0.15)
        self.assertAlmostEqual(parse_rate("15%", default=0.15), 0.15)


class TestEligibility(unittest.TestCase):
    def test_rules_and_duplicate_order(self):
        cfg = Config(CONFIG, Path("configs/test.yaml"))
        records = [
            {"source": "inv", "include": True, "status": "Paid", "record_id": "1", "invoice": "A", "description": "bolts"},
            {"source": "inv", "include": True, "status": "Cancelled", "record_id": "2", "invoice": "B", "description": "nuts"},
            {"source": "inv", "include": True, "status": "Paid", "record_id": "3", "invoice": "C", "description": "tax", "account_description": "IC01"},
            {"source": "inv", "include": True, "status": "Paid", "record_id": "4", "invoice": "A", "description": "bolts"},
        ]
        stats = apply_eligibility(records, cfg)
        self.assertEqual([r["include"] for r in records], [True, False, False, False])
        self.assertEqual(records[1]["exclusion_reason"], "CANCELLED")
        self.assertEqual(records[2]["exclusion_reason"], "NON_PROCUREMENT")
        self.assertEqual(records[3]["exclusion_reason"], "DUPLICATE")
        self.assertEqual(stats["NON_PROCUREMENT"], 1)


class TestOutputContract(unittest.TestCase):
    def test_schema_and_validation(self):
        cfg = Config(CONFIG, Path("configs/test.yaml"))
        taxonomy = pd.DataFrame(
            [
                {"category_l1": "Materials", "category_l2": "Metal", "category_l3": "Bolts"},
            ]
        )
        records = [
            {
                "record_id": "1",
                "include": True,
                "amount_reporting": 12.345,
                "supplier_key": "SUP:A",
                "item_key": "ITEM:A",
                "category_path": ["Materials", "Metal", "Bolts"],
                "classification_confidence": 0.8,
            },
            {
                "record_id": "2",
                "include": False,
                "exclusion_reason": "DUPLICATE",
                "amount_reporting": 99.0,
                "supplier_key": "SUP:A",
                "item_key": None,
                "eligibility_confidence": 0.9,
            },
        ]
        frame = build_frame(records, cfg, taxonomy)
        self.assertEqual(list(frame.columns), OUTPUT_COLUMNS)
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.loc[1, "spend"], 0.0)
        self.assertTrue(validate_frame(frame, cfg, taxonomy, 2)["valid"])


if __name__ == "__main__":
    unittest.main()
