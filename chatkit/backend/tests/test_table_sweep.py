"""Smoke test: verify that common_table_live_metadata.json is fresh and well-formed."""
import json
import datetime
from pathlib import Path

METADATA_PATH = Path(__file__).parents[1] / "app/investigation/knowledge/common_table_live_metadata.json"
STALE_DAYS = 30


def test_metadata_exists_and_parseable():
    assert METADATA_PATH.exists(), f"Metadata file not found: {METADATA_PATH}"
    data = json.loads(METADATA_PATH.read_text())
    assert isinstance(data.get("tables"), list)
    assert data["table_count"] > 0


def test_no_error_tables():
    data = json.loads(METADATA_PATH.read_text())
    error_tables = [t["table_name"] for t in data["tables"] if t.get("status") == "error"]
    assert error_tables == [], f"Error tables present: {error_tables}"


def test_freshness():
    data = json.loads(METADATA_PATH.read_text())
    stale_threshold = int(
        (datetime.date.today() - datetime.timedelta(days=STALE_DAYS)).strftime("%Y%m%d")
    )
    stale = []
    for t in data["tables"]:
        md = t.get("max_sales_date")
        if md and int(md) < stale_threshold:
            stale.append((t["table_name"], md))
    assert stale == [], f"Stale tables: {stale}"


def test_tier_assigned():
    data = json.loads(METADATA_PATH.read_text())
    missing_tier = [t["table_name"] for t in data["tables"] if not t.get("tier")]
    assert missing_tier == [], f"Tables missing tier: {missing_tier}"


def test_key_tables_present():
    data = json.loads(METADATA_PATH.read_text())
    names = {t["table_name"] for t in data["tables"]}
    required = ["prod.monitoring.provider_combined_audit"]
    for tbl in required:
        assert tbl in names, f"Required table missing: {tbl}"
