"""Smoke test: verify that common_table_live_metadata.json is well-formed.

The freshness check is a nightly health monitor, not a build gate — the
metadata snapshot only refreshes when `scripts/refresh_table_metadata.py`
runs (~55 min against 3VDEV). Failing CI on stale data blocks unrelated
merges; instead we warn via the test report and let the nightly KB
rebuild (Phase 1 infra/kb-refresh) keep things fresh automatically.
"""
import datetime
import json
import warnings
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


def test_freshness_warning():
    """Warns (does NOT fail) when the metadata snapshot is stale.

    Re-run `scripts/refresh_table_metadata.py` to refresh, or wait for
    the nightly KB-refresh ECS task (Phase 1 infra) to roll the snapshot.
    """
    data = json.loads(METADATA_PATH.read_text())
    stale_threshold = int(
        (datetime.date.today() - datetime.timedelta(days=STALE_DAYS)).strftime("%Y%m%d")
    )
    stale = []
    for t in data["tables"]:
        md = t.get("max_sales_date")
        if md and int(md) < stale_threshold:
            stale.append((t["table_name"], md))
    if stale:
        warnings.warn(
            f"{len(stale)} tables are stale (>{STALE_DAYS}d old); "
            f"re-run scripts/refresh_table_metadata.py. First 5: {stale[:5]}",
            stacklevel=2,
        )


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
