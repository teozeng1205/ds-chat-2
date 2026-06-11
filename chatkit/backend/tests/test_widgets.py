"""Unit tests for the reusable result-table widget builder."""

from chatkit.widgets import Card

from app.tools.widgets import (
    MAX_TABLE_COLS,
    MAX_TABLE_ROWS,
    _fmt_cell,
    result_table_card,
)


def _roundtrip(card: Card) -> dict:
    """Dump + re-validate so we catch any invalid widget schema."""
    dumped = card.model_dump(exclude_none=True)
    Card.model_validate(dumped)
    return dumped


def test_result_table_card_builds_and_validates() -> None:
    card = result_table_card(
        columns=["provider", "request_count", "error_rate"],
        rows=[
            {"provider": "QL2", "request_count": 12345, "error_rate": 0.0234},
            {"provider": "XYZ", "request_count": 9876, "error_rate": 1.5},
        ],
        row_count=842,
        title="Query results",
        column_types={"provider": "object", "request_count": "int64", "error_rate": "float64"},
        subtitle="redshift_analytics · 1.2s",
    )
    dumped = _roundtrip(card)
    assert dumped["type"] == "Card"
    # title row + table + caption
    assert len(dumped["children"]) == 3


def test_caption_reports_truncation_of_rows_and_columns() -> None:
    columns = [f"c{i}" for i in range(MAX_TABLE_COLS + 3)]
    rows = [{c: i for c in columns} for i in range(MAX_TABLE_ROWS + 5)]
    card = result_table_card(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        title="Wide",
    )
    dumped = _roundtrip(card)
    caption = dumped["children"][-1]["value"]
    assert f"Showing {MAX_TABLE_ROWS} of" in caption
    assert "+3 more columns" in caption
    # Only the capped number of columns are rendered in the header row.
    table = dumped["children"][1]
    header_cells = table["children"][0]["children"]
    assert len(header_cells) == MAX_TABLE_COLS


def test_numeric_columns_are_right_aligned() -> None:
    card = result_table_card(
        columns=["name", "count"],
        rows=[{"name": "a", "count": 5}],
        row_count=1,
        title="T",
        column_types={"name": "object", "count": "int64"},
    )
    dumped = _roundtrip(card)
    header_cells = dumped["children"][1]["children"][0]["children"]
    # name -> start, count -> end
    name_align = header_cells[0]["children"][0]["textAlign"]
    count_align = header_cells[1]["children"][0]["textAlign"]
    assert name_align == "start"
    assert count_align == "end"


def test_fmt_cell_formats_values() -> None:
    assert _fmt_cell(None) == "—"
    assert _fmt_cell(1234567) == "1,234,567"
    assert _fmt_cell(3.0) == "3"  # whole-valued float renders as int
    assert _fmt_cell(float("nan")) == "—"
    long = "x" * 100
    assert _fmt_cell(long).endswith("…")
