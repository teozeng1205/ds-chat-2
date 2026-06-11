"""Reusable ChatKit widget builders for the DS Chat agent.

These produce richer, more polished widgets than ad-hoc markdown tables —
notably a structured, zebra-striped result table for SQL / S3 query previews.
Builders return plain dicts (the WidgetComponent form ChatKit validates), so
callers can drop them straight into ``Card(children=[...])`` or stream them via
``ctx.context.stream_widget(...)``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from chatkit.widgets import Card

# Keep tables readable inside the chat column: cap visible rows/cols and let the
# caller surface the true totals in the caption.
MAX_TABLE_ROWS = 12
MAX_TABLE_COLS = 8
MAX_CELL_CHARS = 48

# Theme-aware colors (explicit light/dark so they render regardless of which
# semantic tokens the host theme exposes). Tuned to the slate/indigo UI palette.
_ZEBRA_BG = {"light": "rgba(15, 23, 42, 0.035)", "dark": "rgba(255, 255, 255, 0.05)"}
_HEADER_FG = {"light": "#64748b", "dark": "#94a3b8"}


def _is_numeric_dtype(dtype: str | None) -> bool:
    if not dtype:
        return False
    d = dtype.lower()
    return "int" in d or "float" in d or "decimal" in d or "double" in d


def _looks_numeric(values: Iterable[Any]) -> bool:
    seen = False
    for v in values:
        if v is None:
            continue
        seen = True
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
    return seen


def _fmt_cell(value: Any) -> str:
    """Human-readable cell text: thousands separators, tidy floats, em-dash nulls."""
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        if value == int(value) and abs(value) < 1e15:
            return f"{int(value):,}"
        if abs(value) >= 1000 or abs(value) < 0.001:
            return f"{value:,.3g}"
        return f"{value:,.3f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    text = str(value)
    if len(text) > MAX_CELL_CHARS:
        return text[: MAX_CELL_CHARS - 1] + "…"
    return text


def _cell(text: str, *, align: str, header: bool, zebra: bool) -> dict[str, Any]:
    return {
        "type": "Box",
        "flex": 1,
        "minWidth": 0,
        "padding": {"x": "sm", "y": "xs"},
        "background": (_ZEBRA_BG if (zebra and not header) else None),
        "children": [
            {
                "type": "Text",
                "value": text,
                "size": "xs" if header else "sm",
                "weight": "semibold" if header else "normal",
                "color": (_HEADER_FG if header else None),
                "textAlign": align,
                "truncate": True,
                "maxLines": 1,
            }
        ],
    }


def result_table_card(
    *,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    row_count: int,
    title: str,
    column_types: Mapping[str, str] | None = None,
    subtitle: str | None = None,
    status: Any | None = None,
) -> Card:
    """Build a polished, zebra-striped table card from query-preview rows.

    Args:
        columns: ordered column names.
        rows: list of row dicts (already a bounded preview, e.g. first 20).
        row_count: total rows in the full result (for the caption).
        title: card title (e.g. "Query results").
        column_types: optional pandas-style dtype map for numeric alignment.
        subtitle: optional small caption line (e.g. datasource + timing).
        status: optional ChatKit widget status to attach to the card.
    """
    column_types = column_types or {}
    shown_cols = list(columns[:MAX_TABLE_COLS])
    hidden_cols = len(columns) - len(shown_cols)
    shown_rows = list(rows[:MAX_TABLE_ROWS])

    # Decide alignment per column once (numeric → right-aligned).
    aligns: dict[str, str] = {}
    for col in shown_cols:
        numeric = _is_numeric_dtype(column_types.get(col)) or _looks_numeric(
            r.get(col) for r in shown_rows
        )
        aligns[col] = "end" if numeric else "start"

    header_row = {
        "type": "Row",
        "gap": "xs",
        "children": [
            _cell(col, align=aligns[col], header=True, zebra=False) for col in shown_cols
        ],
    }

    body_rows: list[dict[str, Any]] = []
    for i, row in enumerate(shown_rows):
        body_rows.append(
            {
                "type": "Row",
                "gap": "xs",
                "children": [
                    _cell(_fmt_cell(row.get(col)), align=aligns[col], header=False, zebra=(i % 2 == 1))
                    for col in shown_cols
                ],
            }
        )

    table = {
        "type": "Col",
        "gap": 0,
        "border": 1,
        "radius": "lg",
        "children": [header_row, {"type": "Divider", "flush": True}, *body_rows],
    }

    # Caption summarizes what's visible vs. the full result.
    caption_bits: list[str] = []
    if row_count > len(shown_rows):
        caption_bits.append(f"Showing {len(shown_rows)} of {row_count:,} rows")
    else:
        caption_bits.append(f"{row_count:,} row{'s' if row_count != 1 else ''}")
    if hidden_cols > 0:
        caption_bits.append(f"+{hidden_cols} more column{'s' if hidden_cols != 1 else ''}")
    if subtitle:
        caption_bits.append(subtitle)

    children: list[dict[str, Any]] = [
        {
            "type": "Row",
            "align": "center",
            "justify": "between",
            "gap": "sm",
            "children": [
                {"type": "Title", "value": title, "size": "sm"},
                {"type": "Badge", "label": f"{row_count:,} rows", "color": "info", "variant": "soft", "pill": True},
            ],
        },
        table,
        {"type": "Caption", "value": " · ".join(caption_bits), "size": "xs"},
    ]

    card_kwargs: dict[str, Any] = {"size": "lg", "children": children}
    if status is not None:
        card_kwargs["status"] = status
    return Card(**card_kwargs)
