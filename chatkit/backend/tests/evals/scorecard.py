#!/usr/bin/env python3
"""Score the most recent E2E smoke run and emit a machine-readable scorecard.

Reads a JSONL log produced by tests/run_e2e_smoke.py (by default the most
recent one in tests/smoke_reports/) and writes scorecard.json next to it.

Exit codes:
  0 — overall pass rate ≥ baseline minus tolerance
  1 — no smoke log found or unreadable
  2 — overall pass rate regressed vs baseline

Usage:
  python tests/evals/scorecard.py                        # latest log
  python tests/evals/scorecard.py --log PATH             # specific log
  python tests/evals/scorecard.py --tolerance 0.05       # allow 5pp slack
  python tests/evals/scorecard.py --update-baseline      # write current into baseline.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = EVALS_DIR.parents[1]
SMOKE_DIR = BACKEND_ROOT / "tests" / "smoke_reports"
CATEGORIES_PATH = EVALS_DIR / "categories.json"
BASELINE_PATH = EVALS_DIR / "baseline.json"


def _load_categories() -> dict[str, str]:
    payload = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
    return dict(payload.get("categories") or {})


def _latest_smoke_log() -> Path | None:
    if not SMOKE_DIR.exists():
        return None
    logs = sorted(SMOKE_DIR.glob("e2e_smoke_*.log"))
    return logs[-1] if logs else None


def _load_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _score(rows: list[dict[str, Any]], categories: dict[str, str]) -> dict[str, Any]:
    total = len(rows)
    skipped = sum(1 for r in rows if r.get("skipped"))
    considered = [r for r in rows if not r.get("skipped")]
    passed_rows = [r for r in considered if r.get("passed")]
    overall = round(len(passed_rows) / len(considered), 4) if considered else 0.0

    per_cat_total: Counter[str] = Counter()
    per_cat_pass: Counter[str] = Counter()
    uncategorized: list[str] = []
    for r in considered:
        cat = categories.get(r.get("name") or "") or "uncategorized"
        if cat == "uncategorized":
            uncategorized.append(r.get("name") or "")
        per_cat_total[cat] += 1
        if r.get("passed"):
            per_cat_pass[cat] += 1

    per_category = {
        cat: {
            "total": per_cat_total[cat],
            "passed": per_cat_pass[cat],
            "pass_rate": round(per_cat_pass[cat] / per_cat_total[cat], 4),
        }
        for cat in sorted(per_cat_total)
    }

    tool_calls: Counter[str] = Counter()
    for r in considered:
        for tc in r.get("tool_calls") or []:
            tool_calls[tc.get("name") or "unknown"] += 1

    errors = [{"name": r.get("name"), "error": r.get("error")} for r in considered if r.get("error")]
    failed = [
        {"name": r.get("name"), "failures": r.get("failures") or []}
        for r in considered
        if not r.get("passed") and not r.get("error")
    ]

    model = next((r.get("model") for r in rows if r.get("model")), None)

    return {
        "overall_pass_rate": overall,
        "total_cases": total,
        "skipped": skipped,
        "considered": len(considered),
        "passed": len(passed_rows),
        "per_category_pass_rate": {cat: per_category[cat]["pass_rate"] for cat in per_category},
        "per_category": per_category,
        "tool_call_counts": dict(tool_calls),
        "errors": errors,
        "failed": failed,
        "uncategorized": sorted(set(x for x in uncategorized if x)),
        "model": model,
    }


def _load_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.exists():
        return {"overall_pass_rate": 0.0, "per_category_pass_rate": {}}
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _regression_ok(current: dict[str, Any], baseline: dict[str, Any], tolerance: float) -> list[str]:
    regressions: list[str] = []
    base_overall = float(baseline.get("overall_pass_rate") or 0.0)
    cur_overall = float(current.get("overall_pass_rate") or 0.0)
    if cur_overall + tolerance < base_overall:
        regressions.append(f"overall: {cur_overall:.3f} < baseline {base_overall:.3f} - {tolerance}")
    base_per_cat = dict(baseline.get("per_category_pass_rate") or {})
    cur_per_cat = dict(current.get("per_category_pass_rate") or {})
    for cat, base_rate in base_per_cat.items():
        cur_rate = float(cur_per_cat.get(cat, 0.0))
        if cur_rate + tolerance < float(base_rate):
            regressions.append(f"{cat}: {cur_rate:.3f} < baseline {base_rate:.3f} - {tolerance}")
    return regressions


def main() -> int:
    p = argparse.ArgumentParser(description="Emit scorecard from an e2e smoke log")
    p.add_argument("--log", type=Path, default=None, help="path to e2e_smoke_*.log (default: latest)")
    p.add_argument("--out", type=Path, default=None, help="path to scorecard.json (default: next to log)")
    p.add_argument("--tolerance", type=float, default=0.05, help="allowed pp slack vs baseline")
    p.add_argument("--update-baseline", action="store_true", help="write current into baseline.json")
    args = p.parse_args()

    log_path = args.log or _latest_smoke_log()
    if not log_path or not Path(log_path).exists():
        print("No smoke log found. Run tests/run_e2e_smoke.py first.", file=sys.stderr)
        return 1

    rows = _load_log(Path(log_path))
    if not rows:
        print(f"Log is empty: {log_path}", file=sys.stderr)
        return 1

    categories = _load_categories()
    card = _score(rows, categories)
    card["source_log"] = str(log_path)
    card["recorded_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    out_path = args.out or Path(log_path).with_suffix(".scorecard.json")
    out_path.write_text(json.dumps(card, indent=2), encoding="utf-8")

    baseline = _load_baseline()
    regressions = _regression_ok(card, baseline, args.tolerance)

    print(f"Scorecard: {out_path}")
    print(f"Overall: {card['overall_pass_rate']:.3f}  ({card['passed']}/{card['considered']})  skipped={card['skipped']}")
    for cat, info in card["per_category"].items():
        print(f"  {cat:<14} {info['pass_rate']:.3f}  ({info['passed']}/{info['total']})")
    if card["uncategorized"]:
        print(f"Uncategorized cases: {', '.join(card['uncategorized'])}")
    if regressions:
        print("Regressions:")
        for r in regressions:
            print(f"  - {r}")

    if args.update_baseline:
        new_baseline = {
            "overall_pass_rate": card["overall_pass_rate"],
            "per_category_pass_rate": card["per_category_pass_rate"],
            "recorded_at": card["recorded_at"],
            "source_log": card["source_log"],
        }
        BASELINE_PATH.write_text(json.dumps(new_baseline, indent=2), encoding="utf-8")
        print(f"Updated baseline: {BASELINE_PATH}")

    return 2 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
