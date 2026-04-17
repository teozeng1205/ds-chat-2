# Evals

Thin layer on top of `tests/e2e_investigation_cases.json` + `tests/run_e2e_smoke.py` that turns the existing smoke suite into a **measurable, per-category scorecard** for CI use.

Produces a single `scorecard.json` artifact per run so regressions can be gated automatically. Does not replace the smoke runner; it reads its output.

## Layout

- `categories.json` — maps each smoke-case name to a category (`sql`, `kb`, `shell`, `cross_db`, `meta`, `aws_ops`, `python`, `code_nav`). Lets us report per-category pass rates without touching the smoke cases file.
- `scorecard.py` — reads the latest `tests/smoke_reports/e2e_smoke_*.log` (JSONL produced by `run_e2e_smoke.py`) and writes a single `scorecard.json` with overall pass rate, per-category pass rate, tool-call counts, and regression delta vs `baseline.json`.
- `baseline.json` — current accepted baseline. Updated manually when a new run is considered the reference. `scorecard.py` returns nonzero if the current run regresses vs baseline by more than `--tolerance`.

## Usage

```bash
cd chatkit/backend
# 1. Run the existing smoke suite (unchanged)
.venv/bin/python tests/run_e2e_smoke.py --model gpt-5.4-mini --concurrency 5

# 2. Score the most recent run
.venv/bin/python tests/evals/scorecard.py            # reads latest smoke log
.venv/bin/python tests/evals/scorecard.py --log tests/smoke_reports/e2e_smoke_20260417_120000.log
```

Exit codes:
- `0` — pass rate ≥ baseline (within tolerance)
- `2` — pass rate regressed
- `1` — couldn't read a smoke log

## Updating the baseline

After a confirmed-good run, copy the relevant fields from `scorecard.json` into `baseline.json` and commit.
