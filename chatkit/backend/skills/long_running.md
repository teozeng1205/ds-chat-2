---
name: long_running
description: Patterns for running long-lived Python scripts from bash — unbuffered output, progress prints, background + tail, reasonable timeout values.
keywords: [long, running, timeout, background, nohup, tail, capacity, metrics, etl, analytics, streaming, unbuffered, flush, progress, script]
---

## Long-running scripts

When running scripts that take minutes (capacity metrics, large ETL, analytics pipelines):

**Always use unbuffered output + merged stderr:**
```bash
python3 -u ~/git/ds-priceeye-analytics/scripts/capacity_metrics.py --weeks 2 2>&1
```
- `python3 -u`: forces unbuffered stdout so `print()` statements stream line-by-line.
- `2>&1`: merges stderr into stdout so errors appear inline.
- Pass `timeout=1200` (or higher, up to 1800s) to `bash()` for long jobs.

**Add periodic progress prints inside scripts (when you control them):**
```python
import sys
print(f"[{i}/{total}] Processing {item}...", flush=True)
sys.stdout.flush()  # belt-and-suspenders for unbuffered mode
```

**For very long jobs (>30 min) — background + tail:**
```bash
nohup python3 -u ~/git/ds-priceeye-analytics/scripts/capacity_metrics.py --weeks 4 > /tmp/capacity.log 2>&1 &
echo "PID: $!"

# Stream the log
tail -f /tmp/capacity.log
```

**Typical timeout values:**

| Job type | Suggested timeout |
|---|---|
| Quick scripts (<2 min) | 120 (default) |
| Medium ETL (2–10 min) | 600 |
| Capacity / analytics runs | 1200 |
| Full pipeline reproduction | 1800 (max) |
