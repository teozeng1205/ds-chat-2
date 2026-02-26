"""DataFrame analysis utilities for autonomous investigations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


class DataAnalyzer:
    """Runs built-in analyses and deep table EDA."""

    def analyze(self, *, frames: dict[str, pd.DataFrame], analysis_spec: dict[str, Any]) -> dict[str, Any]:
        analysis_type = str(analysis_spec.get("type", "summary"))
        if analysis_type == "distribution":
            return self._distribution(frames, analysis_spec)
        if analysis_type == "issue_impact":
            return self._issue_impact(frames, analysis_spec)
        if analysis_type == "anomaly_summary":
            return self._anomaly_summary(frames, analysis_spec)
        if analysis_type == "table_eda":
            return self._table_eda(frames, analysis_spec)
        return self._summary(frames)

    @staticmethod
    def _primary_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        for frame in frames.values():
            return frame
        return pd.DataFrame()

    def _summary(self, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
        rows = {name: int(len(frame)) for name, frame in frames.items()}
        cols = {name: [str(col) for col in frame.columns] for name, frame in frames.items()}
        return {
            "analysis_type": "summary",
            "results": {"row_counts": rows, "columns": cols},
            "summary_stats": {
                "dataset_count": len(frames),
                "total_rows": int(sum(rows.values())),
            },
            "report_markdown": "\n".join(
                [
                    "## Investigation Summary",
                    f"- Datasets: {len(frames)}",
                    f"- Total rows: {int(sum(rows.values()))}",
                ]
            ),
            "caveats": [],
        }

    def _distribution(self, frames: dict[str, pd.DataFrame], spec: dict[str, Any]) -> dict[str, Any]:
        frame = self._primary_frame(frames)
        column = str(spec.get("column", "impact_score"))
        bucket_count = max(5, int(spec.get("bucket_count", 12)))
        if frame.empty or column not in frame.columns:
            return {
                "analysis_type": "distribution",
                "results": {"error": f"column '{column}' not found"},
                "summary_stats": {},
                "report_markdown": f"No data available for distribution column `{column}`.",
                "caveats": ["Dataset empty or missing requested column."],
            }

        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            return {
                "analysis_type": "distribution",
                "results": {"error": f"column '{column}' has no numeric values"},
                "summary_stats": {},
                "report_markdown": f"Column `{column}` has no numeric values for distribution.",
                "caveats": ["All values were non-numeric or null."],
            }

        histogram, edges = np.histogram(series, bins=bucket_count)
        buckets = []
        for idx, count in enumerate(histogram):
            buckets.append(
                {
                    "bucket": idx + 1,
                    "left": float(edges[idx]),
                    "right": float(edges[idx + 1]),
                    "count": int(count),
                }
            )

        summary = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "p50": float(series.quantile(0.5)),
            "p90": float(series.quantile(0.9)),
            "p95": float(series.quantile(0.95)),
            "max": float(series.max()),
        }
        markdown = [
            f"## Distribution: `{column}`",
            f"- Count: {summary['count']}",
            f"- Mean: {summary['mean']:.4f}",
            f"- P50: {summary['p50']:.4f}",
            f"- P90: {summary['p90']:.4f}",
            f"- P95: {summary['p95']:.4f}",
            f"- Max: {summary['max']:.4f}",
            "",
            "### Histogram Buckets",
        ]
        markdown.extend([f"- [{b['left']:.4f}, {b['right']:.4f}): {b['count']}" for b in buckets])
        return {
            "analysis_type": "distribution",
            "results": {"buckets": buckets},
            "summary_stats": summary,
            "report_markdown": "\n".join(markdown),
            "caveats": [],
        }

    def _issue_impact(self, frames: dict[str, pd.DataFrame], spec: dict[str, Any]) -> dict[str, Any]:
        del spec
        keys = list(frames.keys())
        issues = frames.get(keys[0], pd.DataFrame()) if keys else pd.DataFrame()
        impact = frames.get(keys[1], pd.DataFrame()) if len(keys) > 1 else pd.DataFrame()

        top_rows = []
        if not issues.empty:
            cols = [c for c in ["issue_sources", "issue_reasons", "issue_count", "providercode", "sitecode"] if c in issues.columns]
            top_rows = issues[cols].head(10).fillna("").to_dict(orient="records") if cols else []

        max_rate = None
        if not impact.empty and "issue_rate_pct" in impact.columns:
            rate_series = pd.to_numeric(impact["issue_rate_pct"], errors="coerce").dropna()
            if not rate_series.empty:
                max_rate = float(rate_series.max())

        return {
            "analysis_type": "issue_impact",
            "results": {"top_issues": top_rows},
            "summary_stats": {
                "issue_groups": len(issues),
                "impact_rows": len(impact),
                "max_issue_rate_pct": max_rate,
            },
            "report_markdown": "\n".join(
                [
                    "## Top Site Issues",
                    f"- Issue groups: {len(issues)}",
                    f"- Impact rows: {len(impact)}",
                    f"- Max issue rate pct: {max_rate if max_rate is not None else 'n/a'}",
                ]
            ),
            "caveats": [],
        }

    def _anomaly_summary(self, frames: dict[str, pd.DataFrame], spec: dict[str, Any]) -> dict[str, Any]:
        frame = self._primary_frame(frames)
        confirmed_only = bool(spec.get("confirmed_only", True))
        filtered = frame

        candidate_cols = ["confirmed", "is_confirmed", "confirmed_anomaly", "status"]
        chosen_col = next((col for col in candidate_cols if col in frame.columns), None)
        if confirmed_only and chosen_col is not None:
            if frame[chosen_col].dtype == bool:
                filtered = frame[frame[chosen_col]]
            else:
                normalized = frame[chosen_col].astype(str).str.lower()
                filtered = frame[normalized.isin({"1", "true", "yes", "confirmed", "y"})]

        top_n = max(1, int(spec.get("top_n", 15)))
        preview = filtered.head(top_n).to_dict(orient="records") if not filtered.empty else []

        return {
            "analysis_type": "anomaly_summary",
            "results": {"preview": preview},
            "summary_stats": {
                "rows": int(len(frame)),
                "confirmed_anomalies": int(len(filtered)),
            },
            "report_markdown": "\n".join(
                [
                    "## Customer Collection Anomalies",
                    f"- Total rows loaded: {len(frame)}",
                    f"- Confirmed anomalies: {len(filtered)}",
                ]
            ),
            "caveats": [],
        }

    def _table_eda(self, frames: dict[str, pd.DataFrame], spec: dict[str, Any]) -> dict[str, Any]:
        table_name = str(spec.get("table_name", "table"))
        frame = self._primary_frame(frames)
        if frame.empty:
            return {
                "analysis_type": "table_eda",
                "results": {"error": "No rows returned"},
                "summary_stats": {},
                "report_markdown": f"# EDA: `{table_name}`\n\nNo data returned for this table.",
                "caveats": ["EDA computed on empty dataset."],
            }

        numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
        datetime_cols = [col for col in frame.columns if "date" in str(col).lower() or pd.api.types.is_datetime64_any_dtype(frame[col])]

        missingness = []
        for col in frame.columns:
            nulls = int(frame[col].isna().sum())
            missingness.append(
                {
                    "column": str(col),
                    "null_count": nulls,
                    "null_pct": round((nulls / max(1, len(frame))) * 100.0, 2),
                }
            )
        missingness.sort(key=lambda item: item["null_pct"], reverse=True)

        cardinality = []
        for col in frame.columns:
            nunique = int(frame[col].nunique(dropna=True))
            cardinality.append({"column": str(col), "distinct": nunique})
        cardinality.sort(key=lambda item: item["distinct"], reverse=True)

        numeric_summary = {}
        for col in numeric_cols[:12]:
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
            outliers = int((series > upper).sum()) if iqr and not math.isnan(float(iqr)) else 0
            numeric_summary[str(col)] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "p25": float(q1),
                "p50": float(series.quantile(0.5)),
                "p75": float(q3),
                "p95": float(series.quantile(0.95)),
                "outlier_count": outliers,
            }

        temporal = {}
        for col in datetime_cols[:4]:
            parsed = pd.to_datetime(frame[col], errors="coerce")
            parsed = parsed.dropna()
            if parsed.empty:
                continue
            temporal[str(col)] = {
                "min": parsed.min().isoformat(),
                "max": parsed.max().isoformat(),
                "days_covered": int((parsed.max() - parsed.min()).days),
            }

        correlations: list[dict[str, Any]] = []
        if len(numeric_cols) >= 2:
            corr = frame[numeric_cols[:8]].corr(numeric_only=True)
            for left in corr.columns:
                for right in corr.columns:
                    if left >= right:
                        continue
                    value = corr.loc[left, right]
                    if pd.isna(value):
                        continue
                    correlations.append({"left": str(left), "right": str(right), "corr": float(value)})
            correlations.sort(key=lambda item: abs(item["corr"]), reverse=True)

        risks = []
        if missingness and missingness[0]["null_pct"] > 40:
            risks.append("High missingness on key columns may bias aggregates.")
        if numeric_summary and any(item["outlier_count"] > 0 for item in numeric_summary.values()):
            risks.append("Numeric outliers detected; verify whether they are valid spikes.")

        follow_up_sql = [
            f"SELECT * FROM {table_name} LIMIT 200;",
            f"SELECT COUNT(*) FROM {table_name};",
        ]

        report = [
            f"# EDA Report: `{table_name}`",
            "",
            "## Schema and Row Overview",
            f"- Rows sampled: {len(frame)}",
            f"- Columns: {len(frame.columns)}",
            "",
            "## Missingness",
        ]
        report.extend([f"- {m['column']}: {m['null_count']} ({m['null_pct']}%)" for m in missingness[:15]])
        report.append("")
        report.append("## Cardinality")
        report.extend([f"- {c['column']}: {c['distinct']} distinct" for c in cardinality[:15]])
        report.append("")
        report.append("## Numeric Distribution Summary")
        for col, stats in numeric_summary.items():
            report.append(
                f"- {col}: p25={stats['p25']:.4f}, p50={stats['p50']:.4f}, p75={stats['p75']:.4f}, p95={stats['p95']:.4f}, outliers={stats['outlier_count']}"
            )
        report.append("")
        report.append("## Temporal Coverage")
        if temporal:
            report.extend([f"- {col}: {info['min']} to {info['max']} ({info['days_covered']} days)" for col, info in temporal.items()])
        else:
            report.append("- No date-like columns detected.")
        report.append("")
        report.append("## Correlation Highlights")
        if correlations:
            report.extend([f"- {item['left']} vs {item['right']}: {item['corr']:.3f}" for item in correlations[:8]])
        else:
            report.append("- Not enough numeric columns for correlation analysis.")
        report.append("")
        report.append("## Key Risks")
        if risks:
            report.extend([f"- {risk}" for risk in risks])
        else:
            report.append("- No critical risks detected from sampled data.")
        report.append("")
        report.append("## Recommended Follow-up SQL")
        report.extend([f"- `{sql}`" for sql in follow_up_sql])

        return {
            "analysis_type": "table_eda",
            "results": {
                "missingness": missingness,
                "cardinality": cardinality,
                "numeric_summary": numeric_summary,
                "temporal": temporal,
                "correlations": correlations[:20],
                "risks": risks,
                "follow_up_sql": follow_up_sql,
            },
            "summary_stats": {
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "numeric_columns": int(len(numeric_cols)),
            },
            "report_markdown": "\n".join(report),
            "caveats": ["EDA based on sampled rows and may not represent full-table distribution."],
        }


__all__ = ["DataAnalyzer"]
