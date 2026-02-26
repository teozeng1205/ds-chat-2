"""Generic DataFrame analysis modes for autonomous investigations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class DataAnalyzer:
    """Runs generic profile/custom/summarization analyses without task templates."""

    def analyze(self, *, frames: dict[str, pd.DataFrame], analysis_spec: dict[str, Any]) -> dict[str, Any]:
        mode = str(analysis_spec.get("mode", "profile_dataset")).strip().lower()
        if mode in {"profile_dataset", "profile", "summarize_findings", "summary"}:
            return self._profile_dataset(frames, analysis_spec)
        if mode in {"custom"}:
            return self._custom(frames, analysis_spec)
        # Unknown mode falls back to generic profile.
        return self._profile_dataset(frames, analysis_spec)

    @staticmethod
    def _primary_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        for frame in frames.values():
            return frame
        return pd.DataFrame()

    def _profile_dataset(self, frames: dict[str, pd.DataFrame], spec: dict[str, Any]) -> dict[str, Any]:
        focus = str(spec.get("focus", "general")).strip()

        frame = self._primary_frame(frames)
        if frame.empty:
            return {
                "analysis_mode": "profile_dataset",
                "results": {
                    "dataset_count": len(frames),
                    "error": "No rows available for profiling.",
                },
                "summary_stats": {"dataset_count": len(frames), "total_rows": 0},
                "report_markdown": "## Dataset Profile\n- No rows available for profiling.",
                "caveats": ["Analysis executed on empty dataset(s)."],
            }

        # Column classes
        numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
        date_like_cols = [
            col
            for col in frame.columns
            if "date" in str(col).lower() or pd.api.types.is_datetime64_any_dtype(frame[col])
        ]

        # Missingness
        missingness: list[dict[str, Any]] = []
        for col in frame.columns:
            null_count = int(frame[col].isna().sum())
            missingness.append(
                {
                    "column": str(col),
                    "null_count": null_count,
                    "null_pct": round((null_count / max(1, len(frame))) * 100.0, 2),
                }
            )
        missingness.sort(key=lambda row: row["null_pct"], reverse=True)

        # Distinct counts
        cardinality: list[dict[str, Any]] = []
        for col in frame.columns:
            cardinality.append({"column": str(col), "distinct": int(frame[col].nunique(dropna=True))})
        cardinality.sort(key=lambda row: row["distinct"], reverse=True)

        # Numeric summary
        numeric_summary: dict[str, Any] = {}
        for col in numeric_cols[:20]:
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            if series.empty:
                continue
            numeric_summary[str(col)] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if series.count() > 1 else 0.0,
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.5)),
                "p75": float(series.quantile(0.75)),
                "p95": float(series.quantile(0.95)),
                "min": float(series.min()),
                "max": float(series.max()),
            }

        # Temporal range summary
        temporal_coverage: dict[str, Any] = {}
        for col in date_like_cols[:6]:
            parsed = pd.to_datetime(frame[col], errors="coerce").dropna()
            if parsed.empty:
                continue
            temporal_coverage[str(col)] = {
                "min": parsed.min().isoformat(),
                "max": parsed.max().isoformat(),
                "days_covered": int((parsed.max() - parsed.min()).days),
            }

        # Correlations
        corr_pairs: list[dict[str, Any]] = []
        if len(numeric_cols) >= 2:
            corr = frame[numeric_cols[:10]].corr(numeric_only=True)
            for left in corr.columns:
                for right in corr.columns:
                    if left >= right:
                        continue
                    value = corr.loc[left, right]
                    if pd.isna(value):
                        continue
                    corr_pairs.append({"left": str(left), "right": str(right), "corr": float(value)})
            corr_pairs.sort(key=lambda row: abs(row["corr"]), reverse=True)

        # Build report
        lines = [
            "## Dataset Profile",
            f"- Focus: {focus}",
            f"- Datasets profiled: {len(frames)}",
            f"- Rows: {len(frame)}",
            f"- Columns: {len(frame.columns)}",
            "",
            "### Missingness (Top)",
        ]
        lines.extend([f"- {row['column']}: {row['null_count']} ({row['null_pct']}%)" for row in missingness[:15]])
        lines.append("")
        lines.append("### Cardinality (Top)")
        lines.extend([f"- {row['column']}: {row['distinct']} distinct" for row in cardinality[:15]])
        lines.append("")
        lines.append("### Numeric Summary")
        if numeric_summary:
            for col, stats in numeric_summary.items():
                lines.append(
                    f"- {col}: p50={stats['p50']:.4f}, p95={stats['p95']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}"
                )
        else:
            lines.append("- No numeric columns detected.")
        lines.append("")
        lines.append("### Temporal Coverage")
        if temporal_coverage:
            for col, stats in temporal_coverage.items():
                lines.append(f"- {col}: {stats['min']} to {stats['max']} ({stats['days_covered']} days)")
        else:
            lines.append("- No date-like columns detected.")
        lines.append("")
        lines.append("### Correlation Highlights")
        if corr_pairs:
            lines.extend([f"- {item['left']} vs {item['right']}: {item['corr']:.3f}" for item in corr_pairs[:8]])
        else:
            lines.append("- Not enough numeric columns for correlation analysis.")

        caveats: list[str] = []
        if len(frame) >= 100000:
            caveats.append("Profile may be based on bounded extraction sample, not full-table scan.")

        return {
            "analysis_mode": "profile_dataset",
            "results": {
                "missingness": missingness,
                "cardinality": cardinality,
                "numeric_summary": numeric_summary,
                "temporal_coverage": temporal_coverage,
                "correlation_pairs": corr_pairs[:20],
            },
            "summary_stats": {
                "dataset_count": len(frames),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "numeric_columns": int(len(numeric_cols)),
            },
            "report_markdown": "\n".join(lines),
            "caveats": caveats,
        }

    def _custom(self, frames: dict[str, pd.DataFrame], spec: dict[str, Any]) -> dict[str, Any]:
        frame = self._primary_frame(frames)
        columns = [str(col) for col in frame.columns]

        requested_column = str(spec.get("column", "")).strip()
        results: dict[str, Any] = {
            "row_count": int(len(frame)),
            "column_count": len(columns),
            "columns": columns,
        }

        if requested_column and requested_column in frame.columns:
            series = frame[requested_column]
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if not numeric.empty:
                    results["column_summary"] = {
                        "column": requested_column,
                        "count": int(numeric.count()),
                        "mean": float(numeric.mean()),
                        "p50": float(numeric.quantile(0.5)),
                        "p90": float(numeric.quantile(0.9)),
                        "max": float(numeric.max()),
                    }
            else:
                top = series.astype(str).value_counts().head(20)
                results["column_summary"] = {
                    "column": requested_column,
                    "top_values": [{"value": idx, "count": int(val)} for idx, val in top.items()],
                }

        return {
            "analysis_mode": "custom",
            "results": results,
            "summary_stats": {
                "rows": int(len(frame)),
                "columns": len(columns),
            },
            "report_markdown": "## Custom Dataset Analysis\n- Rows: {}\n- Columns: {}".format(len(frame), len(columns)),
            "caveats": [],
        }


__all__ = ["DataAnalyzer"]
