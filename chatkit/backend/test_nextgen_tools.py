from __future__ import annotations

import pandas as pd

from app.nextgen_tools import _build_raw_dataset_profile, _select_dataset_ids_for_synthesis
from app.workspace_manager import WorkspaceManager


def test_build_raw_dataset_profile_uses_direct_dataframe_values() -> None:
    frame = pd.DataFrame(
        {
            "market": ["A-B", "A-B", "C-D", None],
            "impact_score": [10.0, 20.5, None, -1.0],
            "count": [1, 2, 3, 4],
        }
    )

    profile = _build_raw_dataset_profile(frame)
    assert profile["row_count"] == 4
    assert profile["column_count"] == 3
    assert "preview_rows" in profile
    assert "numeric_stats" in profile
    assert "top_values" in profile

    impact_stats = profile["numeric_stats"]["impact_score"]
    assert impact_stats["count"] == 3.0
    assert impact_stats["min"] == -1.0
    assert impact_stats["max"] == 20.5

    market_top = profile["top_values"]["market"]
    assert market_top[0]["value"] == "A-B"
    assert market_top[0]["count"] == 2


def test_select_dataset_ids_for_synthesis_prefers_requested_then_artifacts(tmp_path) -> None:
    manager = WorkspaceManager(runtime_root=tmp_path / "runtime")
    workspace = manager.create_turn_workspace("thread", "turn")

    frame = pd.DataFrame({"x": [1, 2, 3]})
    workspace.write_dataset(
        df=frame,
        dataset_id="base_a",
        source_type="sql",
        source_ref="t.a",
    )
    workspace.write_dataset(
        df=frame,
        dataset_id="base_b",
        source_type="sql",
        source_ref="t.b",
    )

    selected_requested = _select_dataset_ids_for_synthesis(
        workspace,
        requested_dataset_id="base_a",
        parsed_analysis={"artifacts": ["base_b", "base_a"]},
    )
    assert selected_requested[0] == "base_a"
    assert "base_b" in selected_requested

    selected_from_artifacts = _select_dataset_ids_for_synthesis(
        workspace,
        requested_dataset_id=None,
        parsed_analysis={"artifacts": ["base_b"]},
    )
    assert selected_from_artifacts == ["base_b"]

    selected_fallback = _select_dataset_ids_for_synthesis(
        workspace,
        requested_dataset_id=None,
        parsed_analysis=None,
    )
    assert selected_fallback
