from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.workspace_manager import WorkspaceManager


def test_workspace_dataset_contract_and_cleanup(tmp_path: Path) -> None:
    manager = WorkspaceManager(runtime_root=tmp_path)
    workspace = manager.create_turn_workspace("thread-a", "turn-b")

    df = pd.DataFrame(
        [
            {"id": 1, "providercode": "QL2", "count": 10},
            {"id": 2, "providercode": "AA", "count": 4},
        ]
    )

    handle, manifest = workspace.write_dataset(
        df=df,
        dataset_id="top_site_issues",
        source_type="sql",
        source_ref="prod.monitoring.provider_combined_audit",
        query="SELECT ...",
        partitions={"sales_date": "20260211", "customers": "AA"},
    )

    assert Path(handle.path).exists()
    assert Path(handle.manifest_path).exists()
    assert manifest.dataset_id == "top_site_issues"
    assert manifest.row_count == 2

    loaded = workspace.read_dataset("top_site_issues")
    assert len(loaded) == 2

    cleanup = workspace.cleanup()
    assert cleanup["deleted"] is True
