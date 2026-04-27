"""Tool registry for the general computer-agent harness.

This keeps tool selection explicit and testable while preserving the
existing public tool objects and names.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


ToolFactory = Callable[[], Iterable[Any]]


@dataclass(frozen=True)
class ToolBundle:
    id: str
    description: str
    factory: ToolFactory
    default_enabled: bool = True


class ToolRegistry:
    def __init__(self, bundles: Iterable[ToolBundle] = ()) -> None:
        self._bundles: list[ToolBundle] = list(bundles)

    @property
    def bundles(self) -> tuple[ToolBundle, ...]:
        return tuple(self._bundles)

    def register(self, bundle: ToolBundle) -> None:
        if any(existing.id == bundle.id for existing in self._bundles):
            raise ValueError(f"Duplicate tool bundle id: {bundle.id}")
        self._bundles.append(bundle)

    def build_tools(self, *, include_disabled: bool = False) -> list[Any]:
        tools: list[Any] = []
        for bundle in self._bundles:
            if not include_disabled and not bundle.default_enabled:
                continue
            tools.extend(list(bundle.factory()))
        return tools


def build_default_tool_registry(
    *,
    model: str,
    include_apply_patch: bool,
    include_orchestration: bool = False,
    include_aws_ops: bool = True,
) -> ToolRegistry:
    """Build the default DS Chat tool registry.

    Orchestration tools default off for the general computer harness. The
    modules remain available for experiments and direct tests.
    """
    from agents import WebSearchTool

    from ..agents.planner import as_agent_tool as planner_as_tool
    from ..agents.planner import build_planner_agent
    from ..agents.reviewer import as_agent_tool as reviewer_as_tool
    from ..agents.reviewer import build_reviewer_agent
    from ..tools.apply_patch import apply_patch_tool
    from ..tools.catalog_tools import catalog_tools
    from ..tools.investigation_tools import investigation_tools_core
    from ..tools.lineage_tools import lineage_tools
    from ..tools.ops_tools import ops_tools
    from ..tools.shell_tools import shell_tools
    from ..tools.streams_tools import streams_tools

    registry = ToolRegistry()
    registry.register(ToolBundle(
        id="web",
        description="OpenAI-hosted web search.",
        factory=lambda: [WebSearchTool(search_context_size="medium")],
    ))
    if include_orchestration:
        registry.register(ToolBundle(
            id="orchestration",
            description="Planner and reviewer sub-agents exposed as tools.",
            factory=lambda: [
                planner_as_tool(build_planner_agent()),
                reviewer_as_tool(build_reviewer_agent()),
            ],
        ))
    if include_apply_patch:
        registry.register(ToolBundle(
            id="patch",
            description="Hosted apply-patch editor.",
            factory=apply_patch_tool,
        ))
    registry.register(ToolBundle(
        id="computer_shell",
        description="Persistent shell, filesystem, git, fetch, and artifacts.",
        factory=shell_tools,
    ))
    registry.register(ToolBundle(
        id="priceeye_data",
        description="PriceEye SQL, S3 dataset, Python dataset, KB, and entity tools.",
        factory=investigation_tools_core,
    ))
    if include_aws_ops:
        registry.register(ToolBundle(
            id="aws_ops",
            description="Read-only AWS ops wrappers for compatibility.",
            factory=ops_tools,
        ))
    registry.register(ToolBundle(
        id="streams",
        description="Kinesis stream inspection.",
        factory=streams_tools,
    ))
    registry.register(ToolBundle(
        id="catalog",
        description="Glue catalog and QuickSight tools.",
        factory=catalog_tools,
    ))
    registry.register(ToolBundle(
        id="lineage",
        description="Pipeline graph lineage tools.",
        factory=lineage_tools,
    ))
    return registry


__all__ = ["ToolBundle", "ToolRegistry", "build_default_tool_registry"]
