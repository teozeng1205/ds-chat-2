"""Pass 1b — auto-discover repo sub-modules as stage nodes.

Many ATPCO repos ship as multi-module monorepos with no `.properties`
manifest per stage — e.g. `priceeye-v2/source/common-output-generator/`,
`priceeye-v2/source/daily-billing/`, `ingest/ingest-sources/`,
`ds-priceeye-enrichment/enrichment-calculators/`.

Pass 1 (`.properties` + CFN templates) misses these entirely because
the modules have no config manifest of their own. Pass 3 (code
patterns) sees the Java / Python files but can't attribute them to
a stage since `known_stages` is seeded from `aliases.yaml` only.

This pass walks each repo's conventional module layout and emits a
`stage` + `app` node per sub-module. Orchestrator merges these node
names into `known_stages` before running Pass 3, so Pass 3 can
attribute files inside each module to its stage.

Heuristic: a directory is a module when any of these is true:
  - contains a `pom.xml` (Maven)
  - contains a `src/main/java/` tree
  - contains a `src/main/python/` tree
  - contains a `build.sbt` / `build.gradle` / `build.gradle.kts`
  - contains a `package.json`
  - contains a `setup.py` / `pyproject.toml`
  - contains a `Dockerfile` + source file

Looked-at parents (in this order):
  source/*, packages/*, services/*, apps/*, lambdas/*, functions/*,
  jobs/*, modules/*, <repo-root>/*

Stage name = the directory's basename, canonicalized through the
alias table. No bucket/table edges are emitted here — that's Pass 3's
job once the stage is registered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .canonicalize import AliasTable, Node, RepoEntry

log = logging.getLogger(__name__)


# Parents whose direct children are candidate modules.
_MODULE_PARENTS: tuple[str, ...] = (
    "source", "packages", "services", "apps", "lambdas",
    "functions", "jobs", "modules",
)

# Marker files/dirs that identify a directory as a real module.
_MODULE_MARKERS: tuple[str, ...] = (
    "pom.xml", "build.sbt", "build.gradle", "build.gradle.kts",
    "package.json", "setup.py", "pyproject.toml",
    "Dockerfile", "Makefile",
)

_MODULE_MARKER_DIRS: tuple[str, ...] = (
    "src/main/java", "src/main/python", "src/main/scala",
    "src/main/kotlin", "src",
)

# Directory names that are NOT modules (internal plumbing / tooling).
_SKIP_MODULE_NAMES = frozenset({
    "common", "dao", "data", "database-util", "env-check",
    "mock-testing", "test-utils", "testing", "tests",
    "utils", "util", "shared",
    "src", "lib", "libs", "third_party",
    ".git", "docs", "doc", "node_modules", ".venv", "venv",
    "dist", "build", "target", ".idea", ".vscode", "archive",
})


@dataclass
class DiscoveryResult:
    nodes: list[Node]
    modules_scanned: int
    modules_with_marker: int


def discover(
    repos: Iterable[RepoEntry],
    *,
    aliases: AliasTable | None = None,
) -> DiscoveryResult:
    """Walk each repo looking for sub-modules. Emit stage + app nodes.

    No edges are emitted — those come from Pass 3 (code patterns) once
    `known_stages` includes these module names.
    """
    aliases = aliases or AliasTable.load()
    nodes: list[Node] = []
    scanned = 0
    with_marker = 0

    for repo in repos:
        if not repo.local_path.exists():
            continue

        candidates: list[Path] = []
        for parent in _MODULE_PARENTS:
            pdir = repo.local_path / parent
            if pdir.exists() and pdir.is_dir():
                try:
                    for child in pdir.iterdir():
                        if child.is_dir():
                            candidates.append(child)
                except (OSError, PermissionError):
                    continue

        # Also look at repo root for repos that DON'T nest under
        # source/ — e.g. small single-module repos. Only accept if
        # the top-level dir itself has a marker (rare but happens).
        # This avoids adding spurious top-level stages.
        try:
            for child in repo.local_path.iterdir():
                if (
                    child.is_dir()
                    and child.name not in _SKIP_MODULE_NAMES
                    and child.name not in _MODULE_PARENTS
                    and not child.name.startswith(".")
                    and _has_module_marker(child)
                ):
                    candidates.append(child)
        except (OSError, PermissionError):
            pass

        for cand in candidates:
            scanned += 1
            if cand.name in _SKIP_MODULE_NAMES:
                continue
            if not _has_module_marker(cand):
                continue
            with_marker += 1

            canonical = aliases.resolve(cand.name)
            if not canonical:
                continue
            try:
                rel_path = str(cand.relative_to(repo.local_path))
            except ValueError:
                rel_path = str(cand)
            source_tag = f"config:{repo.name}/{rel_path}"

            nodes.append(Node(
                kind="stage",
                name=canonical,
                aliases=(cand.name,) if cand.name != canonical else (),
                metadata={
                    "repo": repo.name,
                    "module_path": rel_path,
                    "discovered_via": "module_layout",
                },
                source=source_tag,
            ))
            nodes.append(Node(
                kind="app",
                name=canonical,
                aliases=(cand.name,) if cand.name != canonical else (),
                metadata={"repo": repo.name},
                source=source_tag,
            ))

    return DiscoveryResult(
        nodes=nodes, modules_scanned=scanned, modules_with_marker=with_marker,
    )


def _has_module_marker(path: Path) -> bool:
    """True if `path` has one of the marker files or dirs."""
    for marker in _MODULE_MARKERS:
        if (path / marker).exists():
            return True
    for marker in _MODULE_MARKER_DIRS:
        if (path / marker).exists() and (path / marker).is_dir():
            return True
    return False


__all__ = ["DiscoveryResult", "discover"]
