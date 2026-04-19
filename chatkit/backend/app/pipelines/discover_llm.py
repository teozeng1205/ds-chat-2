"""Pass 4 — LLM-assisted per-repo summary (low confidence, content-cached).

For every repo, feed README.md + a truncated listing of file names +
any detected `docs/*.md` to one gpt-5.4-mini call. The model returns a
strict JSON envelope naming apps the repo deploys, their inputs /
outputs / triggers, and a one-sentence purpose. We translate that
into graph nodes / edges with `confidence=low`.

Only used to fill the long tail — repos with no config (Pass 1 empty),
no deployed resources visible to the caller (Pass 2 empty), and no
helpful code patterns (Pass 3 empty). Everything this pass produces
is tagged low-confidence so the merge layer prefers any corroborating
signal from the deterministic passes.

Cached by SHA-256 of the prompt input so re-runs against an unchanged
repo cost nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .canonicalize import (
    AliasTable,
    Edge,
    Node,
    RepoEntry,
    canonical_redshift_table,
    canonical_s3_prefix,
    node_id,
)

log = logging.getLogger(__name__)

ClientFactory = Callable[[], Any]  # returns an OpenAI-like client

DEFAULT_MODEL = os.environ.get("DS_CHAT_LLM_DISCOVERY_MODEL", "gpt-5.4-mini")
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".data" / "llm_repo_cache"
MAX_README_CHARS = 12_000
MAX_LISTING_LINES = 200


# ── Public API ─────────────────────────────────────────────────────────


@dataclass
class DiscoveryResult:
    nodes: list[Node]
    edges: list[Edge]
    repos_scanned: int
    repos_with_signal: int
    cache_hits: int


_PROMPT_SYSTEM = (
    "You identify data pipeline apps in a Python/Java repo. "
    "Given a README and a truncated file listing, list every app or "
    "service this repo deploys and map its data inputs/outputs. "
    "Return ONLY valid JSON matching the requested schema. Never invent "
    "table / bucket names you don't see in the source material."
)

_PROMPT_USER_TEMPLATE = """Repo name: {repo}

README (truncated to {max_readme} chars):
---
{readme}
---

File listing (up to {max_lines} lines):
---
{listing}
---

Return JSON of shape:
{{
  "apps": [
    {{
      "name": "<kebab-case app name>",
      "inputs":  ["<bucket-or-table-or-app>", ...],
      "outputs": ["<bucket-or-table-or-app>", ...],
      "triggers": ["<schedule or upstream app>", ...],
      "purpose": "<one sentence>"
    }}
  ]
}}

If the repo is clearly not a data pipeline (e.g. pure library, frontend),
return {{"apps": []}}. No prose outside the JSON."""


def discover(
    repos: Iterable[RepoEntry],
    *,
    aliases: AliasTable | None = None,
    client_factory: ClientFactory | None = None,
    model: str = DEFAULT_MODEL,
    cache_dir: Path | None = None,
    max_repos: int | None = None,
) -> DiscoveryResult:
    """Run Pass 4 over every repo. Any repo whose prompt hash is already
    cached skips the LLM call.
    """
    aliases = aliases or AliasTable.load()
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    nodes: list[Node] = []
    edges: list[Edge] = []
    scanned = 0
    with_signal = 0
    cache_hits = 0

    client: Any | None = None
    for repo in repos:
        if max_repos is not None and scanned >= max_repos:
            break
        scanned += 1
        if not repo.local_path.exists():
            continue

        prompt = _build_prompt(repo)
        if not prompt:
            continue
        key = hashlib.sha256(f"{model}|{prompt}".encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{key}.json"

        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                cache_hits += 1
            except Exception:
                payload = None
        else:
            payload = None

        if payload is None:
            if client is None:
                try:
                    client = (client_factory or _default_client_factory)()
                except Exception as exc:  # noqa: BLE001
                    log.warning("discover_llm: no OpenAI client available: %s", exc)
                    break
            try:
                payload = _llm_call(client, model, prompt)
            except Exception as exc:  # noqa: BLE001
                log.warning("discover_llm call for %s failed: %s", repo.name, exc)
                continue
            try:
                cache_path.write_text(
                    json.dumps(payload, ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("discover_llm cache write failed: %s", exc)

        got = _emit_from_payload(
            repo=repo, payload=payload, aliases=aliases,
            nodes=nodes, edges=edges,
        )
        if got:
            with_signal += 1

    return DiscoveryResult(
        nodes=nodes, edges=edges,
        repos_scanned=scanned,
        repos_with_signal=with_signal,
        cache_hits=cache_hits,
    )


# ── Prompt build ──────────────────────────────────────────────────────


def _build_prompt(repo: RepoEntry) -> str | None:
    readme = _read_readme(repo.local_path)
    listing = _summarize_listing(repo.local_path)
    if not readme and not listing:
        return None
    return _PROMPT_USER_TEMPLATE.format(
        repo=repo.name,
        readme=(readme or "(no README)")[:MAX_README_CHARS],
        listing=listing[:MAX_README_CHARS],
        max_readme=MAX_README_CHARS,
        max_lines=MAX_LISTING_LINES,
    )


def _read_readme(root: Path) -> str:
    for candidate in ("README.md", "README.rst", "README.txt", "README"):
        p = root / candidate
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return ""
    return ""


_IGNORE_DIRS = {".git", "node_modules", ".venv", "venv", ".mypy_cache",
                ".pytest_cache", "__pycache__", "dist", "build", "target"}


def _summarize_listing(root: Path) -> str:
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if len(lines) >= MAX_LISTING_LINES:
            lines.append("...")
            break
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        # Skip ignored roots
        if any(seg in _IGNORE_DIRS for seg in rel.parts):
            continue
        if path.is_dir():
            continue
        lines.append(str(rel))
    return "\n".join(lines)


# ── LLM call ─────────────────────────────────────────────────────────


def _default_client_factory() -> Any:
    from openai import OpenAI
    return OpenAI()


def _llm_call(client: Any, model: str, prompt: str) -> dict[str, Any]:
    # Prefer Responses API if present; fall back to chat.completions.
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": _PROMPT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        text = getattr(resp, "output_text", None) or ""
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _PROMPT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
    return _parse_json_envelope(text)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_envelope(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"apps": []}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_BLOCK.search(text)
    if not m:
        return {"apps": []}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"apps": []}


# ── Node / edge emission ──────────────────────────────────────────────


def _emit_from_payload(
    *,
    repo: RepoEntry,
    payload: dict[str, Any],
    aliases: AliasTable,
    nodes: list[Node],
    edges: list[Edge],
) -> int:
    apps = payload.get("apps") if isinstance(payload, dict) else None
    if not isinstance(apps, list):
        return 0
    emitted = 0
    for app in apps:
        if not isinstance(app, dict):
            continue
        name_raw = app.get("name")
        if not isinstance(name_raw, str) or not name_raw.strip():
            continue
        canonical = aliases.resolve(name_raw.strip())
        if not canonical:
            continue
        src = f"llm:{repo.name}:{name_raw.strip()}"
        nodes.append(Node(
            kind="stage", name=canonical,
            aliases=(name_raw.strip(),),
            metadata={"repo": repo.name, "purpose": app.get("purpose")},
            source=src,
        ))
        stage_id = node_id("stage", canonical)

        for target in _iter_strs(app.get("inputs")):
            tgt = _classify_ref(target, aliases, source=src)
            if tgt is None:
                continue
            nodes.append(tgt)
            edges.append(Edge(
                source_id=stage_id, target_id=tgt.id, rel="reads",
                weight=0.3,  # low confidence
                source=src,
            ))
            emitted += 1

        for target in _iter_strs(app.get("outputs")):
            tgt = _classify_ref(target, aliases, source=src)
            if tgt is None:
                continue
            nodes.append(tgt)
            edges.append(Edge(
                source_id=stage_id, target_id=tgt.id, rel="writes",
                weight=0.3,
                source=src,
            ))
            emitted += 1
    return emitted


def _iter_strs(value: Any) -> Iterable[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str) and v.strip()]


_S3_URI = re.compile(r"s3://([A-Za-z0-9.\-]+)(?:/([^\s]+))?")


def _classify_ref(
    raw: str,
    aliases: AliasTable,
    *,
    source: str,
) -> Node | None:
    s = raw.strip()
    if not s:
        return None
    m = _S3_URI.match(s)
    if m:
        bucket, prefix = m.group(1), (m.group(2) or "")
        return Node(
            kind="s3_prefix",
            name=canonical_s3_prefix(bucket, prefix),
            source=source,
        )
    if re.fullmatch(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+", s):
        return Node(
            kind="redshift_table",
            name=canonical_redshift_table(s),
            source=source,
        )
    # Otherwise treat it as an app / stage name
    canonical = aliases.resolve(s)
    if not canonical:
        return None
    return Node(
        kind="stage", name=canonical, aliases=(s,), source=source,
    )


__all__ = [
    "DiscoveryResult",
    "discover",
]
