"""Skills: dynamically-loaded instruction packs.

A Skill is a folder-less markdown file with YAML frontmatter:

    ---
    name: sql_investigation
    description: When to run SQL vs. S3 fetches; partition rules.
    keywords: [sql, redshift, partition, mysql, analytics, monitoring]
    tier: high            # optional — used as a tiebreaker in choose_skills
    ---
    <markdown body used as-is>

`SkillRegistry.load(dir)` reads every *.md file under the skills
directory. `choose_skills(prompt, registry, k)` returns the top-k
skills whose keywords best match the user prompt (plain lexical
overlap with simple token scoring).

This is the minimum-viable mechanism; once the semantic KB from Phase 1
is live, choose_skills can switch to embedding-based matching without
changing the loader or the on-disk format.

Kept deliberately free of a YAML dependency — frontmatter is parsed
with a small handwritten reader that handles scalar / list values.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILLS_DIR = BACKEND_ROOT / "skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    keywords: tuple[str, ...]
    body: str
    tier: str = "normal"
    path: Path | None = None


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.S)
_LIST_RE = re.compile(r"^\s*-\s*(.+)\s*$")


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Parse a YAML-ish frontmatter block. Returns (fields, body)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    block, body = m.group(1), m.group(2)
    fields: dict[str, object] = {}
    cur_key: str | None = None
    cur_list: list[str] | None = None

    for line in block.splitlines():
        if not line.strip():
            continue

        list_match = _LIST_RE.match(line)
        if list_match and cur_list is not None:
            cur_list.append(list_match.group(1).strip())
            continue

        if ":" in line:
            # flush pending list
            if cur_key is not None and cur_list is not None:
                fields[cur_key] = tuple(cur_list)
                cur_list = None

            key, _, rest = line.partition(":")
            key = key.strip()
            rest = rest.strip()
            if not rest or rest == "[]":
                # list form starting next line, OR empty inline list
                if rest == "[]":
                    fields[key] = tuple()
                    cur_key = None
                    continue
                cur_key = key
                cur_list = []
                continue
            if rest.startswith("[") and rest.endswith("]"):
                inner = rest[1:-1]
                fields[key] = tuple(x.strip().strip('"').strip("'") for x in inner.split(",") if x.strip())
                continue
            # scalar
            fields[key] = rest.strip().strip('"').strip("'")
            cur_key = None
            cur_list = None

    # flush trailing list
    if cur_key is not None and cur_list is not None:
        fields[cur_key] = tuple(cur_list)
    return fields, body.strip()


def _load_skill_file(path: Path) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to read skill %s: %s", path, exc)
        return None
    fields, body = _parse_frontmatter(text)
    name = str(fields.get("name") or path.stem).strip()
    description = str(fields.get("description") or "").strip()
    keywords_raw = fields.get("keywords") or ()
    if isinstance(keywords_raw, str):
        keywords = tuple(k.strip() for k in keywords_raw.split(",") if k.strip())
    else:
        keywords = tuple(str(k).strip() for k in keywords_raw if str(k).strip())
    tier = str(fields.get("tier") or "normal").strip()
    return Skill(name=name, description=description, keywords=keywords, body=body, tier=tier, path=path)


@dataclass
class SkillRegistry:
    skills: tuple[Skill, ...] = field(default_factory=tuple)

    @classmethod
    def load(cls, skills_dir: Path | str | None = None) -> "SkillRegistry":
        root = Path(skills_dir) if skills_dir else DEFAULT_SKILLS_DIR
        if not root.exists():
            return cls(skills=())
        out: list[Skill] = []
        for md in sorted(root.glob("*.md")):
            if md.name.startswith("_"):  # _README.md etc.
                continue
            s = _load_skill_file(md)
            if s is not None:
                out.append(s)
        return cls(skills=tuple(out))

    def by_name(self, name: str) -> Skill | None:
        for s in self.skills:
            if s.name == name:
                return s
        return None


_TOKEN = re.compile(r"[A-Za-z0-9_]{2,}")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


_TIER_BONUS = {"high": 0.25, "normal": 0.0, "low": -0.25}


def choose_skills(
    prompt: str,
    registry: SkillRegistry,
    *,
    k: int = 3,
    min_score: float = 0.1,
) -> list[Skill]:
    """Return up to k skills best matching the prompt, by keyword overlap."""
    if not registry.skills:
        return []
    prompt_tokens = set(_tokenize(prompt))
    if not prompt_tokens:
        return []

    scored: list[tuple[float, Skill]] = []
    for skill in registry.skills:
        if not skill.keywords:
            continue
        kw_tokens = [kw.lower() for kw in skill.keywords]
        hits = sum(1 for kw in kw_tokens if kw in prompt_tokens)
        if hits == 0:
            continue
        score = hits / len(kw_tokens)
        score += _TIER_BONUS.get(skill.tier, 0.0)
        scored.append((score, skill))

    scored.sort(key=lambda s: (-s[0], s[1].name))
    return [s for score, s in scored[:k] if score >= min_score]


def render_skills(skills: Iterable[Skill]) -> str:
    """Join selected skill bodies into a single string to append to the
    main system prompt. Each skill is wrapped in a tagged section so
    the model can tell boundaries apart."""
    parts: list[str] = []
    for s in skills:
        parts.append(f"<skill name=\"{s.name}\">\n{s.body.strip()}\n</skill>")
    return "\n\n".join(parts)


__all__ = [
    "Skill",
    "SkillRegistry",
    "choose_skills",
    "render_skills",
    "DEFAULT_SKILLS_DIR",
]
