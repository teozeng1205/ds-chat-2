"""Regression tests for investigation-agent instruction policy."""

from __future__ import annotations

from app.agents.investigation_agent import _build_instructions


def test_domain_knowledge_lookup_requires_kb_and_repo_verification() -> None:
    instructions = _build_instructions()

    assert "Use **both** the KB and the real codebase." in instructions
    assert "Do not answer these from memory and do not stop at `search_kb`." in instructions
    assert 'bash("ls ~/git/{repo}/")' in instructions
    assert "code-verified facts" in instructions
