"""Tools and instructions for the knowledge agent."""

from __future__ import annotations

from agents import WebSearchTool
from agents.tool import WebSearchToolFilters


def knowledge_instructions() -> str:
    return (
        "You are a concise knowledge assistant.\n"
        "Use web_search to answer questions and focus on docs.zanlit.com.\n"
        "If the answer is not on docs.zanlit.com, say so plainly."
    )


def knowledge_tools() -> list[object]:
    return [
        WebSearchTool(
            filters=WebSearchToolFilters(allowed_domains=["docs.zanlit.com"])
        )
    ]
