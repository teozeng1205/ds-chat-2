#!/usr/bin/env python3
"""Refresh investigation knowledge base."""

from app.investigation.runtime import get_runtime


def main() -> None:
    runtime = get_runtime()
    result = runtime.refresh_knowledge_base(force=True)
    print(result)


if __name__ == "__main__":
    main()
