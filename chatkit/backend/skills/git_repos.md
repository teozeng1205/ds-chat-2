---
name: git_repos
description: Where ATPCO git repositories are checked out and how to find their docs.
keywords: [git, repo, repository, codebase, documentation, docs, priceeye, ds, ingest, analytics, scheduling, enrichment, data, collection]
---

## Git repositories

All git repos live under `~/git/`. Use `bash('ls ~/git')` to list them.

Full per-repo docs live at `~/git/documentations/{repo-name}.md`. Alternatively, call
`search_kb` with the repo name or the concept you're interested in; the KB indexes
`~/git/documentations/` and returns short snippets with file-line references.

Typical investigation flow:
1. `search_kb("auto-scheduler")` — get doc hints + file-line references.
2. `bash("ls ~/git/priceeye-scheduling")` — see what's in the repo.
3. `read_file("~/git/priceeye-scheduling/path/to/file.py")` — read before you edit.
4. `bash("cd ~/git/priceeye-scheduling && grep -rn 'ClassName' --include='*.py'")` — narrow by symbol.
