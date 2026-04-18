export interface ChangelogEntry {
  version: string;   // used as localStorage key, e.g. "2026-03-14"
  date: string;      // display date
  items: string[];   // bullet points
}

export const CHANGELOG: ChangelogEntry[] = [
  {
    version: "2026-04-18",
    date: "Apr 18, 2026",
    items: [
      "14 new AWS tools: Step Functions inspection, Lambda errors, Logs Insights, ECS tasks, alarms, EventBridge, Kinesis tails, Glue catalog, QuickSight dashboards",
      "Semantic knowledge base — ask in plain English (\"JetBlue\", \"auto-scheduler\") and get the right tables, codes, and docs; 697-chunk embedded index",
      "Real planner + reviewer sub-agents — planner probes the environment before emitting a plan; reviewer fact-checks numeric claims in the final answer",
      "New \"details\" button in the header opens a side drawer — session stats, 👍/👎 on the last answer, and editable preferences the agent remembers across threads",
      "Agent can now remember things you tell it (\"I'm on the B6 team\", \"default to prod.analytics\") via remember / recall / list_memories / forget tools",
      "Live Glue catalog backs `inspect_table` and partition-warning checks — always current, no stale snapshots",
      "New `write_file` tool fixes the long-running Python/plotting hangs — scripts reliably write and run in ~10–30s instead of stalling",
      "Running tokens and dollar cost visible in the header bar; per-turn cost logged to SQLite for review",
      "Agent traces persisted locally — every run's spans + tool calls queryable after the fact",
      "Task-specific \"skills\" load automatically: SQL investigation, pipeline ops, AWS read-only, Python venv, long-running scripts, git repos",
      "Query-result cache: identical SQL within 15 minutes returns instantly",
      "All feature gates removed — nothing to configure, the full agent is on by default",
    ],
  },
  {
    version: "2026-03-28",
    date: "Mar 28, 2026",
    items: [
      "Upgraded to gpt-5.4 — latest model, smarter analysis and code generation",
      "SQL and S3 queries now show elapsed time in the progress stream",
      "Agent retries automatically on transient API failures — fewer dropped runs",
      "Updated agent SDK to 0.13.2 with MCP resource support and stability fixes",
    ],
  },
  {
    version: "2026-03-14",
    date: "Mar 14, 2026",
    items: [
      "Download cards now render correctly with a working Download button",
      "Bash exploration commands no longer clutter chat — only final outputs (files, charts) appear as cards",
    ],
  },
  {
    version: "2026-03-13",
    date: "Mar 13, 2026",
    items: [
      "Agent responses are faster and cheaper — system prompt trimmed by ~50%",
      "New: agent can push files directly into chat with a Download button",
      "Long-running scripts now stream output live with a heartbeat indicator",
    ],
  },
  {
    version: "2026-03-11",
    date: "Mar 11, 2026",
    items: [
      "Image charts render inline correctly; Download button works on all cards",
      "Upgraded to gpt-5.2 — smarter analysis, better code generation",
    ],
  },
  {
    version: "2026-03-09",
    date: "Mar 9, 2026",
    items: [
      "Shell agent added: run git, bash, and file commands directly in chat",
      "Agent now knows your full AWS topology (3VDEV + 3VPROD)",
      "Agent defaults to production tables — no more dev/local confusion",
    ],
  },
  {
    version: "2026-03-02",
    date: "Mar 2, 2026",
    items: [
      "MySQL (PriceEye production) tables available for queries",
      "Browse repo files: ask about code and the agent can read source directly",
    ],
  },
  {
    version: "2026-02-26",
    date: "Feb 26, 2026",
    items: [
      "Investigation engine rewritten: fully autonomous, KB-driven, no rigid pipeline",
      "PriceEye system architecture added to knowledge base — 18 investigation patterns",
    ],
  },
];

export const LATEST_VERSION = CHANGELOG[0].version;
export const SEEN_KEY = "changelog_seen_version";
