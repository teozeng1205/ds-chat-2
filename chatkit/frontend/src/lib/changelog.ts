export interface ChangelogEntry {
  version: string;   // used as localStorage key, e.g. "2026-03-14"
  date: string;      // display date
  items: string[];   // bullet points
}

export const CHANGELOG: ChangelogEntry[] = [
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
