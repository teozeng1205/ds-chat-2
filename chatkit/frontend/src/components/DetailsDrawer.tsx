import { useCallback, useEffect, useState } from "react";

import { postFeedback } from "../lib/feedback";

interface MemoryItem {
  key: string;
  value: string;
  updated_at: string;
}

interface SessionInfo {
  model: string | null;
  turn_count: number;
  totals?: { tokens: number; dollars: number };
  alive?: boolean;
  cwd?: string | null;
}

interface FeedbackSummary {
  total: number;
  up: number;
  down: number;
}

interface DetailsDrawerProps {
  open: boolean;
  onClose: () => void;
  threadId: string | null;
}

function formatDollars(d: number): string {
  if (d < 0.01) return `$${d.toFixed(4)}`;
  if (d < 1) return `$${d.toFixed(3)}`;
  return `$${d.toFixed(2)}`;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return `${n}`;
}

export function DetailsDrawer({ open, onClose, threadId }: DetailsDrawerProps) {
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [memory, setMemory] = useState<MemoryItem[]>([]);
  const [feedback, setFeedback] = useState<FeedbackSummary | null>(null);
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");
  const [saveBusy, setSaveBusy] = useState(false);
  const [feedbackBusy, setFeedbackBusy] = useState(false);
  const [feedbackMsg, setFeedbackMsg] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const tasks: Promise<void>[] = [
      fetch("/chatkit/memory")
        .then(r => r.json() as Promise<{ items?: MemoryItem[] }>)
        .then(b => setMemory(b.items ?? []))
        .catch(() => setMemory([])),
    ];
    if (threadId) {
      tasks.push(
        fetch(`/chatkit/session/${threadId}`)
          .then(r => r.json() as Promise<SessionInfo>)
          .then(b => setSession(b))
          .catch(() => setSession(null))
      );
      tasks.push(
        fetch(`/chatkit/feedback/summary/${threadId}`)
          .then(r => r.json() as Promise<FeedbackSummary>)
          .then(b => setFeedback(b))
          .catch(() => setFeedback(null))
      );
    } else {
      setSession(null);
      setFeedback(null);
    }
    await Promise.all(tasks);
  }, [threadId]);

  useEffect(() => {
    if (!open) return;
    void refresh();
  }, [open, refresh]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const addMemory = async () => {
    const key = newKey.trim();
    const value = newValue.trim();
    if (!key || !value) return;
    setSaveBusy(true);
    try {
      const r = await fetch("/chatkit/memory", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key, value }),
      });
      if (r.ok) {
        setNewKey("");
        setNewValue("");
        await refresh();
      }
    } finally {
      setSaveBusy(false);
    }
  };

  const deleteMemory = async (key: string) => {
    await fetch(`/chatkit/memory/${encodeURIComponent(key)}`, { method: "DELETE" });
    await refresh();
  };

  const sendFeedback = async (verdict: 1 | -1) => {
    if (!threadId) return;
    setFeedbackBusy(true);
    setFeedbackMsg(null);
    try {
      const r = await postFeedback({ threadId, verdict });
      if (r.ok) {
        setFeedbackMsg(verdict === 1 ? "Thanks for the 👍" : "Thanks for the 👎");
        await refresh();
      } else {
        setFeedbackMsg(`Feedback failed: ${r.error ?? "unknown"}`);
      }
    } finally {
      setFeedbackBusy(false);
      setTimeout(() => setFeedbackMsg(null), 3000);
    }
  };

  if (!open) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-slate-900/30 dark:bg-black/50 z-40"
        onClick={onClose}
        aria-hidden="true"
      />
      {/* Drawer */}
      <aside
        className={[
          "fixed top-0 right-0 z-50 h-full w-[360px] max-w-full",
          "bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800",
          "shadow-2xl flex flex-col text-sm",
        ].join(" ")}
        role="dialog"
        aria-label="Session details"
      >
        <header className="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-800">
          <h2 className="font-semibold text-slate-700 dark:text-slate-200">Details</h2>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
            aria-label="Close details"
          >
            ✕
          </button>
        </header>

        <div className="flex-1 overflow-y-auto">
          {/* Session stats */}
          <section className="px-4 py-3 border-b border-slate-100 dark:border-slate-800">
            <h3 className="text-xs uppercase tracking-wide text-slate-400 mb-2">
              This session
            </h3>
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
              <dt className="text-slate-400">Model</dt>
              <dd className="font-mono">{session?.model ?? "—"}</dd>
              <dt className="text-slate-400">Turns</dt>
              <dd className="font-mono">{session?.turn_count ?? 0}</dd>
              <dt className="text-slate-400">Tokens</dt>
              <dd className="font-mono">
                {session?.totals ? formatTokens(session.totals.tokens) : "0"}
              </dd>
              <dt className="text-slate-400">Cost</dt>
              <dd className="font-mono">
                {session?.totals ? formatDollars(session.totals.dollars) : "$0.0000"}
              </dd>
              <dt className="text-slate-400">Shell</dt>
              <dd className="font-mono truncate">
                {session?.alive
                  ? (session.cwd ?? "active")
                  : <span className="text-slate-400">idle</span>}
              </dd>
            </dl>
          </section>

          {/* Feedback */}
          <section className="px-4 py-3 border-b border-slate-100 dark:border-slate-800">
            <h3 className="text-xs uppercase tracking-wide text-slate-400 mb-2">
              Rate the last answer
            </h3>
            <div className="flex items-center gap-2">
              <button
                onClick={() => { void sendFeedback(1); }}
                disabled={feedbackBusy || !threadId}
                className="px-3 py-1.5 rounded border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-50"
                aria-label="Thumbs up"
              >
                👍
              </button>
              <button
                onClick={() => { void sendFeedback(-1); }}
                disabled={feedbackBusy || !threadId}
                className="px-3 py-1.5 rounded border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-50"
                aria-label="Thumbs down"
              >
                👎
              </button>
              <span className="ml-auto text-xs text-slate-400 font-mono">
                {feedback ? `${feedback.up} 👍 · ${feedback.down} 👎` : "—"}
              </span>
            </div>
            {feedbackMsg && (
              <p className="mt-2 text-xs text-slate-500">{feedbackMsg}</p>
            )}
          </section>

          {/* Memory */}
          <section className="px-4 py-3">
            <h3 className="text-xs uppercase tracking-wide text-slate-400 mb-2">
              Your preferences <span className="text-slate-500 normal-case">(remembered across threads)</span>
            </h3>

            {memory.length === 0 && (
              <p className="text-xs text-slate-500 mb-3">
                No preferences yet. The agent can save things here with the <code>remember</code> tool,
                or add one below.
              </p>
            )}

            <ul className="space-y-1.5 mb-3">
              {memory.map(item => (
                <li key={item.key} className="flex items-start gap-2 text-xs">
                  <div className="flex-1 min-w-0">
                    <div className="font-mono font-semibold text-slate-700 dark:text-slate-200 truncate">
                      {item.key}
                    </div>
                    <div className="text-slate-500 dark:text-slate-400 break-words">
                      {item.value}
                    </div>
                  </div>
                  <button
                    onClick={() => { void deleteMemory(item.key); }}
                    className="text-slate-400 hover:text-red-500 text-xs shrink-0"
                    aria-label={`Delete memory ${item.key}`}
                  >
                    ✕
                  </button>
                </li>
              ))}
            </ul>

            <div className="space-y-1.5">
              <input
                type="text"
                placeholder="key (e.g., default_customer)"
                value={newKey}
                onChange={e => setNewKey(e.target.value)}
                maxLength={120}
                className="w-full px-2 py-1.5 text-xs rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950 font-mono"
              />
              <textarea
                placeholder="value"
                value={newValue}
                onChange={e => setNewValue(e.target.value)}
                maxLength={4000}
                rows={2}
                className="w-full px-2 py-1.5 text-xs rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-950"
              />
              <button
                onClick={() => { void addMemory(); }}
                disabled={saveBusy || !newKey.trim() || !newValue.trim()}
                className="w-full px-3 py-1.5 rounded bg-slate-800 dark:bg-slate-100 text-slate-100 dark:text-slate-900 text-xs disabled:opacity-40"
              >
                {saveBusy ? "Saving…" : "Save preference"}
              </button>
            </div>
          </section>
        </div>
      </aside>
    </>
  );
}
