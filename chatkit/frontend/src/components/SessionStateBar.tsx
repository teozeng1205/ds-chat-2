import { useEffect, useState } from "react";

interface SessionState {
  alive: boolean;
  cwd: string | null;
  idle_secs: number | null;
  model: string | null;
  turn_count: number;
  totals?: { tokens: number; dollars: number };
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

interface SessionStateBarProps {
  threadId: string | null;
}

export function SessionStateBar({ threadId }: SessionStateBarProps) {
  const [state, setState] = useState<SessionState | null>(null);
  useEffect(() => {
    if (!threadId) {
      setState(null);
      return;
    }
    let mounted = true;

    const poll = async () => {
      try {
        const r = await fetch(`/chatkit/session/${threadId}`);
        if (mounted) setState(await r.json());
      } catch {
        /* ignore network errors */
      }
    };

    poll();
    const id = setInterval(poll, 3000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, [threadId]);

  const shortCwd = state?.cwd
    ? state.cwd.replace(/^\/Users\/[^/]+/, "~").slice(-50)
    : null;

  return (
    <div
      className={[
        "flex items-center gap-2 px-4 py-1.5 text-xs border-b font-mono shrink-0",
        "border-slate-100 dark:border-slate-800",
        "bg-slate-50 dark:bg-slate-900/60",
      ].join(" ")}
    >
      {state && (
        <>
          <span
            className={`w-2 h-2 rounded-full ${state.alive ? "bg-green-500" : "bg-slate-400"}`}
          />
          <span className="text-slate-400">shell</span>
          <span className="text-slate-600 dark:text-slate-300 truncate">{shortCwd ?? "—"}</span>
        </>
      )}
      <div className="ml-auto flex items-center gap-3">
        {state?.model && (
          <span className="text-slate-400 shrink-0">
            {({
              "gpt-5.4": "5.4",
              "gpt-5.4-mini": "mini",
              "gpt-5-mini": "mini",
              "gpt-5.2": "5.2",
            } as Record<string, string>)[state.model] ?? state.model}
          </span>
        )}
        {state != null && state.turn_count > 0 && (
          <span className="text-slate-400 shrink-0">{state.turn_count}t</span>
        )}
        {state?.totals && state.totals.tokens > 0 && (
          <span
            className="text-slate-400 shrink-0"
            title={`${state.totals.tokens.toLocaleString()} tokens · ${formatDollars(state.totals.dollars)}`}
          >
            {formatTokens(state.totals.tokens)} · {formatDollars(state.totals.dollars)}
          </span>
        )}
        {state?.alive && (state.idle_secs ?? 0) > 10 && (
          <span className="text-slate-400">idle {state.idle_secs}s</span>
        )}
      </div>
    </div>
  );
}
