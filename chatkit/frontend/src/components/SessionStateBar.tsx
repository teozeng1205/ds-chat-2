import { useEffect, useState } from "react";

interface SessionState {
  alive: boolean;
  cwd: string | null;
  idle_secs: number | null;
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

  if (!state) return null;

  const shortCwd = state.cwd
    ? state.cwd.replace(/^\/Users\/[^/]+/, "~").slice(-50)
    : "—";

  return (
    <div
      className={[
        "flex items-center gap-2 px-4 py-1.5 text-xs border-b font-mono shrink-0",
        "border-slate-100 dark:border-slate-800",
        "bg-slate-50 dark:bg-slate-900/60",
      ].join(" ")}
    >
      <span
        className={`w-2 h-2 rounded-full ${state.alive ? "bg-green-500" : "bg-slate-400"}`}
      />
      <span className="text-slate-400">shell</span>
      <span className="text-slate-600 dark:text-slate-300 truncate">{shortCwd}</span>
      {state.alive && (state.idle_secs ?? 0) > 10 && (
        <span className="ml-auto text-slate-400">idle {state.idle_secs}s</span>
      )}
    </div>
  );
}
