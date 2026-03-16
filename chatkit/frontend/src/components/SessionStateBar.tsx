import { useEffect, useState } from "react";
import { LATEST_VERSION, SEEN_KEY } from "../lib/changelog";
import { ChangelogModal } from "./ChangelogModal";

interface SessionState {
  alive: boolean;
  cwd: string | null;
  idle_secs: number | null;
  model: string | null;
  turn_count: number;
}

interface SessionStateBarProps {
  threadId: string | null;
}

export function SessionStateBar({ threadId }: SessionStateBarProps) {
  const [state, setState] = useState<SessionState | null>(null);
  const [showChangelog, setShowChangelog] = useState(false);
  const [hasUnseen, setHasUnseen] = useState(
    () => localStorage.getItem(SEEN_KEY) !== LATEST_VERSION
  );

  const openChangelog = () => {
    localStorage.setItem(SEEN_KEY, LATEST_VERSION);
    setHasUnseen(false);
    setShowChangelog(true);
  };

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
            {state.model === "gpt-5-mini" ? "mini" : state.model}
          </span>
        )}
        {state != null && state.turn_count > 0 && (
          <span className="text-slate-400 shrink-0">{state.turn_count}t</span>
        )}
        {state?.alive && (state.idle_secs ?? 0) > 10 && (
          <span className="text-slate-400">idle {state.idle_secs}s</span>
        )}
        <button onClick={openChangelog}
                className="relative text-slate-400 hover:text-slate-600 text-xs flex items-center gap-1">
          {hasUnseen && (
            <span className="absolute -top-1 -right-1 w-1.5 h-1.5 rounded-full bg-blue-500" />
          )}
          What's new
        </button>
      </div>
      {showChangelog && <ChangelogModal onClose={() => setShowChangelog(false)} />}
    </div>
  );
}
