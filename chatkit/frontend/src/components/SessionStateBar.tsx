import { useEffect, useState } from "react";

interface SessionState {
  alive: boolean;
  cwd: string | null;
  idle_secs: number | null;
  model: string | null;
  turn_count: number;
  aws_profile?: string | null;
  data_env?: "prod" | "dev" | "gold" | null;
}

interface SessionStateBarProps {
  threadId: string | null;
}

function isSessionState(value: unknown): value is SessionState {
  if (!value || typeof value !== "object") {
    return false;
  }
  const item = value as Record<string, unknown>;
  return (
    typeof item.alive === "boolean" &&
    (typeof item.cwd === "string" || item.cwd === null) &&
    (typeof item.idle_secs === "number" || item.idle_secs === null) &&
    (typeof item.model === "string" || item.model === null) &&
    typeof item.turn_count === "number"
  );
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
        const payload: unknown = await r.json();
        if (mounted && isSessionState(payload)) {
          setState(payload);
        }
      } catch {
        /* ignore network errors */
      }
    };

    void poll();
    const id = setInterval(() => {
      void poll();
    }, 3000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, [threadId]);

  const shortCwd = state?.cwd
    ? state.cwd.replace(/^\/Users\/[^/]+/, "~").slice(-50)
    : null;

  const pill =
    "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium " +
    "bg-slate-100/80 text-slate-600 ring-1 ring-slate-200/70 " +
    "dark:bg-white/5 dark:text-slate-300 dark:ring-white/10 backdrop-blur";

  return (
    <div
      className={[
        "flex items-center gap-2.5 px-4 h-11 shrink-0 text-xs",
        "border-b border-slate-200/70 dark:border-white/10",
        "bg-white/70 dark:bg-slate-950/60 backdrop-blur-xl",
      ].join(" ")}
    >
      {/* Brand */}
      <div className="flex items-center gap-2 select-none">
        <span className="grid h-6 w-6 place-items-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-500 text-[11px] font-bold text-white shadow-sm">
          3V
        </span>
        <span className="font-semibold tracking-tight text-slate-700 dark:text-slate-100">
          3Vchat
        </span>
      </div>

      {state && (
        <span className={pill} title={state.alive ? "Shell session active" : "No active shell"}>
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              state.alive ? "bg-emerald-500 shadow-[0_0_0_3px_rgba(16,185,129,0.18)]" : "bg-slate-400"
            }`}
          />
          <span className="font-mono truncate max-w-[34ch]">{shortCwd ?? "shell"}</span>
        </span>
      )}

      {state?.data_env && (
        <span
          className={[
            "inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide ring-1",
            state.data_env === "prod"
              ? "bg-amber-100 text-amber-800 ring-amber-300/60 dark:bg-amber-500/15 dark:text-amber-200 dark:ring-amber-400/30"
              : "bg-sky-100 text-sky-800 ring-sky-300/60 dark:bg-sky-500/15 dark:text-sky-200 dark:ring-sky-400/30",
          ].join(" ")}
          title={
            state.data_env === "prod"
              ? `Data default: PROD (reading 3VPROD via cross-account from ${state.aws_profile ?? "3VDEV"} creds). Say "use dev" in chat to switch.`
              : `Data default: ${state.data_env.toUpperCase()}`
          }
        >
          {state.data_env === "prod" ? "PROD" : state.data_env.toUpperCase()}
        </span>
      )}

      <div className="ml-auto flex items-center gap-2 text-slate-500 dark:text-slate-400">
        {state?.model && (
          <span className={pill}>
            {({
              "gpt-5.5": "5.5",
              "gpt-5.4": "5.4",
              "gpt-5.4-mini": "mini",
              "gpt-5-mini": "mini",
              "gpt-5.2": "5.2",
            } as Record<string, string>)[state.model] ?? state.model}
          </span>
        )}
        {state != null && state.turn_count > 0 && (
          <span className="tabular-nums">{state.turn_count} turns</span>
        )}
        {state?.alive && (state.idle_secs ?? 0) > 10 && (
          <span className="tabular-nums">idle {state.idle_secs}s</span>
        )}
      </div>
    </div>
  );
}
