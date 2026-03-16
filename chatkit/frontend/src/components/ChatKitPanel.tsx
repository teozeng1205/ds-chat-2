import {
  ChatKit,
  type UseChatKitReturn,
  type Widgets,
  useChatKit,
} from "@openai/chatkit-react";
import { useCallback, useMemo, useRef, useState } from "react";
import { CHATKIT_API_DOMAIN_KEY, CHATKIT_API_URL } from "../lib/config";
import { CHANGELOG, LATEST_VERSION, SEEN_KEY } from "../lib/changelog";
import { SessionStateBar } from "./SessionStateBar";

const THREAD_STORAGE_KEY = "ds-chat:last-thread-id";
const THEME_STORAGE_KEY = "ds-chat:theme";

type Theme = "dark" | "light";
type WidgetAction = { type: string; payload?: Record<string, unknown> };
type WidgetActionItem = { id: string; widget: Widgets.Card | Widgets.ListView };

// Inner component is keyed by theme so useChatKit remounts cleanly on theme change.
// Thread ID is persisted in localStorage so conversation history is never lost.
function ChatKitCore({
  theme,
  initialThreadId,
  onThreadChange,
  onWidgetAction,
  chatkitApiRef,
  onAbout,
  onToggleTheme,
}: {
  theme: Theme;
  initialThreadId: string | null;
  onThreadChange: (args: { threadId: string | null }) => void;
  onWidgetAction: (action: WidgetAction, widgetItem: WidgetActionItem) => Promise<void>;
  chatkitApiRef: React.MutableRefObject<Pick<
    UseChatKitReturn,
    "sendCustomAction" | "sendUserMessage" | "setComposerValue"
  > | null>;
  onAbout: () => void;
  onToggleTheme: () => void;
}) {
  const chatkitOptions = useMemo(
    () => ({
      api: {
        url: CHATKIT_API_URL,
        domainKey: CHATKIT_API_DOMAIN_KEY,
        uploadStrategy: { type: "two_phase" as const },
      },
      initialThread: initialThreadId,
      onResponseStart: () => {
        void 0;
      },
      onResponseEnd: () => {
        void 0;
      },
      onThreadChange,
      onError: ({ error }: { error: Error }) => {
        void error;
      },
      onLog: ({ name, data }: { name: string; data?: Record<string, unknown> }) => {
        void name;
        void data;
      },
      theme,
      threadItemActions: { retry: true },
      header: {
        leftAction: {
          icon: (theme === "dark" ? "light-mode" : "dark-mode") as "light-mode" | "dark-mode",
          onClick: onToggleTheme,
        },
        rightAction: {
          icon: "book-open" as const,
          onClick: onAbout,
        },
      },
      history: {
        enabled: true,
        showRename: true,
        showDelete: true,
      },
      startScreen: {
        greeting: "Welcome to 3Vchat, Chat about Anything",
        prompts: [
          {
            label: "What's wrong with QL2?",
            prompt: "What's wrong with QL2?",
            icon: "analytics" as const,
          },
          {
            label: "How does autoscheduler work?",
            prompt: "Super detailed how autoscheduler work?",
            icon: "square-code" as const,
          },
          {
            label: "Anomalies for B6 today",
            prompt: "What are the anomalies today for customer B6?",
            icon: "analytics" as const,
          },
          {
            label: "Daily PriceEye Report",
            prompt: "Give me a daily Priceeye Report",
            icon: "chart" as const,
          },
        ],
      },
      widgets: {
        onAction: onWidgetAction,
      },
      composer: {
        attachments: { enabled: true, maxCount: 5, maxSize: 25 * 1024 * 1024 },
        placeholder: "Code, shell, git, data investigation, web search...",
        dictation: { enabled: true },
        models: [
          {
            id: "gpt-5.2",
            label: "Default",
            description: "gpt-5.2 — most capable",
            default: true,
          },
          {
            id: "gpt-5-mini",
            label: "Fast",
            description: "gpt-5-mini — fastest",
          },
        ],
      },
    }),
    [initialThreadId, onThreadChange, onWidgetAction, theme, onAbout, onToggleTheme],
  );

  const chatkit = useChatKit(chatkitOptions);
  chatkitApiRef.current = chatkit;

  return <ChatKit control={chatkit.control} className="flex-1 min-h-0 block w-full" />;
}

export function ChatKitPanel() {
  const [showAbout, setShowAbout] = useState(false);
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window === "undefined") return "dark";
    return (window.localStorage.getItem(THEME_STORAGE_KEY) as Theme | null) ?? "light";
  });

  const [initialThreadId] = useState<string | null>(() => {
    if (typeof window === "undefined") {
      return null;
    }
    return window.localStorage.getItem(THREAD_STORAGE_KEY);
  });

  const [currentThreadId, setCurrentThreadId] = useState<string | null>(initialThreadId);

  const chatkitApiRef = useRef<
    Pick<UseChatKitReturn, "sendCustomAction" | "sendUserMessage" | "setComposerValue"> | null
  >(null);

  const onThreadChange = useCallback(({ threadId }: { threadId: string | null }) => {
    if (typeof window === "undefined") {
      return;
    }
    setCurrentThreadId(threadId);
    if (threadId) {
      window.localStorage.setItem(THREAD_STORAGE_KEY, threadId);
      return;
    }
    window.localStorage.removeItem(THREAD_STORAGE_KEY);
  }, []);

  const onWidgetAction = useCallback(async (action: WidgetAction, widgetItem: WidgetActionItem) => {
    const api = chatkitApiRef.current;
    if (!api) {
      return;
    }

    const prompt =
      typeof action.payload?.prompt === "string" && action.payload.prompt.trim()
        ? action.payload.prompt
        : null;

    if (action.type === "prefill_prompt" && prompt) {
      await api.setComposerValue({ text: prompt });
      return;
    }

    if (action.type === "send_prompt" && prompt) {
      await api.sendUserMessage({ text: prompt });
      return;
    }

    if (action.type === "open_url") {
      const url = typeof action.payload?.url === "string" ? action.payload.url.trim() : "";
      if (url && typeof window !== "undefined") {
        window.open(url, "_blank", "noopener,noreferrer");
      }
      return;
    }

    if (action.type === "download_url") {
      const url = typeof action.payload?.url === "string" ? action.payload.url.trim() : "";
      const filename =
        typeof action.payload?.filename === "string" && action.payload.filename.trim()
          ? action.payload.filename.trim()
          : "plot.png";
      if (url && typeof document !== "undefined") {
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = filename;
        anchor.rel = "noopener noreferrer";
        anchor.style.display = "none";
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
      }
      return;
    }

    if (action.type === "copy_to_clipboard") {
      const text = typeof action.payload?.text === "string" ? action.payload.text : "";
      if (text && typeof navigator !== "undefined") {
        navigator.clipboard.writeText(text).catch(() => void 0);
      }
      return;
    }

    await api.sendCustomAction(action, widgetItem.id);
  }, []);

  const onToggleTheme = useCallback(() => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    if (typeof window !== "undefined") {
      window.localStorage.setItem(THEME_STORAGE_KEY, next);
    }
    setTheme(next);
  }, [theme]);

  const onAbout = useCallback(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(SEEN_KEY, LATEST_VERSION);
    }
    setShowAbout(true);
  }, []);

  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden">
      <SessionStateBar threadId={currentThreadId} />
      <ChatKitCore
        key={theme}
        theme={theme}
        initialThreadId={initialThreadId}
        onThreadChange={onThreadChange}
        onWidgetAction={onWidgetAction}
        chatkitApiRef={chatkitApiRef}
        onAbout={onAbout}
        onToggleTheme={onToggleTheme}
      />

      {showAbout && (
        <div
          className="absolute inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
          onClick={() => setShowAbout(false)}
        >
          <div
            className={[
              "relative mx-6 w-full max-w-4xl rounded-xl shadow-xl overflow-hidden",
              theme === "dark" ? "bg-[#1a1a1a] text-white" : "bg-white text-black",
            ].join(" ")}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className={[
                "absolute right-5 top-5 w-7 h-7 flex items-center justify-center rounded-full text-sm transition-colors",
                theme === "dark" ? "text-white/30 hover:text-white/70 hover:bg-white/10" : "text-black/30 hover:text-black/60 hover:bg-black/5",
              ].join(" ")}
              onClick={() => setShowAbout(false)}
              aria-label="Close"
            >
              ✕
            </button>

            <div className="flex">
              {/* Left: About */}
              <div className="flex-1 px-9 py-9">
                <p className={["text-[11px] font-medium uppercase tracking-widest mb-4", theme === "dark" ? "text-white/30" : "text-black/30"].join(" ")}>
                  3Vchat
                </p>
                <h2 className="text-2xl font-semibold tracking-tight mb-2">Ask anything.</h2>
                <p className={["text-sm mb-8 leading-relaxed", theme === "dark" ? "text-white/50" : "text-black/50"].join(" ")}>
                  Full access to 3VDEV. No dashboards, no context switching.
                </p>

                <div className="space-y-2.5">
                  {[
                    ["Databases", "Redshift, MySQL PriceEye — any table"],
                    ["AWS & S3", "Files, reports, objects from 3VDEV"],
                    ["Python", "Run experiments, crunch numbers, plot"],
                    ["Codebase", "Read, explain, summarize any repo or file"],
                    ["Charts & downloads", "Plots, CSVs, outputs — direct in chat"],
                    ["Analysis", "Anomalies, trends, reports — just ask"],
                  ].map(([label, desc]) => (
                    <div key={label} className={["flex gap-3 text-sm py-2.5 border-b", theme === "dark" ? "border-white/5" : "border-black/5"].join(" ")}>
                      <span className="font-medium w-36 shrink-0">{label}</span>
                      <span className={theme === "dark" ? "text-white/40" : "text-black/40"}>{desc}</span>
                    </div>
                  ))}
                </div>

                <div className="mt-8">
                  <p className={["text-[11px] font-medium uppercase tracking-widest mb-3", theme === "dark" ? "text-white/30" : "text-black/30"].join(" ")}>
                    Try asking
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {[
                      "What's wrong with QL2?",
                      "Daily PriceEye report",
                      "Anomalies for B6 today",
                      "Plot B6 fares this week",
                      "How does autoscheduler work?",
                    ].map((q) => (
                      <span
                        key={q}
                        className={[
                          "rounded-md px-2.5 py-1 text-xs",
                          theme === "dark" ? "bg-white/5 text-white/50" : "bg-black/5 text-black/50",
                        ].join(" ")}
                      >
                        {q}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Divider */}
              <div className={["w-px shrink-0 my-6", theme === "dark" ? "bg-white/8" : "bg-black/6"].join(" ")} />

              {/* Right: changelog */}
              <div className="flex-1 px-9 py-9 overflow-y-auto max-h-[70vh]">
                <p className={["text-[11px] font-medium uppercase tracking-widest mb-6", theme === "dark" ? "text-white/30" : "text-black/30"].join(" ")}>
                  What's New
                </p>
                <div className="space-y-7">
                  {CHANGELOG.map((entry, i) => (
                    <div key={entry.version}>
                      <div className="flex items-center gap-2 mb-2.5">
                        <span className={["text-xs font-medium", theme === "dark" ? "text-white/30" : "text-black/30"].join(" ")}>
                          {entry.date}
                        </span>
                        {i === 0 && (
                          <span className={["text-[10px] font-medium px-1.5 py-0.5 rounded", theme === "dark" ? "bg-white/10 text-white/50" : "bg-black/6 text-black/40"].join(" ")}>
                            latest
                          </span>
                        )}
                      </div>
                      <ul className="space-y-1.5">
                        {entry.items.map((item, j) => (
                          <li key={j} className={["text-sm leading-relaxed", theme === "dark" ? "text-white/60" : "text-black/60"].join(" ")}>
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
