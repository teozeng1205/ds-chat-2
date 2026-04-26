import {
  ChatKit,
  type UseChatKitReturn,
  type Widgets,
  useChatKit,
} from "@openai/chatkit-react";
import type { MutableRefObject } from "react";
import { useCallback, useMemo, useRef, useState } from "react";
import { CHATKIT_API_DOMAIN_KEY, CHATKIT_API_URL } from "../lib/config";
import { SessionStateBar } from "./SessionStateBar";

const THEME_STORAGE_KEY = "ds-chat:theme";

type Theme = "dark" | "light";
type WidgetAction = { type: string; payload?: Record<string, unknown> };
type WidgetActionItem = { id: string; widget: Widgets.Card | Widgets.ListView };

// Inner component is keyed by theme so useChatKit remounts cleanly on theme change.
function ChatKitCore({
  theme,
  onThreadChange,
  onWidgetAction,
  chatkitApiRef,
  onToggleTheme,
}: {
  theme: Theme;
  onThreadChange: (args: { threadId: string | null }) => void;
  onWidgetAction: (action: WidgetAction, widgetItem: WidgetActionItem) => Promise<void>;
  chatkitApiRef: MutableRefObject<Pick<
    UseChatKitReturn,
    "sendCustomAction" | "sendUserMessage" | "setComposerValue"
  > | null>;
  onToggleTheme: () => void;
}) {
  const chatkitOptions = useMemo(
    () => ({
      api: {
        url: CHATKIT_API_URL,
        domainKey: CHATKIT_API_DOMAIN_KEY,
        uploadStrategy: { type: "two_phase" as const },
      },
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
          icon: theme === "dark" ? "light-mode" : "dark-mode",
          onClick: onToggleTheme,
        },
      },
      history: {
        enabled: false,
      },
      startScreen: {
        greeting: "Welcome to 3Vchat, Chat about Anything",
        prompts: [
          {
            label: "B6 anomalies by impact score",
            prompt: "What are the anomalies for B6 today? Give me a distribution by impact score and render it as a chart in the chat.",
            icon: "analytics" as const,
          },
          {
            label: "How does PriceEye work?",
            prompt: "Give me a PDF document explaining how PriceEye works — cover the core concepts, data flow, and key components.",
            icon: "book-open" as const,
          },
          {
            label: "Site issues + past trends",
            prompt: "Give me visualizations on the site issues today, along with past trends over the last 30 days.",
            icon: "chart" as const,
          },
          {
            label: "Smartest 3V customer?",
            prompt: "Which customer is the smartest among 3V's customers? Define your own metrics and justify your ranking.",
            icon: "agent" as const,
          },
          {
            label: "QL2: MySQL config vs Redshift errors",
            prompt: "Look up QL2's site configuration in MySQL (query priceeye.site for QL2) and then check their top collection errors in Redshift (prod.monitoring.provider_combined_audit for today). What do the site settings tell us about the error patterns you see?",
            icon: "bug" as const,
          },
          {
            label: "Top providers bar chart",
            prompt: "Query prod.monitoring.provider_combined_audit for today and get the top 10 providers by request count. Then use Python to plot a horizontal bar chart of the results and publish it as an image card in the chat.",
            icon: "chart" as const,
          },
          {
            label: "How does auto-scheduler work?",
            prompt: "How does the auto-scheduler work in priceeye-scheduling? Look it up in the knowledge base and then check the actual codebase to show me the real class names and entry points.",
            icon: "square-code" as const,
          },
          {
            label: "How fresh is the data?",
            prompt: "What is today's date, and how fresh is the data in the analytics tables? Check the actual latest sales_date available in prod.analytics.market_level_anomalies.",
            icon: "calendar" as const,
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
            id: "gpt-5.4",
            label: "Default",
            description: "gpt-5.4 — most capable",
            default: true,
          },
          {
            id: "gpt-5.4-mini",
            label: "Fast",
            description: "gpt-5.4-mini — fastest",
          },
        ],
      },
    }),
    [onThreadChange, onWidgetAction, theme, onToggleTheme],
  );

  const chatkit = useChatKit(chatkitOptions);
  chatkitApiRef.current = chatkit;

  return <ChatKit control={chatkit.control} className="flex-1 min-h-0 block w-full" />;
}

export function ChatKitPanel() {
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window === "undefined") return "dark";
    return (window.localStorage.getItem(THEME_STORAGE_KEY) as Theme | null) ?? "light";
  });

  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);

  const chatkitApiRef = useRef<
    Pick<UseChatKitReturn, "sendCustomAction" | "sendUserMessage" | "setComposerValue"> | null
  >(null);

  const onThreadChange = useCallback(({ threadId }: { threadId: string | null }) => {
    setCurrentThreadId(threadId);
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

  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden">
      <SessionStateBar threadId={currentThreadId} />
      <ChatKitCore
        key={theme}
        theme={theme}
        onThreadChange={onThreadChange}
        onWidgetAction={onWidgetAction}
        chatkitApiRef={chatkitApiRef}
        onToggleTheme={onToggleTheme}
      />
    </div>
  );
}
