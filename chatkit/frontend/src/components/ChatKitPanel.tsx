import {
  ChatKit,
  type Entity,
  type UseChatKitReturn,
  type Widgets,
  useChatKit,
} from "@openai/chatkit-react";
import { useCallback, useMemo, useRef, useState } from "react";
import { CHATKIT_API_DOMAIN_KEY, CHATKIT_API_URL } from "../lib/config";

const THREAD_STORAGE_KEY = "ds-chat:last-thread-id";
const MAX_TELEMETRY_ROWS = 12;

type TelemetryRow = {
  timestamp: string;
  event: string;
  detail?: string;
};

type WidgetAction = { type: string; payload?: Record<string, unknown> };
type WidgetActionItem = { id: string; widget: Widgets.Card | Widgets.ListView };

const ENTITY_CATALOG: Entity[] = [
  {
    id: "customer:TS",
    title: "TS",
    group: "Customer",
    icon: "user",
    interactive: true,
    data: { kind: "customer", description: "Customer TS anomaly scope." },
  },
  {
    id: "customer:B6",
    title: "B6",
    group: "Customer",
    icon: "user",
    interactive: true,
    data: { kind: "customer", description: "Customer B6 anomaly scope." },
  },
  {
    id: "provider:AI",
    title: "AI",
    group: "Provider",
    icon: "agent",
    interactive: true,
    data: { kind: "provider", description: "Provider AI anomaly scope." },
  },
  {
    id: "site:AI|BW",
    title: "AI|BW",
    group: "Provider Site",
    icon: "map-pin",
    interactive: true,
    data: { kind: "provider_site", description: "Provider AI at site BW." },
  },
  {
    id: "repo:ds-threevictors",
    title: "ds-threevictors",
    group: "Repository",
    icon: "square-code",
    interactive: true,
    data: { kind: "repository", description: "Internal data-service utilities package." },
  },
  {
    id: "repo:ds-chat-2",
    title: "ds-chat-2",
    group: "Repository",
    icon: "square-code",
    interactive: true,
    data: { kind: "repository", description: "Chat frontend/backend orchestration project." },
  },
];

function formatTelemetryDetail(detail: unknown): string | undefined {
  if (detail == null) {
    return undefined;
  }
  if (typeof detail === "string") {
    return detail;
  }
  try {
    return JSON.stringify(detail);
  } catch {
    return "[unserializable detail]";
  }
}

function buildEntityPreview(entity: Entity): Widgets.BasicRoot {
  return {
    type: "Basic",
    direction: "col",
    gap: 2,
    padding: 3,
    children: [
      { type: "Title", value: entity.title, size: "lg" },
      {
        type: "Caption",
        value: `${entity.group ?? "Entity"} • ${entity.id}`,
      },
      {
        type: "Text",
        value: entity.data?.description ?? "No additional metadata available.",
      },
      {
        type: "Badge",
        label: entity.data?.kind ?? "entity",
        color: "info",
        variant: "soft",
      },
    ],
  };
}

export function ChatKitPanel() {
  const [initialThreadId] = useState<string | null>(() => {
    if (typeof window === "undefined") {
      return null;
    }
    return window.localStorage.getItem(THREAD_STORAGE_KEY);
  });
  const [activeThreadId, setActiveThreadId] = useState<string | null>(initialThreadId);
  const [lastThreadId, setLastThreadId] = useState<string | null>(initialThreadId);
  const [actionBusy, setActionBusy] = useState(false);
  const [telemetry, setTelemetry] = useState<TelemetryRow[]>([]);

  const chatkitApiRef = useRef<
    Pick<
      UseChatKitReturn,
      "fetchUpdates" | "sendCustomAction" | "sendUserMessage" | "setComposerValue" | "setThreadId"
    > | null
  >(null);

  const pushTelemetry = useCallback((event: string, detail?: unknown) => {
    const row: TelemetryRow = {
      timestamp: new Date().toLocaleTimeString(),
      event,
      detail: formatTelemetryDetail(detail),
    };
    setTelemetry((previous) => [row, ...previous].slice(0, MAX_TELEMETRY_ROWS));
  }, []);

  const runProgrammaticAction = useCallback(
    async (label: string, action: () => Promise<void>) => {
      setActionBusy(true);
      pushTelemetry("programmatic.start", { label });
      try {
        await action();
        pushTelemetry("programmatic.success", { label });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        pushTelemetry("programmatic.error", { label, message });
      } finally {
        setActionBusy(false);
      }
    },
    [pushTelemetry],
  );

  const onThreadChange = useCallback(
    ({ threadId }: { threadId: string | null }) => {
      setActiveThreadId(threadId);
      if (threadId) {
        setLastThreadId(threadId);
      }
      if (typeof window !== "undefined") {
        if (threadId) {
          window.localStorage.setItem(THREAD_STORAGE_KEY, threadId);
        } else {
          window.localStorage.removeItem(THREAD_STORAGE_KEY);
        }
      }
      pushTelemetry("thread.change", { threadId });
    },
    [pushTelemetry],
  );

  const onEntityClick = useCallback(
    (entity: Entity) => {
      pushTelemetry("entity.click", { id: entity.id, title: entity.title });
      const api = chatkitApiRef.current;
      if (!api) {
        return;
      }
      void api.setComposerValue({
        content: [
          { type: "input_text", text: "Investigate " },
          {
            type: "input_tag",
            id: entity.id,
            text: entity.title,
            group: entity.group,
            data: entity.data,
            interactive: entity.interactive ?? true,
          },
          { type: "input_text", text: " anomalies for today." },
        ],
      });
    },
    [pushTelemetry],
  );

  const onWidgetAction = useCallback(
    async (action: WidgetAction, widgetItem: WidgetActionItem) => {
      pushTelemetry("widget.action", { actionType: action.type, widgetItemId: widgetItem.id });

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

      await api.sendCustomAction(action, widgetItem.id);
    },
    [pushTelemetry],
  );

  const chatkitOptions = useMemo(
    () => ({
      api: { url: CHATKIT_API_URL, domainKey: CHATKIT_API_DOMAIN_KEY },
      initialThread: initialThreadId,
      onResponseStart: () => pushTelemetry("response.start"),
      onResponseEnd: () => pushTelemetry("response.end"),
      onThreadChange,
      onError: ({ error }: { error: Error }) => pushTelemetry("error", error.message),
      onLog: ({
        name,
        data,
      }: {
        name: string;
        data?: Record<string, unknown>;
      }) => pushTelemetry(`log.${name}`, data),
      history: {
        enabled: true,
        showRename: true,
        showDelete: true,
      },
      startScreen: {
        greeting: "What do you want to analyze today?",
        prompts: [
          {
            label: "Internal monitoring anomalies",
            prompt:
              "Check internal monitoring anomalies for today and summarize the top customer/provider/site issues.",
            icon: "analytics" as const,
          },
          {
            label: "Analytics market anomalies",
            prompt:
              "Find major market anomalies for today and explain likely drivers with actionable next steps.",
            icon: "chart" as const,
          },
          {
            label: "Explain codebase architecture",
            prompt:
              "Explain the architecture of the repositories under ~/git and highlight key services and data flow.",
            icon: "square-code" as const,
          },
        ],
      },
      entities: {
        showComposerMenu: true,
        onTagSearch: (query: string) => {
          const normalized = query.trim().toLowerCase();
          if (!normalized) {
            return Promise.resolve(ENTITY_CATALOG.slice(0, 8));
          }
          return Promise.resolve(
            ENTITY_CATALOG.filter((entity) => {
              const searchable = `${entity.id} ${entity.title} ${entity.group ?? ""}`.toLowerCase();
              return searchable.includes(normalized);
            }).slice(0, 8),
          );
        },
        onClick: onEntityClick,
        onRequestPreview: (entity: Entity) =>
          Promise.resolve({
            preview: buildEntityPreview(entity),
          }),
      },
      widgets: {
        onAction: onWidgetAction,
      },
      composer: {
        attachments: { enabled: false },
        placeholder: "Ask about anomalies, monitoring, or codebase questions...",
        models: [
          {
            id: "gpt-5.2",
            label: "Default",
            description: "Default gpt-5.2",
            default: true,
          },
          {
            id: "gpt-5-mini",
            label: "Fast",
            description: "for speed optimized gpt-5-mini",
          },
        ],
      },
    }),
    [initialThreadId, onEntityClick, onThreadChange, onWidgetAction, pushTelemetry],
  );

  const chatkit = useChatKit(chatkitOptions);
  chatkitApiRef.current = chatkit;

  return (
    <div className="relative flex h-[90vh] w-full flex-col overflow-hidden rounded-2xl bg-white shadow-sm transition-colors dark:bg-slate-900">
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 bg-slate-50 px-3 py-2 dark:border-slate-800 dark:bg-slate-950/40">
        <button
          className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-700 disabled:opacity-50 dark:border-slate-700 dark:text-slate-200"
          disabled={actionBusy}
          onClick={() =>
            void runProgrammaticAction("setComposerValue", () =>
              chatkit.setComposerValue({
                text: "Summarize today's internal monitoring anomalies by provider and site.",
              }),
            )
          }
          type="button"
        >
          Prefill Prompt
        </button>
        <button
          className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-700 disabled:opacity-50 dark:border-slate-700 dark:text-slate-200"
          disabled={actionBusy}
          onClick={() =>
            void runProgrammaticAction("sendUserMessage", () =>
              chatkit.sendUserMessage({
                text: "Check internal monitoring anomalies for today and provide top risks.",
                toolChoice: { id: "internal_monitoring" },
              }),
            )
          }
          type="button"
        >
          Send Monitoring Query
        </button>
        <button
          className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-700 disabled:opacity-50 dark:border-slate-700 dark:text-slate-200"
          disabled={actionBusy}
          onClick={() => void runProgrammaticAction("setThreadId(null)", () => chatkit.setThreadId(null))}
          type="button"
        >
          New Thread
        </button>
        <button
          className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-700 disabled:opacity-50 dark:border-slate-700 dark:text-slate-200"
          disabled={actionBusy || !lastThreadId}
          onClick={() =>
            lastThreadId
              ? void runProgrammaticAction("setThreadId(last)", () => chatkit.setThreadId(lastThreadId))
              : undefined
          }
          type="button"
        >
          Reopen Last Thread
        </button>
        <button
          className="rounded-md border border-slate-300 px-2 py-1 text-xs text-slate-700 disabled:opacity-50 dark:border-slate-700 dark:text-slate-200"
          disabled={actionBusy}
          onClick={() => void runProgrammaticAction("fetchUpdates", () => chatkit.fetchUpdates())}
          type="button"
        >
          Sync Updates
        </button>
      </div>

      <ChatKit control={chatkit.control} className="block h-full w-full flex-1" />

      <div className="border-t border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600 dark:border-slate-800 dark:bg-slate-950/40 dark:text-slate-300">
        <div className="mb-1 flex items-center justify-between">
          <span className="font-medium">Telemetry</span>
          <span className="truncate">
            {activeThreadId ? `thread: ${activeThreadId}` : "thread: new"}
          </span>
        </div>
        <ul className="max-h-24 space-y-1 overflow-y-auto">
          {telemetry.length === 0 ? (
            <li className="text-slate-500 dark:text-slate-400">No telemetry events yet.</li>
          ) : (
            telemetry.map((row, index) => (
              <li key={`${row.timestamp}-${row.event}-${index}`} className="truncate">
                <span className="mr-2 text-slate-500 dark:text-slate-400">{row.timestamp}</span>
                <span className="font-medium">{row.event}</span>
                {row.detail ? <span className="ml-2">{row.detail}</span> : null}
              </li>
            ))
          )}
        </ul>
      </div>
    </div>
  );
}
