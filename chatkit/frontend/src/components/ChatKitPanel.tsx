import {
  ChatKit,
  type UseChatKitReturn,
  type Widgets,
  useChatKit,
} from "@openai/chatkit-react";
import { useCallback, useMemo, useRef, useState } from "react";
import { CHATKIT_API_DOMAIN_KEY, CHATKIT_API_URL } from "../lib/config";

const THREAD_STORAGE_KEY = "ds-chat:last-thread-id";

type WidgetAction = { type: string; payload?: Record<string, unknown> };
type WidgetActionItem = { id: string; widget: Widgets.Card | Widgets.ListView };

export function ChatKitPanel() {
  const [initialThreadId] = useState<string | null>(() => {
    if (typeof window === "undefined") {
      return null;
    }
    return window.localStorage.getItem(THREAD_STORAGE_KEY);
  });

  const chatkitApiRef = useRef<
    Pick<UseChatKitReturn, "sendCustomAction" | "sendUserMessage" | "setComposerValue"> | null
  >(null);

  const onThreadChange = useCallback(({ threadId }: { threadId: string | null }) => {
    if (typeof window === "undefined") {
      return;
    }
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

    await api.sendCustomAction(action, widgetItem.id);
  }, []);

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
            prompt: "what are the top site issues for provider QL2 today",
            icon: "analytics" as const,
          },
          {
            label: "Analytics market anomalies",
            prompt: "what are the anomalies for B6 today",
            icon: "chart" as const,
          },
          {
            label: "Explain codebase architecture",
            prompt: "explain ds-priceeye-analytics repo",
            icon: "square-code" as const,
          },
        ],
      },
      widgets: {
        onAction: onWidgetAction,
      },
      composer: {
        attachments: { enabled: true, maxCount: 5, maxSize: 25 * 1024 * 1024 },
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
    [initialThreadId, onThreadChange, onWidgetAction],
  );

  const chatkit = useChatKit(chatkitOptions);
  chatkitApiRef.current = chatkit;

  return (
    <div className="relative flex h-[90vh] w-full flex-col overflow-hidden rounded-2xl bg-white shadow-sm transition-colors dark:bg-slate-900">
      <ChatKit control={chatkit.control} className="block h-full w-full" />
    </div>
  );
}
