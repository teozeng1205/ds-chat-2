import { ChatKit, useChatKit } from "@openai/chatkit-react";
import { CHATKIT_API_DOMAIN_KEY, CHATKIT_API_URL } from "../lib/config";

export function ChatKitPanel() {
  const chatkit = useChatKit({
    api: { url: CHATKIT_API_URL, domainKey: CHATKIT_API_DOMAIN_KEY },
    composer: {
      // File uploads are disabled for the demo backend.
      attachments: { enabled: false },
      tools: [
      {
        id: "market_anomalies",
        icon: "book-open",
        label: "Market Anomalies",
        placeholderOverride: "Market Anomalies Tool: Summarize market anomalies for a customer",

      },
      {
        id: "internal_monitor",
        icon: "search",
        label: "Internal Monitoring",
        shortLabel: "Monitoring",
        placeholderOverride:
          "Teo's Monitoring Tool: Ask questions about site monitoring issues.",
      },
      {
        id: "knowledge_docs",
        icon: "book-open",
        label: "Knowledge Docs",
        placeholderOverride: "Knowledge Tool: Search docs.zanlit.com",
      },
    ],
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
  });

  return (
    <div className="relative pb-8 flex h-[90vh] w-full rounded-2xl flex-col overflow-hidden bg-white shadow-sm transition-colors dark:bg-slate-900">
      <ChatKit control={chatkit.control} className="block h-full w-full" />
    </div>
  );
}
