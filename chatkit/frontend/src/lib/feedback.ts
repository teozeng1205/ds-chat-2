// Thin client for POST /chatkit/feedback. Intended for the future
// MessageActions component (Phase 4 UX). Kept as a standalone helper
// so it can be imported without touching the backend endpoint code.

export type FeedbackVerdict = 1 | -1;

export interface FeedbackPayload {
  threadId: string;
  verdict: FeedbackVerdict;
  messageId?: string;
  comment?: string;
}

export interface FeedbackResult {
  ok: boolean;
  id?: number;
  error?: string;
}

export async function postFeedback(payload: FeedbackPayload): Promise<FeedbackResult> {
  try {
    const r = await fetch("/chatkit/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        thread_id: payload.threadId,
        verdict: payload.verdict,
        message_id: payload.messageId,
        comment: payload.comment,
      }),
    });
    if (!r.ok) {
      const text = await r.text().catch(() => "");
      return { ok: false, error: `HTTP ${r.status}${text ? `: ${text}` : ""}` };
    }
    const body = (await r.json()) as { ok?: boolean; id?: number; detail?: string };
    return { ok: body.ok === true, id: body.id, error: body.detail };
  } catch (exc) {
    return { ok: false, error: exc instanceof Error ? exc.message : String(exc) };
  }
}
