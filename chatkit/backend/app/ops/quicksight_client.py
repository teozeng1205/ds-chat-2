"""Read-only QuickSight helpers.

Exposes two operations: list dashboards (with name/name-prefix filter)
and generate an embed URL for anonymous viewing of a single
dashboard. Embedding requires QuickSight Enterprise Edition and the
right IAM; callers should gracefully degrade when neither is set up.

All calls are read-only apart from `GenerateEmbedUrlForAnonymousUser`,
which creates a short-lived signed URL (no persistent side effects).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


ClientFactory = Callable[[str], Any]


def _default_factory() -> ClientFactory:
    def _make(service: str) -> Any:
        import boto3
        return boto3.client(service)

    return _make


def _isoformat(value: Any) -> Optional[str]:
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except Exception:
            return str(value)
    return str(value)


def list_dashboards(
    *,
    aws_account_id: str | None = None,
    name_substring: str | None = None,
    max_results: int = 100,
    client_factory: ClientFactory | None = None,
) -> list[dict[str, Any]]:
    """List QuickSight dashboards. Optionally filter by name substring (case-insensitive)."""
    client = (client_factory or _default_factory())("quicksight")
    account = aws_account_id or _resolve_account_id(client_factory)
    if not account:
        return []
    paginator = client.get_paginator("list_dashboards")
    out: list[dict[str, Any]] = []
    needle = (name_substring or "").lower().strip() or None
    try:
        for page in paginator.paginate(AwsAccountId=account):
            for d in page.get("DashboardSummaryList", []):
                name = d.get("Name") or ""
                if needle and needle not in name.lower():
                    continue
                out.append({
                    "dashboardId": d.get("DashboardId"),
                    "name": name,
                    "arn": d.get("Arn"),
                    "createdTime": _isoformat(d.get("CreatedTime")),
                    "lastUpdatedTime": _isoformat(d.get("LastUpdatedTime")),
                    "publishedVersionNumber": d.get("PublishedVersionNumber"),
                })
                if len(out) >= max_results:
                    return out
    except Exception as exc:  # noqa: BLE001
        log.warning("quicksight list_dashboards failed: %s", exc)
    return out


def generate_anonymous_embed_url(
    dashboard_id: str,
    *,
    aws_account_id: str | None = None,
    session_lifetime_minutes: int = 60,
    namespace: str = "default",
    allowed_domain: str | None = None,
    client_factory: ClientFactory | None = None,
) -> dict[str, Any]:
    """Create a short-lived anonymous embed URL for a dashboard.

    Requires QuickSight Enterprise + IAM permission quicksight:
    GenerateEmbedUrlForAnonymousUser + a capacity-pricing session.
    Returns {ok, embed_url, request_id} or {ok=False, error, error_type}.
    """
    client = (client_factory or _default_factory())("quicksight")
    account = aws_account_id or _resolve_account_id(client_factory)
    if not account:
        return {"ok": False, "error": "unable to resolve AWS account id", "error_type": "MissingAccount"}

    kwargs: dict[str, Any] = {
        "AwsAccountId": account,
        "Namespace": namespace,
        "AuthorizedResourceArns": [_dashboard_arn(account, dashboard_id)],
        "ExperienceConfiguration": {"Dashboard": {"InitialDashboardId": dashboard_id}},
        "SessionLifetimeInMinutes": max(15, min(600, session_lifetime_minutes)),
    }
    if allowed_domain:
        kwargs["AllowedDomains"] = [allowed_domain]
    try:
        resp = client.generate_embed_url_for_anonymous_user(**kwargs)
    except Exception as exc:  # noqa: BLE001
        log.warning("quicksight generate_embed_url_for_anonymous_user failed: %s", exc)
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
    return {
        "ok": True,
        "embed_url": resp.get("EmbedUrl"),
        "request_id": resp.get("RequestId"),
        "anonymous_user_arn": resp.get("AnonymousUserArn"),
    }


def _dashboard_arn(account_id: str, dashboard_id: str) -> str:
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return f"arn:aws:quicksight:{region}:{account_id}:dashboard/{dashboard_id}"


def _resolve_account_id(client_factory: ClientFactory | None) -> str | None:
    """Best-effort STS GetCallerIdentity."""
    try:
        sts = (client_factory or _default_factory())("sts")
        return sts.get_caller_identity().get("Account")
    except Exception as exc:  # noqa: BLE001
        log.warning("unable to resolve AWS account id: %s", exc)
        return None


__all__ = ["list_dashboards", "generate_anonymous_embed_url", "ClientFactory"]
