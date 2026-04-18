"""Local disk attachment store for ChatKit two-phase uploads."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from chatkit.store import AttachmentStore
from chatkit.types import (
    AttachmentCreateParams,
    AttachmentUploadDescriptor,
    FileAttachment,
    ImageAttachment,
)


def default_attachment_dir() -> Path:
    env_path = os.getenv("CHATKIT_ATTACHMENTS_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path(__file__).resolve().parent / "chatkit_attachments").resolve()


class LocalDiskAttachmentStore(AttachmentStore[dict[str, Any]]):
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_attachment(self, attachment_id: str) -> Path:
        return (self.root_dir / f"{attachment_id}.bin").resolve()

    def _build_upload_url(self, attachment_id: str, context: dict[str, Any]) -> str:
        """Always return an absolute URL. ChatKit's pydantic validator rejects
        relative URLs on AttachmentUploadDescriptor, so every branch below
        produces a scheme://host/path string. The order prefers explicit
        configuration, then load-balancer hints, then the request itself,
        then a localhost fallback for internal / test contexts.
        """
        request = context.get("request") if isinstance(context, dict) else None
        attachment_path = f"/chatkit/uploads/{quote(attachment_id, safe='')}"

        explicit_public_base_url = os.getenv("CHATKIT_PUBLIC_BASE_URL")
        if explicit_public_base_url:
            return f"{explicit_public_base_url.rstrip('/')}{attachment_path}"

        if request is not None:
            forwarded_host = request.headers.get("x-forwarded-host")
            if forwarded_host:
                forwarded_proto = request.headers.get("x-forwarded-proto") or request.url.scheme
                return f"{forwarded_proto}://{forwarded_host}{attachment_path}"

            origin = request.headers.get("origin")
            if origin:
                return f"{origin.rstrip('/')}{attachment_path}"

            # Fall back to the request's own scheme + host (always set on a
            # real FastAPI Request).
            scheme = request.url.scheme or "http"
            netloc = request.url.netloc
            if netloc:
                return f"{scheme}://{netloc}{attachment_path}"

        # No request context (internal / test call). Use a localhost default
        # plus optional env override so the URL is always absolute.
        fallback_base = os.getenv("CHATKIT_INTERNAL_BASE_URL", "http://localhost:8000")
        return f"{fallback_base.rstrip('/')}{attachment_path}"

    async def create_attachment(
        self, input: AttachmentCreateParams, context: dict[str, Any]
    ) -> FileAttachment | ImageAttachment:
        attachment_id = self.generate_attachment_id(input.mime_type, context)
        upload_url = self._build_upload_url(attachment_id, context)
        metadata = {
            "local_path": str(self._path_for_attachment(attachment_id)),
            "size": input.size,
        }
        upload_descriptor = AttachmentUploadDescriptor(
            url=upload_url,
            method="PUT",
            headers={},
        )

        if input.mime_type.startswith("image/"):
            return ImageAttachment(
                id=attachment_id,
                name=input.name,
                mime_type=input.mime_type,
                preview_url=upload_url,
                upload_descriptor=upload_descriptor,
                metadata=metadata,
            )

        return FileAttachment(
            id=attachment_id,
            name=input.name,
            mime_type=input.mime_type,
            upload_descriptor=upload_descriptor,
            metadata=metadata,
        )

    async def delete_attachment(self, attachment_id: str, context: dict[str, Any]) -> None:
        path = self._path_for_attachment(attachment_id)
        if path.exists():
            path.unlink()

    async def write_attachment_bytes(self, attachment_id: str, payload: bytes) -> None:
        path = self._path_for_attachment(attachment_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    async def read_attachment_bytes(self, attachment_id: str) -> bytes:
        path = self._path_for_attachment(attachment_id)
        return path.read_bytes()
