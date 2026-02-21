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


DEFAULT_PUBLIC_BASE_URL = "http://127.0.0.1:8000"


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
        request = context.get("request") if isinstance(context, dict) else None
        base_url = None
        if request is not None and hasattr(request, "base_url"):
            base_url = str(request.base_url).rstrip("/")
        if not base_url:
            base_url = os.getenv("CHATKIT_PUBLIC_BASE_URL", DEFAULT_PUBLIC_BASE_URL).rstrip("/")
        return f"{base_url}/chatkit/uploads/{quote(attachment_id, safe='')}"

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

