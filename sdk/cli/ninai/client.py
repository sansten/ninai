"""Small HTTP client used by CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class CliConfig:
    api_key: str = ""
    org_id: str = ""
    base_url: str = "http://localhost:8000/api/v1"


class CliClient:
    def __init__(self, config: CliConfig):
        self.config = config
        self._client = httpx.Client(base_url=config.base_url, timeout=30.0)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        if self.config.org_id:
            headers["X-Organization-ID"] = self.config.org_id
        return headers

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self._client.get(path, headers=self._headers(), params=params)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self._client.post(path, headers=self._headers(), json=payload or {})
        resp.raise_for_status()
        return resp.json()

    def patch(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self._client.patch(path, headers=self._headers(), json=payload or {})
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()
