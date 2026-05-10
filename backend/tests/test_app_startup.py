from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import app.main as main_module


@pytest.mark.asyncio
async def test_lifespan_ensures_qdrant_collection_when_configured(monkeypatch):
    ensure_collection = AsyncMock()
    create_db_and_tables = AsyncMock()

    monkeypatch.setattr(main_module.settings, "APP_ENV", "development")
    monkeypatch.setattr(main_module.settings, "QDRANT_HOST", "localhost")
    monkeypatch.setattr(main_module.settings, "QDRANT_PORT", 6333)
    monkeypatch.setattr(main_module, "create_db_and_tables", create_db_and_tables)
    monkeypatch.setattr(main_module.QdrantService, "ensure_collection", ensure_collection)
    monkeypatch.setattr(main_module, "engine", SimpleNamespace(dispose=AsyncMock()))

    async with main_module.lifespan(SimpleNamespace()):
        pass

    create_db_and_tables.assert_awaited_once()
    ensure_collection.assert_awaited_once()
