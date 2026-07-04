"""Regression tests: set_tenant_context() interpolated clearance_level into
`SET LOCAL app.current_clearance_level = '{clearance_level}'` without the
escape() applied to every other field, even though the function signature
declares it as int — a non-int value passed at runtime (Python doesn't
enforce type hints) would go straight into the SQL string. int(...) now
enforces the contract at the point of use, matching the string fields'
escape() guard.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.database import set_tenant_context


def _executed_sql(mock_session: AsyncMock) -> list[str]:
    return [str(call.args[0]) for call in mock_session.execute.call_args_list]


@pytest.mark.asyncio
async def test_integer_clearance_level_is_interpolated_unchanged():
    session = AsyncMock()
    with patch("app.services.rls_guard.attach_org_filter"):
        await set_tenant_context(session, "u1", "org1", "member", clearance_level=3)

    sql = _executed_sql(session)
    assert any("current_clearance_level = '3'" in s for s in sql)


@pytest.mark.asyncio
async def test_none_clearance_level_defaults_to_zero():
    session = AsyncMock()
    with patch("app.services.rls_guard.attach_org_filter"):
        await set_tenant_context(session, "u1", "org1", "member", clearance_level=None)

    sql = _executed_sql(session)
    assert any("current_clearance_level = '0'" in s for s in sql)


@pytest.mark.asyncio
async def test_malicious_string_clearance_level_raises_instead_of_injecting():
    """A caller that (against the type hint) passes an attacker-influenced
    string must fail loudly rather than have it land in the SQL text."""
    session = AsyncMock()
    malicious = "0'; DROP TABLE memories; --"

    with patch("app.services.rls_guard.attach_org_filter"):
        with pytest.raises(ValueError):
            await set_tenant_context(session, "u1", "org1", "member", clearance_level=malicious)  # type: ignore[arg-type]

    # The dangerous value must never have reached a SET LOCAL statement.
    sql = _executed_sql(session)
    assert not any("DROP TABLE" in s for s in sql)
