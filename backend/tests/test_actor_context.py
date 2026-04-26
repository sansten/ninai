from __future__ import annotations

from types import SimpleNamespace

from app.api.v1.endpoints.memories import _memory_matches_reader_role
from app.core.actor_context import ActorType, ROLE_RESPONSIBILITIES, normalize_actor_context


def test_normalize_actor_context_defaults_to_anonymous() -> None:
    ctx = normalize_actor_context(
        actor_id=None,
        actor_type=None,
        role=None,
        responsibility=None,
    )

    assert ctx["actor_id"] == "anonymous"
    assert ctx["actor_type"] == ActorType.ANONYMOUS.value
    assert ctx["role"] == "anonymous"
    assert ctx["responsibility"] == ROLE_RESPONSIBILITIES["anonymous"]
    assert ctx["is_anonymous"] is True
    assert ctx["role_specific"] is False


def test_normalize_actor_context_role_specific_for_employee() -> None:
    ctx = normalize_actor_context(
        actor_id="emp-42",
        actor_type=ActorType.EMPLOYEE,
        role="employee_engineering",
        responsibility=None,
    )

    assert ctx["actor_id"] == "emp-42"
    assert ctx["actor_type"] == ActorType.EMPLOYEE.value
    assert ctx["role"] == "employee_engineering"
    assert ctx["responsibility"] == ROLE_RESPONSIBILITIES["employee_engineering"]
    assert ctx["is_anonymous"] is False
    assert ctx["role_specific"] is True


def test_memory_matches_reader_role_for_role_specific_reader() -> None:
    memory = SimpleNamespace(extra_metadata={"write_role": "employee_engineering"})
    reader_ctx = normalize_actor_context(
        actor_id="emp-10",
        actor_type="employee",
        role="employee_engineering",
        responsibility=None,
    )

    assert _memory_matches_reader_role(memory, reader_ctx) is True

    other_reader_ctx = normalize_actor_context(
        actor_id="emp-11",
        actor_type="employee",
        role="employee_support",
        responsibility=None,
    )
    assert _memory_matches_reader_role(memory, other_reader_ctx) is False


def test_memory_matches_reader_role_for_anonymous_reader() -> None:
    memory = SimpleNamespace(extra_metadata={"write_role": "employee_engineering"})
    reader_ctx = normalize_actor_context(
        actor_id=None,
        actor_type="anonymous",
        role=None,
        responsibility=None,
    )

    assert _memory_matches_reader_role(memory, reader_ctx) is True
