"""Strawberry GraphQL schema for Cognitive OS."""

from __future__ import annotations

import strawberry

from app.graphql.resolvers.cognitive import GqlDecideResult, resolve_cognitive_decide
from app.graphql.resolvers.goals import GqlGoal, resolve_goals
from app.graphql.resolvers.memory import GqlMemory, resolve_search_memory
from app.graphql.subscriptions import MemoryEvent, Subscription


@strawberry.type
class Query:
    @strawberry.field
    async def search_memory(
        self,
        info: strawberry.Info,
        query: str,
        limit: int = 5,
    ) -> list[GqlMemory]:
        return await resolve_search_memory(info, query=query, limit=limit)

    @strawberry.field
    async def cognitive_decide(
        self,
        info: strawberry.Info,
        situation: str,
    ) -> GqlDecideResult:
        return await resolve_cognitive_decide(info, situation=situation)

    @strawberry.field
    async def goals(
        self,
        info: strawberry.Info,
        limit: int = 10,
        status: str | None = None,
    ) -> list[GqlGoal]:
        return await resolve_goals(info, limit=limit, status=status)


schema = strawberry.Schema(query=Query, subscription=Subscription)

__all__ = ["schema", "MemoryEvent"]
