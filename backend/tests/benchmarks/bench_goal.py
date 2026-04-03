from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.agents.classification_agent import ClassificationAgent
from tests.benchmarks._fixtures import make_goal_samples


async def run(*, mode: str, strategy: str, dataset: str = "synthetic") -> dict[str, Any]:
    agent = ClassificationAgent()
    data = make_goal_samples(n=64, dataset=dataset)

    tool_events: list[dict] = []

    async def _sink(event: dict) -> None:
        tool_events.append(dict(event or {}))

    correct = 0
    scored_correct = 0
    scored_samples = 0
    for content, truth in data:
        before_calls = len([e for e in tool_events if (e.get("payload") or {}).get("tool") == "ollama.generate"])
        before_success = len([e for e in tool_events if (e.get("payload") or {}).get("tool") == "ollama.generate" and (e.get("payload") or {}).get("ok") is True])

        ctx = {
            "tenant": {"org_id": str(uuid4()), "org_slug": None},
            "memory": {"content": content, "classification": "internal"},
            "runtime": {"job_id": str(uuid4())},
            "tool_event_sink": _sink,
        }
        result = await agent.run(str(uuid4()), ctx)
        predicted = str((result.outputs or {}).get("classification", "")).strip().lower()
        if predicted == truth:
            correct += 1

        after_calls = len([e for e in tool_events if (e.get("payload") or {}).get("tool") == "ollama.generate"])
        after_success = len([e for e in tool_events if (e.get("payload") or {}).get("tool") == "ollama.generate" and (e.get("payload") or {}).get("ok") is True])
        llm_attempted = after_calls > before_calls
        llm_succeeded = after_success > before_success

        if strategy == "llm":
            if llm_attempted and llm_succeeded:
                scored_samples += 1
                if predicted == truth:
                    scored_correct += 1
        else:
            scored_samples += 1
            if predicted == truth:
                scored_correct += 1

    llm_tool_calls = 0
    llm_success_calls = 0
    for event in tool_events:
        payload = event.get("payload") or {}
        if payload.get("tool") == "ollama.generate":
            llm_tool_calls += 1
            if payload.get("ok") is True:
                llm_success_calls += 1

    if strategy == "llm" and llm_success_calls == 0:
        raise RuntimeError("llm strategy selected but no successful ollama.generate events observed")

    llm_success_rate = (llm_success_calls / llm_tool_calls) if llm_tool_calls else 0.0
    fallback_rate = 1.0 - llm_success_rate if strategy == "llm" else 0.0
    scored_accuracy = (scored_correct / scored_samples) if scored_samples else 0.0

    return {
        "benchmark": "goal",
        "mode": mode,
        "strategy": strategy,
        "dataset": dataset,
        "accuracy": round(correct / len(data), 4),
        "scored_accuracy": round(scored_accuracy, 4),
        "scored_samples": scored_samples,
        "llm_tool_calls": llm_tool_calls,
        "llm_success_calls": llm_success_calls,
        "llm_success_rate": round(llm_success_rate, 4),
        "fallback_rate": round(fallback_rate, 4),
        "samples": len(data),
        "status": "ok",
    }
