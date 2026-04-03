"""Benchmark run storage.

Persists the output of `tests.benchmarks.run_all` so the admin dashboard
can display COSE scores over time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, Float, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


class BenchmarkRun(Base, UUIDMixin):
    __tablename__ = "benchmark_runs"

    run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="UTC timestamp when the benchmark run completed",
    )
    mode: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset: Mapped[str] = mapped_column(String(64), nullable=False)
    ollama_model: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, doc="Ollama model tag used, null for heuristic runs"
    )
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    composite_score: Mapped[float] = mapped_column(
        Float, nullable=False, doc="quality × reliability composite (0–1)"
    )
    results: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, doc="Full per-benchmark result rows"
    )

    __table_args__ = (Index("ix_benchmark_runs_run_at", "run_at"),)
