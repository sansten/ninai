"""Output formatters for the demo notebook.

Renders enrichment results as:
  - CSR briefing cards (rich markdown)
  - Before/after comparison tables
  - Episode membership tables
  - Anomaly charts (matplotlib)

All functions return strings (markdown or plain text) that Jupyter
renders inline, so the notebook has no dependency on external templates.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# ANSI / Jupyter colour helpers
# ---------------------------------------------------------------------------

_TONE_EMOJI = {
    "urgent": "🚨",
    "cautionary": "⚠️",
    "informational": "ℹ️",
}

_PRIORITY_LABEL = {
    "critical": "🔴 CRITICAL",
    "high": "🟠 HIGH",
    "medium": "🟡 MEDIUM",
    "low": "🟢 LOW",
}


def _prio(p: str) -> str:
    return _PRIORITY_LABEL.get(str(p).lower(), str(p).upper())


# ---------------------------------------------------------------------------
# CSR Briefing Card
# ---------------------------------------------------------------------------

def render_csr_card(
    case_number: int,
    case_title: str,
    hero_record: dict,
    enrichment: dict,
    agent_outputs: dict,
) -> str:
    """Return a markdown CSR briefing card.

    Parameters
    ----------
    case_number:   1-5
    case_title:    Short case title
    hero_record:   The primary demo memory record dict
    enrichment:    Accumulated enrichment dict after all agents ran
    agent_outputs: {agent_name: outputs_dict} for the hero record
    """
    meta = hero_record.get("extra_metadata", {})
    tone = enrichment.get("tone", enrichment.get("narrative_tone", "informational"))
    tone_emoji = _TONE_EMOJI.get(str(tone).lower(), "ℹ️")

    narrative = enrichment.get("narrative_text") or enrichment.get(
        "narrative", "No narrative generated."
    )
    key_entities = enrichment.get("key_entities") or []
    action_items = enrichment.get("action_items") or []
    timeline = enrichment.get("timeline_summary") or ""

    # goal info
    goal_detected = enrichment.get("goal_detected", False)
    blocking = enrichment.get("blocking_subtask", "")
    completion = enrichment.get("completion_fraction")
    pct = f"{int(float(completion) * 100)}%" if completion is not None else "—"

    # playbook
    pb_id = enrichment.get("matched_playbook_id", "")
    pb_conf = enrichment.get("playbook_confidence")
    pb_str = f"`{pb_id}` ({int(float(pb_conf)*100)}% match)" if pb_id and pb_conf else "—"

    # episode
    episode = enrichment.get("episode_label", enrichment.get("episode_id", ""))

    # conflict
    conflict_count = enrichment.get("conflict_count", 0)

    # credibility
    cred = enrichment.get("credibility_score")
    cred_str = f"{float(cred):.2f}" if cred is not None else "—"
    source_tier = enrichment.get("source_tier", "—")

    # uncertainty
    uncertainty = enrichment.get("uncertainty_level", "—")

    lines = [
        f"---",
        f"## {tone_emoji} Case {case_number}: {case_title}",
        f"**Tone:** `{tone.upper()}` &nbsp;&nbsp; **Credibility:** {cred_str} ({source_tier}) &nbsp;&nbsp; **Uncertainty:** {uncertainty}",
        f"",
        f"### Ticket Context",
        f"- **Ticket ID:** `{meta.get('ticket_id', hero_record.get('id', '—'))}`",
        f"- **Priority:** {_prio(meta.get('priority', 'medium'))}",
        f"- **Category:** `{meta.get('category', '—')}`",
        f"- **Team:** `{meta.get('team_slug', '—')}` &nbsp; **Reporter:** {meta.get('user_email', '—')}",
        f"",
    ]

    if key_entities:
        lines += [
            f"### Key Entities",
            f"{', '.join(f'`{e}`' for e in key_entities)}",
            f"",
        ]

    if narrative:
        lines += [
            f"### Ninai Narrative",
            f"> {narrative}",
            f"",
        ]

    if timeline:
        lines += [f"### Timeline", f"{timeline}", f""]

    if episode:
        lines += [f"### Episode", f"`{episode}`", f""]

    if goal_detected:
        lines += [
            f"### Goal Tracking",
            f"- Progress: **{pct}** complete",
        ]
        if blocking:
            lines.append(f"- Blocking subtask: `{blocking}`")
        lines.append("")

    if pb_id:
        lines += [f"### Matched Playbook", f"{pb_str}", f""]

    if conflict_count:
        lines += [
            f"### Conflicts",
            f"**{conflict_count}** conflict(s) detected — cross-reference before resolving.",
            f"",
        ]

    if action_items:
        lines += [f"### Action Items for CSR"]
        for item in action_items:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Before / After comparison table
# ---------------------------------------------------------------------------

def render_comparison_table(
    questions: list[str],
    raw_answers: list[str],
    ninai_answers: list[str],
) -> str:
    """Return a markdown table: question | raw ticket | Ninai.

    Parameters
    ----------
    questions:    List of questions a CSR needs answered
    raw_answers:  What the raw ticket alone can tell you
    ninai_answers: What Ninai surfaces
    """
    header = "| Question | Raw ticket | Ninai |\n|---|---|---|"
    rows = []
    for q, r, n in zip(questions, raw_answers, ninai_answers):
        rows.append(f"| {q} | {r} | **{n}** |")
    return "\n".join([header] + rows)


# ---------------------------------------------------------------------------
# Episode membership table
# ---------------------------------------------------------------------------

def render_episode_table(records: list[dict], enrichments: dict) -> str:
    """Return a markdown table of records and their episode membership."""
    header = "| Record | Team | Days Ago | Episode | Credibility |"
    sep =    "|--------|------|----------|---------|-------------|"
    rows = [header, sep]
    for rec in records:
        local_id = rec["id"]
        meta = rec.get("extra_metadata", {})
        enr = enrichments.get(local_id, {})
        episode = enr.get("episode_label") or enr.get("episode_id") or "—"
        cred = enr.get("credibility_score")
        cred_str = f"{float(cred):.2f}" if cred is not None else "—"
        rows.append(
            f"| `{local_id}` | `{meta.get('team_slug','—')}` "
            f"| {meta.get('days_ago', '—')} "
            f"| {episode} | {cred_str} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Case delta summary (used in epilogue)
# ---------------------------------------------------------------------------

_CASE_QUESTIONS = [
    "Is this a new issue?",
    "Has this person called before?",
    "Did the previous fix work?",
    "Who else is affected?",
    "What is the root cause?",
    "What should the CSR do next?",
]


def render_delta_table(case_ninai_answers: list[str]) -> str:
    """Render a 6-row table for the epilogue comparison.

    Parameters
    ----------
    case_ninai_answers: 6 strings — one answer per question from Ninai
    """
    raw_answers = [
        "Unknown — no memory",
        "No — ticket system has no cross-reference",
        "Unknown — no follow-up tracked",
        "Unknown — tickets are isolated",
        "Not available from this ticket",
        "Re-read the ticket and guess",
    ]
    return render_comparison_table(_CASE_QUESTIONS, raw_answers, case_ninai_answers)


# ---------------------------------------------------------------------------
# Anomaly chart (returns matplotlib Figure)
# ---------------------------------------------------------------------------

def render_anomaly_chart(
    daily_counts: list[int],
    labels: list[str],
    baseline: float,
    anomaly_day_index: int,
    title: str = "Ticket volume vs baseline",
):
    """Return a matplotlib Figure showing the anomaly spike.

    Parameters
    ----------
    daily_counts:       Tickets per day (list of ints, newest last)
    labels:             Day labels (same length as daily_counts)
    baseline:           Average daily count (horizontal line)
    anomaly_day_index:  Index of the anomaly day in daily_counts
    title:              Chart title
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 4))
    colours = [
        "#e74c3c" if i == anomaly_day_index else "#3498db"
        for i in range(len(daily_counts))
    ]
    bars = ax.bar(labels, daily_counts, color=colours, width=0.6)
    ax.axhline(baseline, color="#2ecc71", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline:.0f}/day)")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Tickets")
    ax.set_xlabel("Day")

    # Annotate anomaly bar
    if anomaly_day_index < len(daily_counts):
        y = daily_counts[anomaly_day_index]
        ax.annotate(
            f"ANOMALY\n+{int((y/baseline-1)*100)}% above baseline",
            xy=(anomaly_day_index, y),
            xytext=(anomaly_day_index, y + max(daily_counts) * 0.08),
            ha="center",
            fontsize=9,
            color="#e74c3c",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#e74c3c"),
        )

    normal_patch = mpatches.Patch(color="#3498db", label="Normal day")
    anomaly_patch = mpatches.Patch(color="#e74c3c", label="Anomaly day")
    ax.legend(handles=[normal_patch, anomaly_patch, *ax.get_legend_handles_labels()[0][-1:]])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Slow-boil trend chart
# ---------------------------------------------------------------------------

def render_trend_chart(
    daily_counts: list[int],
    labels: list[str],
    sla_threshold: int,
    title: str = "Ticket trend — slow-boil detection",
):
    """Return a matplotlib Figure showing exponential growth vs SLA threshold."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(labels, daily_counts, marker="o", linewidth=2.5, color="#e67e22", label="Tickets/day")
    ax.axhline(sla_threshold, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"SLA threshold ({sla_threshold}/day)")
    ax.fill_between(range(len(daily_counts)), daily_counts, alpha=0.15, color="#e67e22")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Tickets")
    ax.set_xlabel("Day")
    ax.legend()
    fig.tight_layout()
    return fig
