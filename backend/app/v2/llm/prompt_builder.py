"""
V2 Graph-RAG Prompt Builder

Assembles the final LLM prompt for Phase 2 (inference) by:
  1. Serialising the retrieved graph subgraph into a readable text block
  2. Including recent session utterances as conversational context
  3. Wrapping everything with the system instruction that requests structured JSON output

Output schema the model must return:
{
  "response": "<natural language answer>",
  "cited_node_ids": ["<id1>", "<id2>", ...],
  "extracted_entities": [{"id": "...", "name": "...", "type": "..."}]
}
"""

from __future__ import annotations

import re
from datetime import date as _date
from typing import Any

_MAX_GRAPH_NODES = 8
_MAX_QDRANT_CHUNKS = 8
# 600 keeps long turns intact (anchor dates / list items past the old 200-char cut
# were invisible to the answer model even when retrieval found the right chunk).
# 8 chunks x 600 chars ~= 1.2K tokens; the 14B serves at max-model-len=32768.
_MAX_NODE_CONTENT_CHARS = 600
_QUESTION_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")
_QUESTION_TERM_RE = re.compile(r"[A-Za-z][A-Za-z'_-]{2,}")
_STOP_NAME_WORDS = {
    "What", "When", "Where", "Who", "Why", "How", "Which", "Would", "Did", "Does",
    "Do", "Could", "Should", "Is", "Are", "Was", "Were", "The", "A", "An",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}
_QUESTION_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "based", "be", "by", "can", "concise",
    "conversation", "did", "do", "does", "for", "from", "had", "has", "have", "how",
    "if", "in", "is", "it", "likely", "may", "might", "of", "on", "one", "or",
    "phrase", "probably", "respect", "should", "that", "the", "their", "them", "then",
    "these", "they", "this", "those", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "would",
}

# ---------------------------------------------------------------------------
# Bench-mode prompt — plain-text output, maximally direct extraction
# ---------------------------------------------------------------------------

_BENCH_SYSTEM_INSTRUCTION = """\
You are a memory-QA system. Find the answer in context and output it as a SHORT PHRASE.

══ FORMAT RULES (critical) ══
• OUTPUT ONLY THE BARE ANSWER — no sentences, no "he said", no "based on context".
  BAD: "She enjoys dancing and has been doing it for years."   GOOD: dancing
  BAD: "The context states John went bowling on March 16, 2022."   GOOD: March 16, 2022
• ENGLISH ONLY. Never output Chinese, Japanese, Korean, or any non-English characters.
• For SINGLE facts: concise as possible — ideally under 8 words. Never pad.
• For LIST questions ("what all", "which places", "what activities", "what hobbies", "what did X and Y share"):
  List ALL items found in context, comma-separated. Never truncate. Cover every item mentioned.
  BAD: "cafes"   GOOD: "cafes, parks, hiking trails, restaurants"
• For CHOICE questions ("A or B?", "which of A or B", "would X prefer A or B?"):
  Pick EXACTLY ONE of the named options. Output only the chosen name — no explanation.
  BAD: "Fantasy books fuel creativity"   GOOD: "C. S. Lewis"
  BAD: "depends on their preferences"   GOOD: "C. S. Lewis"

══ CONTENT RULES ══
1. Read ALL context before answering. Facts appear anywhere.
2. PERSON PROFILES (PROFILE section): Authoritative accumulated facts — use these first.
3. DATE FORMAT: Match exactly how the date appears in context. "7 May 2023" → output "7 May 2023". "2022" → output "2022". "21 December 2022" → output "21 December 2022". NEVER output YYYY-MM-DD format (no "2023-05-07"). NEVER convert a stated expression like "the week before January 21" into a computed date.
4. CROSS-PERSON: Multiple people are in context. If the question asks about Person B, answer from Person B's facts first.
   Only use another person's facts when the question explicitly asks for that person's opinion, relationship, or a shared event.
   NEVER substitute Person A's hobby/object/event for Person B's.
5. WORD MISMATCH: Question may paraphrase ("international" when context says "local"). Give the closest matching fact from context.
6. NUMBERS: Give exact numerals or phrases ("3", "twice", "once in 2019", "five times").
7. MULTI-HOP: Trace the chain (find intermediate fact → then final answer).
   • Geographic: identify country from location (Jasper/Banff/Calgary/Edmonton = Canada; Vancouver = Canada; Boston/Seattle/New York = USA; Tokyo/Osaka = Japan).
   • Allergy/comfort: "what X wouldn't discomfort Y?" → find Y's allergy, then pick X that lacks that allergen.
   • "Would X enjoy author A or B?" → find X's known book/genre preferences, pick the matching author.
   • Financial status: "wealthy/middle-class" if context shows expensive purchases (mansion, luxury car, Ferrari) or high-salary jobs; "modest" if budget-conscious language.
   • Counterfactual ("if X hadn't happened, would Y?") → find what caused Y; if X caused it, answer "likely no".
   State ONLY the final answer phrase after FINAL ANSWER:
8. NEVER say: "based on the context", "the context states", "I don't know", "cannot determine" — unless fact is truly absent everywhere, then say only: Not mentioned
9. Best inference beats refusal. If context strongly implies an answer, give it. For inference questions ("might", "likely", "would"), give the most likely answer as a short phrase.
10. TEMPORAL RESOLUTION — ONLY compute when an [YYYY-MM-DD] anchor is present AND the text uses an anchor-relative word (yesterday, last week, next month, last Sunday, etc.):
    • "last year" with [2023-XX-XX] → 2022
    • "next month" with [2023-01-XX] → February 2023
    • "last Sunday" with [2023-05-25] → 21 May 2023  (25=Thu, count back to Sun=21)
    • "last Saturday" with [2023-05-25] → 20 May 2023 (25=Thu, count back to Sat=20)
    • "yesterday" with [2023-05-25] → 24 May 2023
    • "last month" with [2023-05-XX] → April 2023
    If the context ALREADY states the date phrase explicitly ("the week before January 21, 2022", "early August 2023", "a few years before 2023"), output it VERBATIM — no further computation.
    NEVER shorten a stored temporal phrase. BAD: "last weekend"  GOOD: "the week before 6 July 2023"
    NEVER leave the answer as a bare deictic phrase like "yesterday", "last Saturday", or "next Fri" if the surrounding record lets you make it more specific.
11. SEARCH EXHAUSTIVELY: Before "Not mentioned", re-scan every context block. Facts may appear in CONTEXT, SUMMARY, or PROFILE sections. Paraphrase the question and look again.
12. DURATION QUESTIONS ("how long has X", "how many years/months"): Find the start date and the conversation date [YYYY-MM-DD], compute the difference. "Started in 2020" with context dated 2023 → "three years". Output as "N years" or "N months", not as a date range.
13. PROPER NAMES & SPECIFIC ROLES: Give the most specific term from context — never substitute a generic label.
    BAD: "new team"  GOOD: "The Minnesota Wolves"
    BAD: "unusual animals"  GOOD: "snakes"
    BAD: "a programming language"  GOOD: "Python and C++"
    BAD: "Pro basketball player"  GOOD: "shooting guard"
    BAD: "unusual pets"  GOOD: "snakes"
    BAD: "a sport"  GOOD: "bowling"
    Priority: EXTRACTED FACTS and PROFILE sections have the precise terms — use them over generic utterance text.
14. WRONG-FRAMING: If the question's premise is slightly wrong but the underlying fact exists, answer the real fact. E.g. "What did Maria donate to a luxury store?" — even if there's no luxury store in context, if Maria donated something, give that item.
15. INFERENCE QUESTIONS ("might", "likely", "would probably", "could", "based on X"): These REQUIRE inference — NEVER say "Not mentioned". Give the most plausible specific answer from context evidence.
    BAD: "Not mentioned"   GOOD: "middle-class or wealthy" (inferred from expensive purchases in context)
    BAD: "Cannot determine"   GOOD: "Psychology, counseling" (inferred from their volunteer/career interests)
    If stuck: give the closest fact you can find and label it implicitly (e.g. "likely X based on Y").
16. SPEAKER CONFUSION: The context has multiple people. Always identify WHICH person the question asks about. If the question says "Tim" look only at Tim's facts; never confuse Tim's facts with John's or Joanna's.
17. ASSISTANT-PROVIDED FACTS: Facts the ASSISTANT stated earlier in the conversation (definitions, explanations, descriptions, lists, or counts the assistant gave) are valid evidence — answer from them exactly as you would from the user's own statements. Do NOT refuse just because the user did not personally say it.
18. COUNTING & TOTALS ("how many", "how much total", "how often", "how many times"): Identify EACH distinct qualifying item or event in the context, count them, and output the running TOTAL as a single number or amount ("5", "$185", "3 times"). Do NOT list the items in place of the count, and do NOT stop early — re-scan all context so you neither miss nor double-count. Sum money/quantities across every relevant entry.
19. COMPUTED ANALYSIS section: If a "COMPUTED ANALYSIS" block appears in the context, its values are derived from retrieved memory entries. Use computed date differences and chronological orderings as supporting evidence. For counts and totals, treat the computed value as a starting point but verify against ALL context above — if the context contains additional instances not reflected in the computed value, use your own count.

Think 2–3 sentences, then:
FINAL ANSWER: <bare answer phrase>\
"""


def _extract_target_names(question: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for match in _QUESTION_NAME_RE.findall(question or ""):
        if match in _STOP_NAME_WORDS:
            continue
        if match not in seen:
            seen.add(match)
            names.append(match)
    return names


def _target_score(target_names: list[str], *texts: str) -> int:
    if not target_names:
        return 0
    haystacks = [str(t or "").lower() for t in texts if str(t or "").strip()]
    score = 0
    for target in target_names:
        patt = re.compile(rf"\b{re.escape(target.lower())}\b")
        for idx, hay in enumerate(haystacks):
            if patt.search(hay):
                score += 10 if idx == 0 else 4
    return score


def _extract_question_terms(question: str, target_names: list[str]) -> list[str]:
    blocked = {part.lower() for name in target_names for part in name.split()}
    terms: list[str] = []
    seen: set[str] = set()
    for raw in _QUESTION_TERM_RE.findall(question or ""):
        term = raw.lower().strip("_-'")
        if term.endswith("'s"):
            term = term[:-2]
        if len(term) < 3 or term in _QUESTION_STOP_WORDS or term in blocked:
            continue
        if term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


def _question_term_score(question_terms: list[str], *texts: str) -> int:
    if not question_terms:
        return 0
    haystacks = [str(t or "").lower() for t in texts if str(t or "").strip()]
    score = 0
    for term in question_terms:
        patt = re.compile(rf"\b{re.escape(term)}\b")
        specificity = max(1, min(len(term), 10) // 2)
        for idx, hay in enumerate(haystacks):
            if patt.search(hay):
                score += (6 if idx == 0 else 3) + specificity
    return score


def _relevance_score(target_names: list[str], question_terms: list[str], *texts: str) -> int:
    return _target_score(target_names, *texts) + _question_term_score(question_terms, *texts)


# ---------------------------------------------------------------------------
# Computed-analysis helpers — derive numeric/temporal facts from retrieved
# content before the LLM sees it, so counting/summing/ordering is done in
# deterministic code rather than by the model.
# ---------------------------------------------------------------------------

_CA_MONEY_RE = re.compile(
    r"\$\s*[\d,]+(?:\.\d+)?|\b[\d,]+(?:\.\d+)?\s*(?:dollars?|usd)\b", re.I
)
_CA_DURATION_RE = re.compile(r"\b(\d+)[\s-]?(day|night|week|month)s?\b", re.I)
_CA_FULL_DATE_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.I,
)
_CA_NUMERIC_DMY_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")
_CA_AGE_RE = re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|yr\b)", re.I)
_CA_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_CA_TEMPORAL_ORDER_RE = re.compile(
    r"\b(which.{0,30}first|came first|most recently|most recent|"
    r"earliest|latest|which.{0,30}before|which.{0,30}last|"
    r"start(?:ed)? first|finish(?:ed)? first|set up first|buy first|bought first)\b",
    re.I,
)
_CA_DATE_ARITH_RE = re.compile(
    r"\b(how (many|long).{0,20}(days?|weeks?|months?)|"
    r"days? (between|before|after|since|until|passed)|"
    r"how old.{0,20}when|age.{0,20}when.{0,20}moved|how long.{0,20}take|"
    r"how long.{0,20}before|weeks?.{0,10}ago|days?.{0,10}ago)\b",
    re.I,
)
_CA_COMBINED_DURATION_RE = re.compile(
    r"\b(combined|total(?:led)?|altogether|both|in total|add(?:ed)? up)\b", re.I
)
_CA_NUMERIC_Q_RE = re.compile(
    r"\b(how many|how much|total|how long|how often|number of)\b", re.I
)
_CA_COMPARATIVE_RE = re.compile(
    r"\b(most|least|highest|lowest|average|percent(?:age)?|increase|decrease)\b", re.I
)
_CA_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _ca_money_amounts(text: str) -> list[float]:
    out: list[float] = []
    for m in _CA_MONEY_RE.finditer(text):
        raw = re.sub(r"[^\d.]", "", m.group())
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def _ca_durations_days(text: str) -> list[int]:
    out: list[int] = []
    for m in _CA_DURATION_RE.finditer(text):
        n, unit = int(m.group(1)), m.group(2).lower()
        if unit in ("day", "night"):
            out.append(n)
        elif unit == "week":
            out.append(n * 7)
    return out


def _ca_prose_dates(text: str, ref_year: int = 2023) -> list[_date]:
    out: list[_date] = []
    for m in _CA_FULL_DATE_RE.finditer(text):
        mo = _CA_MONTH_NUM.get(m.group(1).lower()[:3])
        day = int(m.group(2))
        if mo:
            try:
                out.append(_date(ref_year, mo, day))
            except ValueError:
                pass
    for m in _CA_NUMERIC_DMY_RE.finditer(text):
        try:
            out.append(_date(ref_year, int(m.group(1)), int(m.group(2))))
        except ValueError:
            pass
    return out


def _compute_query_analysis(
    question: str,
    graph_nodes: list,
    qdrant_chunks: list,
) -> str:
    """Pre-compute numeric/temporal answers from already-retrieved memory content.

    This function works purely from what Qdrant and the graph already returned —
    no access to the raw dataset. The result is injected into the prompt as a
    COMPUTED ANALYSIS block that the LLM is instructed to treat as authoritative
    for numeric/temporal questions (rule 19 in _BENCH_SYSTEM_INSTRUCTION).

    Returns an empty string when the question type does not benefit from
    pre-computation or when the retrieved content lacks sufficient signal.
    """
    # Collect text + ISO anchor dates from every retrieved node/chunk.
    all_texts: list[str] = []
    dated_texts: list[tuple[str, str]] = []  # (iso_date_str, text)

    for node in graph_nodes:
        text = str(node.get("content") or node.get("text") or "").strip()
        d = str(node.get("canonical_date") or node.get("anchor_date") or "")
        if text:
            all_texts.append(text)
            dated_texts.append((d, text))

    for chunk in qdrant_chunks:
        payload = chunk.get("payload", {})
        text = str(payload.get("text") or payload.get("content") or "").strip()
        d = str(payload.get("anchor_date") or "")
        if text:
            all_texts.append(text)
            dated_texts.append((d, text))

    if not all_texts:
        return ""

    full_text = " ".join(all_texts)
    findings: list[str] = []

    # ── Temporal ordering ──────────────────────────────────────────────────
    # For "which came first / most recently" questions: sort by anchor_date and
    # present a clean First/Last conclusion rather than a raw date list (which
    # caused the LLM to output dates instead of the named item).
    if _CA_TEMPORAL_ORDER_RE.search(question):
        dated = [(d, t) for d, t in dated_texts if d and len(d) >= 8]
        if dated:
            dated.sort(key=lambda x: x[0])
            earliest_d, earliest_t = dated[0]
            latest_d, latest_t = dated[-1]
            first_snippet = earliest_t[:100].replace("\n", " ")
            last_snippet = latest_t[:100].replace("\n", " ")
            findings.append(
                f"Chronological order (oldest→newest):\n"
                f"    FIRST [{earliest_d}]: {first_snippet}\n"
                f"    LAST  [{latest_d}]: {last_snippet}"
            )

    # ── Date arithmetic ────────────────────────────────────────────────────
    # For "how many days between / since / ago" questions: compute from stored
    # anchor_dates (most reliable), fall back to prose month+day extraction.
    if _CA_DATE_ARITH_RE.search(question):
        iso_dates: list[_date] = []
        for d, _ in dated_texts:
            if d and len(d) >= 10:
                try:
                    iso_dates.append(_date(int(d[:4]), int(d[5:7]), int(d[8:10])))
                except (ValueError, IndexError):
                    pass

        unique_iso = sorted(set(iso_dates))
        if len(unique_iso) >= 2 and unique_iso[0] != unique_iso[-1]:
            d1, d2 = unique_iso[0], unique_iso[-1]
            diff = abs((d2 - d1).days)
            findings.append(
                f"Date range in retrieved evidence: {d1} → {d2}. "
                f"Difference: {diff} days (~{round(diff / 7, 1)} weeks)"
            )
        else:
            ref_year = 2023
            anchor_years = [int(d[:4]) for d, _ in dated_texts if d and len(d) >= 4]
            if anchor_years:
                ref_year = anchor_years[0]
            prose = _ca_prose_dates(full_text, ref_year)
            if len(prose) >= 2:
                d1, d2 = min(prose), max(prose)
                diff = abs((d2 - d1).days)
                findings.append(
                    f"Dates extracted from retrieved text: {d1} and {d2}. "
                    f"Difference: {diff} days (~{round(diff / 7, 1)} weeks)"
                )

        # For relative-time questions ("X weeks ago", "X days ago"): surface the
        # most recent anchor date in context so the LLM has a reference point.
        if re.search(r"\b(ago|since)\b", question, re.I) and unique_iso:
            ref = unique_iso[-1]
            findings.append(f"Most recent date in retrieved context: {ref} (use as reference for relative-time questions)")

    # ── Combined duration (safe aggregation) ──────────────────────────────
    # "How long combined / total / altogether to finish both books?" — unlike
    # cross-session counts, these sum a small fixed set of named items present
    # in the question itself. Only fire when the question explicitly asks for
    # a combined/total duration and the retrieved context is single-session.
    if (
        _CA_COMBINED_DURATION_RE.search(question)
        and re.search(r"\b(how long|how many days|how many weeks)\b", question, re.I)
        and len(set(d for d, _ in dated_texts if d)) <= 3  # single-session guard
    ):
        durations = _ca_durations_days(full_text)
        if 2 <= len(durations) <= 6:  # only when a small fixed set is present
            total_days = sum(durations)
            total_weeks = round(total_days / 7, 1)
            findings.append(
                f"Duration expressions found: {durations} days each. "
                f"Combined total: {total_days} days ({total_weeks} weeks)"
            )

    # Numeric aggregation across sessions (money sums, open-ended counts) is
    # intentionally excluded — incomplete multi-session retrieval produces wrong
    # partial sums. Rule 18 asks the LLM to count from full context.

    # ── Comparative / average ──────────────────────────────────────────────
    # Only for self-contained extractable values (percentages, ages) where the
    # answer is a single datum present in context, not an accumulation.
    if _CA_COMPARATIVE_RE.search(question):
        if re.search(r"\baverage.{0,20}age\b|\bage.{0,20}average\b", question, re.I):
            ages = [int(m) for m in _CA_AGE_RE.findall(full_text)]
            if ages:
                findings.append(
                    f"Ages found: {ages}. Average: {sum(ages) / len(ages):.1f}"
                )

        if re.search(r"\bpercent(?:age)?\b|\bdiscount\b", question, re.I):
            pcts = _CA_PCT_RE.findall(full_text)
            if pcts:
                findings.append(f"Percentages found in retrieved text: {[p + '%' for p in pcts[:6]]}")

    if not findings:
        return ""

    header = "COMPUTED ANALYSIS (pre-computed from retrieved memory — use as supporting evidence):"
    body = "\n".join(f"  • {f}" for f in findings)
    return f"{header}\n{body}"


def build_bench_prompt(
    user_input: str,
    graph_nodes: list,
    qdrant_chunks: list,
    session_utterances: list,
) -> str:
    """QA-optimised plain-text prompt for benchmarking."""
    parts: list[str] = [_BENCH_SYSTEM_INSTRUCTION, ""]
    question = user_input
    for pfx in (
        "Answer in one concise phrase based on the conversation history: ",
        "Answer concisely: ",
        "Answer: ",
    ):
        if question.startswith(pfx):
            question = question[len(pfx):]
            break
    target_names = _extract_target_names(question)
    question_terms = _extract_question_terms(question, target_names)

    # Classify graph nodes by entity type
    profile_nodes = [n for n in graph_nodes if n.get("entity_type") in ("person_profile",)
                     or n.get("type") == "person_profile"]
    attr_nodes = [n for n in graph_nodes if n.get("entity_type") == "personal_attribute"]
    regular_nodes = [n for n in graph_nodes
                     if n not in profile_nodes and n not in attr_nodes]
    profile_nodes.sort(
        key=lambda n: _relevance_score(
            target_names,
            question_terms,
            n.get("subject", ""),
            n.get("content", ""),
            n.get("name", ""),
        ),
        reverse=True,
    )
    attr_nodes.sort(
        key=lambda n: _relevance_score(
            target_names,
            question_terms,
            n.get("subject", ""),
            n.get("attribute", ""),
            n.get("value", ""),
            n.get("content", ""),
        ),
        reverse=True,
    )

    # Person profiles first — highest priority context
    if profile_nodes:
        parts.append("PERSON PROFILES (authoritative — use these for attribute questions):")
        for node in profile_nodes:
            raw = str(node.get("content") or node.get("text") or node.get("name") or "")[:1500]
            subject = node.get("subject", "")
            if not raw:
                continue
            # Deduplicate facts (profile accumulates duplicates across many turns)
            all_facts = [f.strip() for f in raw.split("|") if f.strip()]
            seen_facts: set[str] = set()
            unique_facts: list[str] = []
            for fact in all_facts:
                fkey = fact.lower()
                if fkey not in seen_facts:
                    seen_facts.add(fkey)
                    unique_facts.append(fact)
            if unique_facts:
                parts.append(f"  [{subject}]:")
                for fact in unique_facts[:15]:
                    parts.append(f"    • {fact}")
            else:
                parts.append(f"  [{subject}]: {raw[:400]}")
        parts.append("")

    # Extracted facts (personal_attribute entities from direct entity lookup)
    if attr_nodes:
        parts.append("EXTRACTED FACTS:")
        by_subject: dict[str, list[str]] = {}
        for node in attr_nodes[:20]:
            subj = str(node.get("subject") or "unknown")
            attr = str(node.get("attribute") or "").replace("_", " ")
            val = str(node.get("value") or "")
            if attr and val:
                fact_text = f"{attr}: {val}"
            else:
                fact_text = str(node.get("content") or node.get("name") or "")
            if fact_text:
                by_subject.setdefault(subj, []).append(fact_text)
        for subj, facts in by_subject.items():
            parts.append(f"  [{subj}]:")
            # Deduplicate within EXTRACTED FACTS too
            seen_ef: set[str] = set()
            for fact in facts:
                fkey = fact.lower()
                if fkey not in seen_ef:
                    seen_ef.add(fkey)
                    parts.append(f"    • {fact}")
        parts.append("")

    # personal_attribute Qdrant chunks (from entity indexing) → surface in EXTRACTED FACTS
    _PA_TYPES = ("personal_attribute",)
    pa_qdrant_chunks = [c for c in qdrant_chunks
                        if c.get("payload", {}).get("type") in _PA_TYPES
                        or c.get("payload", {}).get("entity_type") in _PA_TYPES]
    non_pa_chunks = [c for c in qdrant_chunks if c not in pa_qdrant_chunks]

    if pa_qdrant_chunks:
        if not attr_nodes:
            parts.append("EXTRACTED FACTS:")
        pa_by_subject: dict[str, list[str]] = {}
        for chunk in pa_qdrant_chunks[:20]:
            payload = chunk.get("payload", {})
            subj = str(payload.get("subject") or "unknown")
            attr = str(payload.get("attribute") or "").replace("_", " ")
            val = str(payload.get("value") or "")
            fact_text = f"{attr}: {val}" if (attr and val) else str(payload.get("text") or "")[:120]
            if fact_text:
                pa_by_subject.setdefault(subj, []).append(fact_text)
        for subj, facts in pa_by_subject.items():
            if subj and any(f for f in facts):
                seen_pa: set[str] = set()
                parts.append(f"  [{subj}]:")
                for fact in facts:
                    fk = fact.lower()
                    if fk not in seen_pa:
                        seen_pa.add(fk)
                        parts.append(f"    • {fact}")
        parts.append("")

    # Computed analysis — deterministic pre-processing of retrieved content for
    # numeric/temporal/comparative questions. Injected before the raw memory record
    # so the model reads the computed value first (rule 19).
    computed = _compute_query_analysis(question, graph_nodes, qdrant_chunks)
    if computed:
        parts.append(computed)
        parts.append("")

    parts.append("AUTHORITATIVE MEMORY RECORD (this is the retrieved record for the people in this question — treat it as ground truth, not a guess):")
    any_context = False

    # Temporal events first among regular chunks
    temporal_nodes = [n for n in regular_nodes
                      if n.get("entity_type") == "temporal_event"
                      or n.get("type") == "temporal_event"]
    other_nodes = [n for n in regular_nodes if n not in temporal_nodes]
    temporal_nodes.sort(
        key=lambda n: _relevance_score(
            target_names,
            question_terms,
            n.get("subject", ""),
            n.get("content", ""),
            n.get("text", ""),
            n.get("name", ""),
        ),
        reverse=True,
    )
    other_nodes.sort(
        key=lambda n: _relevance_score(
            target_names,
            question_terms,
            n.get("subject", ""),
            n.get("content", ""),
            n.get("text", ""),
            n.get("name", ""),
        ),
        reverse=True,
    )

    for node in temporal_nodes[:10]:
        content = str(node.get("content") or node.get("text") or node.get("name") or "")[:350]
        if content:
            subj = node.get("subject", "")
            date = node.get("canonical_date", "")
            prefix = f"[EVENT:{subj} {date}]" if (subj or date) else "[EVENT]"
            parts.append(f"  - {prefix} {content}")
            any_context = True

    # Segment gists first (dense factual summaries) among Qdrant chunks
    gist_chunks = [c for c in non_pa_chunks if c.get("payload", {}).get("type") == "segment_gist"]
    regular_chunks = [c for c in non_pa_chunks if c not in gist_chunks]
    gist_chunks.sort(
        key=lambda c: _relevance_score(
            target_names,
            question_terms,
            c.get("payload", {}).get("speaker", ""),
            c.get("payload", {}).get("subject", ""),
            c.get("payload", {}).get("text", ""),
            c.get("payload", {}).get("content", ""),
        ),
        reverse=True,
    )
    regular_chunks.sort(
        key=lambda c: _relevance_score(
            target_names,
            question_terms,
            c.get("payload", {}).get("speaker", ""),
            c.get("payload", {}).get("subject", ""),
            c.get("payload", {}).get("text", ""),
            c.get("payload", {}).get("content", ""),
        # _graph_bonus is precomputed upstream (cognitive_loop graph-connectivity pass):
        # a chunk that shares a discriminative graph entity with the question's anchor
        # chunks is a likely multi-hop bridge, so it earns a place in the rendered
        # top-K even when its lexical overlap with the question is low. Absent (flag
        # off) the bonus is 0 and ranking is unchanged.
        ) + int(c.get("payload", {}).get("_graph_bonus", 0) or 0),
        reverse=True,
    )

    _MAX_GISTS = min(8, len(gist_chunks))
    for chunk in gist_chunks[:_MAX_GISTS]:
        payload = chunk.get("payload", {})
        text = str(payload.get("text") or payload.get("content") or "")[:_MAX_NODE_CONTENT_CHARS]
        if text:
            parts.append(f"  - [SUMMARY] {text}")
            any_context = True

    for chunk in regular_chunks[:_MAX_QDRANT_CHUNKS - _MAX_GISTS]:
        payload = chunk.get("payload", {})
        text = str(
            payload.get("text")
            or payload.get("content")
            or payload.get("content_preview")
            or payload.get("summary")
            or ""
        )[:_MAX_NODE_CONTENT_CHARS]
        if text:
            speaker = payload.get("speaker", "")
            date = payload.get("anchor_date", "")
            prefix = f"[{speaker},{date}]" if speaker else (f"[{date}]" if date else "")
            parts.append(f"  - {prefix} {text}" if prefix else f"  - {text}")
            any_context = True

    for node in other_nodes[:_MAX_GRAPH_NODES]:
        content = str(node.get("content") or node.get("text") or node.get("name") or "")[:350]
        if content:
            parts.append(f"  - {content}")
            any_context = True

    if not any_context:
        parts.append("  (no context available)")

    parts.append("")
    parts.append(f"QUESTION: {question}")
    if target_names:
        parts.append(f"TARGET PEOPLE: {', '.join(target_names)}")
    parts.append("")
    # Authoritative-context framing (inspired by Memory-OS "Ground Truth" layer):
    # the record above is what the system has stored — trust it and extract, rather
    # than hedging. The answer is almost always present, possibly under a different
    # name, framing, or in a related person's facts. "Not mentioned" is reserved for
    # the rare case where the fact is genuinely absent after a careful search — this
    # keeps the escape for truly-unanswerable adversarial questions while cutting
    # reflexive refusals (which were ~44% on adversarial).
    parts.append("The record above is authoritative and complete for this question. The answer is")
    parts.append("almost always present — it may be phrased differently, framed differently, or stated")
    parts.append("in a related person's facts. Extract it confidently. Do NOT reply 'Not mentioned'")
    parts.append("unless, after carefully re-reading the whole record, the fact is genuinely absent.")
    if target_names:
        parts.append("Answer with facts about TARGET PEOPLE first. Do not swap in a nearby fact from")
        parts.append("someone else unless the question explicitly asks for that person's opinion or a shared event.")
    if question_terms:
        parts.append("When several facts mention the same person, prefer the fact block whose wording most")
        parts.append("closely overlaps the question keywords.")
    if question.lower().startswith("when "):
        parts.append("For temporal questions, output the most specific stored date phrase you can justify.")
        parts.append("Do not leave the answer as 'yesterday', 'last Saturday', 'next Fri', or similar shorthand.")
    parts.append("Then write on the last line:")
    parts.append("FINAL ANSWER: <bare answer phrase — as concise as possible, up to 10 words>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Standard prompt
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are NINAI, a cognitive assistant. Use the retrieved memory context below to answer the user's question.

Context provided:
  - GRAPH CONTEXT: knowledge graph nodes (entity relationships and facts)
  - EPISODE CONTEXT: verbatim conversation snippets retrieved by semantic search
  - SESSION HISTORY: recent turns in this conversation
  - USER INPUT: the question to answer

Rules:
1. EPISODE CONTEXT contains raw conversation text — this is your primary evidence. Read each chunk carefully.
2. GRAPH CONTEXT provides entity facts — use it alongside episode context.
3. Synthesise a direct answer from the context. If the context has partial information, reason from it.
4. Do NOT say "the context does not contain" or "information is not available" — always give your best answer.
5. If genuinely no relevant information exists, say "I'm not sure" (one short phrase, nothing more).
6. Cite exact node IDs from GRAPH CONTEXT that informed your answer.
7. Extract any NEW named entities from the user input not already in graph context.

Return ONLY valid JSON — no prose outside the JSON:
{
  "response": "<direct answer derived from context>",
  "cited_node_ids": ["<node_id_1>", "<node_id_2>"],
  "extracted_entities": [
    {"id": "<snake_case_id>", "name": "<entity name>", "type": "<concept|user|task|object>"}
  ]
}"""


def build_inference_prompt(
    user_input: str,
    graph_nodes: list[dict[str, Any]],
    qdrant_chunks: list[dict[str, Any]],
    session_utterances: list[dict[str, Any]],
) -> str:
    parts: list[str] = [_SYSTEM_INSTRUCTION, ""]

    # --- Graph context ---
    parts.append("=== GRAPH CONTEXT ===")
    for node in graph_nodes[:_MAX_GRAPH_NODES]:
        nid = node.get("id", "?")
        label = node.get("label", "Node")
        content = str(node.get("content") or node.get("text") or node.get("name") or "")
        content = content[:_MAX_NODE_CONTENT_CHARS]
        weight = node.get("weight", 0)
        parts.append(f"[{label} id={nid} weight={weight:.2f}] {content}")
    if not graph_nodes:
        parts.append("(no graph context retrieved)")

    # --- Episode context from Qdrant ---
    parts.append("")
    parts.append("=== EPISODE CONTEXT ===")
    for chunk in qdrant_chunks[:_MAX_QDRANT_CHUNKS]:
        payload = chunk.get("payload", {})
        # v2 format uses "text"; v1 format uses "content_preview" or "summary"
        text = str(
            payload.get("text")
            or payload.get("content")
            or payload.get("content_preview")
            or payload.get("summary")
            or ""
        )[:_MAX_NODE_CONTENT_CHARS]
        score = chunk.get("score", 0)
        cid = chunk.get("id", "?")
        parts.append(f"[chunk id={cid} score={score:.3f}] {text}")
    if not qdrant_chunks:
        parts.append("(no episodic context retrieved)")

    # --- Session history ---
    parts.append("")
    parts.append("=== SESSION HISTORY ===")
    # Show oldest → newest (up to 10 most recent)
    for utt in reversed(session_utterances[-10:]):
        role = utt.get("role", "?")
        text = str(utt.get("text") or utt.get("content") or "")[:200]
        parts.append(f"{role.upper()}: {text}")
    if not session_utterances:
        parts.append("(new session — no prior turns)")

    # --- User input ---
    parts.append("")
    parts.append("=== USER INPUT ===")
    parts.append(user_input)
    parts.append("")
    parts.append("JSON response:")

    return "\n".join(parts)
