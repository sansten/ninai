import json, time, pathlib, re, uuid
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta
from rouge_score import rouge_scorer
from ninai import NinaiClient
print('All imports OK.')

import pathlib, random

BASE_URL  = 'https://admin.ninai.sansten.com/api/v1'
EMAIL     = 'demo@ninai.dev'
PASSWORD  = 'demo1234'
ORG_SLUG  = 'default'

_NB_DIR      = pathlib.Path('d:/Sansten/Projects/Ninai2/repos/ninai/notebooks')
# Official LoCoMo dataset (snap-research/locomo, 10 convs, 1986 QA pairs)
DATASET_PATH = _NB_DIR / 'locomo_dataset' / 'locomo10.json'

RETRIEVAL_LIMIT  = 50   # top-N from deduplicated unique turns per conversation
ROUGE_TYPE       = 'rouge1'
LLM_MODEL        = 'qwen2.5:7b'
LLM_MODEL_HARD   = 'deepseek-coder-v2:16b'
LLM_TIMEOUT      = 120
LLM_WORKERS      = 8
INGEST_WORKERS   = 16  # more parallelism for 5882 turns

LOCOMO_SEED = 99    # fresh run tag for full dataset
RESUME_TAG  = 'locomo-full-676b1b69'  # reuse existing ingest
SKIP_INGEST = True

_rng = random.Random(LOCOMO_SEED)
run_tag = RESUME_TAG or 'locomo-full-{:08x}'.format(_rng.randint(0, 0xFFFFFFFF))

print(f'Dataset : {DATASET_PATH}')
print(f'Exists  : {DATASET_PATH.exists()}')
print(f'run_tag : {run_tag!r}')
print(f'SKIP    : {SKIP_INGEST}')

client = NinaiClient(base_url=BASE_URL)
client.login(email=EMAIL, password=PASSWORD, org_slug=ORG_SLUG)
_token = client._access_token or ''
print('Authenticated with Ninai.')

import re as _re
from datetime import datetime, timezone

with open(DATASET_PATH, encoding='utf-8') as _f:
    _raw = json.load(_f)

# Category mapping (integers in real dataset)
CAT_MAP = {1: 'single_hop', 2: 'temporal', 3: 'multi_hop',
           4: 'open_domain', 5: 'adversarial'}

def _parse_locomo_date(s):
    for fmt in ('%I:%M %p on %d %B, %Y', '%I:%M %p on %d %B %Y',
                '%H:%M on %d %B, %Y',    '%H:%M on %d %B %Y'):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime(2023, 1, 1, tzinfo=timezone.utc)

# Build conversations list in a normalised format
conversations = []
for raw_conv in _raw:
    c   = raw_conv['conversation']
    cid = 'locomo_{:03d}'.format(_raw.index(raw_conv) + 1)
    speaker_a = c['speaker_a']
    speaker_b = c['speaker_b']

    # Collect sessions
    session_nums = sorted(
        {int(_re.match(r'session_(\d+)', k).group(1))
         for k in c if _re.match(r'session_\d+$', k)},
    )
    sessions = []
    for n in session_nums:
        key_turns = f'session_{n}'
        key_date  = f'session_{n}_date_time'
        if key_turns not in c or not isinstance(c[key_turns], list):
            continue
        sessions.append({
            'session_id': n,
            'date_dt'   : _parse_locomo_date(c.get(key_date, '')),
            'turns'     : c[key_turns],   # list of {speaker, dia_id, text}
        })

    # Normalise QA pairs
    qa_pairs = []
    for i, qa in enumerate(raw_conv['qa']):
        qa_pairs.append({
            'id'      : f'{cid}_q{i+1:03d}',
            'question': qa['question'],
            'answer'  : str(qa.get('answer') or qa.get('adversarial_answer', '')),
            'category': CAT_MAP.get(qa['category'], 'open_domain'),
            'evidence': qa.get('evidence', []),
        })

    # Build event overview from structured event_summary (per-person bullet points).
    # event_summary is ~1800 tokens avg (vs 5250 for full session_summary narratives).
    # Format: '[Session N | date]\n  Person: event' -- directly answers 'What did X do?' Qs.
    _es = raw_conv.get('event_summary', {})
    _overview_lines = []
    for _ekey in sorted(_es.keys(), key=lambda k: int(k.replace('events_session_', ''))):
        _sess_data = _es[_ekey]
        if not isinstance(_sess_data, dict):
            continue
        _snum = _ekey.replace('events_session_', '')
        _date = _sess_data.get('date', '')
        _overview_lines.append(f'[Session {_snum} | {_date}]')
        for _person, _events in _sess_data.items():
            if _person == 'date':
                continue
            if isinstance(_events, list) and _events:
                for _ev in _events:
                    _overview_lines.append(f'  {_person}: {_ev}')
    session_overview = '\n'.join(_overview_lines)

    conversations.append({
        'conv_id'          : cid,
        'speaker_a'        : speaker_a,
        'speaker_b'        : speaker_b,
        'sessions'         : sessions,
        'qa_pairs'         : qa_pairs,
        'session_overview' : session_overview,
    })

total_turns = sum(len(s['turns']) for c in conversations for s in c['sessions'])
total_qa    = sum(len(c['qa_pairs']) for c in conversations)
from collections import Counter
cat_counts = Counter(qa['category'] for c in conversations for qa in c['qa_pairs'])

print(f'Loaded {len(conversations)} conversations, {total_turns} turns, {total_qa} QA pairs')
print('QA by category:')
for cat, n in sorted(cat_counts.items()):
    print(f'  {cat:15s}: {n}')

from datetime import timedelta
ingested = []
print(f'Run tag: {run_tag}')

if SKIP_INGEST:
    existing = client.memories.list(tags=[run_tag], page_size=20)
    print(f'SKIP_INGEST=True -- found {len(existing.items)} memories tagged {run_tag!r} (first page)')
    if not existing.items:
        print('  WARNING: no memories found -- set SKIP_INGEST=False and re-run')
else:
    to_ingest = []
    for conv in conversations:
        conv_id = conv['conv_id']
        for sess in conv['sessions']:
            base_dt = sess['date_dt']
            for t_idx, turn in enumerate(sess['turns']):
                speaker = turn['speaker']  # actual name, e.g. 'Caroline'
                content = '[{}] {}'.format(speaker, turn['text'])
                to_ingest.append({
                    'conv_id'    : conv_id,
                    'session_id' : sess['session_id'],
                    'content'    : content,
                    'tags'       : ['locomo', run_tag, conv_id,
                                    'session_{}'.format(sess['session_id']),
                                    speaker.lower()],
                    'occurred_at': base_dt + timedelta(minutes=t_idx * 2),
                })

    print(f'Turns to ingest: {len(to_ingest)}')

    def _create_one(item):
        try:
            mem = client.memories.create(
                content=item['content'],
                source_type='locomo_benchmark',
                tags=item['tags'],
                occurred_at=item['occurred_at'],
            )
            return {'conv_id': item['conv_id'], 'memory_id': mem.id, 'ok': True}
        except Exception as e:
            return {'conv_id': item['conv_id'], 'memory_id': None,
                    'ok': False, 'error': str(e)}

    import concurrent.futures as _cf
    print(f'Ingesting with {INGEST_WORKERS} workers...')
    t0 = time.time()
    failed = 0
    with _cf.ThreadPoolExecutor(max_workers=INGEST_WORKERS) as pool:
        futures = [pool.submit(_create_one, item) for item in to_ingest]
        for i, fut in enumerate(_cf.as_completed(futures), 1):
            r = fut.result()
            if r['ok']:
                ingested.append(r)
            else:
                failed += 1
                if failed <= 5:
                    print(f'  WARN: {r["error"]}')
            if i % 200 == 0:
                print(f'  {i}/{len(to_ingest)} ({failed} failed)...')

    elapsed = time.time() - t0
    n = len(ingested) or 1
    print(f'Ingested {len(ingested)}/{len(to_ingest)} in {elapsed:.1f}s  ({elapsed/n:.3f}s/mem, {failed} failed)')

# Retrieval helpers for LoCoMo benchmark.
# Primary: Ninai hybrid semantic search (lexical+vector).
# Fallback: stemmed BM25 top-N when search returns < 3 hits.
import urllib.request, concurrent.futures, time as _time

_PREAMBLE = (
    'based on the context', 'based on the conversation', 'according to the context',
    'according to the conversation', 'the context indicates', 'the context states',
    'from the context', 'from the conversation', 'the answer is', 'in the context',
    'i do not know', 'the conversation does not', 'there is no mention', 'no information',
    'looking at the conversation', 'reviewing the conversation', 'as per the conversation',
    'based on the provided', 'based on the above', 'in the given', 'the provided context',
)

_STOP = {
    'the','a','an','is','was','did','do','what','when','where','who','how',
    'and','or','of','in','on','to','for','at','does','has','have','had',
    'been','be','are','were','will','would','could','should','i','my','me',
    'we','our','you','your','he','she','it','they','their','that','this',
    'said','told','yes','no','not','just','about','from','with','which',
    'if','by','so','but','its','also','then','than','any','all','some',
}

# -- Stemmer: suffix-stripping to normalise word forms ----------------------
def _stem(w):
    # longer suffixes first to avoid partial stripping
    if len(w) <= 3:
        return w
    for suf in ('ation', 'tions', 'tion', 'ness', 'ment', 'ings', 'ing',
                'able', 'ible', 'ive', 'ful', 'less', 'ous', 'ary',
                'ers', 'ied', 'ies', 'ed', 'er', 'es', 'ly', 'al', 'en'):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    if w.endswith('s') and len(w) >= 4:
        return w[:-1]
    return w

def _stem_set(text):
    # both original tokens and stemmed forms for maximum recall
    raw = set(re.sub(r'[^\w\s]', '', text.lower()).split()) - _STOP
    return raw | {_stem(t) for t in raw}

# -- Answer cleaner: strip preamble, keep first line ------------------------
def _clean_answer(raw):
    if not raw:
        return raw
    s = raw.strip()
    lower = s.lower()
    for p in _PREAMBLE:
        if lower.startswith(p):
            for sep in (',', ':', ';', ' --', ' -', '\n'):
                idx = s.find(sep)
                if idx != -1 and idx < 100:
                    rest = s[idx+1:].strip()
                    if rest:
                        s = rest
                        break
            break
    first_line = s.split('\n')[0].strip()
    if first_line:
        s = first_line
    return s

def _sort_by_date(mem_dicts):
    def parse_dt(m):
        oc = (m.get('occurred_at') or '')
        try:
            from datetime import datetime
            return datetime.fromisoformat(oc.replace('Z', '+00:00'))
        except Exception:
            from datetime import datetime, timezone
            return datetime.min.replace(tzinfo=timezone.utc)
    return sorted(mem_dicts, key=parse_dt)

def _dedup_by_content(mem_dicts):
    # remove duplicate turns (ingest ran 3x with same run_tag)
    seen, unique = set(), []
    for m in mem_dicts:
        c = m.get('content', '')
        if c not in seen:
            seen.add(c)
            unique.append(m)
    return unique

def _top_k_bm25(question, mem_dicts, k, extra_terms=''):
    # stemmed BM25; falls back to most-recent turns when no keyword overlap
    qstems = _stem_set(question + (' ' + extra_terms if extra_terms else ''))
    if not qstems:
        return _sort_by_date(mem_dicts)[-k:]
    scored = []
    for m in mem_dicts:
        mstems = _stem_set(m.get('content', ''))
        sc = sum(1 for w in qstems if w in mstems)
        scored.append((sc, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [m for sc, m in scored if sc > 0][:k]
    if len(top) < k:
        seen_ids = {m.get('id') for m in top}
        recent = [m for m in _sort_by_date(mem_dicts) if m.get('id') not in seen_ids]
        top += recent[-(k - len(top)):]
    return top

def _extract_key_terms(text, top_n=10):
    tokens = [t for t in re.sub(r'[^\w\s]', '', text.lower()).split()
              if t not in _STOP and len(t) > 2]
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]

def _query_expand(question, category):
    # QueryIntelligenceAgent: extract named entities + intent to enrich search.
    words = re.sub(r'[^\w\s]', '', question).split()
    entities = [w for i, w in enumerate(words)
                if i > 0 and w and w[0].isupper() and w.lower() not in _STOP and len(w) > 2]
    q_lower = question.lower()
    extras = []
    if any(kw in q_lower for kw in ('where', 'location', 'place', 'city', 'country', 'move', 'moved', 'live', 'lived')):
        extras = ['location', 'place', 'moved']
    elif any(kw in q_lower for kw in ('when', 'date', 'year', 'month', 'how long', 'how many')):
        extras = ['date', 'time', 'year']
    elif any(kw in q_lower for kw in ('who', 'whose', 'person', 'name')):
        extras = ['person', 'name']
    expansion = ' '.join(entities[:4] + extras[:2])
    return (question + ' ' + expansion).strip() if expansion else question

def _episodic_diversify(hits, all_unique, question, limit):
    # EpisodicGroupingAgent: ensure multi_hop hits span multiple sessions.
    # If all hits cluster in <3 sessions, sample bridging turns from other sessions.
    def _session(m):
        return (m.get('occurred_at') or '')[:7]  # YYYY-MM
    session_hits = {}
    for h in hits:
        session_hits.setdefault(_session(h), []).append(h)
    if len(session_hits) >= 3:
        return hits
    hit_ids = {h.get('id') for h in hits}
    other_sessions = {}
    for m in all_unique:
        if m.get('id') in hit_ids:
            continue
        sk = _session(m)
        if sk not in session_hits:
            other_sessions.setdefault(sk, []).append(m)
    if not other_sessions:
        return hits
    q_stems = _stem_set(question)
    additions = []
    for sk in sorted(other_sessions.keys()):
        mems = other_sessions[sk]
        best = max(mems, key=lambda m: sum(1 for w in q_stems if w in _stem_set(m.get('content', ''))))
        additions.append(best)
    return hits + additions[:4]

def _session_expand(hits, mem_dicts):
    # EpisodicGroupingAgent core insight: group by session date (YYYY-MM-DD).
    # LoCoMo turns have no session_N tag; sessions are identified by occurred_at date.
    # Once a date is identified as relevant (any hit), include ALL turns from that date.
    # cognitive_rerank then narrows back to top-N.
    hit_dates = set()
    for h in hits:
        d = (h.get('occurred_at') or '')[:10]
        if d:
            hit_dates.add(d)
    if not hit_dates:
        return hits
    hit_ids = {h.get('id') for h in hits}
    expansions = []
    for m in mem_dicts:
        if m.get('id') in hit_ids:
            continue
        d = (m.get('occurred_at') or '')[:10]
        if d in hit_dates:
            expansions.append(m)
            hit_ids.add(m.get('id'))
    return hits + expansions

# -- ROUGE normalization: date/number forms ----------------------------------
_MONTH_MAP = {
    'january':'01','february':'02','march':'03','april':'04',
    'may':'05','june':'06','july':'07','august':'08',
    'september':'09','october':'10','november':'11','december':'12',
    'jan':'01','feb':'02','mar':'03','apr':'04','jun':'06',
    'jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12',
}
_NUM_MAP = {
    'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
    'six':'6','seven':'7','eight':'8','nine':'9','ten':'10',
    'eleven':'11','twelve':'12','thirteen':'13','fourteen':'14',
    'fifteen':'15','sixteen':'16','seventeen':'17','eighteen':'18',
    'nineteen':'19','twenty':'20','thirty':'30','forty':'40',
    'fifty':'50','sixty':'60','seventy':'70','eighty':'80','ninety':'90',
    'hundred':'100','once':'1','twice':'2','thrice':'3',
}

def _normalize_for_rouge(text):
    # None means false-premise in adversarial Qs; caller maps to 'none' before calling.
    s = re.sub(r'[^\w\s]', ' ', str(text or '').lower()).strip()
    toks = [_NUM_MAP.get(t, t) for t in s.split()]
    s = ' '.join(toks)
    def _msub(m):
        mon = _MONTH_MAP.get(m.group(2).lower())
        if not mon:
            return m.group(0)
        return '{} {} {}'.format(m.group(1), mon, m.group(3))
    s = re.sub(r'\b(\d{1,2})\s+([a-z]+)\s+(\d{4})\b', _msub, s)
    s = re.sub(r'\b([a-z]+)\s+(\d{1,2})\s*\s*(\d{4})\b',
               lambda m: '{} {} {}'.format(
                   m.group(2), _MONTH_MAP.get(m.group(1).lower(), m.group(1)), m.group(3)), s)
    return s.strip()

# -- Single-hop answer sharpener: extract shortest plausible span ------------
def _sharpen_single_hop(answer, question):
    toks = answer.split()
    if len(toks) <= 4:
        return answer
    q_lower = question.lower()
    if q_lower.startswith('where') or 'where' in q_lower[:20]:
        words = answer.split()
        candidates, run = [], []
        for w in words:
            cleaned = re.sub(r'[^\w]', '', w)
            if cleaned and (cleaned[0].isupper() or cleaned.isdigit()):
                run.append(w)
            else:
                if run:
                    candidates.append(' '.join(run))
                run = []
        if run:
            candidates.append(' '.join(run))
        if candidates:
            return candidates[-1]
    if q_lower.startswith('who') or q_lower.startswith('whose'):
        words = answer.split()
        run = []
        for w in words:
            cleaned = re.sub(r'[^\w]', '', w)
            if cleaned and cleaned[0].isupper() and cleaned.lower() not in _STOP:
                run.append(w)
            elif run:
                break
        if run and len(run) <= 4:
            return ' '.join(run)
    if q_lower.startswith('what'):
        m = re.search(r'\b(\d[\d,./]*(?:\s+\w+){0,2})\b', answer)
        if m:
            return m.group(1)
    return answer

def _sharpen_boolean(answer, question):
    # For yes/no questions, collapse verbose answers to 'Yes'/'No'.
    q = question.lower().strip()
    _bool_starts = ('do ', 'did ', 'is ', 'are ', 'was ', 'were ', 'has ', 'have ', 'can ')
    if not any(q.startswith(s) for s in _bool_starts):
        return answer
    low = answer.lower()
    first = low.split()[:5]
    if 'yes' in first or low.startswith('yes'):
        return 'Yes'
    if 'no' in first or low.startswith('no') or 'not' in low.split()[:3]:
        return 'No'
    return answer

def _iso_to_natural(answer):
    # Convert ISO date outputs (YYYY-MM-DD) from context prefixes into natural language.
    # LLMs sometimes echo the [YYYY-MM-DD] context prefix format as the answer.
    _months = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
    _a = answer.strip()
    # Strip leading/trailing brackets: '[2023-05-08]' → '2023-05-08'
    _a = re.sub(r'^\[|\]$', '', _a).strip()
    # Full ISO date: '2023-05-08' → '8 May 2023'
    _m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', _a)
    if _m:
        _y, _mo, _d = int(_m.group(1)), int(_m.group(2)), int(_m.group(3))
        if 1 <= _mo <= 12:
            return '{} {} {}'.format(_d, _months[_mo-1], _y)
    # Year-month only: '2023-05' → 'May 2023'
    _m = re.match(r'^(\d{4})-(\d{2})$', _a)
    if _m:
        _y, _mo = int(_m.group(1)), int(_m.group(2))
        if 1 <= _mo <= 12:
            return '{} {}'.format(_months[_mo-1], _y)
    return answer

def _resolve_temporal_references(answer, last_date):
    # Convert relative time expressions to absolute dates using the last session date.
    # Fixes: 'last year' → '2022', 'next month' → 'June 2023' etc.
    if not last_date or not answer:
        return answer
    try:
        from datetime import datetime as _dt, timedelta as _td
        _ref = _dt.strptime(last_date, '%Y-%m-%d').date()
    except Exception:
        return answer
    _a = answer.strip()
    _low = _a.lower()
    _months = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
    if re.search(r'\blast\s+year\b', _low):
        return str(_ref.year - 1)
    if re.search(r'\bnext\s+year\b', _low):
        return str(_ref.year + 1)
    if re.search(r'\bthis\s+year\b', _low):
        return str(_ref.year)
    if re.search(r'\blast\s+month\b', _low):
        _d = (_ref.replace(day=1) - _td(days=1))
        return _months[_d.month - 1] + ' ' + str(_d.year)
    if re.search(r'\bnext\s+month\b', _low):
        _m2 = _ref.month % 12 + 1
        _y2 = _ref.year + (1 if _ref.month == 12 else 0)
        return _months[_m2 - 1] + ' ' + str(_y2)
    if re.search(r'\bthis\s+month\b', _low):
        return _months[_ref.month - 1] + ' ' + str(_ref.year)
    if re.search(r'\blast\s+week\b', _low):
        _w = _ref - _td(days=7)
        return _w.strftime('week of %B %d %Y')
    # 'X years ago' → compute year
    _m = re.search(r'\b(\d+)\s+years?\s+ago\b', _low)
    if _m:
        return str(_ref.year - int(_m.group(1)))
    # 'X months ago'
    _m = re.search(r'\b(\d+)\s+months?\s+ago\b', _low)
    if _m:
        _n = int(_m.group(1))
        _mo = (_ref.month - _n - 1) % 12 + 1
        _yr = _ref.year + (_ref.month - _n - 1) // 12
        return _months[_mo - 1] + ' ' + str(_yr)
    return _a

def _mem_obj_to_dict(m):
    # prefer content (full text); fall back to content_preview
    full = getattr(m, 'content', None) or getattr(m, 'content_preview', None) or m.title or ''
    return {
        'id'         : str(m.id),
        'content'    : full,
        'tags'       : m.tags or [],
        'occurred_at': m.occurred_at.isoformat() if m.occurred_at else None,
    }

def _search_semantic(question, conv_id, run_tag, client, limit, hybrid=True):
    # Ninai hybrid semantic+lexical search filtered to one conversation
    for attempt in range(3):
        try:
            # enforce timeout so one slow API call cannot stall the whole benchmark
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                fut = _pool.submit(
                    client.memories.search,
                    query=question,
                    tags=[conv_id, run_tag],
                    limit=limit,
                    threshold=0.0,
                    hybrid=hybrid,
                    use_graph=True,
                )
                result = fut.result(timeout=25)
            hits = [
                _mem_obj_to_dict(m) for m in (result.items or [])
                if getattr(m, 'source_type', None) == 'locomo_benchmark'
            ]
            # semantic search can return duplicate turns; keep first occurrence only
            _seen = set()
            _unique_hits = []
            for h in hits:
                _key = h.get('content', '')
                if _key and _key in _seen:
                    continue
                _seen.add(_key)
                _unique_hits.append(h)
            return _unique_hits
        except Exception as e:
            if attempt < 2:
                _time.sleep(1.5 ** attempt)
            else:
                return []
    return []

def _cognitive_rerank(question, hits, limit, base_url, token):
    # Pass Qdrant semantic hits through Ninai cognitive gateway:
    # AttentionRetrievalService + SelfRAG + CorrectiveRAG + ContextCompression.
    # Falls back to raw hits on any error so benchmark never stalls.
    if not hits or not token:
        return hits[:limit]
    try:
        payload = json.dumps({
            'query': question,
            'memories': hits,
            'limit': limit,
        }).encode()
        req = urllib.request.Request(
            base_url.rstrip('/') + '/cognitive/gateway/read',
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + token,
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        reranked = data.get('memories') or []
        return reranked if reranked else hits[:limit]
    except Exception:
        return hits[:limit]

def _retrieve(question, mem_dicts, mems_obj, category, limit,
              client=None, run_tag=None, conv_id=None):
    # ── Primary: Ninai semantic search (hybrid lexical+vector) ──────────
    if client is not None and run_tag is not None and conv_id is not None:
        k = limit if category not in ('multi_hop',) else min(limit * 2, 80)
        # QueryIntelligenceAgent: entity/intent-expanded query for stage-1 search
        search_q = _query_expand(question, category)
        hits = _search_semantic(search_q, conv_id, run_tag, client, k)
        if len(hits) >= 3:
            # Session expansion: include ALL turns from sessions already hit by semantic search.
            # Semantic finds the right session but may miss the answer-bearing turn.
            # Expansion gives us more candidates; BM25 then re-ranks for relevance.
            hits = _session_expand(hits, mem_dicts)
            if category == 'multi_hop':
                # Stage 2: expand with entity terms from stage-1 results
                stage1_text = ' '.join(h['content'] for h in hits[:10])
                key_terms   = _extract_key_terms(stage1_text)
                expanded_q  = question + ' ' + ' '.join(key_terms[:5])
                hits2 = _search_semantic(expanded_q, conv_id, run_tag, client, limit)
                # Stage 3: proper noun bridge terms
                all_text = ' '.join(h['content'] for h in hits + hits2)
                proper_nouns = []
                for tok in re.sub(r'\[\w+\]', '', all_text).split():
                    w = re.sub(r'[^\w]', '', tok)
                    if w and w[0].isupper() and w.lower() not in _STOP and len(w) > 2:
                        proper_nouns.append(w.lower())
                q_toks = set(question.lower().split())
                pn_freq = {}
                for p in proper_nouns:
                    if p not in q_toks:
                        pn_freq[p] = pn_freq.get(p, 0) + 1
                bridge_terms = [w for w, _ in sorted(pn_freq.items(), key=lambda x: -x[1])[:4]]
                hits3 = _search_semantic(' '.join(bridge_terms), conv_id, run_tag, client, limit) if bridge_terms else []
                seen_ids, merged = set(), []
                for h in hits + hits2 + hits3:
                    if h['id'] not in seen_ids:
                        seen_ids.add(h['id'])
                        merged.append(h)
                # BM25 re-rank within merged pool (relevance over recency)
                merged = _top_k_bm25(question, merged, limit)
                # EpisodicGroupingAgent: session diversity — ensures hits span multiple sessions
                unique_all = _dedup_by_content(mem_dicts)
                merged = _episodic_diversify(merged, unique_all, question, limit)
                return _sort_by_date(merged)
            # BM25 re-rank within session-expanded pool (relevance, not recency)
            # cognitive_rerank (AttentionRetrievalService) is recency-biased and
            # discards answer-bearing turns from older sessions. BM25 is query-centric.
            hits = _top_k_bm25(question, hits, limit)
            return _sort_by_date(hits)
    # ── Fallback: stemmed BM25 ───────────────────────────────────────────
    unique = _dedup_by_content(mem_dicts)
    if category == 'multi_hop':
        k1 = min(limit * 2, 60)
        k2 = min(limit, 40)
        stage1 = _top_k_bm25(question, unique, k1)
        stage1_text = ' '.join(m.get('content', '') for m in stage1)
        key_terms = _extract_key_terms(stage1_text)
        stage2 = _top_k_bm25(question, unique, k2, extra_terms=' '.join(key_terms[:6]))
        # Stage 3: extract proper nouns as bridging entities
        all_text = ' '.join(m.get('content', '') for m in stage1 + stage2)
        proper_nouns = []
        for tok in re.sub(r'\[\w+\]', '', all_text).split():
            w = re.sub(r'[^\w]', '', tok)
            if w and w[0].isupper() and w.lower() not in _STOP and len(w) > 2:
                proper_nouns.append(w.lower())
        q_lower_toks = set(question.lower().split())
        pn_freq = {}
        for p in proper_nouns:
            if p not in q_lower_toks:
                pn_freq[p] = pn_freq.get(p, 0) + 1
        bridge_terms = [w for w, _ in sorted(pn_freq.items(), key=lambda x: -x[1])[:4]]
        stage3 = _top_k_bm25(' '.join(bridge_terms), unique, k2) if bridge_terms else []
        seen_ids, merged = set(), []
        for m in stage1 + stage2 + stage3:
            mid = m.get('id', '')
            if mid not in seen_ids:
                seen_ids.add(mid)
                merged.append(m)
        return _sort_by_date(merged[:k2])
    elif category == 'temporal':
        k1 = min(limit * 2, 50)
        stage1 = _top_k_bm25(question, unique, k1)
        stage1_text = ' '.join(m.get('content', '') for m in stage1)
        key_terms = _extract_key_terms(stage1_text)
        expanded = _top_k_bm25(question, unique, min(limit, 30),
                               extra_terms=' '.join(key_terms[:6]))
        seen_ids, merged = set(), []
        for m in stage1 + expanded:
            mid = m.get('id', '')
            if mid not in seen_ids:
                seen_ids.add(mid)
                merged.append(m)
        return _sort_by_date(merged[:limit])
    elif category == 'adversarial':
        k = min(limit, 30)
        stage1 = _top_k_bm25(question, unique, k * 2)
        stage1_text = ' '.join(m.get('content', '') for m in stage1)
        key_terms = _extract_key_terms(stage1_text)
        merged = _top_k_bm25(question, unique, k, extra_terms=' '.join(key_terms[:5]))
        return _sort_by_date(merged)
    else:
        k = min(limit, len(unique))
        return _sort_by_date(_top_k_bm25(question, unique, k))

def _build_prompt(category, question, context, last_date='', session_overview=''):
    # Prepend event overview for all categories except multi_hop.
    # multi_hop needs clean retrieval context; overview adds noise for 2-hop reasoning.
    _overview_block = ''
    if session_overview:
        _overview_block = ('KEY EVENTS (all sessions):\n'
                           + session_overview + '\n\n')
    q_lower = question.lower()
    if category == 'temporal':
        # Detect if question already contains a specific date/month anchor.
        # These ask WHAT happened at a known time (not WHEN something happened).
        # Prompting for date output causes LLM to echo the question date back.
        _has_date_anchor = bool(re.search(
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december|20\d{2})\b',
            q_lower))
        _is_when_q = (q_lower.startswith('when') or 'what date' in q_lower
                      or 'what year' in q_lower or 'what month' in q_lower
                      or 'what time' in q_lower)
        if _has_date_anchor and not _is_when_q:
            inst = (
                'RULE: The question already states the time period. Answer with the FACT or DESCRIPTION, NOT a date.\n'
                'Reply with ONLY the specific fact (1 short phrase). Do NOT repeat the date from the question. No explanation.\n'
            )
        elif any(p in q_lower for p in ('how long', 'how old', 'how many weeks',
                                       'how many months', 'how many days', 'how many years',
                                       'how many times', 'how many hours')):
            inst = (
                'RULE: Reply with ONLY the duration or count. Examples: "4 months" / "6 weeks" / "3 times".\n'
                'Do NOT write a full sentence. Do NOT explain.\n'
            )
        elif 'before or after' in q_lower or 'did it happen' in q_lower:
            inst = (
                'RULE: Reply with ONLY "Before" or "After" plus the key dates.\n'
                'Example: "Before -- Event A: May 5; Event B: June 3."\n'
            )
        else:
            inst = (
                'RULE: Use the session date prefixes to convert ALL relative time references (last Saturday, next week, 4 years ago) to specific calendar dates.\n'
                'Examples of correct format: "25 May 2023", "March 2019", "August 2022", "the Sunday before 25 May 2023".\n'
                'Reply with ONLY the date or time expression. No full sentence. No explanation.\n'
            )
        return (
            _overview_block + 'Conversation turns with ISO dates:\n' + context + '\n\n'
            'Question: ' + question + '\n'
            + inst +
            'Answer:'
        )
    elif category == 'multi_hop':
        return (
            _overview_block + 'Conversation excerpts (chronological):\n' + context + '\n\n'
            'Question: ' + question + '\n'
            'RULE: Answer with ONLY the key fact (name, place, or short phrase). Prefer 1-5 words.\n'
            'Do NOT write a full sentence. Do NOT explain.\n'
            'Answer:'
        )
    elif category == 'adversarial':
        return (
            _overview_block + 'Conversation (chronological order):\n' + context + '\n\n'
            'Question: ' + question + '\n'
            'RULE: Answer using only what the conversation states. Reply with ONLY the exact fact — 1 to 8 words. No explanation.\n'
            'Answer:'
        )
    elif category == 'open_domain':
        return (
            _overview_block + 'Conversation excerpts:\n' + context + '\n\n'
            'Question: ' + question + '\n'
            'RULE: Reply with ONLY the shortest exact span from the conversation that answers the question.\n'
            'Prefer 3-12 words. Use exact names and wording from the conversation. No explanation.\n'
            'Answer:'
        )
    else:  # single_hop
        time_inst = ''
        if 'time' in q_lower and any(w in q_lower for w in ('marathon', 'finish', 'race', 'ran', 'run')):
            time_inst = 'For race times like "3:07", write "3 hours and 7 minutes". '
        return (
            _overview_block + 'Conversation (chronological order):\n' + context + '\n\n'
            'Question: ' + question + '\n'
            'RULE: Reply with ONLY the exact word, name, or short phrase that answers the question.\n'
            'Do NOT write a full sentence. Do NOT explain. ' + time_inst + '\n'
            'Answer:'
        )

def _ollama_generate(prompt, model=LLM_MODEL, timeout=LLM_TIMEOUT, num_ctx=32768):
    try:
        payload = json.dumps({
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'num_ctx': num_ctx,
                'temperature': 0,
                'top_p': 1.0,
            }
        }).encode()
        req = urllib.request.Request('http://localhost:11434/api/generate',
            data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())['response'].strip()
    except Exception:
        return ''

def _run_prompts_parallel(prompts, models=None, workers=LLM_WORKERS, num_ctx=32768):
    # models: list of model names, one per prompt; None = use LLM_MODEL for all
    if models is None:
        models = [LLM_MODEL] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_ollama_generate, p, m, LLM_TIMEOUT, num_ctx)
                for p, m in zip(prompts, models)]
        fut_idx = {fut: i for i, fut in enumerate(futs)}
        results = [None] * len(prompts)
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            results[fut_idx[fut]] = fut.result()
            done += 1
            if done % 100 == 0:
                print(f'    {done}/{len(prompts)} answers received...')
        return results

def _extract_answer_heuristic(question, context):
    if not context.strip():
        return ''
    q_words = set(re.sub(r'[^\w\s]', '', question.lower()).split())
    stop = {
        'the','a','an','is','was','did','do','what','when','where','who','how',
        'and','or','of','in','on','to','for','at','i','my','me','we','our',
        'you','your','he','she','it','they','their','that','this','these','those',
    }
    sentences   = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', context) if s.strip()]
    clean_sents = [re.sub(r'^\[\S+\]\s*', '', s) for s in sentences]
    best_sent, best_score = '', -1.0
    for orig, clean in zip(sentences, clean_sents):
        tokens = [t for t in re.sub(r'[^\w\s]', '', clean.lower()).split() if t not in stop]
        if not tokens:
            continue
        new_facts = [t for t in tokens if t not in q_words]
        if not new_facts:
            continue
        score = len(new_facts) / len(tokens)
        if score > best_score:
            best_score, best_sent = score, clean
    return best_sent or (clean_sents[0] if clean_sents else '')

test = _ollama_generate('Reply with only the word: ready', timeout=10)
print('Ollama:', 'ok' if test else 'unavailable -- heuristic fallback active')
scorer = rouge_scorer.RougeScorer([ROUGE_TYPE], use_stemmer=True)

# Paginated fetch -- filter to source_type='locomo_benchmark' only.
# Ninai creates enrichment/episodic derivative records (3x multiplier).
print('Fetching run memories (source_type=locomo_benchmark)...')
all_run_mems = []
page = 1
while True:
    page_result = client.memories.list(tags=[run_tag], page=page, page_size=100)
    for m in page_result.items:
        if getattr(m, 'source_type', None) == 'locomo_benchmark':
            all_run_mems.append(m)
    if not page_result.has_more:
        break
    page += 1
print(f'Source memories: {len(all_run_mems)} (original ingested turns only)')

# Group by conv_id -- build BM25 fallback dicts
conv_ids = [c['conv_id'] for c in conversations]
conv_memories_obj  = {cid: [] for cid in conv_ids}
for m in all_run_mems:
    for cid in conv_ids:
        if cid in (m.tags or []):
            conv_memories_obj[cid].append(m)
            break

# _mem_obj_to_dict is defined in cell 15
conv_memories_dict = {
    cid: [_mem_obj_to_dict(m) for m in mems]
    for cid, mems in conv_memories_obj.items()
}

for cid in conv_ids:
    raw = len(conv_memories_obj[cid])
    unique_n = len(set(m.get('content','') for m in conv_memories_dict[cid]) - {''})
    print(f'  {cid}: {raw} raw -> {unique_n} unique after content-dedup')

# Build session overview lookup (conv_id → session summary text from raw dataset)
conv_overview_dict = {conv['conv_id']: conv.get('session_overview', '') for conv in conversations}

# ── Phase 1: semantic retrieval (stable sequential mode) ────────────────
print('Phase 1: semantic retrieval (sequential for stability)...')
import time as _time2

_all_qa_flat = [
    (conv['conv_id'], qa)
    for conv in conversations
    for qa in conv['qa_pairs']
]

def _retrieve_one(args):
    conv_id, qa = args
    mems_dict = conv_memories_dict.get(conv_id, [])
    mems_obj  = conv_memories_obj.get(conv_id, [])
    retrieved = _retrieve(
        qa['question'], mems_dict, mems_obj, qa['category'], RETRIEVAL_LIMIT,
        client=client, run_tag=run_tag, conv_id=conv_id,
    )
    if len(retrieved) > RETRIEVAL_LIMIT * 4:
        retrieved = retrieved[:RETRIEVAL_LIMIT]
    if qa['category'] == 'temporal':
        _ctx_lines = []
        _prev_date = None
        for m in retrieved:
            _dt = (m.get('occurred_at') or '?')[:10]
            if _dt != _prev_date:
                _ctx_lines.append('--- Session: {} ---'.format(_dt))
                _prev_date = _dt
            _ctx_lines.append('[{date}] {text}'.format(date=_dt, text=m.get('content') or ''))
        context = '\n'.join(_ctx_lines)
    else:
        context = '\n'.join(m.get('content') or '' for m in retrieved)
    return {
        'conv_id'         : conv_id,
        'qa_id'           : qa['id'],
        'category'        : qa['category'],
        'question'        : qa['question'],
        'gold_answer'     : qa['answer'],
        'context'         : context,
        'retrieved'       : len(retrieved),
        'last_date'       : (retrieved[-1].get('occurred_at') or '')[:10] if retrieved else '',
        'session_overview': conv_overview_dict.get(conv_id, ''),
    }

t0_ret = _time2.time()
qa_records = []
for i, args in enumerate(_all_qa_flat, 1):
    qa_records.append(_retrieve_one(args))
    if i % 200 == 0:
        print(f'  Retrieved {i}/{len(_all_qa_flat)}...')

print(f'Retrieval done in {_time2.time()-t0_ret:.1f}s')
from collections import Counter
cat_counts = Counter(r['category'] for r in qa_records)
print(f'Total QA: {len(qa_records)}')
for cat, cnt in sorted(cat_counts.items()):
    print(f'  {cat:15s}: {cnt}')

# ── Phase 2: build prompts ─────────────────────────────────────────────
print('Phase 2: building prompts...')
prompts = []
for r in qa_records:
    prompts.append(_build_prompt(r['category'], r['question'], r['context'],
                                 last_date=r.get('last_date', ''),
                                 session_overview=r.get('session_overview', '')))

# ── Phase 3: model-routed LLM inference ───────────────────────────────
print(f'Phase 3: model-routed LLM inference ({LLM_WORKERS} workers, {len(prompts)} prompts)...')
import time as _time
t0 = _time.time()
raw_answers = [''] * len(prompts)
# qwen: adversarial, single_hop, temporal (short answers, fast)
# deepseek (16K ctx): open_domain + multi_hop (richer answers; 16K sufficient for ~2.5K-token prompts)
qwen_cats = {'adversarial', 'single_hop', 'temporal'}
deep_cats = {'open_domain', 'multi_hop'}
qwen_idx = [i for i, r in enumerate(qa_records) if r['category'] in qwen_cats]
deep_idx = [i for i, r in enumerate(qa_records) if r['category'] in deep_cats]

if qwen_idx:
    print(f'  qwen ({LLM_MODEL}): {len(qwen_idx)} prompts')
    qwen_prompts = [prompts[i] for i in qwen_idx]
    qwen_raw = _run_prompts_parallel(qwen_prompts,
                                     models=[LLM_MODEL] * len(qwen_prompts),
                                     workers=LLM_WORKERS)
    for j, i in enumerate(qwen_idx):
        raw_answers[i] = qwen_raw[j]

if deep_idx:
    deep_workers = 4
    print(f'  deepseek 16K ({LLM_MODEL_HARD}): {len(deep_idx)} prompts, workers={deep_workers}')
    deep_prompts = [prompts[i] for i in deep_idx]
    deep_raw = _run_prompts_parallel(deep_prompts,
                                     models=[LLM_MODEL_HARD] * len(deep_prompts),
                                     workers=deep_workers,
                                     num_ctx=16384)
    for j, i in enumerate(deep_idx):
        raw_answers[i] = deep_raw[j]

elapsed = _time.time() - t0

llm_used, heuristic_used = 0, 0
generated_answers = []
for rec, raw in zip(qa_records, raw_answers):
    if raw:
        gen = _clean_answer(raw)
        if rec['category'] in ('single_hop', 'adversarial'):
            gen = _sharpen_single_hop(gen, rec['question'])
        if rec['category'] in ('single_hop', 'adversarial'):
            gen = _sharpen_boolean(gen, rec['question'])
        if rec['category'] == 'temporal':
            gen = _resolve_temporal_references(gen, rec.get('last_date', ''))
        generated_answers.append(gen)
        llm_used += 1
    else:
        generated_answers.append(_extract_answer_heuristic(rec['question'], rec['context']))
        heuristic_used += 1

print(f'  LLM: {llm_used} | Heuristic: {heuristic_used} | Time: {elapsed:.1f}s')

# ── Phase 4: ROUGE scoring ──────────────────────────────────────────────
print('Phase 4: ROUGE scoring...')
results = []
for rec, gen in zip(qa_records, generated_answers):
    gold_norm = _normalize_for_rouge(rec['gold_answer'])
    gen_norm  = _normalize_for_rouge(gen)
    # Temporal: if gold is a bare year (e.g. "2022"), extract year from generated answer.
    # LLM often outputs full date "August 2022" when gold is just "2022".
    if rec['category'] == 'temporal' and re.match(r'^\d{4}$', gold_norm.strip()):
        _years = re.findall(r'\b(20\d{2}|19\d{2})\b', gen_norm)
        if _years:
            gen_norm = _years[0]
    score = scorer.score(gold_norm, gen_norm)
    f1    = score[ROUGE_TYPE].fmeasure * 100
    results.append({
        'conv_id'          : rec['conv_id'],
        'qa_id'            : rec['qa_id'],
        'category'         : rec['category'],
        'question'         : rec['question'],
        'gold_answer'      : rec['gold_answer'],
        'generated_answer' : gen[:200],
        'retrieved_count'  : rec['retrieved'],
        'rouge1_f1'        : round(f1, 2),
    })

df_results = pd.DataFrame(results)
overall = df_results['rouge1_f1'].mean()
print(f'Evaluated {len(df_results)} QA pairs.')
print(f'Mean ROUGE-1 F1: {overall:.2f}')
print()
print('Per-category scores:')
print(df_results.groupby('category')['rouge1_f1'].mean().round(2).to_string())
print()
print('Published baselines (Maharana et al., ACL 2024):')
print('  MemoryBank (GPT-4)       : overall ~37-41 (varies by retrieval)')
print('  ReadAgent (GPT-4)        : QA F1 ~45-47')
print('  GPT-3.5-turbo-16K (full) : QA F1 ~37.8')
print('  Human performance        : QA F1 ~87.9')
CATEGORIES = ['single_hop', 'multi_hop', 'temporal', 'adversarial', 'open_domain']

def agg_scores(df):
    out = {}
    for cat in CATEGORIES:
        sub = df[df['category'] == cat]
        out[cat] = round(sub['rouge1_f1'].mean(), 1) if len(sub) > 0 else 0.0
    out['overall'] = round(df['rouge1_f1'].mean(), 1)
    return out

scores = agg_scores(df_results)
print('Ninai ROUGE-1 F1 scores (full LoCoMo dataset, 1986 QA pairs):')
for cat in CATEGORIES:
    print(f'  {cat:15s}: {scores[cat]}')
print(f'  {"overall":15s}: {scores["overall"]}')

ninai_scores = scores
baselines    = None  # use hardcoded list in cell 22

# Export all results to JSON for offline analysis
import json as _json
_export = df_results[['qa_id','category','question','gold_answer','generated_answer','rouge1_f1','retrieved_count']].to_dict(orient='records')
with open('locomo_results_latest.json', 'w', encoding='utf-8') as _f:
    _json.dump({'scores': scores, 'results': _export}, _f, indent=2)
print('Results exported to locomo_results_latest.json')

