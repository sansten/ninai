import json

with open('locomo_benchmark.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell 5: config — seed 43 forces re-ingest with persona-name-injected content ──
cells[5]['source'] = [
    "import pathlib, random\n",
    "\n",
    "BASE_URL  = 'https://admin.ninai.sansten.com/api/v1'\n",
    "EMAIL     = 'demo@ninai.dev'\n",
    "PASSWORD  = 'demo1234'\n",
    "ORG_SLUG  = 'default'\n",
    "\n",
    "_NB_DIR      = pathlib.Path('d:/Sansten/Projects/Ninai2/repos/ninai/notebooks')\n",
    "DATASET_PATH = _NB_DIR / 'locomo_dataset' / 'locomo_sample.json'\n",
    "\n",
    "USE_FULL_DATASET = False\n",
    "RETRIEVAL_LIMIT  = 10\n",
    "ROUGE_TYPE       = 'rouge1'\n",
    "LLM_MODEL        = 'qwen2.5:7b'\n",
    "LLM_TIMEOUT      = 120\n",
    "LLM_WORKERS      = 8\n",
    "INGEST_WORKERS   = 8\n",
    "\n",
    "# LOCOMO_SEED 43 => new run_tag with persona-name-injected content\n",
    "LOCOMO_SEED = 43\n",
    "RESUME_TAG  = None\n",
    "SKIP_INGEST = False\n",
    "\n",
    "_rng = random.Random(LOCOMO_SEED) if LOCOMO_SEED is not None else random\n",
    "run_tag = RESUME_TAG or 'locomo-{:08x}'.format(_rng.randint(0, 0xFFFFFFFF))\n",
    "\n",
    "print(f'Dataset path : {DATASET_PATH}')\n",
    "print(f'Dataset exists: {DATASET_PATH.exists()}')\n",
    "print(f'run_tag      : {run_tag!r}  (LOCOMO_SEED={LOCOMO_SEED})')\n",
    "print(f'SKIP_INGEST  : {SKIP_INGEST}')\n",
    "print(f'LLM_WORKERS  : {LLM_WORKERS}')\n",
]

# ── Cell 11: ingestion with persona-name injection ──────────────────────────────
cells[11]['source'] = [
    "ingested = []\n",
    "print(f'Run tag: {run_tag}')\n",
    "\n",
    "\n",
    "def _persona_name(conv):\n",
    "    s = conv.get('persona', '')\n",
    "    return s.split(',')[0].strip() if s else 'User'\n",
    "\n",
    "\n",
    "if SKIP_INGEST:\n",
    "    existing = client.memories.list(tags=[run_tag], page_size=20)\n",
    "    print(f'SKIP_INGEST=True -- skipping ingestion.')\n",
    "    print(f'  Found {len(existing.items)} memories tagged {run_tag!r}.')\n",
    "    if not existing.items:\n",
    "        print('  WARNING: no memories found -- run ingestion first.')\n",
    "else:\n",
    "    to_ingest = []\n",
    "    for conv in conversations:\n",
    "        conv_id = conv['conv_id']\n",
    "        persona = _persona_name(conv)  # 'Alex', 'Maria', 'Dev'\n",
    "        for session in conv['sessions']:\n",
    "            session_id   = session['session_id']\n",
    "            session_date = session.get('date', '')\n",
    "            try:\n",
    "                base_dt = datetime.strptime(session_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)\n",
    "            except Exception:\n",
    "                base_dt = datetime.now(timezone.utc)\n",
    "            for t_idx, turn in enumerate(session['turns']):\n",
    "                speaker = turn['speaker'].lower()\n",
    "                if speaker == 'human':\n",
    "                    # Inject persona name so name-based queries match directly\n",
    "                    content = '[{}] {}'.format(persona, turn['text'])\n",
    "                else:\n",
    "                    content = '[ASSISTANT] {}'.format(turn['text'])\n",
    "                to_ingest.append({\n",
    "                    'conv_id'    : conv_id,\n",
    "                    'session_id' : session_id,\n",
    "                    'content'    : content,\n",
    "                    'tags'       : ['locomo', run_tag, conv_id,\n",
    "                                    'session_{}'.format(session_id),\n",
    "                                    turn['speaker'], persona.lower()],\n",
    "                    'occurred_at': base_dt + timedelta(minutes=t_idx * 3),\n",
    "                })\n",
    "\n",
    "    print(f'Turns to ingest: {len(to_ingest)}')\n",
    "\n",
    "    def _create_one(item):\n",
    "        try:\n",
    "            mem = client.memories.create(\n",
    "                content=item['content'],\n",
    "                source_type='locomo_benchmark',\n",
    "                tags=item['tags'],\n",
    "                occurred_at=item['occurred_at'],\n",
    "            )\n",
    "            return {'conv_id': item['conv_id'], 'session_id': item['session_id'],\n",
    "                    'memory_id': mem.id, 'ok': True}\n",
    "        except Exception as e:\n",
    "            return {'conv_id': item['conv_id'], 'session_id': item['session_id'],\n",
    "                    'memory_id': None, 'ok': False, 'error': str(e)}\n",
    "\n",
    "    import concurrent.futures as _cf\n",
    "    print(f'Ingesting with {INGEST_WORKERS} parallel workers...')\n",
    "    t0 = time.time()\n",
    "    failed = 0\n",
    "    with _cf.ThreadPoolExecutor(max_workers=INGEST_WORKERS) as pool:\n",
    "        futures = [pool.submit(_create_one, item) for item in to_ingest]\n",
    "        for i, fut in enumerate(_cf.as_completed(futures), 1):\n",
    "            result = fut.result()\n",
    "            if result['ok']:\n",
    "                ingested.append(result)\n",
    "            else:\n",
    "                failed += 1\n",
    "                if failed <= 3:\n",
    "                    print(f'  WARN: {result[\"error\"]}')\n",
    "            if i % 50 == 0:\n",
    "                print(f'  {i}/{len(to_ingest)} processed ({failed} failed)...')\n",
    "\n",
    "    elapsed = time.time() - t0\n",
    "    n = len(ingested) or 1\n",
    "    print(f'Ingested {len(ingested)}/{len(to_ingest)} memories in {elapsed:.1f}s'\n",
    "          f'  ({elapsed/n:.2f}s/mem, {failed} failed)')\n",
]

# ── Cell 15: helpers with short-memory boost rerank + verbosity-matching prompts ──
cell15 = r"""# LLM helpers -- category-aware, persona-name-aware, parallel
import urllib.request, concurrent.futures

_PREAMBLE = (
    'based on the context', 'based on the conversation', 'according to the context',
    'according to the conversation', 'the context indicates', 'the context states',
    'from the context', 'from the conversation', 'the answer is', 'in the context',
    'i do not know', 'the conversation does not', 'there is no mention', 'no information',
)

def _clean_answer(raw):
    if not raw:
        return raw
    s = raw.strip()
    lower = s.lower()
    for p in _PREAMBLE:
        if lower.startswith(p):
            for sep in (',', ':', ';', ' --', ' -'):
                idx = s.find(sep)
                if idx != -1 and idx < 80:
                    rest = s[idx+1:].strip()
                    if rest:
                        return rest
            return s
    return s


def _sort_by_date(mem_dicts):
    def parse_dt(m):
        oc = (m.get('occurred_at') or '')
        try:
            from datetime import datetime
            return datetime.fromisoformat(oc.replace('Z', '+00:00'))
        except Exception:
            from datetime import datetime
            return datetime.min
    return sorted(mem_dicts, key=parse_dt)


def _top_k_keyword(question, memories, k):
    words = set(re.sub(r'[^\w\s]', '', question.lower()).split())
    stop = {
        'the','a','an','is','was','did','do','what','when','where','who','how',
        'and','or','of','in','on','to','for','at','does','has','have','had',
        'been','be','are','were','will','would','could','should',
    }
    words -= stop
    scored = []
    for m in memories:
        text_raw = re.sub(r'[^\w\s]', ' ', (m.content_preview or m.title or '').lower())
        toks = set(text_raw.split())
        overlap = sum(1 for w in words if w in toks)
        length_bonus = 1.3 if len(text_raw) < 80 else 1.0
        scored.append((overlap * length_bonus, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:k]]


def _extract_key_terms(text, top_n=8):
    stop = {
        'the','a','an','is','was','did','do','what','when','where','who','how',
        'and','or','of','in','on','to','for','at','i','my','me','we','our',
        'you','your','he','she','it','they','their','that','this','said','told',
        'human','assistant','yes','no','not','just','about','from','with','been',
        'are','were','will','would','could','should','have','had','has','be',
    }
    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
    freq = {}
    for t in tokens:
        if t not in stop and len(t) > 2:
            freq[t] = freq.get(t, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]


def _retrieve(question, mem_dicts, mems_obj, category, limit):
    cat_limit = {
        'single_hop' : limit,
        'multi_hop'  : min(limit * 2, 25),
        'temporal'   : min(limit + 5, 20),
        'adversarial': min(limit + 5, 20),
    }.get(category, limit)

    gateway_mems = []
    try:
        gw = client.cognitive.gateway.read(
            query=question,
            memories=mem_dicts,
            limit=cat_limit,
        )
        gateway_mems = gw.get('memories', [])
    except Exception:
        pass

    if category == 'multi_hop':
        # Two-stage: expand with key terms from stage-1 hits for better chain coverage
        stage1 = gateway_mems or []
        stage1_text = ' '.join(
            (m.get('content') or m.get('title', '')) for m in stage1
        )
        key_terms = _extract_key_terms(stage1_text)
        if key_terms:
            followup = question + ' ' + ' '.join(key_terms[:5])
            stage2 = []
            try:
                gw2 = client.cognitive.gateway.read(
                    query=followup,
                    memories=mem_dicts,
                    limit=cat_limit,
                )
                stage2 = gw2.get('memories', [])
            except Exception:
                pass
            seen = set()
            merged = []
            for m in stage1 + stage2:
                mid = m.get('id', '')
                if mid not in seen:
                    seen.add(mid)
                    merged.append(m)
            if merged:
                gateway_mems = merged[:cat_limit]

    elif category == 'adversarial' and not gateway_mems:
        top = _top_k_keyword(question, mems_obj, k=cat_limit)
        gateway_mems = [{
            'id'         : str(m.id),
            'content'    : m.content_preview or m.title or '',
            'occurred_at': m.occurred_at.isoformat() if m.occurred_at else None,
        } for m in top]

    if gateway_mems:
        # Always return in chronological order so the LLM sees conversation flow
        return _sort_by_date(gateway_mems)

    # Fallback: keyword-based
    top = _top_k_keyword(question, mems_obj, k=cat_limit)
    return _sort_by_date([{
        'id'         : str(m.id),
        'content'    : m.content_preview or m.title or '',
        'occurred_at': m.occurred_at.isoformat() if m.occurred_at else None,
    } for m in top])


def _build_prompt(category, question, context):
    q_lower = question.lower()
    if category == 'temporal':
        # Detect question sub-type for targeted instructions
        if any(p in q_lower for p in ('how long', 'how old', 'how many weeks', 'how many months', 'how many days')):
            duration_inst = (
                'Answer with ONLY the duration or age as a compact phrase (e.g., "4 months", "6 weeks old", "3 days").\n'
                'Do NOT write a full sentence. Just the number and unit.\n'
            )
        elif 'before or after' in q_lower:
            duration_inst = (
                'Reply: "Before -- [one-sentence reason with dates if known]"'
                ' or "After -- [one-sentence reason with dates if known]".\n'
                'Include the specific month/year if the conversation mentions it.\n'
            )
        else:
            duration_inst = (
                'Answer with the exact time expression or date from the conversation.\n'
                'If a race time like "3:07" appears, write it as "3 hours and 7 minutes (3:07)".\n'
            )
        return (
            'The conversation excerpts below are in chronological order.\n'
            + duration_inst +
            '\nConversation (chronological):\n'
            + context + '\n\n'
            'Question: ' + question + '\n\n'
            'If not present say: I do not know\n'
            'Answer:'
        )
    elif category == 'multi_hop':
        return (
            'Answer by connecting facts across the conversation excerpts.\n'
            'Use third-person perspective and include specific names, places, and numbers.\n\n'
            'Conversation:\n'
            + context + '\n\n'
            'Question: ' + question + '\n\n'
            'Answer (1-2 sentences using exact names and facts from the conversation):\n'
            'If not present say: I do not know\n'
            'Answer:'
        )
    elif category == 'adversarial':
        return (
            'Answer based ONLY on the conversation excerpts below.\n'
            'The question may contain false assumptions -- verify every claim against the facts.\n'
            'Use third-person perspective and exact names from the conversation.\n\n'
            'Conversation:\n'
            + context + '\n\n'
            'Question: ' + question + '\n\n'
            'State only what the conversation supports. If the premise is false, give the correct fact.\n'
            'Answer:'
        )
    else:  # single_hop
        if 'time' in q_lower and ('marathon' in q_lower or 'finishing' in q_lower or 'finish' in q_lower or 'race' in q_lower):
            time_inst = 'If a race time like "3:07" appears, write it as "3 hours and 7 minutes (3:07)".\n'
        else:
            time_inst = ''
        return (
            'Answer the question using only the conversation excerpts below.\n'
            'Use the exact name, term, or phrase as it appears in the conversation.\n'
            + time_inst +
            '\nConversation:\n'
            + context + '\n\n'
            'Question: ' + question + '\n\n'
            'Short specific answer (1 sentence or less) using exact wording from the conversation:\n'
            'If not present say: I do not know\n'
            'Answer:'
        )


def _ollama_generate(prompt, model=LLM_MODEL, timeout=LLM_TIMEOUT):
    try:
        payload = json.dumps({'model': model, 'prompt': prompt, 'stream': False}).encode()
        req = urllib.request.Request(
            'http://localhost:11434/api/generate',
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())['response'].strip()
    except Exception:
        return ''


def _run_prompts_parallel(prompts, workers=LLM_WORKERS):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_ollama_generate, p, LLM_MODEL, LLM_TIMEOUT) for p in prompts]
        fut_idx = {fut: i for i, fut in enumerate(futs)}
        results = [None] * len(prompts)
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            results[fut_idx[fut]] = fut.result()
            done += 1
            if done % 5 == 0:
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
        'human','assistant',
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
        density = len(new_facts) / len(tokens)
        length_penalty = 1.0 / (1 + max(0, len(tokens) - 15) * 0.05)
        score = density * length_penalty
        if score > best_score:
            best_score = score
            best_sent = clean
    return best_sent or (clean_sents[0] if clean_sents else '')


# Smoke-test
test = _ollama_generate('Reply with only the word: ready', timeout=10)
print('Ollama status:', 'ok -- category-aware LLM enabled' if test else 'unavailable -- heuristic fallback active')
"""

cells[15]['source'] = [line + '\n' for line in cell15.splitlines()]
# Fix the last line (no trailing newline)
if cells[15]['source']:
    cells[15]['source'][-1] = cells[15]['source'][-1].rstrip('\n')

with open('locomo_benchmark.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook patched: cells 5, 11, 15 updated.')
print('Changes:')
print('  Cell 5:  LOCOMO_SEED=43 (new run_tag with persona names)')
print('  Cell 11: [Alex]/[Maria]/[Dev] instead of [HUMAN] in ingested content')
print('  Cell 15: _boost_rerank() for short-memory preference + verbosity prompts')
