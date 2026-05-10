import json
from collections import Counter

RESULTS_FILE = 'd:/Sansten/Projects/Ninai2/locomo_results_latest.json'

with open(RESULTS_FILE) as f:
    data = json.load(f)

results = data['results']
categories = ['single_hop', 'multi_hop', 'temporal', 'open_domain', 'adversarial']

def classify_failure(row):
    gold = (row.get('gold_answer') or '').strip().lower()
    gen  = (row.get('generated_answer') or '').strip().lower()
    r1   = row.get('rouge1_f1', 0)

    if r1 >= 90:
        return 'correct'
    if r1 == 0 and gen in ('not mentioned', 'none', '', 'not found', 'unknown'):
        if gold in ('not mentioned', 'none', ''):
            return 'correct_abstention'
        return 'wrong_abstention'       # model said "not mentioned" but gold has a real answer
    if r1 == 0 and gold in ('not mentioned', 'none', ''):
        return 'wrong_answer_should_abstain'  # model gave answer when it should abstain
    if r1 == 0:
        return 'wrong_answer'           # retrieved wrong fact entirely
    if 0 < r1 < 40:
        return 'partial_paraphrase'     # right idea, wrong words
    if 40 <= r1 < 90:
        return 'near_miss'              # close but missing words (truncation, modifier drop)
    return 'other'

print('\n=== FAILURE MODE DISTRIBUTION ===\n')
for cat in categories:
    rows = [r for r in results if r['category'] == cat]
    counts = Counter(classify_failure(r) for r in rows)
    total  = len(rows)
    avg    = sum(r.get('rouge1_f1', 0) for r in rows) / total if total else 0
    sem_rows = [r.get('semantic_correct', -1) for r in rows if r.get('semantic_correct', -1) >= 0]
    sem_pct  = round(sum(sem_rows) / len(sem_rows) * 100, 1) if sem_rows else None
    sem_str  = f'  semantic={sem_pct}%' if sem_pct is not None else ''
    print(f'{cat} (n={total}, rouge1={avg:.1f}{sem_str}):')
    for mode, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        print(f'  {mode:<35} {cnt:>4} ({pct:.0f}%)')
    print()

print('\n=== 5 WORST EXAMPLES PER CATEGORY ===\n')
for cat in categories:
    rows = sorted(
        [r for r in results if r['category'] == cat],
        key=lambda x: x.get('rouge1_f1', 0)
    )
    print(f'--- {cat} ---')
    for r in rows[:5]:
        sem = r.get('semantic_correct', -1)
        sem_str = f'  sem={sem}' if sem >= 0 else ''
        print(f'  Q   : {r["question"][:90]}')
        print(f'  Gold: {r.get("gold_answer","")[:60]}')
        print(f'  Gen : {r.get("generated_answer","")[:60]}')
        print(f'  R1  : {r.get("rouge1_f1",0):.1f}{sem_str}  |  mode: {classify_failure(r)}')
        print()

print('\n=== 5 BEST EXAMPLES (to confirm what IS working) ===\n')
for cat in categories:
    rows = sorted(
        [r for r in results if r['category'] == cat],
        key=lambda x: -x.get('rouge1_f1', 0)
    )
    print(f'--- {cat} ---')
    for r in rows[:3]:
        print(f'  Q   : {r["question"][:90]}')
        print(f'  Gold: {r.get("gold_answer","")[:60]}')
        print(f'  Gen : {r.get("generated_answer","")[:60]}')
        print(f'  R1  : {r.get("rouge1_f1",0):.1f}')
    print()

print('\n=== ZERO-SCORE RATE (ROUGE-1 = 0) ===\n')
for cat in categories:
    rows = [r for r in results if r['category'] == cat]
    zeros = sum(1 for r in rows if r.get('rouge1_f1', 0) == 0)
    print(f'  {cat:<15}: {zeros}/{len(rows)} = {zeros/len(rows)*100:.0f}% zero-score')

# Overall semantic summary
all_sem = [r.get('semantic_correct', -1) for r in results if r.get('semantic_correct', -1) >= 0]
if all_sem:
    print(f'\n=== OVERALL SEMANTIC ACCURACY ===\n')
    print(f'  {sum(all_sem)}/{len(all_sem)} = {sum(all_sem)/len(all_sem)*100:.1f}% judged correct')
    print(f'  ({len(results) - len(all_sem)} pairs excluded — judge timed out)')
