import json

with open('locomo_results_20260426_2219_quick_55q.json') as f:
    data = json.load(f)

results = data.get('results', [])
print(f'Total results: {len(results)}')

# Show multi_hop failures
print('\n=== MULTI_HOP FAILURES (rouge1 < 5) ===')
mh_fail = [r for r in results if r['category'] == 'multi_hop' and r['rouge1_f1'] < 5]
print(f'Count: {len(mh_fail)}/{len([r for r in results if r["category"]=="multi_hop"])}')
for r in mh_fail[:12]:
    q = r['question'][:85]
    gold = r['gold_answer'][:65]
    gen = r['generated_answer'][:65]
    rc = r.get('retrieved_count', '?')
    print(f'  Q: {q}')
    print(f'  Gold: {gold}')
    print(f'  Gen:  {gen}')
    print(f'  R1={r["rouge1_f1"]:.1f}  retrieved={rc}')
    print()

print('\n=== ADVERSARIAL FAILURES (rouge1 < 5) ===')
adv_fail = [r for r in results if r['category'] == 'adversarial' and r['rouge1_f1'] < 5]
print(f'Count: {len(adv_fail)}/{len([r for r in results if r["category"]=="adversarial"])}')
for r in adv_fail[:12]:
    q = r['question'][:85]
    gold = r['gold_answer'][:65]
    gen = r['generated_answer'][:65]
    rc = r.get('retrieved_count', '?')
    print(f'  Q: {q}')
    print(f'  Gold: {gold}')
    print(f'  Gen:  {gen}')
    print(f'  R1={r["rouge1_f1"]:.1f}  retrieved={rc}')
    print()

# Not-mentioned breakdown for adversarial
nm = [r for r in results if r['category'] == 'adversarial' and 'not mentioned' in r['generated_answer'].lower()]
print(f'\nAdversarial "Not mentioned" count: {len(nm)}/{len([r for r in results if r["category"]=="adversarial"])}')

# Multi_hop: check yes/no pattern
yn_wrong = [r for r in results if r['category'] == 'multi_hop' and r['rouge1_f1'] < 5 and r['generated_answer'].strip().lower() in ('yes', 'no', 'likely yes', 'likely no')]
print(f'\nMulti_hop bare Yes/No misses: {len(yn_wrong)}')
for r in yn_wrong:
    print(f'  Gen={r["generated_answer"]!r}  Gold={r["gold_answer"][:60]!r}')
