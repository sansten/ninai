import json

with open('locomo_results_20260426_2219_quick_55q.json') as f:
    run1 = {r['qa_id']: r for r in json.load(f)['results']}

with open('locomo_results_20260426_2232_quick_55q.json') as f:
    run2 = {r['qa_id']: r for r in json.load(f)['results']}

mh1 = [(k, v) for k, v in run1.items() if v['category'] == 'multi_hop']
mh2 = {k: v for k, v in run2.items() if v['category'] == 'multi_hop'}

print("=== MULTI_HOP: run1 vs run2 ===")
print(f"{'qa_id':<22} {'R1_v1':>7} {'R1_v2':>7}  {'Gold':<50}  {'Gen_v1':<50}  {'Gen_v2'}")
improved, regressed, same = [], [], []
for qa_id, r1 in sorted(mh1, key=lambda x: x[1]['rouge1_f1']):
    r2 = mh2.get(qa_id, {})
    diff = r2.get('rouge1_f1', 0) - r1['rouge1_f1']
    entry = (qa_id, r1['rouge1_f1'], r2.get('rouge1_f1', 0), r1['gold_answer'][:55], r1['generated_answer'][:55], r2.get('generated_answer', '')[:55])
    if diff > 3:
        improved.append(entry)
    elif diff < -3:
        regressed.append(entry)
    else:
        same.append(entry)

print(f"\nIMPROVED ({len(improved)}):")
for e in improved:
    print(f"  {e[0]}: {e[1]:.1f} -> {e[2]:.1f}")
    print(f"    Gold:  {e[3]}")
    print(f"    v1:    {e[4]}")
    print(f"    v2:    {e[5]}")

print(f"\nREGRESSED ({len(regressed)}):")
for e in regressed:
    print(f"  {e[0]}: {e[1]:.1f} -> {e[2]:.1f}")
    print(f"    Gold:  {e[3]}")
    print(f"    v1:    {e[4]}")
    print(f"    v2:    {e[5]}")

adv1 = [(k, v) for k, v in run1.items() if v['category'] == 'adversarial']
adv2 = {k: v for k, v in run2.items() if v['category'] == 'adversarial'}
print(f"\n=== ADVERSARIAL REGRESSIONS ===")
for qa_id, r1 in adv1:
    r2 = adv2.get(qa_id, {})
    diff = r2.get('rouge1_f1', 0) - r1['rouge1_f1']
    if diff < -3:
        print(f"  {qa_id}: {r1['rouge1_f1']:.1f} -> {r2.get('rouge1_f1',0):.1f}")
        print(f"    Gold:  {r1['gold_answer'][:60]}")
        print(f"    v1:    {r1['generated_answer'][:60]}")
        print(f"    v2:    {r2.get('generated_answer','')[:60]}")
