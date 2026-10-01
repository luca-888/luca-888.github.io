"""Summarize a run directory into summary.json (speedup vs matched baseline, acceptance, KV-limit flags)."""
import glob, json, os, re, sys, statistics
RUN, OUT = sys.argv[1], sys.argv[2]
MATCH = lambda n: 'baseline' if n.startswith('eagle3') else 'baseline-mrv1-noasync' if (n.startswith('ngram') and not n.startswith('ngram_gpu')) else 'baseline-mrv1'
names = sorted(p.split('/')[-2] for p in glob.glob(f'{RUN}/*/cells.jsonl'))
cells, pools, meta = [], {}, {}
D = {}
for n in names:
    D[n] = {}
    for l in open(f'{RUN}/{n}/cells.jsonl'):
        r = json.loads(l); D[n][(r['task'], r['temp'], r['c'])] = r
    m = re.search(r'GPU KV cache size: ([\d,]+) tokens', open(f'{RUN}/{n}.log', errors='ignore').read())
    pools[n] = int(m.group(1).replace(',', '')) if m else None
    meta[n] = json.load(open(f'{RUN}/{n}/meta.json'))
for n in names:
    for (task, temp, c), r in sorted(D[n].items()):
        row = {'cfg': n, 'method': meta[n]['method'], 'k': meta[n]['k'], 'task': task, 'temp': temp, 'c': c,
               'tok_per_s': r['tok_per_s']}
        toks = statistics.mean(o['n_in'] + o['n_out'] for o in r['outputs'])
        row['demand_tokens'] = round(c * toks * 1.05)
        row['kv_pool'] = pools[n]
        row['kv_limited'] = bool(pools[n] and row['demand_tokens'] > pools[n])
        if r['spec']:
            s = r['spec']
            row['al'] = 1 + s['num_accepted_tokens'] / s['num_drafts']
            row['pos'] = [x / s['num_drafts'] for x in s['num_accepted_tokens_per_pos']]
        if not n.startswith('baseline'):
            b = D[MATCH(n)][(task, temp, c)]
            row['matched'] = MATCH(n); row['speedup'] = r['tok_per_s'] / b['tok_per_s']
            row['speedup_vs_plain'] = r['tok_per_s'] / D['baseline'][(task, temp, c)]['tok_per_s']
        cells.append(row)
# T=0 exact-match rate vs plain baseline in the same cell (numeric-noise control: baseline c=1 vs c=X)
def eq_rate(n):
    tot = eq = 0
    for k, r in D[n].items():
        if k[1] != 0: continue
        for o, b in zip(r['outputs'], D['baseline'][k]['outputs']):
            tot += 1; eq += o['ids'] == b['ids']
    return eq, tot
exact = {n: eq_rate(n) for n in names if n != 'baseline'}
json.dump({'configs': {n: {'method': meta[n]['method'], 'k': meta[n]['k'], 'kv_pool_tokens': pools[n], 'load_s': meta[n]['load_s'], 'mrv1': meta[n].get('mrv1', False), 'no_async': meta[n].get('no_async', False)} for n in names},
           'cells': cells, 'exact_match_vs_baseline': exact}, open(OUT, 'w'))
print(len(cells), 'cells;', sum(c['kv_limited'] for c in cells), 'kv-limited')
