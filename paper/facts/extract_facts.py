"""Every number the paper quotes, extracted from the Polyp-PVT repo's outputs.
Usage: python extract_facts.py /path/to/polyp-pvt   -> facts.json and facts.md next to this file."""
import csv, datetime as dt, json, math, os, re, statistics as st, sys

R = sys.argv[1]
OUT = os.path.dirname(os.path.abspath(__file__))
SETS = ['Kvasir', 'ClinicDB', 'ColonDB', 'CVC-300', 'ETIS', 'mean']
LOGSET = {'Kvasir': 'Kvasir', 'CVC-ClinicDB': 'ClinicDB', 'CVC-ColonDB': 'ColonDB', 'CVC-300': 'CVC-300',
          'ETIS-LaribPolypDB': 'ETIS'}
T975 = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582}   # t(0.975, df)
F = {}


def summary(path, van, fct):
    """The last six-seed seed_summary block for fct vs van: per-set statistics and per-run values."""
    text = open(os.path.join(R, path)).read()
    head = 'difference = %s - %s with the same seed' % (fct, van)
    i = text.rfind(head)
    assert i > 0 and 'seeds [1, 2, 3, 4, 5, 6]' in text[i - 60:i], (path, fct)
    start = text.rfind('meanDic per run', 0, i)
    runs = {}
    for line in text[start:i].splitlines()[1:]:
        m = re.match(r'^(\S+)_s(\d)\s+([\d.\s]+?)(\s+seed.*)?$', line.strip())
        if m and m.group(1) in (van, fct):
            runs.setdefault(m.group(1), {})[int(m.group(2))] = [float(x) for x in m.group(3).split()]
    stats = {}
    for line in text[i:].splitlines()[2:9]:
        m = re.match(r'^(\S+)\s+([\d.]+)\+-([\d.]+)\s+([\d.]+)\+-([\d.]+)\s+([+-][\d.]+)\s+\[\s*([+-][\d.]+),\s*([+-][\d.]+)\]'
                     r'\s+([\d.]+)\s+([\d.]+)\s+(\d)/6', line.strip())
        if m:
            g = m.groups()
            stats[g[0]] = dict(van=float(g[1]), van_sd=float(g[2]), fct=float(g[3]), fct_sd=float(g[4]), diff=float(g[5]),
                               lo=float(g[6]), hi=float(g[7]), p_t=float(g[8]), p_perm=float(g[9]), wins=int(g[10]))
    assert set(stats) == set(SETS), (path, fct, stats.keys())
    assert len(runs.get(van, {})) == 6 and len(runs.get(fct, {})) == 6, (path, fct, {k: len(v) for k, v in runs.items()})
    return dict(stats=stats, runs=runs)


def paired(a, b):
    """mean of b - a over the seeds, 95% t interval with len - 1 degrees of freedom, wins."""
    d = [y - x for x, y in zip(a, b)]
    m, s = st.mean(d), st.stdev(d)
    h = T975[len(d) - 1] * s / math.sqrt(len(d))
    return dict(diff=m, lo=m - h, hi=m + h, wins=sum(x > 0 for x in d))


S = {}
S['pvt_s3lmup'] = summary('seeds_s3lmup.out', 'PVT_VANILLA', 'S3LMUP')
S['pvt_s3lm'] = summary('seeds_s3lm.out', 'PVT_VANILLA', 'S3LM')
S['pvt_s3sig'] = summary('seeds_hosts.out', 'PVT_VANILLA', 'S3SIG')
for a in ['A05', 'A10', 'A15']:
    S['pvt_' + a.lower()] = summary('seeds_sweep.out', 'PVT_VANILLA', 'S3LMUP_' + a)
S['pranet_s3lmup'] = summary('seeds_pranetv2.out', 'PRANETV2_VANILLA', 'PRANETV2_S3LMUP')
if os.path.isfile(os.path.join(R, 'results', 'ASGNET_1288.md')):      # recorded by hand with its seed_summary block
    S['asgnet_s3lmup'] = summary(os.path.join('results', 'ASGNET_1288.md'), 'ASGNET_VANILLA', 'ASGNET_S3LMUP')
# Component dissection (Polyp-PVT). An arm is filled in once it has at least three seeds: from
# full_precision.json when that holds its maps' scores, else from the last seed_summary block that
# lists it. Its statistics are paired with vanilla on the arm's own seeds (three or six).
import glob
DISSECT = {'pvt_s3': 'S3', 'pvt_s3up': 'S3UP', 'pvt_s3fup': 'S3FUP', 'pvt_s3relup': 'S3RELUP', 'pvt_s3f': 'S3F'}
FP_PATH = os.path.join(OUT, 'full_precision.json')
FPJ = json.load(open(FP_PATH)) if os.path.isfile(FP_PATH) else {}
SETFULL = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
def runs_from_fp(arm):
    out = {}
    for sd in range(1, 7):
        k = '%s_s%d' % (arm, sd)
        if k in FPJ and all(c in FPJ[k] for c in SETFULL):
            v = [FPJ[k][c][0] for c in SETFULL]
            out[sd] = v + [sum(v) / 5]
    return out
def runs_from_summary(arm):
    best = {}
    for path in sorted(glob.glob(os.path.join(R, 'seeds*.out')) + glob.glob(os.path.join(R, 'results', '*.md')),
                       key=os.path.getmtime):
        text = open(path, errors='replace').read()
        i = text.rfind('difference = %s - PVT_VANILLA with the same seed' % arm)
        if i < 0:
            continue
        runs = {}
        for line in text[text.rfind('meanDic per run', 0, i):i].splitlines()[1:]:
            m = re.match(r'^(\S+)_s(\d)\s+([\d.\s]+?)(\s+seed.*)?$', line.strip())
            if m and m.group(1) == arm:
                runs[int(m.group(2))] = [float(x) for x in m.group(3).split()]
        if len(runs) >= len(best):
            best = runs
    return best
for key, arm in DISSECT.items():
    r = runs_from_fp(arm) or runs_from_summary(arm)
    if len(r) < 3:
        continue
    seeds = sorted(r)
    van_r = S['pvt_s3lmup']['runs']['PVT_VANILLA']
    stats = {}
    for j, name in enumerate(SETS):
        a, b = [van_r[sd][j] for sd in seeds], [r[sd][j] for sd in seeds]
        stats[name] = dict(paired(a, b), van=st.mean(a), van_sd=st.stdev(a), fct=st.mean(b), fct_sd=st.stdev(b))
    S[key] = dict(stats=stats, runs={'PVT_VANILLA': {sd: van_r[sd] for sd in seeds}, arm: r}, seeds=seeds)
F['dissection_pending'] = [DISSECT[k] for k in DISSECT if k not in S]

# Full precision: seed_summary prints per-run scores to 4 decimals, which shifts derived intervals in
# the last digit. full_precision.py rescored every run from its saved maps; when its output exists,
# the per-run values and every statistic derived from them are recomputed from it.
FP_PATH = os.path.join(OUT, 'full_precision.json')
SETFULL = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
F['full_precision'] = os.path.isfile(FP_PATH)
if F['full_precision']:
    FPJ = json.load(open(FP_PATH))
    def fp_runs(arm):
        return runs_from_fp(arm)
    changed = []
    for key, rec in S.items():
        sds = rec.get('seeds', list(range(1, 7)))
        replaced = 0
        for arm in list(rec['runs']):
            if all('%s_s%d' % (arm, sd) in FPJ for sd in sds):
                old = rec['runs'][arm]
                new_runs = fp_runs(arm)
                rec['runs'][arm] = {sd: new_runs[sd] for sd in sds}
                assert all(abs(old[sd][j] - rec['runs'][arm][sd][j]) < 6e-5 for sd in sds for j in range(6)), (key, arm)
                replaced += 1
        if replaced < 2:                # no saved maps here (ASGNet): keep seed_summary's own statistics
            continue
        van_arm, fct_arm = [a for a in rec['runs'] if 'VANILLA' in a][0], [a for a in rec['runs'] if 'VANILLA' not in a][0]
        for j, name in enumerate(SETS):
            a = [rec['runs'][van_arm][sd][j] for sd in sds]
            b = [rec['runs'][fct_arm][sd][j] for sd in sds]
            new = paired(a, b)
            new.update(van=st.mean(a), van_sd=st.stdev(a), fct=st.mean(b), fct_sd=st.stdev(b))
            old = rec['stats'][name]
            for f in ['van', 'van_sd', 'fct', 'fct_sd', 'diff', 'lo', 'hi']:
                if '%.2f' % (100 * old[f]) != '%.2f' % (100 * new[f]):
                    changed.append('%s %s %s: %.2f -> %.2f' % (key, name, f, 100 * old[f], 100 * new[f]))
            if old['wins'] != new['wins']:
                changed.append('%s %s wins %d -> %d' % (key, name, old['wins'], new['wins']))
            old.update({f: new[f] for f in ['van', 'van_sd', 'fct', 'fct_sd', 'diff', 'lo', 'hi', 'wins']})
    F['full_precision_changes'] = changed
    print('full precision: %d two-decimal changes against the 4-decimal summaries' % len(changed))
    for c in changed:
        print('  ', c)
F['summaries'] = {k: v['stats'] for k, v in S.items()}
F['runs'] = {k: v['runs'] for k, v in S.items()}
# the vanilla runs must be the same runs in every Polyp-PVT summary
van = S['pvt_s3lmup']['runs']['PVT_VANILLA']
for k in ['pvt_s3lm', 'pvt_s3sig', 'pvt_a05', 'pvt_a10', 'pvt_a15'] + [k for k in DISSECT if k in S]:
    assert all(S[k]['runs']['PVT_VANILLA'][sd] == van[sd] for sd in S[k]['runs']['PVT_VANILLA']), k

# head to head, paired over seeds, from the per-run values (4-decimal)
def col(runs, arm, j):
    return [runs[arm][s][j] for s in range(1, 7)]
H = {}
lmup = S['pvt_s3lmup']['runs']['S3LMUP']
for name, key, arm in [('S3LMUP - S3LM', 'pvt_s3lm', 'S3LM'), ('S3LMUP - S3SIG', 'pvt_s3sig', 'S3SIG'),
                       ('S3LMUP(a=2) - a=0.5', 'pvt_a05', 'S3LMUP_A05'), ('S3LMUP(a=2) - a=1', 'pvt_a10', 'S3LMUP_A10'),
                       ('S3LMUP(a=2) - a=1.5', 'pvt_a15', 'S3LMUP_A15')]:
    other = S[key]['runs'][arm]
    H[name] = {SETS[j]: paired([other[s][j] for s in range(1, 7)], [lmup[s][j] for s in range(1, 7)]) for j in range(6)}
# dissection effects, paired over the seeds both rows have: each pull alone and both on top of the
# three views; each pull on top of the other; FCT minus each change of its weight function; and the
# lag-or-margin weight minus the confidence-graded one (both without the upright pull)
ARMOF = dict(DISSECT, pvt_s3lmup='S3LMUP', pvt_s3lm='S3LM', pvt_s3sig='S3SIG')
STEPS = [('S3UP - S3', 'pvt_s3', 'pvt_s3up'), ('S3LM - S3', 'pvt_s3', 'pvt_s3lm'), ('S3LMUP - S3', 'pvt_s3', 'pvt_s3lmup'),
         ('S3LMUP - S3LM', 'pvt_s3lm', 'pvt_s3lmup'), ('S3LMUP - S3UP', 'pvt_s3up', 'pvt_s3lmup'),
         ('S3LMUP - S3FUP', 'pvt_s3fup', 'pvt_s3lmup'), ('S3LMUP - S3RELUP', 'pvt_s3relup', 'pvt_s3lmup'),
         ('S3LM - S3SIG', 'pvt_s3sig', 'pvt_s3lm'), ('S3LM - S3F', 'pvt_s3f', 'pvt_s3lm')]
D = {}
for name, ka, kb in STEPS:
    if ka in S and kb in S:
        ra, rb = S[ka]['runs'][ARMOF[ka]], S[kb]['runs'][ARMOF[kb]]
        sds = sorted(set(ra) & set(rb))
        D[name] = {SETS[j]: dict(paired([ra[sd][j] for sd in sds], [rb[sd][j] for sd in sds]), n=len(sds)) for j in range(6)}
F['dissection_seeds'] = {k: len(S[k].get('seeds', range(1, 7))) for k in DISSECT if k in S}
F['dissection_steps'] = D
F['head_to_head'] = H

# flip gap, mean over seeds, per set
rows = list(csv.reader(open(os.path.join(R, 'flip_gap.tsv')), delimiter='\t'))
head = rows[0]
def gap(arm, single=False):
    pat = re.escape(arm) if single else r'%s_s[1-6]/PolypPVT' % re.escape(arm)
    keep = [r for r in rows[1:] if re.fullmatch(pat, r[0])]
    assert keep and (single or len(keep) == 6), (arm, len(keep))
    out = {c: 100 * st.mean(float(r[head.index(c)]) for r in keep) for c in head[1:]}
    out['per_seed'] = {c: [100 * float(r[head.index(c)]) for r in keep] for c in head[1:]}
    out['n'] = len(keep)
    return out
G = {a: gap(a) for a in ['PVT_VANILLA', 'S3LMUP', 'S3LM', 'S3SIG', 'S3LMUP_A05', 'S3LMUP_A10', 'S3LMUP_A15',
                         'PRANETV2_VANILLA', 'PRANETV2_S3LMUP']}
for a in ['PVT_AUG', 'PVT_AUG_FCT', 'S3F', 'S3']:
    G[a] = gap(a, single=True)
for key, a in DISSECT.items():      # the dissection arms: the flip gap over the seeds each was trained with
    if key in S:
        sds = S[key]['seeds']
        keep = [r for r in rows[1:] if re.fullmatch(r'%s_s[%s]/PolypPVT' % (re.escape(a), ''.join(map(str, sds))), r[0])]
        if len(keep) == len(sds):
            G[a + ' (all seeds)'] = {c: 100 * st.mean(float(r[head.index(c)]) for r in keep) for c in head[1:]}
F['flip_gap'] = G
for v, f in [('PVT_VANILLA', 'S3LMUP'), ('PRANETV2_VANILLA', 'PRANETV2_S3LMUP')]:
    F.setdefault('flip_gap_every_seed_lower', {})[f] = {
        c: all(b < a for a, b in zip(G[v]['per_seed'][c], G[f]['per_seed'][c])) for c in head[1:]}

# training logs: time per epoch, gradient diagnostics, mechanism counters, per-epoch evaluation
LINE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?) Epoch \[(\d+)/(\d+)\], Step \[(\d+)/(\d+)\](.*)$')
def log(arm, seed):
    path = os.path.join(R, 'logs_%s_s%d.txt' % (arm, seed))
    pts, cos, ratio, last, evals, cur = [], [], [], '', [], {}
    for line in open(path, errors='replace'):
        m = LINE.match(line.strip())
        if m:
            t = dt.datetime.fromisoformat(m.group(1)); e, E, s, S_ = map(int, m.groups()[1:5])
            pts.append((t, e - 1 + s / S_))
            c = re.search(r'cos: ([+-][\d.]+)', m.group(6)); r_ = re.search(r'\|g_cons\|/\|g_sup\|: ([\d.]+)', m.group(6))
            if c: cos.append(float(c.group(1)))
            if r_: ratio.append(float(r_.group(1)))
            last = m.group(6)
            continue
        m = re.match(r'^(CVC-300|CVC-ClinicDB|Kvasir|CVC-ColonDB|ETIS-LaribPolypDB) :\s+([\d.]+)', line.strip())
        if m:
            cur[LOGSET[m.group(1)]] = float(m.group(2)); continue
        m = re.match(r'^mean:\s+([\d.]+)', line.strip())
        if m:
            cur['mean'] = float(m.group(1)); evals.append(cur); cur = {}
    (t0, p0), (t1, p1) = pts[0], pts[-1]
    counters = {k: float(v) for k, v in re.findall(r'(up|lead|wl): ([\d.]+)', last)}
    return dict(min_per_epoch=(t1 - t0).total_seconds() / 60 / (p1 - p0), cos=st.mean(cos) if cos else None,
                ratio=st.mean(ratio) if ratio else None, counters=counters, evals=evals)
L = {}
for arm in ['PVT_VANILLA', 'S3LMUP', 'S3LM', 'S3SIG', 'PRANETV2_VANILLA', 'PRANETV2_S3LMUP']:
    L[arm] = [log(arm, s) for s in range(1, 7)]
for arm in ['ASGNET_VANILLA', 'ASGNET_S3LMUP']:
    if all(os.path.isfile(os.path.join(R, 'logs_%s_s%d.txt' % (arm, sd))) for sd in range(1, 7)):
        L[arm] = [log(arm, sd) for sd in range(1, 7)]
for key, arm in DISSECT.items():    # dissection arms with flip pulls: gradient statistics over their seeds
    if key in S and arm in ('S3F', 'S3FUP', 'S3RELUP'):
        sds = S[key]['seeds']
        if all(os.path.isfile(os.path.join(R, 'logs_%s_s%d.txt' % (arm, sd))) for sd in sds):
            L[arm] = [log(arm, sd) for sd in sds]
s3f = log.__wrapped__ if hasattr(log, '__wrapped__') else None
F['time_min_per_epoch'] = {a: st.median(x['min_per_epoch'] for x in v) for a, v in L.items()}
F['grad_cos'] = {a: st.mean(x['cos'] for x in v) for a, v in L.items() if v[0]['cos'] is not None}
F['grad_ratio'] = {a: st.mean(x['ratio'] for x in v) for a, v in L.items() if v[0]['ratio'] is not None}
F['final_counters'] = {a: {k: st.mean(x['counters'].get(k, float('nan')) for x in v) for k in ['up', 'lead', 'wl']}
                       for a, v in L.items() if v[0]['counters']}
# the single naive-consistency run (constant weight), same FCT recipe otherwise
p = os.path.join(R, 'logs_S3F.txt')
cos = [float(c) for c in re.findall(r'cos: ([+-][\d.]+)', open(p, errors='replace').read())]
rat = [float(c) for c in re.findall(r'\|g_cons\|/\|g_sup\|: ([\d.]+)', open(p, errors='replace').read())]
F['grad_cos']['S3F (1 run)'] = st.mean(cos); F['grad_ratio']['S3F (1 run)'] = st.mean(rat)
# per-run gradient cosine for the paired comparison of weight rules
F['grad_cos_per_seed'] = {a: [x['cos'] for x in v] for a, v in L.items() if v[0]['cos'] is not None}

# selection-free: training-time evaluation at the last epoch and over the last 10 epochs
SF = {}
for host, v, f in [('Polyp-PVT', 'PVT_VANILLA', 'S3LMUP'), ('PraNet-V2', 'PRANETV2_VANILLA', 'PRANETV2_S3LMUP'),
                   ('ASGNet', 'ASGNET_VANILLA', 'ASGNET_S3LMUP')]:
    if v not in L or f not in L:
        continue
    SF[host] = {}
    for label, sel in [('last epoch', lambda ev: ev[-1]), ('mean of last 10 epochs',
                       lambda ev: {k: st.mean(e[k] for e in ev[-10:]) for k in SETS})]:
        a = [sel(x['evals']) for x in L[v]]; b = [sel(x['evals']) for x in L[f]]
        SF[host][label] = {k: paired([100 * r[k] for r in a], [100 * r[k] for r in b]) for k in SETS}
    SF[host]['best epoch (vanilla, FCT)'] = ([max(range(len(x['evals'])), key=lambda i: x['evals'][i]['mean']) + 1 for x in L[v]],
                                             [max(range(len(x['evals'])), key=lambda i: x['evals'][i]['mean']) + 1 for x in L[f]])
    n_ep = min(len(x['evals']) for x in L[v] + L[f])
    SF[host]['epochs'] = n_ep
    mv = [st.mean(x['evals'][e]['mean'] for x in L[v]) for e in range(n_ep)]   # six-seed mean curves
    mf = [st.mean(x['evals'][e]['mean'] for x in L[f]) for e in range(n_ep)]
    SF[host]['fct_above'] = sum(b > a for a, b in zip(mv, mf))
F['selection_free'] = SF

json.dump(F, open(os.path.join(OUT, 'facts.json'), 'w'), indent=1, default=str)
md = ['# Facts extracted from %s' % R, '']
for k, v in F['summaries'].items():
    md.append('## %s' % k)
    for s_, d in v.items():
        md.append('- %-8s vanilla %.4f±%.4f  arm %.4f±%.4f  diff %+.4f [%+.4f, %+.4f]  p_t %.3f  p_perm %.3f  %d/6' % (
            s_, d['van'], d['van_sd'], d['fct'], d['fct_sd'], d['diff'], d['lo'], d['hi'], d.get('p_t', float('nan')), d.get('p_perm', float('nan')), d['wins']))
md.append('\n## head to head (points of Dice x100 not applied; fractions)')
for k, v in H.items():
    md.append('- %s: mean %+.4f [%+.4f, %+.4f] %d/6 | ETIS %+.4f [%+.4f, %+.4f] %d/6' % (
        k, v['mean']['diff'], v['mean']['lo'], v['mean']['hi'], v['mean']['wins'], v['ETIS']['diff'], v['ETIS']['lo'], v['ETIS']['hi'], v['ETIS']['wins']))
md.append('\n## flip gap (%, mean over seeds)')
for a, g in G.items():
    md.append('- %-18s n=%d  ' % (a, g['n']) + '  '.join('%s %.1f' % (c, g[c]) for c in head[1:]))
md.append('- every seed lower: %s' % json.dumps(F['flip_gap_every_seed_lower']))
md.append('\n## time (min/epoch, median over seeds): %s' % json.dumps({k: round(v, 2) for k, v in F['time_min_per_epoch'].items()}))
md.append('## gradient cosine (mean over logged steps and seeds): %s' % json.dumps({k: round(v, 3) for k, v in F['grad_cos'].items()}))
md.append('## |g_cons|/|g_sup|: %s' % json.dumps({k: round(v, 3) for k, v in F['grad_ratio'].items()}))
md.append('## final counters (EMA, mean over seeds): %s' % json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in F['final_counters'].items()}))
md.append('\n## selection-free (training-time evaluator, Dice points)')
for host, v in SF.items():
    for label in ['last epoch', 'mean of last 10 epochs']:
        md.append('- %s, %s: ' % (host, label) + '  '.join('%s %+.2f [%+.2f, %+.2f] %d/6' % (k, d['diff'], d['lo'], d['hi'], d['wins']) for k, d in v[label].items()))
    md.append('- %s best epochs vanilla %s FCT %s (of %d)' % (host, v['best epoch (vanilla, FCT)'][0], v['best epoch (vanilla, FCT)'][1], v['epochs']))
open(os.path.join(OUT, 'facts.md'), 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
