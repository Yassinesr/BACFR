"""Figure 1: flip gap and mDice per test set, vanilla -> FCT, for each network (dumbbell rows).
Data: flip_gap.tsv and the six-seed summaries in the Polyp-PVT repo (means over seeds 1-6).
Usage: python make_fig1.py /path/to/polyp-pvt OUT.pdf"""
import csv, os, re, statistics as st, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO, OUT = sys.argv[1], sys.argv[2]
SETS = [  # summary name, flip_gap.tsv column, label, seen
    ('Kvasir', 'Kvasir', 'Kvasir', True), ('ClinicDB', 'CVC-ClinicDB', 'CVC-ClinicDB', True),
    ('ColonDB', 'CVC-ColonDB', 'CVC-ColonDB', False), ('CVC-300', 'CVC-300', 'CVC-300', False),
    ('ETIS', 'ETIS-LaribPolypDB', 'ETIS', False)]
HOSTS = [('Polyp-PVT', 'seeds_s3lmup.out', 'PVT_VANILLA', 'S3LMUP', '#2a78d6'),        # categorical slot 1
         ('PraNet-V2', 'seeds_pranetv2.out', 'PRANETV2_VANILLA', 'PRANETV2_S3LMUP', '#eb6834')]  # slot 2
INK, INK2, GRID, AXIS = '#0b0b0b', '#4a4a46', '#e1e0d9', '#c3c2b7'


def dice(path, van, fct):
    text = open(os.path.join(REPO, path)).read()
    i = text.rfind('difference = %s - %s with the same seed' % (fct, van))
    out = {}
    for line in text[i:].splitlines()[1:10]:
        m = re.match(r'^(\S+)\s+([\d.]+)\+-[\d.]+\s+([\d.]+)\+-', line.strip())
        if m:
            out[m.group(1)] = (100 * float(m.group(2)), 100 * float(m.group(3)))
    return out


def gaps(arm):
    rows = list(csv.reader(open(os.path.join(REPO, 'flip_gap.tsv')), delimiter='\t'))
    head = rows[0]
    keep = [r for r in rows[1:] if re.fullmatch(r'%s_s[1-6]/PolypPVT' % arm, r[0])]
    assert len(keep) == 6, (arm, len(keep))
    return {c: 100 * st.mean(float(r[head.index(c)]) for r in keep) for c in head[1:]}


plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False, 'font.size': 8.5,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 8, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.6,
                     'xtick.color': INK2, 'ytick.color': INK, 'xtick.major.size': 0, 'ytick.major.size': 0})
fig, (axg, axd) = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
ROWGAP, GROUPGAP, OFFSET = 1.0, 0.55, 0.19
ypos, y = {}, 0.0
for i, (_, _, label, seen) in enumerate(SETS):
    if i > 0 and seen != SETS[i - 1][3]:
        y -= GROUPGAP
    ypos[label] = y
    y -= ROWGAP
data = {}
for name, path, van, fct, color in HOSTS:
    d, gv, gf = dice(path, van, fct), gaps(van), gaps(fct)
    data[name] = {label: (gv[col], gf[col], d[key][0], d[key][1]) for key, col, label, _ in SETS}
for ax, which, xlabel, xlim in [(axg, 0, 'Flip gap (%), lower is better', (0, 28)), (axd, 2, 'mDice (%), higher is better', (64, 96))]:
    ax.grid(True, axis='x', color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    for k, (name, _, _, _, color) in enumerate(HOSTS):
        for _, _, label, _ in SETS:
            yy = ypos[label] + (OFFSET if k == 0 else -OFFSET)
            x0, x1 = data[name][label][which], data[name][label][which + 1]
            if abs(x1 - x0) > 0.045 * (xlim[1] - xlim[0]):   # shorter than its head: markers only
                ax.annotate('', xy=(x1, yy), xytext=(x0, yy),
                            arrowprops=dict(arrowstyle='-|>', color=color, lw=1.15, shrinkA=3.6, shrinkB=3.6, mutation_scale=7))
            ax.plot(x0, yy, 'o', ms=5.6, mfc='white', mec=color, mew=1.25, zorder=3)
            ax.plot(x1, yy, 'o', ms=5.6, mfc=color, mec='white', mew=0.8, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, color=INK)
axg.set_yticks([ypos[l] for _, _, l, _ in SETS], [l for _, _, l, _ in SETS])
axg.tick_params(axis='y', pad=2)
ymin, ymax = min(ypos.values()) - 0.6, max(ypos.values()) + 0.6
axg.set_ylim(ymin, ymax)
first_unseen = ypos['CVC-ColonDB']
for ax in (axg, axd):
    ax.axhline(first_unseen + ROWGAP / 2 + GROUPGAP / 2, color=AXIS, linewidth=0.6)
axg.text(-0.36, ypos['Kvasir'] + 0.45, 'Seen', transform=axg.get_yaxis_transform(), fontsize=7.6, style='italic', color=INK2)
axg.text(-0.36, first_unseen + 0.45, 'Unseen', transform=axg.get_yaxis_transform(), fontsize=7.6, style='italic', color=INK2)
handles = [Line2D([], [], color=c, lw=1.15, marker='o', ms=5.6, mfc=c, mec='white', label=n) for n, _, _, _, c in HOSTS]
handles += [Line2D([], [], ls='', marker='o', ms=5.6, mfc='white', mec=INK2, mew=1.25, label='Vanilla'),
            Line2D([], [], ls='', marker='o', ms=5.6, mfc=INK2, mec='white', label='FCT (ours)')]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=7.8, handletextpad=0.4,
           columnspacing=1.8, bbox_to_anchor=(0.55, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.0)
fig.savefig(OUT, bbox_inches='tight', pad_inches=0.02)
for name in data:
    for label, (g0, g1, d0, d1) in data[name].items():
        print('%-10s %-13s gap %5.1f -> %5.1f   mDice %5.2f -> %5.2f' % (name, label, g0, g1, d0, d1))
print('wrote', OUT)
