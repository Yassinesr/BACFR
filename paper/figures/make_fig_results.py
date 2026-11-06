"""Result figures, all read from the Polyp-PVT repo's logs and the extracted facts:
  fig_seeds.pdf   per-seed paired gains (FCT - vanilla, mDice points) per test set, both networks
  fig_cos.pdf     cosine between the flip-pull gradient and the rest of the gradient, per epoch,
                  for the three label-aware weight rules on Polyp-PVT (mean +- SD over six seeds)
  fig_curves.pdf  five-set mean Dice of the training-time evaluator after every epoch, vanilla and
                  FCT, both networks (mean +- SD over six seeds): the view without epoch selection
Usage: python make_fig_results.py REPO FACTS_JSON OUTDIR"""
import json, math, os, re, statistics as st, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO, FACTS, OUTD = sys.argv[1:4]
F = json.load(open(FACTS))
BLUE, ORANGE, GREY, LBLUE = '#2a78d6', '#eb6834', '#7a7974', '#86b6ef'
INK, INK2, GRID, AXIS = '#0b0b0b', '#4a4a46', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False, 'font.size': 7.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.6,
                     'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.size': 2, 'ytick.major.size': 2,
                     'legend.frameon': False, 'legend.fontsize': 6.8})
T975 = 2.570582


def style(ax, grid='x'):
    ax.grid(True, axis=grid, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ per-seed paired gains
SETS = ['Kvasir', 'ClinicDB', 'ColonDB', 'CVC-300', 'ETIS', 'mean']
LABEL = {'Kvasir': 'Kvasir', 'ClinicDB': 'CVC-ClinicDB', 'ColonDB': 'CVC-ColonDB', 'CVC-300': 'CVC-300',
         'ETIS': 'ETIS', 'mean': 'Five-set mean'}
HOSTS = [('Polyp-PVT', 'pvt_s3lmup', 'PVT_VANILLA', 'S3LMUP', BLUE),
         ('PraNet-V2', 'pranet_s3lmup', 'PRANETV2_VANILLA', 'PRANETV2_S3LMUP', ORANGE)]
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.35), sharey=True)
ypos = {s: -i - (0.5 if s == 'mean' else 0) for i, s in enumerate(SETS)}
for ax, (name, key, van, fct, color) in zip(axes, HOSTS):
    style(ax)
    runs = F['runs'][key]
    ax.axvline(0, color=INK2, lw=0.6)
    for j, s in enumerate(SETS):
        d = [100 * (runs[fct][str(k)][j] - runs[van][str(k)][j]) for k in range(1, 7)]
        m, h = st.mean(d), T975 * st.stdev(d) / math.sqrt(6)
        ref = F['summaries'][key][s]   # the summary the tables print: the same numbers
        assert abs(100 * ref['diff'] - m) < 0.011, (name, s, m, ref['diff'])
        y = ypos[s]
        jit = np.linspace(-0.17, 0.17, 6)
        ax.scatter(d, y + jit, s=7, facecolor='white', edgecolor=color, linewidth=0.6, zorder=3)
        ax.plot([m - h, m + h], [y - 0.32, y - 0.32], color=color, lw=1.1, solid_capstyle='butt', zorder=2)
        ax.scatter([m], [y - 0.32], s=14, color=color, zorder=4)
    ax.set_title(name, fontsize=7.5, color=INK, pad=3)
    ax.set_xlabel('Gain (mDice points)')
axes[0].set_yticks([ypos[s] for s in SETS])
axes[0].set_yticklabels([LABEL[s] for s in SETS])
axes[0].tick_params(axis='y', length=0)
axes[1].tick_params(axis='y', length=0)
fig.tight_layout(w_pad=0.6)
fig.savefig(os.path.join(OUTD, 'fig_seeds.pdf'))
print('wrote fig_seeds.pdf')

# ------------------------------------------------------------------ logs: per-epoch cos and evaluation
LINE = re.compile(r'^\S+ \S+ Epoch \[(\d+)/(\d+)\], Step \[(\d+)/(\d+)\](.*)$')
SETRE = re.compile(r'^(CVC-300|CVC-ClinicDB|Kvasir|CVC-ColonDB|ETIS-LaribPolypDB) :\s+([\d.]+)')


def parse(arm, seed):
    cos, evals, cur = {}, [], {}
    for line in open(os.path.join(REPO, 'logs_%s_s%d.txt' % (arm, seed)), errors='replace'):
        line = line.strip()
        m = LINE.match(line)
        if m:
            c = re.search(r'cos: ([+-][\d.]+)', m.group(5))
            if c:
                cos.setdefault(int(m.group(1)), []).append(float(c.group(1)))
            continue
        if SETRE.match(line):
            continue
        m = re.match(r'^mean:\s+([\d.]+)', line)
        if m:
            evals.append(float(m.group(1)))
    return {e: st.mean(v) for e, v in cos.items()}, evals


# gradient alignment of the three label-aware rules (Polyp-PVT)
RULES = [('Confidence-graded weight', 'S3SIG', GREY, '-'), ('Lag-or-margin, no upright pull', 'S3LM', LBLUE, '-'),
         ('FCT', 'S3LMUP', BLUE, '-')]
fig, ax = plt.subplots(figsize=(3.5, 1.9))
style(ax, 'y')
summary = {}
for label, arm, color, ls in RULES:
    per = [parse(arm, s)[0] for s in range(1, 7)]
    ep = sorted(set.intersection(*[set(p) for p in per]))
    M = np.array([[p[e] for e in ep] for p in per])
    summary[arm] = float(M.mean())
    # each seed's per-epoch means, smoothed by a centred 5-epoch moving average (said in the caption)
    k = np.ones(5) / 5
    M = np.array([np.convolve(np.pad(r, 2, mode='edge'), k, mode='valid') for r in M])
    mu, sd = M.mean(0), M.std(0, ddof=1)
    ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)
    ax.plot(ep, mu, color=color, lw=1.1, ls=ls, label=label)
ax.set_xlabel('Epoch')
ax.set_ylabel('Gradient cosine')
ax.set_xlim(1, 99)
ax.set_ylim(0, 0.8)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.4, columnspacing=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUTD, 'fig_cos.pdf'))
print('wrote fig_cos.pdf; mean over epochs and seeds of the per-epoch means:', summary)

# per-epoch five-set mean Dice, vanilla and FCT, both networks
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.85))
for ax, (name, key, van, fct, color) in zip(axes, HOSTS):
    style(ax, 'y')
    for arm, ls, lab in [(van, (0, (3, 1.5)), 'vanilla'), (fct, '-', 'FCT')]:
        ev = [parse(arm, s)[1] for s in range(1, 7)]
        n = min(len(e) for e in ev)
        M = 100 * np.array([e[:n] for e in ev])
        x = np.arange(1, n + 1)
        mu, sd = M.mean(0), M.std(0, ddof=1)
        ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15 if arm == fct else 0.08, linewidth=0)
        ax.plot(x, mu, color=color, lw=1.0, ls=ls, label=lab)
    ax.set_title(name, fontsize=7.5, color=INK, pad=3)
    ax.set_xlabel('Epoch')
    ax.set_xlim(1, n)
    ax.legend(loc='lower right', handlelength=1.8)
axes[0].set_ylabel('Five-set mean Dice (%)')
axes[0].set_ylim(76, 88.5)
axes[1].set_ylim(66, 84)
fig.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(OUTD, 'fig_curves.pdf'))
print('wrote fig_curves.pdf')
