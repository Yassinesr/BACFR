"""Figure (Section III): what a consistency pull does to the student's error, per pixel.
Axes: student error e_s and teacher error e_t, e = |p - y|, for a binary label y.
(a) unweighted consistency: pull toward the label e_s - e_t (Proposition 1);
(b) the FCT weight w (Eq. lag-or-margin, a = 2);
(c) the FCT pull w (e_s - e_t).
Pure analysis: no data. Usage: python make_fig_pull.py OUT.pdf"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = sys.argv[1]
INK, INK2, AXIS = '#0b0b0b', '#4a4a46', '#c3c2b7'
# Diverging: blue (toward the label) <-> red (away from it), neutral grey midpoint, equal arms.
DIV = LinearSegmentedColormap.from_list('div', ['#9e2626', '#e34948', '#f0a3a2', '#f0efec',
                                                '#86b6ef', '#2a78d6', '#104281'])
SEQ = LinearSegmentedColormap.from_list('seq', ['#f0efec', '#c3c2b7', '#7a7974', '#2b2b29'])

n = 801
e = np.linspace(0, 1, n)
ES, ET = np.meshgrid(e, e)                     # x: student error, y: teacher error
toward = ES - ET                               # unweighted pull toward the label
w = np.where(ES >= ET, 1.0, np.clip(1.0 - 2.0 * ET, 0.0, None))
fct = w * toward

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False, 'font.size': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.edgecolor': AXIS,
                     'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2})
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35))
panels = [(toward, DIV, (-1, 1), '(a) Unweighted pull toward the label'),
          (w, SEQ, (0, 1), '(b) FCT weight $w$'),
          (fct, DIV, (-1, 1), '(c) FCT pull toward the label')]
for ax, (z, cmap, (lo, hi), title) in zip(axes, panels):
    im = ax.imshow(z, origin='lower', extent=(0, 1, 0, 1), cmap=cmap, vmin=lo, vmax=hi,
                   interpolation='bilinear', rasterized=True)
    ax.plot([0, 1], [0, 1], color=INK, lw=0.6)
    ax.axhline(0.5, color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.axvline(0.5, color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('student error $e_s$')
    ax.set_title(title, fontsize=8, color=INK, pad=4)
    ax.set_aspect('equal')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[lo, 0, hi] if lo < 0 else [0, 0.5, 1])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5, length=2, color=INK2)
axes[0].set_ylabel('teacher error $e_t$')
kw = dict(fontsize=6.6, color=INK, ha='center', va='center')
axes[0].text(0.70, 0.22, 'toward\nthe label', **kw)
axes[0].text(0.27, 0.80, 'away from\nthe label', **kw)
axes[1].text(0.70, 0.22, '$w=1$\n(lag)', **dict(kw, color='#f0efec'))
axes[1].text(0.25, 0.80, '$w=0$\n(teacher wrong)', **kw)
axes[1].annotate('$w=1-2e_t$\n(margin)', xy=(0.13, 0.32), xytext=(0.30, 0.60), fontsize=6.6, color=INK,
                 ha='center', va='center', arrowprops=dict(arrowstyle='-', lw=0.5, color=INK))
axes[2].text(0.70, 0.22, 'unchanged', **kw)
axes[2].text(0.27, 0.80, 'removed', **kw)
axes[2].annotate('$|\\mathrm{pull}|\\leq 1/8$', xy=(0.12, 0.30), xytext=(0.31, 0.60), fontsize=6.6, color=INK,
                 ha='center', va='center', arrowprops=dict(arrowstyle='-', lw=0.5, color=INK))
fig.tight_layout(w_pad=0.8)
fig.savefig(OUT, dpi=300)
print('wrote', OUT)
