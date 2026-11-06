"""Qualitative figure: unseen-set test images, their labels and the seed-1 maps of both networks,
without and with FCT. Rows are chosen from the per-image scores of all six seeds (per_image_compare.py
CSVs): the images with the largest mean gain over both networks, one image per polyp, plus the image
with the largest mean loss. The Dice under each map is that map's own (seed 1, MATLAB-style mDice).
Usage: python make_fig_qual.py REPO PER_IMAGE_DIR OUT.pdf"""
import csv, collections, os, statistics as st, sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO, PI, OUT = sys.argv[1:4]
FOLDER = {'ETIS': 'ETIS-LaribPolypDB', 'ColonDB': 'CVC-ColonDB', 'CVC-300': 'CVC-300'}
# (set, image): picked by the ranking printed below; 22/20 and 244/247/250 are frames of one polyp each
ROWS = [('ETIS', '22'), ('ETIS', '57'), ('ColonDB', '106'), ('ColonDB', '131'), ('ColonDB', '75')]
COLS = [('Polyp-PVT', 'PVT_VANILLA_s1', 'S3LMUP_s1', 'pvt.csv'),
        ('PraNet-V2', 'PRANETV2_VANILLA_s1', 'PRANETV2_S3LMUP_s1', 'pranet.csv')]


def scores(name):
    d = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(PI, name))):
        d[(r['set'], r['image'])][r['run']] = float(r['meandic'])
    return d


S = {c[3]: scores(c[3]) for c in COLS}
for key in ROWS:   # print the six-seed evidence behind each row
    msg = []
    for net, van, fct, f in COLS:
        a, b = van[:-1], fct[:-1]
        g = [S[f][key][b + str(s)] - S[f][key][a + str(s)] for s in range(1, 7)]
        msg.append('%s six-seed mean gain %+.3f, seed-1 %+.3f' % (net, st.mean(g), g[0]))
    print(key, ' | '.join(msg))

INK = '#0b0b0b'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                     'axes.unicode_minus': False, 'font.size': 7.5})
W, H = 240, 190
fig, axes = plt.subplots(len(ROWS), 6, figsize=(7.0, 0.93 * len(ROWS) + 0.35))
heads = ['Image', 'Label', 'Polyp-PVT', 'Polyp-PVT + FCT', 'PraNet-V2', 'PraNet-V2 + FCT']
for r, (s, n) in enumerate(ROWS):
    root = os.path.join(REPO, 'dataset', 'TestDataset', FOLDER[s])
    tiles = [Image.open(os.path.join(root, 'images', n + '.png')).convert('RGB'),
             Image.open(os.path.join(root, 'masks', n + '.png')).convert('L')]
    notes = ['%s %s' % (s, n), '']
    for net, van, fct, f in COLS:
        for run in (van, fct):
            tiles.append(Image.open(os.path.join(REPO, 'result_map', run, FOLDER[s], n + '.png')).convert('L'))
            notes.append('Dice %.2f' % S[f][(s, n)][run])
    for c, (t, note) in enumerate(zip(tiles, notes)):
        ax = axes[r, c]
        a = np.asarray(t.resize((W, H), Image.BILINEAR))
        ax.imshow(a, cmap='gray' if a.ndim == 2 else None, vmin=0, vmax=255, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3); sp.set_color('#c3c2b7')
        if note:
            ax.text(0.5, -0.04, note, transform=ax.transAxes, ha='center', va='top', fontsize=6.6, color=INK)
        if r == 0:
            ax.set_title(heads[c], fontsize=7.5, color=INK, pad=3)
fig.subplots_adjust(left=0.005, right=0.995, top=0.955, bottom=0.03, wspace=0.03, hspace=0.24)
fig.savefig(OUT, dpi=300)
print('wrote', OUT)
