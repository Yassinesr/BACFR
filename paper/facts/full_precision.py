"""Full-precision mDice of every seed run, from the saved maps, with the MATLAB-style evaluation
(eval_dice.py through per_image_compare.load_set). Writes full_precision.json:
{run: {set: [mDice, n_images]}}. Usage: python full_precision.py REPO OUT.json"""
import argparse, json, os, sys
REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
os.chdir(REPO)
import per_image_compare as P

SETS = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']
ARMS = ['PVT_VANILLA', 'S3LMUP', 'S3LM', 'S3SIG', 'S3LMUP_A05', 'S3LMUP_A10', 'S3LMUP_A15',
        'PRANETV2_VANILLA', 'PRANETV2_S3LMUP']
DISSECT = ['S3', 'S3UP', 'S3FUP', 'S3RELUP', 'S3F']        # the dissection arms: whichever seeds have maps
def has_maps(run):
    return all(os.path.isdir(os.path.join('./result_map', run, s)) and
               len(os.listdir(os.path.join('./result_map', run, s))) ==
               len(os.listdir(os.path.join('./dataset/TestDataset', s, 'masks'))) for s in SETS)
runs = ['%s_s%d' % (a, s) for a in ARMS for s in range(1, 7)]
runs += [r for r in ('%s_s%d' % (a, s) for a in DISSECT for s in range(1, 7)) if has_maps(r)]
opt = argparse.Namespace(result_root='./result_map', gt_root='./dataset/TestDataset')
out = {r: {} for r in runs}
for s in SETS:
    names, area, sc, missing, resized = P.load_set(opt, s, runs)
    assert missing == 0, (s, missing)
    for i, r in enumerate(runs):
        out[r][s] = [float(sc['meandic'][i].mean()), len(names)]
    print(s, len(names), 'images', flush=True)
# PolypGen, for the runs that have its maps (vanilla and the confidence-graded weight)
pg = ['PolypGen_C%d' % c for c in range(1, 7)]
pg_runs = [r for r in ['%s_s%d' % (a, s) for a in ['PVT_VANILLA', 'S3SIG'] for s in range(1, 7)]
           if all(os.path.isdir(os.path.join('./result_map', r, c)) for c in pg)]
if pg_runs and os.path.isdir('./dataset/PolypGen'):
    opt_pg = argparse.Namespace(result_root='./result_map', gt_root='./dataset/PolypGen')
    for c in pg:
        names, area, sc, missing, resized = P.load_set(opt_pg, c, pg_runs)
        for i, r in enumerate(pg_runs):
            out.setdefault(r, {})[c] = [float(sc['meandic'][i].mean()), len(names)]
        print(c, len(names), 'images, missing', missing, flush=True)
json.dump(out, open(OUT, 'w'), indent=1)
print('wrote', OUT)
