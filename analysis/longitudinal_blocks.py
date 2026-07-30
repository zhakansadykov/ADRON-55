"""
longitudinal_blocks.py - longitudinal profile on the full good-run sample, in blocks.

Accumulates exactly the quantities of fig_longitudinal.py (per-plane occupancy,
summed deposition and sum of squares), but one raw file at a time and saving a
partial result per block of files, so that the full-sample pass is resumable.

Run:
    python analysis/longitudinal_blocks.py --config config/settings.yaml --block 0 --nblocks 16
    ...
    python analysis/longitudinal_blocks.py --config config/settings.yaml --merge
"""
import os
import io
import sys
import glob
import argparse
import contextlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data_loader, preprocessing
from analysis.fig_longitudinal import X_IDX, Y_IDX, wilson, cumulative_depth


def process(files, cfg, exclude_set):
    expected = cfg['geometry'].get('expected_channels', {})
    ped = cfg['filters'].get('hit_threshold', 5)
    trigger = cfg['filters'].get('min_total_energy', 500)

    occ = np.zeros(8); esum = np.zeros(8); esq = np.zeros(8)
    n = seen = excluded = below = 0

    for path in files:
        if os.path.basename(path) in exclude_set:
            excluded += sum(1 for _ in data_loader.parse_dat_file(path))
            continue
        for ev in data_loader.parse_dat_file(path):
            seen += 1
            with contextlib.redirect_stdout(io.StringIO()):
                X_layers, _, Y_layers, _ = preprocessing.unify_layers(ev['arrays'], expected)
            row_e = np.zeros(8); row_hit = np.zeros(8, dtype=bool)
            for layers, idxs in ((X_layers, X_IDX), (Y_layers, Y_IDX)):
                for layer, idx in zip(layers, idxs):
                    a = np.asarray(layer, dtype=np.float64)
                    m = a >= ped
                    row_e[idx] = float(a[m].sum())
                    row_hit[idx] = bool(np.any(m))
            if row_e.sum() < trigger:
                below += 1
                continue
            n += 1
            esum += row_e
            esq += row_e * row_e
            occ += row_hit.astype(float)
    return occ, esum, esq, n, seen, excluded, below


def main():
    ap = argparse.ArgumentParser(description="Longitudinal profile in resumable blocks")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--block', type=int, default=None)
    ap.add_argument('--nblocks', type=int, default=16)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--out', default='output/blocks_long')
    args = ap.parse_args()

    import yaml
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    exclude_set = set(data_cfg.get('exclude_files', []) or [])
    os.makedirs(args.out, exist_ok=True)

    if args.merge:
        blocks = sorted(glob.glob(os.path.join(args.out, 'block_*.npz')))
        if not blocks:
            raise SystemExit("No blocks found; run the per-block passes first.")
        data = [np.load(b) for b in blocks]
        occ = sum(d['occ'] for d in data); esum = sum(d['esum'] for d in data)
        esq = sum(d['esq'] for d in data); cnt = sum(d['counts'] for d in data)
        n = int(cnt[0])
        nch = np.array([cfg['geometry']['expected_channels'][f'row_{i+1}'] for i in range(8)],
                       dtype=float)
        x0, li, mm = cumulative_depth(cfg)
        proj = ['X', 'Y'] * 4
        print(f"blocks: {len(blocks)}  events used: {n}  seen: {int(cnt[1])}  "
              f"excluded(bad run): {int(cnt[2])}  below trigger: {int(cnt[3])}")
        print(f"\n{'plane':>5} {'proj':>4} {'lam_I':>7} {'occupancy':>20} "
              f"{'<E>/Nch':>16} {'<E>/Nch|fired':>16}")
        rows = []
        for i in range(8):
            p = occ[i] / n
            lo, hi = wilson(occ[i], n)
            mean = esum[i] / n
            var = max(esq[i] / n - mean ** 2, 0.0)
            sem = np.sqrt(var / n)
            epc, sepc = mean / nch[i], sem / nch[i]
            cond, scond = (epc / p, sepc / p) if p > 0 else (float('nan'),) * 2
            rows.append(dict(row=i + 1, proj=proj[i], lam=li[i], occ=p,
                             docc=max(hi - p, p - lo), e=epc, de=sepc, ec=cond, dec=scond))
            print(f"{i+1:>5} {proj[i]:>4} {li[i]:>7.2f} "
                  f"{p:>12.4f} ± {max(hi-p,p-lo):.4f} "
                  f"{epc:>10.1f} ± {sepc:<4.1f} {cond:>10.1f} ± {scond:<4.1f}")
        np.savez(os.path.join(args.out, 'merged.npz'), occ=occ, esum=esum, esq=esq, counts=cnt)
        import json
        with open(os.path.join(args.out, 'response.json'), 'w', encoding='utf-8') as f:
            json.dump({'n': n, 'rows': rows}, f, indent=1)
        print(f"\nWritten: {os.path.join(args.out, 'response.json')}")
        return

    files = []
    for y in data_cfg.get('years'):
        files.extend(sorted(glob.glob(os.path.join(data_cfg['raw_dir'], str(y), "*.dat"))))
    chunk = files[args.block::args.nblocks]
    occ, esum, esq, n, seen, excluded, below = process(chunk, cfg, exclude_set)
    path = os.path.join(args.out, f'block_{args.block:02d}.npz')
    np.savez(path, occ=occ, esum=esum, esq=esq,
             counts=np.array([n, seen, excluded, below]))
    print(f"block {args.block}/{args.nblocks}: {len(chunk)} files, {n} events used -> {path}")


if __name__ == '__main__':
    main()
