"""
transverse_blocks.py - transverse profile on the full good-run sample, in blocks.

Accumulates exactly the histograms of fig_transverse.py, but one raw file at a
time and saving a partial result per block of files. This makes the full-sample
pass resumable, and it also yields a block-level estimate of the uncertainty on
the profile widths: each block is an independent group of runs, so the spread of
the per-block widths gives the standard error of the combined value, capturing
run-to-run variation as well as counting statistics.

Run (blocks may be run in any order, and re-running a block overwrites it):
    python analysis/transverse_blocks.py --config config/settings.yaml --block 0 --nblocks 8
    ...
    python analysis/transverse_blocks.py --config config/settings.yaml --merge
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
from analysis.fig_transverse import (layer_profile, width_metrics,
                                     X_IDX, Y_IDX, GAMMA_IDX)

RMAX, NBINS, PEDESTAL, WINDOW_CH, MIN_ROW_E = 1440.0, 12, 30.0, 5, 100.0


def gather_files(raw_dir, years):
    files = []
    for y in years:
        files.extend(sorted(glob.glob(os.path.join(raw_dir, str(y), "*.dat"))))
    return files


def process(files, cfg, exclude_set):
    expected = cfg['geometry'].get('expected_channels', {})
    cw = cfg['geometry']['channel_width']
    ped_trig = cfg['filters'].get('hit_threshold', 5)
    trigger = cfg['filters'].get('min_total_energy', 500)
    r_edges = np.linspace(0, RMAX, NBINS + 1)

    acc = {k: np.zeros(NBINS) for k in ('x', 'y', 'gamma', 'hadron')}
    acc.update({f'row{i}': np.zeros(NBINS) for i in range(8)})
    n = seen = excluded = below = 0

    for path in files:
        if os.path.basename(path) in exclude_set:
            excluded += sum(1 for _ in data_loader.parse_dat_file(path))
            continue
        for ev in data_loader.parse_dat_file(path):
            seen += 1
            with contextlib.redirect_stdout(io.StringIO()):
                X_layers, _, Y_layers, _ = preprocessing.unify_layers(ev['arrays'], expected)
            tot = 0.0
            for layer in list(X_layers) + list(Y_layers):
                a = np.asarray(layer, dtype=np.float64)
                tot += float(a[a >= ped_trig].sum())
            if tot < trigger:
                below += 1
                continue
            n += 1
            for layers, idxs, key in ((X_layers, X_IDX, 'x'), (Y_layers, Y_IDX, 'y')):
                for layer, idx in zip(layers, idxs):
                    prof = layer_profile(layer, cw, r_edges, PEDESTAL, WINDOW_CH, MIN_ROW_E)
                    if prof is None:
                        continue
                    acc[key] += prof
                    acc[f'row{idx}'] += prof
                    acc['gamma' if idx in GAMMA_IDX else 'hadron'] += prof
    return acc, n, seen, excluded, below


def main():
    ap = argparse.ArgumentParser(description="Transverse profile in resumable blocks")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--block', type=int, default=None)
    ap.add_argument('--nblocks', type=int, default=8)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--out', default='output/blocks')
    args = ap.parse_args()

    import yaml
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    exclude_set = set(data_cfg.get('exclude_files', []) or [])
    os.makedirs(args.out, exist_ok=True)
    r_centers = 0.5 * (np.linspace(0, RMAX, NBINS + 1)[:-1] + np.linspace(0, RMAX, NBINS + 1)[1:])

    if args.merge:
        blocks = sorted(glob.glob(os.path.join(args.out, 'block_*.npz')))
        if not blocks:
            raise SystemExit("No blocks found; run the per-block passes first.")
        data = [np.load(b) for b in blocks]
        tot = {k: sum(d[k] for d in data) for k in data[0].files if k != 'counts'}
        counts = sum(d['counts'] for d in data)
        print(f"blocks: {len(blocks)}  events used: {int(counts[0])}  "
              f"seen: {int(counts[1])}  excluded(bad run): {int(counts[2])}  "
              f"below trigger: {int(counts[3])}")
        print(f"\n{'group':<10}{'RMS (m)':>18}{'R90 (m)':>18}")
        for key, label in (('x', 'X planes'), ('y', 'Y planes'),
                           ('gamma', 'gamma block'), ('hadron', 'hadron block')):
            w = [width_metrics(r_centers, d[key]) for d in data]
            rms_b = np.array([a for a, _ in w]); r90_b = np.array([b for _, b in w])
            rms, r90 = width_metrics(r_centers, tot[key])
            # standard error of the mean over independent blocks
            e_rms = rms_b.std(ddof=1) / np.sqrt(len(rms_b))
            e_r90 = r90_b.std(ddof=1) / np.sqrt(len(r90_b))
            print(f"{label:<10}{rms/1000:>12.3f} ± {e_rms/1000:.3f}"
                  f"{r90/1000:>12.3f} ± {e_r90/1000:.3f}")
        np.savez(os.path.join(args.out, 'merged.npz'), counts=counts, **tot)
        print(f"\nMerged accumulators written to {os.path.join(args.out, 'merged.npz')}")
        return

    files = gather_files(data_cfg['raw_dir'], data_cfg.get('years'))
    chunk = files[args.block::args.nblocks]
    acc, n, seen, excluded, below = process(chunk, cfg, exclude_set)
    path = os.path.join(args.out, f'block_{args.block:02d}.npz')
    np.savez(path, counts=np.array([n, seen, excluded, below]), **acc)
    print(f"block {args.block}/{args.nblocks}: {len(chunk)} files, "
          f"{n} events used, {excluded} excluded -> {path}")


if __name__ == '__main__':
    main()
