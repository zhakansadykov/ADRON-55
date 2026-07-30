"""
diag_slopes.py - locating the residual azimuthal systematic.

Histograms slope_x and slope_y (mm per mm) SEPARATELY for single 3D tracks. For
an isotropic flux both must be symmetric about zero. If only one is displaced,
the systematic lives in that projection.

Also counts how many 3D hits fall in each physical plane (0..7), which shows
whether the deepest Y plane (idx 7) enters the fit at all.

Run:
    python analysis/diag_slopes.py --config config/settings.yaml --min-quality GOOD --limit 20000
"""
import os
import sys
import argparse
import contextlib
import io
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_loader, preprocessing, event_quality, tracker

QUALITY_ORDER = {'EXCELLENT': 4, 'GOOD': 3, 'POOR': 2, 'NOISE': 1, 'NOT_ANALYZED': 0}


def _decode(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def load_target_ids(catalog_path, min_quality):
    import h5py
    thr = QUALITY_ORDER.get(min_quality, 3)
    targets = set()
    with h5py.File(catalog_path, 'r') as hf:
        meta = hf['metadata']
        ids = meta['event_id'][:]
        status = meta['filter_status'][:]
        qcls = meta['quality_class'][:] if 'quality_class' in meta else None
        for i in range(len(ids)):
            if _decode(status[i]) != 'passed':
                continue
            q = _decode(qcls[i]) if qcls is not None else 'NOT_ANALYZED'
            if QUALITY_ORDER.get(q, 0) >= thr:
                targets.add(int(ids[i]))
    return targets


def ascii_hist(values, nbins, lo, hi, label, width=50):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        print(f"\n[{label}] no data")
        return
    counts, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    mx = counts.max() if counts.max() > 0 else 1
    print(f"\n[{label}]  N={len(v)}  mean={v.mean():+.4f}  median={np.median(v):+.4f}")
    for i in range(nbins):
        bar = '#' * int(round(width * counts[i] / mx))
        print(f"  {edges[i]:+6.3f}..{edges[i+1]:+6.3f} | {bar:<{width}} {counts[i]}")


def main():
    ap = argparse.ArgumentParser(description="slope_x / slope_y distributions")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--min-quality', default='GOOD', choices=list(QUALITY_ORDER.keys()))
    ap.add_argument('--limit', type=int, default=20000)
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    catalog = os.path.join(data_cfg['processed_dir'], data_cfg['catalog_file'])
    targets = load_target_ids(catalog, args.min_quality)
    print(f"To process: {len(targets)} events (limit {args.limit})")

    heights = cfg['geometry']['heights']
    slope_x_single, slope_y_single = [], []
    hits_per_row = defaultdict(int)
    tracks_with_row = defaultdict(int)
    n_single = 0
    done = 0

    for ev in data_loader.iter_all_events(data_cfg['raw_dir'], years=data_cfg.get('years')):
        if ev['global_id'] not in targets:
            continue
        done += 1
        with contextlib.redirect_stdout(io.StringIO()):
            X_layers, _, Y_layers, _ = preprocessing.unify_layers(
                ev['arrays'], cfg['geometry'].get('expected_channels', {}))
            stats = data_loader.compute_event_stats(ev, hit_threshold=cfg['filters']['hit_threshold'])
            report = event_quality.analyze_event(
                ev['global_id'], ev['event_time'], X_layers, Y_layers, stats, cfg)
            res = tracker.reconstruct_3d(report, cfg)

        t3d = res['tracks_3d']
        if len(t3d) == 1:
            n_single += 1
            t = t3d[0]
            slope_x_single.append(t.slope_x)
            slope_y_single.append(t.slope_y)
            rows_in_track = set()
            for (x_mm, y_mm, z_mm) in t.hits_3d:
                li = int(np.argmin([abs(z_mm - h) for h in heights]))
                hits_per_row[li] += 1
                rows_in_track.add(li)
            for li in rows_in_track:
                tracks_with_row[li] += 1
        if done >= args.limit:
            break

    print(f"\nProcessed: {done} | single 3D tracks: {n_single}")
    ascii_hist(slope_x_single, 20, -0.6, 0.6, "SLOPE_X, single tracks (expect mean ~0)")
    ascii_hist(slope_y_single, 20, -0.6, 0.6, "SLOPE_Y, single tracks (expect mean ~0)")

    print("\nPlane participation in single tracks (idx 0 = top .. 7 = bottom):")
    print(f"{'row_idx':>7} {'proj':>5} {'n_hits':>8} {'n_tracks_with_hit':>18} {'%tracks':>8}")
    proj = {0: 'X', 2: 'X', 4: 'X', 6: 'X', 1: 'Y', 3: 'Y', 5: 'Y', 7: 'Y'}
    for li in range(8):
        pct = tracks_with_row[li] / n_single if n_single else 0
        print(f"{li:>7} {proj[li]:>5} {hits_per_row[li]:>8} {tracks_with_row[li]:>18} {pct:>7.1%}")

    print("\nInterpretation:")
    print(" - mean(slope_y) != 0 with mean(slope_x) ~ 0 -> a Y-specific systematic.")
    print(" - idx7 (deepest Y) with ~0% participation -> it is not in the fit, so not the cause.")


if __name__ == '__main__':
    main()