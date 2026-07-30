"""
diag_residuals.py - track-based alignment from the mean hit position.

Measures the mean MEASURED hit position (mm) in each plane over a sample of
SINGLE 3D tracks, with the current row_alignment_mm already applied. For an
isotropic flux it must be zero; a non-zero value means the plane is not aligned,
and that residual is added to its offset.

The update is ADDITIVE and ITERATIVE: new_offset = current_offset + mean_position.
Repeat until mean(slope_x) and mean(slope_y) are ~0, at which point the azimuth
is flat.

Which coordinate of a hit is the measured one follows from its weights
(for an x hit, wx > wy).

Run:
    python analysis/diag_residuals.py --config config/settings.yaml --min-quality GOOD --limit 20000
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
ROW_PROJ = {0: 'X', 2: 'X', 4: 'X', 6: 'X', 1: 'Y', 3: 'Y', 5: 'Y', 7: 'Y'}


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


def main():
    ap = argparse.ArgumentParser(description="Track-based alignment from hit positions")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--min-quality', default='GOOD', choices=list(QUALITY_ORDER.keys()))
    ap.add_argument('--limit', type=int, default=40000)
    ap.add_argument('--min-rows', type=int, default=6,
                    help="minimum planes per track (6 = 3+3, continuous slope)")
    ap.add_argument('--min-hits-row', type=int, default=50,
                    help="minimum hits in a plane before its offset is updated")
    ap.add_argument('--season', default=None,
                    help="restrict to one observing season, e.g. 2021. Filtering is on the "
                         "event timestamp, not on the directory, so it is unaffected by "
                         "misfiled runs. Note that restricting data.years in the "
                         "configuration instead would renumber global_id and silently "
                         "break the match against the catalogue.")
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    catalog = os.path.join(data_cfg['processed_dir'], data_cfg['catalog_file'])
    current_align = cfg['geometry'].get('row_alignment_mm', {}) or {}
    targets = load_target_ids(catalog, args.min_quality)
    print(f"To process: {len(targets)} events (limit {args.limit})"
          + (f", season {args.season} only" if args.season else ""))

    real_pos = defaultdict(list)   # layer_idx -> measured positions, in mm
    slope_x_all, slope_y_all = [], []
    n_single, done = 0, 0

    for ev in data_loader.iter_all_events(data_cfg['raw_dir'], years=data_cfg.get('years')):
        if ev['global_id'] not in targets:
            continue
        if args.season and not ev['event_time'].startswith(args.season):
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
            t = t3d[0]
            n_rows = len(set(w[2] for w in t.hits_weights))
            if n_rows < args.min_rows:   # clean multi-point tracks only, free of quantization
                continue
            n_single += 1
            slope_x_all.append(t.slope_x)
            slope_y_all.append(t.slope_y)
            for (x, y, z), (wx, wy, li) in zip(t.hits_3d, t.hits_weights):
                if wx > wy:          # measured X
                    real_pos[li].append(x)
                else:                # measured Y
                    real_pos[li].append(y)
        if done >= args.limit:
            break

    mx = float(np.mean(slope_x_all)) if slope_x_all else float('nan')
    my = float(np.mean(slope_y_all)) if slope_y_all else float('nan')
    print(f"\nProcessed: {done} | single 3D tracks: {n_single}")
    print(f"CURRENT mean(slope_x) = {mx:+.4f} | mean(slope_y) = {my:+.4f}  (target: both ~0)\n")

    print(f"{'row':>4} {'proj':>5} {'n_hits':>8} {'mean_pos_mm':>12} "
          f"{'current_off':>12} {'-> new_off':>12}")
    print("-" * 60)
    new_offsets = {}
    for li in range(8):
        cur = float(current_align.get(li, current_align.get(str(li), 0.0)))
        vals = real_pos.get(li, [])
        if len(vals) < args.min_hits_row:
            new_offsets[li] = cur
            note = f"(too few hits, {len(vals)}) -> kept as is"
            print(f"{li:>4} {ROW_PROJ[li]:>5} {len(vals):>8} {'—':>12} {cur:>12.1f}  {note}")
            continue
        mean_pos = float(np.mean(vals))
        new_off = cur + mean_pos   # additive: shift the plane so the mean position goes to zero
        new_offsets[li] = new_off
        print(f"{li:>4} {ROW_PROJ[li]:>5} {len(vals):>8} {mean_pos:>12.1f} "
              f"{cur:>12.1f} {new_off:>12.1f}")

    print("\nUpdated row_alignment_mm (additive) - replace in settings.yaml:")
    print("row_alignment_mm:")
    for li in range(8):
        print(f"  {li}: {new_offsets[li]:.1f}")
    print("\nRecompute reco and plots. If mean(slope) is not yet ~0, run diag_residuals again")
    print("(it converges in one to three passes).")


if __name__ == '__main__':
    main()