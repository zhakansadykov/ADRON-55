"""
diag_track_quality.py - track metric distributions, to find spurious fragments.

Multi-track events proliferated once the road-search seeds were extended to the
deepest planes. The hypothesis: broad showers leave several disconnected clusters
in the lower planes, each of which seeds a track, producing short two-point
fragments that the deduplication does not merge because they do not overlap.

For GOOD events the script reports:
  - the number of 3D tracks per event;
  - the penetration depth (number of planes) of the 2D tracks;
  - the energy of the 2D tracks (summed cluster amplitude), separately for
    depth 2 and depth >= 3;
  - a comparison of the proliferating events (>=5 3D tracks) with normal ones
    (a single 3D track).

The distributions guide the choice of a rejection threshold, for instance
depth >= 3 or an energy cut.

Run:
    python analysis/diag_track_quality.py --config config/settings.yaml --min-quality GOOD --limit 20000
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


def track_energy(track):
    return sum(h.total_amplitude for h in track.hits)


def ascii_hist(values, nbins, lo, hi, label, width=46):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        print(f"\n[{label}] no data")
        return
    counts, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    mx = counts.max() if counts.max() > 0 else 1
    print(f"\n[{label}]  N={len(v)}  median={np.median(v):.1f}  mean={v.mean():.1f}")
    for i in range(nbins):
        bar = '#' * int(round(width * counts[i] / mx))
        print(f"  {edges[i]:7.0f}..{edges[i+1]:7.0f} | {bar:<{width}} {counts[i]}")


def main():
    ap = argparse.ArgumentParser(description="Track metric distributions")
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

    n3d_per_event = []
    depth_2d = []                       # depth of every 2D track
    energy_by_depth = defaultdict(list) # depth -> energies of the 2D tracks
    # proliferating events against normal ones
    exploded_n2d, normal_n2d = [], []
    exploded_depth, normal_depth = [], []
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

        n3d = len(res['tracks_3d'])
        n3d_per_event.append(n3d)
        n2d = len(report.tracks)
        for t in report.tracks:
            d = t.penetration_depth
            depth_2d.append(d)
            energy_by_depth[d].append(track_energy(t))
        if n3d >= 5:
            exploded_n2d.append(n2d)
            exploded_depth.extend([t.penetration_depth for t in report.tracks])
        elif n3d == 1:
            normal_n2d.append(n2d)
            normal_depth.extend([t.penetration_depth for t in report.tracks])
        if done >= args.limit:
            break

    print(f"\nProcessed: {done}")
    ascii_hist(n3d_per_event, 15, 0, 15, "3D tracks per event")
    ascii_hist(depth_2d, 6, 1, 7, "Depth of the 2D tracks (planes)")

    print("\n2D track energy by depth (median / N):")
    for d in sorted(energy_by_depth):
        e = energy_by_depth[d]
        print(f"  depth={d}: N={len(e):>7}  median_energy={np.median(e):>10.0f}  "
              f"mean={np.mean(e):>10.0f}")

    print("\nEvent comparison:")
    if exploded_n2d:
        print(f"  EXPLODED (>=5 3D tracks): {len(exploded_n2d)} events, "
              f"mean 2D tracks/event = {np.mean(exploded_n2d):.1f}, "
              f"median 2D depth = {np.median(exploded_depth):.1f}")
    if normal_n2d:
        print(f"  NORMAL   (1 3D track):     {len(normal_n2d)} events, "
              f"mean 2D tracks/event = {np.mean(normal_n2d):.1f}, "
              f"median 2D depth = {np.median(normal_depth):.1f}")

    # Fraction of two-point tracks with low energy: noise candidates
    if 2 in energy_by_depth:
        e2 = np.array(energy_by_depth[2])
        deep_energies = np.concatenate([energy_by_depth[d] for d in energy_by_depth if d >= 3]) \
            if any(d >= 3 for d in energy_by_depth) else np.array([])
        thr_e = np.median(deep_energies) if len(deep_energies) else 0
        low = (e2 < thr_e).mean() if len(e2) else 0
        print(f"\nTwo-point tracks below the median energy of the deep ones (threshold {thr_e:.0f}): "
              f"{low:.1%} - candidates for spurious fragments.")

    print("\nInterpretation:")
    print(" - Many 2D tracks with a median depth of 2 in exploded events -> spurious fragments.")
    print(" - If depth=2 tracks carry much less energy than depth>=3 -> cut on energy.")


if __name__ == '__main__':
    main()