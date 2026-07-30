"""
diag_clusters_bottom.py - cluster yield per plane against the hit threshold.

The question is whether the deepest planes (7 at idx 6, 8 at idx 7) can be
brought into the tracking. diag_slopes showed that the fit uses two points
(planes 3 and 5, or 4 and 6) and that the deepest planes drop out. There are two
possible causes: (A) the seeds do not reach them, and (B) clustering at a
threshold of 30 may lose weak signal behind the iron. This script tests (B)
without touching the code: it runs find_clusters_in_layer at several thresholds
and reports, per plane,
  - the fraction of events in which the plane yields at least one cluster;
  - the mean number of clusters and the median peak cluster amplitude.

If planes 7 and 8 yield almost nothing at 30 but appear with a decent amplitude
at 15, the threshold can be lowered. If the yield stays low even at 5, the
signal is genuinely weak and the cause is absorption.

Run:
    python analysis/diag_clusters_bottom.py --config config/settings.yaml --sample 20000 --stride 14
"""
import os
import sys
import argparse
import contextlib
import io
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_loader, preprocessing
from src.event_quality import find_clusters_in_layer

ROW_PROJ = {0: 'X', 2: 'X', 4: 'X', 6: 'X', 1: 'Y', 3: 'Y', 5: 'Y', 7: 'Y'}
# X_layers run [7, 5, 3, 1] -> layer_idx [6, 4, 2, 0]; Y [8, 6, 4, 2] -> [7, 5, 3, 1]
X_IDX = [6, 4, 2, 0]
Y_IDX = [7, 5, 3, 1]


def main():
    ap = argparse.ArgumentParser(description="Cluster yield per plane against the hit threshold")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--sample', type=int, default=20000)
    ap.add_argument('--stride', type=int, default=14)
    ap.add_argument('--thresholds', default='5,15,30',
                    help="comma-separated hit_threshold values")
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    expected = cfg['geometry'].get('expected_channels', {})
    eq_cfg = cfg.get('event_quality', {})
    min_cluster_size = eq_cfg.get('min_cluster_size', 1)
    min_cluster_amplitude = eq_cfg.get('min_cluster_amplitude', 50.0)
    thresholds = [float(x) for x in args.thresholds.split(',')]

    # thr -> layer_idx -> {'events_with_cluster', 'n_clusters', 'max_amps'}
    stats = {thr: {li: {'ev': 0, 'nc': 0, 'amps': []} for li in range(8)} for thr in thresholds}
    n, seen = 0, 0

    for ev in data_loader.iter_all_events(data_cfg['raw_dir'], years=data_cfg.get('years')):
        seen += 1
        if (seen - 1) % args.stride != 0:
            continue
        n += 1
        with contextlib.redirect_stdout(io.StringIO()):
            X_layers, _, Y_layers, _ = preprocessing.unify_layers(ev['arrays'], expected)

        layers_map = list(zip(X_layers, X_IDX)) + list(zip(Y_layers, Y_IDX))
        for thr in thresholds:
            for layer_data, li in layers_map:
                cls = find_clusters_in_layer(
                    layer_data, li, ROW_PROJ[li],
                    hit_threshold=thr,
                    min_cluster_size=min_cluster_size,
                    min_cluster_amplitude=min_cluster_amplitude,
                )
                if cls:
                    stats[thr][li]['ev'] += 1
                    stats[thr][li]['nc'] += len(cls)
                    stats[thr][li]['amps'].append(max(c.max_amplitude for c in cls))
        if n >= args.sample:
            break

    print(f"\nExamined: {n} events | min_cluster_amplitude={min_cluster_amplitude}\n")
    print("Fraction of events with >=1 cluster in the plane:")
    header = f"{'row_idx':>7} {'proj':>5}" + "".join(f"{'thr='+str(int(t)):>10}" for t in thresholds)
    print(header)
    print("-" * len(header))
    # Print top to bottom: idx 0..7
    for li in range(8):
        row = f"{li:>7} {ROW_PROJ[li]:>5}"
        for thr in thresholds:
            frac = stats[thr][li]['ev'] / n if n else 0
            row += f"{frac:>9.1%} "
        print(row)

    print("\nMedian peak cluster amplitude per plane (when a cluster is present):")
    print(header)
    print("-" * len(header))
    for li in range(8):
        row = f"{li:>7} {ROW_PROJ[li]:>5}"
        for thr in thresholds:
            amps = stats[thr][li]['amps']
            med = int(np.median(amps)) if amps else 0
            row += f"{med:>10}"
        print(row)

    print("\nInterpretation (row_idx 6 = plane 7, 7 = plane 8, the deepest):")
    print(" - If idx6/idx7 are low at thr=30 but clearly higher at thr=15,")
    print("   and the amplitude exceeds min_cluster_amplitude, lower the threshold.")
    print(" - If the fraction stays low even at thr=5, the signal is genuinely weak")


if __name__ == '__main__':
    main()