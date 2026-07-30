"""
diag_occupancy.py - X/Y asymmetry of the selection.

Walks over a sample of raw events and reports, for each PHYSICAL plane (1-8):
  - occupancy: the fraction of events in which the plane is active, that is
    carries a channel at or above hit_threshold;
  - mean_signal: the mean summed amplitude of the plane, over positive samples;
  - mean_nhits: the mean number of channels above threshold.

It also histograms the number of active X and Y planes per event, which shows
why `few_y_rows` fires several times more often than `few_x_rows`.

No rescan is needed: the raw files are read directly, limited by --sample.

Run:
    python analysis/diag_occupancy.py --config config/settings.yaml --sample 20000
"""
import os
import sys
import argparse
from collections import defaultdict

import numpy as np

# Allow `python analysis/diag_occupancy.py` from the project root: put the root
# (the parent of analysis/) on the path so that the src package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import iter_all_events, CAMERA_TO_ROW, X_ROWS, Y_ROWS

ROW_PROJ = {1: 'X', 2: 'Y', 3: 'X', 4: 'Y', 5: 'X', 6: 'Y', 7: 'X', 8: 'Y'}


def accumulate(arrays, acc, threshold):
    """Update the accumulator with one event. Returns (n_active_x, n_active_y)."""
    row_active = {r: False for r in range(1, 9)}
    row_sig = defaultdict(float)
    row_nhits = defaultdict(int)

    for name, a in arrays.items():
        r = CAMERA_TO_ROW.get(name)
        if r is None:
            continue
        a = np.asarray(a, dtype=np.float64)
        pos = a[a > 0]
        row_sig[r] += float(pos.sum())
        nh = int(np.count_nonzero(a >= threshold))
        row_nhits[r] += nh
        if nh > 0:
            row_active[r] = True

    for r in range(1, 9):
        if row_active[r]:
            acc['occ'][r] += 1
        acc['sig'][r] += row_sig[r]
        acc['nhits'][r] += row_nhits[r]

    n_active = {r for r in range(1, 9) if row_active[r]}
    return len(n_active & X_ROWS), len(n_active & Y_ROWS)


def main():
    ap = argparse.ArgumentParser(description="Per-plane X/Y occupancy diagnostic")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--sample', type=int, default=20000, help="how many events to examine")
    ap.add_argument('--stride', type=int, default=1,
                    help="take every Nth event, for uniform coverage of the whole data set (e.g. 14)")
    ap.add_argument('--hit-threshold', type=int, default=None,
                    help="hit threshold (defaults to filters.hit_threshold from the config)")
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    thr = args.hit_threshold if args.hit_threshold is not None \
        else cfg.get('filters', {}).get('hit_threshold', 5)

    acc = {'occ': defaultdict(int), 'sig': defaultdict(float), 'nhits': defaultdict(int)}
    hist_x = np.zeros(5, dtype=int)  # index 0..4 = number of active X planes
    hist_y = np.zeros(5, dtype=int)

    n = 0
    seen = 0
    for ev in iter_all_events(data_cfg['raw_dir'], years=data_cfg.get('years')):
        seen += 1
        if (seen - 1) % args.stride != 0:
            continue
        ax, ay = accumulate(ev['arrays'], acc, thr)
        hist_x[ax] += 1
        hist_y[ay] += 1
        n += 1
        if n >= args.sample:
            break

    if n == 0:
        raise SystemExit("No events found; check raw_dir and years in the config.")

    print(f"\nHit threshold: {thr} | events examined: {n}\n")
    print(f"{'Row':>4} {'Proj':>5} {'Occupancy':>11} {'mean_signal':>13} {'mean_nhits':>12}")
    print("-" * 48)
    for r in range(1, 9):
        occ = acc['occ'][r] / n
        msig = acc['sig'][r] / n
        mnh = acc['nhits'][r] / n
        print(f"{r:>4} {ROW_PROJ[r]:>5} {occ:>10.1%} {msig:>13.1f} {mnh:>12.2f}")

    print("\nActive planes per event (histogram):")
    print(f"{'k':>3} {'active_X_rows':>15} {'active_Y_rows':>15}")
    for k in range(5):
        print(f"{k:>3} {hist_x[k]:>15} {hist_y[k]:>15}")

    # Summary per projection
    def summary(hist):
        tot = hist.sum()
        mean = sum(k * hist[k] for k in range(5)) / tot if tot else 0
        lt3 = hist[0] + hist[1] + hist[2]
        return mean, lt3 / tot if tot else 0
    mx, ltx = summary(hist_x)
    my, lty = summary(hist_y)
    print(f"\nMean active X planes: {mx:.2f} | events with <3 X planes: {ltx:.1%}")
    print(f"Mean active Y planes: {my:.2f} | events with <3 Y planes: {lty:.1%}")
    print("\nIf occupancy and mean_signal are systematically lower in the Y planes, the")
    print("cause is Y-specific (efficiency, pedestal, assembly). If they are comparable")
    print("but <3 Y is still frequent, look at which individual Y plane drops out.\n")


if __name__ == '__main__':
    main()