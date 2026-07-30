"""
diag_runs.py - per-run search for dead readout sections (good-run selection).

Many events carry fewer than three active Y planes, and this is not absorption:
a deep X plane is often more active than a shallow Y plane. Individual Y sections
must therefore be absent or dead in part of the runs. The script reports:

  1. Globally, the fraction of events in which each section is actually written
     to the raw file. A section that is almost always absent is zero-filled by
     the preprocessing.
  2. Per file, the presence of the Y sections and the fraction of events with
     fewer than three active Y planes, which separates good from bad periods.
     The full table is written to CSV.

Run:
    python analysis/diag_runs.py --config config/settings.yaml
    python analysis/diag_runs.py --config config/settings.yaml --bad-threshold 0.20
"""
import os
import sys
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import iter_all_events, CAMERA_TO_ROW, X_ROWS, Y_ROWS

ALL_CAMERAS = [
    'left_g', 'right_g', 'front_g', 'back_g',
    'left_1', 'middle_1', 'right_1', 'front_1', 'back_1',
    'left_2', 'middle_2', 'right_2', 'front_2', 'back_2',
    'left_3', 'middle_3', 'right_3', 'front_3', 'back_3',
]
Y_CAMERAS = [c for c in ALL_CAMERAS if CAMERA_TO_ROW[c] in Y_ROWS]


def active_y_rows(arrays, threshold):
    rows = set()
    for name, a in arrays.items():
        r = CAMERA_TO_ROW.get(name)
        if r in Y_ROWS:
            a = np.asarray(a, dtype=np.float64)
            if np.any(a >= threshold):
                rows.add(r)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Per-run readout section presence (dead camera search)")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--hit-threshold', type=int, default=None)
    ap.add_argument('--bad-threshold', type=float, default=0.20,
                    help="fraction of events with <3 Y planes above which a file is flagged")
    ap.add_argument('--out', default='output/paper_plots')
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    thr = args.hit_threshold if args.hit_threshold is not None \
        else cfg.get('filters', {}).get('hit_threshold', 5)

    # Accumulators
    global_present = defaultdict(int)
    global_n = 0
    per_file_n = defaultdict(int)
    per_file_present = defaultdict(lambda: defaultdict(int))  # file -> cam -> count present
    per_file_lt3y = defaultdict(int)  # file -> events with <3 active Y-rows

    for ev in iter_all_events(data_cfg['raw_dir'], years=data_cfg.get('years')):
        arrays = ev['arrays']
        src = ev.get('source_file', '?')
        global_n += 1
        per_file_n[src] += 1
        for cam in ALL_CAMERAS:
            if cam in arrays:
                global_present[cam] += 1
                per_file_present[src][cam] += 1
        if active_y_rows(arrays, thr) < 3:
            per_file_lt3y[src] += 1

    if global_n == 0:
        raise SystemExit("No events; check raw_dir and years.")

    print(f"\nHit threshold: {thr} | total events: {global_n} | files: {len(per_file_n)}\n")

    print("Global section presence (fraction of events in which the section is recorded):")
    print(f"{'camera':>10} {'proj':>5} {'row':>4} {'present':>10}")
    print("-" * 34)
    for cam in ALL_CAMERAS:
        r = CAMERA_TO_ROW[cam]
        proj = 'Y' if r in Y_ROWS else 'X'
        frac = global_present[cam] / global_n
        mark = '  <-- rarely present' if frac < 0.5 else ''
        print(f"{cam:>10} {proj:>5} {r:>4} {frac:>9.1%}{mark}")

    # Files flagged by the <3 Y fraction
    print(f"\nFiles with a fraction of events (<3 active Y planes) above {args.bad_threshold:.0%}:")
    print(f"{'file':>18} {'n_ev':>7} {'<3Y':>7}  " + " ".join(f"{c:>8}" for c in Y_CAMERAS))
    bad_files, bad_events = [], 0
    rows_csv = []
    for src in sorted(per_file_n):
        n = per_file_n[src]
        lt3 = per_file_lt3y[src] / n if n else 0
        ypres = {c: (per_file_present[src][c] / n if n else 0) for c in Y_CAMERAS}
        rows_csv.append((src, n, lt3, ypres))
        if lt3 > args.bad_threshold:
            bad_files.append(src)
            bad_events += n
            print(f"{src:>18} {n:>7} {lt3:>6.0%}  " + " ".join(f"{ypres[c]:>7.0%}" for c in Y_CAMERAS))

    good_files = [s for s in per_file_n if s not in set(bad_files)]
    good_events = global_n - bad_events
    print(f"\nBy the <3Y criterion: {len(good_files)} good files ({good_events} events), "
          f"{len(bad_files)} flagged ({bad_events} events, {bad_events/global_n:.1%}).")

    # === Mode A: structurally dead Y sections (hard defect) ===
    # As opposed to mode B, where every section is present but <3Y arises from
    # physics or an efficiency drift.
    REQUIRED_Y = ['front_1', 'back_1', 'front_3', 'back_3']
    dead_files, dead_events = [], 0
    for src in sorted(per_file_n):
        n = per_file_n[src]
        if any((per_file_present[src][c] / n if n else 0) < 0.5 for c in REQUIRED_Y):
            dead_files.append(src)
            dead_events += n
    print(f"\nMode A (dead Y sections {REQUIRED_Y}): "
          f"{len(dead_files)} files, {dead_events} events ({dead_events/global_n:.1%}).")
    print("These are candidates for hard exclusion (good-run selection).")

    excl_path = os.path.join(args.out, 'exclude_files.txt')
    with open(excl_path, 'w', encoding='utf-8') as f:
        f.write("# Runs with dead Y sections (mode A), for data.exclude_files in settings.yaml\n")
        f.write("exclude_files:\n")
        for src in dead_files:
            f.write(f"  - {src}\n")
    print(f"Exclusion list for the config: {excl_path}")

    # Full per-file table, as CSV
    os.makedirs(args.out, exist_ok=True)
    import csv
    path = os.path.join(args.out, 'run_quality.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['source_file', 'n_events', 'frac_lt3_y_rows'] + [f'present_{c}' for c in Y_CAMERAS])
        for src, n, lt3, ypres in rows_csv:
            w.writerow([src, n, f"{lt3:.4f}"] + [f"{ypres[c]:.4f}" for c in Y_CAMERAS])
    print(f"\nFull per-file table: {path}")
    print("Flagged files are candidates for exclusion (good-run selection).\n")


if __name__ == '__main__':
    main()