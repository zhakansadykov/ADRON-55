"""
diag_camera_lengths.py - measured channel count of each readout section.

Establishes how the parallel sections make up a plane:
  - if a section is about (plane width / number of sections) long, the sections
    TILE the plane and must be concatenated;
  - if a section is about as long as the whole plane, they OVERLAP and must be
    summed.

The length is the number of channels before the first -1 terminator. The spread
(min/max) shows how stable it is: a large spread means -1 also occurs mid-array
(dead channels), in which case a fixed padding per section is required.

Run:
    python analysis/diag_camera_lengths.py --config config/settings.yaml --sample 20000 --stride 14
"""
import os
import sys
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_loader

# Planes and their sections, in assembly order
ROW_CAMERAS = {
    1: ['left_g', 'right_g'],
    2: ['front_g', 'back_g'],
    3: ['left_1', 'middle_1', 'right_1'],
    4: ['front_1', 'back_1'],
    5: ['left_2', 'middle_2', 'right_2'],
    6: ['front_2', 'back_2'],
    7: ['left_3', 'middle_3', 'right_3'],
    8: ['front_3', 'back_3'],
}


def real_length(arr):
    a = np.asarray(arr)
    neg = np.where(a == -1)[0]
    return int(neg[0]) if len(neg) > 0 else len(a)


def main():
    ap = argparse.ArgumentParser(description="Measured length of each readout section")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--sample', type=int, default=20000)
    ap.add_argument('--stride', type=int, default=14)
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    expected = cfg['geometry'].get('expected_channels', {})

    lengths = defaultdict(list)
    present = defaultdict(int)
    n, seen = 0, 0
    for ev in data_loader.iter_all_events(data_cfg['raw_dir'], years=data_cfg.get('years')):
        seen += 1
        if (seen - 1) % args.stride != 0:
            continue
        arrays = ev['arrays']
        for cams in ROW_CAMERAS.values():
            for cam in cams:
                if cam in arrays:
                    present[cam] += 1
                    lengths[cam].append(real_length(arrays[cam]))
        n += 1
        if n >= args.sample:
            break

    print(f"\nExamined: {n} events\n")
    print(f"{'row':>4} {'expected':>9} {'camera':>10} {'present%':>9} "
          f"{'median_len':>11} {'min':>5} {'max':>5} {'sum_median':>11}")
    print("-" * 72)
    for row, cams in ROW_CAMERAS.items():
        exp = expected.get(f'row_{row}', '?')
        sum_med = 0
        rows_out = []
        for cam in cams:
            if not lengths[cam]:
                rows_out.append((cam, 0, 0, 0, 0))
                continue
            arr = np.array(lengths[cam])
            med = int(np.median(arr))
            sum_med += med
            rows_out.append((cam, present[cam] / n, med, int(arr.min()), int(arr.max())))
        for i, (cam, pr, med, mn, mx) in enumerate(rows_out):
            exp_str = str(exp) if i == 0 else ''
            sm_str = str(sum_med) if i == 0 else ''
            print(f"{row if i==0 else '':>4} {exp_str:>9} {cam:>10} {pr:>8.0%} "
                  f"{med:>11} {mn:>5} {mx:>5} {sm_str:>11}")
        print()

    print("Interpretation:")
    print(" - sum_median of the sections ~ expected row length -> TILING (concatenate).")
    print(" - median of each section ~ expected row length -> OVERLAP (sum).")
    print(" - min << max for a section -> -1 also appears mid-array (dead channels); fixed padding needed.")


if __name__ == '__main__':
    main()