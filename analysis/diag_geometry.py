"""
diag_geometry.py - data-driven plane alignment.

The planes differ in width and assembly, and the upper ones sit far above the
lower block. Any displacement of one plane centre relative to the others acts
over a long lever arm and imposes a fixed tilt, which biases the azimuth.

For an isotropic cosmic-ray flux the mean signal position in each plane must
coincide with its geometric centre. This script measures that mean position, in
channels and in millimetres under the current channel_to_mm, for each of the
eight planes, and prints the offsets (mm) to be SUBTRACTED so that every plane
centre maps to zero.

Run:
    python analysis/diag_geometry.py --config config/settings.yaml --sample 20000 --stride 14
"""
import os
import sys
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_loader, preprocessing
from src.preprocessing import channel_to_mm

# (layer_idx, projection) from unify_layers: X = [7, 5, 3, 1] at idx [6, 4, 2, 0],
# Y = [8, 6, 4, 2] at idx [7, 5, 3, 1]
ROW_INFO = {0: 'X', 2: 'X', 4: 'X', 6: 'X', 1: 'Y', 3: 'Y', 5: 'Y', 7: 'Y'}


def weighted_centroid(layer):
    layer = np.asarray(layer, dtype=np.float64)
    s = layer.sum()
    if s <= 0:
        return None
    idx = np.arange(len(layer))
    return float((idx * layer).sum() / s)


def main():
    ap = argparse.ArgumentParser(description="Data-driven per-plane alignment offsets")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--sample', type=int, default=20000)
    ap.add_argument('--stride', type=int, default=14)
    ap.add_argument('--min-row-signal', type=float, default=100.0,
                    help="minimum summed plane signal for its centroid to be counted")
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    expected = cfg['geometry'].get('expected_channels', {})

    cent_ch = defaultdict(list)   # layer_idx -> centroids, in channels
    cent_mm = defaultdict(list)   # layer_idx -> positions, in mm under the current mapping

    n, seen = 0, 0
    import contextlib, io
    for ev in data_loader.iter_all_events(data_cfg['raw_dir'], years=data_cfg.get('years')):
        seen += 1
        if (seen - 1) % args.stride != 0:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            X_layers, X_idx, Y_layers, Y_idx = preprocessing.unify_layers(ev['arrays'], expected)
        for layers, idxs in ((X_layers, X_idx), (Y_layers, Y_idx)):
            for layer, li in zip(layers, idxs):
                if np.asarray(layer, dtype=np.float64).sum() < args.min_row_signal:
                    continue
                c = weighted_centroid(layer)
                if c is None:
                    continue
                cent_ch[li].append(c)
                cent_mm[li].append(channel_to_mm(c, li, cfg))
        n += 1
        if n >= args.sample:
            break

    print(f"\nEvents examined: {n}\n")
    print(f"{'row':>4} {'proj':>5} {'n_ch':>5} {'mean_centroid_ch':>17} {'n_ch/2':>8} "
          f"{'mean_mm(now)':>14} {'-> offset_mm':>13}")
    print("-" * 72)
    offsets = {}
    for li in range(8):
        proj = ROW_INFO[li]
        n_ch = expected.get(f'row_{li+1}', cfg['geometry'].get('max_channels', 72))
        if not cent_mm[li]:
            print(f"{li+1:>4} {proj:>5} {n_ch:>5}  (no data)")
            continue
        mc = float(np.mean(cent_ch[li]))
        mm = float(np.mean(cent_mm[li]))
        offsets[li] = mm  # millimetres to SUBTRACT so that the plane centre maps to zero
        print(f"{li+1:>4} {proj:>5} {n_ch:>5} {mc:>17.2f} {n_ch/2:>8.1f} "
              f"{mm:>14.1f} {mm:>13.1f}")

    print("\nInterpretation:")
    print(" - If mean_mm(now) is ~0 for every plane, the geometry is centred; look elsewhere.")
    print(" - If mean_mm is systematically non-zero and differs between the upper (1, 2)")
    print("   and the lower planes, that is the source of the fixed tilt. offset_mm is the correction.")
    print("\nOffsets (mm) for channel_to_mm (row_alignment_mm):")
    print("row_alignment_mm:")
    for li in range(8):
        if li in offsets:
            print(f"  {li}: {offsets[li]:.1f}")


if __name__ == '__main__':
    main()