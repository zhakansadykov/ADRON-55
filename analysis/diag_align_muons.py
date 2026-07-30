"""
diag_align_muons.py - track-based alignment on clean single muons.

Unlike diag_geometry.py, which uses the centroid of the whole plane and is
therefore dominated by broad showers, this script uses the cluster centroids of
single muon tracks: narrow, clean positions. Selection: exactly one X track and
one Y track, each with at least three clusters. For an isotropic flux the mean
cluster position in a plane coincides with its geometric centre.

Prints an updated row_alignment_mm block (mm, layer_idx 0..7) for channel_to_mm.

Run:
    python analysis/diag_align_muons.py --config config/settings.yaml --sample 40000 --stride 7
"""
import os
import sys
import argparse
import contextlib
import io
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_loader, preprocessing, event_quality

ROW_PROJ = {0: 'X', 2: 'X', 4: 'X', 6: 'X', 1: 'Y', 3: 'Y', 5: 'Y', 7: 'Y'}


def collect_muon_clusters(cfg, data_cfg, sample, stride, min_hits, selection='single_muon'):
    """Collect cluster centroids per plane.

    selection='single_muon': exactly one X and one Y track, each with at least
      min_hits clusters. Clean muons, but they rarely reach the deepest planes
      behind the iron.
    selection='all_tracks': clusters of every track. Covers the deepest planes
      through showers and gives more statistics; for an isotropic flux the mean
      position is still the plane centre.
    """
    expected = cfg['geometry'].get('expected_channels', {})
    hit_thr = cfg['filters']['hit_threshold']
    cent = defaultdict(list)  # layer_idx -> centroids (channels)
    n, seen, n_clean = 0, 0, 0

    for ev in data_loader.iter_all_events(data_cfg['raw_dir'], years=data_cfg.get('years')):
        seen += 1
        if (seen - 1) % stride != 0:
            continue
        n += 1
        with contextlib.redirect_stdout(io.StringIO()):
            X_layers, _, Y_layers, _ = preprocessing.unify_layers(ev['arrays'], expected)
            stats = data_loader.compute_event_stats(ev, hit_threshold=hit_thr)
            report = event_quality.analyze_event(
                ev['global_id'], ev['event_time'], X_layers, Y_layers, stats, cfg)

        if selection == 'all_tracks':
            used = [t for t in report.tracks if len(t.hits) >= min_hits]
            if used:
                n_clean += 1
            for t in used:
                for h in t.hits:
                    cent[h.layer_idx].append(h.centroid)
        else:
            x_tracks = [t for t in report.tracks if t.projection == 'X']
            y_tracks = [t for t in report.tracks if t.projection == 'Y']
            if len(x_tracks) != 1 or len(y_tracks) != 1:
                continue
            if len(x_tracks[0].hits) < min_hits or len(y_tracks[0].hits) < min_hits:
                continue
            n_clean += 1
            for t in (x_tracks[0], y_tracks[0]):
                for h in t.hits:
                    cent[h.layer_idx].append(h.centroid)
        if n >= sample:
            break
    return cent, n, n_clean


def main():
    ap = argparse.ArgumentParser(description="Alignment from single muon tracks")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--sample', type=int, default=40000)
    ap.add_argument('--stride', type=int, default=7)
    ap.add_argument('--min-hits', type=int, default=3, help="minimum clusters per track")
    ap.add_argument('--selection', default='single_muon',
                    choices=['single_muon', 'all_tracks'],
                    help="single_muon: clean muons; all_tracks: every track (covers plane 8)")
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    geom = cfg['geometry']
    cw = geom['channel_width']
    expected = geom.get('expected_channels', {})
    current_align = geom.get('row_alignment_mm', {}) or {}

    cent, n_seen, n_clean = collect_muon_clusters(
        cfg, data_cfg, args.sample, args.stride, args.min_hits, args.selection)

    print(f"\nExamined: {n_seen} | clean single muons: {n_clean}\n")
    print(f"{'row':>4} {'proj':>5} {'n_clusters':>11} {'mean_centroid':>14} "
          f"{'n_ch/2':>8} {'new_offset_mm':>14} {'current':>10}")
    print("-" * 70)
    new_offsets = {}
    for li in range(8):
        n_ch = expected.get(f'row_{li+1}', geom.get('max_channels', 72))
        cur = current_align.get(li, current_align.get(str(li), 0.0))
        if not cent[li]:
            print(f"{li+1:>4} {ROW_PROJ[li]:>5}  (no data)")
            continue
        mc = float(np.mean(cent[li]))
        # Absolute offset: mean cluster centroid -> mm from the plane centre,
        # without the alignment currently in the config
        off = (mc - n_ch / 2.0) * cw
        new_offsets[li] = off
        print(f"{li+1:>4} {ROW_PROJ[li]:>5} {len(cent[li]):>11} {mc:>14.2f} "
              f"{n_ch/2:>8.1f} {off:>14.1f} {cur:>10.1f}")

    print("\nUpdated row_alignment_mm (from clean muons) - replace in settings.yaml:")
    print("row_alignment_mm:")
    for li in range(8):
        if li in new_offsets:
            print(f"  {li}: {new_offsets[li]:.1f}")
    print("\nAfter replacing: python analysis/reconstruct_all.py ... && python analysis/paper_plots.py ...")
    print("Then check chi2/ndf of the single tracks: target ~1-3 (isotropy).")


if __name__ == '__main__':
    main()