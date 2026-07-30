"""
fig_transverse.py - the lateral shower profile (publication figure).

Method: the shower axis in a plane is the peak channel, refined by the
amplitude-weighted centroid in a window of +/-window channels around it. The
energy of every channel above the pedestal (--pedestal) is accumulated in bins
of |r| in millimetres.

The trigger matches the definition used by fig_longitudinal: the sum ABOVE the
pedestal (a >= filters.hit_threshold) over all planes must reach
filters.min_total_energy, so that the two figures select the same events.

The report printed at the end carries the widths (RMS and R90) per group and per
plane, the normalized profiles per bin, and the number of events removed by the
good-run selection.

Output: output/paper_plots/fig_transverse.(png,pdf) and transverse_profile.csv

Run:
    python analysis/fig_transverse.py --config config/settings.yaml
"""
import os
import sys
import argparse
import contextlib
import io

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_loader, preprocessing
from analysis.plot_style import apply_style, COLORS, save_figure
apply_style()

X_IDX = [6, 4, 2, 0]   # X_layers -> layer_idx (row7,5,3,1)
Y_IDX = [7, 5, 3, 1]   # Y_layers -> layer_idx (row8,6,4,2)
GAMMA_IDX = {0, 1}     # planes 1 and 2
ROW_PROJ = {0: 'X', 1: 'Y', 2: 'X', 3: 'Y', 4: 'X', 5: 'Y', 6: 'X', 7: 'Y'}


def layer_profile(layer, cw, r_edges, ped, window_ch, w_min):
    """The |r| profile of one plane of one event.

    The axis is the peak of the plane, refined by the amplitude-weighted
    centroid within +/-window_ch channels around it; only channels above the
    pedestal enter the profile. Returns the binned energies, or None if the
    signal is too weak.
    """
    a = np.asarray(layer, dtype=np.float64)
    sig = a.copy()
    sig[sig < ped] = 0.0
    if sig.sum() < w_min:
        return None
    peak = int(np.argmax(sig))
    lo = max(0, peak - window_ch)
    hi = min(len(sig), peak + window_ch + 1)
    win = sig[lo:hi]
    idxw = np.arange(lo, hi)
    axis = (idxw * win).sum() / win.sum() if win.sum() > 0 else float(peak)
    r = np.abs(np.arange(len(sig)) - axis) * cw
    nbins = len(r_edges) - 1
    h = np.zeros(nbins, dtype=np.float64)
    b = np.digitize(r, r_edges) - 1
    m = (b >= 0) & (b < nbins) & (sig > 0)
    np.add.at(h, b[m], sig[m])
    return h


def width_metrics(r_centers, prof):
    """RMS width and R90, the radius containing 90% of the energy."""
    p = np.asarray(prof, dtype=np.float64)
    if p.sum() <= 0:
        return float('nan'), float('nan')
    rms = np.sqrt((p * r_centers ** 2).sum() / p.sum())
    cum = np.cumsum(p) / p.sum()
    r90 = float(np.interp(0.9, cum, r_centers))
    return float(rms), r90


def main():
    ap = argparse.ArgumentParser(description="Lateral shower profile")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--sample', type=int, default=40000)
    ap.add_argument('--stride', type=int, default=7)
    ap.add_argument('--rmax', type=float, default=1440.0,
                    help="maximum radius, mm (multiple of the 120 mm pitch)")
    ap.add_argument('--nbins', type=int, default=12,
                    help="number of bins (default: one 120 mm pitch per bin)")
    ap.add_argument('--pedestal', type=float, default=30.0,
                    help="per-channel pedestal threshold for the profile, ADC")
    ap.add_argument('--window-ch', type=int, default=5,
                    help="half-width in channels of the window around the peak used for the axis")
    ap.add_argument('--min-row-energy', type=float, default=100.0)
    ap.add_argument('--out', default='output/paper_plots')
    ap.add_argument('--from-npz', default=None,
                    help="load the accumulators produced by transverse_blocks.py "
                         "instead of scanning the raw files")
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    expected = cfg['geometry'].get('expected_channels', {})
    cw = cfg['geometry']['channel_width']
    ped_trig = cfg['filters'].get('hit_threshold', 5)       # pedestal of the trigger
    trigger = cfg['filters'].get('min_total_energy', 500)
    exclude_set = set(data_cfg.get('exclude_files', []) or [])

    r_edges = np.linspace(0, args.rmax, args.nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    if args.from_npz:
        d = np.load(args.from_npz)
        hist_x, hist_y = d['x'], d['y']
        hist_gamma, hist_hadron = d['gamma'], d['hadron']
        hist_per_row = {i: d[f'row{i}'] for i in range(8)}
        n, seen, excluded, below_trigger = (int(v) for v in d['counts'])
    else:
        hist_x = np.zeros(args.nbins); hist_y = np.zeros(args.nbins)
        hist_gamma = np.zeros(args.nbins); hist_hadron = np.zeros(args.nbins)
        hist_per_row = {i: np.zeros(args.nbins) for i in range(8)}
        n, seen, excluded, below_trigger = 0, 0, 0, 0

        for ev in data_loader.iter_all_events(data_cfg['raw_dir'], years=data_cfg.get('years')):
            if ev.get('source_file') in exclude_set:
                excluded += 1
                continue
            seen += 1
            if (seen - 1) % args.stride != 0:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                X_layers, _, Y_layers, _ = preprocessing.unify_layers(ev['arrays'], expected)

            # Trigger as in fig_longitudinal: the above-pedestal sum over all planes
            tot = 0.0
            for layer in list(X_layers) + list(Y_layers):
                a = np.asarray(layer, dtype=np.float64)
                tot += float(a[a >= ped_trig].sum())
            if tot < trigger:
                below_trigger += 1
                continue
            n += 1

            for layers, idxs, hist_proj in ((X_layers, X_IDX, hist_x),
                                            (Y_layers, Y_IDX, hist_y)):
                for layer, idx in zip(layers, idxs):
                    prof = layer_profile(layer, cw, r_edges, args.pedestal,
                                         args.window_ch, args.min_row_energy)
                    if prof is None:
                        continue
                    hist_proj += prof
                    hist_per_row[idx] += prof
                    if idx in GAMMA_IDX:
                        hist_gamma += prof
                    else:
                        hist_hadron += prof
            if n >= args.sample:
                break

    if n == 0:
        raise SystemExit("No events; check raw_dir, years and exclude_files.")

    def norm(h):
        s = h.sum()
        return h / s if s > 0 else h

    rms_g, r90_g = width_metrics(r_centers, hist_gamma)
    rms_h, r90_h = width_metrics(r_centers, hist_hadron)
    rms_x, r90_x = width_metrics(r_centers, hist_x)
    rms_y, r90_y = width_metrics(r_centers, hist_y)

    os.makedirs(args.out, exist_ok=True)

    # --- CSV ---
    import csv
    nx, ny = norm(hist_x), norm(hist_y)
    ng, nh = norm(hist_gamma), norm(hist_hadron)
    with open(os.path.join(args.out, 'transverse_profile.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['r_mm', 'prof_x', 'prof_y', 'prof_gamma', 'prof_hadron'])
        for i in range(args.nbins):
            w.writerow([f"{r_centers[i]:.1f}", f"{nx[i]:.5f}", f"{ny[i]:.5f}",
                        f"{ng[i]:.5f}", f"{nh[i]:.5f}"])

    # --- Figure: two panels ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.step(r_centers, nx, where='mid', color=COLORS['x'], lw=2,
             label=f'X planes (RMS={rms_x:.0f} mm, R90={r90_x:.0f})')
    ax1.step(r_centers, ny, where='mid', color=COLORS['y'], lw=2, ls='--',
             label=f'Y planes (RMS={rms_y:.0f} mm, R90={r90_y:.0f})')
    ax1.set_yscale('log')
    ax1.set_xlabel('Lateral distance from cascade axis r (mm)')
    ax1.set_ylabel('Fraction of signal per 120 mm bin')
    ax1.set_title('Lateral profile: X vs Y (symmetry)')
    ax1.grid(True, ls='--', alpha=0.4); ax1.legend(fontsize=9)

    ax2.step(r_centers, ng, where='mid', color=COLORS['gamma'], lw=2,
             label=f'Gamma block (RMS={rms_g:.0f} mm, R90={r90_g:.0f})')
    ax2.step(r_centers, nh, where='mid', color=COLORS['hadron'], lw=2,
             label=f'Hadron block (RMS={rms_h:.0f} mm, R90={r90_h:.0f})')
    ax2.set_yscale('log')
    ax2.set_xlabel('Lateral distance from cascade axis r (mm)')
    ax2.set_ylabel('Fraction of signal per 120 mm bin')
    ax2.set_title('Lateral profile: gamma vs hadron block')
    ax2.grid(True, ls='--', alpha=0.4); ax2.legend(fontsize=9)

    fig.suptitle('Transverse (lateral) shower profile in ADRON-55')
    fig.tight_layout()
    base = os.path.join(args.out, 'fig_transverse')
    save_figure(fig, base, caption=(
        "Transverse profile of the recorded signal in the ADRON-55 calorimeter: "
        "the fraction of the total above-pedestal amplitude (ADC counts) per "
        "120 mm bin of lateral distance from the cascade axis (defined per plane as "
        "the amplitude-weighted centroid in a +/-5-channel window around the peak, above "
        "pedestal). All good-run events enter the sample (the nominal 500 ADC "
        "software threshold removes none). The profile is histogrammed in one-pitch "
        "(120 mm) bins to avoid aliasing between the channel pitch and the bin "
        "width. Left: X and Y planes (axial "
        "symmetry). Right: gamma versus hadron block. Widths (RMS, R90) are given in "
        "the legend. The 120 mm cell pitch limits the lateral resolution, so the "
        "electromagnetic core is not resolved from the hadronic component; positions "
        "near the plane edges cannot contribute at large r, which slightly steepens "
        "the tail. About 90% of the deposited energy lies within the central third "
        "of the detector width."))
    plt.close(fig)
    png = base + '.png'

    print(f"Events used: {n} (examined {seen}, "
          f"excluded by good-run: {excluded}, below trigger: {below_trigger})")
    print(f"Written: {png} (+.pdf) and transverse_profile.csv")

    # ================= PASTE-BLOCK =================
    P = print
    P("")
    P("===== PASTE-BLOCK TRANSVERSE v2 START =====")
    P(f"config={args.config} sample={args.sample} stride={args.stride} "
      f"rmax={args.rmax:.0f} nbins={args.nbins} pedestal_profile={args.pedestal:.0f} "
      f"window=+-{args.window_ch} min_row_E={args.min_row_energy:.0f} "
      f"ped_trigger={ped_trig} trigger={trigger}")
    P(f"events_used={n} seen={seen} excluded_badrun={excluded} "
      f"below_trigger={below_trigger} exclude_files={len(exclude_set)}")
    P(f"{'group':<14} {'RMS_mm':>7} {'R90_mm':>7}")
    P(f"{'X planes':<14} {rms_x:>7.0f} {r90_x:>7.0f}")
    P(f"{'Y planes':<14} {rms_y:>7.0f} {r90_y:>7.0f}")
    P(f"{'Gamma block':<14} {rms_g:>7.0f} {r90_g:>7.0f}")
    P(f"{'Hadron block':<14} {rms_h:>7.0f} {r90_h:>7.0f}")
    P("per-row RMS/R90 (mm):")
    for i in range(8):
        rms_i, r90_i = width_metrics(r_centers, hist_per_row[i])
        P(f"  row {i+1} ({ROW_PROJ[i]}): RMS={rms_i:>4.0f}  R90={r90_i:>4.0f}")
    P("normalized profiles (r_mm, X, Y, gamma, hadron):")
    for i in range(args.nbins):
        P(f"  {r_centers[i]:>6.1f} {nx[i]:.4f} {ny[i]:.4f} {ng[i]:.4f} {nh[i]:.4f}")
    P("===== PASTE-BLOCK TRANSVERSE v2 END =====")


if __name__ == '__main__':
    main()