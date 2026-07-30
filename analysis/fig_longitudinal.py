"""
fig_longitudinal.py - the longitudinal absorption profile (publication figure).

For each plane of the detector:
  - occupancy, the fraction of events with signal in the plane, which traces the
    absorption down the stack;
  - the mean deposition per channel, that is the summed above-pedestal signal of
    the plane divided by its channel count.

Normalizing per channel is essential: the planes carry between 48 and 72
channels, so the raw X and Y sums are not comparable and the channel count would
otherwise masquerade as a property of the gamma block.

The occupancy uncertainties are Wilson intervals, which stay meaningful at
p = 1, where the binomial sqrt(pq/n) degenerates to zero.

The depth axis is the cumulative nuclear interaction length; the CSV also holds
the radiation length and the depth in millimetres. The report printed at the end
carries every number of the figure, including how many events the good-run
selection removed.

Output: output/paper_plots/fig_longitudinal.(png,pdf) and
        longitudinal_profile.csv

Run:
    python analysis/fig_longitudinal.py --config config/settings.yaml
"""
import os
import sys
import math
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

ROW_PROJ = {0: 'X', 1: 'Y', 2: 'X', 3: 'Y', 4: 'X', 5: 'Y', 6: 'X', 7: 'Y'}
X_IDX = [6, 4, 2, 0]   # order of X_layers -> layer_idx
Y_IDX = [7, 5, 3, 1]


def wilson(k, n, z=1.0):
    """Wilson interval for the fraction k/n. Returns (lo, hi)."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - hw), min(1.0, center + hw)


def cumulative_depth(cfg):
    """Cumulative depth (X0, lambda_I, mm) above each plane (idx 0..7).

    before_row_N is either a single layer {material, thickness_mm} or a list of
    them, for example Pb 220 + air 2200 + Pb 130 + Fe 100 above plane 3.
    """
    ab = cfg['absorber']
    mats = ab['materials']
    x0 = np.zeros(8); li = np.zeros(8); mm = np.zeros(8)
    cx = clam = cmm = 0.0
    for i in range(8):
        entry = ab['layers'].get(f'before_row_{i+1}')
        layers = entry if isinstance(entry, list) else ([entry] if entry else [])
        for layer in layers:
            m = mats[layer['material']]
            th = layer['thickness_mm']
            cx += th / m['X0_mm']
            clam += th / m['lambda_I_mm']
            cmm += th
        x0[i], li[i], mm[i] = cx, clam, cmm
    return x0, li, mm


def main():
    ap = argparse.ArgumentParser(description="Longitudinal absorption profile")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--sample', type=int, default=40000)
    ap.add_argument('--stride', type=int, default=7)
    ap.add_argument('--out', default='output/paper_plots')
    ap.add_argument('--from-npz', default=None,
                    help="load the accumulators produced by longitudinal_blocks.py "
                         "instead of scanning the raw files")
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    expected = cfg['geometry'].get('expected_channels', {})
    ped = cfg['filters'].get('hit_threshold', 5)
    trigger = cfg['filters'].get('min_total_energy', 500)
    exclude_set = set(data_cfg.get('exclude_files', []) or [])

    n_ch_row = np.array([expected.get(f'row_{i+1}',
                                      cfg['geometry'].get('max_channels', 72))
                         for i in range(8)], dtype=float)

    x0, li, mm = cumulative_depth(cfg)

    if args.from_npz:
        d = np.load(args.from_npz)
        occ, esum, esq = d['occ'], d['esum'], d['esq']
        n, seen, excluded, below_trigger = (int(v) for v in d['counts'])
    else:
        occ = np.zeros(8, dtype=float)          # events with signal in the plane
        esum = np.zeros(8, dtype=float)         # summed plane energy
        esq = np.zeros(8, dtype=float)          # sum of squares, for the standard error
        n = 0
        seen = 0
        excluded = 0
        below_trigger = 0
        for ev in data_loader.iter_all_events(data_cfg['raw_dir'], years=data_cfg.get('years')):
            if ev.get('source_file') in exclude_set:      # good-run selection
                excluded += 1
                continue
            seen += 1
            if (seen - 1) % args.stride != 0:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                X_layers, _, Y_layers, _ = preprocessing.unify_layers(ev['arrays'], expected)
            row_e = np.zeros(8, dtype=np.float64)
            row_hit = np.zeros(8, dtype=bool)
            for layers, idxs in ((X_layers, X_IDX), (Y_layers, Y_IDX)):
                for layer, idx in zip(layers, idxs):
                    a = np.asarray(layer, dtype=np.float64)
                    m = a >= ped
                    row_e[idx] = float(a[m].sum())
                    row_hit[idx] = bool(np.any(m))
            # Software trigger: total energy above threshold
            if row_e.sum() < trigger:
                below_trigger += 1
                continue
            n += 1
            esum += row_e
            esq += row_e * row_e
            occ += row_hit.astype(float)
            if n >= args.sample:
                break

    if n == 0:
        raise SystemExit("No events; check raw_dir, years and exclude_files.")

    occupancy = occ / n
    occ_lo = np.zeros(8); occ_hi = np.zeros(8)
    for i in range(8):
        occ_lo[i], occ_hi[i] = wilson(occ[i], n)
    occ_err2 = np.vstack([occupancy - occ_lo, occ_hi - occupancy])  # (2,8)

    mean_e = esum / n
    var_e = esq / n - mean_e ** 2
    e_err = np.sqrt(np.maximum(var_e, 0) / n)              # SEM
    e_perch = mean_e / n_ch_row
    e_perch_err = e_err / n_ch_row

    os.makedirs(args.out, exist_ok=True)

    # --- CSV ---
    import csv
    csv_path = os.path.join(args.out, 'longitudinal_profile.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['row', 'proj', 'n_channels', 'depth_X0', 'depth_lambdaI',
                    'depth_mm', 'occupancy', 'occ_lo', 'occ_hi',
                    'mean_energy_adc', 'energy_err',
                    'energy_per_channel', 'energy_per_channel_err'])
        for i in range(8):
            w.writerow([i + 1, ROW_PROJ[i], int(n_ch_row[i]),
                        f"{x0[i]:.3f}", f"{li[i]:.4f}", f"{mm[i]:.0f}",
                        f"{occupancy[i]:.5f}", f"{occ_lo[i]:.5f}", f"{occ_hi[i]:.5f}",
                        f"{mean_e[i]:.1f}", f"{e_err[i]:.1f}",
                        f"{e_perch[i]:.2f}", f"{e_perch_err[i]:.2f}"])

    # --- Figure: separate X and Y curves ---
    xrows = np.array([2, 4, 6])          # hadron-block X: planes 3, 5, 7
    yrows = np.array([3, 5, 7])          # hadron-block Y: planes 4, 6, 8
    xg = np.array([0]); yg = np.array([1])   # gamma planes 1 (X) and 2 (Y)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True)

    def plot_xy(ax, y, yerr, ylabel, logy=False):
        def sub(err, idxs):
            e = np.asarray(err)
            return e[:, idxs] if e.ndim == 2 else e[idxs]
        ax.errorbar(li[xrows], y[xrows], yerr=sub(yerr, xrows), fmt='o-', ms=7,
                    color=COLORS['x'], capsize=3, lw=1.5,
                    label='X planes (rows 3,5,7)')
        ax.errorbar(li[yrows], y[yrows], yerr=sub(yerr, yrows), fmt='s-', ms=7,
                    color=COLORS['y'], capsize=3, lw=1.5,
                    label='Y planes (rows 4,6,8)')
        ax.errorbar(li[xg], y[xg], yerr=sub(yerr, xg), fmt='o', ms=9, mfc='none',
                    color=COLORS['x'], capsize=3, label='Gamma X (row 1)')
        ax.errorbar(li[yg], y[yg], yerr=sub(yerr, yg), fmt='s', ms=9, mfc='none',
                    color=COLORS['y'], capsize=3, label='Gamma Y (row 2)')
        for i in range(8):
            ax.annotate(f"r{i+1}", (li[i], y[i]), textcoords="offset points",
                        xytext=(5, 6), fontsize=8, color='gray')
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale('log')
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=8)

    plot_xy(ax1, occupancy, occ_err2, 'Occupancy (fraction of events)')
    ax1.set_ylim(0, 1.05)
    ax1.axvspan(li[1], li[2], color='gray', alpha=0.12)
    ax1.annotate('22 cm Pb \u00b7 2.2 m air\n13 cm Pb \u00b7 10 cm Fe', (0.5 * (li[1] + li[2]), 0.15),
                 ha='center', fontsize=8, color='dimgray')

    plot_xy(ax2, e_perch, e_perch_err,
            'Mean deposition per channel (ADC, arb. units)')
    ax2.axvspan(li[1], li[2], color='gray', alpha=0.12)
    ax2.set_xlabel(r'Cumulative nuclear interaction length $\lambda_I$')

    fig.tight_layout()
    base = os.path.join(args.out, 'fig_longitudinal')
    save_figure(fig, base, caption=(
        "Longitudinal development in the ADRON-55 ionization calorimeter. "
        "Top: plane occupancy (fraction of events above the software threshold with "
        "signal in the plane) versus cumulative nuclear interaction length lambda_I; "
        "X and Y planes are shown separately to expose the per-projection efficiency "
        "offset. Bottom: mean energy deposition per channel (ADC, arbitrary units); "
        "normalizing by the number of channels (48-72 per plane) makes the X and Y "
        "planes directly comparable. The shaded band marks the material between the "
        "gamma and hadron blocks: the 22 cm lead target, the 2.2 m air gap, the "
        "13 cm lead hadron converter, and the first 10 cm iron layer. Good runs only; error "
        "bars are Wilson intervals (occupancy) and the standard error of the mean "
        "(energy)."))
    plt.close(fig)
    png = base + '.png'

    # --- terminal output ---
    print(f"Events used: {n} (examined {seen}, "
          f"excluded by good-run: {excluded}, below trigger: {below_trigger})")
    print(f"Written: {png}\n         {png.replace('.png', '.pdf')}\n         {csv_path}")

    # ================= PASTE-BLOCK =================
    def monotone(vals):
        return "yes" if np.all(np.diff(vals) <= 1e-9) else "NO"

    x_seq = occupancy[[0, 2, 4, 6]]   # rows 1,3,5,7
    y_seq = occupancy[[1, 3, 5, 7]]   # rows 2,4,6,8
    P = print
    P("")
    P("===== PASTE-BLOCK LONGITUDINAL v2 START =====")
    P(f"config={args.config} sample={args.sample} stride={args.stride} "
      f"ped={ped} trigger={trigger}")
    P(f"events_used={n} seen={seen} excluded_badrun={excluded} "
      f"below_trigger={below_trigger} exclude_files={len(exclude_set)}")
    P(f"{'row':>3} {'proj':>4} {'n_ch':>4} {'lamI':>6} "
      f"{'occ':>8} {'occ_lo':>8} {'occ_hi':>8} "
      f"{'E_row':>9} {'E/ch':>8} {'err':>6}")
    for i in range(8):
        P(f"{i+1:>3} {ROW_PROJ[i]:>4} {int(n_ch_row[i]):>4} {li[i]:>6.2f} "
          f"{occupancy[i]:>8.5f} {occ_lo[i]:>8.5f} {occ_hi[i]:>8.5f} "
          f"{mean_e[i]:>9.1f} {e_perch[i]:>8.2f} {e_perch_err[i]:>6.2f}")
    P(f"check: occ monotonic X(1,3,5,7)={monotone(x_seq)}  "
      f"Y(2,4,6,8)={monotone(y_seq)}")
    P(f"check: E/ch ratios  row2/row1={e_perch[1]/e_perch[0]:.3f}  "
      f"row4/row3={e_perch[3]/e_perch[2]:.3f}  "
      f"(raw-sum ratios were {mean_e[1]/mean_e[0]:.3f} / {mean_e[3]/mean_e[2]:.3f}; "
      f"channel ratios 69/50={69/50:.2f}, 72/48={72/48:.2f})")
    P("===== PASTE-BLOCK LONGITUDINAL v2 END =====")


if __name__ == '__main__':
    main()