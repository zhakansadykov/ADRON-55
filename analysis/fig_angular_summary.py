"""
fig_angular_summary.py - the combined angular figure for the paper.

  (a) Zenith distribution of the clean 3+3 sample in 6-degree bins, which
      averages over the comb of quantized slopes; the cos^2(theta)*sin(theta)
      shape is overlaid, normalized to the populated bins and shown without an
      acceptance correction.
  (b) Azimuth of the clean sample with the isotropic expectation and the
      harmonic fit a0 + A1*cos(phi - phi1) + A2*cos(2*(phi - phi2)), so that the
      residual modulation is shown explicitly as instrumental.
  (c) chi2/ndf against uniformity by track class (n_rows <= 4, == 5, >= 6),
      computed from reco.csv. The open marker is the level of the uncorrected
      development-stage pipeline, kept as a reference point.

Output: output/paper_plots/fig_angular_summary.(png,pdf)

Run:
    python analysis/fig_angular_summary.py --reco output/paper_plots/reco.csv
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.plot_style import apply_style, COLORS, save_figure
apply_style()

AZ_BINS = 36
RAW_PIPELINE_CHI2 = 1324.0   # measured at the development stage, before the
                             # corrections; a reference point, not reproducible
                             # from the current reco.csv


def chi2_uniform(a, nbins=AZ_BINS):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    counts, _ = np.histogram(a, bins=nbins, range=(0, 360))
    u = len(a) / nbins
    chi2 = float(np.sum((counts - u) ** 2 / u)) if u > 0 else float('nan')
    return chi2 / (nbins - 1), counts, u


def azimuth_harmonics(counts, nbins=AZ_BINS):
    centers = (np.arange(nbins) + 0.5) * (360.0 / nbins)
    phi = np.radians(centers)
    M = np.column_stack([np.ones_like(phi), np.cos(phi), np.sin(phi),
                         np.cos(2 * phi), np.sin(2 * phi)])
    coef, *_ = np.linalg.lstsq(M, counts.astype(float), rcond=None)
    a0, c1, s1, c2, s2 = coef
    return {'a0': float(a0),
            'A1': float(np.hypot(c1, s1)),
            'phi1': float(np.degrees(np.arctan2(s1, c1)) % 360.0),
            'A2': float(np.hypot(c2, s2)),
            'phi2': float((np.degrees(np.arctan2(s2, c2)) / 2.0) % 180.0),
            'coef': coef}


def main():
    ap = argparse.ArgumentParser(description="Combined angular figure (Fig. 6)")
    ap.add_argument('--reco', default='output/paper_plots/reco.csv')
    ap.add_argument('--min-rows', type=int, default=6)
    ap.add_argument('--zen-bin-deg', type=float, default=6.0)
    ap.add_argument('--out', default='output/paper_plots')
    args = ap.parse_args()

    df = pd.read_csv(args.reco)
    df = df[df['zenith_deg'].notna()]
    singles = df[df['n_tracks_3d'] == 1]
    clean = singles[singles['n_rows'] >= args.min_rows]
    z = clean['zenith_deg'].to_numpy()
    z = z[np.isfinite(z)]
    a = clean['azimuth_deg'].to_numpy()
    a = a[np.isfinite(a)]

    os.makedirs(args.out, exist_ok=True)
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(13.2, 4.3))

    # --- (a) zenith ---
    nzen = int(round(90.0 / args.zen_bin_deg))
    counts, edges, _ = axa.hist(z, bins=nzen, range=(0, 90), color=COLORS['data'],
                                edgecolor='black', alpha=0.85,
                                label=f'Data (N={len(z)})')
    centers = 0.5 * (edges[:-1] + edges[1:])
    theta = np.radians(centers)
    model = np.cos(theta) ** 2 * np.sin(theta)
    mask = counts > 0
    if model[mask].sum() > 0:
        model = model / model[mask].sum() * counts[mask].sum()
    axa.plot(centers, model, '--', color=COLORS['model'], lw=2,
             label=r'$\cos^2\theta\,\sin\theta$' + '\n(no acceptance corr.)')
    axa.set_xlabel(r'Zenith angle $\theta$ (deg)')
    axa.set_ylabel('Events')
    axa.legend(fontsize=8.5)
    axa.grid(True, ls='--', alpha=0.4)
    axa.set_title('(a)', loc='left', fontsize=11)

    # --- (b) azimuth and the harmonic fit ---
    chi2_clean, az_counts, u = chi2_uniform(a)
    harm = azimuth_harmonics(az_counts)
    axb.hist(a, bins=AZ_BINS, range=(0, 360), color=COLORS['accent'],
             edgecolor='black', alpha=0.85, label=f'Data (N={len(a)})')
    axb.axhline(u, color=COLORS['model'], ls='--', lw=1.8, label='Isotropic')
    phi_d = np.linspace(0, 360, 361)
    pr = np.radians(phi_d)
    c = harm['coef']
    fitc = (c[0] + c[1] * np.cos(pr) + c[2] * np.sin(pr)
            + c[3] * np.cos(2 * pr) + c[4] * np.sin(2 * pr))
    axb.plot(phi_d, fitc, '-', color=COLORS['x'], lw=1.8,
             label='harmonic fit\n' r'$a_0{+}A_1\cos(\varphi{-}\varphi_1)'
                   r'{+}A_2\cos2(\varphi{-}\varphi_2)$')
    axb.set_xlabel(r'Azimuth angle $\varphi$ (deg)')
    axb.set_ylabel('Events')
    axb.set_xlim(0, 360)
    axb.legend(fontsize=7.6, loc='upper left')
    axb.grid(True, ls='--', alpha=0.4)
    axb.set_title('(b)', loc='left', fontsize=11)

    # --- (c) chi2/ndf progression ---
    groups = [('$n_\\mathrm{rows}{\\leq}4$\n(2+2)', singles['n_rows'] <= 4),
              ('$n_\\mathrm{rows}{=}5$\n(3+2)', singles['n_rows'] == 5),
              ('$n_\\mathrm{rows}{\\geq}6$\n(3+3)',
               singles['n_rows'] >= args.min_rows)]
    xs, vals, ns = [], [], []
    for i, (_, m) in enumerate(groups):
        aa = singles.loc[m, 'azimuth_deg'].to_numpy()
        aa = aa[np.isfinite(aa)]
        cc, _, _ = chi2_uniform(aa)
        xs.append(i + 1)
        vals.append(cc)
        ns.append(len(aa))
    axc.plot(xs, vals, 'o-', color=COLORS['x'], ms=9, lw=1.8, zorder=3,
             label='this pipeline,\nby track class')
    axc.plot([0], [RAW_PIPELINE_CHI2], 's', ms=9, mfc='none', mew=1.6,
             color=COLORS['y'], zorder=3,
             label='raw pipeline\n(development stage)')
    for x, v, n in zip(xs, vals, ns):
        axc.annotate(f'{v:.0f}\nN={n}', (x, v), textcoords='offset points',
                     xytext=(8, 2), fontsize=8, color='#4a5a66')
    axc.annotate(f'{RAW_PIPELINE_CHI2:.0f}', (0, RAW_PIPELINE_CHI2),
                 textcoords='offset points', xytext=(8, -3), fontsize=8,
                 color='#4a5a66')
    axc.axhline(1.0, color=COLORS['span'], ls=':', lw=1.5)
    axc.text(3.3, 1.18, 'statistical limit', ha='right', fontsize=8,
             color=COLORS['span'])
    axc.set_yscale('log')
    axc.set_xlim(-0.6, 3.5)
    axc.set_ylim(0.5, 4000)
    axc.set_xticks([0] + xs, ['raw'] + [g[0] for g in groups], fontsize=8.5)
    axc.set_ylabel(r'$\chi^2/\mathrm{ndf}$ of azimuth vs uniform')
    axc.grid(True, ls='--', alpha=0.4, which='both')
    axc.legend(fontsize=7.8, loc='upper right')
    axc.set_title('(c)', loc='left', fontsize=11)

    fig.tight_layout()
    base = os.path.join(args.out, 'fig_angular_summary')
    save_figure(fig, base, caption=(
        "Angular reconstruction on the clean sample of single penetrating tracks "
        "(three active planes in each projection). (a) Zenith-angle distribution in "
        "6-degree bins, chosen to average over the residual quantization of the "
        "reconstructed slopes; the dashed curve is the cos^2(theta)*sin(theta) shape "
        "of the atmospheric muon flux, normalized to the populated bins and shown "
        "without acceptance correction. (b) Azimuthal distribution with the isotropic "
        "expectation (dashed) and a first-plus-second-harmonic fit (solid) that "
        "quantifies the residual instrumental modulation. (c) chi2/ndf of the azimuth "
        "distribution against uniformity for single tracks by track class, from "
        "two-point (quantized) to fully penetrating 3+3 tracks; the open marker shows "
        "the level of the uncorrected development-stage pipeline, and the dotted line "
        "the statistical limit."))
    plt.close(fig)
    print(f"Written: {base}.png / .pdf")

    # ================= PASTE-BLOCK =================
    P = print
    P("")
    P("===== PASTE-BLOCK ANGULAR-SUMMARY v1 START =====")
    P(f"reco={args.reco} min_rows={args.min_rows} zen_bin={args.zen_bin_deg:.0f}deg")
    P(f"clean N={len(z)}  zenith median={np.median(z):.1f} mean={z.mean():.1f} "
      f"max={z.max():.1f}")
    P(f"azimuth chi2/ndf={chi2_clean:.1f}  a0={harm['a0']:.1f}  "
      f"A1/a0={harm['A1']/harm['a0']:.1%}@{harm['phi1']:.0f}deg  "
      f"A2/a0={harm['A2']/harm['a0']:.1%}@axis{harm['phi2']:.0f}deg")
    P("progression chi2/ndf: " + "  ".join(
        f"[{lab.splitlines()[0]}]={v:.1f}(N={n})"
        for (lab, _), v, n in zip(groups, vals, ns))
      + f"  [raw]={RAW_PIPELINE_CHI2:.0f}(hist)")
    P("===== PASTE-BLOCK ANGULAR-SUMMARY v1 END =====")


if __name__ == '__main__':
    main()