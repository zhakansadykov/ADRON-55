"""
fig_angular.py - angular distributions on the clean sample (publication figures).

Works from reco.csv, produced by reconstruct_all.py. Clean sample: single tracks
(n_tracks_3d == 1) with n_rows >= 6, that is three points in each projection.

The report printed at the end carries every number needed to check the text:
  * zenith: the mode of the distribution and the median of the truncated
    cos^2(theta)*sin(theta) model on [0, theta_max];
  * azimuth: a harmonic decomposition of the residual modulation,
    N(phi) ~ a0 + A1*cos(phi - phi1) + A2*cos(2*(phi - phi2)), which quantifies
    the non-uniformity. The second harmonic, of period 180 degrees, is expected
    from the different lever arms of the two projections;
  * single tracks split by n_rows (>=6, ==5, <=4) with chi2/ndf, the material
    for the progression panel of the summary figure.

Output: output/paper_plots/fig_zenith_final.(png,pdf),
        fig_azimuth_final.(png,pdf), fig_skymap_final.(png,pdf),
        angular_sample.csv

Run:
    python analysis/fig_angular.py --reco output/paper_plots/reco.csv
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.plot_style import apply_style, COLORS, save_figure
apply_style()

AZ_BINS = 36
ZEN_BINS = 30


def load_clean(reco_path, min_rows):
    df = pd.read_csv(reco_path)
    df = df[df['zenith_deg'].notna()]
    if 'n_rows' not in df.columns:
        raise SystemExit("reco.csv has no n_rows column; rerun reconstruct_all.py.")
    clean = df[(df['n_tracks_3d'] == 1) & (df['n_rows'] >= min_rows)].copy()
    return df, clean


def chi2_uniform(a, nbins=AZ_BINS, lo=0.0, hi=360.0):
    """chi2/ndf of the histogram against uniformity. Returns (chi2_ndf, counts, u)."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    counts, _ = np.histogram(a, bins=nbins, range=(lo, hi))
    u = len(a) / nbins
    if u <= 0:
        return float('nan'), counts, u
    chi2 = float(np.sum((counts - u) ** 2 / u))
    return chi2 / (nbins - 1), counts, u


def azimuth_harmonics(counts, nbins=AZ_BINS):
    """Least-squares fit of N(phi) = a0 + c1 cos + s1 sin + c2 cos2 + s2 sin2
    over the bin centres. Returns a0, A1, phi1_deg (maximum of the first
    harmonic), A2 and phi2_deg (axis of the second harmonic, period 180 deg)."""
    centers = (np.arange(nbins) + 0.5) * (360.0 / nbins)
    phi = np.radians(centers)
    M = np.column_stack([np.ones_like(phi),
                         np.cos(phi), np.sin(phi),
                         np.cos(2 * phi), np.sin(2 * phi)])
    coef, *_ = np.linalg.lstsq(M, counts.astype(float), rcond=None)
    a0, c1, s1, c2, s2 = coef
    A1 = float(np.hypot(c1, s1))
    A2 = float(np.hypot(c2, s2))
    phi1 = float(np.degrees(np.arctan2(s1, c1)) % 360.0)
    phi2 = float((np.degrees(np.arctan2(s2, c2)) / 2.0) % 180.0)
    return {'a0': float(a0), 'A1': A1, 'phi1': phi1, 'A2': A2, 'phi2': phi2}


def truncated_model_median(zmax_deg):
    """Median of the pdf proportional to cos^2(theta)*sin(theta), truncated to [0, zmax]."""
    t = np.linspace(0.0, np.radians(zmax_deg), 4001)
    pdf = np.cos(t) ** 2 * np.sin(t)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return float(np.degrees(np.interp(0.5, cdf, t)))


def ascii_hist(values, nbins, lo, hi, label, width=46):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    counts, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    mx = counts.max() if counts.max() > 0 else 1
    print(f"[{label}]  N={len(v)}")
    for i in range(nbins):
        bar = '#' * int(round(width * counts[i] / mx))
        print(f"  {edges[i]:6.1f}-{edges[i+1]:6.1f} | {bar:<{width}} {counts[i]}")
    return counts, edges


def plot_zenith(clean, out):
    z = clean['zenith_deg'].to_numpy()
    z = z[np.isfinite(z)]
    fig, ax = plt.subplots(figsize=(7, 5))
    counts, edges, _ = ax.hist(z, bins=ZEN_BINS, range=(0, 90), color=COLORS['data'],
                               edgecolor='black', alpha=0.85, label=f'Data (N={len(z)})')
    centers = 0.5 * (edges[:-1] + edges[1:])
    theta = np.radians(centers)
    expected = np.cos(theta) ** 2 * np.sin(theta)
    mask = counts > 0
    if expected[mask].sum() > 0:
        expected = expected / expected[mask].sum() * counts[mask].sum()
    ax.plot(centers, expected, 'r--', lw=2,
            label=r'$\cos^2\theta\,\sin\theta$ (no acceptance corr.)')
    ax.set_xlabel('Zenith angle θ (deg)')
    ax.set_ylabel('Events')
    ax.set_title('Zenith angle distribution (single penetrating tracks, 3+3)')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    save_figure(fig, os.path.join(out, 'fig_zenith_final'), caption=(
        "Zenith-angle distribution of single penetrating tracks reconstructed with the "
        "full set of hadron-block planes (3+3 points). The dashed curve is the "
        "cos^2(theta)*sin(theta) shape expected for atmospheric muons, normalized to "
        "the data over the populated range. The distribution is shown WITHOUT "
        "acceptance correction; the geometric acceptance falls with inclination and "
        "vanishes above ~60 deg. A full acceptance correction requires the Geant4 "
        "simulation (future work)."))
    plt.close(fig)
    return counts, centers


def plot_azimuth(clean, out):
    a = clean['azimuth_deg'].to_numpy()
    a = a[np.isfinite(a)]
    chi2_ndf, counts, u = chi2_uniform(a)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(a, bins=AZ_BINS, range=(0, 360), color=COLORS['accent'],
            edgecolor='black', alpha=0.85)
    ax.axhline(u, color='r', ls='--', lw=2, label='Isotropic')
    ax.set_xlabel('Azimuth angle φ (deg)')
    ax.set_ylabel('Events')
    ax.set_title(f'Azimuth distribution (3+3 tracks, N={len(a)}, χ²/ndf={chi2_ndf:.1f})')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    save_figure(fig, os.path.join(out, 'fig_azimuth_final'), caption=(
        "Azimuthal distribution of single penetrating tracks (3+3 points). The dashed "
        "line marks the isotropic expectation. The chi2/ndf relative to uniform is "
        "given in the title. The residual modulation is instrumental (differing lever "
        "arms of the X and Y projections); local topography is excluded because the "
        "sample is restricted to zenith angles <60 deg, well above the terrain "
        "horizon (~10-12 deg)."))
    plt.close(fig)
    return chi2_ndf, counts


def plot_skymap(clean, out):
    m = clean['ra_deg'].notna() & clean['dec_deg'].notna()
    ra = clean.loc[m, 'ra_deg'].to_numpy()
    dec = clean.loc[m, 'dec_deg'].to_numpy()
    if len(ra) == 0:
        return 0, (np.nan, np.nan), (np.nan, np.nan)
    fig, ax = plt.subplots(figsize=(9, 5))
    hb = ax.hexbin(ra, dec, gridsize=25, cmap='viridis', mincnt=1)
    fig.colorbar(hb, ax=ax, label='Events')
    ax.set_xlabel('Right Ascension (deg)')
    ax.set_ylabel('Declination (deg)')
    ax.set_xlim(0, 360)
    ax.set_title(f'Sky map, equatorial (single penetrating tracks, N={len(ra)})')
    fig.tight_layout()
    save_figure(fig, os.path.join(out, 'fig_skymap_final'), caption=(
        "Equatorial sky map (right ascension vs declination) of single penetrating "
        "tracks (3+3 points), obtained by converting reconstructed local "
        "altitude/azimuth with event timestamps. The coverage reflects the detector's "
        "field of view and exposure over the 2021 and 2023 observing seasons rather "
        "than any intrinsic sky anisotropy."))
    plt.close(fig)
    return len(ra), (float(ra.min()), float(ra.max())), (float(dec.min()), float(dec.max()))


def main():
    ap = argparse.ArgumentParser(description="Angular figures on the clean sample")
    ap.add_argument('--reco', default='output/paper_plots/reco.csv')
    ap.add_argument('--min-rows', type=int, default=6)
    ap.add_argument('--out', default='output/paper_plots')
    args = ap.parse_args()

    df, clean = load_clean(args.reco, args.min_rows)
    os.makedirs(args.out, exist_ok=True)

    n_all = len(df)
    singles = df[df['n_tracks_3d'] == 1]
    n_single = len(singles)
    print(f"Total 3D tracks in reco: {n_all}")
    print(f"Single-track events: {n_single}")
    print(f"Clean sample (single track, n_rows>={args.min_rows}): {len(clean)}")

    z_counts, z_centers = plot_zenith(clean, args.out)
    chi2_clean, az_counts = plot_azimuth(clean, args.out)
    n_sky, ra_rng, dec_rng = plot_skymap(clean, args.out)
    clean.to_csv(os.path.join(args.out, 'angular_sample.csv'), index=False)

    z = clean['zenith_deg'].to_numpy()
    z = z[np.isfinite(z)]
    zen_mode = float(z_centers[int(np.argmax(z_counts))]) if z_counts.sum() else float('nan')
    model_med = truncated_model_median(float(z.max())) if len(z) else float('nan')
    harm = azimuth_harmonics(az_counts)

    print(f"\nAzimuth chi2/ndf against uniformity: {chi2_clean:.1f}")
    print(f"Written to {args.out}/: fig_zenith_final, fig_azimuth_final, "
          f"fig_skymap_final (png+pdf) and angular_sample.csv")

    # ================= PASTE-BLOCK =================
    P = print
    P("")
    P("===== PASTE-BLOCK ANGULAR v2 START =====")
    P(f"reco={args.reco} min_rows={args.min_rows}")
    P(f"tracks_total={n_all} singles={n_single} clean={len(clean)}")
    P(f"zenith: median={np.median(z):.1f} mean={z.mean():.1f} max={z.max():.1f} "
      f"mode_bin_center={zen_mode:.1f} (bin={90/ZEN_BINS:.0f}deg)")
    P(f"zenith model cos2*sin truncated to [0,{z.max():.1f}]: "
      f"median_expected={model_med:.1f}")
    ascii_hist(z, ZEN_BINS, 0, 90, "ZENITH clean (deg)")
    a = clean['azimuth_deg'].to_numpy()
    a = a[np.isfinite(a)]
    P(f"azimuth: chi2/ndf={chi2_clean:.1f}  a0={harm['a0']:.1f}/bin")
    P(f"  A1/a0={harm['A1']/harm['a0']:.1%} at phi1={harm['phi1']:.0f}deg ; "
      f"A2/a0={harm['A2']/harm['a0']:.1%} axis phi2={harm['phi2']:.0f}deg "
      f"(period 180)")
    ascii_hist(a, AZ_BINS, 0, 360, "AZIMUTH clean (deg)")
    P("singles by n_rows (chi2/ndf to uniform, 36 bins):")
    for label, mask in [(">=6", singles['n_rows'] >= 6),
                        ("==5", singles['n_rows'] == 5),
                        ("<=4", singles['n_rows'] <= 4)]:
        aa = singles.loc[mask, 'azimuth_deg'].to_numpy()
        aa = aa[np.isfinite(aa)]
        if len(aa) == 0:
            P(f"  n_rows{label}: N=0")
            continue
        c, _, _ = chi2_uniform(aa)
        P(f"  n_rows{label}: N={len(aa):>6}  chi2/ndf={c:.1f}")
    P(f"skymap: N={n_sky}  RA[{ra_rng[0]:.1f},{ra_rng[1]:.1f}]  "
      f"Dec[{dec_rng[0]:.1f},{dec_rng[1]:.1f}]")
    P("===== PASTE-BLOCK ANGULAR v2 END =====")


if __name__ == '__main__':
    main()