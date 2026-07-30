"""
paper_plots.py - physics figures for the paper, from reco.csv.

The diagnostic distributions (zenith, azimuth, chi2) are produced first, to
establish that the angular reconstruction is unbiased, and only then the sky
map. The azimuth of cosmic rays must be close to isotropic; a strong
concentration signals a systematic that has to be removed before a sky map means
anything.

Produces:
  fig_efficiency.txt   summary of the 3D reconstruction efficiency
  fig_zenith.png       zenith distribution with the cos^2(theta)*sin(theta) shape
  fig_azimuth.png      azimuth distribution with the isotropic expectation
  fig_chi2.png         chi2_3d distribution, logarithmic
  fig_skymap.png       equatorial sky map, to be trusted only after the azimuth
                       check above

Run:
    python analysis/paper_plots.py --reco output/paper_plots/reco.csv
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


def load_reco(path):
    df = pd.read_csv(path)
    df['has_track'] = df['zenith_deg'].notna()
    return df


def report_efficiency(df, out_dir):
    lines = []
    n = len(df)
    ntr = int(df['has_track'].sum())
    lines.append(f"Events in reco: {n}")
    lines.append(f"With a 3D track: {ntr} ({ntr/n:.1%})")
    lines.append(f"Without a track: {n-ntr} ({(n-ntr)/n:.1%})")
    lines.append("\nEfficiency by particle label:")
    for pt, g in df.groupby('particle_type'):
        m = len(g); k = int(g['has_track'].sum())
        lines.append(f"  {pt:<20} {k:>6}/{m:<6} ({k/m:.1%})")
    text = "\n".join(lines)
    with open(os.path.join(out_dir, 'fig_efficiency.txt'), 'w', encoding='utf-8') as f:
        f.write(text + "\n")
    print(text)
    return text


def plot_zenith(df, out_dir):
    z = df.loc[df['has_track'], 'zenith_deg'].to_numpy()
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    counts, edges, _ = ax.hist(z, bins=45, range=(0, 90), color='steelblue',
                               edgecolor='black', alpha=0.8, label='Data')
    # Shape reference dN/dtheta ~ cos^2(theta)*sin(theta), without acceptance correction
    centers = 0.5 * (edges[:-1] + edges[1:])
    theta = np.radians(centers)
    expected = np.cos(theta) ** 2 * np.sin(theta)
    if expected.sum() > 0:
        expected = expected / expected.sum() * counts.sum()
        ax.plot(centers, expected, 'r--', lw=2, label=r'$\cos^2\theta\,\sin\theta$ (no acceptance correction)')
    ax.set_xlabel('Zenith angle θ (deg)')
    ax.set_ylabel('Events')
    ax.set_title('Zenith angle distribution')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_zenith.png'), dpi=200)
    plt.close(fig)


def plot_azimuth(df, out_dir):
    a = df.loc[df['has_track'], 'azimuth_deg'].to_numpy()
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    counts, edges, _ = ax.hist(a, bins=36, range=(0, 360), color='seagreen',
                               edgecolor='black', alpha=0.8, label='Data')
    uniform = len(a) / 36
    ax.axhline(uniform, color='r', ls='--', lw=2, label='Isotropic expectation')
    ax.set_xlabel('Azimuth angle φ (deg)')
    ax.set_ylabel('Events')
    ax.set_title('Azimuth distribution: expected to be flat')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_azimuth.png'), dpi=200)
    plt.close(fig)
    # Numerical measure of the non-uniformity
    chi2_uniform = float(np.sum((counts - uniform) ** 2 / uniform)) if uniform > 0 else float('nan')
    print(f"\nAzimuth: chi2 against uniformity = {chi2_uniform:.1f} (ndf={len(counts)-1}); "
          f"bin-to-bin sigma = {counts.std():.1f}, mean = {counts.mean():.1f}")


def plot_azimuth_by_ntracks(df, out_dir):
    """Split the azimuth into single-track and multi-track events.
    If the single tracks are displaced too, the problem lies in the geometry or
    the 2D reconstruction, not in the X-to-Y matching."""
    d = df[df['has_track']].copy()
    single = d[d['n_tracks_3d'] == 1]['azimuth_deg'].to_numpy()
    multi = d[d['n_tracks_3d'] >= 2]['azimuth_deg'].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, a, name in [(axes[0], single, 'Single track (n=1)'),
                        (axes[1], multi, 'Multi-track (n>=2)')]:
        a = a[np.isfinite(a)]
        if len(a) == 0:
            ax.set_title(f'{name}: no data')
            continue
        counts, _, _ = ax.hist(a, bins=36, range=(0, 360), color='slateblue',
                               edgecolor='black', alpha=0.8)
        u = len(a) / 36
        ax.axhline(u, color='r', ls='--', lw=2)
        chi2 = float(np.sum((counts - u) ** 2 / u)) if u > 0 else float('nan')
        ax.set_title(f'{name}\nN={len(a)}, chi2/ndf={chi2/35:.1f}')
        ax.set_xlabel('Azimuth phi (deg)')
        ax.grid(True, ls='--', alpha=0.4)
        print(f"  {name}: N={len(a)}, chi2/ndf against uniformity = {chi2/35:.1f}")
    axes[0].set_ylabel('Events')
    fig.suptitle('Azimuth by track multiplicity (locating the systematic)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_azimuth_by_ntracks.png'), dpi=200)
    plt.close(fig)


def plot_chi2(df, out_dir):
    c = df.loc[df['has_track'], 'chi2_3d'].to_numpy()
    c = c[np.isfinite(c) & (c > 0)]
    if len(c) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(c, bins=np.logspace(np.log10(max(c.min(), 1e-6)), np.log10(c.max()), 40),
            color='indianred', edgecolor='black', alpha=0.8)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\chi^2/\mathrm{ndf}$ (3D)')
    ax.set_ylabel('Tracks')
    ax.set_title('3D fit quality')
    ax.grid(True, ls='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_chi2.png'), dpi=200)
    plt.close(fig)


def plot_skymap(df, out_dir):
    m = df['has_track'] & df['ra_deg'].notna()
    ra = df.loc[m, 'ra_deg'].to_numpy()
    dec = df.loc[m, 'dec_deg'].to_numpy()
    if len(ra) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    hb = ax.hexbin(ra, dec, gridsize=40, cmap='viridis', mincnt=1)
    fig.colorbar(hb, ax=ax, label='Events')
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_xlim(0, 360)
    ax.set_title('Sky map (RA/Dec): read only after the azimuth check')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_skymap.png'), dpi=200)
    plt.close(fig)


def print_ascii_hist(values, nbins, lo, hi, label, width=50):
    """Print a compact ASCII histogram, as a terminal substitute for the PNG."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        print(f"\n[{label}] no data")
        return
    counts, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    mx = counts.max() if counts.max() > 0 else 1
    print(f"\n[{label}]  N={len(v)}, max_bin={mx}")
    for i in range(nbins):
        c = edges[i]
        bar = '#' * int(round(width * counts[i] / mx))
        print(f"  {c:6.0f}-{edges[i+1]:6.0f} | {bar:<{width}} {counts[i]}")


def print_text_diagnostics(df):
    """Text summary of the distributions."""
    d = df[df['has_track']]
    print("\n" + "#" * 60)
    print("TEXT DISTRIBUTIONS (terminal substitute for the PNG figures)")
    print("#" * 60)
    print_ascii_hist(d['zenith_deg'], 18, 0, 90, "ZENITH (all tracks)")
    print_ascii_hist(d[d['n_tracks_3d'] == 1]['azimuth_deg'], 24, 0, 360,
                     "AZIMUTH, single tracks (n=1)")
    print_ascii_hist(d[d['n_tracks_3d'] >= 2]['azimuth_deg'], 24, 0, 360,
                     "AZIMUTH, multi-track (n>=2)")

    # Single tracks split by plane count (n_rows>=6 is 3+3, a continuous slope)
    if 'n_rows' in d.columns:
        print("\n--- Azimuth of single tracks by plane count (quantized vs continuous) ---")
        singles = d[d['n_tracks_3d'] == 1]
        for label, mask in [("n_rows>=6 (3+3, cleanest)", singles['n_rows'] >= 6),
                            ("n_rows==5 (3+2)", singles['n_rows'] == 5),
                            ("n_rows<=4 (2+2, quantized)", singles['n_rows'] <= 4)]:
            a = singles.loc[mask, 'azimuth_deg'].to_numpy()
            a = a[np.isfinite(a)]
            if len(a) == 0:
                print(f"  {label}: no data")
                continue
            counts, _ = np.histogram(a, bins=36, range=(0, 360))
            u = len(a) / 36
            chi2 = float(np.sum((counts - u) ** 2 / u)) if u > 0 else float('nan')
            print(f"  {label}: N={len(a)}, chi2/ndf={chi2/35:.1f}")
        # ASCII histogram of the cleanest group
        best = singles.loc[singles['n_rows'] >= 6, 'azimuth_deg']
        print_ascii_hist(best, 24, 0, 360, "AZIMUTH, single tracks with n_rows>=6 (3+3)")


def main():
    ap = argparse.ArgumentParser(description="Physics figures from reco.csv")
    ap.add_argument('--reco', default='output/paper_plots/reco.csv')
    ap.add_argument('--out', default='output/paper_plots')
    args = ap.parse_args()

    if not os.path.exists(args.reco):
        raise SystemExit(f"{args.reco} not found. Run first: python analysis/reconstruct_all.py ...")
    os.makedirs(args.out, exist_ok=True)

    df = load_reco(args.reco)
    print("=" * 55)
    report_efficiency(df, args.out)
    print("=" * 55)
    plot_zenith(df, args.out)
    plot_azimuth(df, args.out)
    plot_azimuth_by_ntracks(df, args.out)
    plot_chi2(df, args.out)
    plot_skymap(df, args.out)
    print_text_diagnostics(df)
    print(f"\nFigures written to {args.out}/ (fig_zenith/azimuth/chi2/skymap.png)")
    print("Check fig_azimuth.png first: if it is not flat, there is an angular systematic.")


if __name__ == '__main__':
    main()