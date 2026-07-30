"""
fig_pipeline.py - block diagram of the reconstruction pipeline (publication figure).

Draws the diagram entirely in matplotlib, as vector output, without graphviz and
without reading any data: the main sequence of stages plus two side branches,
the good-run selection driven by the file-level diagnostics, and the
track-based alignment loop. The stages follow Section 3 of the paper.

Output: output/paper_plots/fig_pipeline.(png,pdf)

Run:
    python analysis/fig_pipeline.py
"""
import os
import sys
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.plot_style import apply_style, COLORS, save_figure
apply_style()

DARK = COLORS['x']       # #2c3e50
SIDE = COLORS['y']       # #e67e22
GREY = '#7f8c8d'
FACE = '#f4f6f8'


def add_box(ax, xc, yc, w, h, title, sub=None, ec=DARK, fc=FACE, lw=1.4,
            tfs=10.0, sfs=8.3):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                boxstyle="round,pad=0.25,rounding_size=1.2",
                                linewidth=lw, edgecolor=ec, facecolor=fc,
                                mutation_aspect=1.0, zorder=2))
    if sub:
        ax.text(xc, yc + h * 0.16, title, ha='center', va='center',
                fontsize=tfs, fontweight='bold', color=DARK, zorder=3)
        ax.text(xc, yc - h * 0.21, sub, ha='center', va='center',
                fontsize=sfs, color='#4a5a66', zorder=3)
    else:
        ax.text(xc, yc, title, ha='center', va='center',
                fontsize=tfs, fontweight='bold', color=DARK, zorder=3)


def arrow(ax, p0, p1, ec=DARK, lw=1.5, style='-|>', ls='solid', rad=0.0,
          ms=13, zorder=1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, edgecolor=ec, facecolor=ec,
                                 linestyle=ls, zorder=zorder,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser(description="Pipeline block diagram (Fig. 2)")
    ap.add_argument('--out', default='output/paper_plots')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 9.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.grid(False)

    XC, W, H = 34.0, 44.0, 7.4       # main sequence
    ys = [95.0, 84.4, 73.8, 63.2, 52.6, 42.0, 31.4, 20.8]
    stages = [
        ("Raw .dat files",
         "2021 & 2023 seasons · ~2.9×10$^5$ event records"),
        ("Integrity audit & good-run selection",
         "bad-run events flagged in catalogue · IDs preserved"),
        ("Section assembly",
         "overlap sum by channel index · pad to plane width"),
        ("Cluster finding + event vetoes",
         "saturation · uniformity · empty · flash"),
        ("Road search per projection",
         "hadron block only · seeds include deepest planes"),
        ("X–Y matching",
         "energy-rank pairing · geometric fallback"),
        ("Channel→mm calibration + alignment",
         "per-plane centring (width N/2) · plane offsets"),
        ("3D weighted least-squares fit",
         "γ-planes excluded · multiple-scattering weights"),
    ]
    for (t, s), y in zip(stages, ys):
        add_box(ax, XC, y, W, H, t, s)
    # final block: the results
    y_out = 9.4
    add_box(ax, XC, y_out, W + 8, 8.2,
            "Detector-response observables",
            "longitudinal · transverse · zenith / azimuth · RA, Dec",
            ec=DARK, fc='#eef3f0')
    # arrows of the main sequence
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(ax, (XC, y0 - H / 2), (XC, y1 + H / 2))
    arrow(ax, (XC, ys[-1] - H / 2), (XC, y_out + 8.2 / 2))

    # --- right branch 1: file-level diagnostics (good-run selection) ---
    GX, GW, GH = 81.5, 33.0, 15.0
    GY = 84.4
    add_box(ax, GX, GY, GW, GH,
            "File-level diagnostics",
            "camera presence per run\nduplicate raw files\nwrong-season files\n→ exclusion list",
            ec=SIDE, fc='white', sfs=8.3)
    # raw data -> diagnostics (dashed), diagnostics -> selection (solid)
    arrow(ax, (XC + W / 2, 95.0), (GX - GW / 2, GY + GH * 0.30),
          ec=GREY, lw=1.2, ls=(0, (4, 3)), rad=-0.25)
    arrow(ax, (GX - GW / 2, GY - GH * 0.10), (XC + W / 2, 84.4),
          ec=SIDE, lw=1.6)

    # --- right branch 2: the alignment loop ---
    LX, LW, LH = 81.5, 33.0, 15.0
    LY = 26.1
    add_box(ax, LX, LY, LW, LH,
            "Track-based alignment loop",
            "mean transverse residual\nper plane on 3+3 tracks\n→ offset update (×2 iter.)",
            ec=SIDE, fc='white', sfs=8.3)
    # fit -> loop (residuals), loop -> calibration (updated offsets, dashed)
    arrow(ax, (XC + W / 2, 20.8), (LX - LW / 2, LY - LH * 0.28),
          ec=SIDE, lw=1.6, rad=0.22)
    arrow(ax, (LX - LW / 2, LY + LH * 0.28), (XC + W / 2, 31.4),
          ec=SIDE, lw=1.6, ls=(0, (4, 3)), rad=0.22)
    ax.text(LX, LY - LH / 2 - 1.8, "residuals ↑ · updated offsets ↓",
            ha='center', va='top', fontsize=7.6, color=GREY)

    # --- data tags on the left ---
    ax.annotate("HDF5 event catalogue\n(all events + status flags)",
                xy=(XC - W / 2, 63.2), xytext=(1.0, 63.2),
                ha='left', va='center', fontsize=7.8, color=GREY,
                fontstyle='italic',
                arrowprops=dict(arrowstyle='-', color=GREY, lw=0.9))
    ax.annotate("reco.csv\n(per-track observables)",
                xy=(XC - W / 2, 20.8), xytext=(1.0, 20.8),
                ha='left', va='center', fontsize=7.8, color=GREY,
                fontstyle='italic',
                arrowprops=dict(arrowstyle='-', color=GREY, lw=0.9))

    fig.tight_layout()
    base = os.path.join(args.out, 'fig_pipeline')
    save_figure(fig, base, caption=(
        "Structure of the reconstruction pipeline. The main flow (left) takes the raw "
        "per-section data to three-dimensional tracks and the detector-response "
        "observables of Section 4. Two data-driven procedures branch from it: the "
        "good-run selection, in which file-level diagnostics (per-run camera presence, "
        "duplicate and wrong-season raw files) produce an exclusion list applied while "
        "the event catalogue is built, with excluded events flagged rather than "
        "removed; and the track-based alignment loop, in which the mean transverse "
        "residual of clean penetrating tracks in each plane updates that plane's "
        "offset, converging in two iterations."))
    plt.close(fig)
    print(f"Written: {base}.png / .pdf")


if __name__ == '__main__':
    main()