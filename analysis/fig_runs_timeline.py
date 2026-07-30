"""
fig_runs_timeline.py - data quality over time (publication figure).

From run_quality.csv, produced by analysis/diag_runs.py, plots for every run:
  top panel:    the minimum presence among the Y readout sections; a dead
                section drives it to zero;
  bottom panel: the fraction of events with fewer than three active Y planes.

The horizontal axis is the run date, taken from the first six digits of the
filename (YYMMDD). The 2021 and 2023 seasons are drawn as two columns of width
proportional to their duration. Runs listed in data.exclude_files are marked
with open orange markers, which makes the dead-camera block of December 2021
visible at a glance.

Output: output/paper_plots/fig_runs_timeline.(png,pdf)

Run (generate the CSV first if it does not exist):
    python analysis/diag_runs.py --config config/settings.yaml
    python analysis/fig_runs_timeline.py
"""
import os
import re
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.plot_style import apply_style, COLORS, save_figure
apply_style()

DARK = COLORS['x']
SIDE = COLORS['y']


def file_date(name):
    m = re.search(r'(\d{6})', str(name))
    if not m:
        return None
    s = m.group(1)
    try:
        return datetime.strptime('20' + s, '%Y%m%d')
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description="Run quality over time (Fig. 3)")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--run-quality', default='output/paper_plots/run_quality.csv')
    ap.add_argument('--out', default='output/paper_plots')
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    exclude_set = set(cfg.get('data', {}).get('exclude_files', []) or [])
    dataset_years = set(int(y) for y in (cfg.get('data', {}).get('years') or []))

    if not os.path.exists(args.run_quality):
        raise SystemExit(f"{args.run_quality} not found; run first: "
                         f"python analysis/diag_runs.py --config {args.config}")
    df = pd.read_csv(args.run_quality)
    pres_cols = [c for c in df.columns if c.startswith('present_')]
    if not pres_cols:
        raise SystemExit("run_quality.csv has no present_* columns; rerun diag_runs.py.")

    df['date'] = df['source_file'].map(file_date)
    n_nodate = int(df['date'].isna().sum())
    df = df.dropna(subset=['date']).copy()
    df['min_presence'] = df[pres_cols].min(axis=1)
    df['excluded'] = df['source_file'].isin(exclude_set)
    df['season'] = df['date'].map(lambda d: d.year)

    dropped = pd.DataFrame(columns=df.columns)
    if dataset_years:
        outside = ~df['season'].isin(dataset_years)
        dropped = df[outside].copy()
        df = df[~outside].copy()

    seasons = sorted(df['season'].unique())
    spans = []
    for s in seasons:
        d = df[df['season'] == s]['date']
        spans.append(max((d.max() - d.min()).days, 30))
    ratios = [s / max(spans) for s in spans]

    fig, axes = plt.subplots(2, len(seasons), figsize=(10.5, 6.4), sharey='row',
                             gridspec_kw={'width_ratios': ratios, 'wspace': 0.06,
                                          'hspace': 0.12})
    if len(seasons) == 1:
        axes = axes.reshape(2, 1)

    for j, s in enumerate(seasons):
        sub = df[df['season'] == s]
        good = sub[~sub['excluded']]
        bad = sub[sub['excluded']]
        for i, col in enumerate(['min_presence', 'frac_lt3_y_rows']):
            ax = axes[i, j]
            ax.plot(good['date'], good[col], 'o', ms=4.5, color=DARK,
                    label='good run' if (i == 0 and j == 0) else None)
            ax.plot(bad['date'], bad[col], 'o', ms=5.5, mfc='none', mew=1.4,
                    color=SIDE,
                    label='excluded run' if (i == 0 and j == 0) else None)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, ls='--', alpha=0.4)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2 if j == 0 else 1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            if i == 0:
                ax.set_title(str(s), fontsize=11)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(f'Run date ({s})')
            if j > 0:
                ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel('Min. presence of\nY-readout sections')
    axes[1, 0].set_ylabel('Fraction of events\nwith <3 active Y planes')
    axes[0, 0].legend(loc='center left', fontsize=8.5, framealpha=0.9)

    fig.align_ylabels(axes[:, 0])
    base = os.path.join(args.out, 'fig_runs_timeline')
    os.makedirs(args.out, exist_ok=True)
    save_figure(fig, base, caption=(
        "Run-by-run data quality over the 2021 and 2023 seasons. Each point is one "
        "raw data file (one run). Top: the minimum presence among the Y-projection "
        "readout sections, i.e. the fraction of events in the run in which the least "
        "active Y section is recorded at all; a value near zero signals an inactive "
        "readout camera. Bottom: the fraction of events in the run with fewer than "
        "three active Y planes, the condition required for a Y-projection track. Open "
        "orange markers are the runs excluded by the good-run selection of "
        "Section 3.5; the December 2021 block of dead-camera runs is clearly visible. "
        "Excluded events remain in the catalogue with their status flagged."))
    plt.close(fig)
    print(f"Written: {base}.png / .pdf")

    # ================= PASTE-BLOCK =================
    P = print
    P("")
    P("===== PASTE-BLOCK RUNS-TIMELINE v1 START =====")
    P(f"run_quality={args.run_quality}  files={len(df)}  no_date_skipped={n_nodate}  "
      f"exclude_files_in_config={len(exclude_set)}")
    if len(dropped):
        P(f"outside dataset years {sorted(dataset_years)} (not shown in figure): "
          f"{len(dropped)} files, {int(dropped['n_events'].sum())} events -> "
          + ", ".join(dropped['source_file'].tolist()))
    for s in seasons:
        sub = df[df['season'] == s]
        g, b = sub[~sub['excluded']], sub[sub['excluded']]
        P(f"season {s}: files good={len(g)} excl={len(b)} | "
          f"events good={int(g['n_events'].sum())} excl={int(b['n_events'].sum())} | "
          f"dates {sub['date'].min():%Y-%m-%d}..{sub['date'].max():%Y-%m-%d}")
    P("excluded runs with min_presence>0.5 (excluded for a reason other than a dead camera):")
    odd = df[df['excluded'] & (df['min_presence'] > 0.5)]
    if len(odd):
        for _, r in odd.iterrows():
            P(f"  {r['source_file']:<20} min_pres={r['min_presence']:.2f} "
              f"lt3Y={r['frac_lt3_y_rows']:.2f} n={int(r['n_events'])}")
    else:
        P("  (none)")
    P("worst 5 GOOD runs by frac_lt3_y_rows (candidates for further exclusion):")
    for _, r in df[~df['excluded']].nlargest(5, 'frac_lt3_y_rows').iterrows():
        P(f"  {r['source_file']:<20} lt3Y={r['frac_lt3_y_rows']:.2f} "
          f"min_pres={r['min_presence']:.2f} n={int(r['n_events'])}")
    P("===== PASTE-BLOCK RUNS-TIMELINE v1 END =====")


if __name__ == '__main__':
    main()