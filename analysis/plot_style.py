"""
plot_style.py - a single publication style for every figure script.

Import at the top of any fig_* script, then plot as usual and finish with
save_figure(fig, 'output/paper_plots/name').

The style targets the figure requirements of Applied Sciences: legible fonts at
a sufficient size, vector PDF alongside a 300 dpi raster PNG, and a restrained
palette.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Shared palette: colourblind-friendly and distinguishable in greyscale ---
COLORS = {
    'x':       '#2c3e50',   # X planes, the primary dark tone
    'y':       '#e67e22',   # Y planes
    'gamma':   '#c0392b',   # gamma block
    'hadron':  '#2c3e50',   # hadron block
    'data':    '#34495e',   # data histograms
    'model':   '#c0392b',   # overlaid model curves
    'accent':  '#27ae60',   # accent, used for the azimuth
    'span':    '#7f8c8d',   # shaded spans, such as the absorber
}


def apply_style():
    """Apply the shared rcParams. Call once at the top of a script."""
    plt.rcParams.update({
        # fonts
        'font.family': 'DejaVu Sans',
        'font.size': 12,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        # lines and markers
        'lines.linewidth': 1.8,
        'lines.markersize': 7,
        'errorbar.capsize': 3,
        # axes and grid
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
        'axes.axisbelow': True,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.9,
        # inward ticks, as journals prefer
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        # output
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,       # editable text in the PDF (TrueType)
        'ps.fonttype': 42,
    })


def save_figure(fig, path_no_ext, caption=None):
    """Save a 300 dpi PNG and a vector PDF. path_no_ext carries no extension.
    If caption is given it is printed, to collect the figure captions."""
    png = path_no_ext + '.png'
    pdf = path_no_ext + '.pdf'
    fig.savefig(png)
    fig.savefig(pdf)
    if caption:
        print(f"\n[FIGURE CAPTION] {path_no_ext.split('/')[-1]}:\n{caption}")
    return png, pdf