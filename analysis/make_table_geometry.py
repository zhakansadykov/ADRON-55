"""
make_table_geometry.py - the plane geometry and cumulative depth table.

Built directly from config/settings.yaml (heights, expected_channels, absorber),
so that the table in the paper is guaranteed to match the numbers the pipeline
actually uses.

Output: output/paper_plots/table_geometry.tex (booktabs), plus a LaTeX and a
plain-text version printed to the terminal for a quick check.

Run:
    python analysis/make_table_geometry.py --config config/settings.yaml
"""
import os
import argparse

MAT_SHORT = {'lead': 'Pb', 'iron': 'Fe', 'air': 'air'}
ROW_PROJ = ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y']


def cumulative_depth(cfg):
    ab = cfg['absorber']
    mats = ab['materials']
    rows = []
    cx = clam = 0.0
    for i in range(8):
        entry = ab['layers'].get(f'before_row_{i+1}')
        layers = entry if isinstance(entry, list) else ([entry] if entry else [])
        parts = []
        for layer in layers:
            m = mats[layer['material']]
            th = float(layer['thickness_mm'])
            cx += th / m['X0_mm']
            clam += th / m['lambda_I_mm']
            short = MAT_SHORT.get(layer['material'], layer['material'])
            parts.append(f"{short} {th:g}")
        mat_str = ' + '.join(parts) if parts else '--'
        rows.append((mat_str, cx, clam))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Geometry table (Table 1)")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--out', default='output/paper_plots')
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    heights = cfg['geometry']['heights']
    expected = cfg['geometry']['expected_channels']
    cw = cfg['geometry']['channel_width']
    depth = cumulative_depth(cfg)

    # --- plain-text version, for a quick check ---
    print(f"{'row':>3} {'proj':>4} {'z_mm':>6} {'N_ch':>5} {'width_m':>8} "
          f"{'absorber_above':>30} {'sumX/X0':>8} {'sum_lamI':>9}")
    for i in range(8):
        n_ch = expected[f'row_{i+1}']
        mat, sx0, sli = depth[i]
        print(f"{i+1:>3} {ROW_PROJ[i]:>4} {heights[i]:>6} {n_ch:>5} "
              f"{n_ch*cw/1000:>8.2f} {mat:>30} {sx0:>8.1f} {sli:>9.2f}")

    # --- LaTeX ---
    block = 'gamma'
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Geometry of the eight active chamber planes and the cumulative "
        r"absorber depth above each plane, as used by the reconstruction. Depths are "
        r"summed from the top of the stack; the channel pitch is 120~mm throughout. "
        r"The gamma block (planes 1--2) is separated from the hadron block "
        r"(planes 3--8) by the 220~mm lead target, a 2.2~m air gap, a 130~mm "
        r"lead hadron converter, and the first 100~mm iron layer.}",
        r"\label{tab:geometry}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"Plane & Proj. & $z$ (mm) & $N_\mathrm{ch}$ & Absorber above (mm) & "
        r"$\Sigma X/X_0$ & $\Sigma\lambda_I$ \\",
        r"\midrule",
    ]
    for i in range(8):
        if i == 2 and block == 'gamma':
            lines.append(r"\midrule")
            block = 'hadron'
        n_ch = expected[f'row_{i+1}']
        mat, sx0, sli = depth[i]
        mat_tex = mat.replace('air', 'air')  # kept as is
        lines.append(f"{i+1} & {ROW_PROJ[i]} & {heights[i]} & {n_ch} & "
                     f"{mat_tex} & {sx0:.1f} & {sli:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'table_geometry.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\nWritten: {path}\n")
    print("===== PASTE-BLOCK TABLE-GEOMETRY v1 (LaTeX) START =====")
    print("\n".join(lines))
    print("===== PASTE-BLOCK TABLE-GEOMETRY v1 END =====")


if __name__ == '__main__':
    main()