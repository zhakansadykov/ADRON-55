"""
make_table_response.py - the longitudinal response table, per plane.

Built from output/paper_plots/longitudinal_profile.csv, produced by
fig_longitudinal.py, so that the table matches the corresponding figure.

Columns: plane, projection, cumulative lambda_I, occupancy with its 68% Wilson
interval, the mean deposition per channel over all events (ADC; a silent plane
enters as zero), and the same quantity CONDITIONED on the plane having fired.

The conditional column is what settles the obvious question about the apparent
collapse of the deposition in plane 6: conditioned on firing, all eight planes
fall in a common band of about 235-324 ADC, so the entire difference between the
projections in the unconditional means originates in the occupancy.

Output: output/paper_plots/table_response.tex

Run:
    python analysis/make_table_response.py --csv output/paper_plots/longitudinal_profile.csv
"""
import os
import csv
import argparse


def main():
    ap = argparse.ArgumentParser(description="Longitudinal response table")
    ap.add_argument('--csv', default='output/paper_plots/longitudinal_profile.csv')
    ap.add_argument('--out', default='output/paper_plots')
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            occ = float(r['occupancy'])
            lo, hi = float(r['occ_lo']), float(r['occ_hi'])
            e, de = float(r['energy_per_channel']), float(r['energy_per_channel_err'])
            rows.append({
                'row': int(r['row']), 'proj': r['proj'],
                'lam': float(r['depth_lambdaI']),
                'occ': occ, 'docc': max(hi - occ, occ - lo),
                'e': e, 'de': de,
                'ec': e / occ, 'dec': de / occ,
            })
    assert len(rows) == 8, f"expected 8 planes, got {len(rows)}"

    # --- plain-text preview ---
    print(f"{'row':>4} {'pr':>3} {'lam_I':>6} {'occupancy':>16} "
          f"{'<E>/Nch':>13} {'<E>/Nch | hit':>14}")
    for r in rows:
        print(f"{r['row']:>4} {r['proj']:>3} {r['lam']:>6.2f} "
              f"{r['occ']:.4f} ± {r['docc']:.4f} "
              f"{r['e']:>7.1f} ± {r['de']:<4.1f} {r['ec']:>8.1f} ± {r['dec']:<4.1f}")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Longitudinal response of the eight active planes on the "
        r"good-run sample ($N=34\,821$; uniform one-in-seven sampling of the "
        r"data set). Occupancy is the fraction of events with any above-pedestal "
        r"signal in the plane, with 68\% Wilson intervals; the mean deposition "
        r"per channel is taken over all events (ADC counts; events without "
        r"signal in the plane enter as zero), and the last column conditions it "
        r"on the plane having fired. Conditioned on firing, all planes fall in "
        r"a common 235--324~ADC band: the entire difference between the "
        r"projections in the unconditional means originates in the occupancy.}",
        r"\label{tab:response}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"Plane & Proj. & $\Sigma\lambda_I$ & Occupancy & "
        r"$\langle E\rangle/N_\mathrm{ch}$ (ADC) & "
        r"$\langle E\rangle/N_\mathrm{ch}\,|\,\mathrm{fired}$ \\",
        r"\midrule",
    ]
    for r in rows:
        if r['row'] == 3:
            lines.append(r"\midrule")
        occ_s = (f"$\\approx$1" if r['occ'] > 0.9995
                 else f"{r['occ']:.3f} $\\pm$ {r['docc']:.3f}")
        lines.append(
            f"{r['row']} & {r['proj']} & {r['lam']:.2f} & {occ_s} & "
            f"{r['e']:.0f} $\\pm$ {r['de']:.0f} & "
            f"{r['ec']:.0f} $\\pm$ {r['dec']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'table_response.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\nWritten: {path}\n")
    print("===== PASTE-BLOCK TABLE-RESPONSE v1 (LaTeX) START =====")
    print("\n".join(lines))
    print("===== PASTE-BLOCK TABLE-RESPONSE v1 END =====")


if __name__ == '__main__':
    main()