"""
cutflow.py - the event selection cut-flow table for publication.

Reads the HDF5 catalogue built by `python main.py scan` and prints, and saves,
the selection sequence:

  all recorded events
   -> passed the pre-selection (energy, planes, both projections, bottom pair)
   -> veto-clean (quality_class != NOISE)
   -> GOOD or better
   → EXCELLENT
  plus a breakdown of the survivors by particle label.

Output: a terminal table, cutflow.csv, and cutflow.tex (booktabs) for the paper.

Run:
    python analysis/cutflow.py --config config/settings.yaml
    python analysis/cutflow.py --catalog data/processed/events_catalog.h5 --out output/paper_plots
"""
import os
import argparse
from collections import OrderedDict

import numpy as np


# Quality ordering, as in data_loader.get_event_metadata
QUALITY_ORDER = {'EXCELLENT': 4, 'GOOD': 3, 'POOR': 2, 'NOISE': 1, 'NOT_ANALYZED': 0}


def _decode(arr):
    """h5py returns bytes for strings; convert them to str."""
    out = []
    for x in arr:
        out.append(x.decode() if isinstance(x, (bytes, bytearray)) else str(x))
    return np.array(out, dtype=object)


def read_catalog(catalog_path):
    """Read the required catalogue fields into a dictionary of arrays."""
    import h5py
    with h5py.File(catalog_path, 'r') as hf:
        meta = hf['metadata']
        data = {
            'event_id': meta['event_id'][:],
            'filter_status': _decode(meta['filter_status'][:]),
        }
        # The quality fields exist only if the scan ran with event_quality.enabled
        if 'quality_class' in meta:
            data['quality_class'] = _decode(meta['quality_class'][:])
            data['particle_type'] = _decode(meta['particle_type'][:])
        else:
            n = len(data['event_id'])
            data['quality_class'] = np.array(['NOT_ANALYZED'] * n, dtype=object)
            data['particle_type'] = np.array(['unknown'] * n, dtype=object)
    return data


def build_cutflow(filter_status, quality_class, particle_type):
    """
    Pure aggregation, testable without h5py.
    Returns (stages, rejections, quality_breakdown, particle_breakdown).
    """
    filter_status = np.asarray(filter_status, dtype=object)
    quality_class = np.asarray(quality_class, dtype=object)
    particle_type = np.asarray(particle_type, dtype=object)

    n_total = len(filter_status)
    passed_mask = filter_status == 'passed'
    n_passed = int(passed_mask.sum())

    q_passed = quality_class[passed_mask]
    # Veto-clean: passed the pre-selection and not graded NOISE or NOT_ANALYZED
    veto_clean_mask = ~np.isin(q_passed, ['NOISE', 'NOT_ANALYZED'])
    n_veto_clean = int(veto_clean_mask.sum())

    def q_at_least(level):
        thr = QUALITY_ORDER[level]
        return int(sum(QUALITY_ORDER.get(str(q), 0) >= thr for q in q_passed))

    n_good = q_at_least('GOOD')
    n_excellent = q_at_least('EXCELLENT')

    stages = OrderedDict()
    stages['Total events'] = n_total
    stages['Passed pre-selection'] = n_passed
    stages['VETO-clean (not NOISE)'] = n_veto_clean
    stages['Quality >= GOOD'] = n_good
    stages['Quality EXCELLENT'] = n_excellent

    # Pre-selection rejection reasons
    rejections = OrderedDict()
    for status in sorted(set(filter_status[~passed_mask])):
        rejections[str(status)] = int((filter_status == status).sum())

    # Quality breakdown among the pre-selected events
    quality_breakdown = OrderedDict()
    for cls in ['EXCELLENT', 'GOOD', 'POOR', 'NOISE', 'NOT_ANALYZED']:
        c = int((q_passed == cls).sum())
        if c:
            quality_breakdown[cls] = c

    # Particle labels among the veto-clean events
    pt_clean = particle_type[passed_mask][veto_clean_mask]
    particle_breakdown = OrderedDict()
    for pt in sorted(set(pt_clean.tolist())):
        particle_breakdown[str(pt)] = int((pt_clean == pt).sum())

    return stages, rejections, quality_breakdown, particle_breakdown


def _pct(n, base):
    return f"{100.0 * n / base:.2f}%" if base else "—"


def print_report(stages, rejections, quality_breakdown, particle_breakdown):
    n_total = next(iter(stages.values()))
    print("\n" + "=" * 62)
    print("CUT-FLOW (event selection)")
    print("=" * 62)
    print(f"{'Stage':<38}{'N':>10}{'% of all':>12}")
    print("-" * 62)
    for name, n in stages.items():
        print(f"{name:<38}{n:>10}{_pct(n, n_total):>12}")

    print("\nPre-selection rejection reasons:")
    for reason, n in rejections.items():
        print(f"  {reason:<34}{n:>10}{_pct(n, n_total):>12}")

    print("\nQuality grades among pre-selected events:")
    for cls, n in quality_breakdown.items():
        print(f"  {cls:<34}{n:>10}")

    print("\nParticle labels (veto-clean events):")
    for pt, n in particle_breakdown.items():
        print(f"  {pt:<34}{n:>10}")
    print("=" * 62 + "\n")


def save_csv(stages, out_dir):
    import csv
    path = os.path.join(out_dir, 'cutflow.csv')
    n_total = next(iter(stages.values()))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['stage', 'n_events', 'fraction_percent'])
        for name, n in stages.items():
            w.writerow([name, n, f"{100.0 * n / n_total:.4f}" if n_total else ""])
    return path


def save_latex(stages, out_dir):
    path = os.path.join(out_dir, 'cutflow.tex')
    n_total = next(iter(stages.values()))
    # Labels needing LaTeX markup; the remaining keys are used verbatim.
    en = {
        'Quality >= GOOD': 'Quality $\\geq$ GOOD',
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Event selection cut-flow.}",
        r"\label{tab:cutflow}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Selection stage & $N$ & Fraction \\",
        r"\midrule",
    ]
    for name, n in stages.items():
        frac = f"{100.0 * n / n_total:.2f}\\%" if n_total else "--"
        label = en.get(name, name).replace('%', r'\%')
        lines.append(f"{label} & {n} & {frac} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    return path


def resolve_catalog(args):
    if args.catalog:
        return args.catalog
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    d = cfg['data']
    return os.path.join(d['processed_dir'], d['catalog_file'])


def main():
    ap = argparse.ArgumentParser(description="Event selection cut-flow table")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--catalog', default=None, help="path to events_catalog.h5 (overrides the config)")
    ap.add_argument('--out', default='output/paper_plots', help="output directory for cutflow.csv and cutflow.tex")
    args = ap.parse_args()

    catalog = resolve_catalog(args)
    if not os.path.exists(catalog):
        raise SystemExit(f"Catalogue not found: {catalog}\nBuild it first: python main.py scan")

    os.makedirs(args.out, exist_ok=True)
    data = read_catalog(catalog)
    result = build_cutflow(data['filter_status'], data['quality_class'], data['particle_type'])
    stages, rejections, quality_breakdown, particle_breakdown = result

    print_report(*result)
    p_csv = save_csv(stages, args.out)
    p_tex = save_latex(stages, args.out)
    print(f"Written: {p_csv}\n         {p_tex}")


if __name__ == '__main__':
    main()