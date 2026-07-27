# ADRON-55 Reconstruction Pipeline

[![DOI](https://zenodo.org/badge/1295658994.svg)](https://doi.org/10.5281/zenodo.21631377)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An open, fully scripted reconstruction pipeline for the **ADRON-55
ionization–neutron calorimeter** at the Tien Shan high-altitude station
(3340 m a.s.l., 43.04° N, 76.94° E, Kazakhstan).

The pipeline takes the instrument from its raw per-section data to
reconstructed three-dimensional tracks and per-plane detector-response
observables. It is the software described in:

> Sadykov, T.; Argynova, A.; Makhmet, K.; Piscal, V.; Tautayev, Y.;
> Sadykov, Z. *An Open Reprocessing Pipeline and Detector-Response
> Characterization for the ADRON-55 High-Altitude Ionization Calorimeter.*
> Applied Sciences **2026**, *16*, —. https://doi.org/10.3390/app16XX

---

## Table of Contents

- [Overview](#overview)
- [Detector](#detector)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Data](#data)
- [Reproducing the Paper](#reproducing-the-paper)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Citation](#citation)
- [License](#license)
- [Authors](#authors)
- [Funding](#funding)

---

## Overview

The ADRON-55 calorimeter records extensive air shower (EAS) cores in
crossed ionization-chamber planes. This pipeline:

1. **Decodes and stitches** the overlapping readout sections of each
   chamber row into a single position profile.
2. **Finds clusters** and applies veto conditions to reject pathological
   events.
3. **Reconstructs tracks** via a seeded road-search algorithm in each
   projection, followed by hybrid X–Y matching and a weighted
   least-squares 3-D fit.
4. **Calibrates** channel-to-position mapping and performs iterative
   track-based plane alignment.
5. **Selects good runs** through a file-level integrity audit
   (inactive cameras, duplicate files, wrong-season records).
6. **Characterizes the detector response**: longitudinal occupancy and
   energy deposition, transverse profiles, and angular distributions.

The pipeline processes ~290 000 events (2021 + 2023 seasons) in a
single streaming pass in ≈ 90 s on a laptop-class machine.

---

## Detector

| Parameter | Value |
|---|---|
| Sensitive area | 55 m² |
| Total absorber depth | ~1220 g/cm² (≈ 6.4 λ_I to deepest active plane) |
| Chamber cross-section | 11 × 6 cm², 3 m long |
| Fill gas | Ar at 2 atm |
| Anode voltage | 600 V |
| Channel pitch | 120 mm |
| Planes | 8 active (planes 1–8); plane 9 not operational |
| Projections | X (planes 1, 3, 5, 7) / Y (planes 2, 4, 6, 8) |
| Gamma block | Planes 1–2 (Pb, ~8 X₀) |
| Hadron block | Planes 3–8 (Fe, 980 mm total) |
| Pb target + air gap | 220 mm Pb + 2200 mm air between blocks |

---

## Installation

### Requirements

- Python ≥ 3.10
- ~2 GB RAM for the full data set

### Setup

```bash
git clone https://github.com/adron55/adron55-pipeline.git
cd adron55-pipeline
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Copy and edit the configuration
cp config/settings.yaml.example config/settings.yaml
#    → set data.raw_dir to your raw data directory

# 2. Build the event catalogue (scan + quality + good-run selection)
python main.py scan

# 3. Inspect a single event
python main.py inspect --ids 0 --stage processed --output terminal

# 4. Visualize (2-D histograms + 3-D reconstruction)
python main.py visualize --ids 0,100,500 --type both

# 5. Batch 3-D reconstruction
python main.py reconstruct --min-quality GOOD --save-csv output/reco.csv

# 6. Reproduce the paper figures
cd analysis
python paper_plots.py          # generates all figures into output/paper_plots/
```

---

## Pipeline Stages

```
Raw .dat files
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — Assembly (preprocessing.py)                  │
│  Overlapping sections summed by index (no reversal,     │
│  no truncation). Padded to nominal row length.          │
├─────────────────────────────────────────────────────────┤
│  Stage 2 — Clustering + Veto (event_quality.py)         │
│  Contiguous clusters (thr = 30 ADC, min sum = 50).      │
│  Saturation / uniformity / emptiness / flash vetoes.    │
├─────────────────────────────────────────────────────────┤
│  Stage 3 — Track finding (event_quality.py, tracker.py) │
│  Seeded road search in hadron block (planes 3–8).       │
│  Seeds: planes 8, 7, 6, 5, 4. Adaptive window.          │
│  X–Y matching: energy-ratio ≤ 3 → rank pairing;         │
│  otherwise geometric proximity at z_common.             │
├─────────────────────────────────────────────────────────┤
│  Stage 4 — 3-D fit (tracker.py)                         │
│  Weighted least-squares line; weights ∝ (ΣX/X₀)⁻¹.      │
│  Gamma planes excluded. χ²/ndf < 10 acceptance.         │
├─────────────────────────────────────────────────────────┤
│  Stage 5 — Calibration (preprocessing.py, tracker.py)   │
│  Per-width centring + iterative track-based alignment   │
│  (2 iterations, converges to mean slope ≈ −0.01).       │
├─────────────────────────────────────────────────────────┤
│  Stage 6 — Good-run selection (data_loader.py)          │
│  File-level audit: camera presence, duplicates,         │
│  wrong-season timestamps. 16.3% of events flagged.      │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
Per-plane observables, 3-D tracks, angular distributions
```

---

## Configuration

All parameters are in `config/settings.yaml`. An annotated example is
provided in `config/settings.yaml.example`. Key sections:

| Section | Purpose |
|---|---|
| `geometry` | Plane heights, channel counts, pitch, alignment offsets |
| `data` | Raw/processed directories, years, catalogue filename |
| `filters` | Pre-selection thresholds (energy, active planes, hits) |
| `event_quality` | Clustering, veto, road-search, quality-class thresholds |
| `absorber` | Material stack between planes (Pb, Fe, air) |
| `tracking` | Road-search radius, hybrid matching, 3-D fit parameters |
| `calibration` | ADC → energy conversion (uncalibrated scale) |
| `location` | Station coordinates for astronomical conversion |
| `visualization` | Event selection mode, histogram DPI, output paths |

---

## Data

The raw data set comprises 291 358 event records in 263 daily files:

| Season | Files | Size |
|---|---|---|
| 2021 | ~190 | ~847 MB |
| 2023 | ~73 | ~287 MB |
| **Total** | **263** | **~1.1 GB** |

Raw data are archived on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21631378.svg)](https://doi.org/10.5281/zenodo.21631378)

Each event record is a plain-text block:

```
|EVENT: 06.01.2021 11:09:24 [378] 0 B 378
right_g high sensitivity: 10 20 30 ... -1
left_g high sensitivity: 15 25 35 ... -1
front_g high sensitivity: ...
...
#
```

The `-1` marker delimits valid data within each section.

---

## Reproducing the Paper

All figures and tables in the paper are generated by scripts in
`analysis/`:

| Script | Output |
|---|---|
| `paper_plots.py` | Master script — calls all below |
| `fig_pipeline.py` | `fig_pipeline.pdf` — pipeline schematic |
| `fig_runs_timeline.py` | `fig_runs_timeline.pdf` — run quality |
| `fig_longitudinal.py` | `fig_longitudinal.pdf` — longitudinal profile |
| `fig_transverse.py` | `fig_transverse.pdf` — lateral profile |
| `fig_angular_summary.py` | `fig_angular_summary.pdf` — angular distributions |
| `fig_skymap_final.py` | `fig_skymap_final.pdf` — equatorial sky map |
| `make_table_geometry.py` | `table_geometry.tex` |
| `make_table_response.py` | `table_response.tex` |
| `cutflow.py` | `cutflow.csv`, `cutflow.tex` |
| `audit_raw_files.py` | `run_quality.csv`, `exclude_files.txt` |
| `reconstruct_all.py` | `reco.csv`, `angular_sample.csv` |

Diagnostic scripts (`diag_*.py`) reproduce intermediate checks
described in the paper (slope convergence, occupancy, residuals, etc.).

```bash
cd analysis
python paper_plots.py
# → all outputs in ../output/paper_plots/
```

---

## Project Structure

```
adron55-pipeline/
├── main.py                     # CLI entry point
├── config/
│   └── settings.yaml.example   # Annotated configuration template
├── src/
│   ├── data_loader.py          # Raw parsing, HDF5 catalogue, good-run audit
│   ├── preprocessing.py        # Section assembly, channel→mm calibration
│   ├── event_quality.py        # Clustering, vetoes, road search, scoring
│   ├── tracker.py              # X–Y matching, 3-D fit, vertex finding
│   ├── physics.py              # Alt/Az → RA/Dec (astropy), energy calib
│   ├── visualizer.py           # 2-D histograms (matplotlib), 3-D (plotly)
│   └── exporter.py             # CSV / TXT / JSON export utilities
├── analysis/                   # Paper figure & table generators
│   ├── paper_plots.py
│   ├── fig_*.py
│   ├── diag_*.py
│   ├── make_table_*.py
│   ├── cutflow.py
│   ├── audit_raw_files.py
│   ├── reconstruct_all.py
│   └── plot_style.py
├── tests/                      # Unit tests (pytest)
│   ├── conftest.py
│   └── test_preprocessing.py
├── docs/                       # Extended documentation
├── data/
│   ├── raw/bank0/{2021,2023}/  # Raw .dat files (not in git)
│   └── processed/              # HDF5 catalogue (not in git)
├── output/                     # Generated figures, CSVs (not in git)
├── logs/                       # Run logs (not in git)
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Testing

```bash
pip install pytest
pytest tests/ -v
```

---

## Citation

If you use this software, please cite both the paper and the software
release:

**Paper:**

> Sadykov, T.; Argynova, A.; Makhmet, K.; Piscal, V.; Tautayev, Y.;
> Sadykov, Z. An Open Reprocessing Pipeline and Detector-Response
> Characterization for the ADRON-55 High-Altitude Ionization Calorimeter.
> *Applied Sciences* **2026**, *16*, —.

**Software:**

> Sadykov, T.; Argynova, A.; Makhmet, K.; Piscal, V.; Tautayev, Y.;
> Sadykov, Z. (2026). ADRON-55 Reconstruction Pipeline
> [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.21631378

BibTeX:

```bibtex
@software{adron55_pipeline_2026,
  author       = {Sadykov Turlan and Argynova Alia and
                  Makhmet Khanshaiym and Piscal Vyacheslav and
                  Tautayev Yernar and Sadykov Zhakypbek},
  title        = {ADRON-55 Reconstruction Pipeline},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21631378},
  url          = {https://doi.org/10.5281/zenodo.21631378}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Authors

| Name | Affiliation |
|---|---|
| Turlan Sadykov | Satbayev University, Almaty, Kazakhstan |
| Alia Argynova | Satbayev University, Almaty, Kazakhstan |
| Khanshaiym Makhmet | Satbayev University, Almaty, Kazakhstan |
| Vyacheslav Piscal | Satbayev University, Almaty, Kazakhstan |
| Yernar Tautayev | Satbayev University, Almaty, Kazakhstan |
| Zhakypbek Sadykov ✉ | Satbayev University, Almaty, Kazakhstan |

Correspondence: zhakansadykov@gmail.com

---

## Funding

This research was funded by the Science Committee of the Ministry of
Science and Higher Education of the Republic of Kazakhstan, grant
number **AP23488396** (*Study of hadron interactions at ultrahigh
energies through the longitudinal, transverse, and azimuthal
characteristics of hadrons in EAS cores*).

---

## Acknowledgments

The authors thank the staff of the Tien Shan High-Mountain Scientific
Station for the operation and maintenance of the installation.
```