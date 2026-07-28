# ADRON-55 Reprocessing Pipeline

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

An open, fully scripted reconstruction pipeline for the **ADRON-55**
ionization calorimeter at the Tien Shan high-altitude cosmic-ray station
(3340 m a.s.l., 43.04° N, 76.94° E, Kazakhstan). The pipeline takes the
instrument from its raw per-section data to calibrated three-dimensional
tracks and per-plane detector-response observables.

> **Paper:** *An Open Reprocessing Pipeline and Detector-Response
> Characterization for the ADRON-55 High-Altitude Ionization Calorimeter*
> (Applied Sciences, MDPI, 2026). All figures in the paper are reproducible
> from the raw data using this code.

---

## Overview

The ADRON-55 calorimeter records ~600 000 events per year, a large fraction
of which are electronic noise, empty triggers, electromagnetic flashes, or
saturated readouts. This pipeline provides a transparent, three-stage cascade
that:

1. **Assembles** the overlapping readout sections of each chamber row into a
   single position profile;
2. **Filters and flags** pathological events (VETO checks) and bad data-taking
   runs (good-run selection);
3. **Finds clusters and tracks** with a road-search algorithm confined to the
   hadron block, then matches the two projections into 3D tracks fitted with a
   weighted least-squares line.

The methodological core of the work is three **data-driven** procedures that we
found necessary for unbiased results:

- correct assembly of overlapping (not tiled) chamber sections;
- a channel-to-position calibration with a track-based per-plane alignment;
- an automatic good-run selection that removes runs with inactive readout
  cameras, duplicated files, and misfiled seasons.

### Key results (2021 + 2023 data set)

| Quantity | Value |
|---|---|
| Recorded events | 291 358 |
| Flagged by integrity / good-run audit | 47 612 (16.3 %) |
| Good-run events | 243 746 |
| Pre-selected | 164 227 (56.4 %) |
| Quality ≥ GOOD | 86 433 (29.7 %) |
| ≥ 1 three-dimensional track | 65 620 (22.5 %) |
| Clean single 3+3 track sample | 991 (0.34 %) |
| Azimuthal uniformity χ²/ndf (3+3) | ~1.3 × 10³ → **26** after corrections |

The longitudinal occupancy falls from unity in the gamma block to 0.75 (X
planes) and 0.61 (Y planes) at 6.4 nuclear interaction lengths; the transverse
profile resolves a narrower electromagnetic component in the gamma block
(RMS 0.64 m vs 0.72 m).

---

## The Instrument

The calorimeter is a two-tier, 55 m² coordinate detector with ~1220 g/cm² of
absorber. Its sensitive elements are 3 m long rectangular ionization chambers
(copper waveguide, argon at 2 atm, 120 mm channel pitch) grouped into eight
crossed planes: odd planes measure **X**, even planes measure **Y**.

| Plane | Projection | Height (mm) | Block |
|---|---|---|---|
| 1 | X | 4275 | Gamma |
| 2 | Y | 4200 | Gamma |
| 3 | X | 1460 | Hadron |
| 4 | Y | 1300 | Hadron |
| 5 | X | 1010 | Hadron |
| 6 | Y | 820 | Hadron |
| 7 | X | 630 | Hadron |
| 8 | Y | 470 | Hadron |

The gamma block (planes 1–2) is separated from the hadron block by a 22 cm
lead target and a 2.2 m air gap. Because multiple scattering and secondary
production in the lead break the trajectory, **the gamma planes are excluded
from the direction fit**; tracking uses only the six hadron-block planes.

> **Critical detail — overlapping sections.** Each row is read out as two
> (gamma) or three (hadron X) sections that *overlap* in the transverse
> coordinate rather than tile it. They are therefore summed **channel-by-channel
> with aligned indices** — not concatenated, reversed, or truncated to the
> shortest section. Getting this wrong folds or collapses the position
> measurement and biases every downstream result.

---

## Repository Structure

```
.
├── main.py                     # CLI entry point (scan / inspect / visualize / reconstruct)
├── config/
│   ├── settings.yaml           # Local config (NOT committed — absolute paths)
│   └── settings.yaml.example   # Template (committed)
├── src/                        # Pipeline modules
│   ├── data_loader.py          # .dat parsing, HDF5 catalogue, good-run selection
│   ├── preprocessing.py        # Section assembly (combine_overlap), channel→mm
│   ├── event_quality.py        # VETO, clusters, road search, quality scoring
│   ├── tracker.py              # 3D reconstruction (hybrid X↔Y matching, WLS)
│   ├── physics.py              # Alt/Az → RA/Dec (astropy)
│   ├── visualizer.py           # 2D histograms (matplotlib) + 3D (plotly)
│   └── exporter.py             # TXT / JSON / CSV export
├── analysis/                   # Diagnostic & paper-figure scripts
├── tests/                      # Unit tests (pytest)
├── examples/                   # Demo scripts
├── docs/                       # Documentation
├── data/
│   ├── raw/                    # Raw .dat files (NOT committed)
│   └── processed/              # HDF5 catalogue (NOT committed)
├── output/                     # Figures, HTML, CSV (NOT committed)
├── logs/                       # Run logs (NOT committed)
├── LICENSE                     # BSD 3-Clause
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<USER>/adron55-pipeline.git
cd adron55-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config/settings.yaml.example config/settings.yaml   # edit paths
```

Dependencies: `numpy`, `scipy`, `pyyaml`, `matplotlib`, `plotly`, `astropy`,
`h5py`, `pandas`, `tqdm`.

---

## Usage

```bash
# 1. Build the event catalogue (scan + quality + good-run selection)
python main.py scan

# 2. Inspect events
python main.py info --ids 0,100,500
python main.py info --min-quality GOOD

# 3. Visualize (2D histograms and/or 3D reconstruction)
python main.py visualize --ids 0,1 --type both

# 4. Batch 3D reconstruction → CSV
python main.py reconstruct --min-quality GOOD --save-csv output/reco.csv
```

### Analysis scripts

| Script | Purpose |
|---|---|
| `analysis/reconstruct_all.py` | Streaming 3D reconstruction → `reco.csv` (+RA/Dec) |
| `analysis/paper_plots.py` | Paper figures (zenith, azimuth, χ², sky map) |
| `analysis/cutflow.py` | Selection cut-flow table (CSV + LaTeX) |
| `analysis/diag_residuals.py` | Track-based alignment (additive, on 3+3 tracks) |
| `analysis/diag_runs.py` | Per-run camera presence → `exclude_files.txt` |
| `analysis/diag_occupancy.py` | Longitudinal occupancy per plane |

---

## Known Limitations

- **Absolute energy scale is not finalized.** The inherited conversion factor
  and an independent documentation estimate differ by three orders of
  magnitude; the scale must be anchored to the minimum-ionizing peak. No energy
  spectrum is reported until this is resolved.
- **Angular resolution is geometry-limited.** Only ~30 % of particles reach the
  deepest planes, so most tracks are two-point (quantized slope). The clean 3+3
  sample (991 events) is the basis of the angular results.
- **Particle classifier is not used.** It was written against an earlier plane
  arrangement and needs revision for hadron-block tracking.
- **Acceptance correction** of the zenith distribution requires the
  collaboration's Geant4 model (Hadr55) and is left for a follow-up study.

---

## Documentation

- [`docs/event_quality.md`](docs/event_quality.md) — clustering, VETO, road
  search, and quality scoring.
- [`docs/3d_reconstruction.md`](docs/3d_reconstruction.md) — hybrid X↔Y
  matching, weighted least-squares fitting, and vertex finding.

---

## Data and Code Availability

The pipeline is released under the [BSD 3-Clause License](LICENSE) and
archived on Zenodo (DOI: `10.5281/zenodo.XXXXXXX`). The raw event data are
archived separately (DOI: `10.5281/zenodo.XXXXXXX`).

---

## Citation

If you use this pipeline, please cite the accompanying paper and the software
DOI:

> Sadykov, T.; Argynova, A.; Makhmet, K.; Piscal, V.; Tautayev, Y.;
> Sadykov, Z. *An Open Reprocessing Pipeline and Detector-Response
> Characterization for the ADRON-55 High-Altitude Ionization Calorimeter.*
> Appl. Sci. **2026**.

---

## Funding

This research was funded by the Science Committee of the Ministry of Science
and Higher Education of the Republic of Kazakhstan, grant number AP23488396.

## Authors

Institute of Physics and Technology, Satbayev University, Almaty, Kazakhstan.
Correspondence: zhakansadykov@gmail.com
```

---
