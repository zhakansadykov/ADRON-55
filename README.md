# ADRON-55 Reprocessing Pipeline

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21631377.svg)](https://doi.org/10.5281/zenodo.21631377)

An open, fully scripted reconstruction pipeline for the **ADRON-55** ionization
calorimeter at the Tien Shan high-altitude cosmic-ray station (3340 m a.s.l.,
43.04° N, 76.94° E, Kazakhstan). The pipeline takes the instrument from its raw
per-section data to calibrated three-dimensional tracks and per-plane
detector-response observables.

> **Paper:** *An Open Reprocessing Pipeline and Detector-Response
> Characterization for the ADRON-55 High-Altitude Ionization Calorimeter*
> (Applied Sciences, MDPI, 2026). All figures in the paper are reproducible
> from the raw data using this code.

---

## Overview

A large fraction of recorded triggers are electronic noise, empty events,
electromagnetic flashes, or saturated readouts. The pipeline provides a
transparent cascade that:

1. **assembles** the overlapping readout sections of each chamber row into a
   single position profile;
2. **filters and flags** pathological events (veto conditions) and bad
   data-taking runs (good-run selection);
3. **finds clusters and tracks** with a road search confined to the hadron
   block, then matches the two projections into three-dimensional tracks fitted
   with a weighted least-squares line.

The methodological core of the work is three data-driven procedures that proved
necessary for unbiased results:

- correct assembly of overlapping (not tiled) chamber sections;
- a channel-to-position calibration with a track-based per-plane alignment;
- an automatic good-run selection that removes runs with inactive readout
  cameras, duplicated files, and misfiled seasons.

### Key results (2021 and 2023 data set)

| Quantity                              | Value |
| ------------------------------------- | ----- |
| Recorded events                       | 291 358 |
| Flagged by integrity / good-run audit | 47 612 (16.3%) |
| Good-run events                       | 243 746 |
| Pre-selected                          | 164 227 (56.4%) |
| Quality ≥ GOOD                        | 86 433 (29.7%) |
| ≥ 1 three-dimensional track           | 65 620 (22.5%) |
| Clean single 3+3 track sample         | 991 (0.34%) |
| Azimuthal uniformity χ²/ndf (3+3)     | ≈1.3 × 10³ → 26 after corrections |

Plane occupancy falls from unity in the gamma block to 0.75 (X planes) and 0.61
(Y planes) at 6.4 nuclear interaction lengths; the transverse profile resolves a
narrower electromagnetic component in the gamma block (RMS 0.64 m against
0.72 m).

---

## The Instrument

The calorimeter is a two-tier, 55 m² coordinate detector with about 1220 g/cm²
of absorber. Its sensitive elements are 3 m long rectangular ionization chambers
(copper waveguide, argon at two atmospheres, 120 mm channel pitch) grouped into
eight crossed planes: odd planes measure **X**, even planes measure **Y**.

| Plane | Projection | Height (mm) | Channels | Block  |
| ----- | ---------- | ----------- | -------- | ------ |
| 1     | X          | 4275        | 50       | Gamma  |
| 2     | Y          | 4200        | 69       | Gamma  |
| 3     | X          | 1460        | 48       | Hadron |
| 4     | Y          | 1300        | 72       | Hadron |
| 5     | X          | 1010        | 48       | Hadron |
| 6     | Y          | 820         | 72       | Hadron |
| 7     | X          | 630         | 48       | Hadron |
| 8     | Y          | 470         | 72       | Hadron |

The gamma block (planes 1–2) is separated from the hadron block by a 220 mm
lead target and a 2.2 m air gap. Because multiple scattering and secondary
production in the lead break the trajectory, **the gamma planes are excluded
from the direction fit**; tracking uses only the six hadron-block planes.

> **Critical detail — overlapping sections.** Each row is read out as two
> (gamma) or three (hadron X) sections that *overlap* in the transverse
> coordinate rather than tile it. They are therefore summed channel by channel
> with aligned indices — not concatenated, not reversed, and not truncated to
> the shortest section. Getting this wrong folds or collapses the position
> measurement and biases every downstream result.

---

## Repository Structure

```
.
├── main.py                     # CLI entry point (scan / inspect / visualize / reconstruct)
├── config/
│   ├── settings.yaml           # Local config (not committed — absolute paths)
│   └── settings.yaml.example   # Template with the published parameter values
├── src/                        # Pipeline modules
│   ├── data_loader.py          # .dat parsing, HDF5 catalogue, good-run selection
│   ├── preprocessing.py        # Section assembly (combine_overlap), channel to mm
│   ├── event_quality.py        # Veto, clusters, road search, quality scoring
│   ├── tracker.py              # 3D reconstruction (X-Y matching, weighted fit)
│   ├── physics.py              # Alt/Az to RA/Dec (Astropy), absorber depths
│   ├── visualizer.py           # 2D histograms (matplotlib) and 3D (plotly)
│   └── exporter.py             # TXT / JSON / CSV export
├── analysis/                   # Diagnostic and paper-figure scripts
├── tests/                      # Unit tests (pytest)
├── docs/                       # Documentation
├── data/
│   ├── raw/                    # Raw .dat files (not committed — see Zenodo)
│   └── processed/              # HDF5 catalogue (not committed)
├── output/                     # Figures, HTML, CSV (not committed)
├── logs/                       # Run logs (not committed)
├── CITATION.cff
├── LICENSE                     # BSD 3-Clause
└── README.md
```

---

## Installation

```bash
git clone https://github.com/zhakansadykov/ADRON-55.git
cd ADRON-55
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config/settings.yaml.example config/settings.yaml   # then edit the paths
```

Dependencies: `numpy`, `scipy`, `pyyaml`, `matplotlib`, `plotly`, `astropy`,
`h5py`, `pandas`, `tqdm`.

The raw data are not part of this repository. Download them from the Zenodo data
record (see [Data and Code Availability](#data-and-code-availability)) and
unpack them so that the files sit in `data/raw/bank0/2021/` and
`data/raw/bank0/2023/`, split by the year encoded in the filename (`YYMMDD.dat`).
**Do not rename the files:** event identifiers are assigned in sorted filename
order, so renaming shifts every `global_id`.

---

## Usage

```bash
# 1. Build the event catalogue (scan + quality + good-run selection)
python main.py scan

# 2. Inspect events
python main.py inspect --ids 0 --stage raw --output terminal
python main.py inspect --ids 10 --stage quality --output json

# 3. Visualize (2D histograms and/or 3D reconstruction)
python main.py visualize --ids 0,1 --type both

# 4. Batch 3D reconstruction to CSV
python main.py reconstruct --min-quality GOOD --save-csv output/reco.csv
```

### Reproducing the paper

```bash
python main.py scan                                              # event catalogue
python analysis/audit_raw_files.py  --config config/settings.yaml  # integrity audit
python analysis/cutflow.py          --config config/settings.yaml  # Table 4
python analysis/reconstruct_all.py  --config config/settings.yaml --min-quality GOOD
python analysis/paper_plots.py      --reco output/paper_plots/reco.csv
```

A single streaming pass over the roughly 2.9 × 10⁵ events takes about ninety
seconds on a laptop-class machine.

### Analysis scripts

| Script | Purpose |
| ------ | ------- |
| `analysis/reconstruct_all.py`    | Streaming 3D reconstruction to `reco.csv` (with RA/Dec) |
| `analysis/paper_plots.py`        | Paper figures (zenith, azimuth, χ², sky map) |
| `analysis/cutflow.py`            | Selection cut-flow table (CSV and LaTeX) |
| `analysis/audit_raw_files.py`    | Raw-file integrity audit (duplicates, wrong season) |
| `analysis/diag_runs.py`          | Per-run camera presence, generates the exclusion list |
| `analysis/diag_residuals.py`     | Track-based alignment on 3+3 tracks |
| `analysis/diag_occupancy.py`     | Longitudinal occupancy per plane |
| `analysis/diag_slopes.py`        | Slope distributions and plane participation |
| `analysis/diag_geometry.py`      | Per-plane offset from the mean centroid |
| `analysis/diag_camera_lengths.py`| Measured length of each readout section |
| `analysis/diag_clusters_bottom.py`| Cluster yield per plane against threshold |
| `analysis/diag_track_quality.py` | Track metrics, spurious short-track search |
| `analysis/diag_align_muons.py`   | Alignment cross-check on single muons |
| `analysis/fig_*.py`              | Individual paper figures |
| `analysis/make_table_*.py`       | Geometry and response tables (LaTeX) |

---

## Known Limitations

- **The absolute energy scale is not finalized.** The inherited conversion
  factor and an independent estimate from the detector documentation differ by
  three orders of magnitude; the scale must be anchored to the
  minimum-ionizing peak. No energy spectrum is reported until this is resolved.
- **Angular resolution is geometry-limited.** Only about 30% of particles reach
  the deepest planes, so most tracks are reconstructed from two points and their
  slope is quantized. The clean 3+3 sample (991 events) is the basis of the
  angular results.
- **The particle classifier is not used.** It was written against an earlier
  plane arrangement and needs revision for hadron-block tracking.
- **Acceptance correction** of the zenith distribution requires the
  collaboration's Geant4 model (Hadr55) and is left for a follow-up study.
- **Azimuth is reported in the detector frame.** The recorded offset of the
  detector X axis from north (`location.detector_angle_offset`) is applied only
  in the conversion to equatorial coordinates.

---

## Documentation

- [`docs/EVENT_QUALITY.md`](docs/EVENT_QUALITY.md) — clustering, veto
  conditions, road search, and quality scoring.
- [`docs/3D_RECONSTRUCTION.md`](docs/3D_RECONSTRUCTION.md) — projection
  matching, weighted least-squares fitting, and vertex finding.
The reasoning behind each calibration, and the diagnostics that led to it, are
described in Section 3 of the accompanying paper.

---

## Data and Code Availability

The pipeline is released under the [BSD 3-Clause License](LICENSE) and archived
on Zenodo: **[10.5281/zenodo.21631377](https://doi.org/10.5281/zenodo.21631377)**
(concept DOI, always resolving to the latest version).

The raw event data of the 2021 and 2023 seasons — 263 daily files, 291 358 event
records — are archived as a separate Zenodo record:
**[<DATA-DOI>](https://doi.org/<DATA-DOI>)**.

---

## Citation

If you use this pipeline, please cite the accompanying paper and the software
DOI:

> Sadykov, T.; Argynova, A.; Makhmet, K.; Kantarbayeva, D.; Tautayev, Y.;
> Mukanov, Y.; Sadykov, Z. *An Open Reprocessing Pipeline and
> Detector-Response Characterization for the ADRON-55 High-Altitude Ionization
> Calorimeter.* Appl. Sci. **2026**.

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff).

---

## Funding

This research was funded by the Science Committee of the Ministry of Science and
Higher Education of the Republic of Kazakhstan, grant number AP23488396.

## Authors

Institute of Physics and Technology, Satbayev University, Almaty, Kazakhstan.
Correspondence: <zhakansadykov@gmail.com>

We thank the staff of the Tien Shan High-Mountain Scientific Station for
operating the HADRON-55 complex and maintaining the raw data archive.
