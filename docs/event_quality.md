# Event Quality Assessment

Module `src/event_quality.py` implements the event filtering, clustering,
track finding, and quality scoring pipeline for the ADRON-55 ionization
calorimeter. It operates on assembled per-plane channel arrays (see
`preprocessing.py`) and produces per-event quality metrics used by the
downstream selection and reconstruction stages.

## Table of Contents

- [Overview](#overview)
- [Detector and Absorber Structure](#detector-and-absorber-structure)
- [Algorithms](#algorithms)
  - [Cluster Finder](#1-cluster-finder)
  - [VETO Engine](#2-veto-engine)
  - [Road Search Tracker](#3-road-search-tracker)
  - [Track Deduplication](#4-track-deduplication)
  - [Quality Scoring](#5-quality-scoring)
  - [Particle Classification](#6-particle-classification)
- [Formulas](#formulas)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
- [Results on Real Data](#results-on-real-data)
- [Quality Class Reference](#quality-class-reference)
- [Particle Type Reference](#particle-type-reference)

## Overview

The ADRON-55 calorimeter records 600 000+ events per year. A substantial
fraction of these are not usable physics events:

- Electronic noise (EMI, false ADC triggers)
- Empty events (1–2 chambers firing on background)
- Flash events (all chambers firing simultaneously from electromagnetic
  interference)
- Saturated events (all-max values from high-voltage faults)
- Weak showers without a reconstructable track structure

Track reconstruction and arrival-direction determination require events with
well-defined tracks crossing multiple detector layers. The quality assessment
pipeline addresses this through a cascade of filters, a road-search track
finder, a 0–100 quality score, and a particle-type classifier.

Key design principles:

- **Soft filtering.** Events are flagged, not discarded. Final selection is
  performed downstream via the `--min-quality` CLI parameter.
- **Parallel processing.** Multiprocessing via `multiprocessing.Pool` with a
  configurable number of workers.
- **Adaptive road search.** The search radius grows with the thickness of the
  absorber traversed, accounting for multiple scattering.
- **Physically motivated classification.** Particle type is inferred from the
  penetration depth in the detector.

## Detector and Absorber Structure

The calorimeter consists of two functional blocks:

**Gamma block (rows 1–2).** Measures the electromagnetic component of a
shower core (e±, γ).

```
  ↑ (cosmic ray)
  │
  ├── 3 cm Pb          ← input filter
  ├── Row 1 (X, 4275 mm)
  ├── 1.5 cm Pb
  ├── Row 2 (Y, 4200 mm)
  ├── 22 cm Pb         ← main EM absorber
  │
  │   2.2 m air        ← decay gap (no detectors)
  │
```

**Hadron block (rows 3–8).** Measures hadrons (π, K, p) and penetrating
muons.

```
  ├── 13 cm Pb          ← hadron converter
  ├── 10 cm Fe
  ├── Row 3 (X, 1460 mm)
  ├── 10 cm Fe
  ├── Row 4 (Y, 1300 mm)
  ├── 13 cm Fe
  ├── Row 5 (X, 1010 mm)
  ├── 13 cm Fe
  ├── Row 6 (Y, 820 mm)
  ├── 13 cm Fe
  ├── Row 7 (X, 630 mm)
  ├── 10 cm Fe
  ├── Row 8 (Y, 470 mm)
  ├── 13 cm Fe
  └── (neutron counters — not operational)
```

**Physical consequences for the algorithms.**

| Particle type | Behavior | Where the track is searched |
|---|---|---|
| Muon (penetrating) | Crosses all 8 rows | Seeds in rows 7–8, track up to rows 1–2 |
| Hadron (p, π, K) | Interacts in the hadron block → shower | Seeds in rows 5–7, terminates in rows 3–4 |
| Electron/photon (EAS) | Fully absorbed in 22 cm Pb | Rows 1–2 only |
| Neutral particle | Converts in 13 cm Pb → starts in hadron block | Rows 3+ only, no signal in rows 1–2 |

**Multiple scattering in the road search.** As a particle traverses material,
its track undergoes multiple scattering proportional to

$$\theta_{\mathrm{rms}} \propto \frac{z \sqrt{x/X_0}}{p \beta c}$$

where $z$ is the particle charge, $X_0$ the radiation length of the material,
$x$ the material thickness, $p$ the momentum, and $\beta c$ the velocity.

In practice, the road-search radius grows with absorber thickness:

- Air: `base_radius` (3 channels)
- 10–13 cm Fe: `base_radius + 1`
- 22 cm Pb: `base_radius + 4` (strong scattering)

## Algorithms

### 1. Cluster Finder

A cluster is a contiguous run of channels with signal above threshold within a
single layer.

**Algorithm:**

1. Scan all channels of a layer.
2. If `channel_value >= hit_threshold`, start a cluster.
3. Extend right while channels remain active (respecting `max_gap`).
4. Store the cluster with its channels and amplitudes.

**Parameters:**

| Parameter | Value | Description |
|---|---|---|
| `hit_threshold` | 30 | Channel value threshold |
| `min_cluster_size` | 1 | Minimum channels in a cluster |
| `min_cluster_amplitude` | 50 | Minimum peak amplitude |
| `max_gap` | 0 | Allowed gap between channels |

**Cluster centroid** (amplitude-weighted center of mass):

$$c = \frac{\sum_{i} ch_i \cdot A_i}{\sum_{i} A_i}$$

where $ch_i$ is the channel number and $A_i$ the amplitude.

### 2. VETO Engine

Four fast checks flag pathological events. Events are flagged, not discarded.

**2.1 Saturation veto (EMI / HV fault).**

```
sat_frac = (number of channels ≥ 4000) / (total channels)
if sat_frac > 0.30 → is_saturated = True
```

Physics: ADC saturation from electromagnetic interference or a high-voltage
fault.

**2.2 Uniformity veto (flat noise).**

```
cv = std(nonzero_values) / mean(nonzero_values)
if cv < 0.15 and len(nonzero) > 50 → is_uniform_noise = True
```

Physics: a low coefficient of variation means all values are nearly equal —
this is noise, not a localized shower-core or penetrating-particle deposit.

**2.3 Emptiness veto (empty event).**

```
if total_energy < 100 or n_hits < 3 → is_empty = True
```

**2.4 Flash veto.**

```
max_pair_energy = max(energy[row_i] + energy[row_{i+1}])
if max_pair_energy / total_energy > 0.90 → is_flash = True
```

Physics: more than 90% of the energy concentrated in one pair of adjacent
rows indicates a local electrical disturbance, not a traversing particle.

### 3. Road Search Tracker

The track finder searches from a set of seed layers in the lower detector and
extends each candidate upward through the plane sequence of a given
projection.

**Important (current configuration).** The gamma rows (1–2) are excluded from
the direction fit and from the 2D track search, because they are separated
from the hadron block by 22 cm Pb + 2.2 m air. The road search therefore
operates on the hadron block only:

```yaml
road_search:
  seed_layers: [8, 7, 6, 5, 4]   # lower rows as seeds
  track_rows_x: [7, 5, 3]        # hadron X rows (gamma row 1 excluded)
  track_rows_y: [8, 6, 4]        # hadron Y rows (gamma row 2 excluded)
  min_track_layers: 2            # ≥2 points (rows 7/8 fire in ~30% due to absorption)
```

**Algorithm.** For each seed layer in `seed_layers`, and for each cluster in
that seed layer:

1. Initialize a track: `track.hits = [seed_cluster]`.
2. Move upward through the plane sequence:
   - Compute $\Delta Z = |z_{\mathrm{next}} - z_{\mathrm{prev}}|$.
   - Adaptive radius: $R = \text{base\_radius} + \Delta Z / 250$.
   - Extrapolate the position:
     $$x_{\mathrm{pred}} = x_1 + (x_2 - x_1) \cdot \frac{z_{\mathrm{next}} - z_1}{z_2 - z_1}$$
   - Find the nearest cluster within $\pm R$ channels.
   - If found, append to the track.
   - If not found, allow up to `max_layer_skips` skips.
3. If the track depth ≥ `min_track_layers`, store it.

Candidate tracks from different seeds that share most of their clusters are
merged (the longer track is kept), so that one physical track is not counted
multiple times.

### 4. Track Deduplication

Multiple seeds can produce overlapping tracks.

**Algorithm:**

1. For each pair of tracks, count shared clusters. A cluster is uniquely
   identified by `(layer_idx, projection, centroid)`.
2. If `overlap_fraction ≥ 50%`, the tracks overlap.
3. Among overlapping tracks, keep the one with:
   - greater `penetration_depth` (priority), then
   - lower `chi2_ndf` (tie-breaker).

**Overlap fraction:**

$$\text{overlap} = \frac{|clusters_i \cap clusters_j|}{\min(|clusters_i|, |clusters_j|)}$$

### 5. Quality Scoring

The final 0–100 score is a weighted sum:

$$Q = 0.35 \cdot P + 0.25 \cdot G + 0.20 \cdot T + 0.10 \cdot C + 0.10 \cdot E$$

where:

**P — Penetration score (0–100).**

$$P = \min\left(100, \frac{\text{depth}}{4} \cdot 100\right)$$

Track depth (maximum 4 layers per projection).

**G — Track goodness (0–100).**

$$G = \begin{cases}
100 & \text{if } \chi^2/\text{ndf} \leq 1.0 \\
80 - (\chi^2/\text{ndf} - 1) \cdot 10 & \text{if } 1.0 < \chi^2/\text{ndf} \leq 3.0 \\
\max(0, 60 - (\chi^2/\text{ndf} - 3) \cdot 20) & \text{if } \chi^2/\text{ndf} > 3.0
\end{cases}$$

**T — Topology score (0–100).**

- `n_active_layers ≥ 6` → 100
- `n_active_layers ≥ 4` → 80
- otherwise → 50

**C — Cluster purity (0–100).**

$$C = \min\left(100, \frac{n_{\mathrm{hits}}}{50} \cdot 100\right)$$

**E — Energy profile score (0–100).**

Reserved for future longitudinal-profile analysis (currently fixed at 70.0).

### 6. Particle Classification

The classifier infers particle type from penetration depth:

```python
all_layers = union(track.layers_hit for track in tracks)
has_top    = any(layer in [0, 1] for layer in all_layers)    # rows 1–2
has_bottom = any(layer in [6, 7] for layer in all_layers)    # rows 7–8

if best_track.penetration_depth >= 4 and has_top and has_bottom:
    → penetrating_muon
elif not has_top and has_bottom:
    → hadron_shower
elif has_top and not has_bottom:
    → em_shower
elif not has_top and any(layer in [2, 3, 4, 5]):
    → neutral_candidate
else:
    → unknown
```

**Note.** The classifier was written against an earlier row arrangement and
requires revision for the current hadron-block tracking scheme. It is not
used as a physics result in the current analysis; the angular analysis uses
only single 3+3 tracks (see `docs/3d_reconstruction.md`).

## Formulas

**Track chi-squared.**

$$\chi^2 = \sum_{i=1}^{N} (x_i - x_{\mathrm{pred},i})^2$$

where $x_i$ is the measured position in layer $i$ (cluster centroid) and
$x_{\mathrm{pred},i} = \text{slope} \cdot z_i + \text{intercept}$ is the
predicted position.

Degrees of freedom: $\text{ndf} = N - 2$.

**Linear regression.**

$$\text{slope} = \frac{\sum (z_i - \bar{z})(x_i - \bar{x})}{\sum (z_i - \bar{z})^2}$$

$$\text{intercept} = \bar{x} - \text{slope} \cdot \bar{z}$$

## Configuration

**Absorber section in `settings.yaml`.**

```yaml
absorber:
  materials:
    lead: { X0_mm: 5.6, lambda_I_mm: 170.0, Z: 82 }
    iron: { X0_mm: 17.6, lambda_I_mm: 168.0, Z: 26 }
    air:  { X0_mm: 304000.0, lambda_I_mm: 900000.0, Z: 7.3 }
  layers:
    before_row_1: { material: lead, thickness_mm: 30 }
    before_row_2: { material: lead, thickness_mm: 15 }
    before_row_3: { material: lead, thickness_mm: 220 }    # main EM absorber
    before_row_4: { material: air,  thickness_mm: 2200 }   # decay gap
    before_row_5: { material: iron, thickness_mm: 100 }
    before_row_6: { material: iron, thickness_mm: 130 }
    before_row_7: { material: iron, thickness_mm: 130 }
    before_row_8: { material: iron, thickness_mm: 100 }
```

**Event quality section in `settings.yaml`.**

```yaml
event_quality:
  enabled: true
  # Cluster finding
  hit_threshold: 30
  min_cluster_size: 1
  min_cluster_amplitude: 50
  # VETO thresholds
  veto:
    saturation_fraction: 0.30
    saturation_value: 4000
    uniformity_cv_max: 0.15
    min_total_energy: 100
    min_total_hits: 3
  # Road search
  road_search:
    seed_layers: [8, 7, 6, 5, 4]
    track_rows_x: [7, 5, 3]
    track_rows_y: [8, 6, 4]
    min_track_layers: 2
    base_radius_channels: 3
    max_layer_skips: 1
  # Quality thresholds
  quality_classes:
    excellent: 85
    good: 65
    poor: 40
  # Logging
  logging:
    level: INFO              # INFO / DEBUG / TRACE
    dir: logs
    trace_road_search: false
  # Parallel processing
  parallel:
    n_workers: 4             # 0 = auto (all CPUs)
    chunk_size: 100          # events per worker
```

## CLI Usage

**Scan with quality analysis (parallel).**

```bash
python main.py scan
```

Output:

```
============================================================
 SUMMARY:
============================================================
    Total events:         291,358
    Passed filter:        240,780
    Rejected:              50,578
    Rejection reasons:
       no_bottom_signal: 50,577
       low_energy: 1
```

**View events with quality information.**

```bash
# All events
python main.py info

# Specific events
python main.py info --ids 0,100,500,1234

# Only EXCELLENT events
python main.py info --min-quality EXCELLENT

# GOOD and above
python main.py info --min-quality GOOD

# POOR and above (includes almost everything)
python main.py info --min-quality POOR
```

Example output:

```
      ID     Energy   Hits  Layers     Quality  Score  Tracks          Particle  Time                  Status
------------------------------------------------------------------------------------------------------------------------
       0      12450     85       8   EXCELLENT   88.2       5          em_shower  2021-01-05 00:01:23   passed
     100       8720     42       6   EXCELLENT   88.2       5          em_shower  2021-01-05 00:15:45   passed
     500      25600    120       8        GOOD   82.2       1          em_shower  2021-01-05 01:23:11   passed
    1234      15800     67       7        POOR   55.3       2  hadron_shower     2021-01-05 03:45:02   passed
Total shown: 4

Quality statistics:
    EXCELLENT: 2
         GOOD: 1
         POOR: 1

Particle type statistics:
     em_shower: 3
  hadron_shower: 1
```

**Inspect raw / processed / quality data for a single event.**

```bash
# Raw arrays to terminal
python main.py inspect --ids 0 --stage raw --output terminal

# Assembled (processed) data to a text file
python main.py inspect --ids 5 --stage processed --output txt

# Quality report (clusters, tracks, VETO) to JSON
python main.py inspect --ids 10 --stage quality --output json
```

## Results on Real Data

**Test on 291 358 events (2021, 2023).**

```
Total events:         291,358
Passed filter:        240,780  (82.6%)
Rejected:              50,578  (17.4%)
Analyzed:             196,507
Rejection reasons:
   no_bottom_signal: 50,577
   low_energy: 1
```

**Example quality analysis.**

| Event ID | Clusters | Tracks | Quality | Score | Particle Type | Best θ |
|---|---|---|---|---|---|---|
| 0 | 33 (X=13, Y=20) | 5 | EXCELLENT | 88.2 | em_shower | 39.0° |
| 1 | 33 (X=16, Y=17) | 5 | EXCELLENT | 88.2 | em_shower | 28.9° |
| 2 | 20 (X=9, Y=11) | 1 | GOOD | 82.2 | em_shower | 29.1° |
| 9999 (test) | 34 (X=20, Y=14) | 10 | EXCELLENT | 97.0 | penetrating_muon | 45.8° |

**Performance.**

- Single-threaded: ~1–5 ms per event
- Parallel (4 workers): ~4× speedup
- Total time for 196k events: ~15–20 minutes

## Quality Class Reference

| Class | Score | Criteria | Use |
|---|---|---|---|
| EXCELLENT | 85–100 | depth ≥ 4, χ²/ndf ≤ 1.5, 8 active layers | Scientific analysis, precise angles |
| GOOD | 65–84 | depth ≥ 3, χ²/ndf ≤ 3.0 | Main dataset |
| POOR | 40–64 | depth = 2–3 or high χ² | Exploratory studies, not for science |
| NOISE | 0–39 | VETO flags, no tracks | Statistics only |

## Particle Type Reference

| Type | Criterion | Physics |
|---|---|---|
| penetrating_muon | Track through ≥6 layers (top + bottom) | High-energy muon, does not interact |
| hadron_shower | Terminates in hadron block (rows 3–6) | Proton/nucleus, nuclear interaction |
| em_shower | Rows 1–2 only (before 22 cm Pb) | Electron/photon, EM shower |
| neutral_candidate | Not in top rows, present in middle rows | γ or neutron, converted in Pb |
| unknown | Does not match any criterion | Ambiguous event, requires manual review |
```

---
