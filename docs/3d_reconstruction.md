# 3D Track Reconstruction

Module `src/tracker.py` reconstructs three-dimensional particle tracks from
the per-projection 2D tracks produced by the road-search finder (see
`event_quality.py`). It combines the X and Y projections into 3D tracks,
fits them with a weighted least-squares line, and locates shower vertices.

## Table of Contents

- [Overview](#overview)
- [Hybrid X↔Y Matching](#hybrid-xy-matching)
- [Weighted Least Squares Fitting](#weighted-least-squares-fitting)
- [Shower Vertex Finding](#shower-vertex-finding)
- [Physics: Landau Fluctuations](#physics-landau-fluctuations)
- [Configuration](#configuration)
- [Worked Examples](#worked-examples)
- [Known Limitations](#known-limitations)
- [Validation Results](#validation-results)

## Overview

The 3D reconstruction proceeds in four stages:

1. **Hybrid X↔Y matching.** Combine the X and Y 2D tracks into candidate 3D
   tracks, using energy-ratio matching for simple cases and geometric
   proximity for complex cases.
2. **3D hit construction.** Build 3D points from the matched 2D tracks,
   interpolating the unmeasured coordinate from the partner track.
3. **Weighted least-squares fit.** Fit a straight line to the 3D hits,
   weighting each point by the multiple scattering accumulated above it.
4. **Vertex finding.** Locate the shower vertex using a particle-type-specific
   strategy.

**Gamma-row exclusion.** The gamma rows (1–2) are excluded from the direction
fit. They are separated from the hadron block by 22 cm Pb + 2.2 m air, so a
gamma-row hit does not lie on the straight line defined by the hadron-block
segment. Including them would impose a spurious tilt on the reconstructed
direction. This is configured via `tracking.fit_exclude_row_idx: [0, 1]`.

## Hybrid X↔Y Matching

### The problem with pure energy-ratio matching

The original (MATLAB-style) approach sorts the X and Y tracks by energy and
pairs them by rank:

```python
x_sorted = sorted(x_tracks, key=energy, reverse=True)
y_sorted = sorted(y_tracks, key=energy, reverse=True)
# Pair by rank: X[0]↔Y[0], X[1]↔Y[1], ...
```

Physical motivation: a higher-energy particle leaves proportionally more
ionization in both projections.

**Problem: Landau fluctuations.** When a charged particle traverses matter,
it loses energy in random, discrete amounts (knocking out electrons). This is
described by the Landau distribution:

```
For a 10 GeV muon through 1 cm of lead:
  - Mean energy loss: 2 MeV
  - But it can be: 0.5 MeV (rare event)
  - Or: 15 MeV (delta electron knocked out a lot of ionization)
```

Practical consequence:

```
Real event with 2 muons:
  Muon A: E = 5 GeV
  Muon B: E = 3 GeV

Due to Landau fluctuations, the energy ranks can FLIP:
  X-projection: E_X(A) = 4800, E_X(B) = 3200  → rank: A > B  ✓
  Y-projection: E_Y(A) = 2900, E_Y(B) = 3100  → rank: B > A  ✗

MATLAB-style matching would pair INCORRECTLY:
  X[0] (A) ↔ Y[0] (B)  ← WRONG
  X[1] (B) ↔ Y[1] (A)  ← WRONG
```

Special case — delta electrons: roughly 1 in 100 muons knocks out a delta
electron with E > 10 keV:

```
  - Normal cluster: 300 ADC counts
  - Cluster with delta electron: 3000 ADC counts (×10)
```

The energy ranks can flip entirely.

### The hybrid solution

**Step 1: MATLAB-style matching (fast, for obvious cases).**

```python
for i in range(min(len(x), len(y))):
    x_energy = compute_energy(x[i])
    y_energy = compute_energy(y[i])
    ratio = max(x_energy, y_energy) / min(x_energy, y_energy)
    if ratio <= 3.0:
        # MATLAB-style worked — keep the pair
        matches.append((x[i], y[i], 'energy'))
    else:
        # Ratio too large — need geometry
        geometry_candidates.append((x[i], y[i]))
```

**Step 2: Geometric proximity (for complex cases).**

```python
# Extrapolate all tracks to Z_COMMON = 2000 mm (detector midpoint)
for x_track in geometry_candidates_x:
    x_mm = x_track.slope * Z_COMMON + x_track.intercept
for y_track in geometry_candidates_y:
    y_mm = y_track.slope * Z_COMMON + y_track.intercept

# Build a cost matrix and apply the Hungarian algorithm
cost_matrix[i, j] = sqrt(x_points[i]**2 + y_points[j]**2)
x_indices, y_indices = linear_sum_assignment(cost_matrix)
```

Physical motivation for the geometric approach:

- If the X and Y tracks belong to the same particle, their extrapolations
  meet at nearby points in the XY plane.
- This approach does not depend on energy, so it is robust against Landau
  fluctuations.
- It uses only the track geometry (slope, intercept).

**Comparison of approaches.**

| Criterion | MATLAB-style | Geometric proximity | Hybrid |
|---|---|---|---|
| Single particles | ✓ | ✓ | ✓ |
| Dense showers | ✗ | ✓ | ✓ |
| Robust to Landau fluctuations | ✗ | ✓ | ✓ |
| Robust to delta electrons | ✗ | ✓ | ✓ |
| Speed | O(N log N) | O(N³) | O(N log N + N³) |

## Weighted Least Squares Fitting

### The problem with ordinary least squares

The original approach treats all 3D hits as equally precise:

```python
coeffs_x = np.polyfit(z_vals, x_vals, 1)  # all weights = 1
```

**Problem.** The X and Y layers alternate in height:

```
X-layers: 4275, 1460, 1010, 630 mm (rows 1, 3, 5, 7)
Y-layers: 4200, 1300,  820, 470 mm (rows 2, 4, 6, 8)
```

When constructing 3D hits, interpolation is used:

```
For an X-hit (z = 1460 mm):
  x_mm = measured (from row 3)
  y_mm = interpolated from the Y-track (no Y-layer at z = 1460 mm)
```

The interpolated coordinate has a larger uncertainty, but ordinary least
squares does not distinguish it from a measured coordinate.

### The solution: weighted least squares

```python
# Weights for each point:
# X-hit: measured X (w_x = 1.0), interpolated Y (w_y = 0.1)
# Y-hit: interpolated X (w_x = 0.1), measured Y (w_y = 1.0)
weights_x = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]  # for X-hits
weights_y = [0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0]  # for Y-hits

coeffs_x = np.polyfit(z_vals, x_vals, 1, w=np.sqrt(weights_x))
coeffs_y = np.polyfit(z_vals, y_vals, 1, w=np.sqrt(weights_y))
```

**Chi-squared normalization by cluster size.**

Without normalization, χ² grows with cluster size:

```
Single muon:    χ² = 453 mm²   (looks bad)
Wide shower:    χ² = 5000 mm²  (looks even worse)
```

Normalization by a characteristic cluster size:

```python
SIGMA_TYPICAL_MM = 200.0  # mm (typical cluster size)
chi2_norm = chi2_raw / (SIGMA_TYPICAL_MM ** 2)
chi2_ndf = chi2_norm / ndf  # dimensionless, ~1 for a good track
```

Interpretation:

- `chi2_ndf ≈ 1` → the track is well described by a straight line
- `chi2_ndf > 10` → the track is rejected

## Shower Vertex Finding

The vertex-finding strategy is particle-type-specific.

**Penetrating muon.** Extrapolate all tracks to a height of 15 km, cluster
the extrapolation points (DBSCAN-like), and take the mean position of each
cluster as a vertex. Result: a vertex at ~15 km, corresponding to the muon
production point.

**Hadron shower.** The vertex is the point of the first nuclear interaction,
not the point where tracks converge. Algorithm:

1. Ignore rows 1–2 (gamma block, before 22 cm Pb).
2. Find the first layer with ≥3 clusters in rows 3–8.
3. If none is found, look for a sharp energy rise (>1000 and >3× the
   previous layer).
4. Fallback: the first layer with signal in rows 3–8.

Result: a vertex in row 3 (z ≈ 1.46 m) — the first nuclear interaction.

**EM shower / neutral particle.** Locate the vertex from the energy-deposition
profile (energy maximum).

## Physics: Landau Fluctuations

**Calculation for ADRON-55.**

Ionization chamber:

```
Gas: argon + CO₂ (or similar)
Gas gap thickness: ~1–2 cm
Pressure: ~1 atm
Particle: muon with E = 10 GeV
```

Mean energy loss (Bethe-Bloch formula):

```
dE/dx ≈ 2 keV/cm (for Ar at 1 atm)
E_deposited = dE/dx × thickness = 2 keV/cm × 2 cm = 4 keV
```

Landau fluctuations:

```
ξ ≈ 0.5 keV (for 2 cm Ar)
MPV ≈ 3.5 keV (Most Probable Value)
FWHM ≈ 4 × ξ ≈ 2 keV
```

Relative fluctuation:

```
σ/E ≈ FWHM / (2.35 × E) ≈ 2 / (2.35 × 4) ≈ 21%
```

**Practical consequences.**

For a single layer:

```
Muon E = 10 GeV crossing row 3 (X-projection):
  - Mean cluster energy: 300 ADC counts
  - Due to Landau: E_min ≈ 150, E_max ≈ 900 (factor of 6)
```

For a full track (sum over 4 X-layers):

```
Without Landau: E_total = 300 + 300 + 300 + 300 = 1200
With Landau:    E_total = 150 + 900 + 300 + 450 = 1800   (+50%)
                E_total = 450 + 150 + 200 + 600 = 1400   (+17%)
                E_total = 100 + 200 + 150 + 250 = 700    (-42%)
Spread: 700 – 1800 (factor of 2.5)
```

**Why the 120 mm channel width does not protect against Landau fluctuations.**
Landau fluctuations occur along the particle trajectory (in the gas), not
across it.

```
        ┌─────────────────────────────────┐  ← strip, 120 mm wide
        │                                 │     (does NOT affect Landau)
        │    ╲                            │
        │     ╲  particle track           │
        │      ╲                          │
        │───────╲─────────────────────────│  ← gas thickness (1–2 cm)
        │        ╲                        │     (THIS determines Landau)
        └─────────────────────────────────┘
```

The only mitigations are:

- Summing over many layers: σ decreases by a factor of √N.
- The geometric approach: does not depend on energy at all.
- Delta-electron rejection: a cluster-energy filter.

## Configuration

```yaml
tracking:
  # Basic parameters
  radius: 3
  threshold: 5
  h_max_extrapolation: 500000  # 500 m

  # Cosmic-ray vertex extrapolation
  cosmic_ray:
    extrapolation_mode: "auto"  # auto | fixed
    auto_heights_m:
      penetrating_muon: 15000   # 15 km
      hadron_shower: 8000       # 8 km
      em_shower: 10000          # 10 km
      neutral_candidate: 8000
      unknown: 10000
    fixed_height_m: 10
    vertex_clustering_eps_m: 500  # vertex clustering radius
    vertex_min_tracks: 2

  # Hybrid X↔Y matching
  hybrid_matching:
    # Energy-ratio threshold: below → MATLAB-style, above → geometry
    energy_ratio_threshold: 3.0
    # Extrapolation height for geometric matching (mm)
    # Optimal: detector midpoint
    z_common_mm: 2000.0
    # Max distance for geometric matching (mm)
    geometry_max_distance_mm: 5000.0
    # Method priority order
    method_priority: ["energy", "geometry"]

  # 3D fitting
  fitting:
    # Characteristic cluster size for χ² normalization (mm)
    sigma_typical_mm: 200.0
    # Track rejection threshold on χ²/ndf
    chi2_ndf_threshold: 10.0
    # Minimum number of 3D hits for fitting
    min_hits_3d: 3
    # Use Weighted Least Squares
    weighted_least_squares: true
    # WLS weights
    weights:
      real_hit: 1.0
      interpolated_hit: 0.1

  # Vertex finding
  vertex_finding:
    # Hadron showers: minimum clusters in a layer for shower start
    hadron_shower_min_clusters: 3
    hadron_shower_energy_threshold: 1000
    hadron_shower_energy_ratio: 3.0
    # Ignore rows 1–2 (gamma block) when finding the hadron vertex
    ignore_gamma_block: true

  # Gamma rows (row 1,2 = idx 0,1) are excluded from the direction fit:
  # they are separated from the hadron block by 22 cm Pb + 2.2 m air.
  fit_exclude_row_idx: [0, 1]
```

## Worked Examples

**Example 1: single vertical muon.**

```
Input: 1 X-track (θ = 2°), 1 Y-track (θ = -4°)
Hybrid algorithm:
  Match (energy): X#0 (E = 14000) ↔ Y#0 (E = 13500) [ratio = 1.04]
Result:
  Track3D #0 (energy): 7 hits, zenith = 3°, azimuth = 207°, chi2/ndf = 0.01
  Vertex #0: (0, 0, 15000000) mm = 15.0 km
```

**Example 2: hadron shower (event ID = 0).**

```
Input: 5 X-tracks, 5 Y-tracks
Hybrid algorithm:
  Match (energy): X#0 ↔ Y#0 [ratio = 1.2]
  Match (energy): X#1 ↔ Y#1 [ratio = 1.5]
  Match (geometry): X#2 ↔ Y#3 [cost = 1200 mm]  ← ratio was 5.2
Result:
  2 × Track3D with chi2/ndf < 1
  Vertex #0: (0, 0, 1460) mm = 1.46 m (row 3)
```

## Known Limitations

- **Energy calibration is not finalized.** The `mip_in_mv` value (0.39)
  disagrees with the documentation value (390 mV) by a factor of ~1000 in
  absolute scale. The energy spectrum is not used as a physics result until
  this is resolved.
- **Two-point tracks / angular resolution.** Only ~30% of particles reach the
  lower rows, so most tracks are two-point (quantized slope). Full azimuthal
  isotropy is not achievable with this geometry; this is a physical limit of
  the detector, not a software bug.
- **Noisy multi-track events ("stubs").** Extending the seeds to the lower
  rows produced rare (0.2% of events) but "explosive" multi-track events
  (~12 short 2-point tracks from scattered shower clusters). These are not
  used in the angular analysis (only single 3+3 tracks are used). Cleaning
  shower axes is future work.
- **Particle classifier.** Tied to the earlier row arrangement (gamma,
  penetration); after the row-composition change it requires revision. It is
  not used as a physics result.
- **Monte Carlo comparison (Hadr55/Geant4).** A slot is reserved in the paper
  structure; to be filled when simulation access is available.

## Validation Results

The azimuthal uniformity of single penetrating 3+3 tracks was used to
validate the reconstruction. The progression of the uniformity statistic
(χ²/ndf against a uniform distribution) is:

| Stage | χ²/ndf (n_rows ≥ 6) |
|---|---|
| Original (gamma in fit, old stitching) | ~1324 |
| After stitching fix | ~85 → 42 |
| After road search (lower rows) | ~30 |
| After track-based alignment (converged) | **26** |

The residual (~26, not perfect isotropy) is explained physically: the
difference in the X- and Y-projection lever arms on 3 points, plus absorption
(only ~30% of particles reach the lower rows). This is the limit of the
detector geometry, not a bug.

The track-based alignment (`diag_residuals.py`, additive, on clean 3+3 tracks
only) converged in 2 iterations: mean(slope_x)/mean(slope_y) went from
−0.25/−0.27 to −0.012/−0.002. The mean tilt is removed.

Final alignment (example; re-measure on your own data):

```yaml
row_alignment_mm:
  0: 0.0
  1: 0.0
  2: -214.2
  3: -784.3
  4: -152.3
  5: -691.3
  6: 10.4
  7: -544.4
```
```

---
