# Event Quality: Clustering, Veto Conditions, Road Search and Scoring

Module: `src/event_quality.py`.

This document describes the stage that turns assembled plane profiles into
clusters, rejects pathological events, builds two-dimensional track candidates
in each projection, and assigns a quality grade. It is the stage between
`src/preprocessing.py` (section assembly) and `src/tracker.py` (three-dimensional
reconstruction).

All parameter values quoted here are the defaults of
`config/settings.yaml.example`, which are the values used for the published
results.

---

## 1. Input and plane ordering

`preprocessing.unify_layers` returns two lists ordered **from the bottom of the
calorimeter upward**:

| List | Planes (bottom to top) | Plane indices (0 = top) |
| ---- | ---------------------- | ----------------------- |
| `X_layers` | 7, 5, 3, 1 | 6, 4, 2, 0 |
| `Y_layers` | 8, 6, 4, 2 | 7, 5, 3, 1 |

Throughout the code `layer_idx` is the zero-based plane index counted from the
top of the stack, so `layer_idx = 0` is plane 1 (the uppermost gamma plane) and
`layer_idx = 7` is plane 8 (the lowest active plane). Plane heights are taken
from `geometry.heights` by this index.

---

## 2. Cluster finding

A cluster is a contiguous run of channels whose amplitude reaches
`event_quality.hit_threshold` (30 ADC counts). A run is retained if it contains
at least `min_cluster_size` channels (1) and its maximum amplitude reaches
`min_cluster_amplitude` (50 ADC counts). Gaps inside a run are not bridged in
the released configuration (`max_gap = 0`).

Each cluster is summarized by two quantities:

- the amplitude-weighted centroid, used as the transverse position of the signal
  in that plane,

  $$c = \frac{\sum_i n_i a_i}{\sum_i a_i}$$

  with $n_i$ the channel index and $a_i$ its amplitude;
- the summed amplitude, used as a measure of the deposited energy.

Clusters are the unit on which everything downstream operates: the track finder
links them across planes, and the response profiles are built from them.

The plane occupancy — the fraction of events in which a plane carries any signal
— is insensitive to the threshold over the range 5–30 ADC counts. Where a plane
is hit, the signal stands well above threshold, so the threshold governs which
channels enter a cluster but not whether the plane is counted as active. This
was verified with `analysis/diag_clusters_bottom.py`.

Implementation: `find_clusters_in_layer`, `find_all_clusters`.

---

## 3. Veto conditions

Four conditions remove events that are not candidates for tracking, before any
track finding is attempted. They are deliberately loose: their role is to discard
pathological events cheaply, not to shape the physics sample.

| Condition | Test | Threshold | Physical origin |
| --------- | ---- | --------- | --------------- |
| Saturation | fraction of channels at the maximum digitized value | > 0.30 at 4000 ADC | electromagnetic interference or readout fault |
| Uniformity | coefficient of variation of the non-zero amplitudes | CV < 0.15 with more than 50 non-zero channels | incoherent baseline noise |
| Emptiness | total deposition or hit count | < 100 ADC or < 3 hits | empty trigger |
| Flash | energy concentrated in one adjacent plane pair | > 90% of the total in one pair, with signal in at most three planes | localized electrical disturbance |

The uniformity test is applied only if the saturation flag is not already set,
so that a saturated event is reported under a single, unambiguous cause.

An event carrying any veto flag is graded `NOISE` and does not enter
reconstruction.

Implementation: `veto_engine`.

---

## 4. Road search

The track finder is a classical seeded road search operating independently in
each projection.

### 4.1 Plane sequence and seeds

Track finding is confined to the **hadron block**. The gamma planes are
separated from it by 220 mm of lead and a 2.2 m air gap, so the footprint a
particle leaves in the gamma block does not in general lie on the straight line
defined by its hadron-block segment.

| Setting | Value | Meaning |
| ------- | ----- | ------- |
| `track_rows_x` | `[7, 5, 3]` | X plane sequence, bottom to top |
| `track_rows_y` | `[8, 6, 4]` | Y plane sequence, bottom to top |
| `seed_layers`  | `[8, 7, 6, 5, 4]` | planes allowed to start a track |
| `min_track_layers` | 2 | minimum planes for a retained candidate |
| `max_layer_skips` | 1 | planes that may be missed without ending the track |
| `base_radius_channels` | 3 | base half-width of the search window |

Seeds are intersected with the projection's plane sequence, so plane 3 is not
used as a seed. The deepest planes are deliberately included: at the analysis
thresholds they yield a usable cluster in only about a third of events, and
unless they are permitted to seed a track they are never reached by a search
that only extends upward — and therefore never supply the third point that lifts
a track out of the two-point regime.

### 4.2 Extension

Starting from a seed cluster the finder walks upward through the plane sequence.
The position in the next plane is predicted by linear extrapolation from the last
two hits,

$$c_\mathrm{pred} = c_1 + (c_2 - c_1)\,\frac{z_\mathrm{next} - z_1}{z_2 - z_1},$$

or, for a track with a single hit, by the position of that hit. The nearest
cluster within the search window is attached. The window widens with the vertical
gap between planes,

$$R = R_0 + \left\lfloor \frac{\Delta z}{250\ \mathrm{mm}} \right\rfloor,$$

to absorb the larger extrapolation uncertainty across the thicker absorber
sections. A limited number of planes may be skipped, so that a plane which
happens not to register a cluster does not terminate an otherwise good track.

### 4.3 Deduplication

Candidates from different seeds that share at least half of their clusters are
treated as the same physical track. The survivor is the one with the greater
penetration depth; at equal depth, the one with the smaller $\chi^2/\mathrm{ndf}$.
Track identifiers are renumbered afterwards.

Implementation: `road_search_projection`, `road_search_all`, `fit_track`,
`deduplicate_tracks`.

---

## 5. Quality score

Events that pass the veto are graded on a 0–100 scale built from five terms:

| Term | Weight | Definition |
| ---- | ------ | ---------- |
| Penetration | 0.35 | planes on the best track, normalized to four per projection |
| Track goodness | 0.25 | piecewise function of $\chi^2/\mathrm{ndf}$: 100 at $\leq 1$; 80 falling to 60 between 1 and 3; then falling to 0 at 6 |
| Topology | 0.20 | 100 for at least six active planes, 80 for at least four, 50 otherwise |
| Cluster purity | 0.10 | hits in clusters relative to a reference of 50 |
| Energy profile | 0.10 | fixed placeholder value of 70 |

The best track is the one with the greatest penetration depth, ties broken by the
smaller $\chi^2/\mathrm{ndf}$.

| Grade | Score |
| ----- | ----- |
| EXCELLENT | ≥ 85 |
| GOOD | ≥ 65 |
| POOR | ≥ 40 |
| NOISE | < 40, or any veto flag |

The score and its weights are heuristic. They serve only to remove obviously
pathological events before reconstruction; no physics result depends on the
precise weighting. Events graded GOOD or better enter three-dimensional
reconstruction.

Implementation: `compute_quality_score`.

---

## 6. Particle classification

`classify_particle` assigns one of `penetrating_muon`, `hadron_shower`,
`em_shower`, `neutral_candidate` or `unknown` from the pattern of planes carrying
clusters.

**This classifier is not used for any published result.** It was written against
an earlier arrangement of the planes and needs revision for the hadron-block
tracking adopted here. With the direction fit restricted to the three planes of
each hadron-block projection, its penetrating-muon category can no longer be
populated at all. The labels remain in the catalogue for diagnostic purposes
only.

---

## 7. Catalogue fields

Running `python main.py scan` with `event_quality.enabled: true` adds the
following fields to the HDF5 catalogue:

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `quality_score` | float32 | 0–100 |
| `quality_class` | string | EXCELLENT / GOOD / POOR / NOISE |
| `n_tracks` | int16 | two-dimensional track candidates found |
| `particle_type` | string | classifier label (not used in the paper) |
| `best_track_theta` | float32 | zenith angle of the best track, degrees |
| `best_track_chi2` | float32 | $\chi^2/\mathrm{ndf}$ of the best track |

Events excluded by the good-run selection keep their catalogue entry with a
`bad_run` status, so that `global_id` remains stable and an event can still be
loaded by identifier.

---

## 8. Command-line access

```bash
# Full scan with quality analysis (multiprocessing)
python main.py scan

# Quality report for one event, as JSON
python main.py inspect --ids 10 --stage quality --output json

# Quality report in the terminal
python main.py inspect --ids 10 --stage quality --output terminal
```

Detailed per-event output is written to `logs/event_quality.log` rather than the
terminal.
