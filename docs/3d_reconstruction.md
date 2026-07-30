# Three-Dimensional Reconstruction

Module: `src/tracker.py`.

Two-dimensional track candidates found independently in the X and Y projections
(`src/event_quality.py`) are paired, converted into three-dimensional hits, and
fitted with a straight line. This document describes that chain and the
conventions it relies on.

---

## 1. Coordinate conventions

All positions are in millimetres. A cluster centroid, expressed as a channel
number, is converted by `preprocessing.channel_to_mm`:

$$x = \left(c - \frac{N_p}{2}\right) w - \delta_p,$$

where $c$ is the centroid, $w = 120$ mm the channel pitch, $N_p$ the number of
channels of plane $p$, and $\delta_p$ the alignment offset of that plane.

Centring each plane by **its own width** is not optional. The planes differ in
width — 50, 69, 48 and 72 channels for the gamma-X, gamma-Y, hadron-X and
hadron-Y planes — so referring them all to one common centre would place their
geometric centres at different physical positions and impose a width-dependent
offset between planes, which the fit reads as a tilt even for a perfectly
vertical track.

The offsets $\delta_p$ are stored in `geometry.row_alignment_mm` and are derived
from the data; see Section 5.

---

## 2. Excluding the gamma planes

Planes 1 and 2 (`layer_idx` 0 and 1) are excluded from the direction fit through
`tracking.fit_exclude_row_idx`. They are separated from the hadron block by
220 mm of lead and a 2.2 m air gap, which breaks the trajectory of a traversing
particle: their hits are not two well-aligned points at a long lever arm but two
systematically displaced points that impose a large, spurious tilt. Including
them strongly biases the azimuthal distribution; removing them restores it.

The same restriction applies in the road search through `track_rows_x` and
`track_rows_y`.

---

## 3. Matching the projections

`match_xy_tracks_hybrid` pairs X and Y tracks in two stages.

**Energy ranking.** A particle deposits comparable ionization in the X and Y
chambers it crosses, so the two projections of one particle should carry similar
total amplitude. Tracks in each projection are ordered by summed cluster
amplitude and paired in rank. A pair is accepted when

$$\frac{\max(E_X, E_Y)}{\min(E_X, E_Y)} \leq \texttt{energy\_ratio\_threshold} = 3.$$

**Geometric assignment.** Pairs failing the energy test, together with any
unpaired remainder, are assigned by proximity. Each track is extrapolated to a
common height `z_common_mm` (2000 mm, near the middle of the hadron block) and
the assignment minimizing the total distance is found with the Hungarian
algorithm (`scipy.optimize.linear_sum_assignment`). Pairs further apart than
`geometry_max_distance_mm` (5000 mm) are discarded.

The ratio threshold accommodates Landau fluctuations and delta electrons, which
routinely make the two projections of one particle differ in amplitude by a
factor of a few.

For the angular results of the paper only events yielding a **single** track in
each projection are used, so the details of the multi-track assignment do not
enter them.

---

## 4. Three-dimensional hits and the fit

For each hit of the X track, the measured coordinate is $x$; the orthogonal
coordinate $y$ at that depth, which the plane does not measure, is taken from the
fitted line of the partner Y track, and symmetrically for the Y hits. Each hit
therefore carries one measured and one interpolated coordinate, with base weights
1.0 and 0.1 respectively.

The line $x = s_x z + b_x$, $y = s_y z + b_y$ is fitted by weighted least
squares. The weight of each point is further scaled by the multiple scattering
accumulated above its plane,

$$w_p \propto \left(\Sigma X / X_0\right)_p^{-1},$$

so that planes deeper in the iron, where the direction has been perturbed more,
carry less weight — without any assumption about the particle momentum. The
cumulative radiation lengths come from `physics.get_cumulative_rad_length`,
which reads the absorber structure from the configuration.

The fit quality is reported as a dimensionless quantity, the weighted residual
sum normalized by a characteristic cluster scale and by the degrees of freedom:

$$\chi^2_{\mathrm{3D}} = \frac{\chi^2_x + \chi^2_y}{\sigma_\mathrm{typ}^2\,\mathrm{ndf}},
\qquad \sigma_\mathrm{typ} = 200\ \mathrm{mm},
\qquad \mathrm{ndf} = \max(1,\ 2N_\mathrm{hits} - 4).$$

Tracks with $\chi^2_{\mathrm{3D}} > 10$ or fewer than three hits are rejected
(`tracking.fitting.chi2_ndf_threshold`, `min_hits_3d`).

The arrival direction follows from the fitted slopes:

$$\theta = \arctan\sqrt{s_x^2 + s_y^2},
\qquad \varphi = \operatorname{atan2}(-s_y, -s_x).$$

`tracker.py` reports $\varphi$ in the **detector frame**. The offset of the
detector X axis from north (`location.detector_angle_offset`) is added later, in
`physics.altaz_to_radec`, when the direction is converted to equatorial
coordinates.

---

## 5. Track-based alignment

Residual per-plane offsets — from the mechanical mounting, from inactive channels
that bias a centroid, from the section assembly — displace the effective centre
of a plane by a fraction of its width. A constant offset in a single plane tilts
every track crossing it, and the tilt accumulates over the lever arm of the
stack. Left uncorrected these offsets are the dominant systematic in the
reconstructed direction.

They are removed through $\delta_p$, obtained from the tracks themselves. For a
cosmic-ray flux symmetric in azimuth, the mean transverse position of the tracks
crossing a plane must coincide with the geometric centre of that plane, so the
mean position measures the residual offset:

$$\delta_p \leftarrow \delta_p + \langle r_p \rangle.$$

Two features are essential to convergence:

- the procedure is applied **only to fully penetrating tracks** with three active
  planes in each projection. A global tilt is absorbed into the straight-line fit
  of each individual track and is therefore invisible in the per-track residuals;
  only for a sample with continuous slope does the mean-position criterion pin
  the offsets down. On quantized two-point tracks it does not converge reliably.
- it is **iterative and additive**: reconstruct with the current offsets, measure
  the mean residual in each plane, add it to that plane's offset, repeat. In
  practice two passes suffice — the mean reconstructed slope in both projections
  falls from about $-0.25$ with no alignment to about $-0.01$.

Run with `analysis/diag_residuals.py`. The resulting offsets are those in
`geometry.row_alignment_mm`:

| Plane | Projection | $\delta_p$ (mm) |
| ----- | ---------- | --------------- |
| 3 | X | −214.2 |
| 4 | Y | −784.3 |
| 5 | X | −152.3 |
| 6 | Y | −691.3 |
| 7 | X | +10.4 |
| 8 | Y | −544.4 |

The offsets are effective quantities: they absorb mechanical displacement,
dead-channel bias in the centroids, and any residual convention of the section
assembly alike. Some are sizeable — up to about 0.8 m in the Y planes — and
should not be read as a literal mechanical displacement.

---

## 6. Vertex finding

Vertex reconstruction is a diagnostic and visualization feature; no published
result depends on it.

For `penetrating_muon` and `hadron_shower` labels, tracks are extrapolated to a
type-dependent height (`tracking.cosmic_ray.auto_heights_m`) and clustered in the
transverse plane with a simple single-linkage algorithm of radius
`vertex_clustering_eps_m` (500 m). Each cluster with at least two tracks becomes
a vertex, positioned at the mean of the extrapolated points, with the spread
reported as a quality measure.

For the remaining labels a single vertex is estimated from the longitudinal
energy profile: the first plane at which the deposition rises sharply above the
preceding one, or crosses a tenth of the maximum, is taken as the shower start,
and the best track is evaluated at that height.

---

## 7. Resolution limits

Most tracks have only two active planes per projection. With the lever arm of
0.83 m and the 120 mm granularity, the reconstructed slope of a two-point track
is quantized in steps of $w/L \approx 0.14$, which is 5–8° in angle near the
vertical. Only fully penetrating tracks with three active planes in each
projection have a continuous direction, and only that subset is used for the
angular results.

This is a property of the geometry, not of the algorithm. A larger data set
improves the statistics of the clean sample but not its per-track resolution.

---

## 8. Command-line access

```bash
# 3D reconstruction of selected events, with CSV output
python main.py reconstruct --ids 0,10,100 --save-csv output/reco.csv

# All GOOD events
python main.py reconstruct --min-quality GOOD --save-csv output/reco.csv

# Interactive 3D view of one event
python main.py visualize --ids 5 --type 3d

# Streaming reconstruction of the full data set (single pass over the raw files)
python analysis/reconstruct_all.py --config config/settings.yaml --min-quality GOOD
```

`main.py reconstruct` re-reads the raw files for each requested event and is
intended for small selections. For the full data set use
`analysis/reconstruct_all.py`, which reconstructs inline during a single
streaming pass and computes the equatorial coordinates vectorially at the end.
