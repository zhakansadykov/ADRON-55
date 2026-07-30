"""
event_quality.py - event quality assessment for the ADRON-55 calorimeter.

The module performs:
  1. cluster finding in each plane
  2. veto conditions (saturation, uniform noise, empty event, flash)
  3. topological metrics of the event
  4. road search for tracks from multiple seeds
  5. straight-line fitting and angle computation (zenith, azimuth)
  6. deduplication of overlapping tracks
  7. quality scoring on a 0-100 scale
  8. particle classification (muon/hadron/em/neutral)

Events are flagged rather than discarded; the selection happens later, through
--min-quality on the command line.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from src.preprocessing import channel_to_mm

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Cluster:
    """A contiguous run of channels above threshold within one plane."""
    layer_idx: int              # plane index (0-7, 0 = plane 1, the top)
    projection: str             # 'X' or 'Y'
    channels: List[int]         # channel numbers belonging to the cluster
    amplitudes: List[float]     # signal in those channels
    
    @property
    def size(self) -> int:
        return len(self.channels)
    
    @property
    def max_amplitude(self) -> float:
        return max(self.amplitudes) if self.amplitudes else 0.0
    
    @property
    def total_amplitude(self) -> float:
        return sum(self.amplitudes)
    
    @property
    def centroid(self) -> float:
        """Amplitude-weighted centroid, in channel units."""
        if not self.channels or self.total_amplitude == 0:
            return float(np.mean(self.channels)) if self.channels else 0.0
        return sum(c * a for c, a in zip(self.channels, self.amplitudes)) / self.total_amplitude
    
    @property
    def channel_min(self) -> int:
        return min(self.channels)
    
    @property
    def channel_max(self) -> int:
        return max(self.channels)


@dataclass
class VetoResult:
    """Outcome of the veto conditions. All flags default to False (event is clean)."""
    is_saturated: bool = False
    is_uniform_noise: bool = False
    is_empty: bool = False
    is_flash: bool = False
    
    # Diagnostic details
    saturation_fraction: float = 0.0
    uniformity_cv: float = 0.0
    total_energy: float = 0.0
    total_hits: int = 0
    
    @property
    def any_flag(self) -> bool:
        return self.is_saturated or self.is_uniform_noise or self.is_empty or self.is_flash
    
    @property
    def flags_list(self) -> List[str]:
        flags = []
        if self.is_saturated: flags.append('saturated')
        if self.is_uniform_noise: flags.append('uniform_noise')
        if self.is_empty: flags.append('empty')
        if self.is_flash: flags.append('flash')
        return flags


@dataclass
class TopologyMetrics:
    """Topological metrics of an event."""
    n_hits: int = 0
    n_active_layers: int = 0
    n_clusters: int = 0
    n_clusters_X: int = 0
    n_clusters_Y: int = 0
    max_cluster_amplitude: float = 0.0
    layers_with_clusters: List[int] = field(default_factory=list)
    
    # Cluster distribution across the planes
    clusters_per_layer: Dict[int, int] = field(default_factory=dict)


@dataclass
class TrackCandidate:
    """A track candidate: a sequence of clusters across the planes."""
    track_id: int
    seed_layer: int
    projection: str             # 'X' or 'Y'
    hits: List[Cluster] = field(default_factory=list)
    
    # Fit results
    slope: float = 0.0          # slope (channels per plane)
    intercept: float = 0.0      # intercept
    chi2_ndf: float = 0.0       # chi2 / degrees of freedom
    
    # Angles in the detector frame
    theta_deg: float = 0.0      # angle from the vertical (zenith)
    phi_deg: float = 0.0        # azimuth
    
    # Quality
    penetration_depth: int = 0  # number of planes carrying a hit
    
    @property
    def layers_hit(self) -> List[int]:
        return [h.layer_idx for h in self.hits]
    
    @property
    def is_valid(self) -> bool:
        return self.penetration_depth >= 3


@dataclass
class EventQualityReport:
    """Full quality report for one event."""
    event_id: int
    event_time: str
    
    # VETO
    veto: VetoResult = field(default_factory=VetoResult)
    
    # Topology
    topology: TopologyMetrics = field(default_factory=TopologyMetrics)
    
    # Clusters
    clusters: List[Cluster] = field(default_factory=list)
    
    # Tracks, after deduplication
    tracks: List[TrackCandidate] = field(default_factory=list)
    
    # Event quality
    quality_score: float = 0.0
    quality_class: str = 'UNKNOWN'
    particle_type: str = 'unknown'
    
    # Decision trail, for the logs
    decision_log: List[str] = field(default_factory=list)


# ============================================================================
# CLUSTER FINDER
# ============================================================================

def find_clusters_in_layer(
    arr: np.ndarray,
    layer_idx: int,
    projection: str,
    hit_threshold: float = 30.0,
    min_cluster_size: int = 1,
    min_cluster_amplitude: float = 50.0,
    max_gap: int = 0,
) -> List[Cluster]:
    """
    Find the clusters of one plane.

    A cluster is a contiguous run of channels, allowing gaps of up to max_gap,
    whose amplitude reaches hit_threshold.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return []
    
    # Mask of active channels
    active_mask = arr >= hit_threshold
    
    # Find contiguous runs of active channels
    clusters = []
    i = 0
    n = len(arr)
    
    while i < n:
        if active_mask[i]:
            # Start of a cluster
            start = i
            # Extend right while active, or while a gap of at most max_gap is bridged
            j = i
            last_active = i
            while j < n:
                if active_mask[j]:
                    last_active = j
                    j += 1
                elif max_gap > 0 and (j - last_active) <= max_gap:
                    # Look ahead for activity within the gap
                    lookahead = min(n, j + max_gap + 1)
                    if np.any(active_mask[j:lookahead]):
                        j += 1
                    else:
                        break
                else:
                    break
            
            end = last_active + 1  # exclusive
            
            channels = list(range(start, end))
            amplitudes = [float(arr[k]) for k in channels]
            
            # Filter by size and amplitude
            if len(channels) >= min_cluster_size:
                cl = Cluster(
                    layer_idx=layer_idx,
                    projection=projection,
                    channels=channels,
                    amplitudes=amplitudes,
                )
                if cl.max_amplitude >= min_cluster_amplitude:
                    clusters.append(cl)
            
            i = end
        else:
            i += 1
    
    return clusters


def find_all_clusters(
    X_layers: List[np.ndarray],
    Y_layers: List[np.ndarray],
    cfg: Dict,
) -> List[Cluster]:
    """
    Find the clusters of every plane.

    The input lists run BOTTOM TO TOP, as produced by preprocessing.py:
      X_layers[0] = plane 7 (deepest X), X_layers[3] = plane 1 (top X)
      Y_layers[0] = plane 8 (deepest Y), Y_layers[3] = plane 2 (top Y)

    They are converted here to the standard 0-7 indexing, with 0 = plane 1 (top).
    """
    eq_cfg = cfg.get('event_quality', {})
    hit_threshold = eq_cfg.get('hit_threshold', 30.0)
    min_cluster_size = eq_cfg.get('min_cluster_size', 1)
    min_cluster_amplitude = eq_cfg.get('min_cluster_amplitude', 50.0)
    
    all_clusters = []
    
    # Map the index within X_layers to the plane number (1-based)
    x_row_numbers = [7, 5, 3, 1]
    y_row_numbers = [8, 6, 4, 2]
    
    for i, layer_data in enumerate(X_layers):
        row_num = x_row_numbers[i]
        layer_idx = row_num - 1  # 0-based
        clusters = find_clusters_in_layer(
            layer_data, layer_idx, 'X',
            hit_threshold=hit_threshold,
            min_cluster_size=min_cluster_size,
            min_cluster_amplitude=min_cluster_amplitude,
        )
        all_clusters.extend(clusters)
    
    for i, layer_data in enumerate(Y_layers):
        row_num = y_row_numbers[i]
        layer_idx = row_num - 1
        clusters = find_clusters_in_layer(
            layer_data, layer_idx, 'Y',
            hit_threshold=hit_threshold,
            min_cluster_size=min_cluster_size,
            min_cluster_amplitude=min_cluster_amplitude,
        )
        all_clusters.extend(clusters)
    
    return all_clusters


# ============================================================================
# VETO ENGINE
# ============================================================================

def veto_engine(
    X_layers: List[np.ndarray],
    Y_layers: List[np.ndarray],
    stats: Dict,
    cfg: Dict,
) -> VetoResult:
    """
    Apply the four veto conditions and return the flags.

      1. Saturation - too many channels at the maximum value (interference or fault)
      2. Uniformity - low coefficient of variation (flat baseline noise)
      3. Emptiness - too little energy or too few hits
      4. Flash     - signal confined to one or two adjacent planes
    """
    eq_cfg = cfg.get('event_quality', {})
    veto_cfg = eq_cfg.get('veto', {})
    
    result = VetoResult()
    result.total_energy = stats.get('total_energy', 0)
    result.total_hits = stats.get('n_hits', 0)
    
    # Collect every channel value
    all_values = []
    all_max = 0.0
    sat_threshold = veto_cfg.get('saturation_value', 4000)
    
    for layer in list(X_layers) + list(Y_layers):
        arr = np.asarray(layer, dtype=np.float64)
        if arr.size > 0:
            all_values.append(arr)
            all_max = max(all_max, float(np.max(arr)))
    
    if not all_values:
        result.is_empty = True
        return result
    
    all_values = np.concatenate(all_values)
    nonzero = all_values[all_values > 0]
    
    # 1) Saturation check
    sat_count = int(np.sum(all_values >= sat_threshold))
    total_count = len(all_values)
    sat_frac = sat_count / total_count if total_count > 0 else 0.0
    result.saturation_fraction = sat_frac
    if sat_frac > veto_cfg.get('saturation_fraction', 0.30):
        result.is_saturated = True
    
    # 2) Uniformity check (CV = std/mean) over the non-zero values
    if len(nonzero) > 1 and np.mean(nonzero) > 0:
        cv = float(np.std(nonzero) / np.mean(nonzero))
        result.uniformity_cv = cv
        # Flat noise: many non-zero channels, all of nearly the same amplitude.
        # A saturated event is reported under its own flag instead.
        if (cv < veto_cfg.get('uniformity_cv_max', 0.15) 
                and len(nonzero) > 50 
                and not result.is_saturated):
            result.is_uniform_noise = True
    
    # 3) Empty check
    if (result.total_energy < veto_cfg.get('min_total_energy', 100) 
            or result.total_hits < veto_cfg.get('min_total_hits', 3)):
        result.is_empty = True
    
    # 4) Flash check: signal concentrated in one or two adjacent planes.
    # Sum the energy of each plane (1-8)
    row_energies = {}
    # X_layers: [row_7, row_5, row_3, row_1]
    x_rows = [7, 5, 3, 1]
    y_rows = [8, 6, 4, 2]
    for i, layer in enumerate(X_layers):
        row_energies[x_rows[i]] = float(np.sum(np.asarray(layer)))
    for i, layer in enumerate(Y_layers):
        row_energies[y_rows[i]] = float(np.sum(np.asarray(layer)))
    
    # Adjacent plane pairs (1+2), (2+3), ..., (7+8): find the largest sum
    max_pair_energy = 0.0
    total_energy = sum(row_energies.values())
    for r in range(1, 8):
        pair_e = row_energies.get(r, 0) + row_energies.get(r+1, 0)
        max_pair_energy = max(max_pair_energy, pair_e)
    
    # More than 90% of the energy in one adjacent pair, with few other planes hit
    if (total_energy > 0 
            and max_pair_energy / total_energy > 0.90
            and len([v for v in row_energies.values() if v > 0]) <= 3):
        result.is_flash = True
    
    return result


# ============================================================================
# TOPOLOGY
# ============================================================================

def compute_topology_metrics(clusters: List[Cluster]) -> TopologyMetrics:
    """Compute the topological metrics from the clusters found."""
    metrics = TopologyMetrics()
    metrics.n_clusters = len(clusters)
    metrics.n_clusters_X = sum(1 for c in clusters if c.projection == 'X')
    metrics.n_clusters_Y = sum(1 for c in clusters if c.projection == 'Y')
    metrics.n_hits = sum(c.size for c in clusters)
    
    if clusters:
        metrics.max_cluster_amplitude = max(c.max_amplitude for c in clusters)
    
    # Clusters per plane
    layers_with = set()
    per_layer: Dict[int, int] = {}
    for c in clusters:
        layers_with.add(c.layer_idx)
        per_layer[c.layer_idx] = per_layer.get(c.layer_idx, 0) + 1
    
    metrics.layers_with_clusters = sorted(layers_with)
    metrics.n_active_layers = len(layers_with)
    metrics.clusters_per_layer = per_layer
    
    return metrics


# ============================================================================
# ROAD SEARCH TRACKER
# ============================================================================

def get_layer_height(layer_idx: int, cfg: Dict) -> float:
    """
    Height of a plane, in millimetres.
    layer_idx: 0-7, where 0 = plane 1 (top, 4275 mm) and 7 = plane 8 (470 mm).
    """
    heights = cfg['geometry']['heights']  # [4275, 4200, 1460, 1300, 1010, 820, 630, 470]
    return heights[layer_idx]


def road_search_projection(
    clusters: List[Cluster],
    projection: str,
    cfg: Dict,
) -> List[TrackCandidate]:
    """
    Road search within one projection (X or Y), over an explicit plane
    sequence and with a search window that widens with the plane gap.
    """
    eq_cfg = cfg.get('event_quality', {})
    rs_cfg = eq_cfg.get('road_search', {})
    
    seed_layers = rs_cfg.get('seed_layers', [7, 6, 5, 4])
    min_track_layers = rs_cfg.get('min_track_layers', 3)
    base_radius = rs_cfg.get('base_radius_channels', 3)
    max_layer_skips = rs_cfg.get('max_layer_skips', 1)
    
    # Plane sequence of a track, bottom to top, 1-based. By default the hadron
    # block only: the gamma planes are separated from it by 220 mm of lead and a
    # 2.2 m air gap, so their hits do not lie on the hadron-block trajectory.
    if projection == 'X':
        layer_seq = rs_cfg.get('track_rows_x', [7, 5, 3])
    else:
        layer_seq = rs_cfg.get('track_rows_y', [8, 6, 4])

    valid_seeds = [s for s in seed_layers if s in layer_seq]
    
    # Keep the clusters of this projection and group them by plane (0-based)
    proj_clusters = [c for c in clusters if c.projection == projection]
    clusters_by_layer: Dict[int, List[Cluster]] = {}
    for c in proj_clusters:
        clusters_by_layer.setdefault(c.layer_idx, []).append(c)
    
    tracks = []
    track_id_counter = 0
    
    for seed_layer in valid_seeds:
        seed_idx = seed_layer - 1
        if seed_idx not in clusters_by_layer:
            continue
        
        seq_start = layer_seq.index(seed_layer)
        
        for seed_cluster in clusters_by_layer[seed_idx]:
            track = TrackCandidate(
                track_id=track_id_counter,
                seed_layer=seed_layer,
                projection=projection,
                hits=[seed_cluster],
            )
            
            skipped = 0
            # Walk upward through the sequence
            for i in range(seq_start + 1, len(layer_seq)):
                next_layer = layer_seq[i]
                next_layer_idx = next_layer - 1
                prev_layer_idx = track.hits[-1].layer_idx
                
                # Vertical gap, which sets the search window
                z_next = get_layer_height(next_layer_idx, cfg)
                z_prev = get_layer_height(prev_layer_idx, cfg)
                delta_z = abs(z_next - z_prev)
                
                # One extra channel of window per 250 mm of gap
                search_radius = base_radius + int(delta_z / 250.0)
                
                # Predicted position in the next plane
                if len(track.hits) >= 2:
                    h1, h2 = track.hits[-2], track.hits[-1]
                    z1 = get_layer_height(h1.layer_idx, cfg)
                    z2 = get_layer_height(h2.layer_idx, cfg)
                    if z2 != z1:
                        predicted_pos = h1.centroid + (h2.centroid - h1.centroid) * (z_next - z1) / (z2 - z1)
                    else:
                        predicted_pos = h2.centroid
                else:
                    predicted_pos = track.hits[-1].centroid
                
                # Nearest cluster within the window
                if next_layer_idx in clusters_by_layer:
                    candidates = clusters_by_layer[next_layer_idx]
                    distances = [abs(c.centroid - predicted_pos) for c in candidates]
                    min_dist_idx = int(np.argmin(distances))
                    min_dist = distances[min_dist_idx]
                    
                    if min_dist <= search_radius:
                        track.hits.append(candidates[min_dist_idx])
                        skipped = 0
                    else:
                        if skipped < max_layer_skips:
                            skipped += 1
                        else:
                            break
                else:
                    if skipped < max_layer_skips:
                        skipped += 1
                    else:
                        break
            
            track.penetration_depth = len(track.hits)
            if track.penetration_depth >= min_track_layers:
                tracks.append(track)
                track_id_counter += 1
                
    return tracks


def fit_track(track: TrackCandidate, cfg: Dict) -> TrackCandidate:
    """
    Fit a track with a straight line and derive its angles.

    Least squares: position = slope * z + intercept, with z the plane height in
    millimetres.

      theta = arctan(slope), the angle from the vertical
      phi   = 0 for the X projection, 90 for Y
    """
    if len(track.hits) < 2:
        return track
    
    # Collect the (z, position) points
    z_values = []
    pos_values = []
    for hit in track.hits:
        z = get_layer_height(hit.layer_idx, cfg)
        z_values.append(z)
        pos_values.append(hit.centroid)
    
    z_arr = np.array(z_values)
    pos_arr = np.array(pos_values)
    
    # Least-squares line: pos = slope * z + intercept
    coeffs = np.polyfit(z_arr, pos_arr, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    track.slope = slope
    track.intercept = intercept
    
    # chi2 per degree of freedom
    predicted = slope * z_arr + intercept
    residuals = pos_arr - predicted
    chi2 = np.sum(residuals**2)
    ndf = len(z_arr) - 2  # degrees of freedom
    track.chi2_ndf = chi2 / ndf if ndf > 0 else 0.0
    
    # The angle is computed in millimetre space. The planes differ in width
    # (50/69/48/72 channels), so each hit must be centred by the width of its own
    # plane; otherwise a vertical track acquires a spurious tilt. channel_to_mm
    # accounts for the plane width.
    pos_mm = np.array([
        channel_to_mm(hit.centroid, hit.layer_idx, cfg) for hit in track.hits
    ])
    coeffs_mm = np.polyfit(z_arr, pos_mm, 1)
    slope_mm_per_mm = coeffs_mm[0]  # dimensionless (mm per mm)
    
    # theta = arctan(slope), the angle from the vertical
    theta_rad = np.arctan(slope_mm_per_mm)
    track.theta_deg = np.degrees(theta_rad)

    # Azimuth of a single projection: 0 for X (the track lies in XZ),
    # 90 for Y (the track lies in YZ)
    if track.projection == 'X':
        track.phi_deg = 0.0 if slope >= 0 else 180.0
    else:  # Y
        track.phi_deg = 90.0 if slope >= 0 else 270.0
    
    return track


def road_search_all(
    clusters: List[Cluster],
    cfg: Dict,
) -> List[TrackCandidate]:
    """
    Run the road search in both projections and fit every candidate found.
    Track identifiers are assigned from a single counter.
    """
    all_tracks = []
    global_track_id = 0
    
    for proj in ['X', 'Y']:
        proj_tracks = road_search_projection(clusters, proj, cfg)
        for track in proj_tracks:
            track.track_id = global_track_id
            track = fit_track(track, cfg)
            all_tracks.append(track)
            global_track_id += 1
            
    return all_tracks


# ============================================================================
# TRACK DEDUPLICATION
# ============================================================================

def deduplicate_tracks(tracks: List[TrackCandidate], cfg: Dict) -> List[TrackCandidate]:
    """
    Remove overlapping tracks, keeping the best of each group.

      1. Count the clusters shared by each pair of tracks.
      2. Tracks sharing at least half of their clusters are treated as one.
      3. The survivor is the one with the greater penetration depth; at equal
         depth, the one with the smaller chi2/ndf.
    
    Returns:
        The list of surviving tracks.
    """
    if len(tracks) <= 1:
        return tracks
    
    # Index the tracks by identifier
    track_dict = {t.track_id: t for t in tracks}
    to_remove = set()
    
    # Compare every pair
    for i in range(len(tracks)):
        if tracks[i].track_id in to_remove:
            continue
        
        for j in range(i + 1, len(tracks)):
            if tracks[j].track_id in to_remove:
                continue
            
            # Shared clusters, identified by (layer_idx, projection, centroid)
            clusters_i = {(h.layer_idx, h.projection, round(h.centroid, 1)) for h in tracks[i].hits}
            clusters_j = {(h.layer_idx, h.projection, round(h.centroid, 1)) for h in tracks[j].hits}
            
            common = clusters_i & clusters_j
            overlap_fraction = len(common) / min(len(clusters_i), len(clusters_j)) if clusters_i and clusters_j else 0
            
            # At least half the clusters shared: the tracks overlap
            if overlap_fraction >= 0.5:
                # Keep the deeper track; at equal depth, the smaller chi2
                if tracks[i].penetration_depth > tracks[j].penetration_depth:
                    to_remove.add(tracks[j].track_id)
                elif tracks[j].penetration_depth > tracks[i].penetration_depth:
                    to_remove.add(tracks[i].track_id)
                else:
                    # Equal depth: keep the smaller chi2
                    if tracks[i].chi2_ndf < tracks[j].chi2_ndf:
                        to_remove.add(tracks[j].track_id)
                    else:
                        to_remove.add(tracks[i].track_id)
    
    # Return the survivors
    unique_tracks = [t for t in tracks if t.track_id not in to_remove]
    
    # Renumber the track identifiers
    for new_id, track in enumerate(unique_tracks):
        track.track_id = new_id
    
    return unique_tracks


# ============================================================================
# QUALITY SCORING
# ============================================================================

def compute_quality_score(
    tracks: List[TrackCandidate],
    topology: TopologyMetrics,
    veto: VetoResult,
    cfg: Dict,
) -> Tuple[float, str]:
    """
    Score the event on a 0-100 scale and assign a grade.

    The score is heuristic; it serves only to remove obviously pathological
    events before reconstruction.
      quality_score = (
        penetration_score * 0.35 +      # penetration depth
        track_goodness * 0.25 +         # fit quality (chi2)
        topology_score * 0.20 +         # event topology
        cluster_purity * 0.10 +         # cluster purity
        energy_profile_score * 0.10     # deposition profile
      )
    
    Returns:
        (quality_score, quality_class)
    """
    eq_cfg = cfg.get('event_quality', {})
    qc_cfg = eq_cfg.get('quality_classes', {})
    
    # Any veto flag grades the event NOISE
    if veto.any_flag:
        return 0.0, 'NOISE'
    
    # No tracks: POOR
    if not tracks:
        return 20.0, 'POOR'
    
    # Best track: greatest depth, then smallest chi2
    best_track = max(tracks, key=lambda t: (t.penetration_depth, -t.chi2_ndf))
    
    # 1. Penetration score (0-100)
    # Eight planes in total (4 X + 4 Y), so at most four per projection
    max_depth = 4  # per projection
    penetration_score = min(100, (best_track.penetration_depth / max_depth) * 100)
    
    # 2. Track goodness (0-100)
    # chi2/ndf: ~1 is ideal, above 5 is poor
    if best_track.chi2_ndf <= 1.0:
        track_goodness = 100.0
    elif best_track.chi2_ndf <= 3.0:
        track_goodness = 80.0 - (best_track.chi2_ndf - 1.0) * 10
    else:
        track_goodness = max(0, 60.0 - (best_track.chi2_ndf - 3.0) * 20)
    
    # 3. Topology score (0-100)
    # A good event has six to eight active planes
    if topology.n_active_layers >= 6:
        topology_score = 100.0
    elif topology.n_active_layers >= 4:
        topology_score = 80.0
    else:
        topology_score = 50.0
    
    # 4. Cluster purity (0-100)
    # Hits inside clusters relative to a reference count; high means little noise
    if topology.n_hits > 0:
        cluster_purity = min(100, (topology.n_hits / max(topology.n_hits, 50)) * 100)
    else:
        cluster_purity = 0.0
    
    # 5. Energy profile score (0-100)
    # Placeholder: the longitudinal profile is not yet used here
    energy_profile_score = 70.0  # placeholder
    
    # Combined score
    quality_score = (
        penetration_score * 0.35 +
        track_goodness * 0.25 +
        topology_score * 0.20 +
        cluster_purity * 0.10 +
        energy_profile_score * 0.10
    )
    
    # Grade
    if quality_score >= qc_cfg.get('excellent', 85):
        quality_class = 'EXCELLENT'
    elif quality_score >= qc_cfg.get('good', 65):
        quality_class = 'GOOD'
    elif quality_score >= qc_cfg.get('poor', 40):
        quality_class = 'POOR'
    else:
        quality_class = 'NOISE'
    
    return quality_score, quality_class


def classify_particle(tracks: List[TrackCandidate], topology: TopologyMetrics, cfg: Dict) -> str:
    """
    Classify the particle from the pattern of planes carrying clusters.

    NOTE: this classifier is not used for any published result. It was written
    against an earlier arrangement of the planes and needs revision for the
    hadron-block tracking adopted here.
    """
    if not tracks and topology.n_active_layers == 0:
        return 'unknown'
    
    # Use every cluster, not only those on tracks
    all_layers = set(topology.layers_with_clusters)
    
    has_top = any(layer in [0, 1] for layer in all_layers)
    has_bottom = any(layer in [6, 7] for layer in all_layers)
    has_middle = any(layer in [2, 3, 4, 5] for layer in all_layers)
    
    # Penetrating muon: signal in both the top and the bottom planes
    if has_top and has_bottom:
        best_track = max(tracks, key=lambda t: t.penetration_depth) if tracks else None
        if best_track and best_track.penetration_depth >= 4:
            return 'penetrating_muon'
    
    # Hadron shower: several clusters in the middle and lower planes
    clusters_per_layer = topology.clusters_per_layer
    shower_layers = [2, 3, 4, 5]  # planes 3-6
    max_clusters_in_shower = max(clusters_per_layer.get(i, 0) for i in shower_layers)
    
    if max_clusters_in_shower >= 3 and has_middle:
        return 'hadron_shower'
    
    # Electromagnetic shower: signal confined to the gamma planes
    if has_top and not has_middle and not has_bottom:
        return 'em_shower'
    
    # Neutral candidate: nothing on top, signal in the middle
    if not has_top and has_middle:
        return 'neutral_candidate'
    
    return 'unknown'

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def analyze_event(
    event_id: int,
    event_time: str,
    X_layers: List[np.ndarray],
    Y_layers: List[np.ndarray],
    stats: Dict,
    cfg: Dict,
) -> EventQualityReport:
    """
    Full assessment of one event:
      1. cluster finding
      2. veto conditions
      3. topological metrics
      4. road search
      5. fitting and angles
      6. track deduplication
      7. Quality Scoring
      8. particle classification
    
    Args:
        event_id: global event identifier
        event_time: event timestamp
        X_layers: X planes, bottom to top (7, 5, 3, 1)
        Y_layers: Y planes, bottom to top (8, 6, 4, 2)
        stats: statistics from data_loader.compute_event_stats
        cfg: the full configuration
    
    Returns:
        EventQualityReport
    """
    report = EventQualityReport(event_id=event_id, event_time=event_time)
    
    # 1. Cluster finding
    logger.debug(f"Event {event_id}: starting cluster search")
    report.clusters = find_all_clusters(X_layers, Y_layers, cfg)
    n_x = sum(1 for c in report.clusters if c.projection == 'X')
    n_y = sum(1 for c in report.clusters if c.projection == 'Y')
    logger.debug(f"Event {event_id}: found {len(report.clusters)} clusters (X={n_x}, Y={n_y})")
    report.decision_log.append(
        f"CLUSTERS: found {len(report.clusters)} (X={n_x}, Y={n_y})"
    )
    
    # 2. VETO
    logger.debug(f"Event {event_id}: running VETO checks")
    report.veto = veto_engine(X_layers, Y_layers, stats, cfg)
    if report.veto.any_flag:
        logger.debug(f"Event {event_id}: VETO flagged [{', '.join(report.veto.flags_list)}]")
        report.decision_log.append(
            f"VETO: flagged [{', '.join(report.veto.flags_list)}] "
            f"(sat={report.veto.saturation_fraction:.2%}, "
            f"cv={report.veto.uniformity_cv:.2f})"
        )
    else:
        logger.debug(f"Event {event_id}: VETO passed")
        report.decision_log.append(
            f"VETO: passed (sat={report.veto.saturation_fraction:.2%}, "
            f"cv={report.veto.uniformity_cv:.2f})"
        )
    
    # 3. Topology
    report.topology = compute_topology_metrics(report.clusters)
    logger.info(
        f"Event {event_id}: topology - {report.topology.n_active_layers} active layers, "
        f"{report.topology.n_hits} hits"
    )
    report.decision_log.append(
        f"TOPOLOGY: {report.topology.n_active_layers} active layers, "
        f"{report.topology.n_hits} hits in clusters, "
        f"max_amp={report.topology.max_cluster_amplitude:.0f}"
    )
    
    # 4. Road Search
    logger.debug(f"Event {event_id}: starting Road Search")
    raw_tracks = road_search_all(report.clusters, cfg)
    logger.debug(f"Event {event_id}: found {len(raw_tracks)} raw tracks")
    
    # 5. Deduplication
    report.tracks = deduplicate_tracks(raw_tracks, cfg)
    logger.debug(f"Event {event_id}: {len(report.tracks)} unique tracks after deduplication")
    report.decision_log.append(
        f"ROAD SEARCH: {len(raw_tracks)} raw → {len(report.tracks)} unique tracks"
    )
    
    # Per-track details
    for track in report.tracks:
        logger.debug(
            f"Event {event_id}: track #{track.track_id} - "
            f"seed={track.seed_layer}, depth={track.penetration_depth}, "
            f"chi2/ndf={track.chi2_ndf:.2f}, theta={track.theta_deg:.1f}°"
        )
        report.decision_log.append(
            f"  Track #{track.track_id}: seed layer {track.seed_layer}, "
            f"depth {track.penetration_depth}, χ²={track.chi2_ndf:.2f}, "
            f"θ={track.theta_deg:.1f}°, φ={track.phi_deg:.1f}°"
        )
    
    # 6. Quality Scoring
    report.quality_score, report.quality_class = compute_quality_score(
        report.tracks, report.topology, report.veto, cfg
    )
    logger.info(
        f"Event {event_id}: quality = {report.quality_score:.1f}/100 ({report.quality_class})"
    )
    report.decision_log.append(
        f"QUALITY: {report.quality_score:.1f}/100 → {report.quality_class}"
    )
    
    # 7. Particle classification
    report.particle_type = classify_particle(report.tracks, report.topology, cfg)
    logger.debug(f"Event {event_id}: particle type = {report.particle_type}")
    report.decision_log.append(f"PARTICLE: {report.particle_type}")
    
    return report


def setup_quality_logging(cfg: Dict) -> None:
    """Send the event_quality log to its own file."""
    eq_cfg = cfg.get('event_quality', {})
    log_cfg = eq_cfg.get('logging', {})
    log_dir = log_cfg.get('dir', 'logs')
    level_str = log_cfg.get('level', 'INFO').upper()
    level = getattr(logging, level_str, logging.INFO)
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Dedicated file handler for event_quality
    fh = logging.FileHandler(
        os.path.join(log_dir, 'event_quality.log'),
        encoding='utf-8',
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    
    # Attach to the module logger, avoiding duplicates
    if not any(isinstance(h, logging.FileHandler) 
               and 'event_quality' in (h.baseFilename if hasattr(h, 'baseFilename') else '')
               for h in logger.handlers):
        logger.addHandler(fh)
        logger.setLevel(level)


def summarize_report(report: EventQualityReport) -> str:
    """One-line summary for terminal output."""
    veto_str = '|'.join(report.veto.flags_list) if report.veto.any_flag else 'ok'
    n_tracks = len(report.tracks)
    n_valid_tracks = sum(1 for t in report.tracks if t.is_valid)
    
    return (
        f"ID={report.event_id:>8} | "
        f"clusters={len(report.clusters):>3} | "
        f"layers={report.topology.n_active_layers}/8 | "
        f"hits={report.topology.n_hits:>4} | "
        f"tracks={n_valid_tracks}/{n_tracks} | "
        f"quality={report.quality_class}({report.quality_score:.0f}) | "
        f"particle={report.particle_type} | "
        f"veto=[{veto_str}]"
    )