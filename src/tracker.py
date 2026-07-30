import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.event_quality import TrackCandidate, Cluster, EventQualityReport
from src.physics import get_cumulative_rad_length
from src.preprocessing import channel_to_mm

logger = logging.getLogger(__name__)

# Fallback values; the operative ones come from cfg['tracking']['fitting'].
SIGMA_TYPICAL_MM = 200.0
CHI2_NDF_THRESHOLD = 10.0
MIN_HITS_3D = 3


def _fitting_cfg(cfg):
    """3D-fit parameters from the configuration, falling back to the defaults above."""
    f = cfg.get('tracking', {}).get('fitting', {}) or {}
    return (
        float(f.get('sigma_typical_mm', SIGMA_TYPICAL_MM)),
        float(f.get('chi2_ndf_threshold', CHI2_NDF_THRESHOLD)),
        int(f.get('min_hits_3d', MIN_HITS_3D)),
    )

@dataclass
class Track3D:
    track_id: int
    hits_3d: List[Tuple[float, float, float]]
    hits_weights: List[Tuple[float, float]] = field(default_factory=list)
    slope_x: float = 0.0
    slope_y: float = 0.0
    intercept_x: float = 0.0
    intercept_y: float = 0.0
    chi2_3d: float = 0.0
    chi2_raw: float = 0.0
    zenith_deg: float = 0.0
    azimuth_deg: float = 0.0
    energy: float = 0.0
    vertex: Optional[Tuple[float, float, float]] = None
    vertex_cluster_id: int = -1
    x_track_id: int = -1
    y_track_id: int = -1
    matching_method: str = "unknown"

@dataclass
class ShowerVertex:
    vertex_id: int
    position: Tuple[float, float, float]
    track_ids: List[int]
    n_tracks: int
    spread_mm: float

def compute_track_energy(track: TrackCandidate) -> float:
    return sum(hit.total_amplitude for hit in track.hits)

def match_xy_tracks_hybrid(x_tracks, y_tracks, cfg):
    if not x_tracks or not y_tracks: return []
    hybrid_cfg = cfg.get('tracking', {}).get('hybrid_matching', {})
    ratio_threshold = hybrid_cfg.get('energy_ratio_threshold', 3.0)
    
    x_sorted = sorted(x_tracks, key=compute_track_energy, reverse=True)
    y_sorted = sorted(y_tracks, key=compute_track_energy, reverse=True)
    
    n_matches = min(len(x_sorted), len(y_sorted))
    energy_matches, geo_x, geo_y = [], [], []
    
    for i in range(n_matches):
        x_track, y_track = x_sorted[i], y_sorted[i]
        x_e, y_e = compute_track_energy(x_track), compute_track_energy(y_track)
        ratio = max(x_e, y_e) / min(x_e, y_e) if min(x_e, y_e) > 0 else float('inf')
        
        if ratio <= ratio_threshold:
            energy_matches.append((x_track, y_track, 'energy'))
        else:
            geo_x.append(x_track); geo_y.append(y_track)
            
    geo_x.extend(x_sorted[n_matches:])
    geo_y.extend(y_sorted[n_matches:])
    
    geometry_matches = []
    if geo_x and geo_y:
        z_common = hybrid_cfg.get('z_common_mm', 2000.0)
        max_dist = hybrid_cfg.get('geometry_max_distance_mm', 5000.0)
        
        x_pts = []
        for xt in geo_x:
            s, b = fit_track_mm(xt, cfg)
            x_pts.append(s * z_common + b)
        y_pts = []
        for yt in geo_y:
            s, b = fit_track_mm(yt, cfg)
            y_pts.append(s * z_common + b)
        
        cost_matrix = np.zeros((len(x_pts), len(y_pts)))
        for i in range(len(x_pts)):
            for j in range(len(y_pts)):
                cost_matrix[i, j] = np.sqrt(x_pts[i]**2 + y_pts[j]**2)
                
        x_indices, y_indices = linear_sum_assignment(cost_matrix)
        for i, j in zip(x_indices, y_indices):
            if cost_matrix[i, j] < max_dist:
                geometry_matches.append((geo_x[i], geo_y[j], 'geometry'))
                
    return energy_matches + geometry_matches

def get_layer_height(layer_idx: int, cfg: Dict) -> float:
    return cfg['geometry']['heights'][layer_idx]

def fit_exclude_idx(cfg):
    """Plane indices excluded from the 3D direction fit (gamma planes 0 and 1).
    The gamma block is separated from the hadron block by 220 mm of lead and a
    2.2 m air gap, so its hits do not lie on the hadron-block trajectory."""
    return set(cfg.get('tracking', {}).get('fit_exclude_row_idx', []) or [])

def fit_track_mm(track, cfg):
    """
    Straight-line fit of a track in millimetre space: pos_mm = slope*z + intercept.
    Each hit is centred by the width of its own plane (channel_to_mm). The gamma
    planes are excluded. Returns (slope_mm_per_mm, intercept_mm).
    """
    exclude = fit_exclude_idx(cfg)
    zs, ps = [], []
    for h in track.hits:
        if h.layer_idx in exclude:
            continue
        zs.append(get_layer_height(h.layer_idx, cfg))
        ps.append(channel_to_mm(h.centroid, h.layer_idx, cfg))
    if len(zs) < 2:
        return 0.0, (ps[0] if ps else 0.0)
    coeffs = np.polyfit(np.array(zs), np.array(ps), 1)
    return float(coeffs[0]), float(coeffs[1])


def create_3d_hits(x_track, y_track, cfg):
    hits_3d, hits_weights = [], []
    exclude = fit_exclude_idx(cfg)
    
    # The orthogonal coordinate is taken from the millimetre fit of the partner track,
    # not from its channel-space slope and intercept, which mixed planes of different widths.
    y_slope_mm, y_inter_mm = fit_track_mm(y_track, cfg)
    x_slope_mm, x_inter_mm = fit_track_mm(x_track, cfg)
    
    for x_hit in x_track.hits:
        if x_hit.layer_idx in exclude:
            continue
        z_mm = get_layer_height(x_hit.layer_idx, cfg)
        x_mm = channel_to_mm(x_hit.centroid, x_hit.layer_idx, cfg)
        y_mm = y_slope_mm * z_mm + y_inter_mm
        hits_3d.append((x_mm, y_mm, z_mm))
        # The base weights (1.0, 0.1) are rescaled in the fitting function
        hits_weights.append((1.0, 0.1, x_hit.layer_idx))
        
    for y_hit in y_track.hits:
        if y_hit.layer_idx in exclude:
            continue
        z_mm = get_layer_height(y_hit.layer_idx, cfg)
        y_mm = channel_to_mm(y_hit.centroid, y_hit.layer_idx, cfg)
        x_mm = x_slope_mm * z_mm + x_inter_mm
        hits_3d.append((x_mm, y_mm, z_mm))
        hits_weights.append((0.1, 1.0, y_hit.layer_idx))
        
    return hits_3d, hits_weights

def fit_track_3d_weighted(hits_3d, hits_weights, cfg):
    sigma_typical, _, min_hits = _fitting_cfg(cfg)
    if len(hits_3d) < min_hits: return None

    x_vals = np.array([h[0] for h in hits_3d])
    y_vals = np.array([h[1] for h in hits_3d])
    z_vals = np.array([h[2] for h in hits_3d])
    
    # === MULTIPLE SCATTERING ===
    # The multiple-scattering variance grows with the traversed radiation length (X/X0):
    # sigma_ms^2 ~ (X/X0) * (E_s / p*c)^2
    # For weighted least squares w = 1/sigma^2. The momentum p is unknown, so 1/(X/X0)
    # is used as a relative weight.
    rad_lengths = get_cumulative_rad_length(cfg)
    
    w_x_list, w_y_list = [], []
    for w in hits_weights:
        layer_idx = w[2]
        ms_factor = 1.0 / rad_lengths.get(layer_idx, 1.0) # more material -> less weight
        # w[0] and w[1] are the base weights (1.0 measured hit, 0.1 interpolated)
        w_x_list.append(w[0] * ms_factor)
        w_y_list.append(w[1] * ms_factor)
        
    weights_x = np.array(w_x_list)
    weights_y = np.array(w_y_list)
    
    coeffs_x = np.polyfit(z_vals, x_vals, 1, w=np.sqrt(weights_x))
    slope_x, intercept_x = coeffs_x[0], coeffs_x[1]
    coeffs_y = np.polyfit(z_vals, y_vals, 1, w=np.sqrt(weights_y))
    slope_y, intercept_y = coeffs_y[0], coeffs_y[1]
    
    x_pred = slope_x * z_vals + intercept_x
    y_pred = slope_y * z_vals + intercept_y
    chi2_x_raw = np.sum(weights_x * (x_vals - x_pred)**2)
    chi2_y_raw = np.sum(weights_y * (y_vals - y_pred)**2)
    
    chi2_x_norm = chi2_x_raw / (sigma_typical ** 2)
    chi2_y_norm = chi2_y_raw / (sigma_typical ** 2)
    ndf = max(1, 2 * len(hits_3d) - 4)
    chi2_3d = (chi2_x_norm + chi2_y_norm) / ndf
    chi2_raw = chi2_x_raw + chi2_y_raw
    
    slope_mm = np.sqrt(slope_x**2 + slope_y**2)
    zenith_deg = np.degrees(np.arctan(slope_mm))
    azimuth_deg = np.degrees(np.arctan2(-slope_y, -slope_x))
    if azimuth_deg < 0: azimuth_deg += 360.0
        
    return {
        'slope_x': slope_x, 'slope_y': slope_y,
        'intercept_x': intercept_x, 'intercept_y': intercept_y,
        'chi2_3d': chi2_3d, 'chi2_raw': chi2_raw,
        'zenith_deg': zenith_deg, 'azimuth_deg': azimuth_deg,
    }

def simple_cluster_2d(points, eps):
    n = len(points)
    if n == 0: return np.array([], dtype=int)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    for i in range(n):
        if labels[i] != -1: continue
        labels[i] = cluster_id
        queue = [i]
        while queue:
            curr = queue.pop(0)
            for j in range(n):
                if labels[j] != -1: continue
                dist = np.sqrt((points[curr, 0] - points[j, 0])**2 + (points[curr, 1] - points[j, 1])**2)
                if dist <= eps:
                    labels[j] = cluster_id
                    queue.append(j)
        cluster_id += 1
    return labels

def get_extrapolation_height(particle_type, cfg):
    c_cfg = cfg.get('tracking', {}).get('cosmic_ray', {})
    if c_cfg.get('extrapolation_mode', 'auto') == 'fixed':
        return c_cfg.get('fixed_height_m', 10000) * 1000.0
    return c_cfg.get('auto_heights_m', {}).get(particle_type, 10000) * 1000.0

def extrapolate_track_to_height(track, z_target):
    return (track.slope_x * z_target + track.intercept_x, track.slope_y * z_target + track.intercept_y)

def find_multiple_vertices(tracks_3d, clusters, particle_type, cfg):
    if not tracks_3d: return [], {}
    c_cfg = cfg.get('tracking', {}).get('cosmic_ray', {})
    eps = c_cfg.get('vertex_clustering_eps_m', 500) * 1000.0
    target_h = get_extrapolation_height(particle_type, cfg)
    points = np.array([extrapolate_track_to_height(t, target_h) for t in tracks_3d])
    
    if len(tracks_3d) >= 2: labels = simple_cluster_2d(points, eps)
    else: labels = np.zeros(len(tracks_3d), dtype=int)
        
    c_dict = {}
    for i, label in enumerate(labels): c_dict.setdefault(int(label), []).append(tracks_3d[i])
        
    vertices, t_to_v = [], {}
    for c_id, c_tracks in c_dict.items():
        x_pos = [extrapolate_track_to_height(t, target_h)[0] for t in c_tracks]
        y_pos = [extrapolate_track_to_height(t, target_h)[1] for t in c_tracks]
        x_v, y_v = np.mean(x_pos), np.mean(y_pos)
        spread = np.sqrt(np.var(x_pos) + np.var(y_pos)) if len(c_tracks) > 1 else 0.0
        
        v = ShowerVertex(c_id, (x_v, y_v, target_h), [t.track_id for t in c_tracks], len(c_tracks), spread)
        vertices.append(v)
        for t in c_tracks:
            t_to_v[t.track_id] = (x_v, y_v, target_h)
            t.vertex = (x_v, y_v, target_h)
            t.vertex_cluster_id = c_id
    return vertices, t_to_v

def find_shower_vertex_single(tracks_3d, clusters, particle_type, cfg):
    if not tracks_3d: return None
    l_e = {}
    for c in clusters: l_e[c.layer_idx] = l_e.get(c.layer_idx, 0) + c.total_amplitude
    if not l_e: return None
    s_layers = sorted(l_e.keys())
    max_e = max(l_e.values())
    thresh = max_e * 0.1
    v_layer = s_layers[0]
    for i in range(len(s_layers)):
        c_e = l_e[s_layers[i]]
        if i == 0 and c_e > thresh: v_layer = s_layers[i]; break
        if i > 0:
            p_e = l_e[s_layers[i-1]]
            if (c_e > p_e * 2.0 and c_e > 100) or (p_e < thresh and c_e >= thresh): v_layer = s_layers[i]; break
    best = max(tracks_3d, key=lambda t: len(t.hits_3d))
    z_v = cfg['geometry']['heights'][v_layer]
    return (best.slope_x * z_v + best.intercept_x, best.slope_y * z_v + best.intercept_y, z_v)

def reconstruct_3d(report, cfg):
    x_tracks = [t for t in report.tracks if t.projection == 'X']
    y_tracks = [t for t in report.tracks if t.projection == 'Y']
    
    _, chi2_max, min_hits = _fitting_cfg(cfg)

    matches = match_xy_tracks_hybrid(x_tracks, y_tracks, cfg)
    tracks_3d = []

    for i, (x_t, y_t, method) in enumerate(matches):
        hits_3d, hits_w = create_3d_hits(x_t, y_t, cfg)
        if len(hits_3d) < min_hits: continue

        fit = fit_track_3d_weighted(hits_3d, hits_w, cfg)
        if not fit: continue

        if fit['chi2_3d'] > chi2_max: continue
        
        energy = compute_track_energy(x_t) + compute_track_energy(y_t)
        t3d = Track3D(i, hits_3d, hits_w, energy=energy, x_track_id=x_t.track_id, y_track_id=y_t.track_id, matching_method=method, **fit)
        tracks_3d.append(t3d)
        
    p_type = report.particle_type
    if p_type in ['penetrating_muon', 'hadron_shower']:
        vertices, _ = find_multiple_vertices(tracks_3d, report.clusters, p_type, cfg)
    else:
        v = find_shower_vertex_single(tracks_3d, report.clusters, p_type, cfg)
        vertices = []
        if v:
            vertices.append(ShowerVertex(0, v, [t.track_id for t in tracks_3d], len(tracks_3d), 0.0))
            for t in tracks_3d: t.vertex = v; t.vertex_cluster_id = 0
                
    return {'tracks_3d': tracks_3d, 'vertices': vertices, 'particle_type': p_type}