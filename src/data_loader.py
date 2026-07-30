import os
import glob
import logging
from datetime import datetime
from typing import List, Dict, Optional, Iterator

import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("h5py is not installed. Install it with: pip install h5py")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Handlers are configured by main.py through setup_logging(); everything goes
# to logs/scan_full.log


# ============================================================
# 1. PARSING THE RAW .dat FILES
# ============================================================

def parse_dat_file(filepath: str) -> List[Dict]:
    """
    Read a .dat file and return its events as dictionaries.
    Each event holds: header, event_time, lines (raw) and arrays (sections).
    """
    events = []
    current_event = None

    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return events

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("|EVENT: "):
                try:
                    parts = line.split()
                    date_str = parts[1]
                    time_str = parts[2]
                    dt_obj = datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M:%S")
                    formatted_time = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                except (IndexError, ValueError) as e:
                    logger.debug(f"Could not parse the header: {line} ({e})")
                    current_event = None
                    continue

                current_event = {
                    'header': line,
                    'event_time': formatted_time,
                    'lines': [line],
                    'arrays': {},
                    'source_file': os.path.basename(filepath),
                }

            elif line == "#" and current_event is not None:
                current_event['lines'].append(line)
                events.append(current_event)
                current_event = None

            elif current_event is not None:
                current_event['lines'].append(line)
                if ": " in line and "high sensitivity" in line:
                    parts = line.split(": ")
                    if len(parts) >= 3:
                        layer_name = parts[0].strip()
                        data_str = parts[2].strip()
                        try:
                            data_arr = [int(x) for x in data_str.split()]
                            if layer_name != "scinti":
                                current_event['arrays'][layer_name] = data_arr
                        except ValueError:
                            pass

    return events


def iter_all_events(raw_dir: str, years: Optional[List[int]] = None) -> Iterator[Dict]:
    """
    Lazily iterate over every event of every .dat file, one at a time,
    without holding the data set in memory.
    """
    if not os.path.exists(raw_dir):
        logger.error(f"Directory not found: {raw_dir}")
        return

    # Collect the .dat files
    all_files = []
    if years:
        for year in years:
            year_dir = os.path.join(raw_dir, str(year))
            if os.path.exists(year_dir):
                all_files.extend(sorted(glob.glob(os.path.join(year_dir, "*.dat"))))
    else:
        all_files = sorted(glob.glob(os.path.join(raw_dir, "**/*.dat"), recursive=True))

    if not all_files:
        logger.error(f"No .dat files found in {raw_dir}")
        return

    logger.info(f"Files to process: {len(all_files)}")

    global_event_id = 0
    for fpath in all_files:
        logger.info(f"  parsing: {os.path.basename(fpath)}")
        events = parse_dat_file(fpath)
        for ev in events:
            ev['global_id'] = global_event_id
            yield ev
            global_event_id += 1


# ============================================================
# 2. FILTERS
# ============================================================

def compute_event_stats(event: Dict, hit_threshold: int = 5) -> Dict:
    """Event statistics used for the pre-selection and the catalogue.

    active_layers counts PHYSICAL PLANES (0-8), not readout sections: a plane is
    active if at least one of its parallel sections fired. active_rows_x and
    active_rows_y give the active planes of each projection separately, which
    matters because a three-dimensional direction needs both projections.
    """
    arrays = event.get('arrays', {})

    total_energy = 0
    n_hits = 0
    active_row_set = set()
    bottom_has_signal = False
    bottom_keys = {'front_3', 'back_3', 'left_3', 'middle_3', 'right_3'}

    for name, arr in arrays.items():
        valid = [x for x in arr if x > 0]
        layer_energy = sum(valid)
        layer_hits = sum(1 for x in valid if x >= hit_threshold)

        total_energy += layer_energy
        n_hits += layer_hits

        if layer_hits > 0:
            row = CAMERA_TO_ROW.get(name)
            if row is not None:
                active_row_set.add(row)
        if name in bottom_keys and layer_hits > 0:
            bottom_has_signal = True

    active_rows_x = len(active_row_set & X_ROWS)
    active_rows_y = len(active_row_set & Y_ROWS)

    return {
        'total_energy': total_energy,
        'n_hits': n_hits,
        'active_layers': len(active_row_set),   # physical planes (0-8)
        'active_rows_x': active_rows_x,          # active X planes (0-4)
        'active_rows_y': active_rows_y,          # active Y planes (0-4)
        'bottom_has_signal': bottom_has_signal,
        'n_layers_total': len(arrays),
    }

def passes_filters(stats: Dict, filters_cfg: Dict) -> tuple[bool, str]:
    """Apply the pre-selection to the event statistics."""
    if stats['total_energy'] < filters_cfg.get('min_total_energy', 0):
        return False, "low_energy"

    max_e = filters_cfg.get('max_total_energy', 0)
    if max_e > 0 and stats['total_energy'] > max_e:
        return False, "too_high_energy"

    if filters_cfg.get('require_bottom_layers', False) and not stats['bottom_has_signal']:
        return False, "no_bottom_signal"

    if stats['active_layers'] < filters_cfg.get('min_active_layers', 0):
        return False, "few_active_layers"

    # Both projections are required for a direction fit (0 disables the cut)
    if stats.get('active_rows_x', 0) < filters_cfg.get('min_active_rows_x', 0):
        return False, "few_x_rows"
    if stats.get('active_rows_y', 0) < filters_cfg.get('min_active_rows_y', 0):
        return False, "few_y_rows"

    if stats['n_hits'] < filters_cfg.get('min_hits_above_threshold', 0):
        return False, "few_hits"

    if stats['n_hits'] < filters_cfg.get('min_hits_above_threshold', 0):
        return False, "few_hits"

    return True, "passed"


# ============================================================
# 3. HDF5 CATALOGUE
# ============================================================

EXPECTED_LAYERS = [
    'right_g', 'left_g', 'front_g', 'back_g',
    'right_1', 'middle_1', 'left_1', 'front_1', 'back_1',
    'right_2', 'middle_2', 'left_2', 'front_2', 'back_2',
    'right_3', 'middle_3', 'left_3', 'front_3', 'back_3',
]

# Readout section -> physical plane number (1-8).
# The suffix gives the tier, the prefix the projection
# (left/right/middle -> X, front/back -> Y).
CAMERA_TO_ROW = {
    'left_g': 1, 'right_g': 1,   'front_g': 2, 'back_g': 2,
    'left_1': 3, 'middle_1': 3, 'right_1': 3,   'front_1': 4, 'back_1': 4,
    'left_2': 5, 'middle_2': 5, 'right_2': 5,   'front_2': 6, 'back_2': 6,
    'left_3': 7, 'middle_3': 7, 'right_3': 7,   'front_3': 8, 'back_3': 8,
}
X_ROWS = {1, 3, 5, 7}
Y_ROWS = {2, 4, 6, 8}

def build_catalog(raw_dir: str, output_path: str,
                  years: Optional[List[int]] = None,
                  filters_cfg: Optional[Dict] = None) -> Dict:
    """
    Scan every raw file, compute the statistics and write the catalogue to HDF5.
    The catalogue holds metadata and statistics only; the amplitude arrays are
    not stored, and are re-read from the raw files on demand.

    Returns a summary of the scan.
    """
    if not HDF5_AVAILABLE:
        raise RuntimeError("h5py is not installed")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    filters_cfg = filters_cfg or {}

    stats_list = []
    passed_count = 0
    rejected_reasons = {}

    with h5py.File(output_path, 'w') as hf:
        # Metadata group
        meta_grp = hf.create_group('metadata')

        # Accumulate in lists, then create the datasets in one go
        ids, energies, n_hits, active_layers, bottom_flags = [], [], [], [], []
        timestamps = []
        source_files = []
        filter_statuses = []

        for event in iter_all_events(raw_dir, years=years):
            stats = compute_event_stats(event, filters_cfg.get('hit_threshold', 5))
            ok, reason = passes_filters(stats, filters_cfg)

            ids.append(event['global_id'])
            energies.append(stats['total_energy'])
            n_hits.append(stats['n_hits'])
            active_layers.append(stats['active_layers'])
            bottom_flags.append(stats['bottom_has_signal'])
            timestamps.append(event['event_time'])
            source_files.append(event['source_file'])
            filter_statuses.append(reason)

            if ok:
                passed_count += 1
            else:
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

            if (event['global_id'] + 1) % 10000 == 0:
                logger.info(f"    events processed: {event['global_id'] + 1}")

        # Write the datasets
        meta_grp.create_dataset('event_id', data=np.array(ids, dtype=np.int64))
        meta_grp.create_dataset('total_energy', data=np.array(energies, dtype=np.int64))
        meta_grp.create_dataset('n_hits', data=np.array(n_hits, dtype=np.int32))
        meta_grp.create_dataset('active_layers', data=np.array(active_layers, dtype=np.int8))
        meta_grp.create_dataset('bottom_has_signal', data=np.array(bottom_flags, dtype=np.bool_))
        meta_grp.create_dataset('filter_status', data=np.array(filter_statuses, dtype='S32'))

        # Strings are stored separately (UTF-8)
        dt_str = h5py.string_dtype(encoding='utf-8')
        meta_grp.create_dataset('event_time', data=timestamps, dtype=dt_str)
        meta_grp.create_dataset('source_file', data=source_files, dtype=dt_str)

    total = len(ids)
    logger.info(f"Catalogue written: {output_path}")
    logger.info(f"   Total events: {total}, passed pre-selection: {passed_count}")
    logger.info(f"   Rejection reasons: {rejected_reasons}")

    return {
        'total': total,
        'passed': passed_count,
        'rejected': rejected_reasons,
        'path': output_path,
    }


def get_filtered_ids(catalog_path: str) -> np.ndarray:
    """Identifiers of the events that passed the pre-selection."""
    with h5py.File(catalog_path, 'r') as hf:
        statuses = hf['metadata/filter_status'][:]
        ids = hf['metadata/event_id'][:]
    passed = [s.decode() if isinstance(s, bytes) else s for s in statuses]
    return ids[np.array(passed) == 'passed']


def get_event_metadata(catalog_path: str, event_ids: Optional[List[int]] = None,
                       min_quality: Optional[str] = None,
                       particle_type: Optional[str] = None) -> Dict:
    """Catalogue metadata for the given identifiers, or for all events."""
    with h5py.File(catalog_path, 'r') as hf:
        meta = hf['metadata']
        all_ids = meta['event_id'][:]
        
        if event_ids is None:
            mask = np.ones(len(all_ids), dtype=bool)
        else:
            mask = np.isin(all_ids, event_ids)
        
        # Filter by quality grade
        if min_quality and 'quality_class' in meta:
            quality_order = {'EXCELLENT': 4, 'GOOD': 3, 'POOR': 2, 'NOISE': 1, 'NOT_ANALYZED': 0}
            min_order = quality_order.get(min_quality.upper(), 0)
            quality_classes = meta['quality_class'][:]
            
            quality_mask = np.array([
                quality_order.get(qc.decode() if isinstance(qc, bytes) else qc, 0) >= min_order
                for qc in quality_classes
            ])
            mask = mask & quality_mask
        
        # Filter by particle label
        if particle_type and 'particle_type' in meta:
            particle_types = meta['particle_type'][:]
            pt_mask = np.array([
                (pt.decode() if isinstance(pt, bytes) else pt) == particle_type
                for pt in particle_types
            ])
            mask = mask & pt_mask
        
        result = {
            'event_id': all_ids[mask],
            'total_energy': meta['total_energy'][mask],
            'n_hits': meta['n_hits'][mask],
            'active_layers': meta['active_layers'][mask],
            'event_time': [x.decode() if isinstance(x, bytes) else x
                           for x in meta['event_time'][mask]],
            'source_file': [x.decode() if isinstance(x, bytes) else x
                            for x in meta['source_file'][mask]],
            'filter_status': [x.decode() if isinstance(x, bytes) else x
                              for x in meta['filter_status'][mask]],
        }
        
        # Quality fields, if the catalogue carries them
        if 'quality_score' in meta:
            result['quality_score'] = meta['quality_score'][mask]
            result['quality_class'] = [x.decode() if isinstance(x, bytes) else x
                                       for x in meta['quality_class'][mask]]
            result['n_tracks'] = meta['n_tracks'][mask]
            result['particle_type'] = [x.decode() if isinstance(x, bytes) else x
                                       for x in meta['particle_type'][mask]]
            result['best_track_theta'] = meta['best_track_theta'][mask]
            result['best_track_chi2'] = meta['best_track_chi2'][mask]
        
        return result

def load_event_arrays_from_raw(catalog_path: str, event_id: int,
                               raw_dir: str, years: Optional[List[int]] = None) -> Optional[Dict]:
    """
    Load the section arrays of one event by re-reading its source .dat file.
    The catalogue holds metadata only, so the amplitudes always come from the
    original file.
    """
    meta = get_event_metadata(catalog_path, [event_id])
    if len(meta['event_id']) == 0:
        logger.error(f"Event {event_id} not found in the catalogue")
        return None

    src_file = meta['source_file'][0]

    # Locate the source file
    candidates = []
    if years:
        for year in years:
            p = os.path.join(raw_dir, str(year), src_file)
            if os.path.exists(p):
                candidates.append(p)
    else:
        candidates = glob.glob(os.path.join(raw_dir, "**", src_file), recursive=True)

    if not candidates:
        logger.error(f"Source file not found: {src_file}")
        return None

    # Find the event with this global_id
    for event in iter_all_events(raw_dir, years=years):
        if event['global_id'] == event_id:
            return event

    return None


# ============================================================
# 4. EVENT SELECTION FOR VISUALIZATION
# ============================================================

def select_events_for_visualization(catalog_path: str, viz_cfg: Dict,
                                     min_quality: Optional[str] = None,
                                     particle_type: Optional[str] = None) -> List[int]:
    """Select event identifiers for visualization, honouring the filters."""
    mode = viz_cfg.get('selection_mode', 'by_id')
    
    if mode == 'by_id':
        ids = viz_cfg.get('event_ids', [])
        logger.info(f"Selected {len(ids)} events by identifier: {ids}")
        return list(ids)

    elif mode == 'by_criteria':
        passed_ids = get_filtered_ids(catalog_path)
        meta = get_event_metadata(
            catalog_path, passed_ids.tolist(),
            min_quality=min_quality,
            particle_type=particle_type
        )

        if len(meta['event_id']) == 0:
            return []

        crit = viz_cfg.get('criteria', {})
        sort_key = crit.get('sort_by', 'total_energy')
        ascending = crit.get('ascending', False)
        limit = crit.get('limit', 10)

        order = np.argsort(meta[sort_key])
        if not ascending:
            order = order[::-1]

        selected = meta['event_id'][order[:limit]].tolist()
        logger.info(f"Selected the top {limit} events by {sort_key}: {selected}")
        return selected

    elif mode == 'first_n':
        passed_ids = get_filtered_ids(catalog_path)
        meta = get_event_metadata(
            catalog_path, passed_ids.tolist(),
            min_quality=min_quality,
            particle_type=particle_type
        )
        n = viz_cfg.get('first_n', 5)
        selected = meta['event_id'][:n].tolist()
        logger.info(f"Selected the first {n} pre-selected events: {selected}")
        return selected

    else:
        raise ValueError(f"Unknown selection_mode: {mode}")
    
def analyze_event_from_raw(event: Dict, cfg: Dict) -> Optional[Dict]:
    """
    Assemble one raw event and run the quality analysis on it.
    
    Args:
        event: a raw event from parse_dat_file()
        cfg: the full configuration
    
    Returns:
        A dictionary of quality metrics, or None on failure.
        {
            'quality_score': float,
            'quality_class': str,
            'n_tracks': int,
            'particle_type': str,
            'best_track_theta': float,
            'best_track_chi2': float,
        }
    """
    try:
        from src import preprocessing, event_quality
        
        # Assemble the sections into planes
        X_layers, X_idx, Y_layers, Y_idx = preprocessing.unify_layers(
            event['arrays'],
            cfg['geometry'].get('expected_channels', {}),
        )
        
        # Basic statistics
        stats = compute_event_stats(event, hit_threshold=cfg.get('filters', {}).get('hit_threshold', 5))
        
        # Quality analysis
        report = event_quality.analyze_event(
            event_id=event['global_id'],
            event_time=event['event_time'],
            X_layers=X_layers,
            Y_layers=Y_layers,
            stats=stats,
            cfg=cfg,
        )
        
        # Metrics of the best track
        best_theta = 0.0
        best_chi2 = 0.0
        if report.tracks:
            best_track = max(report.tracks, key=lambda t: (t.penetration_depth, -t.chi2_ndf))
            best_theta = best_track.theta_deg
            best_chi2 = best_track.chi2_ndf
        
        return {
            'quality_score': report.quality_score,
            'quality_class': report.quality_class,
            'n_tracks': len(report.tracks),
            'particle_type': report.particle_type,
            'best_track_theta': best_theta,
            'best_track_chi2': best_chi2,
        }
    except Exception as e:
        logger.warning(f"Event {event.get('global_id', '?')}: quality analysis failed: {e}")
        return None


def _worker_analyze_chunk(args):
    """
    Multiprocessing worker: takes (events_chunk, cfg) and returns the results.
    """
    events_chunk, cfg = args
    results = []
    for event in events_chunk:
        quality = analyze_event_from_raw(event, cfg)
        if quality:
            quality['event_id'] = event['global_id']
            results.append(quality)
    return results

def _init_hdf5_datasets(hf, meta_grp):
    """Create the resizable HDF5 datasets used for incremental writing."""
    dt_str = h5py.string_dtype(encoding='utf-8')
    
    # Basic metadata
    meta_grp.create_dataset('event_id', shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
    meta_grp.create_dataset('total_energy', shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
    meta_grp.create_dataset('n_hits', shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)
    meta_grp.create_dataset('active_layers', shape=(0,), maxshape=(None,), dtype=np.int8, chunks=True)
    meta_grp.create_dataset('bottom_has_signal', shape=(0,), maxshape=(None,), dtype=np.bool_, chunks=True)
    meta_grp.create_dataset('filter_status', shape=(0,), maxshape=(None,), dtype='S32', chunks=True)
    meta_grp.create_dataset('event_time', shape=(0,), maxshape=(None,), dtype=dt_str, chunks=True)
    meta_grp.create_dataset('source_file', shape=(0,), maxshape=(None,), dtype=dt_str, chunks=True)
    
    # Quality metadata
    meta_grp.create_dataset('quality_score', shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True)
    meta_grp.create_dataset('quality_class', shape=(0,), maxshape=(None,), dtype='S16', chunks=True)
    meta_grp.create_dataset('n_tracks', shape=(0,), maxshape=(None,), dtype=np.int16, chunks=True)
    meta_grp.create_dataset('particle_type', shape=(0,), maxshape=(None,), dtype='S32', chunks=True)
    meta_grp.create_dataset('best_track_theta', shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True)
    meta_grp.create_dataset('best_track_chi2', shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True)

def _append_to_hdf5(hf, stats_list, quality_results):
    """Append a chunk to the HDF5 datasets, resizing them first."""
    meta_grp = hf['metadata']
    current_size = meta_grp['event_id'].shape[0]
    chunk_size = len(stats_list)
    new_size = current_size + chunk_size
    
    # Resize all datasets
    for ds_name in meta_grp.keys():
        meta_grp[ds_name].resize((new_size,))
    
    # Prepare arrays
    ids = [s['event_id'] for s in stats_list]
    energies = [s['total_energy'] for s in stats_list]
    n_hits = [s['n_hits'] for s in stats_list]
    active_layers = [s['active_layers'] for s in stats_list]
    bottom_flags = [s['bottom_has_signal'] for s in stats_list]
    timestamps = [s['event_time'] for s in stats_list]
    source_files = [s['source_file'] for s in stats_list]
    filter_statuses = [s['filter_status'] for s in stats_list]
    
    quality_scores, quality_classes, n_tracks_list = [], [], []
    particle_types, best_thetas, best_chi2s = [], [], []
    
    for s in stats_list:
        eid = s['event_id']
        if s['passed'] and eid in quality_results:
            q = quality_results[eid]
            quality_scores.append(q['quality_score'])
            quality_classes.append(q['quality_class'])
            n_tracks_list.append(q['n_tracks'])
            particle_types.append(q['particle_type'])
            best_thetas.append(q['best_track_theta'])
            best_chi2s.append(q['best_track_chi2'])
        else:
            quality_scores.append(-1.0)
            quality_classes.append('NOT_ANALYZED')
            n_tracks_list.append(-1)
            particle_types.append('unknown')
            best_thetas.append(0.0)
            best_chi2s.append(0.0)
            
    # Write to HDF5
    meta_grp['event_id'][current_size:new_size] = np.array(ids, dtype=np.int64)
    meta_grp['total_energy'][current_size:new_size] = np.array(energies, dtype=np.int64)
    meta_grp['n_hits'][current_size:new_size] = np.array(n_hits, dtype=np.int32)
    meta_grp['active_layers'][current_size:new_size] = np.array(active_layers, dtype=np.int8)
    meta_grp['bottom_has_signal'][current_size:new_size] = np.array(bottom_flags, dtype=np.bool_)
    meta_grp['filter_status'][current_size:new_size] = np.array(filter_statuses, dtype='S32')
    meta_grp['event_time'][current_size:new_size] = timestamps
    meta_grp['source_file'][current_size:new_size] = source_files
    
    meta_grp['quality_score'][current_size:new_size] = np.array(quality_scores, dtype=np.float32)
    meta_grp['quality_class'][current_size:new_size] = np.array(quality_classes, dtype='S16')
    meta_grp['n_tracks'][current_size:new_size] = np.array(n_tracks_list, dtype=np.int16)
    meta_grp['particle_type'][current_size:new_size] = np.array(particle_types, dtype='S32')
    meta_grp['best_track_theta'][current_size:new_size] = np.array(best_thetas, dtype=np.float32)
    meta_grp['best_track_chi2'][current_size:new_size] = np.array(best_chi2s, dtype=np.float32)

def build_catalog_with_quality(
    raw_dir: str,
    output_path: str,
    years: Optional[List[int]] = None,
    filters_cfg: Optional[Dict] = None,
    quality_cfg: Optional[Dict] = None,
    full_cfg: Optional[Dict] = None,
    n_workers: int = 0,
    chunk_size: int = 100,
    events_per_flush: int = 5000,  # flush to HDF5 every N events
) -> Dict:
    """
    build_catalog with the quality analysis attached.

    Events are processed in chunks rather than held in memory, and the results
    are written to HDF5 incrementally so that a long scan survives a failure.
    """
    import multiprocessing as mp
    from tqdm import tqdm

    if not HDF5_AVAILABLE:
        raise RuntimeError("h5py is not installed")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    filters_cfg = filters_cfg or {}
    full_cfg = full_cfg or {}

    if n_workers <= 0:
        n_workers = mp.cpu_count()

    exclude_set = set(full_cfg.get('data', {}).get('exclude_files', []) or [])
    if exclude_set:
        logger.info(f"Good-run selection: flagging events from {len(exclude_set)} files as bad_run")

    logger.info(f"Quality analysis with {n_workers} workers (chunk_size={chunk_size})")
    logger.info(f"Flushing to HDF5 every {events_per_flush} events")

    total_events = 0
    passed_count = 0
    rejected_reasons = {}
    quality_analyzed_count = 0
    
    # Open the HDF5 file once for the whole scan
    with h5py.File(output_path, 'w') as hf:
        meta_grp = hf.create_group('metadata')
        _init_hdf5_datasets(hf, meta_grp)
        
        event_buffer = []
        pbar = tqdm(desc="Scanning & Quality", unit=" events")
        
        # Iterate lazily, without loading the data set into memory
        for event in iter_all_events(raw_dir, years=years):
            event_buffer.append(event)
            total_events += 1
            pbar.update(1)
            
            # Flush the buffer to disk when it fills up
            if len(event_buffer) >= events_per_flush:
                p, r, q = _process_and_flush_buffer(
                    event_buffer, hf, filters_cfg, full_cfg, n_workers, chunk_size, exclude_set
                )
                passed_count += p
                for reason, count in r.items():
                    rejected_reasons[reason] = rejected_reasons.get(reason, 0) + count
                quality_analyzed_count += q
                event_buffer = []  # release the memory
                
        # Flush the remaining events
        if event_buffer:
            p, r, q = _process_and_flush_buffer(
                event_buffer, hf, filters_cfg, full_cfg, n_workers, chunk_size, exclude_set
            )
            passed_count += p
            for reason, count in r.items():
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + count
            quality_analyzed_count += q
            
        pbar.close()

    logger.info(f"Catalogue with quality written: {output_path}")
    logger.info(f"   Total events: {total_events}, passed pre-selection: {passed_count}")
    logger.info(f"   Quality analysed: {quality_analyzed_count}")

    return {
        'total': total_events,
        'passed': passed_count,
        'rejected': rejected_reasons,
        'quality_analyzed': quality_analyzed_count,
        'path': output_path,
    }

def _process_and_flush_buffer(event_buffer, hf, filters_cfg, full_cfg, n_workers, chunk_size, exclude_set=None):
    """Process one buffer of events and append the results to the HDF5 file."""
    import multiprocessing as mp

    exclude_set = exclude_set or set()
    stats_list = []
    passed_count = 0
    rejected_reasons = {}
    
    # 1. Pre-selection (fast, single-threaded)
    for event in event_buffer:
        stats = compute_event_stats(event, filters_cfg.get('hit_threshold', 5))
        # Good-run selection: events from flagged runs are excluded from the
        # analysis but stay in the catalogue with a bad_run status, so that
        # global_id remains stable.
        if event.get('source_file') in exclude_set:
            ok, reason = False, 'bad_run'
        else:
            ok, reason = passes_filters(stats, filters_cfg)
                    
        stats_list.append({
            'event_id': event['global_id'],
            'total_energy': stats['total_energy'],
            'n_hits': stats['n_hits'],
            'active_layers': stats['active_layers'],
            'bottom_has_signal': stats['bottom_has_signal'],
            'event_time': event['event_time'],
            'source_file': event['source_file'],
            'filter_status': reason,
            'passed': ok,
        })
        
        if ok: passed_count += 1
        else: rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
            
    # 2. Quality analysis (parallel)
    passed_events = [ev for ev, st in zip(event_buffer, stats_list) if st['passed']]
    quality_results = {}
    
    if passed_events:
        chunks = [passed_events[i:i + chunk_size] for i in range(0, len(passed_events), chunk_size)]
        worker_args = [(chunk, full_cfg) for chunk in chunks]
        
        if n_workers > 1 and len(chunks) > 1:
            with mp.Pool(processes=n_workers) as pool:
                for result_chunk in pool.imap_unordered(_worker_analyze_chunk, worker_args):
                    for q in result_chunk:
                        quality_results[q['event_id']] = q
        else:
            for args in worker_args:
                result_chunk = _worker_analyze_chunk(args)
                for q in result_chunk:
                    quality_results[q['event_id']] = q
                    
    # 3. Append to the HDF5 file
    _append_to_hdf5(hf, stats_list, quality_results)
    
    return passed_count, rejected_reasons, len(quality_results)

