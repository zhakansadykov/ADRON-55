"""
reconstruct_all.py - streaming 3D reconstruction of the selected sample.

Replaces the quadratic behaviour of the `reconstruct` command, which re-read the
whole raw data set for every event. Here a SINGLE pass over the raw files
reconstructs each event inline, and the equatorial coordinates are computed
VECTORIALLY at the end, which is faster and avoids hammering the IERS tables.

Events are taken from the catalogue with filter_status == 'passed' and a
quality_class of at least --min-quality. The result is written to reco.csv.

The energy column (energy_mev) is PROVISIONAL: the absolute scale still has to
be anchored to the minimum-ionizing peak. It does not affect the angles
(zenith, azimuth, RA, Dec).

Run:
    python analysis/reconstruct_all.py --config config/settings.yaml --min-quality GOOD
    python analysis/reconstruct_all.py --config config/settings.yaml --limit 2000   # smoke test
"""
import os
import sys
import csv
import argparse
import contextlib
import io
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_loader, preprocessing, event_quality, tracker, physics

QUALITY_ORDER = {'EXCELLENT': 4, 'GOOD': 3, 'POOR': 2, 'NOISE': 1, 'NOT_ANALYZED': 0}


def _decode(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def load_target_ids(catalog_path, min_quality):
    """The set of global_id values to reconstruct (passed and quality >= min)."""
    import h5py
    thr = QUALITY_ORDER.get(min_quality, 3)
    targets = {}
    with h5py.File(catalog_path, 'r') as hf:
        meta = hf['metadata']
        ids = meta['event_id'][:]
        status = meta['filter_status'][:]
        has_q = 'quality_class' in meta
        qcls = meta['quality_class'][:] if has_q else None
        for i in range(len(ids)):
            if _decode(status[i]) != 'passed':
                continue
            q = _decode(qcls[i]) if has_q else 'NOT_ANALYZED'
            if QUALITY_ORDER.get(q, 0) >= thr:
                targets[int(ids[i])] = q
    return targets


def reconstruct_one(event, cfg):
    """Reconstruct one event. Returns a result row, or None."""
    expected = cfg['geometry'].get('expected_channels', {})
    # unify_layers prints the plane lengths; silence stdout so that 60k events
    # do not flood the terminal
    with contextlib.redirect_stdout(io.StringIO()):
        X_layers, _, Y_layers, _ = preprocessing.unify_layers(event['arrays'], expected)

    stats = data_loader.compute_event_stats(
        event, hit_threshold=cfg['filters']['hit_threshold'])
    report = event_quality.analyze_event(
        event['global_id'], event['event_time'], X_layers, Y_layers, stats, cfg)
    res = tracker.reconstruct_3d(report, cfg)

    zenith = azimuth = chi2 = float('nan')
    n_tracks = len(res['tracks_3d'])
    n_rows = 0
    if res['tracks_3d']:
        best = max(res['tracks_3d'], key=lambda t: len(t.hits_3d))
        zenith, azimuth, chi2 = best.zenith_deg, best.azimuth_deg, best.chi2_3d
        n_rows = len(set(w[2] for w in best.hits_weights))

    energy_adc = stats['total_energy']
    energy_mev = physics.calibrate_energy(energy_adc, cfg)

    return {
        'event_id': event['global_id'],
        'event_time': event['event_time'],
        'particle_type': report.particle_type,
        'n_tracks_3d': n_tracks,
        'n_rows': n_rows,
        'zenith_deg': zenith,
        'azimuth_deg': azimuth,
        'chi2_3d': chi2,
        'energy_adc': energy_adc,
        'energy_mev': energy_mev,
    }


def add_radec_vectorized(rows, cfg):
    """Add ra_deg and dec_deg vectorially from (zenith, azimuth, time).
    Rows with a NaN angle are skipped."""
    for r in rows:
        r['ra_deg'], r['dec_deg'] = float('nan'), float('nan')
    valid = [r for r in rows if not math.isnan(r['zenith_deg'])]
    if not valid:
        return rows

    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u

    loc_cfg = cfg['location']
    location = EarthLocation(lat=loc_cfg['latitude'] * u.deg,
                             lon=loc_cfg['longitude'] * u.deg,
                             height=loc_cfg['altitude'] * u.m)
    offset = loc_cfg.get('detector_angle_offset', 0)

    times = Time([r['event_time'] for r in valid], format='iso', scale='utc')
    alt = np.array([90.0 - r['zenith_deg'] for r in valid]) * u.deg
    az = np.array([(r['azimuth_deg'] + offset) % 360.0 for r in valid]) * u.deg
    altaz = AltAz(alt=alt, az=az, obstime=times, location=location)
    icrs = SkyCoord(altaz).transform_to('icrs')
    for r, ra, dec in zip(valid, icrs.ra.deg, icrs.dec.deg):
        r['ra_deg'], r['dec_deg'] = float(ra), float(dec)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Streaming 3D reconstruction of the whole data set")
    ap.add_argument('--config', default='config/settings.yaml')
    ap.add_argument('--min-quality', default='GOOD', choices=list(QUALITY_ORDER.keys()))
    ap.add_argument('--limit', type=int, default=0, help="limit the number of events (smoke test); 0 = all")
    ap.add_argument('--out', default='output/paper_plots/reco.csv')
    args = ap.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    catalog = os.path.join(data_cfg['processed_dir'], data_cfg['catalog_file'])
    if not os.path.exists(catalog):
        raise SystemExit(f"Catalogue not found: {catalog}. Build it first: python main.py scan")

    print(f"Selecting events from the catalogue (passed & quality>={args.min_quality})...")
    targets = load_target_ids(catalog, args.min_quality)
    print(f"To reconstruct: {len(targets)} events")

    rows, done, n_tracks_ok = [], 0, 0
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(targets), desc="Reconstruct")
    except ImportError:
        pbar = None

    for ev in data_loader.iter_all_events(data_cfg['raw_dir'], years=data_cfg.get('years')):
        gid = ev['global_id']
        if gid not in targets:
            continue
        try:
            row = reconstruct_one(ev, cfg)
        except Exception as e:
            print(f"  event {gid}: reconstruction failed: {e}")
            row = None
        if row is not None:
            rows.append(row)
            if not math.isnan(row['zenith_deg']):
                n_tracks_ok += 1
        done += 1
        if pbar:
            pbar.update(1)
        if args.limit and done >= args.limit:
            break
    if pbar:
        pbar.close()

    print(f"Reconstructed: {done}, with a 3D track: {n_tracks_ok}")
    print("Computing RA/Dec (vectorized, astropy)...")
    rows = add_radec_vectorized(rows, cfg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    headers = ['event_id', 'event_time', 'particle_type', 'n_tracks_3d', 'n_rows',
               'zenith_deg', 'azimuth_deg', 'chi2_3d', 'energy_adc', 'energy_mev',
               'ra_deg', 'dec_deg']
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"Written: {args.out}  ({len(rows)} rows)")


if __name__ == '__main__':
    main()