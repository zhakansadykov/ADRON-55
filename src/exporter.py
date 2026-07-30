import os
import json
import csv
import logging
from typing import Dict, List, Optional
from src.event_quality import EventQualityReport

logger = logging.getLogger(__name__)

def export_raw_to_txt(event: Dict, out_path: str) -> None:
    """Write a raw event to a human-readable text file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"=== RAW EVENT DUMP ===\n")
        f.write(f"Event ID: {event.get('global_id', 'N/A')}\n")
        f.write(f"Time:     {event.get('event_time', 'N/A')}\n")
        f.write(f"Source:   {event.get('source_file', 'N/A')}\n")
        f.write(f"Header:   {event.get('header', 'N/A')}\n")
        f.write("="*40 + "\n\n")
        
        for name, arr in event.get('arrays', {}).items():
            f.write(f"--- {name} ({len(arr)} channels) ---\n")
            # Ten values per line for readability
            for i in range(0, len(arr), 10):
                chunk = arr[i:i+10]
                f.write(" ".join(f"{x:5d}" for x in chunk) + "\n")
            f.write("\n")

def print_raw_to_terminal(event: Dict) -> None:
    """Print the raw event to the terminal (first ten values of each section)."""
    print("\n" + "="*50)
    print(f"EVENT ID: {event.get('global_id', 'N/A')} | Time: {event.get('event_time', 'N/A')}")
    print("="*50)
    
    for name, arr in event.get('arrays', {}).items():
        preview = " ".join(f"{x:5d}" for x in arr[:10])
        print(f"  {name:<15} (len={len(arr):<3}): {preview} ...")

def export_processed_to_txt(X_layers, Y_layers, event_time: str, out_path: str, cfg: Dict) -> None:
    """Write the assembled planes to a text file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    heights = cfg['geometry']['heights']
    x_rows = [7, 5, 3, 1] # bottom to top
    y_rows = [8, 6, 4, 2]
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"=== PROCESSED (STITCHED) EVENT DUMP ===\n")
        f.write(f"Time: {event_time}\n\n")
        
        f.write("--- X PROJECTION ---\n")
        for i, layer in enumerate(X_layers):
            row_num = x_rows[i]
            z = heights[row_num-1]
            f.write(f"Row {row_num} (Z={z}mm, {len(layer)}ch): {list(layer)}\n")
            
        f.write("\n--- Y PROJECTION ---\n")
        for i, layer in enumerate(Y_layers):
            row_num = y_rows[i]
            z = heights[row_num-1]
            f.write(f"Row {row_num} (Z={z}mm, {len(layer)}ch): {list(layer)}\n")

def print_processed_to_terminal(X_layers, Y_layers, event_time: str) -> None:
    """Print the assembled planes to the terminal."""
    print("\n" + "="*50)
    print(f"PROCESSED EVENT | Time: {event_time}")
    print("="*50)
    print(" X PROJECTION (Bottom to Top):")
    for i, layer in enumerate(X_layers):
        print(f"  Row {[7,5,3,1][i]}: {list(layer[:10])}...")
    print(" Y PROJECTION (Bottom to Top):")
    for i, layer in enumerate(Y_layers):
        print(f"  Row {[8,6,4,2][i]}: {list(layer[:10])}...")

def export_quality_report(report: EventQualityReport, out_path: str, fmt: str = 'json') -> None:
    """Write the quality report as JSON or plain text."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    data = {
        'event_id': report.event_id,
        'event_time': report.event_time,
        'quality_score': report.quality_score,
        'quality_class': report.quality_class,
        'particle_type': report.particle_type,
        'veto_flags': report.veto.flags_list,
        'topology': {
            'n_hits': report.topology.n_hits,
            'n_active_layers': report.topology.n_active_layers,
            'n_clusters': report.topology.n_clusters,
        },
        'tracks': []
    }
    
    for t in report.tracks:
        data['tracks'].append({
            'id': t.track_id, 'projection': t.projection,
            'penetration_depth': t.penetration_depth, 'chi2_ndf': t.chi2_ndf,
            'theta_deg': t.theta_deg, 'phi_deg': t.phi_deg
        })
        
    if fmt == 'json':
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"Event {report.event_id} ({report.event_time})\n")
            f.write(f"Quality: {report.quality_class} ({report.quality_score:.1f})\n")
            f.write(f"Particle: {report.particle_type}\n")
            f.write(f"Veto: {report.veto.flags_list}\n")
            f.write(f"Tracks: {len(report.tracks)}\n")

def print_quality_report(report: EventQualityReport) -> None:
    """Print the quality report to the terminal."""
    print("\n" + "="*50)
    print(f"QUALITY REPORT | Event {report.event_id}")
    print("="*50)
    print(f"Class: {report.quality_class} ({report.quality_score:.1f}/100)")
    print(f"Particle: {report.particle_type}")
    print(f"Veto: {report.veto.flags_list if report.veto.any_flag else 'Passed'}")
    print(f"Topology: {report.topology.n_clusters} clusters, {report.topology.n_active_layers} layers")
    print(f"Tracks found: {len(report.tracks)}")
    for t in report.tracks:
        print(f"  - Track {t.track_id} ({t.projection}): depth={t.penetration_depth}, chi2={t.chi2_ndf:.2f}, theta={t.theta_deg:.1f}°")

def export_reco_to_csv(results: List[Dict], out_path: str) -> None:
    """Write the 3D reconstruction results to CSV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not results:
        logger.warning("No results to write to CSV")
        return
        
    # The headers must match the keys produced by main.py::cmd_reconstruct;
    # otherwise extrasaction='ignore' silently drops columns (ra_deg, dec_deg, energy in MeV).
    headers = ['event_id', 'particle_type', 'n_tracks_3d', 'best_zenith_deg',
               'best_azimuth_deg', 'best_chi2_ndf', 'vertex_z_mm',
               'total_energy_adc', 'total_energy_mev', 'ra_deg', 'dec_deg']

    missing = set(headers) - set(results[0].keys())
    if missing:
        logger.warning(f"Fields absent from the results: {sorted(missing)} - those columns will be empty")


    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        for res in results:
            writer.writerow(res)
    logger.info(f"Reconstruction results written to {out_path}")