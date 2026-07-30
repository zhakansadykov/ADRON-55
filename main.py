import argparse
import os
import sys
import yaml
import logging
import numpy as np
from src import data_loader, preprocessing, event_quality, tracker, visualizer, exporter, physics

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[::]:
        root_logger.removeHandler(handler)
    fh = logging.FileHandler('logs/scan_full.log', encoding='utf-8', mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
    root_logger.addHandler(fh)

def load_config(path="config/settings.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_catalog(cfg):
    data_cfg = cfg['data']
    catalog_path = os.path.join(data_cfg['processed_dir'], data_cfg['catalog_file'])
    if not os.path.exists(catalog_path):
        print("=" * 60)
        print("EVENT CATALOGUE NOT FOUND")
        print("=" * 60)
        print(f"Expected file: {catalog_path}\nBuild it first: python main.py scan")
        sys.exit(1)
    return catalog_path

# ============================================================
# COMMANDS
# ============================================================

def cmd_scan(cfg):
    data_cfg = cfg['data']
    catalog_path = os.path.join(data_cfg['processed_dir'], data_cfg['catalog_file'])
    print("=" * 60)
    print("SCAN: building the event catalogue")
    print("=" * 60)
    
    if cfg.get('event_quality', {}).get('enabled', False):
        result = data_loader.build_catalog_with_quality(
            raw_dir=data_cfg['raw_dir'], output_path=catalog_path,
            years=data_cfg.get('years'), filters_cfg=cfg.get('filters', {}),
            full_cfg=cfg, n_workers=cfg['event_quality']['parallel'].get('n_workers', 0)
        )
    else:
        result = data_loader.build_catalog(
            raw_dir=data_cfg['raw_dir'], output_path=catalog_path,
            years=data_cfg.get('years'), filters_cfg=cfg.get('filters', {})
        )
    print(f"Total: {result['total']}, passed pre-selection: {result['passed']}")

def cmd_inspect(cfg, args):
    catalog_path = ensure_catalog(cfg)
    data_cfg = cfg['data']
    event_ids = [int(x.strip()) for x in args.ids.split(",")] if args.ids else [0]
    
    for eid in event_ids:
        event = data_loader.load_event_arrays_from_raw(catalog_path, eid, data_cfg['raw_dir'], data_cfg.get('years'))
        if event is None: continue
        stage, out_fmt = args.stage, args.output
        out_dir = "output/inspection"
        
        if stage == "raw":
            if out_fmt == "terminal": exporter.print_raw_to_terminal(event)
            else: exporter.export_raw_to_txt(event, os.path.join(out_dir, f"event_{eid:06d}_raw.{out_fmt}"))
        elif stage == "processed":
            X_layers, _, Y_layers, _ = preprocessing.unify_layers(event['arrays'], cfg['geometry'].get('expected_channels', {}))
            if out_fmt == "terminal": exporter.print_processed_to_terminal(X_layers, Y_layers, event['event_time'])
            else: exporter.export_processed_to_txt(X_layers, Y_layers, event['event_time'], os.path.join(out_dir, f"event_{eid:06d}_processed.{out_fmt}"), cfg)
        elif stage == "quality":
            X_layers, _, Y_layers, _ = preprocessing.unify_layers(event['arrays'], cfg['geometry'].get('expected_channels', {}))
            stats = data_loader.compute_event_stats(event, hit_threshold=cfg['filters']['hit_threshold'])
            report = event_quality.analyze_event(event['global_id'], event['event_time'], X_layers, Y_layers, stats, cfg)
            if out_fmt == "terminal": exporter.print_quality_report(report)
            else: exporter.export_quality_report(report, os.path.join(out_dir, f"event_{eid:06d}_quality.{out_fmt}"), fmt='json' if out_fmt=='json' else 'txt')

def cmd_visualize(cfg, args):
    catalog_path = ensure_catalog(cfg)
    data_cfg = cfg['data']
    event_ids = [int(x.strip()) for x in args.ids.split(",")] if args.ids else [0]
    out_dir = args.save_dir if args.save_dir else "output/visualize"
    
    for eid in event_ids:
        event = data_loader.load_event_arrays_from_raw(catalog_path, eid, data_cfg['raw_dir'], data_cfg.get('years'))
        if event is None: continue
        X_layers, _, Y_layers, _ = preprocessing.unify_layers(event['arrays'], cfg['geometry'].get('expected_channels', {}))
        event_out_dir = os.path.join(out_dir, f"event_{eid:06d}")
        
        if args.type in ['2d', 'both']:
            visualizer.plot_academic_histograms(X_layers, Y_layers, event['event_time'], event_out_dir, expected_channels=cfg['geometry'].get('expected_channels'))
        if args.type in ['3d', 'both']:
            stats = data_loader.compute_event_stats(event, hit_threshold=cfg['filters']['hit_threshold'])
            report = event_quality.analyze_event(event['global_id'], event['event_time'], X_layers, Y_layers, stats, cfg)
            res_3d = tracker.reconstruct_3d(report, cfg)
            if res_3d['tracks_3d']:
                visualizer.plot_event_3d(eid, res_3d['tracks_3d'], res_3d['vertices'], res_3d['particle_type'], cfg, os.path.join(event_out_dir, f"event_{eid:06d}_3d.html"))

def cmd_reconstruct(cfg, args):
    catalog_path = ensure_catalog(cfg)
    data_cfg = cfg['data']
    
    if args.ids:
        selected_ids = [int(x.strip()) for x in args.ids.split(",")]
    else:
        viz_cfg = cfg.get('visualization', {})
        selected_ids = data_loader.select_events_for_visualization(catalog_path, viz_cfg, min_quality=args.min_quality, particle_type=args.particle_type)
            
    if not selected_ids: return
    
    print(f"\n3D reconstruction of {len(selected_ids)} events...")
    results = []
    
    for eid in selected_ids:
        event = data_loader.load_event_arrays_from_raw(catalog_path, eid, data_cfg['raw_dir'], data_cfg.get('years'))
        if event is None: continue
        
        X_layers, _, Y_layers, _ = preprocessing.unify_layers(event['arrays'], cfg['geometry'].get('expected_channels', {}))
        stats = data_loader.compute_event_stats(event, hit_threshold=cfg['filters']['hit_threshold'])
        report = event_quality.analyze_event(event['global_id'], event['event_time'], X_layers, Y_layers, stats, cfg)
        res_3d = tracker.reconstruct_3d(report, cfg)
        
        best_zenith, best_azimuth, best_chi2, ra, dec = float('nan'), float('nan'), float('nan'), None, None
        vert_z = 0.0
        
        if res_3d['tracks_3d']:
            best_track = max(res_3d['tracks_3d'], key=lambda t: len(t.hits_3d))
            best_zenith = best_track.zenith_deg
            best_azimuth = best_track.azimuth_deg
            best_chi2 = best_track.chi2_3d
            
            # Equatorial coordinates
            ra, dec = physics.altaz_to_radec(best_zenith, best_azimuth, event['event_time'], cfg)
            
        if res_3d['vertices']:
            vert_z = max(res_3d['vertices'], key=lambda v: v.n_tracks).position[2]
            
        # Energy on the (uncalibrated) MeV scale
        energy_mev = physics.calibrate_energy(stats['total_energy'], cfg)
            
        results.append({
            'event_id': eid, 'particle_type': report.particle_type,
            'n_tracks_3d': len(res_3d['tracks_3d']), 'best_zenith_deg': best_zenith,
            'best_azimuth_deg': best_azimuth, 'best_chi2_ndf': best_chi2,
            'vertex_z_mm': vert_z, 'total_energy_adc': stats['total_energy'],
            'total_energy_mev': energy_mev, 'ra_deg': ra, 'dec_deg': dec
        })
        print(f"  ID {eid}: theta={best_zenith:.1f} deg | phi={best_azimuth:.1f} deg | RA={ra:.2f} deg Dec={dec:.2f} deg"
              if ra else f"  ID {eid}: equatorial conversion failed")

    if args.save_csv:
        exporter.export_reco_to_csv(results, args.save_csv)

def cmd_find_showcase(cfg, args):
    """Search for a display event: several tracks converging on one vertex."""
    catalog_path = ensure_catalog(cfg)
    data_cfg = cfg['data']
    
    # Top-N EXCELLENT events by deposited energy
    viz_cfg = {'selection_mode': 'by_criteria', 'criteria': {'sort_by': 'total_energy', 'limit': args.limit}}
    selected_ids = data_loader.select_events_for_visualization(catalog_path, viz_cfg, min_quality='EXCELLENT')
    
    print(f"\nSearching for a showcase event among the top {len(selected_ids)} events...")
    best_eid, best_score = None, 0
    
    for eid in selected_ids:
        event = data_loader.load_event_arrays_from_raw(catalog_path, eid, data_cfg['raw_dir'], data_cfg.get('years'))
        if event is None: continue
        
        X_layers, _, Y_layers, _ = preprocessing.unify_layers(event['arrays'], cfg['geometry'].get('expected_channels', {}))
        stats = data_loader.compute_event_stats(event, hit_threshold=cfg['filters']['hit_threshold'])
        report = event_quality.analyze_event(event['global_id'], event['event_time'], X_layers, Y_layers, stats, cfg)
        res_3d = tracker.reconstruct_3d(report, cfg)
        
        # Score the event for display quality
        if len(res_3d['tracks_3d']) >= 2 and res_3d['vertices']:
            # Vertex with the most tracks and the smallest spread
            best_v = max(res_3d['vertices'], key=lambda v: v.n_tracks)
            if best_v.n_tracks >= 2:
                # 10 points per track in the vertex, minus 1 per 100 mm of spread
                score = (best_v.n_tracks * 10) - (best_v.spread_mm / 100.0)
                if score > best_score:
                    best_score = score
                    best_eid = eid
                    print(f"  Candidate ID={eid}: {best_v.n_tracks} tracks, spread={best_v.spread_mm:.0f} mm (score={score:.1f})")

    if best_eid:
        print(f"\nBest showcase event: ID={best_eid}")
        print(f"   To plot it: python main.py visualize --ids {best_eid} --type 3d")
    else:
        print("\nNo showcase event found. Try increasing --limit.")

# ============================================================
# ENTRY POINT
# ============================================================

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="ADRON-55 ionization calorimeter: event catalogue, inspection, "
                    "visualization and three-dimensional track reconstruction.")
    parser.add_argument("command", choices=["scan", "inspect", "visualize", "reconstruct", "find-showcase"],
                        help="scan: build the event catalogue; inspect: dump one event; "
                             "visualize: plot events; reconstruct: 3D tracks; "
                             "find-showcase: search for a multi-track event")
    parser.add_argument("--config", default="config/settings.yaml",
                        help="path to the configuration file")
    parser.add_argument("--ids", type=str,
                        help="comma-separated event identifiers, e.g. 0,100,500")
    parser.add_argument("--min-quality", choices=["GOOD", "EXCELLENT", "POOR"],
                        help="minimum quality grade of the selected events")
    parser.add_argument("--particle-type", choices=["penetrating_muon", "hadron_shower", "em_shower", "neutral_candidate"],
                        help="restrict the selection to one classifier label "
                             "(diagnostic only, not used for published results)")

    parser.add_argument("--stage", choices=["raw", "processed", "quality"], default="raw",
                        help="(inspect) processing stage to dump")
    parser.add_argument("--output", choices=["terminal", "txt", "json"], default="terminal",
                        help="(inspect) output format")
    parser.add_argument("--type", choices=["2d", "3d", "both"], default="both",
                        help="(visualize) 2D histograms, interactive 3D view, or both")
    parser.add_argument("--save-dir", default="output/visualize",
                        help="(visualize) directory for the generated plots")
    parser.add_argument("--save-csv", default=None,
                        help="(reconstruct) write the results to this CSV file")
    parser.add_argument("--limit", type=int, default=500,
                        help="(find-showcase) how many of the highest-energy events to test")
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    if args.command == "scan": cmd_scan(cfg)
    elif args.command == "inspect": cmd_inspect(cfg, args)
    elif args.command == "visualize": cmd_visualize(cfg, args)
    elif args.command == "reconstruct": cmd_reconstruct(cfg, args)
    elif args.command == "find-showcase": cmd_find_showcase(cfg, args)

if __name__ == "__main__":
    main()