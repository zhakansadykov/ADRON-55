import os
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def plot_academic_histograms(X_layers, Y_layers, event_time, output_dir, 
                            expected_channels=None, dpi=300):
    """Plot the X and Y projections as histograms, one panel per plane."""
    os.makedirs(output_dir, exist_ok=True)
    
    if expected_channels is None:
        expected_channels = {
            'row_1': len(X_layers[3]), 'row_3': len(X_layers[2]),
            'row_5': len(X_layers[1]), 'row_7': len(X_layers[0]),
            'row_2': len(Y_layers[3]), 'row_4': len(Y_layers[2]),
            'row_6': len(Y_layers[1]), 'row_8': len(Y_layers[0]),
        }
    
    # === X projection ===
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
    layer_names_x = ['Row 1 (Top, 50 ch)', 'Row 3 (48 ch)', 'Row 5 (48 ch)', 'Row 7 (Bottom, 48 ch)']
    x_order = [3, 2, 1, 0]  # plane 1 on top, plane 7 at the bottom
    x_expected = [expected_channels['row_1'], expected_channels['row_3'],
                  expected_channels['row_5'], expected_channels['row_7']]
    
    for plot_i, data_i in enumerate(x_order):
        data = X_layers[data_i]
        axs[plot_i].bar(np.arange(len(data)), data, color='dimgray', edgecolor='black', linewidth=0.3)
        axs[plot_i].set_ylabel('Signal')
        axs[plot_i].grid(True, linestyle='--', alpha=0.6)
        axs[plot_i].text(0.02, 0.8, layer_names_x[plot_i], transform=axs[plot_i].transAxes,
                         fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        axs[plot_i].set_xlim(-1, x_expected[plot_i])
        axs[plot_i].set_ylim(bottom=0)
    
    axs[-1].set_xlabel('Channel Number (X projection)')
    fig.suptitle(f'Ionization Calorimeter — X Projection | Event: {event_time}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hist_X_projection.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # === Y projection ===
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
    layer_names_y = ['Row 2 (Top, 69 ch)', 'Row 4 (72 ch)', 'Row 6 (72 ch)', 'Row 8 (Bottom, 72 ch)']
    y_order = [3, 2, 1, 0]
    y_expected = [expected_channels['row_2'], expected_channels['row_4'],
                  expected_channels['row_6'], expected_channels['row_8']]
    
    for plot_i, data_i in enumerate(y_order):
        data = Y_layers[data_i]
        axs[plot_i].bar(np.arange(len(data)), data, color='dimgray', edgecolor='black', linewidth=0.3)
        axs[plot_i].set_ylabel('Signal')
        axs[plot_i].grid(True, linestyle='--', alpha=0.6)
        axs[plot_i].text(0.02, 0.8, layer_names_y[plot_i], transform=axs[plot_i].transAxes,
                         fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        axs[plot_i].set_xlim(-1, y_expected[plot_i])
        axs[plot_i].set_ylim(bottom=0)
    
    axs[-1].set_xlabel('Channel Number (Y projection)')
    fig.suptitle(f'Ionization Calorimeter — Y Projection | Event: {event_time}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hist_Y_projection.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Histograms written to {output_dir}/")

def plot_event_3d(event_id, tracks_3d, vertices, particle_type, cfg, out_path):
    """Build an interactive 3D view with the tracks, clipped to the detector, and the vertices."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fig = go.Figure()
    heights = cfg['geometry']['heights']
    det_half_x = 4320.0  # half width in X (72 ch * 120 mm / 2)
    det_half_y = 4140.0  # half width in Y (69 ch * 120 mm / 2)
    
    # 1. Detector frame
    z_min_det, z_max_det = min(heights), max(heights)
    for h in heights:
        fig.add_trace(go.Scatter3d(
            x=[-det_half_x, det_half_x, det_half_x, -det_half_x, -det_half_x],
            y=[-det_half_y, -det_half_y, det_half_y, det_half_y, -det_half_y],
            z=[h, h, h, h, h],
            mode='lines', line=dict(color='gray', width=3), opacity=0.6, showlegend=False
        ))
    
    # 2. Tracks
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    for i, t3d in enumerate(tracks_3d):
        color = colors[i % len(colors)]
        
        # Vertex this track belongs to
        vertex_for_track = None
        for v in vertices:
            if t3d.track_id in v.track_ids:
                vertex_for_track = v
                break
                
        # --- solid line inside the detector volume only ---
        z_line_det = np.linspace(z_min_det, z_max_det, 100)
        x_line_det = t3d.slope_x * z_line_det + t3d.intercept_x
        y_line_det = t3d.slope_y * z_line_det + t3d.intercept_y
        
        fig.add_trace(go.Scatter3d(
            x=x_line_det, y=y_line_det, z=z_line_det,
            mode='lines', line=dict(color=color, width=5),
            name=f'Track {t3d.track_id} (θ={t3d.zenith_deg:.1f}°)'
        ))
        
        # --- dashed line up to the vertex, if it lies outside ---
        if vertex_for_track and vertex_for_track.position[2] > z_max_det:
            z_vert = np.linspace(z_max_det, vertex_for_track.position[2], 50)
            x_vert = t3d.slope_x * z_vert + t3d.intercept_x
            y_vert = t3d.slope_y * z_vert + t3d.intercept_y
            fig.add_trace(go.Scatter3d(
                x=x_vert, y=y_vert, z=z_vert,
                mode='lines', line=dict(color=color, width=2, dash='dash'),
                showlegend=False
            ))
            
        # Hit positions
        x_hits = [h[0] for h in t3d.hits_3d]
        y_hits = [h[1] for h in t3d.hits_3d]
        z_hits = [h[2] for h in t3d.hits_3d]
        fig.add_trace(go.Scatter3d(
            x=x_hits, y=y_hits, z=z_hits,
            mode='markers', marker=dict(size=6, color=color), showlegend=False
        ))
    
    # 3. Vertices
    for v in vertices:
        fig.add_trace(go.Scatter3d(
            x=[v.position[0]], y=[v.position[1]], z=[v.position[2]],
            mode='markers+text',
            marker=dict(size=12, color='yellow', symbol='diamond', line=dict(color='black', width=2)),
            text=[f'V{v.vertex_id} ({v.n_tracks}t, {v.position[2]/1000:.1f}km)'],
            textposition='top center', name=f'Vertex {v.vertex_id}'
        ))
    
    # Fix the axis ranges so that the tracks stay in view
    all_z = list(heights) + [v.position[2] for v in vertices]
    z_min_plot = min(0, min(all_z))
    z_max_plot = max(all_z) * 1.1
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (mm)', range=[-det_half_x*1.5, det_half_x*1.5]),
            yaxis=dict(title='Y (mm)', range=[-det_half_y*1.5, det_half_y*1.5]),
            zaxis=dict(title='Z (mm)', range=[z_min_plot, z_max_plot]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=2) # stretch Z for legibility
        ),
        title=f'Event {event_id} | {particle_type} | {len(tracks_3d)} tracks',
        width=1200, height=900
    )
    
    fig.write_html(out_path)
    logger.info(f"3D view written to {out_path}")