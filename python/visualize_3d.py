#!/usr/bin/env python3
"""
Advanced 3D Visualization for fNIRS Monte Carlo Results
-------------------------------------------------------
Generates publication-quality 3D figures:
  1. Interactive 3D photon paths with tissue surfaces
  2. 3D detector array visualization
  3. Volume rendering of fluence distribution
  4. 3D sensitivity map
  5. Animated photon propagation (optional)

Usage:
    python visualize_3d.py --data-dir ../results --output-dir ../figures
"""

import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TISSUE_COLORS = {
    0: [0.0, 0.0, 0.0, 0.0],      # air
    1: [0.9, 0.7, 0.6, 0.3],      # scalp
    2: [0.95, 0.95, 0.85, 0.5],   # skull
    3: [0.6, 0.8, 1.0, 0.4],      # CSF
    4: [0.8, 0.5, 0.5, 0.6],      # gray matter
    5: [0.95, 0.95, 0.95, 0.7],   # white matter
    6: [1.0, 0.2, 0.2, 1.0],      # amygdala
}

TISSUE_NAMES = ["Air", "Scalp", "Skull", "CSF", "Gray Matter", "White Matter", "Amygdala"]

# Head model ellipsoid parameters (from geometry.cu)
HEAD_ELLIPSOIDS = {
    'scalp': (78.0, 95.0, 85.0),
    'skull_outer': (74.0, 91.0, 81.0),
    'skull_inner_temporal': (71.5, 88.5, 78.5),
    'skull_inner_vertex': (67.0, 84.0, 74.0),
    'csf': (67.0, 84.0, 74.0),
    'gm': (65.5, 82.5, 72.5),
    'wm': (62.0, 79.0, 69.0),
}

AMYGDALA_RIGHT = {
    'center': (24.0, -2.0, -18.0),
    'axes': (5.0, 9.0, 6.0)
}

AMYGDALA_LEFT = {
    'center': (-24.0, -2.0, -18.0),
    'axes': (5.0, 9.0, 6.0)
}


def load_data(data_dir):
    """Load simulation results and metadata."""
    meta_path = data_dir / "volume_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    
    results = {}
    for wl in ["730nm", "850nm"]:
        fp = data_dir / f"results_{wl}.json"
        if fp.exists():
            with open(fp) as f:
                results[wl] = json.load(f)
    
    # Load photon paths if available
    paths = {}
    MAX_PATH_STEPS = 2048
    for wl in ["730nm", "850nm"]:
        meta_fp = data_dir / f"paths_meta_{wl}.bin"
        pos_fp = data_dir / f"paths_pos_{wl}.bin"
        if meta_fp.exists() and pos_fp.exists():
            raw_meta = np.fromfile(meta_fp, dtype=np.int32)
            num_paths = len(raw_meta) // 2
            det_ids = raw_meta[0::2]
            path_lens = raw_meta[1::2]
            raw_pos = np.fromfile(pos_fp, dtype=np.float32)
            positions = raw_pos.reshape(num_paths, MAX_PATH_STEPS, 3)
            valid = path_lens > 0
            paths[wl] = {
                "det_ids": det_ids[valid],
                "path_lens": path_lens[valid],
                "positions": positions[valid],
            }
    
    return meta, results, paths


def create_ellipsoid_mesh(center, axes, n_theta=30, n_phi=30):
    """Create a triangulated ellipsoid mesh."""
    theta = np.linspace(0, 2*np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    
    theta, phi = np.meshgrid(theta, phi)
    
    x = center[0] + axes[0] * np.sin(phi) * np.cos(theta)
    y = center[1] + axes[1] * np.sin(phi) * np.sin(theta)
    z = center[2] + axes[2] * np.cos(phi)
    
    # Create triangles
    vertices = []
    faces = []
    
    for i in range(n_phi - 1):
        for j in range(n_theta - 1):
            # Two triangles per quad
            p1 = [x[i,j], y[i,j], z[i,j]]
            p2 = [x[i+1,j], y[i+1,j], z[i+1,j]]
            p3 = [x[i,j+1], y[i,j+1], z[i,j+1]]
            p4 = [x[i+1,j+1], y[i+1,j+1], z[i+1,j+1]]
            
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])
    
    return faces


def plot_3d_head_model(output_dir):
    """Create a 3D visualization of the head model with all layers."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    center = (0, 0, 0)
    
    # Plot ellipsoid layers (wireframe for clarity)
    layer_colors = {
        'scalp': '#D2B48C',
        'skull_outer': '#DEB887',
        'csf': '#87CEEB',
        'gm': '#CD5C5C',
        'wm': '#F5F5DC',
    }
    
    for name, axes in HEAD_ELLIPSOIDS.items():
        if name in layer_colors:
            # Create wireframe
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x = axes[0] * np.outer(np.cos(u), np.sin(v))
            y = axes[1] * np.outer(np.sin(u), np.sin(v))
            z = axes[2] * np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x, y, z, alpha=0.1, color=layer_colors[name], 
                          linewidth=0, antialiased=True)
            ax.plot_wireframe(x, y, z, alpha=0.2, color=layer_colors[name],
                            rstride=10, cstride=10, linewidth=0.5)
    
    # Plot amygdalae as solid ellipsoids
    for amyg, color in [(AMYGDALA_RIGHT, 'red'), (AMYGDALA_LEFT, 'red')]:
        cx, cy, cz = amyg['center']
        a, b, c = amyg['axes']
        
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = cx + a * np.outer(np.cos(u), np.sin(v))
        y = cy + b * np.outer(np.sin(u), np.sin(v))
        z = cz + c * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, alpha=0.6, color=color, linewidth=0)
    
    # Labels
    ax.set_xlabel('X [mm] (Left-Right)', fontsize=11)
    ax.set_ylabel('Y [mm] (Posterior-Anterior)', fontsize=11)
    ax.set_zlabel('Z [mm] (Inferior-Superior)', fontsize=11)
    ax.set_title('3D Head Model with Amygdala Targets\n(Non-uniform skull: 2.5mm temporal, 7mm vertex)', 
                 fontsize=13, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', alpha=0.5, label=name.replace('_', ' ').title())
                      for name, color in layer_colors.items()]
    legend_elements.append(Patch(facecolor='red', edgecolor='black', alpha=0.6, label='Amygdala'))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Set equal aspect ratio
    max_range = 100
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # View angle
    ax.view_init(elev=15, azim=60)
    
    plt.tight_layout()
    plt.savefig(output_dir / "3d_01_head_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: 3d_01_head_model.png")


def plot_3d_detector_array(results, output_dir):
    """Plot the detector array configuration in 3D."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get detector positions from first wavelength
    r = results["730nm"] if "730nm" in results else list(results.values())[0]
    detectors = r["detectors"]
    
    # Extract positions
    dets_3d = []
    for det in detectors:
        dets_3d.append({
            'id': det['id'],
            'pos': (det['sds_mm'], 0, 0),  # Will compute actual 3D position
            'sds': det['sds_mm'],
            'angle': det.get('angle_deg', 0),
            'x': det.get('x', 0),
            'y': det.get('y', 0),
            'z': det.get('z', 0),
        })
    
    # Plot scalp surface (transparent)
    a, b, c = HEAD_ELLIPSOIDS['scalp']
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='tan', linewidth=0)
    
    # Plot detectors by angle group
    angle_groups = {}
    for det in dets_3d:
        angle = det['angle']
        if angle not in angle_groups:
            angle_groups[angle] = []
        angle_groups[angle].append(det)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(angle_groups)))
    
    for idx, (angle, dets) in enumerate(sorted(angle_groups.items())):
        xs = [d['x'] for d in dets]
        ys = [d['y'] for d in dets]
        zs = [d['z'] for d in dets]
        sds_vals = [d['sds'] for d in dets]
        
        # Color by SDS
        scatter = ax.scatter(xs, ys, zs, c=sds_vals, cmap='viridis', 
                           s=100, alpha=0.8, edgecolors='black', linewidth=0.5,
                           label=f'{angle:+.0f}°')
        
        # Add detector labels
        for d, sds in zip(dets, sds_vals):
            if sds <= 30 or d['id'] % 3 == 0:  # Label some detectors
                ax.text(d['x'], d['y'], d['z'], f'  D{d["id"]}', fontsize=6)
    
    # Plot source position
    # Source is at first detector position minus direction
    ax.scatter([76], [-6], [-57], c='red', s=300, marker='*', 
              edgecolors='black', linewidth=1, label='Source', zorder=5)
    
    # Plot amygdala
    cx, cy, cz = AMYGDALA_RIGHT['center']
    a, b, c = AMYGDALA_RIGHT['axes']
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = cx + a * np.outer(np.cos(u), np.sin(v))
    y = cy + b * np.outer(np.sin(u), np.sin(v))
    z = cz + c * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.3, color='red', linewidth=0)
    ax.text(cx, cy, cz, 'R Amygdala', fontsize=9, color='red')
    
    ax.set_xlabel('X [mm]', fontsize=11)
    ax.set_ylabel('Y [mm]', fontsize=11)
    ax.set_zlabel('Z [mm]', fontsize=11)
    ax.set_title('3D Detector Array Configuration\n(Source positioned over right temporal bone)', 
                 fontsize=13, fontweight='bold')
    
    # Colorbar for SDS
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array([d['sds'] for d in dets_3d])
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Source-Detector Separation [mm]', fontsize=10)
    
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-100, 100])
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "3d_02_detector_array.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: 3d_02_detector_array.png")


def plot_3d_photon_paths(paths, results, output_dir, max_paths=500):
    """Plot photon paths in 3D with tissue context."""
    for wl_key, pdata in paths.items():
        if len(pdata["det_ids"]) == 0:
            continue
            
        fig = plt.figure(figsize=(18, 14))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot tissue surfaces (wireframe)
        for name, axes in [('scalp', HEAD_ELLIPSOIDS['scalp']), 
                          ('skull_outer', HEAD_ELLIPSOIDS['skull_outer'])]:
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = axes[0] * np.outer(np.cos(u), np.sin(v))
            y = axes[1] * np.outer(np.sin(u), np.sin(v))
            z = axes[2] * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_wireframe(x, y, z, alpha=0.1, color='gray', 
                            rstride=5, cstride=5, linewidth=0.3)
        
        # Plot amygdala
        cx, cy, cz = AMYGDALA_RIGHT['center']
        a, b, c = AMYGDALA_RIGHT['axes']
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = cx + a * np.outer(np.cos(u), np.sin(v))
        y = cy + b * np.outer(np.sin(u), np.sin(v))
        z = cz + c * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.3, color='red', linewidth=0)
        
        # Sample paths to plot
        n_paths = len(pdata["det_ids"])
        rng = np.random.default_rng(42)
        indices = rng.choice(n_paths, min(max_paths, n_paths), replace=False)
        
        amyg_count = 0
        for idx in indices:
            nsteps = pdata["path_lens"][idx]
            if nsteps < 2:
                continue
                
            pts = pdata["positions"][idx, :nsteps, :]
            
            # Center coordinates
            cx_head = 100  # meta["nx"] * meta["dx"] / 2
            cy_head = 100
            cz_head = 100
            
            xs = pts[:, 0] - cx_head
            ys = pts[:, 1] - cy_head
            zs = pts[:, 2] - cz_head
            
            # Check if path goes through amygdala
            ax_r = (xs - 24) / 5
            ay_r = (ys - (-2)) / 9
            az_r = (zs - (-18)) / 6
            in_amyg = np.any(ax_r**2 + ay_r**2 + az_r**2 <= 1.0)
            
            if in_amyg:
                color = 'red'
                alpha = 0.6
                linewidth = 1.0
                amyg_count += 1
            else:
                color = 'blue'
                alpha = 0.1
                linewidth = 0.3
            
            ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=linewidth)
        
        # Plot source
        ax.scatter([76-100], [-6-100], [-57-100], c='green', s=200, 
                  marker='*', edgecolors='black', linewidth=1, label='Source')
        
        ax.set_xlabel('X [mm]', fontsize=11)
        ax.set_ylabel('Y [mm]', fontsize=11)
        ax.set_zlabel('Z [mm]', fontsize=11)
        ax.set_title(f'3D Photon Paths - {wl_key}\n({max_paths} paths shown, {amyg_count} reach amygdala)',
                     fontsize=13, fontweight='bold')
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=1, alpha=0.6, label='Reaches amygdala'),
            Line2D([0], [0], color='blue', linewidth=0.5, alpha=0.3, label='Other paths'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='green', 
                  markersize=15, label='Source', linestyle='None'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        ax.set_xlim([-80, 80])
        ax.set_ylim([-80, 80])
        ax.set_zlim([-80, 80])
        ax.view_init(elev=15, azim=60)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"3d_03_photon_paths_{wl_key}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: 3d_03_photon_paths_{wl_key}.png")


def plot_3d_sensitivity_map(results, output_dir):
    """Create a 3D visualization of sensitivity regions."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use 730nm results
    r = results.get("730nm", list(results.values())[0])
    detectors = r["detectors"]
    
    # Plot head surface
    a, b, c = HEAD_ELLIPSOIDS['scalp']
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='tan', linewidth=0)
    
    # Plot sensitivity regions as ellipsoids from each detector
    # Approximate: sensitive region is an ellipsoid between source and detector
    for det in detectors:
        if abs(det.get('angle_deg', 0)) > 5:  # Only primary direction
            continue
        if det['sds_mm'] > 40:  # Skip longest SDS
            continue
            
        # Get best gate amygdala sensitivity
        gates = det.get('time_gates', [])
        best_sens = 0
        for gate in gates:
            ppl = gate.get('partial_pathlength_mm', {})
            amyg = ppl.get('amygdala', 0)
            total = sum(ppl.values()) if ppl else 1
            sens = amyg / total if total > 0 else 0
            if sens > best_sens:
                best_sens = sens
        
        if best_sens < 0.001:  # Skip low sensitivity
            continue
        
        # Detector position
        dx, dy, dz = det.get('x', 0), det.get('y', 0), det.get('z', 0)
        
        # Draw sensitivity "bubble"
        # Size proportional to sensitivity
        size = best_sens * 20  # Scale factor
        
        # Draw sphere at detector
        u_s = np.linspace(0, 2*np.pi, 10)
        v_s = np.linspace(0, np.pi, 10)
        xs = dx + size * np.outer(np.cos(u_s), np.sin(v_s))
        ys = dy + size * np.outer(np.sin(u_s), np.sin(v_s))
        zs = dz + size * np.outer(np.ones(np.size(u_s)), np.cos(v_s))
        
        color = plt.cm.YlOrRd(best_sens / 0.02)  # Normalize to 2% max
        ax.plot_surface(xs, ys, zs, alpha=0.3, color=color, linewidth=0)
    
    # Plot amygdala
    cx, cy, cz = AMYGDALA_RIGHT['center']
    a, b, c = AMYGDALA_RIGHT['axes']
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = cx + a * np.outer(np.cos(u), np.sin(v))
    y = cy + b * np.outer(np.sin(u), np.sin(v))
    z = cz + c * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.5, color='red', linewidth=0, label='Amygdala')
    ax.text(cx, cy, cz, 'R Amygdala', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('X [mm]', fontsize=11)
    ax.set_ylabel('Y [mm]', fontsize=11)
    ax.set_zlabel('Z [mm]', fontsize=11)
    ax.set_title('3D Sensitivity Map\n(Bubble size = amygdala sensitivity)', 
                 fontsize=13, fontweight='bold')
    
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([-100, 100])
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "3d_04_sensitivity_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: 3d_04_sensitivity_map.png")


def plot_cross_section_with_paths(paths, results, output_dir):
    """Plot 2D cross-sections with overlaid photon paths."""
    for wl_key, pdata in paths.items():
        if len(pdata["det_ids"]) == 0:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Sagittal view (X-Z plane at Y=-2)
        ax = axes[0]
        
        # Draw tissue boundaries
        y_slice = -2
        for name, (a, b, c) in HEAD_ELLIPSOIDS.items():
            if name in ['scalp', 'skull_outer', 'csf', 'gm', 'wm']:
                # Ellipse at Y = y_slice
                # (x/a)^2 + (y_slice/b)^2 + (z/c)^2 = 1
                # => (x/a)^2 + (z/c)^2 = 1 - (y_slice/b)^2
                rhs = 1 - (y_slice / b)**2
                if rhs > 0:
                    theta = np.linspace(0, 2*np.pi, 100)
                    x_ell = a * np.sqrt(rhs) * np.cos(theta)
                    z_ell = c * np.sqrt(rhs) * np.sin(theta)
                    ax.plot(x_ell, z_ell, 'k-', alpha=0.3, linewidth=0.5)
        
        # Draw amygdala
        cx, cy, cz = AMYGDALA_RIGHT['center']
        a, b, c = AMYGDALA_RIGHT['axes']
        theta = np.linspace(0, 2*np.pi, 100)
        # At y = -2, check if inside amygdala
        y_rel = (y_slice - cy) / b
        if abs(y_rel) <= 1:
            x_amyg = cx + a * np.sqrt(1 - y_rel**2) * np.cos(theta)
            z_amyg = cz + c * np.sqrt(1 - y_rel**2) * np.sin(theta)
            ax.fill(x_amyg, z_amyg, color='red', alpha=0.3)
            ax.plot(x_amyg, z_amyg, 'r-', linewidth=2)
        
        # Plot photon paths (sagittal projection)
        n_paths = len(pdata["det_ids"])
        rng = np.random.default_rng(42)
        indices = rng.choice(n_paths, min(300, n_paths), replace=False)
        
        for idx in indices:
            nsteps = pdata["path_lens"][idx]
            if nsteps < 2:
                continue
            pts = pdata["positions"][idx, :nsteps, :]
            
            # Project to Y = -2 plane (or just use X-Z)
            xs = pts[:, 0] - 100
            zs = pts[:, 2] - 100
            
            ax.plot(xs, zs, 'b-', alpha=0.1, linewidth=0.3)
        
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Z [mm]')
        ax.set_title(f'Sagittal View (Y={y_slice}mm) - {wl_key}')
        ax.set_aspect('equal')
        ax.set_xlim([-80, 80])
        ax.set_ylim([-80, 80])
        ax.grid(True, alpha=0.3)
        
        # Coronal view (Y-Z plane at X=24)
        ax = axes[1]
        x_slice = 24
        
        # Draw tissue boundaries
        for name, (a, b, c) in HEAD_ELLIPSOIDS.items():
            if name in ['scalp', 'skull_outer', 'csf', 'gm', 'wm']:
                rhs = 1 - (x_slice / a)**2
                if rhs > 0:
                    theta = np.linspace(0, 2*np.pi, 100)
                    y_ell = b * np.sqrt(rhs) * np.cos(theta)
                    z_ell = c * np.sqrt(rhs) * np.sin(theta)
                    ax.plot(y_ell, z_ell, 'k-', alpha=0.3, linewidth=0.5)
        
        # Draw amygdala
        cx, cy, cz = AMYGDALA_RIGHT['center']
        a, b, c = AMYGDALA_RIGHT['axes']
        x_rel = (x_slice - cx) / a
        if abs(x_rel) <= 1:
            y_amyg = cy + b * np.sqrt(1 - x_rel**2) * np.cos(theta)
            z_amyg = cz + c * np.sqrt(1 - x_rel**2) * np.sin(theta)
            ax.fill(y_amyg, z_amyg, color='red', alpha=0.3)
            ax.plot(y_amyg, z_amyg, 'r-', linewidth=2)
        
        # Plot photon paths
        for idx in indices:
            nsteps = pdata["path_lens"][idx]
            if nsteps < 2:
                continue
            pts = pdata["positions"][idx, :nsteps, :]
            
            ys = pts[:, 1] - 100
            zs = pts[:, 2] - 100
            
            ax.plot(ys, zs, 'b-', alpha=0.1, linewidth=0.3)
        
        ax.set_xlabel('Y [mm]')
        ax.set_ylabel('Z [mm]')
        ax.set_title(f'Coronal View (X={x_slice}mm) - {wl_key}')
        ax.set_aspect('equal')
        ax.set_xlim([-80, 80])
        ax.set_ylim([-80, 80])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"3d_05_cross_sections_{wl_key}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: 3d_05_cross_sections_{wl_key}.png")


def main():
    parser = argparse.ArgumentParser(
        description="3D Visualization for fNIRS Monte Carlo Results")
    parser.add_argument("--data-dir", type=str, default="../results",
                       help="Directory containing simulation results")
    parser.add_argument("--output-dir", type=str, default="../figures",
                       help="Output directory for figures")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("3D Visualization for fNIRS Monte Carlo")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    meta, results, paths = load_data(data_dir)
    print(f"\nLoaded results for wavelengths: {list(results.keys())}")
    print(f"Loaded paths for: {list(paths.keys())}")
    
    # Generate visualizations
    print("\nGenerating 3D visualizations...")
    
    plot_3d_head_model(output_dir)
    
    if results:
        plot_3d_detector_array(results, output_dir)
        plot_3d_sensitivity_map(results, output_dir)
    
    if paths:
        plot_3d_photon_paths(paths, results, output_dir)
        plot_cross_section_with_paths(paths, results, output_dir)
    
    print("\n" + "="*70)
    print("3D visualization complete!")
    print(f"Figures saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
