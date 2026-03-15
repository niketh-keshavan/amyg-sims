#!/usr/bin/env python3
"""
Create Interactive 3D Viewer for fNIRS MC Results
------------------------------------------------
Generates a standalone HTML file with Three.js visualization:
  - 3D head model with tissues
  - Photon paths
  - Detector positions
  - Interactive rotation/zoom

Usage:
    python create_3d_viewer.py --data-dir ../results --output viewer.html
"""

import json
import argparse
import numpy as np
from pathlib import Path
import struct
from tqdm import tqdm

HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>fNIRS MC 3D Viewer</title>
    <style>
        body {{ margin: 0; overflow: hidden; font-family: Arial, sans-serif; }}
        #info {{
            position: absolute; top: 10px; left: 10px;
            background: rgba(0,0,0,0.7); color: white;
            padding: 15px; border-radius: 5px;
            max-width: 300px; font-size: 12px;
        }}
        #controls {{
            position: absolute; top: 10px; right: 10px;
            background: rgba(0,0,0,0.7); color: white;
            padding: 15px; border-radius: 5px;
        }}
        button {{
            margin: 5px; padding: 8px 12px;
            cursor: pointer; background: #444; color: white;
            border: none; border-radius: 3px;
        }}
        button:hover {{ background: #666; }}
        button.active {{ background: #0a0; }}
    </style>
</head>
<body>
    <div id="info">
        <h3>fNIRS MC 3D Viewer</h3>
        <p><b>Mouse:</b> Rotate/Zoom</p>
        <p><b>Tissues:</b> Click to toggle</p>
        <p><b>Photons:</b> {photon_count}</p>
        <p><b>Detectors:</b> {detector_count}</b></p>
    </div>
    
    <div id="controls">
        <button id="btn_scalp" class="active" onclick="toggleLayer('scalp')">Scalp</button><br>
        <button id="btn_skull" class="active" onclick="toggleLayer('skull')">Skull</button><br>
        <button id="btn_csf" class="active" onclick="toggleLayer('csf')">CSF</button><br>
        <button id="btn_gray" class="active" onclick="toggleLayer('gray')">Gray Matter</button><br>
        <button id="btn_white" class="active" onclick="toggleLayer('white')">White Matter</button><br>
        <button id="btn_amygdala" class="active" onclick="toggleLayer('amygdala')">Amygdala</button><br>
        <button id="btn_paths" class="active" onclick="toggleLayer('paths')">Photon Paths</button><br>
        <button id="btn_detectors" class="active" onclick="toggleLayer('detectors')">Detectors</button><br>
        <hr>
        <button onclick="resetView()">Reset View</button>
        <button onclick="toggleAnimation()">Animate</button>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111111);
        
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 1000);
        camera.position.set(150, 100, 150);
        
        const renderer = new THREE.WebGLRenderer({{antialias: true}});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(100, 100, 100);
        scene.add(directionalLight);
        
        // Tissue data
        const tissues = {{
            scalp: {{ color: 0xE8C4A0, opacity: 0.3, data: {scalp_data} }},
            skull: {{ color: 0xF5F5DC, opacity: 0.5, data: {skull_data} }},
            csf: {{ color: 0x87CEEB, opacity: 0.4, data: {csf_data} }},
            gray: {{ color: 0xCD5C5C, opacity: 0.6, data: {gray_data} }},
            white: {{ color: 0xFFFFFF, opacity: 0.7, data: {white_data} }},
            amygdala: {{ color: 0xFF3333, opacity: 0.9, data: {amygdala_data} }}
        }};
        
        // Create point clouds for tissues
        const layers = {{}};
        const pointSize = 0.5;
        
        Object.keys(tissues).forEach(name => {{
            const tissue = tissues[name];
            if (!tissue.data || tissue.data.length === 0) return;
            
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(tissue.data.length * 3);
            const colors = new Float32Array(tissue.data.length * 3);
            
            const color = new THREE.Color(tissue.color);
            
            for (let i = 0; i < tissue.data.length; i++) {{
                positions[i*3] = tissue.data[i][0];
                positions[i*3+1] = tissue.data[i][1];
                positions[i*3+2] = tissue.data[i][2];
                
                colors[i*3] = color.r;
                colors[i*3+1] = color.g;
                colors[i*3+2] = color.b;
            }}
            
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            
            const material = new THREE.PointsMaterial({{
                size: pointSize,
                vertexColors: true,
                opacity: tissue.opacity,
                transparent: true
            }});
            
            layers[name] = new THREE.Points(geometry, material);
            scene.add(layers[name]);
        }});
        
        // Photon paths
        const pathData = {path_data};
        if (pathData && pathData.length > 0) {{
            const pathGeometry = new THREE.BufferGeometry();
            const positions = [];
            const colors = [];
            
            pathData.forEach(path => {{
                const isAmyg = path.amyg;
                const color = isAmyg ? new THREE.Color(0xFF0000) : new THREE.Color(0x4444FF);
                
                for (let i = 0; i < path.points.length - 1; i++) {{
                    positions.push(...path.points[i]);
                    positions.push(...path.points[i+1]);
                    colors.push(color.r, color.g, color.b);
                    colors.push(color.r, color.g, color.b);
                }}
            }});
            
            pathGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            pathGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            const pathMaterial = new THREE.LineBasicMaterial({{
                vertexColors: true,
                opacity: 0.3,
                transparent: true
            }});
            
            layers['paths'] = new THREE.LineSegments(pathGeometry, pathMaterial);
            scene.add(layers['paths']);
        }}
        
        // Detectors
        const detectorData = {detector_data};
        if (detectorData && detectorData.length > 0) {{
            const detGeometry = new THREE.BufferGeometry();
            const positions = new Float32Array(detectorData.length * 3);
            
            for (let i = 0; i < detectorData.length; i++) {{
                positions[i*3] = detectorData[i][0];
                positions[i*3+1] = detectorData[i][1];
                positions[i*3+2] = detectorData[i][2];
            }}
            
            detGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            const detMaterial = new THREE.PointsMaterial({{
                size: 2.0,
                color: 0x00FF00
            }});
            
            layers['detectors'] = new THREE.Points(detGeometry, detMaterial);
            scene.add(layers['detectors']);
        }}
        
        // Toggle layers
        function toggleLayer(name) {{
            if (layers[name]) {{
                layers[name].visible = !layers[name].visible;
                const btn = document.getElementById('btn_' + name);
                if (btn) btn.classList.toggle('active');
            }}
        }}
        
        // Reset view
        function resetView() {{
            camera.position.set(150, 100, 150);
            controls.reset();
        }}
        
        // Animation
        let animating = false;
        function toggleAnimation() {{
            animating = !animating;
        }}
        
        // Render loop
        function animate() {{
            requestAnimationFrame(animate);
            
            if (animating) {{
                scene.rotation.y += 0.005;
            }}
            
            controls.update();
            renderer.render(scene, camera);
        }}
        
        animate();
        
        // Handle resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
'''


def downsample_volume(vol, meta, target_points=100000):
    """Downsample volume for web viewer."""
    nx, ny, nz = meta['nx'], meta['ny'], meta['nz']
    dx = meta['dx']
    
    # Find tissue boundaries
    tissues = {}
    for tissue_id in range(1, 7):  # Skip air
        mask = vol == tissue_id
        indices = np.argwhere(mask)
        
        if len(indices) == 0:
            tissues[tissue_id] = []
            continue
        
        # Downsample
        step = max(1, int(np.sqrt(len(indices) / target_points)))
        sampled = indices[::step]
        
        # Convert to mm coordinates (centered)
        cx, cy, cz = nx * dx * 0.5, ny * dx * 0.5, nz * dx * 0.5
        points = [[float((i - cx)), float((j - cy)), float((k - cz))] 
                  for k, j, i in sampled]
        
        tissues[tissue_id] = points
        print(f"  Tissue {tissue_id}: {len(points)} points (from {len(indices)} voxels)")
    
    return tissues


def load_photon_paths(data_dir, max_paths=1000):
    """Load and downsample photon paths."""
    paths_file = data_dir / "paths_meta_730nm.bin"
    pos_file = data_dir / "paths_pos_730nm.bin"
    
    if not paths_file.exists() or not pos_file.exists():
        return []
    
    # Load metadata
    meta = np.fromfile(paths_file, dtype=np.int32)
    num_paths = len(meta) // 2
    det_ids = meta[0::2]
    path_lens = meta[1::2]
    
    # Load positions
    MAX_STEPS = 2048
    pos = np.fromfile(pos_file, dtype=np.float32)
    positions = pos.reshape(num_paths, MAX_STEPS, 3)
    
    # Center coordinates
    cx, cy, cz = 100, 100, 100  # Adjust based on actual grid
    
    path_data = []
    rng = np.random.default_rng(42)
    indices = rng.choice(num_paths, min(max_paths, num_paths), replace=False)
    
    for idx in indices:
        nsteps = path_lens[idx]
        if nsteps < 2:
            continue
        
        pts = positions[idx, :nsteps, :]
        
        # Center and check amygdala
        xs = pts[:, 0] - cx
        ys = pts[:, 1] - cy
        zs = pts[:, 2] - cz
        
        # Check if through amygdala
        ax_r = (xs - 24) / 5
        ay_r = (ys - (-2)) / 9
        az_r = (zs - (-18)) / 6
        in_amyg = np.any(ax_r**2 + ay_r**2 + az_r**2 <= 1.0)
        
        # Downsample path points
        points = [[float(xs[i]), float(ys[i]), float(zs[i])] 
                  for i in range(0, nsteps, max(1, nsteps // 50))]
        
        path_data.append({'points': points, 'amyg': bool(in_amyg)})
    
    return path_data


def create_viewer(data_dir, output_file):
    """Generate interactive HTML viewer."""
    print("="*60)
    print("Creating Interactive 3D Viewer")
    print("="*60)
    
    data_dir = Path(data_dir)
    
    # Load metadata
    with open(data_dir / "volume_meta.json") as f:
        meta = json.load(f)
    
    # Load results for detector positions
    with open(data_dir / "results_730nm.json") as f:
        results = json.load(f)
    
    # Load volume (if available, else use smaller sample)
    vol_file = data_dir / "volume.bin"
    if vol_file.exists():
        print(f"\nLoading volume ({vol_file.stat().st_size / 1e9:.1f} GB)...")
        vol = np.fromfile(vol_file, dtype=np.uint8)
        vol = vol.reshape(meta['nz'], meta['ny'], meta['nx'])
    else:
        print("\nVolume.bin not found, using ellipsoid approximation...")
        # Create approximate volume
        vol = np.zeros((meta['nz'], meta['ny'], meta['nx']), dtype=np.uint8)
        cx, cy, cz = meta['nx'] // 2, meta['ny'] // 2, meta['nz'] // 2
        
        for z in range(meta['nz']):
            for y in range(meta['ny']):
                for x in range(meta['nx']):
                    dx = (x - cx) / (78 / meta['dx'])
                    dy = (y - cy) / (95 / meta['dx'])
                    dz = (z - cz) / (85 / meta['dx'])
                    if dx*dx + dy*dy + dz*dz <= 1:
                        vol[z, y, x] = 1  # Scalp
    
    print("\nDownsampling tissues for web...")
    tissues = downsample_volume(vol, meta)
    
    print("\nLoading photon paths...")
    paths = load_photon_paths(data_dir)
    print(f"  Loaded {len(paths)} paths")
    
    print("\nExtracting detector positions...")
    detectors = [[d.get('x', 0) - 100, d.get('y', 0) - 100, d.get('z', 0) - 100] 
                 for d in results['detectors']]
    
    # Format for JavaScript
    def fmt_array(arr):
        if not arr:
            return "[]"
        return json.dumps(arr)
    
    # Generate HTML
    html = HTML_TEMPLATE.format(
        photon_count=results['num_photons'],
        detector_count=len(detectors),
        scalp_data=fmt_array(tissues.get(1, [])),
        skull_data=fmt_array(tissues.get(2, [])),
        csf_data=fmt_array(tissues.get(3, [])),
        gray_data=fmt_array(tissues.get(4, [])),
        white_data=fmt_array(tissues.get(5, [])),
        amygdala_data=fmt_array(tissues.get(6, [])),
        path_data=json.dumps(paths),
        detector_data=json.dumps(detectors)
    )
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"\n{'='*60}")
    print(f"Interactive 3D viewer saved to: {output_file}")
    print(f"Open in browser: file://{Path(output_file).absolute()}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Create interactive 3D HTML viewer")
    parser.add_argument("--data-dir", type=str, default="../results",
                       help="Directory with simulation results")
    parser.add_argument("--output", type=str, default="viewer.html",
                       help="Output HTML file")
    args = parser.parse_args()
    
    create_viewer(args.data_dir, args.output)


if __name__ == "__main__":
    main()
