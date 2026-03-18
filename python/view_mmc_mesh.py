#!/usr/bin/env python3
"""
View MMC Mesh - Create Interactive 3D HTML Viewer
-------------------------------------------------
Generates a standalone HTML file with Three.js visualization of the MMC tetrahedral mesh.

Usage:
    python view_mmc_mesh.py --mesh mni152_head.mmcmesh --output mesh_viewer.html
    python view_mmc_mesh.py --mesh mni152_head.mmcmesh --max-tets 100000 --output mesh_viewer.html

The viewer runs locally in your browser - no server needed.
"""

import struct
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MMC Mesh 3D Viewer</title>
    <style>
        body {{ margin: 0; overflow: hidden; font-family: Arial, sans-serif; background: #111; }}
        #info {{
            position: absolute; top: 10px; left: 10px;
            background: rgba(0,0,0,0.8); color: white;
            padding: 15px; border-radius: 5px;
            max-width: 350px; font-size: 12px;
            z-index: 100;
        }}
        #controls {{
            position: absolute; top: 10px; right: 10px;
            background: rgba(0,0,0,0.8); color: white;
            padding: 15px; border-radius: 5px;
            z-index: 100;
        }}
        button {{
            margin: 3px; padding: 8px 12px;
            cursor: pointer; background: #444; color: white;
            border: none; border-radius: 3px; font-size: 11px;
            width: 120px;
        }}
        button:hover {{ background: #666; }}
        button.active {{ background: #2a2; }}
        button.inactive {{ background: #444; }}
        h3 {{ margin: 0 0 10px 0; color: #0f0; }}
        .stat {{ color: #ff0; }}
        .label {{ color: #aaa; }}
    </style>
</head>
<body>
    <div id="info">
        <h3>MMC Mesh Viewer</h3>
        <p><span class="label">Nodes:</span> <span class="stat">{num_nodes:,}</span></p>
        <p><span class="label">Tetrahedra:</span> <span class="stat">{num_elements:,}</span></p>
        <p><span class="label">Displayed:</span> <span class="stat">{displayed_tets:,}</span></p>
        <p><span class="label">Tissues:</span></p>
        <p style="margin-left: 10px;">
            <span style="color: #ff6666;">[ ]</span> Scalp<br>
            <span style="color: #dddddd;">[ ]</span> Skull<br>
            <span style="color: #66ccff;">[ ]</span> CSF<br>
            <span style="color: #ff99cc;">[ ]</span> Gray Matter<br>
            <span style="color: #eeeeee;">[ ]</span> White Matter<br>
            <span style="color: #ffcc00;">[ ]</span> <b>Amygdala</b>
        </p>
        <hr style="border-color: #444;">
        <p><b>Controls:</b></p>
        <p>• Left drag: Rotate</p>
        <p>• Right drag: Pan</p>
        <p>• Scroll: Zoom</p>
        <p>• Click buttons: Toggle tissues</p>
    </div>
    
    <div id="controls">
        <button id="btn_scalp" class="active" onclick="toggleLayer(1)">Scalp</button><br>
        <button id="btn_skull" class="active" onclick="toggleLayer(2)">Skull</button><br>
        <button id="btn_csf" class="active" onclick="toggleLayer(3)">CSF</button><br>
        <button id="btn_gray" class="active" onclick="toggleLayer(4)">Gray</button><br>
        <button id="btn_white" class="active" onclick="toggleLayer(5)">White</button><br>
        <button id="btn_amygdala" class="active" onclick="toggleLayer(6)"><b>Amygdala</b></button><br>
        <hr style="border-color: #444;">
        <button id="btn_wireframe" onclick="toggleWireframe()">Wireframe</button><br>
        <button onclick="resetView()">Reset View</button><br>
        <button onclick="toggleRotation()">Auto-Rotate</button><br>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x050505);
        scene.fog = new THREE.Fog(0x050505, 100, 400);
        
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 1000);
        camera.position.set(120, 80, 120);
        
        const renderer = new THREE.WebGLRenderer({{antialias: true}});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        document.body.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, -10, 0);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);
        
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(100, 100, 50);
        dirLight.castShadow = true;
        scene.add(dirLight);
        
        const dirLight2 = new THREE.DirectionalLight(0x6666ff, 0.3);
        dirLight2.position.set(-100, 50, -100);
        scene.add(dirLight2);
        
        // Tissue colors
        const tissueColors = {{
            1: 0xff6666,  // Scalp - red
            2: 0xdddddd,  // Skull - gray
            3: 0x66ccff,  // CSF - light blue
            4: 0xff99cc,  // Gray matter - pink
            5: 0xeeeeee,  // White matter - white
            6: 0xffcc00   // Amygdala - gold
        }};
        
        const tissueOpacity = {{
            1: 0.3,   // Scalp - transparent
            2: 0.5,   // Skull
            3: 0.4,   // CSF
            4: 0.6,   // Gray
            5: 0.5,   // White
            6: 1.0    // Amygdala - opaque
        }};
        
        // Create mesh from tetrahedra
        const layers = {{}};
        let wireframeMode = false;
        
        function createTetrahedronMesh(nodes, elements, tissues, tissueFilter) {{
            const geometries = [];
            
            for (let i = 0; i < elements.length; i++) {{
                const tissue = tissues[i];
                if (tissueFilter && tissue !== tissueFilter) continue;
                
                const tet = elements[i];
                const v0 = new THREE.Vector3(...nodes[tet[0]]);
                const v1 = new THREE.Vector3(...nodes[tet[1]]);
                const v2 = new THREE.Vector3(...nodes[tet[2]]);
                const v3 = new THREE.Vector3(...nodes[tet[3]]);
                
                // Create tetrahedron geometry (4 triangles)
                const geom = new THREE.BufferGeometry();
                const vertices = [];
                const normals = [];
                
                // Helper to add triangle with normal
                function addTriangle(a, b, c) {{
                    vertices.push(...a, ...b, ...c);
                    const normal = new THREE.Vector3();
                    const ab = new THREE.Vector3().subVectors(b, a);
                    const ac = new THREE.Vector3().subVectors(c, a);
                    normal.crossVectors(ab, ac).normalize();
                    normals.push(...normal, ...normal, ...normal);
                }}
                
                // 4 faces of tetrahedron
                addTriangle(v0, v2, v1);  // face opposite v3
                addTriangle(v0, v1, v3);  // face opposite v2
                addTriangle(v0, v3, v2);  // face opposite v1
                addTriangle(v1, v2, v3);  // face opposite v0
                
                geom.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
                geom.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
                geometries.push({{geom, tissue}});
            }}
            
            return geometries;
        }}
        
        // Load mesh data
        const nodes = {nodes_json};
        const elements = {elements_json};
        const tissues = {tissues_json};
        
        // Create layers for each tissue
        const tissueNames = {{1: 'scalp', 2: 'skull', 3: 'csf', 4: 'gray', 5: 'white', 6: 'amygdala'}};
        
        Object.keys(tissueNames).forEach(tid => {{
            const tissueId = parseInt(tid);
            const name = tissueNames[tid];
            const geoms = createTetrahedronMesh(nodes, elements, tissues, tissueId);
            
            if (geoms.length === 0) return;
            
            // Merge geometries for this tissue
            const mergedGeom = new THREE.BufferGeometry();
            const allVertices = [];
            const allNormals = [];
            
            geoms.forEach(g => {{
                allVertices.push(...g.geom.attributes.position.array);
                allNormals.push(...g.geom.attributes.normal.array);
            }});
            
            mergedGeom.setAttribute('position', new THREE.Float32BufferAttribute(allVertices, 3));
            mergedGeom.setAttribute('normal', new THREE.Float32BufferAttribute(allNormals, 3));
            
            const material = new THREE.MeshPhongMaterial({{
                color: tissueColors[tissueId],
                transparent: tissueOpacity[tissueId] < 1.0,
                opacity: tissueOpacity[tissueId],
                side: THREE.DoubleSide,
                shininess: 30
            }});
            
            const mesh = new THREE.Mesh(mergedGeom, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            layers[name] = mesh;
            scene.add(mesh);
        }});
        
        // Add coordinate axes
        const axesHelper = new THREE.AxesHelper(50);
        scene.add(axesHelper);
        
        // Toggle layer visibility
        function toggleLayer(tissueId) {{
            const name = tissueNames[tissueId];
            if (layers[name]) {{
                layers[name].visible = !layers[name].visible;
                const btn = document.getElementById('btn_' + name);
                btn.classList.toggle('active');
                btn.classList.toggle('inactive');
            }}
        }}
        
        // Toggle wireframe
        function toggleWireframe() {{
            wireframeMode = !wireframeMode;
            Object.values(layers).forEach(mesh => {{
                mesh.material.wireframe = wireframeMode;
            }});
            document.getElementById('btn_wireframe').classList.toggle('active');
        }}
        
        // Reset view
        function resetView() {{
            camera.position.set(120, 80, 120);
            controls.target.set(0, -10, 0);
            controls.reset();
        }}
        
        // Auto-rotation
        let autoRotate = false;
        function toggleRotation() {{
            autoRotate = !autoRotate;
            controls.autoRotate = autoRotate;
        }}
        
        // Render loop
        function animate() {{
            requestAnimationFrame(animate);
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


def load_mmcmesh(mesh_path, max_tets=None):
    """Load MMC mesh from binary file."""
    with open(mesh_path, 'rb') as f:
        # Header (48 bytes)
        magic = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('I', f.read(4))[0]
        num_nodes = struct.unpack('I', f.read(4))[0]
        num_elems = struct.unpack('I', f.read(4))[0]
        bbox = struct.unpack('6f', f.read(24))
        
        print(f"Loading mesh: {num_nodes:,} nodes, {num_elems:,} elements")
        
        # Read nodes
        nodes = []
        for i in range(num_nodes):
            x, y, z = struct.unpack('3f', f.read(12))
            nodes.append([x, y, z])
        
        # Read elements
        elements = []
        for i in range(num_elems):
            v0, v1, v2, v3 = struct.unpack('4i', f.read(16))
            elements.append([v0, v1, v2, v3])
        
        # Read tissues
        tissues = struct.unpack(f'{num_elems}i', f.read(num_elems * 4))
        
        print(f"  Tissue distribution:")
        tissue_names = {0: 'Air', 1: 'Scalp', 2: 'Skull', 3: 'CSF', 
                       4: 'Gray', 5: 'White', 6: 'Amygdala'}
        for tid in range(7):
            count = sum(1 for t in tissues if t == tid)
            if count > 0:
                print(f"    {tissue_names.get(tid, 'Unknown')}: {count:,} tets")
    
    # Downsample if requested
    if max_tets and num_elems > max_tets:
        print(f"\nDownsampling to {max_tets:,} tets...")
        
        # Always include all amygdala tets
        amyg_indices = [i for i, t in enumerate(tissues) if t == 6]
        other_indices = [i for i, t in enumerate(tissues) if t != 6]
        
        # Sample from other tissues
        n_other = min(len(other_indices), max_tets - len(amyg_indices))
        if n_other > 0:
            sampled = np.random.choice(other_indices, n_other, replace=False)
            keep_indices = list(amyg_indices) + list(sampled)
        else:
            keep_indices = amyg_indices[:max_tets]
        
        keep_indices = sorted(keep_indices)
        elements = [elements[i] for i in keep_indices]
        tissues = [tissues[i] for i in keep_indices]
        print(f"  Displaying: {len(elements):,} tets (all {len(amyg_indices)} amygdala)")
    
    return nodes, elements, tissues


def create_mesh_viewer(mesh_path, output_file, max_tets=50000):
    """Generate interactive HTML viewer for MMC mesh."""
    print("="*60)
    print("Creating MMC Mesh 3D Viewer")
    print("="*60)
    
    # Load mesh
    nodes, elements, tissues = load_mmcmesh(mesh_path, max_tets)
    
    # Convert to JSON format for JavaScript
    nodes_json = json.dumps(nodes)
    elements_json = json.dumps(elements)
    tissues_json = json.dumps(list(tissues))
    
    # Generate HTML
    html = HTML_TEMPLATE.format(
        num_nodes=len(nodes),
        num_elements=len(elements),
        displayed_tets=len(elements),
        nodes_json=nodes_json,
        elements_json=elements_json,
        tissues_json=tissues_json
    )
    
    # Write output with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n{'='*60}")
    print(f"Viewer saved to: {output_file}")
    print(f"Open in browser: file://{Path(output_file).absolute()}")
    print(f"{'='*60}")
    print("\nControls:")
    print("  - Left drag: Rotate")
    print("  - Right drag: Pan")
    print("  - Scroll: Zoom")
    print("  - Buttons: Toggle tissue visibility")


def main():
    parser = argparse.ArgumentParser(description='Create 3D viewer for MMC mesh')
    parser.add_argument('--mesh', default='mni152_head.mmcmesh',
                       help='Input MMC mesh file')
    parser.add_argument('--output', default='mesh_viewer.html',
                       help='Output HTML file')
    parser.add_argument('--max-tets', type=int, default=50000,
                       help='Maximum tetrahedra to display (default: 50000)')
    
    args = parser.parse_args()
    
    if not Path(args.mesh).exists():
        print(f"Error: Mesh file not found: {args.mesh}")
        return 1
    
    create_mesh_viewer(args.mesh, args.output, args.max_tets)
    return 0


if __name__ == '__main__':
    exit(main())
