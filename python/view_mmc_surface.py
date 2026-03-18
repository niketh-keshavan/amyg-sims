#!/usr/bin/env python3
"""
View MMC Mesh Surface - Extract and visualize boundary surfaces only
---------------------------------------------------------------------
Shows the actual head shape by extracting exterior faces (neighbors == -1).

Usage:
    python view_mmc_surface.py --mesh mni152_head.mmcmesh --output surface_viewer.html

Features:
    - Scalp outer surface (actual head shape)
    - Amygdala surface (deep target)
    - Optional: interior tissue boundaries
"""

import struct
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MMC Surface Viewer</title>
    <style>
        body {{ margin: 0; overflow: hidden; font-family: Arial, sans-serif; background: #050505; }}
        #info {{
            position: absolute; top: 10px; left: 10px;
            background: rgba(0,0,0,0.85); color: white;
            padding: 15px; border-radius: 5px;
            max-width: 300px; font-size: 12px;
            z-index: 100;
            border: 1px solid #333;
        }}
        #controls {{
            position: absolute; top: 10px; right: 10px;
            background: rgba(0,0,0,0.85); color: white;
            padding: 15px; border-radius: 5px;
            z-index: 100;
            border: 1px solid #333;
        }}
        button {{
            margin: 3px; padding: 8px 12px;
            cursor: pointer; background: #444; color: white;
            border: none; border-radius: 3px; font-size: 11px;
            width: 140px;
            transition: background 0.2s;
        }}
        button:hover {{ background: #666; }}
        button.active {{ background: #2a2; }}
        button.inactive {{ background: #444; }}
        h3 {{ margin: 0 0 10px 0; color: #0f0; }}
        h4 {{ margin: 10px 0 5px 0; color: #aaa; font-size: 11px; }}
        .stat {{ color: #ff0; }}
        .label {{ color: #aaa; }}
        .color-box {{ display: inline-block; width: 12px; height: 12px; margin-right: 5px; border: 1px solid #666; }}
    </style>
</head>
<body>
    <div id="info">
        <h3>MMC Surface Viewer</h3>
        <p><span class="label">Nodes:</span> <span class="stat">{num_nodes:,}</span></p>
        <p><span class="label">Tets:</span> <span class="stat">{num_elements:,}</span></p>
        <p><span class="label">Boundary Triangles:</span> <span class="stat">{num_tris:,}</span></p>
        
        <h4>Exterior Surfaces</h4>
        <p style="margin-left: 5px; font-size: 11px;">
            <span class="color-box" style="background: #ff6666;"></span>Scalp (outer)<br>
            <span class="color-box" style="background: #ffcc00;"></span><b>Amygdala</b><br>
        </p>
        
        <h4>Interior Boundaries</h4>
        <p style="margin-left: 5px; font-size: 11px;">
            <span class="color-box" style="background: #dddddd;"></span>Skull<br>
            <span class="color-box" style="background: #66ccff;"></span>CSF<br>
            <span class="color-box" style="background: #ff99cc;"></span>Gray Matter<br>
            <span class="color-box" style="background: #eeeeee;"></span>White Matter
        </p>
        
        <hr style="border-color: #444; margin: 10px 0;">
        <p><b>Controls:</b></p>
        <p style="font-size: 11px; margin: 3px 0;">• Left drag: Rotate</p>
        <p style="font-size: 11px; margin: 3px 0;">• Right drag: Pan</p>
        <p style="font-size: 11px; margin: 3px 0;">• Scroll: Zoom</p>
        <p style="font-size: 11px; margin: 3px 0;">• Buttons: Toggle surfaces</p>
    </div>
    
    <div id="controls">
        <h4>Exterior Surfaces</h4>
        <button id="btn_scalp_ext" class="active" onclick="toggleLayer('scalp_ext')">Scalp (Outer)</button><br>
        <button id="btn_amygdala" class="active" onclick="toggleLayer('amygdala')"><b>Amygdala</b></button><br>
        
        <h4>Interior Boundaries</h4>
        <button id="btn_scalp_int" class="inactive" onclick="toggleLayer('scalp_int')">Scalp (Inner)</button><br>
        <button id="btn_skull" class="inactive" onclick="toggleLayer('skull')">Skull</button><br>
        <button id="btn_csf" class="inactive" onclick="toggleLayer('csf')">CSF</button><br>
        <button id="btn_gray" class="inactive" onclick="toggleLayer('gray')">Gray Matter</button><br>
        <button id="btn_white" class="inactive" onclick="toggleLayer('white')">White Matter</button><br>
        
        <hr style="border-color: #444; margin: 10px 0;">
        <button onclick="toggleWireframe()">Toggle Wireframe</button><br>
        <button onclick="resetView()">Reset View</button><br>
        <button onclick="toggleRotation()">Auto-Rotate</button><br>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x050505);
        
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 1000);
        camera.position.set(100, 0, 100);
        
        const renderer = new THREE.WebGLRenderer({{antialias: true}});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, -5, -10);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.7);
        scene.add(ambientLight);
        
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(100, 100, 50);
        scene.add(dirLight);
        
        const dirLight2 = new THREE.DirectionalLight(0x6666ff, 0.3);
        dirLight2.position.set(-100, 50, -100);
        scene.add(dirLight2);
        
        const dirLight3 = new THREE.DirectionalLight(0xff6666, 0.2);
        dirLight3.position.set(0, -100, 0);
        scene.add(dirLight3);
        
        // Layer colors and settings
        const layerConfig = {{
            'scalp_ext': {{ color: 0xff6666, opacity: 0.9, name: 'Scalp (Exterior)' }},
            'scalp_int': {{ color: 0xcc4444, opacity: 0.3, name: 'Scalp (Interior)' }},
            'skull':     {{ color: 0xdddddd, opacity: 0.4, name: 'Skull' }},
            'csf':       {{ color: 0x66ccff, opacity: 0.3, name: 'CSF' }},
            'gray':      {{ color: 0xff99cc, opacity: 0.5, name: 'Gray Matter' }},
            'white':     {{ color: 0xeeeeee, opacity: 0.5, name: 'White Matter' }},
            'amygdala':  {{ color: 0xffcc00, opacity: 1.0, name: 'Amygdala', emissive: 0x332200 }}
        }};
        
        const layers = {{}};
        let wireframeMode = false;
        
        // Load surface data
        const surfaces = {surfaces_json};
        
        // Create mesh for each surface
        Object.keys(surfaces).forEach(name => {{
            const data = surfaces[name];
            if (!data || data.vertices.length === 0) return;
            
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.vertices, 3));
            geometry.setAttribute('normal', new THREE.Float32BufferAttribute(data.normals, 3));
            
            const config = layerConfig[name];
            const material = new THREE.MeshPhongMaterial({{
                color: config.color,
                transparent: config.opacity < 1.0,
                opacity: config.opacity,
                side: THREE.DoubleSide,
                shininess: 30,
                emissive: config.emissive || 0x000000,
                emissiveIntensity: 0.3
            }});
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            layers[name] = mesh;
            scene.add(mesh);
        }});
        
        // Add coordinate axes
        const axesHelper = new THREE.AxesHelper(30);
        axesHelper.position.set(-80, -80, -50);
        scene.add(axesHelper);
        
        // Toggle layer
        function toggleLayer(name) {{
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
        }}
        
        // Reset view
        function resetView() {{
            camera.position.set(100, 0, 100);
            controls.target.set(0, -5, -10);
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


def compute_neighbors(elements):
    """Compute tet-to-tet neighbor connectivity from element faces."""
    print("Computing neighbor connectivity...")
    
    # Build face -> element mapping
    # Face is represented as sorted tuple of 3 vertex indices
    face_to_elem = {}  # face_key -> [(elem_idx, face_idx), ...]
    
    for elem_idx, elem in enumerate(elements):
        for face_idx in range(4):
            fv = FACE_VERTS[face_idx]
            # Get face vertices
            v0, v1, v2 = elem[fv[0]], elem[fv[1]], elem[fv[2]]
            # Sort to get canonical face representation
            face_key = tuple(sorted([v0, v1, v2]))
            
            if face_key not in face_to_elem:
                face_to_elem[face_key] = []
            face_to_elem[face_key].append((elem_idx, face_idx))
    
    # Build neighbor arrays
    num_elems = len(elements)
    neighbors = [[-1, -1, -1, -1] for _ in range(num_elems)]
    
    boundary_count = 0
    shared_count = 0
    for face_key, elem_list in face_to_elem.items():
        if len(elem_list) == 2:
            # Shared face between two elements
            elem1, face1 = elem_list[0]
            elem2, face2 = elem_list[1]
            neighbors[elem1][face1] = elem2
            neighbors[elem2][face2] = elem1
            shared_count += 1
        elif len(elem_list) == 1:
            # Boundary face
            boundary_count += 1
        else:
            # Should not happen in valid mesh
            print(f"  Warning: face {face_key} has {len(elem_list)} elements")
    
    print(f"  {boundary_count} boundary faces, {shared_count} shared faces")
    return neighbors


def load_mmcmesh(mesh_path):
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
        
        # Neighbors are NOT in the file - compute them
        neighbors = compute_neighbors(elements)
        
    return nodes, elements, tissues, neighbors


# Face vertex indices (opposite vertex k)
FACE_VERTS = [
    [1, 2, 3],  # face 0 (opposite vertex 0)
    [0, 2, 3],  # face 1 (opposite vertex 1)
    [0, 1, 3],  # face 2 (opposite vertex 2)
    [0, 1, 2],  # face 3 (opposite vertex 3)
]


def compute_face_normal(v0, v1, v2):
    """Compute outward-facing normal for triangle."""
    ax = v1[0] - v0[0]
    ay = v1[1] - v0[1]
    az = v1[2] - v0[2]
    
    bx = v2[0] - v0[0]
    by = v2[1] - v0[1]
    bz = v2[2] - v0[2]
    
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    
    length = (nx*nx + ny*ny + nz*nz) ** 0.5
    if length > 0:
        nx /= length
        ny /= length
        nz /= length
    
    return [nx, ny, nz]


def extract_boundary_surfaces(nodes, elements, tissues, neighbors):
    """
    Extract boundary surfaces for each tissue.
    
    For each tissue, find faces that are:
    - Exterior: neighbor is -1 (air)
    - Interior: neighbor is different tissue
    """
    
    # Group faces by (tissue, neighbor_tissue)
    # We want faces where tissue != neighbor_tissue
    surfaces = defaultdict(list)
    
    print("Extracting boundary surfaces...")
    
    for elem_idx, (elem, tissue, neighbor) in enumerate(zip(elements, tissues, neighbors)):
        for face_idx in range(4):
            neighbor_elem = neighbor[face_idx]
            
            # Get face vertices
            fv = FACE_VERTS[face_idx]
            v0 = nodes[elem[fv[0]]]
            v1 = nodes[elem[fv[1]]]
            v2 = nodes[elem[fv[2]]]
            
            # Compute normal (roughly outward)
            normal = compute_face_normal(v0, v1, v2)
            
            if neighbor_elem < 0:
                # Exterior boundary (faces air)
                if tissue == 1:  # Scalp exterior
                    surfaces['scalp_ext'].append((v0, v1, v2, normal))
                elif tissue == 6:  # Amygdala exterior (unlikely, amygdala is deep)
                    surfaces['amygdala'].append((v0, v1, v2, normal))
            else:
                # Interior boundary
                neighbor_tissue = tissues[neighbor_elem]
                if neighbor_tissue != tissue:
                    # This is an interface between two tissues
                    if tissue == 1 and neighbor_tissue == 2:  # Scalp-Skull
                        surfaces['scalp_int'].append((v0, v1, v2, normal))
                    elif tissue == 2 and neighbor_tissue == 3:  # Skull-CSF
                        surfaces['skull'].append((v0, v1, v2, normal))
                    elif tissue == 3 and neighbor_tissue == 4:  # CSF-Gray
                        surfaces['csf'].append((v0, v1, v2, normal))
                    elif tissue == 4 and neighbor_tissue == 5:  # Gray-White
                        surfaces['gray'].append((v0, v1, v2, normal))
                    elif tissue == 5 and neighbor_tissue == 6:  # White-Amygdala
                        surfaces['white'].append((v0, v1, v2, normal))
                    elif tissue == 6:  # Amygdala boundary with anything
                        surfaces['amygdala'].append((v0, v1, v2, normal))
    
    # Convert to flat arrays for JSON
    result = {}
    for name, faces in surfaces.items():
        if not faces:
            continue
        
        vertices = []
        normals = []
        for v0, v1, v2, n in faces:
            vertices.extend(v0)
            vertices.extend(v1)
            vertices.extend(v2)
            normals.extend(n)
            normals.extend(n)
            normals.extend(n)
        
        result[name] = {'vertices': vertices, 'normals': normals}
        print(f"  {name}: {len(faces)} triangles ({len(vertices)//9} faces)")
    
    return result


def create_surface_viewer(mesh_path, output_file):
    """Generate interactive HTML viewer showing boundary surfaces."""
    print("="*60)
    print("Creating MMC Surface Viewer")
    print("="*60)
    
    # Load mesh
    nodes, elements, tissues, neighbors = load_mmcmesh(mesh_path)
    
    # Extract boundary surfaces
    surfaces = extract_boundary_surfaces(nodes, elements, tissues, neighbors)
    
    # Convert to JSON
    surfaces_json = json.dumps(surfaces)
    
    # Count triangles
    num_tris = sum(len(s['vertices']) // 9 for s in surfaces.values())
    
    # Generate HTML
    html = HTML_TEMPLATE.format(
        num_nodes=len(nodes),
        num_elements=len(elements),
        num_tris=num_tris,
        surfaces_json=surfaces_json
    )
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n{'='*60}")
    print(f"Surface viewer saved to: {output_file}")
    print(f"Open in browser: file://{Path(output_file).absolute()}")
    print(f"{'='*60}")
    print("\nThis viewer shows:")
    print("  - Scalp outer surface (the actual head shape)")
    print("  - Amygdala surface (deep target)")
    print("  - Optional interior boundaries")


def main():
    parser = argparse.ArgumentParser(description='Create 3D surface viewer for MMC mesh')
    parser.add_argument('--mesh', default='mni152_head.mmcmesh',
                       help='Input MMC mesh file')
    parser.add_argument('--output', default='surface_viewer.html',
                       help='Output HTML file')
    
    args = parser.parse_args()
    
    if not Path(args.mesh).exists():
        print(f"Error: Mesh file not found: {args.mesh}")
        return 1
    
    create_surface_viewer(args.mesh, args.output)
    return 0


if __name__ == '__main__':
    exit(main())
