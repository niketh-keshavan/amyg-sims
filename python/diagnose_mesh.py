#!/usr/bin/env python3
"""
Diagnose MMC mesh: tissue volumes, scalp thickness, amygdala placement,
and detector SDS accuracy.
"""

import struct
import json
import numpy as np
from pathlib import Path


TISSUE_NAMES = {0: 'air', 1: 'scalp', 2: 'skull', 3: 'csf',
                4: 'gray_matter', 5: 'white_matter', 6: 'amygdala'}


def load_mmcmesh(path):
    with open(path, 'rb') as f:
        magic, version, num_nodes, num_elems = struct.unpack('4I', f.read(16))
        bbox = struct.unpack('6f', f.read(24))
        f.read(8)  # padding

        nodes = np.frombuffer(f.read(num_nodes * 12), dtype=np.float32).reshape(num_nodes, 3)
        elems = np.frombuffer(f.read(num_elems * 16), dtype=np.int32).reshape(num_elems, 4)
        tissue = np.frombuffer(f.read(num_elems * 4), dtype=np.int32)
        neighbor_data = f.read(num_elems * 16)
        if len(neighbor_data) == num_elems * 16:
            neighbors = np.frombuffer(neighbor_data, dtype=np.int32).reshape(num_elems, 4)
        else:
            neighbors = None

    return nodes, elems, tissue, neighbors, bbox


def tet_volumes(nodes, elems):
    v0 = nodes[elems[:, 0]]
    v1 = nodes[elems[:, 1]]
    v2 = nodes[elems[:, 2]]
    v3 = nodes[elems[:, 3]]
    d1 = v1 - v0
    d2 = v2 - v0
    d3 = v3 - v0
    cross = np.cross(d2, d3)
    vol = np.abs(np.sum(d1 * cross, axis=1)) / 6.0
    return vol


def find_scalp_boundary_faces(elems, tissue, neighbors):
    """Find external faces of scalp tets (neighbor == -1)."""
    faces = []
    face_verts = [[1,2,3], [0,2,3], [0,1,3], [0,1,2]]
    scalp_mask = tissue == 1
    for e_idx in np.where(scalp_mask)[0]:
        for f in range(4):
            if neighbors[e_idx, f] == -1:
                vi = [elems[e_idx, face_verts[f][k]] for k in range(3)]
                centroid = nodes[vi].mean(axis=0)
                e1 = nodes[vi[1]] - nodes[vi[0]]
                e2 = nodes[vi[2]] - nodes[vi[0]]
                normal = np.cross(e1, e2)
                nm = np.linalg.norm(normal)
                if nm > 0:
                    normal /= nm
                faces.append((centroid, normal))
    return faces


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', default='mni152_head.mmcmesh')
    parser.add_argument('--results', default='data_mmc_10B')
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        # Try relative to script parent
        mesh_path = Path(__file__).parent.parent / args.mesh
    print(f"Loading mesh: {mesh_path}")
    nodes, elems, tissue, neighbors, bbox = load_mmcmesh(mesh_path)

    print(f"\n{'='*70}")
    print(f"MESH DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"Nodes: {len(nodes):,}")
    print(f"Elements: {len(elems):,}")
    print(f"BBox: x=[{bbox[0]:.1f}, {bbox[3]:.1f}]  y=[{bbox[1]:.1f}, {bbox[4]:.1f}]  z=[{bbox[2]:.1f}, {bbox[5]:.1f}]")

    # --- Tissue volumes ---
    print(f"\n{'='*70}")
    print("TISSUE VOLUMES")
    print(f"{'='*70}")
    vols = tet_volumes(nodes, elems)
    total_vol = vols.sum()
    print(f"{'Tissue':<15} {'Tets':>8} {'Volume(cm3)':>12} {'Fraction':>10}")
    print("-" * 50)
    for tid in range(7):
        mask = tissue == tid
        count = mask.sum()
        vol_mm3 = vols[mask].sum()
        vol_cm3 = vol_mm3 / 1000.0
        frac = vol_mm3 / total_vol * 100
        name = TISSUE_NAMES.get(tid, f'type_{tid}')
        print(f"{name:<15} {count:>8,} {vol_cm3:>12.1f} {frac:>9.1f}%")
    print(f"{'TOTAL':<15} {len(elems):>8,} {total_vol/1000:>12.1f}")

    # --- Amygdala details ---
    print(f"\n{'='*70}")
    print("AMYGDALA DETAILS")
    print(f"{'='*70}")
    amyg_mask = tissue == 6
    amyg_count = amyg_mask.sum()
    if amyg_count > 0:
        amyg_centroids = nodes[elems[amyg_mask]].mean(axis=1)
        amyg_center = amyg_centroids.mean(axis=0)
        amyg_vol = vols[amyg_mask].sum() / 1000
        # Split by hemisphere
        right = amyg_centroids[:, 0] > 0
        left = ~right
        print(f"Total amygdala tets: {amyg_count}")
        print(f"Total volume: {amyg_vol:.2f} cm3")
        if right.sum() > 0:
            rc = amyg_centroids[right].mean(axis=0)
            rv = vols[amyg_mask][right].sum() / 1000
            print(f"Right amygdala: {right.sum()} tets, center=({rc[0]:.1f}, {rc[1]:.1f}, {rc[2]:.1f}), vol={rv:.2f} cm3")
        if left.sum() > 0:
            lc = amyg_centroids[left].mean(axis=0)
            lv = vols[amyg_mask][left].sum() / 1000
            print(f"Left amygdala:  {left.sum()} tets, center=({lc[0]:.1f}, {lc[1]:.1f}, {lc[2]:.1f}), vol={lv:.2f} cm3")
        print(f"Expected MNI coords: right=(+24, -2, -20), left=(-24, -2, -20)")
    else:
        print("WARNING: No amygdala tets found!")

    # --- Source and detector analysis ---
    print(f"\n{'='*70}")
    print("SOURCE & DETECTOR ANALYSIS")
    print(f"{'='*70}")

    results_dir = Path(args.results)
    if not results_dir.exists():
        results_dir = Path(__file__).parent.parent / args.results
    meta_path = results_dir / 'mesh_meta.json'
    r730_path = results_dir / 'results_730nm.json'

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        src = np.array(meta['source_position_mm'])
        print(f"Source position: ({src[0]:.2f}, {src[1]:.2f}, {src[2]:.2f})")

        # Distance from source to amygdala centroids
        if amyg_count > 0 and right.sum() > 0:
            dist_src_amyg = np.linalg.norm(src - rc)
            print(f"Source to right amygdala center: {dist_src_amyg:.1f} mm")

        # Replicate the source placement logic from mmc_main.cu
        # The code uses ellipsoidal projection: ea = cx/78, eb = cy/95, ec = cz/85
        if amyg_count > 0 and right.sum() > 0:
            ea = rc[0] / 78.0
            eb = rc[1] / 95.0
            ec = rc[2] / 85.0
            t_scalp = 1.0 / np.sqrt(ea**2 + eb**2 + ec**2)
            ellip_target = t_scalp * rc
            print(f"\nEllipsoidal projection target: ({ellip_target[0]:.1f}, {ellip_target[1]:.1f}, {ellip_target[2]:.1f})")
            print(f"Actual source position:        ({src[0]:.1f}, {src[1]:.1f}, {src[2]:.1f})")
            print(f"Offset: {np.linalg.norm(src - ellip_target):.1f} mm")

    # --- Detector SDS analysis ---
    target_sds = [8, 8, 15, 20, 22, 25, 28, 30, 33, 35, 40,
                  20, 25, 30, 35, 20, 25, 30, 35, 25, 35, 25, 35]
    target_angles = [0, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     30, 30, 30, 30, -30, -30, -30, -30, 60, 60, -60, -60]

    if r730_path.exists():
        with open(r730_path) as f:
            r730 = json.load(f)
        print(f"\n{'Det':>4} {'Target':>7} {'Actual':>7} {'Delta':>7} {'Angle':>6} {'Photons':>10}")
        print("-" * 50)
        for i, det in enumerate(r730['detectors']):
            actual = det['sds_mm']
            target = target_sds[i] if i < len(target_sds) else '?'
            delta = actual - target if isinstance(target, (int, float)) else 0
            angle = det.get('angle_deg', 0)
            photons = det['detected_photons']
            flag = " *** WRONG" if abs(delta) > 5 else ""
            print(f"{det['id']:>4} {target:>7} {actual:>7.1f} {delta:>+7.1f} {angle:>6.0f} {photons:>10,}{flag}")

    # --- Scalp thickness near source ---
    print(f"\n{'='*70}")
    print("SCALP THICKNESS ANALYSIS")
    print(f"{'='*70}")

    if meta_path.exists():
        # Ray from source inward: what tissues does it pass through?
        src_dir = -src / np.linalg.norm(src)  # inward direction (toward origin)
        print(f"Casting ray from source inward: dir=({src_dir[0]:.3f}, {src_dir[1]:.3f}, {src_dir[2]:.3f})")

        # Sample points along ray and classify by nearest tet centroid
        centroids = nodes[elems].mean(axis=1)
        ray_points = []
        for t in np.arange(0, 60, 0.5):  # 0 to 60mm inward in 0.5mm steps
            pt = src + src_dir * t
            # Find nearest tet centroid
            dists = np.sum((centroids - pt)**2, axis=1)
            nearest_tet = np.argmin(dists)
            tissue_type = tissue[nearest_tet]
            ray_points.append((t, tissue_type, TISSUE_NAMES.get(tissue_type, '?')))

        # Report tissue transitions
        print(f"\nRay from source toward head center:")
        print(f"{'Depth(mm)':>10} {'Tissue':<15}")
        print("-" * 30)
        prev_tissue = -1
        for depth, tid, name in ray_points:
            if tid != prev_tissue:
                print(f"{depth:>10.1f} {name:<15}")
                prev_tissue = tid

        # Compute thicknesses
        tissue_depths = {}
        for depth, tid, name in ray_points:
            if name not in tissue_depths:
                tissue_depths[name] = [depth, depth]
            tissue_depths[name][1] = depth

        print(f"\nTissue layer thicknesses along ray:")
        for name, (d0, d1) in tissue_depths.items():
            if name != 'air':
                print(f"  {name}: {d0:.1f} - {d1:.1f} mm (thickness: {d1-d0:.1f} mm)")

    # --- Brain tissue depth from scalp at source location ---
    print(f"\n{'='*70}")
    print("NEAREST SCALP SURFACE POINTS TO SOURCE")
    print(f"{'='*70}")

    if neighbors is not None and meta_path.exists():
        # Find external scalp faces near source
        scalp_mask = tissue == 1
        ext_faces = []
        face_verts = [[1,2,3], [0,2,3], [0,1,3], [0,1,2]]
        for e_idx in np.where(scalp_mask)[0]:
            for f in range(4):
                if neighbors[e_idx, f] == -1:
                    vi = [elems[e_idx, face_verts[f][k]] for k in range(3)]
                    c = nodes[vi].mean(axis=0)
                    d = np.linalg.norm(c - src)
                    if d < 50:  # within 50mm of source
                        ext_faces.append((d, c))

        ext_faces.sort()
        print(f"Found {len(ext_faces)} external scalp faces within 50mm of source")
        if ext_faces:
            print(f"\nNearest 10 scalp surface points:")
            for i, (d, c) in enumerate(ext_faces[:10]):
                print(f"  {d:.1f} mm: ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")

            # Check: for target SDS=8mm, what's the nearest scalp face at ~8mm geodesic?
            print(f"\nScalp face distances from source (histogram):")
            dists_arr = np.array([d for d, c in ext_faces])
            for lo, hi in [(0,5), (5,10), (10,15), (15,20), (20,25), (25,30), (30,40), (40,50)]:
                n = np.sum((dists_arr >= lo) & (dists_arr < hi))
                print(f"  {lo:>3}-{hi:>3} mm: {n:>5} faces")

    print(f"\n{'='*70}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
