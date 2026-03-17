#!/usr/bin/env python3
"""
Generate a tetrahedral head mesh from the MNI152 (ICBM2009c) atlas for MMC.

Pipeline:
  1. Download MNI152 tissue probability maps via nilearn
  2. Segment tissues: scalp, skull, CSF, gray matter, white matter, amygdala
  3. Extract scalp outer surface with marching cubes
  4. Generate tetrahedral mesh using TetGen
  5. Assign tissue labels to each tetrahedron
  6. Compute tet-to-tet neighbor connectivity
  7. Write binary .mmcmesh file for the CUDA MMC solver

Output file format:
  Header  (48 bytes): magic=0x42434D4D, version, num_nodes, num_elems, bbox[6], pad[2]
  Nodes   (N×3 float32): x,y,z in mm (MNI152 space)
  Elems   (M×4 int32):   v0..v3 node indices (0-based)
  Tissue  (M×1 int32):   tissue type per element (TissueType enum from types.cuh)
  Neighbors (M×4 int32): neighbor element per face (-1 = external boundary)

Usage:
  python generate_mni152_mesh.py [--output mni152_head.mmcmesh] [--res 1] [--max-vol 10]

Dependencies:
  pip install nilearn nibabel scikit-image scipy numpy tetgen
"""

import argparse
import struct
import sys
import os
import time
import numpy as np
from scipy import ndimage

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback tqdm stub
    class tqdm:
        def __init__(self, iterable=None, desc='', total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc
            if iterable is not None:
                self.total = len(iterable) if total is None and hasattr(iterable, '__len__') else total
            else:
                self.total = total
            self.n = 0
            if desc:
                print(f"{desc}...")
        def __iter__(self):
            if self.iterable is None:
                return iter([])
            for item in self.iterable:
                yield item
                self.n += 1
            return
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            self.n += n
        def close(self):
            pass

# Tissue label constants matching TissueType enum in types.cuh
TISSUE_AIR      = 0
TISSUE_SCALP    = 1
TISSUE_SKULL    = 2
TISSUE_CSF      = 3
TISSUE_GRAY     = 4
TISSUE_WHITE    = 5
TISSUE_AMYGDALA = 6

MMC_MESH_MAGIC   = 0x42434D4D
MMC_MESH_VERSION = 1


# ---------------------------------------------------------------------------
# Step 1: Load MNI152 segmentations
# ---------------------------------------------------------------------------

def load_mni152_atlas():
    """Download and return MNI152 tissue probability maps via nilearn."""
    try:
        from nilearn import datasets
    except ImportError:
        raise ImportError("Install nilearn: pip install nilearn")

    print("Fetching ICBM152 2009c atlas (downloads ~100 MB on first run)...")
    icbm = datasets.fetch_icbm152_2009()

    import nibabel as nib
    img_t1  = nib.load(icbm['t1'])
    img_gm  = nib.load(icbm['gm'])
    img_wm  = nib.load(icbm['wm'])
    img_csf = nib.load(icbm['csf'])
    img_msk = nib.load(icbm['mask'])

    t1_data  = img_t1.get_fdata(dtype=np.float32)
    gm_data  = img_gm.get_fdata(dtype=np.float32)
    wm_data  = img_wm.get_fdata(dtype=np.float32)
    csf_data = img_csf.get_fdata(dtype=np.float32)
    mask_data= img_msk.get_fdata(dtype=np.float32)
    affine   = img_t1.affine.astype(np.float64)

    print(f"  T1 shape: {t1_data.shape}, voxel size: {np.abs(affine.diagonal()[:3]).round(3)} mm")
    return t1_data, gm_data, wm_data, csf_data, mask_data, affine


# ---------------------------------------------------------------------------
# Step 2: Build tissue label volume
# ---------------------------------------------------------------------------

def build_tissue_labels(t1, gm, wm, csf, brain_mask, voxel_size_mm=1.0):
    """
    Create an integer label volume from probabilistic maps.
    Priority (innermost wins): amygdala > WM > GM > CSF > skull > scalp > air

    Returns:
        labels: np.uint8 array of TissueType values
        affine: mapping from voxel indices to MNI mm
    """
    shape = t1.shape
    labels = np.zeros(shape, dtype=np.uint8)  # TISSUE_AIR = 0

    # Scalp: T1 intensity > low threshold
    t1_norm = (t1 - t1.min()) / (t1.max() - t1.min() + 1e-9)
    head_mask = t1_norm > 0.05
    # Fill internal holes
    head_mask = ndimage.binary_fill_holes(head_mask)
    labels[head_mask] = TISSUE_SCALP

    # Brain mask → everything inside is at least scalp-level
    brain = brain_mask > 0.5

    # Skull: between outer head surface and brain (after erosion)
    # Dilate brain by ~7 mm and subtract brain
    skull_thickness_vox = max(2, int(7.0 / voxel_size_mm))
    brain_dilated = ndimage.binary_dilation(brain, iterations=skull_thickness_vox)
    skull_mask = brain_dilated & ~brain & head_mask
    labels[skull_mask] = TISSUE_SKULL

    # Thin temporal skull (~2.5 mm): lateral + inferior voxels get thinner skull
    # (This is an approximation; the exact geometry is encoded in the mesh element
    #  properties rather than a separate label here)

    # CSF
    csf_mask = csf > 0.3
    labels[csf_mask] = TISSUE_CSF

    # Gray matter
    gm_mask = (gm > 0.5) & brain
    labels[gm_mask] = TISSUE_GRAY

    # White matter
    wm_mask = (wm > 0.5) & brain
    labels[wm_mask] = TISSUE_WHITE

    # Amygdala: bilateral spherical ROIs at MNI coords
    # Right: (+24, -2, -20) mm   Left: (-24, -2, -20) mm
    # Radius ~6 mm
    amyg_centers_mni = np.array([
        [ 24., -2., -20.],
        [-24., -2., -20.],
    ], dtype=np.float64)
    amyg_radius_mm = 6.0

    # Build voxel coordinate grid (in mm)
    ii, jj, kk = np.indices(shape)
    coords_vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel(),
                           np.ones(ii.size)], axis=0).astype(np.float64)

    # Use the affine placeholder; the actual affine is passed in later
    # Here we just mark the amygdala using approximate atlas coords
    # We'll refine when building the mesh node coordinates
    for center_mni in amyg_centers_mni:
        # Find voxel center (rough estimate using voxel_size assumption)
        # Will be refined using affine in the caller
        cx, cy, cz = center_mni / voxel_size_mm + np.array(shape)/2
        ci, cj, ck = int(round(cx)), int(round(cy)), int(round(cz))
        r_vox = amyg_radius_mm / voxel_size_mm
        # Sphere mask
        di = (ii - ci) * voxel_size_mm
        dj = (jj - cj) * voxel_size_mm
        dk = (kk - ck) * voxel_size_mm
        sphere = (di**2 + dj**2 + dk**2) <= amyg_radius_mm**2
        labels[sphere & brain] = TISSUE_AMYGDALA

    print(f"Tissue label summary:")
    tissue_names = ['air','scalp','skull','csf','gray','white','amygdala']
    for t, name in enumerate(tqdm(tissue_names, desc="  Counting voxels", leave=False)):
        n = np.sum(labels == t)
        print(f"  {name}: {n:,} voxels")

    return labels


def build_tissue_labels_with_affine(t1, gm, wm, csf, brain_mask, affine):
    """Build tissue labels with proper MNI coordinate transform for amygdala."""
    shape = t1.shape
    voxel_size = float(np.abs(affine[0,0]))
    labels = build_tissue_labels(t1, gm, wm, csf, brain_mask, voxel_size)

    # Refine amygdala using proper affine inverse
    try:
        inv_affine = np.linalg.inv(affine)
    except np.linalg.LinAlgError:
        return labels

    amyg_centers_mni = np.array([
        [24., -2., -20., 1.],
        [-24., -2., -20., 1.],
    ], dtype=np.float64)
    amyg_radius_mm = 6.5

    ii, jj, kk = np.indices(shape)
    coords_hom = np.stack([ii.ravel()*voxel_size, jj.ravel()*voxel_size,
                           kk.ravel()*voxel_size, np.ones(ii.size)], axis=0)

    # Transform each center to voxel space
    for center_hom in tqdm(amyg_centers_mni, desc="  Processing amygdala regions", leave=False):
        center_vox = inv_affine @ center_hom
        ci, cj, ck = center_vox[:3]
        di = (ii - ci) * voxel_size
        dj = (jj - cj) * voxel_size
        dk = (kk - ck) * voxel_size
        sphere = (di**2 + dj**2 + dk**2) <= amyg_radius_mm**2
        brain = (gm > 0.3) | (wm > 0.3)
        labels[sphere & brain] = TISSUE_AMYGDALA

    amyg_count = np.sum(labels == TISSUE_AMYGDALA)
    print(f"  amygdala (refined): {amyg_count:,} voxels")
    return labels


# ---------------------------------------------------------------------------
# Step 3: Extract scalp surface with marching cubes
# ---------------------------------------------------------------------------

def extract_scalp_surface(labels, affine, smooth_sigma=1.5):
    """
    Extract the outer scalp surface as a triangle mesh using marching cubes.
    Applies morphological operations to ensure manifold surface.
    Returns:
        verts: (Nv, 3) float64 array in MNI mm
        faces: (Nf, 3) int32 array
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError("Install scikit-image: pip install scikit-image")

    # Binary head mask (anything non-air)
    head_bin = (labels > TISSUE_AIR).astype(np.float32)

    # Ensure manifold: fill holes and close gaps
    print("Cleaning surface mask (morphological operations)...")
    morph_ops = [
        ("Filling holes", lambda x: ndimage.binary_fill_holes(x)),
        ("Opening (remove bridges)", lambda x: ndimage.binary_opening(x, iterations=1)),
        ("Closing (fill gaps)", lambda x: ndimage.binary_closing(x, iterations=2)),
        ("Final hole fill", lambda x: ndimage.binary_fill_holes(x)),
    ]
    for desc, op in tqdm(morph_ops, desc="  Morphological ops", leave=False):
        head_bin = op(head_bin)

    # Smooth to reduce staircase artifacts
    head_smooth = ndimage.gaussian_filter(head_bin.astype(np.float32), sigma=smooth_sigma)

    print("Extracting scalp surface (marching cubes)...")
    voxel_size = float(np.abs(affine[0,0]))

    # marching_cubes returns verts in voxel index space
    verts_vox, faces, normals, values = marching_cubes(
        head_smooth, level=0.5, spacing=(voxel_size,)*3)

    # Transform to MNI mm using affine
    verts_hom = np.hstack([verts_vox, np.ones((len(verts_vox),1))])
    verts_mni = (affine @ verts_hom.T).T[:, :3]

    faces = faces.astype(np.int32)
    print(f"  Scalp surface: {len(verts_mni):,} vertices, {len(faces):,} triangles")
    return verts_mni.astype(np.float64), faces


# ---------------------------------------------------------------------------
# Step 4: Tetrahedral meshing with TetGen
# ---------------------------------------------------------------------------

def mesh_with_tetgen(surf_verts, surf_faces, max_vol=15.0, min_dihedral=15.0):
    """
    Generate a conforming tetrahedral mesh from the scalp surface using TetGen.

    Args:
        surf_verts: (Nv,3) float64 vertices in mm
        surf_faces: (Nf,3) int32 triangle indices
        max_vol: maximum tet volume in mm³ (controls mesh density)
        min_dihedral: minimum dihedral angle in degrees

    Returns:
        nodes: (N,3) float64 node positions in mm
        elems: (M,4) int64 tetrahedra (node indices)
    """
    try:
        import tetgen
    except ImportError:
        raise ImportError(
            "Install tetgen Python package: pip install tetgen\n"
            "  (wraps TetGen by Hang Si, 2002-2018)")

    print(f"Meshing with TetGen (max_vol={max_vol:.1f} mm³, min_dihedral={min_dihedral:.0f}°)...")
    t0 = time.time()

    tet = tetgen.TetGen(surf_verts, surf_faces)
    tet.tetrahedralize(
        order=1,
        quality=True,
        mindihedral=min_dihedral,
        minratio=1.5,
        maxvolume=max_vol,
        verbose=0,
    )

    nodes = np.asarray(tet.node, dtype=np.float64)
    elems = np.asarray(tet.elem, dtype=np.int64)

    elapsed = time.time() - t0
    print(f"  TetGen done in {elapsed:.1f}s: {len(nodes):,} nodes, {len(elems):,} tets")
    return nodes, elems


# ---------------------------------------------------------------------------
# Step 5: Assign tissue labels to tetrahedra
# ---------------------------------------------------------------------------

def assign_tissue_labels(nodes_mm, elems, labels_vol, affine):
    """
    For each tetrahedron, assign the tissue type of its centroid.

    Uses trilinear-sampled tissue label from the segmented atlas.
    """
    print("  Computing centroids...")
    inv_aff = np.linalg.inv(affine)

    # Compute centroids in MNI mm
    centroids = nodes_mm[elems[:, :4]].mean(axis=1)  # (M, 3)

    # Map MNI mm → voxel indices
    print("  Mapping to voxel space...")
    cents_hom = np.hstack([centroids, np.ones((len(centroids),1))])
    vox_coords = (inv_aff @ cents_hom.T).T[:, :3]  # (M, 3)

    # Nearest-neighbor lookup
    print("  Sampling tissue labels...")
    vi = np.round(vox_coords[:,0]).astype(int)
    vj = np.round(vox_coords[:,1]).astype(int)
    vk = np.round(vox_coords[:,2]).astype(int)

    shape = labels_vol.shape
    vi = np.clip(vi, 0, shape[0]-1)
    vj = np.clip(vj, 0, shape[1]-1)
    vk = np.clip(vk, 0, shape[2]-1)

    tissue_labels = labels_vol[vi, vj, vk].astype(np.int32)

    # Post-process: override amygdala labels using direct MNI sphere check.
    # The centroid voxel lookup misses small ROIs (6.5mm radius) due to
    # coarse tet resolution. Directly check centroid-to-sphere distance.
    amyg_centers = np.array([[24., -2., -20.], [-24., -2., -20.]])
    amyg_radius = 6.5
    for ac in amyg_centers:
        dist2 = np.sum((centroids - ac)**2, axis=1)
        in_sphere = dist2 <= amyg_radius**2
        # Override any non-air tissue within sphere (deep brain, so no scalp/skull expected)
        non_air = tissue_labels != 0
        override = in_sphere & non_air
        tissue_labels[override] = TISSUE_AMYGDALA
        n_override = np.sum(override)
        if n_override > 0:
            print(f"    Amygdala override: {n_override} tets near ({ac[0]:.0f},{ac[1]:.0f},{ac[2]:.0f})")

    print("  Counting tissue types...")
    for t, name in enumerate(tqdm(['air','scalp','skull','csf','gray','white','amygdala'],
                                   desc="  Tissue counts", leave=False)):
        n = np.sum(tissue_labels == t)
        if n > 0:
            print(f"    {name}: {n:,} tets")

    return tissue_labels


# ---------------------------------------------------------------------------
# Step 6: Build neighbor connectivity
# ---------------------------------------------------------------------------

def compute_tet_neighbors(elems):
    """
    For each tetrahedron, find the neighboring element sharing each face.
    Face convention:  face k is opposite vertex k.
      face 0: (v1,v2,v3)   face 1: (v0,v2,v3)
      face 2: (v0,v1,v3)   face 3: (v0,v1,v2)

    Returns:
        neighbors: (M, 4) int32 array, -1 for external boundary
    """
    M = len(elems)
    neighbors = -np.ones((M, 4), dtype=np.int32)

    # Face vertex indices relative to element (which 3 of 4 nodes form each face)
    FACE_VERTS = [
        (1, 2, 3),  # face 0
        (0, 2, 3),  # face 1
        (0, 1, 3),  # face 2
        (0, 1, 2),  # face 3
    ]

    print("Computing neighbor connectivity...")
    t0 = time.time()

    # Map sorted face (i,j,k) -> (elem_id, local_face_id)
    face_map = {}

    for e in tqdm(range(M), desc="  Processing elements", unit="tet", ncols=80):
        v = elems[e]
        for f, (i, j, k) in enumerate(FACE_VERTS):
            key = tuple(sorted([int(v[i]), int(v[j]), int(v[k])]))
            if key in face_map:
                other_e, other_f = face_map.pop(key)
                neighbors[e][f]       = other_e
                neighbors[other_e][other_f] = e
            else:
                face_map[key] = (e, f)

    elapsed = time.time() - t0
    boundary_faces = sum(1 for row in neighbors for n in row if n == -1)
    print(f"  Done in {elapsed:.1f}s  boundary faces: {boundary_faces:,}")
    return neighbors


# ---------------------------------------------------------------------------
# Step 7: Write binary mesh file
# ---------------------------------------------------------------------------

def save_mmcmesh(path, nodes_mm, elems, tissue_labels, neighbors):
    """
    Write the binary .mmcmesh file consumed by the CUDA MMC solver.

    Header (48 bytes):
      uint32 magic, version, num_nodes, num_elems
      float32[3] bbox_min, float32[3] bbox_max
      uint32[2] reserved
    Data:
      float32[N×3]  nodes
      int32[M×4]    elems
      int32[M]      tissue
      int32[M×4]    neighbors
    """
    nodes_f32 = nodes_mm.astype(np.float32)
    elems_i32 = elems[:, :4].astype(np.int32)
    tissue_i32 = tissue_labels.astype(np.int32)
    neigh_i32  = neighbors.astype(np.int32)

    N = len(nodes_f32)
    M = len(elems_i32)

    bbox_min = nodes_f32.min(axis=0)
    bbox_max = nodes_f32.max(axis=0)

    print(f"Writing {path} ...")
    print(f"  {N:,} nodes, {M:,} elements")
    print(f"  BBox: ({bbox_min[0]:.1f},{bbox_min[1]:.1f},{bbox_min[2]:.1f}) → "
          f"({bbox_max[0]:.1f},{bbox_max[1]:.1f},{bbox_max[2]:.1f}) mm")

    with open(path, 'wb') as f:
        # Header
        hdr = struct.pack('=IIII ffffff II',
            MMC_MESH_MAGIC,
            MMC_MESH_VERSION,
            N, M,
            float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2]),
            float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2]),
            0, 0)
        f.write(hdr)

        # Data sections
        f.write(nodes_f32.tobytes())
        f.write(elems_i32.tobytes())
        f.write(tissue_i32.tobytes())
        f.write(neigh_i32.tobytes())

    file_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {file_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def generate_mni152_mesh(output_path, max_vol=15.0, min_dihedral=15.0):
    t_total = time.time()

    print("="*60)
    print("  MNI152 Head Mesh Generation for MMC fNIRS")
    print("="*60)

    # Define pipeline steps for overall progress
    steps = [
        ("Load atlas", lambda: load_mni152_atlas()),
        ("Build tissue labels", lambda d: build_tissue_labels_with_affine(d[0], d[1], d[2], d[3], d[4], d[5])),
        ("Extract scalp surface", lambda d: extract_scalp_surface(d, None, smooth_sigma=2.5)),
        ("Tetrahedral mesh", lambda d: mesh_with_tetgen(d[0], d[1], max_vol=max_vol, min_dihedral=min_dihedral)),
        ("Assign tissue labels", lambda d: assign_tissue_labels(d[0], d[1], d[2], d[3])),
        ("Compute neighbors", lambda d: compute_tet_neighbors(d)),
        ("Save mesh", lambda d: save_mmcmesh(output_path, d[0], d[1], d[2], d[3])),
    ]

    # 1. Load atlas
    print("\n[1/7] Loading atlas...")
    t1, gm, wm, csf, brain_mask, affine = load_mni152_atlas()

    # 2. Build tissue label volume
    print("\n[2/7] Building tissue label volume...")
    labels = build_tissue_labels_with_affine(t1, gm, wm, csf, brain_mask, affine)

    # 3. Extract scalp surface
    print("\n[3/7] Extracting scalp surface...")
    surf_verts, surf_faces = extract_scalp_surface(labels, affine, smooth_sigma=2.5)

    # 4. Tetrahedral mesh
    print("\n[4/7] Generating tetrahedral mesh...")
    nodes, elems = mesh_with_tetgen(surf_verts, surf_faces,
                                     max_vol=max_vol,
                                     min_dihedral=min_dihedral)

    # 5. Assign tissue labels
    print("\n[5/7] Assigning tissue labels...")
    tissue_labels = assign_tissue_labels(nodes, elems, labels, affine)

    # 6. Neighbor connectivity
    print("\n[6/7] Building neighbor connectivity...")
    neighbors = compute_tet_neighbors(elems)

    # 7. Save
    print("\n[7/7] Saving binary mesh...")
    save_mmcmesh(output_path, nodes, elems, tissue_labels, neighbors)

    print(f"\nTotal time: {time.time()-t_total:.1f}s")
    print(f"Output: {output_path}")

    # Print mesh stats
    print("\nMesh statistics:")
    print(f"  Nodes:    {len(nodes):>10,}")
    print(f"  Tets:     {len(elems):>10,}")
    vol = np.abs(np.sum(
        np.cross(nodes[elems[:,1]]-nodes[elems[:,0]],
                 nodes[elems[:,2]]-nodes[elems[:,0]])
        * (nodes[elems[:,3]]-nodes[elems[:,0]]), axis=1
    )) / 6.0
    print(f"  Mean tet volume: {vol.mean():.2f} mm³  (max: {vol.max():.1f})")
    amyg_vols = vol[tissue_labels == TISSUE_AMYGDALA]
    if len(amyg_vols) > 0:
        print(f"  Amygdala tets: {len(amyg_vols):,}  (total vol: {amyg_vols.sum():.0f} mm³)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output',   default='mni152_head.mmcmesh',
                   help='Output .mmcmesh file path (default: mni152_head.mmcmesh)')
    p.add_argument('--max-vol',  type=float, default=15.0,
                   help='Max tet volume mm³ — smaller = denser mesh (default: 15)')
    p.add_argument('--min-dihedral', type=float, default=15.0,
                   help='Min dihedral angle in degrees (default: 15)')

    args = p.parse_args()

    # Dependency check
    missing = []
    for pkg in ['nilearn','nibabel','skimage','tetgen','scipy','tqdm']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing packages. Install with:")
        install_pkgs = ' '.join(missing).replace('skimage','scikit-image')
        print(f"  pip install {install_pkgs}")
        sys.exit(1)

    generate_mni152_mesh(args.output,
                          max_vol=args.max_vol,
                          min_dihedral=args.min_dihedral)


if __name__ == '__main__':
    main()
