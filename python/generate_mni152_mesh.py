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

def mesh_with_cgalmesh(seg_dict, max_vol=100.0, voxel_size=1.0):
    """
    Generate multi-tissue brain mesh using cgalmesh from iso2mesh.
    
    This uses CGAL's 3D mesh generation from labeled volumes, which is more
    robust than brain2mesh's surface boolean approach for complex geometries.
    
    Args:
        seg_dict: dict with keys 'scalp', 'skull', 'csf', 'gm', 'wm'
                  Each is a 3D binary numpy array
        max_vol: maximum tet volume
        voxel_size: voxel size in mm (default 1.0)
    
    Returns:
        nodes: (N,3) float64 node positions in mm
        elems: (M,4) int64 tetrahedra (node indices)
        tissue_labels: (M,) int32 tissue type per element
    """
    try:
        import iso2mesh
    except ImportError:
        raise ImportError(
            "Install iso2mesh: pip install iso2mesh or from GitHub\n"
            "  https://github.com/NeuroJSON/pyiso2mesh")
    
    print(f"Meshing with cgalmesh (max_vol={max_vol:.1f} mm³)...")
    print("  Using CGAL 3D mesh generation from labeled volume...")
    t0 = time.time()
    
    # Create a labeled volume where each voxel has a tissue label
    # Priority: inner tissues overwrite outer ones
    shape = seg_dict['scalp'].shape
    labels_3d = np.zeros(shape, dtype=np.uint8)
    
    # Assign labels: 1=scalp, 2=skull, 3=csf, 4=gm, 5=wm
    # Use priority: inner tissues overwrite outer
    labels_3d[seg_dict['scalp'] > 0.5] = 1
    labels_3d[seg_dict['skull'] > 0.5] = 2
    labels_3d[seg_dict['csf'] > 0.5] = 3
    labels_3d[seg_dict['gm'] > 0.5] = 4
    labels_3d[seg_dict['wm'] > 0.5] = 5
    
    print(f"  Label volume shape: {labels_3d.shape}")
    print(f"  Tissue voxel counts:")
    for i, name in enumerate(['bg', 'scalp', 'skull', 'csf', 'gm', 'wm']):
        print(f"    {name}: {np.sum(labels_3d == i):,}")
    
    # Use vol2mesh to create mesh from labeled volume
    # This uses CGAL's mesh generation directly
    print("  Running cgalv2m (CGAL volume meshing)...")
    # Use cgalv2m which meshes a binary volume directly
    # Create a binary mask of all brain tissues
    brain_mask = (labels_3d > 0).astype(np.uint8)
    
    # Create options dictionary
    opt = {
        'maxvol': max_vol,
        'radbound': 3.0,
    }
    
    print(f"    Meshing with maxvol={max_vol}...")
    
    # Mesh the brain volume - cgalv2m(vol, opt, maxvol)
    mesh = iso2mesh.cgalv2m(brain_mask, opt, max_vol)
    
    # cgalv2m returns either a dict or a tuple (node, elem, face)
    if isinstance(mesh, tuple):
        nodes = np.asarray(mesh[0], dtype=np.float64) * voxel_size  # Scale to mm
        elems = np.asarray(mesh[1], dtype=np.int64)
    else:
        nodes = np.asarray(mesh['node'], dtype=np.float64) * voxel_size  # Scale to mm
        elems = np.asarray(mesh['elem'], dtype=np.int64)
    
    print(f"    Raw mesh: {len(nodes)} nodes, elems shape {elems.shape}")
    
    # cgalv2m returns elements with format [n1, n2, n3, n4, region_id] (1-indexed)
    # Extract just the first 4 columns and convert to 0-indexed
    if elems.shape[1] >= 5:
        elems = elems[:, :4]  # Extract node indices only
    elems = elems - 1  # Convert to 0-indexed
    
    print(f"    Processed elems shape: {elems.shape}")
    
    # cgalv2m returns a single region - we need to relabel based on centroid positions
    print("  Assigning tissue labels based on centroid positions...")
    # Build affine: voxel to mm transform (isotropic voxels)
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    tissue_labels = assign_tissue_labels_to_mesh(nodes, elems, labels_3d, affine)
    
    elapsed = time.time() - t0
    print(f"  cgalv2m done in {elapsed:.1f}s: {len(nodes):,} nodes, {len(elems):,} tets")
    
    # Print tissue distribution
    tissue_names = ['air', 'scalp', 'skull', 'csf', 'gray', 'white']
    print("  Tissue distribution:")
    for i, name in enumerate(tissue_names):
        count = np.sum(tissue_labels == i)
        if count > 0:
            print(f"    {name}: {count:,} tets ({100.0*count/len(tissue_labels):.1f}%)")
    
    return nodes, elems, tissue_labels


def assign_tissue_labels_to_mesh(nodes_mm, elems, labels_vol, affine):
    """
    Assign tissue labels to mesh elements based on their centroid positions.
    
    Args:
        nodes_mm: (N,3) node positions in mm
        elems: (M,4) tetrahedra node indices
        labels_vol: 3D label volume (0=bg, 1=scalp, 2=skull, 3=csf, 4=gm, 5=wm)
        affine: 4x4 voxel-to-mm transformation
    
    Returns:
        tissue_labels: (M,) int32 tissue type per element
    """
    print("    Computing centroids and sampling labels...")
    inv_aff = np.linalg.inv(affine)
    
    # Compute centroids
    centroids = nodes_mm[elems[:, :4]].mean(axis=1)
    
    # Transform to voxel space
    cents_hom = np.hstack([centroids, np.ones((len(centroids), 1))])
    vox_coords = (inv_aff @ cents_hom.T).T[:, :3]
    
    # Round to nearest voxel
    vi = np.round(vox_coords[:, 0]).astype(int)
    vj = np.round(vox_coords[:, 1]).astype(int)
    vk = np.round(vox_coords[:, 2]).astype(int)
    
    # Clip to valid range
    shape = labels_vol.shape
    vi = np.clip(vi, 0, shape[0] - 1)
    vj = np.clip(vj, 0, shape[1] - 1)
    vk = np.clip(vk, 0, shape[2] - 1)
    
    # Sample labels
    tissue_labels = labels_vol[vi, vj, vk].astype(np.int32)
    
    # Count distribution
    print("    Tissue distribution from volume sampling:")
    names = ['bg', 'scalp', 'skull', 'csf', 'gm', 'wm']
    for i, name in enumerate(names):
        count = np.sum(tissue_labels == i)
        if count > 0:
            print(f"      {name}: {count:,} tets ({100.0*count/len(tissue_labels):.1f}%)")
    
    return tissue_labels


def add_amygdala_to_mesh(nodes_mm, elems, tissue_labels, affine, amyg_radius=6.5):
    """
    Post-process: re-label tets within amygdala spheres as TISSUE_AMYGDALA.
    
    brain2mesh only handles 5 tissue types. We add amygdala (type 6) by
    checking which tets fall within the amygdala regions.
    
    Args:
        nodes_mm: (N,3) node positions in MNI mm
        elems: (M,4) tetrahedra node indices  
        tissue_labels: (M,) current tissue labels (0-4 from brain2mesh)
        affine: 4x4 voxel-to-mm transformation
        amyg_radius: radius in mm for amygdala region (default 6.5mm)
    
    Returns:
        tissue_labels: updated with TISSUE_AMYGDALA (6) for tets in spheres
    """
    print("  Adding amygdala labels post-brain2mesh...")
    
    # Compute centroids
    centroids = nodes_mm[elems[:, :4]].mean(axis=1)
    
    # Amygdala centers in MNI space
    amyg_centers = np.array([[24., -2., -20.], [-24., -2., -20.]])
    
    n_total = 0
    for ac in amyg_centers:
        dist2 = np.sum((centroids - ac)**2, axis=1)
        in_sphere = dist2 <= amyg_radius**2
        
        # Override any tissue within sphere (should be GM/WM/CSF)
        tissue_labels[in_sphere] = TISSUE_AMYGDALA
        n = int(np.sum(in_sphere))
        n_total += n
        min_dist = float(np.sqrt(dist2.min()))
        print(f"    Amygdala sphere ({ac[0]:+.0f},{ac[1]:+.0f},{ac[2]:+.0f}): "
              f"{n} tets relabeled, closest centroid {min_dist:.1f}mm")
    
    print(f"  Total amygdala tets: {n_total}")
    return tissue_labels


# ---------------------------------------------------------------------------
# Step 5: Assign tissue labels to tetrahedra (legacy, kept for reference)
# ---------------------------------------------------------------------------

def assign_tissue_labels(nodes_mm, elems, labels_vol, affine):
    """
    For each tetrahedron, assign the tissue type of its centroid.

    Uses nearest-neighbor voxel lookup from the segmented atlas volume.
    """
    print("  Computing centroids...")
    inv_aff = np.linalg.inv(affine)

    centroids = nodes_mm[elems[:, :4]].mean(axis=1)  # (M, 3) in MNI mm

    # --- DIAGNOSTICS: verify coordinate systems ---
    print(f"  Centroid MNI range: "
          f"x=[{centroids[:,0].min():.1f}, {centroids[:,0].max():.1f}]  "
          f"y=[{centroids[:,1].min():.1f}, {centroids[:,1].max():.1f}]  "
          f"z=[{centroids[:,2].min():.1f}, {centroids[:,2].max():.1f}]")
    print(f"  Affine diagonal: {np.diag(affine)}")
    print(f"  Affine translation: {affine[:3, 3]}")

    # Map MNI mm → voxel indices
    print("  Mapping to voxel space...")
    cents_hom = np.hstack([centroids, np.ones((len(centroids),1))])
    vox_coords = (inv_aff @ cents_hom.T).T[:, :3]  # (M, 3)

    print(f"  Voxel coord range: "
          f"i=[{vox_coords[:,0].min():.1f}, {vox_coords[:,0].max():.1f}]  "
          f"j=[{vox_coords[:,1].min():.1f}, {vox_coords[:,1].max():.1f}]  "
          f"k=[{vox_coords[:,2].min():.1f}, {vox_coords[:,2].max():.1f}]")
    print(f"  Label volume shape: {labels_vol.shape}")

    # Nearest-neighbor lookup
    vi = np.round(vox_coords[:,0]).astype(int)
    vj = np.round(vox_coords[:,1]).astype(int)
    vk = np.round(vox_coords[:,2]).astype(int)

    shape = labels_vol.shape
    oob = ((vi < 0) | (vi >= shape[0]) |
           (vj < 0) | (vj >= shape[1]) |
           (vk < 0) | (vk >= shape[2]))
    print(f"  Out-of-bounds centroids (clipped): {oob.sum():,} / {len(vi):,}")

    vi = np.clip(vi, 0, shape[0]-1)
    vj = np.clip(vj, 0, shape[1]-1)
    vk = np.clip(vk, 0, shape[2]-1)

    tissue_labels = labels_vol[vi, vj, vk].astype(np.int32)

    # --- DIAGNOSTICS: voxel volume distribution for comparison ---
    print("  Label VOLUME distribution (reference):")
    tissue_names = ['air','scalp','skull','csf','gray','white','amygdala']
    for t, name in enumerate(tissue_names):
        n = int(np.sum(labels_vol == t))
        if n > 0:
            print(f"    {name}: {n:,} voxels ({100.0*n/labels_vol.size:.1f}%)")

    print("  Label TET distribution (before amygdala override):")
    for t, name in enumerate(tissue_names):
        n = int(np.sum(tissue_labels == t))
        if n > 0:
            print(f"    {name}: {n:,} tets ({100.0*n/len(tissue_labels):.1f}%)")

    # --- DIAGNOSTICS: spot-check a known brain point ---
    test_mni = np.array([0.0, 0.0, 0.0, 1.0])
    test_vox = inv_aff @ test_mni
    ti, tj, tk = int(round(test_vox[0])), int(round(test_vox[1])), int(round(test_vox[2]))
    if 0 <= ti < shape[0] and 0 <= tj < shape[1] and 0 <= tk < shape[2]:
        print(f"  Spot-check: MNI(0,0,0) -> vox({ti},{tj},{tk}) = label {labels_vol[ti,tj,tk]} "
              f"({tissue_names[int(labels_vol[ti,tj,tk])]})")

    test_mni2 = np.array([24.0, -2.0, -20.0, 1.0])
    test_vox2 = inv_aff @ test_mni2
    ti2, tj2, tk2 = int(round(test_vox2[0])), int(round(test_vox2[1])), int(round(test_vox2[2]))
    if 0 <= ti2 < shape[0] and 0 <= tj2 < shape[1] and 0 <= tk2 < shape[2]:
        print(f"  Spot-check: MNI(24,-2,-20) -> vox({ti2},{tj2},{tk2}) = label {labels_vol[ti2,tj2,tk2]} "
              f"({tissue_names[int(labels_vol[ti2,tj2,tk2])]})")

    # Amygdala override: UNCONDITIONAL within sphere (no tissue filter).
    # The amygdala is deep brain — any tet centroid within 6.5mm must be amygdala.
    amyg_centers = np.array([[24., -2., -20.], [-24., -2., -20.]])
    amyg_radius = 6.5
    for ac in amyg_centers:
        dist2 = np.sum((centroids - ac)**2, axis=1)
        in_sphere = dist2 <= amyg_radius**2
        n_in = int(np.sum(in_sphere))
        min_dist = float(np.sqrt(dist2.min()))
        tissue_labels[in_sphere] = TISSUE_AMYGDALA
        print(f"    Amygdala sphere ({ac[0]:+.0f},{ac[1]:+.0f},{ac[2]:+.0f}): "
              f"{n_in} tets overridden, closest tet {min_dist:.1f}mm away")

    print("  Final tet distribution:")
    for t, name in enumerate(tissue_names):
        n = int(np.sum(tissue_labels == t))
        if n > 0:
            print(f"    {name}: {n:,} tets ({100.0*n/len(tissue_labels):.1f}%)")

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

import pickle
import os

def save_checkpoint(checkpoint_dir, step_name, data):
    """Save checkpoint data to resume later."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"{step_name}.pkl")
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  💾 Checkpoint saved: {checkpoint_file}")

def load_checkpoint(checkpoint_dir, step_name):
    """Load checkpoint data if it exists."""
    checkpoint_file = os.path.join(checkpoint_dir, f"{step_name}.pkl")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  📂 Checkpoint loaded: {checkpoint_file}")
        return data
    return None

def generate_mni152_mesh(output_path, max_vol=50.0, min_dihedral=15.0, 
                         checkpoint_dir=None, resume=True):
    """
    Generate MNI152 head mesh with checkpoint/resume support.
    
    Args:
        output_path: Path to output .mmcmesh file
        max_vol: Maximum tetrahedron volume
        min_dihedral: Minimum dihedral angle
        checkpoint_dir: Directory to save checkpoints (default: output_path + '.checkpoints')
        resume: If True, resume from last checkpoint if available
    """
    t_total = time.time()
    
    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = output_path + '.checkpoints'
    
    print("="*60)
    print("  MNI152 Head Mesh Generation for MMC fNIRS")
    if resume and os.path.exists(checkpoint_dir):
        print(f"  Resume mode: checkpoints in {checkpoint_dir}")
    print("="*60)

    # Step 1: Load atlas
    step = "step1_atlas"
    if resume:
        checkpoint = load_checkpoint(checkpoint_dir, step)
        if checkpoint:
            t1, gm, wm, csf, brain_mask, affine = checkpoint
        else:
            print("\n[1/7] Loading atlas...")
            with tqdm(total=1, desc="  Downloading/loading MNI152 atlas", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
                t1, gm, wm, csf, brain_mask, affine = load_mni152_atlas()
                pbar.update(1)
            save_checkpoint(checkpoint_dir, step, (t1, gm, wm, csf, brain_mask, affine))
    else:
        print("\n[1/7] Loading atlas...")
        with tqdm(total=1, desc="  Downloading/loading MNI152 atlas", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
            t1, gm, wm, csf, brain_mask, affine = load_mni152_atlas()
            pbar.update(1)
        save_checkpoint(checkpoint_dir, step, (t1, gm, wm, csf, brain_mask, affine))

    # Step 2: Build tissue labels
    step = "step2_labels"
    if resume:
        checkpoint = load_checkpoint(checkpoint_dir, step)
        if checkpoint:
            labels = checkpoint
        else:
            print("\n[2/7] Building tissue label volume...")
            with tqdm(total=1, desc="  Segmenting tissues", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
                labels = build_tissue_labels_with_affine(t1, gm, wm, csf, brain_mask, affine)
                pbar.update(1)
            save_checkpoint(checkpoint_dir, step, labels)
    else:
        print("\n[2/7] Building tissue label volume...")
        with tqdm(total=1, desc="  Segmenting tissues", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
            labels = build_tissue_labels_with_affine(t1, gm, wm, csf, brain_mask, affine)
            pbar.update(1)
        save_checkpoint(checkpoint_dir, step, labels)

    # Step 3: Prepare segmentations with smoothing to fix self-intersections
    step = "step3_segdict"
    if resume:
        checkpoint = load_checkpoint(checkpoint_dir, step)
        if checkpoint:
            seg_dict = checkpoint
            print("\n[3/7] Preparing segmentation volumes... (loaded from checkpoint)")
        else:
            print("\n[3/7] Preparing segmentation volumes...")
            print("  Applying Gaussian smoothing to reduce self-intersections...")
            seg_dict = {}
            tissue_names = ['scalp', 'skull', 'csf', 'gm', 'wm']
            
            # Create raw masks first
            raw_masks = {}
            for name in tissue_names:
                if name == 'scalp':
                    raw_masks[name] = (labels == TISSUE_SCALP).astype(np.float64)
                elif name == 'skull':
                    raw_masks[name] = (labels == TISSUE_SKULL).astype(np.float64)
                elif name == 'csf':
                    raw_masks[name] = ((labels == TISSUE_CSF) | (labels == TISSUE_AMYGDALA)).astype(np.float64)
                elif name == 'gm':
                    raw_masks[name] = (labels == TISSUE_GRAY).astype(np.float64)
                elif name == 'wm':
                    raw_masks[name] = (labels == TISSUE_WHITE).astype(np.float64)
            
            # Apply smoothing to reduce surface artifacts
            # This prevents the "ran out of tries to perturb the mesh" error
            for name in tqdm(tissue_names, desc="  Smoothing tissue masks", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
                # Gaussian smoothing with sigma=1 voxel
                smoothed = ndimage.gaussian_filter(raw_masks[name], sigma=1.0)
                # Threshold at 0.5 to get binary mask back
                seg_dict[name] = (smoothed > 0.5).astype(np.float64)
            
            save_checkpoint(checkpoint_dir, step, seg_dict)
    else:
        print("\n[3/7] Preparing segmentation volumes...")
        print("  Applying Gaussian smoothing to reduce self-intersections...")
        seg_dict = {}
        tissue_names = ['scalp', 'skull', 'csf', 'gm', 'wm']
        
        # Create raw masks first
        raw_masks = {}
        for name in tissue_names:
            if name == 'scalp':
                raw_masks[name] = (labels == TISSUE_SCALP).astype(np.float64)
            elif name == 'skull':
                raw_masks[name] = (labels == TISSUE_SKULL).astype(np.float64)
            elif name == 'csf':
                raw_masks[name] = ((labels == TISSUE_CSF) | (labels == TISSUE_AMYGDALA)).astype(np.float64)
            elif name == 'gm':
                raw_masks[name] = (labels == TISSUE_GRAY).astype(np.float64)
            elif name == 'wm':
                raw_masks[name] = (labels == TISSUE_WHITE).astype(np.float64)
        
        # Apply smoothing
        for name in tqdm(tissue_names, desc="  Smoothing tissue masks", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
            smoothed = ndimage.gaussian_filter(raw_masks[name], sigma=1.0)
            seg_dict[name] = (smoothed > 0.5).astype(np.float64)
        
        save_checkpoint(checkpoint_dir, step, seg_dict)
    
    print(f"  Segmentation volumes: {labels.shape}")
    for name, vol in seg_dict.items():
        print(f"    {name}: {int(vol.sum()):,} voxels")

    # Step 4: brain2mesh (the slow step - checkpoint after this!)
    step = "step4_mesh"
    if resume:
        checkpoint = load_checkpoint(checkpoint_dir, step)
        if checkpoint:
            nodes, elems, tissue_labels = checkpoint
            print("\n[4/7] Generating tetrahedral mesh... (loaded from checkpoint)")
            print(f"    Mesh: {len(nodes):,} nodes, {len(elems):,} tets")
        else:
            print("\n[4/7] Generating tetrahedral mesh with brain2mesh...")
            print("  ⚠️  This step takes 10-30 minutes - generating multi-tissue brain mesh...")
            t_mesh = time.time()
            nodes, elems, tissue_labels = mesh_with_cgalmesh(seg_dict, max_vol=max_vol)
            print(f"  Mesh generation completed in {time.time()-t_mesh:.1f}s")
            save_checkpoint(checkpoint_dir, step, (nodes, elems, tissue_labels))
    else:
        print("\n[4/7] Generating tetrahedral mesh with brain2mesh...")
        print("  ⚠️  This step takes 10-30 minutes - generating multi-tissue brain mesh...")
        t_mesh = time.time()
        nodes, elems, tissue_labels = mesh_with_brain2mesh(seg_dict, max_vol=max_vol)
        print(f"  Mesh generation completed in {time.time()-t_mesh:.1f}s")
        save_checkpoint(checkpoint_dir, step, (nodes, elems, tissue_labels))

    # Step 5: Amygdala labeling
    step = "step5_amygdala"
    if resume:
        checkpoint = load_checkpoint(checkpoint_dir, step)
        if checkpoint:
            tissue_labels = checkpoint
            print("\n[5/7] Adding amygdala tissue labels... (loaded from checkpoint)")
        else:
            print("\n[5/7] Adding amygdala tissue labels...")
            with tqdm(total=1, desc="  Labeling amygdala regions", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
                tissue_labels = add_amygdala_to_mesh(nodes, elems, tissue_labels, affine)
                pbar.update(1)
            save_checkpoint(checkpoint_dir, step, tissue_labels)
    else:
        print("\n[5/7] Adding amygdala tissue labels...")
        with tqdm(total=1, desc="  Labeling amygdala regions", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
            tissue_labels = add_amygdala_to_mesh(nodes, elems, tissue_labels, affine)
            pbar.update(1)
        save_checkpoint(checkpoint_dir, step, tissue_labels)

    # Step 6: Neighbors
    step = "step6_neighbors"
    if resume:
        checkpoint = load_checkpoint(checkpoint_dir, step)
        if checkpoint:
            neighbors = checkpoint
            print("\n[6/7] Building neighbor connectivity... (loaded from checkpoint)")
        else:
            print("\n[6/7] Building neighbor connectivity...")
            neighbors = compute_tet_neighbors(elems)
            save_checkpoint(checkpoint_dir, step, neighbors)
    else:
        print("\n[6/7] Building neighbor connectivity...")
        neighbors = compute_tet_neighbors(elems)
        save_checkpoint(checkpoint_dir, step, neighbors)

    # Step 7: Save final mesh
    print("\n[7/7] Saving binary mesh...")
    with tqdm(total=1, desc="  Writing .mmcmesh file", bar_format='{l_bar}{bar}| {elapsed}') as pbar:
        save_mmcmesh(output_path, nodes, elems, tissue_labels, neighbors)
        pbar.update(1)

    # Clean up checkpoints on success (optional)
    if os.path.exists(checkpoint_dir):
        print(f"\n  🧹 Cleaning up checkpoints in {checkpoint_dir}")
        import shutil
        shutil.rmtree(checkpoint_dir)
        print("  ✅ Checkpoints removed")

    print(f"\n{'='*60}")
    print(f"  Total time: {time.time()-t_total:.1f}s")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

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
        print(f"  ✅ Amygdala tets: {len(amyg_vols):,}  (total vol: {amyg_vols.sum():.0f} mm³)")
    else:
        print(f"  ⚠️  WARNING: No amygdala tets found!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output',   default='mni152_head.mmcmesh',
                   help='Output .mmcmesh file path (default: mni152_head.mmcmesh)')
    p.add_argument('--max-vol',  type=float, default=50.0,
                   help='Max tet volume mm³ — smaller = denser mesh (default: 50)')
    p.add_argument('--min-dihedral', type=float, default=15.0,
                   help='Min dihedral angle in degrees (default: 15)')
    p.add_argument('--resume', action='store_true', default=True,
                   help='Resume from checkpoints if available (default: True)')
    p.add_argument('--no-resume', dest='resume', action='store_false',
                   help='Disable checkpoint resume (start from scratch)')
    p.add_argument('--checkpoint-dir', default=None,
                   help='Directory for checkpoints (default: <output>.checkpoints)')

    args = p.parse_args()

    # Dependency check
    missing = []
    for pkg in ['nilearn','nibabel','skimage','scipy','tqdm']:
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
                          min_dihedral=args.min_dihedral,
                          checkpoint_dir=args.checkpoint_dir,
                          resume=args.resume)


if __name__ == '__main__':
    main()
