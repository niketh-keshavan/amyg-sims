#!/usr/bin/env python3
"""
Debug script for face_pair array in MMC mesh.
Checks for:
1. -1 values (boundary faces)
2. Invalid face indices (should be 0-3)
3. Asymmetry (if face_pair[a][f] = g, then face_pair[neighbor][g] should point back)
"""

import struct
import sys

def read_mmcmesh(mesh_path):
    """Read mesh and return neighbor array"""
    with open(mesh_path, 'rb') as f:
        # Header
        magic = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('I', f.read(4))[0]
        num_nodes = struct.unpack('I', f.read(4))[0]
        num_elems = struct.unpack('I', f.read(4))[0]
        bbox = struct.unpack('6f', f.read(24))
        
        print(f"Mesh: {num_nodes} nodes, {num_elems} elements")
        
        # Skip nodes and elements
        f.seek(48 + num_nodes*3*4 + num_elems*4*4)
        
        # Read tissue (skip)
        f.seek(num_elems*4, 1)
        
        # Read neighbors
        neighbors = []
        for i in range(num_elems):
            n = struct.unpack('4i', f.read(16))
            neighbors.append(n)
        
        return num_elems, neighbors

def compute_face_pair(num_elems, neighbors):
    """Compute face_pair array same as mmc_mesh.cu"""
    face_pair = []
    
    for e in range(num_elems):
        elem_pair = []
        for f in range(4):
            neighbor = neighbors[e][f]
            if neighbor >= 0:
                # Find which face of neighbor points back to e
                entry_face = -1
                for nf in range(4):
                    if neighbors[neighbor][nf] == e:
                        entry_face = nf
                        break
                elem_pair.append(entry_face)
            else:
                elem_pair.append(-1)  # boundary face
        face_pair.append(elem_pair)
    
    return face_pair

def validate_face_pair(num_elems, neighbors, face_pair):
    """Validate face_pair array"""
    errors = []
    warnings = []
    
    boundary_count = 0
    valid_count = 0
    invalid_count = 0
    
    for e in range(num_elems):
        for f in range(4):
            neighbor = neighbors[e][f]
            fp = face_pair[e][f]
            
            if neighbor < 0:
                # Boundary face
                if fp != -1:
                    errors.append(f"Element {e}, face {f}: boundary but face_pair = {fp} (should be -1)")
                boundary_count += 1
            else:
                # Interior face
                if fp < 0 or fp > 3:
                    errors.append(f"Element {e}, face {f}: face_pair = {fp} (should be 0-3)")
                    invalid_count += 1
                else:
                    valid_count += 1
                    
                    # Check symmetry: face_pair[neighbor][fp] should point back to e
                    back_face = face_pair[neighbor][fp]
                    if back_face != f:
                        errors.append(f"Asymmetry: e={e}, f={f}, neighbor={neighbor}, fp={fp}, but back_face={back_face}")
    
    print(f"\nFace Pair Statistics:")
    print(f"  Boundary faces: {boundary_count}")
    print(f"  Valid interior: {valid_count}")
    print(f"  Invalid: {invalid_count}")
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for err in errors[:10]:  # Show first 10
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print(f"\n✓ All face_pair entries valid")
    
    return len(errors) == 0

def main():
    mesh_path = "mni152_head.mmcmesh"
    
    print(f"Reading mesh: {mesh_path}")
    num_elems, neighbors = read_mmcmesh(mesh_path)
    
    print(f"\nComputing face_pair array...")
    face_pair = compute_face_pair(num_elems, neighbors)
    
    print(f"Validating...")
    valid = validate_face_pair(num_elems, neighbors, face_pair)
    
    # Sample some entries
    print(f"\nSample face_pair entries:")
    for e in [0, 100, 1000, 10000]:
        if e < num_elems:
            print(f"  Element {e}: neighbors={neighbors[e]}, face_pair={face_pair[e]}")
    
    # Find elements with amygdala neighbors
    print(f"\nLooking for amygdala-adjacent faces...")
    # Note: We don't have tissue array loaded here, skipping
    
    return 0 if valid else 1

if __name__ == '__main__':
    sys.exit(main())
