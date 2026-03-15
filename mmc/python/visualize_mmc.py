#!/usr/bin/env python3
"""
Visualization utilities for MMC simulation results.

Creates plots of:
- TPSF histograms for each detector
- Sensitivity maps on MNI152 surface
- Photon path visualizations (sample)
"""

import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_tpsf(filepath: str) -> tuple:
    """Load TPSF histogram from binary file."""
    with open(filepath, 'rb') as f:
        num_dets = struct.unpack('i', f.read(4))[0]
        num_bins = struct.unpack('i', f.read(4))[0]
        data = np.fromfile(f, dtype=np.float32)
    
    tpsf = data.reshape((num_dets, num_bins))
    return num_dets, num_bins, tpsf


def load_path_records(filepath: str, max_records: int = 10000) -> np.ndarray:
    """Load photon path records."""
    record_dtype = np.dtype([
        ('exit_x', np.float32), ('exit_y', np.float32), ('exit_z', np.float32),
        ('dir_x', np.float32), ('dir_y', np.float32), ('dir_z', np.float32),
        ('total_pathlen', np.float32), ('amyg_pathlen', np.float32),
        ('tof', np.float32), ('weight', np.float32),
        ('bounces', np.int32), ('hit_amyg', np.int32),
    ])
    
    with open(filepath, 'rb') as f:
        count = struct.unpack('Q', f.read(8))[0]
        count = min(count, max_records)
        data = np.fromfile(f, dtype=record_dtype, count=count)
    
    return data


def plot_tpsf(tpsf: np.ndarray, wavelength: int, output_path: str):
    """Plot TPSF histograms for all detectors."""
    num_dets, num_bins = tpsf.shape
    time_axis = np.arange(num_bins) * 10  # 10 ps bins
    
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    axes = axes.flatten()
    
    for d in range(min(num_dets, 22)):
        ax = axes[d]
        tpsf_d = tpsf[d, :]
        
        ax.semilogy(time_axis, tpsf_d + 1e-10)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Counts')
        ax.set_title(f'Detector {d}')
        ax.set_xlim(0, 3000)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(num_dets, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'TPSF Histograms at {wavelength} nm', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved TPSF plot: {output_path}")


def plot_sensitivity_comparison(tpsf_760: np.ndarray, tpsf_850: np.ndarray, 
                                 output_path: str):
    """Plot sensitivity comparison between wavelengths."""
    
    # Sum over time to get CW sensitivity
    cw_760 = np.sum(tpsf_760, axis=1)
    cw_850 = np.sum(tpsf_850, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(cw_760))
    
    axes[0].bar(x, cw_760, alpha=0.7, label='760 nm', color='red')
    axes[0].set_xlabel('Detector Index')
    axes[0].set_ylabel('CW Sensitivity (counts)')
    axes[0].set_title('CW Sensitivity at 760 nm')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(x, cw_850, alpha=0.7, label='850 nm', color='blue')
    axes[1].set_xlabel('Detector Index')
    axes[1].set_ylabel('CW Sensitivity (counts)')
    axes[1].set_title('CW Sensitivity at 850 nm')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sensitivity comparison: {output_path}")


def plot_path_length_distribution(records_760: np.ndarray, 
                                   records_850: np.ndarray,
                                   output_path: str):
    """Plot path length distributions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if records_760 is not None and len(records_760) > 0:
        amyg_paths_760 = records_760[records_760['hit_amyg'] == 1]['amyg_pathlen']
        if len(amyg_paths_760) > 0:
            axes[0].hist(amyg_paths_760, bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[0].set_xlabel('Amygdala Path Length (mm)')
            axes[0].set_ylabel('Count')
            axes[0].set_title(f'760 nm - Amygdala Path Length\n(n={len(amyg_paths_760)})')
            axes[0].axvline(np.mean(amyg_paths_760), color='darkred', linestyle='--',
                          label=f'Mean: {np.mean(amyg_paths_760):.2f} mm')
            axes[0].legend()
    
    if records_850 is not None and len(records_850) > 0:
        amyg_paths_850 = records_850[records_850['hit_amyg'] == 1]['amyg_pathlen']
        if len(amyg_paths_850) > 0:
            axes[1].hist(amyg_paths_850, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[1].set_xlabel('Amygdala Path Length (mm)')
            axes[1].set_ylabel('Count')
            axes[1].set_title(f'850 nm - Amygdala Path Length\n(n={len(amyg_paths_850)})')
            axes[1].axvline(np.mean(amyg_paths_850), color='darkblue', linestyle='--',
                          label=f'Mean: {np.mean(amyg_paths_850):.2f} mm')
            axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved path length distribution: {output_path}")


def plot_time_of_flight_vs_amyg_path(records_760: np.ndarray,
                                      records_850: np.ndarray,
                                      output_path: str):
    """Plot correlation between TOF and amygdala path length."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if records_760 is not None and len(records_760) > 0:
        amyg_recs = records_760[records_760['hit_amyg'] == 1]
        if len(amyg_recs) > 0:
            axes[0].scatter(amyg_recs['tof'], amyg_recs['amyg_pathlen'], 
                          alpha=0.3, s=1, color='red')
            axes[0].set_xlabel('Time of Flight (ps)')
            axes[0].set_ylabel('Amygdala Path Length (mm)')
            axes[0].set_title('760 nm: TOF vs Amygdala Path')
            axes[0].set_xlim(0, 5000)
    
    if records_850 is not None and len(records_850) > 0:
        amyg_recs = records_850[records_850['hit_amyg'] == 1]
        if len(amyg_recs) > 0:
            axes[1].scatter(amyg_recs['tof'], amyg_recs['amyg_pathlen'],
                          alpha=0.3, s=1, color='blue')
            axes[1].set_xlabel('Time of Flight (ps)')
            axes[1].set_ylabel('Amygdala Path Length (mm)')
            axes[1].set_title('850 nm: TOF vs Amygdala Path')
            axes[1].set_xlim(0, 5000)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved TOF vs path plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize MMC results')
    parser.add_argument('--data-dir', default='results_mmc',
                       help='Directory containing simulation outputs')
    parser.add_argument('--output-dir', default='figures_mmc',
                       help='Directory for output figures')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load TPSF data
    tpsf_760_path = data_dir / 'tpsf_760mm.bin'
    tpsf_850_path = data_dir / 'tpsf_850mm.bin'
    
    tpsf_760 = None
    tpsf_850 = None
    
    if tpsf_760_path.exists():
        print("Loading 760 nm TPSF...")
        _, _, tpsf_760 = load_tpsf(str(tpsf_760_path))
        plot_tpsf(tpsf_760, 760, str(output_dir / 'tpsf_760nm.png'))
    
    if tpsf_850_path.exists():
        print("Loading 850 nm TPSF...")
        _, _, tpsf_850 = load_tpsf(str(tpsf_850_path))
        plot_tpsf(tpsf_850, 850, str(output_dir / 'tpsf_850nm.png'))
    
    if tpsf_760 is not None and tpsf_850 is not None:
        plot_sensitivity_comparison(tpsf_760, tpsf_850,
                                    str(output_dir / 'sensitivity_comparison.png'))
    
    # Load path records
    paths_760_path = data_dir / 'paths_760mm.bin'
    paths_850_path = data_dir / 'paths_850mm.bin'
    
    records_760 = None
    records_850 = None
    
    if paths_760_path.exists():
        print("Loading 760 nm path records...")
        records_760 = load_path_records(str(paths_760_path))
        print(f"  Loaded {len(records_760)} records")
    
    if paths_850_path.exists():
        print("Loading 850 nm path records...")
        records_850 = load_path_records(str(paths_850_path))
        print(f"  Loaded {len(records_850)} records")
    
    if records_760 is not None or records_850 is not None:
        plot_path_length_distribution(records_760, records_850,
                                      str(output_dir / 'path_length_dist.png'))
        plot_time_of_flight_vs_amyg_path(records_760, records_850,
                                         str(output_dir / 'tof_vs_path.png'))
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
