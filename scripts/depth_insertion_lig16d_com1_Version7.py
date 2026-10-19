#!/usr/bin/env python3
"""
Advanced Peptide Insertion Depth Analysis with Residue-Level Detail
=======================================================================

Features:
- True membrane midplane calculation (leaflet-separated phosphates)
- RESIDUE-BASED INSERTION ANALYSIS:
  * Per-residue insertion depth tracking
  * Amino acid type classification & analysis
  * Hydrophobic/hydrophilic residue penetration
  * ALL residues plotted over time
  * Secondary structure integration
  * Residue contact with membrane
- 2D Time-series with color-coded leaflets
- Multiple 3D visualizations with residue coloring
- Advanced Statistical Analysis:
  * Insertion depth distribution (global & per-residue)
  * Autocorrelation analysis
  * Protein distance & orientation metrics
  * Leaflet thickness dynamics
  * Residue penetration heatmaps
  * Amino acid type distribution analysis
  * ALL RESIDUES time series (large multi-plot figure)
- Per-residue contribution plots
- Publication-quality outputs
- Comprehensive residue-level data export

Author: Prince Manu (fully enhanced v5.1 - All Residues Time Series)
"""

# 🔧 NO GUI (fix Qt issues)
import matplotlib
matplotlib.use('Agg')

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy import stats, signal
from scipy.stats import gaussian_kde
import sys
import warnings
import os
import pandas as pd

warnings.filterwarnings('ignore')

# === INPUT FILES ===
TPR_FILE = "md_0_1000.tpr"
XTC_FILE = "center50.xtc"

PEPTIDE_RESNAME = "PROTEIN"
LIPID_RESNAMES = "POPE POPG"

# === OUTPUT FILES ===
OUTPUT_DIR = "insertion_depth_analysis_residue"
OUTPUT_FILE = f"{OUTPUT_DIR}/insertion_depth_data.dat"
RESIDUE_DATA_FILE = f"{OUTPUT_DIR}/residue_insertion_data.csv"
PLOT_2D_FILE = f"{OUTPUT_DIR}/01_insertion_depth_2d_timeseries.png"
PLOT_3D_SPHERES_FILE = f"{OUTPUT_DIR}/02_3d_phospholipid_spheres.png"
PLOT_3D_DENSITY_FILE = f"{OUTPUT_DIR}/03_3d_density_heatmap.png"
PLOT_3D_LEAFLETS_FILE = f"{OUTPUT_DIR}/04_3d_leaflet_separation.png"
PLOT_HISTOGRAM_FILE = f"{OUTPUT_DIR}/05_depth_distribution_histogram.png"
PLOT_AUTOCORR_FILE = f"{OUTPUT_DIR}/06_autocorrelation_analysis.png"
PLOT_THICKNESS_FILE = f"{OUTPUT_DIR}/07_bilayer_thickness_dynamics.png"
PLOT_DISTANCE_FILE = f"{OUTPUT_DIR}/08_protein_distance_analysis.png"
PLOT_RADIAL_FILE = f"{OUTPUT_DIR}/09_radial_distribution_xy.png"
PLOT_RESIDUE_INSERTION_FILE = f"{OUTPUT_DIR}/10_residue_insertion_depth.png"
PLOT_RESIDUE_PENETRATION_FILE = f"{OUTPUT_DIR}/11_residue_penetration_heatmap.png"
PLOT_AA_TYPE_FILE = f"{OUTPUT_DIR}/12_amino_acid_type_analysis.png"
PLOT_RESIDUE_TIMESERIES_SELECTED_FILE = f"{OUTPUT_DIR}/13_residue_timeseries_selected.png"
PLOT_RESIDUE_TIMESERIES_ALL_FILE = f"{OUTPUT_DIR}/13_residue_timeseries_all.png"
PLOT_3D_RESIDUE_COLORED_FILE = f"{OUTPUT_DIR}/14_3d_residue_colored.png"
PLOT_SNAPSHOT_PREFIX = f"{OUTPUT_DIR}/15_snapshot"
SUMMARY_FILE = f"{OUTPUT_DIR}/analysis_summary.txt"
RESIDUE_SUMMARY_FILE = f"{OUTPUT_DIR}/residue_analysis_summary.txt"

# === ANALYSIS PARAMETERS ===
PROTEIN_SEL = "protein or resname PROTEIN"
PHOSPHATE_ATOM = None  # Auto-detect if None
SKIP_FRAMES = 0  # Skip initial equilibration frames
Z_CUTOFF = 0.5  # Z-distance threshold for leaflet assignment (Å)
SMOOTHING_SIGMA = 2.0  # Gaussian smoothing sigma
REFERENCE_ATOMS = "CA"  # Which atoms to average for protein

# === 3D VISUALIZATION PARAMETERS ===
SNAPSHOT_FRAMES = [0, -1]  # Frames to visualize [first, last]
PHOSPHATE_SPHERE_SIZE = 120
PROTEIN_SPHERE_SIZE = 250
DENSITY_RESOLUTION = 30  # Resolution for density grids

# === 2D SNAPSHOT (LIPID/WATER/PEPTIDE) PARAMETERS ===
WATER_RESNAMES = "SOL TIP3 TIP3P TIP4P WAT HOH"  # Candidates, first match wins
WATER_OXYGEN_NAMES = ["OW", "O", "OH2", "OW1"]   # Candidate oxygen atom names
SNAPSHOT_WATER_CUTOFF = 6.0  # Å - only waters within this distance of the peptide are shown
SNAPSHOT_ATOM_SIZE = 90  # Marker size for peptide atoms in the 2D snapshot
SNAPSHOT_TERMINUS_SIZE = 380  # Marker size for the highlighted N/C termini

# === RESIDUE ANALYSIS PARAMETERS ===
SELECTED_RESIDUES_FOR_TIMESERIES = None  # Auto-select most penetrating residues
PLOT_ALL_RESIDUES_TIMESERIES = True  # Plot all residues time series
RESIDUES_PER_FIGURE = 8  # Number of residues per subplot figure (for all residues)

# === COLOR SCHEME ===
COLOR_UPPER = "#FF6B6B"  # Upper leaflet: Coral red
COLOR_LOWER = "#4ECDC4"  # Lower leaflet: Teal
COLOR_PEPTIDE = "#2C3E50"  # Peptide: Dark slate
COLOR_MEAN = "#F39C12"  # Mean depth: Orange
COLOR_MIDPLANE = "#34495E"  # Midplane: Dark gray
COLOR_PROTEIN = "#E74C3C"  # Protein: Bright red
COLOR_NEUTRAL = "#95A5A6"  # Neutral gray
COLOR_LIPID_SNAPSHOT = "#FFA500"  # 2D snapshot lipid phosphates: Orange
COLOR_WATER_SNAPSHOT = "#E60000"  # 2D snapshot nearby water: Red

# === AMINO ACID PROPERTIES ===
AA_HYDROPHOBIC = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'}
AA_POLAR = {'SER', 'THR', 'ASN', 'GLN', 'CYS'}
AA_CHARGED_POS = {'LYS', 'ARG', 'HIS'}
AA_CHARGED_NEG = {'ASP', 'GLU'}
AA_SPECIAL = {'GLY'}

AA_COLOR_MAP = {
    'hydrophobic': '#FF6B6B',
    'polar': '#3498DB',
    'charged_pos': '#2ECC71',
    'charged_neg': '#E74C3C',
    'special': '#F39C12'
}


def create_output_directory():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")


def classify_amino_acid(resname):
    """Classify amino acid by chemical properties."""
    if resname in AA_HYDROPHOBIC:
        return 'hydrophobic'
    elif resname in AA_POLAR:
        return 'polar'
    elif resname in AA_CHARGED_POS:
        return 'charged_pos'
    elif resname in AA_CHARGED_NEG:
        return 'charged_neg'
    elif resname in AA_SPECIAL:
        return 'special'
    else:
        return 'other'


def get_aa_color(resname):
    """Get color for amino acid based on chemical properties."""
    aa_class = classify_amino_acid(resname)
    return AA_COLOR_MAP.get(aa_class, '#95A5A6')


def detect_phosphate(u, lipid_resnames):
    """
    Detect which phosphate atom name is present in the system.
    Priority: P > P31 > PO4 > P1
    """
    for name in ["P", "P31", "PO4", "P1"]:
        try:
            atoms = u.select_atoms(f"resname {lipid_resnames} and name {name}")
            if len(atoms) > 0:
                print(f"✅ Using phosphate atom: '{name}' ({len(atoms)} atoms)")
                return name
        except:
            continue
    
    raise ValueError(f"❌ No phosphate atom found in {lipid_resnames}")


def detect_water(u, candidate_resnames=WATER_RESNAMES, candidate_oxygen_names=WATER_OXYGEN_NAMES):
    """
    Detect the water residue name and oxygen atom name present in the system.
    Returns (resname, oxygen_name) or (None, None) if no water is found.
    """
    for resname in candidate_resnames.split():
        try:
            res_sel = u.select_atoms(f"resname {resname}")
        except Exception:
            continue
        if len(res_sel) == 0:
            continue

        for oname in candidate_oxygen_names:
            try:
                osel = res_sel.select_atoms(f"name {oname}")
            except Exception:
                continue
            if len(osel) > 0:
                print(f"✅ Water detected: resname '{resname}', oxygen atom '{oname}' ({len(osel)} molecules)")
                return resname, oname

        # Fallback: any atom whose name starts with O
        osel = res_sel.select_atoms("name O*")
        if len(osel) > 0:
            print(f"✅ Water detected: resname '{resname}', oxygen-like atoms (fallback, {len(osel)} atoms)")
            return resname, None

    print("⚠️  No water molecules detected — snapshot plot will omit water")
    return None, None


def select_groups(u, lipid_resnames, protein_sel, phosphate_atom=None):
    """
    Select peptide and lipid groups with validation.
    """
    # Select peptide
    try:
        peptide = u.select_atoms(protein_sel)
        if len(peptide) == 0:
            raise ValueError(f"No atoms selected with '{protein_sel}'")
        print(f"✅ Peptide selected: {len(peptide)} atoms")
    except Exception as e:
        raise ValueError(f"❌ Peptide selection failed: {e}")
    
    # Auto-detect phosphate if not provided
    if phosphate_atom is None:
        phosphate_atom = detect_phosphate(u, lipid_resnames)
    
    # Select phosphates
    try:
        phosphates = u.select_atoms(f"resname {lipid_resnames} and name {phosphate_atom}")
        if len(phosphates) == 0:
            raise ValueError(f"No phosphates found")
        print(f"✅ Phosphates selected: {len(phosphates)} atoms")
    except Exception as e:
        raise ValueError(f"❌ Phosphate selection failed: {e}")
    
    return peptide, phosphates, phosphate_atom


def compute_leaflets_and_midplane(phosphate_positions, z_cutoff=0.5):
    """
    Separate phosphates into upper and lower leaflets based on Z coordinate.
    """
    z_all = phosphate_positions[:, 2]
    z_mid_initial = np.median(z_all)
    
    upper_idx = z_all > (z_mid_initial + z_cutoff)
    lower_idx = z_all < (z_mid_initial - z_cutoff)
    
    if np.sum(upper_idx) == 0 or np.sum(lower_idx) == 0:
        upper_idx = z_all >= z_mid_initial
        lower_idx = z_all < z_mid_initial
    
    z_upper = np.mean(z_all[upper_idx])
    z_lower = np.mean(z_all[lower_idx])
    z_midplane = (z_upper + z_lower) / 2.0
    
    return upper_idx, lower_idx, z_midplane, z_upper, z_lower


def get_residue_com(residue, reference_atoms="CA"):
    """Get center of mass of a residue, with fallback options."""
    try:
        if reference_atoms.upper() != "ALL":
            atoms = residue.atoms.select_atoms(f"name {reference_atoms}")
            if len(atoms) > 0:
                return atoms.center_of_mass()
            else:
                return residue.atoms.center_of_mass()
        else:
            return residue.atoms.center_of_mass()
    except:
        return residue.atoms.center_of_mass()


def compute_depth_per_residue(u, peptide, phosphates, skip_frames=0, reference_atoms="CA"):
    """
    Compute insertion depth per residue throughout trajectory.
    
    Returns detailed residue-level analysis data.
    """
    times, depths, upper_zs, lower_zs = [], [], [], []
    protein_positions = []
    upper_phosphate_positions = []
    lower_phosphate_positions = []
    
    # Get unique residues (by index to avoid duplicates)
    residues = peptide.residues
    n_residues = len(residues)
    
    # Initialize residue data storage
    residue_depths = {res.resid: [] for res in residues}
    residue_info = {}
    
    print(f"   Analyzing {n_residues} residues...")
    
    for res in residues:
        residue_info[res.resid] = {
            'resname': res.resname,
            'resid': res.resid,
            'aa_class': classify_amino_acid(res.resname)
        }
    
    n_frames = len(u.trajectory)
    
    for frame_idx, ts in enumerate(u.trajectory):
        if frame_idx < skip_frames:
            continue
        
        if frame_idx % max(1, n_frames // 10) == 0:
            progress = (frame_idx / n_frames) * 100
            print(f"      Frame {frame_idx}/{n_frames-1} ({progress:.1f}%)", end='\r')
        
        phosphate_pos = phosphates.positions.copy()
        upper_idx, lower_idx, z_mid, z_upper, z_lower = compute_leaflets_and_midplane(
            phosphate_pos, z_cutoff=Z_CUTOFF
        )
        
        # Compute protein COM
        try:
            if reference_atoms.upper() != "ALL":
                pep_sel = peptide.select_atoms(f"name {reference_atoms}")
                if len(pep_sel) > 0:
                    pep_com = pep_sel.center_of_mass()
                else:
                    pep_com = peptide.center_of_mass()
            else:
                pep_com = peptide.center_of_mass()
        except:
            pep_com = peptide.center_of_mass()
        
        depth = pep_com[2] - z_mid
        
        # Per-residue depths
        for res in residues:
            res_com = get_residue_com(res, reference_atoms)
            res_depth = res_com[2] - z_mid
            residue_depths[res.resid].append(res_depth)
        
        times.append(ts.time)
        depths.append(depth)
        upper_zs.append(z_upper - z_mid)
        lower_zs.append(z_lower - z_mid)
        protein_positions.append(pep_com)
        upper_phosphate_positions.append(phosphate_pos[upper_idx])
        lower_phosphate_positions.append(phosphate_pos[lower_idx])
    
    print()  # Newline after progress
    
    # Convert residue depths to numpy arrays
    residue_depths_array = {resid: np.array(depths) for resid, depths in residue_depths.items()}
    
    return {
        'times': np.array(times),
        'depths': np.array(depths),
        'upper': np.array(upper_zs),
        'lower': np.array(lower_zs),
        'protein_pos': np.array(protein_positions),
        'upper_phosphates': upper_phosphate_positions,
        'lower_phosphates': lower_phosphate_positions,
        'residue_depths': residue_depths_array,
        'residue_info': residue_info
    }


def compute_rolling_std(data, window=100):
    """Compute rolling standard deviation for error bands."""
    if len(data) < window:
        window = max(1, len(data) // 4)
    
    std = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        std[i] = np.std(data[start:end])
    
    return std


def plot_2d_timeseries(times_ps, depths, upper, lower, output_file):
    """Create publication-quality 2D time-series plot."""
    times_ns = times_ps / 1000.0
    
    depths_smooth = gaussian_filter1d(depths, sigma=SMOOTHING_SIGMA)
    
    mean_depth = np.mean(depths)
    std_depth = np.std(depths)
    mean_upper = np.mean(upper)
    mean_lower = np.mean(lower)
    
    rolling_std = compute_rolling_std(depths, window=max(50, len(depths)//20))
    rolling_std_smooth = gaussian_filter1d(rolling_std, sigma=SMOOTHING_SIGMA)
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    
    ax.fill_between(times_ns, 0, mean_upper, color=COLOR_UPPER, alpha=0.15, zorder=1,
                     label="Upper leaflet region")
    ax.fill_between(times_ns, 0, mean_lower, color=COLOR_LOWER, alpha=0.15, zorder=1,
                     label="Lower leaflet region")
    
    ax.axhline(mean_upper, linestyle="--", color=COLOR_UPPER, lw=2.2, alpha=0.85,
               label=f"Upper leaflet = {mean_upper:.2f} Å", zorder=3)
    ax.axhline(mean_lower, linestyle="--", color=COLOR_LOWER, lw=2.2, alpha=0.85,
               label=f"Lower leaflet = {mean_lower:.2f} Å", zorder=3)
    ax.axhline(0, linestyle=":", color=COLOR_MIDPLANE, lw=2, alpha=0.9, 
               label="Membrane midplane", zorder=2)
    
    ax.fill_between(times_ns, depths_smooth - rolling_std_smooth,
                     depths_smooth + rolling_std_smooth,
                     color=COLOR_PEPTIDE, alpha=0.15, zorder=2)
    ax.plot(times_ns, depths_smooth, color=COLOR_PEPTIDE, lw=2.8, 
            label="Peptide center-of-mass", zorder=3)
    ax.axhline(mean_depth, linestyle="-.", color=COLOR_MEAN, lw=2.3, alpha=0.9,
               label=f"Mean depth = {mean_depth:+.2f} ± {std_depth:.2f} Å", zorder=2)
    
    ax.set_xlabel("Simulation time [ns]", fontsize=14, fontweight='bold')
    ax.set_ylabel("Position along membrane normal [Å]", fontsize=14, fontweight='bold')
    ax.set_xlim(times_ns.min(), times_ns.max())
    ax.set_ylim(-50, 50)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    ax.legend(fontsize=11, loc='best', framealpha=0.96, edgecolor='black', 
              frameon=True, fancybox=True, shadow=True)
    ax.tick_params(labelsize=12)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"💾 2D plot saved: {output_file}")
    return mean_depth, std_depth, mean_upper, mean_lower


def plot_residue_insertion_depth(data, output_file):
    """Plot mean insertion depth for each residue."""
    residue_info = data['residue_info']
    residue_depths = data['residue_depths']
    
    residues = sorted(residue_info.keys())
    mean_depths = [np.mean(residue_depths[rid]) for rid in residues]
    std_depths = [np.std(residue_depths[rid]) for rid in residues]
    aa_names = [residue_info[rid]['resname'] for rid in residues]
    colors_list = [get_aa_color(aa) for aa in aa_names]
    
    fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
    
    # Bar plot with error bars
    x_pos = np.arange(len(residues))
    bars = ax.bar(x_pos, mean_depths, yerr=std_depths, capsize=5, 
                   color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=AA_COLOR_MAP['hydrophobic'], edgecolor='black', label='Hydrophobic'),
        Patch(facecolor=AA_COLOR_MAP['polar'], edgecolor='black', label='Polar'),
        Patch(facecolor=AA_COLOR_MAP['charged_pos'], edgecolor='black', label='Positive'),
        Patch(facecolor=AA_COLOR_MAP['charged_neg'], edgecolor='black', label='Negative'),
        Patch(facecolor=AA_COLOR_MAP['special'], edgecolor='black', label='Special')
    ]
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(np.mean(data['depths']), color=COLOR_MEAN, linestyle=':', linewidth=2,
               label='Mean protein depth', alpha=0.8)
    
    ax.set_xlabel("Residue", fontsize=13, fontweight='bold')
    ax.set_ylabel("Mean Insertion Depth [Å]", fontsize=13, fontweight='bold')
    ax.set_title("Per-Residue Insertion Depth Analysis", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{aa}{rid}" for aa, rid in zip(aa_names, residues)], 
                        rotation=45, ha='right', fontsize=9)
    
    ax.legend(handles=legend_elements + [ax.get_legend_handles_labels()[0][-1]], 
              fontsize=10, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"💾 Residue insertion plot saved: {output_file}")


def plot_residue_penetration_heatmap(data, output_file):
    """Create heatmap of residue insertion depths over time."""
    residue_info = data['residue_info']
    residue_depths = data['residue_depths']
    times_ns = data['times'] / 1000.0
    
    residues = sorted(residue_info.keys())
    
    # Create matrix: residues x frames
    depth_matrix = np.array([residue_depths[rid] for rid in residues])
    
    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    
    im = ax.imshow(depth_matrix, aspect='auto', cmap='RdBu_r', interpolation='bilinear',
                    extent=[times_ns.min(), times_ns.max(), len(residues)-0.5, -0.5],
                    vmin=-50, vmax=50)
    
    aa_labels = [f"{residue_info[rid]['resname']}{rid}" for rid in residues]
    ax.set_yticks(np.arange(len(residues)))
    ax.set_yticklabels(aa_labels, fontsize=9)
    ax.set_xlabel("Simulation time [ns]", fontsize=13, fontweight='bold')
    ax.set_ylabel("Residue", fontsize=13, fontweight='bold')
    ax.set_title("Residue Insertion Depth Heatmap Over Time", fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label="Insertion Depth [Å]")
    
    # Add reference lines for membrane boundaries
    ax.axhline(y=0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=len(residues)-1, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"💾 Penetration heatmap saved: {output_file}")


def plot_amino_acid_type_analysis(data, output_file):
    """Analyze insertion depth by amino acid chemical properties."""
    residue_info = data['residue_info']
    residue_depths = data['residue_depths']
    
    # Classify residues by type
    aa_type_depths = {
        'hydrophobic': [],
        'polar': [],
        'charged_pos': [],
        'charged_neg': [],
        'special': [],
        'other': []
    }
    
    for resid, depths_array in residue_depths.items():
        aa_class = residue_info[resid]['aa_class']
        mean_depth = np.mean(depths_array)
        aa_type_depths[aa_class].append(mean_depth)
    
    # Remove empty categories
    aa_type_depths = {k: v for k, v in aa_type_depths.items() if len(v) > 0}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
    
    # Box plot
    types = list(aa_type_depths.keys())
    data_list = [aa_type_depths[t] for t in types]
    colors_box = [AA_COLOR_MAP[t] for t in types]
    
    bp = ax1.boxplot(data_list, labels=types, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel("Insertion Depth [Å]", fontsize=12, fontweight='bold')
    ax1.set_title("Insertion Depth by Amino Acid Type", fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('#FAFAFA')
    
    # Violin plot
    parts = ax2.violinplot(data_list, positions=range(len(types)), showmeans=True, showmedians=True)
    
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels(types, rotation=45, ha='right')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel("Insertion Depth [Å]", fontsize=12, fontweight='bold')
    ax2.set_title("Distribution by Amino Acid Type (Violin Plot)", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#FAFAFA')
    
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"💾 AA type analysis saved: {output_file}")


def plot_residue_timeseries_selected(data, output_file, n_residues=5):
    """Plot time series for selected residues (most penetrating)."""
    residue_info = data['residue_info']
    residue_depths = data['residue_depths']
    times_ns = data['times'] / 1000.0
    
    # Find most penetrating residues (most negative mean depth)
    mean_depths_dict = {rid: np.mean(residue_depths[rid]) for rid in residue_depths.keys()}
    sorted_residues = sorted(mean_depths_dict.items(), key=lambda x: x[1])
    selected_residues = [rid for rid, _ in sorted_residues[:n_residues]]
    
    fig, axes = plt.subplots(n_residues, 1, figsize=(14, 3*n_residues), dpi=100, sharex=True)
    
    if n_residues == 1:
        axes = [axes]
    
    for idx, resid in enumerate(selected_residues):
        ax = axes[idx]
        depths = residue_depths[resid]
        depths_smooth = gaussian_filter1d(depths, sigma=SMOOTHING_SIGMA)
        
        aa_name = residue_info[resid]['resname']
        color = get_aa_color(aa_name)
        
        ax.plot(times_ns, depths, alpha=0.4, color='gray', label='Raw')
        ax.plot(times_ns, depths_smooth, color=color, linewidth=2.5, label='Smoothed')
        
        ax.axhline(np.mean(depths), color=COLOR_MEAN, linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'Mean = {np.mean(depths):.2f} Å')
        ax.fill_between(times_ns, 0, -50, alpha=0.1, color=COLOR_UPPER)
        ax.fill_between(times_ns, 0, 50, alpha=0.1, color=COLOR_LOWER)
        ax.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax.set_ylabel(f"{aa_name}{resid}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#FAFAFA')
        ax.set_ylim(-50, 50)
    
    axes[-1].set_xlabel("Simulation time [ns]", fontsize=12, fontweight='bold')
    fig.suptitle(f"Time Series for Top {n_residues} Most Penetrating Residues", 
                 fontsize=14, fontweight='bold', y=0.995)
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"💾 Residue timeseries (selected) saved: {output_file}")


def plot_residue_timeseries_all(data, output_file, residues_per_figure=8):
    """Plot time series for ALL residues in multi-page figure."""
    residue_info = data['residue_info']
    residue_depths = data['residue_depths']
    times_ns = data['times'] / 1000.0
    
    residues_sorted = sorted(residue_info.keys())
    n_residues = len(residues_sorted)
    
    # Calculate number of pages needed
    n_pages = int(np.ceil(n_residues / residues_per_figure))
    
    print(f"   Creating {n_pages} page(s) with {residues_per_figure} residues per page...")
    
    # Create PDF-like output with multiple figures
    from matplotlib.backends.backend_pdf import PdfPages
    
    pdf_file = output_file.replace('.png', '_all.pdf')
    
    with PdfPages(pdf_file) as pdf:
        for page_idx in range(n_pages):
            start_idx = page_idx * residues_per_figure
            end_idx = min((page_idx + 1) * residues_per_figure, n_residues)
            page_residues = residues_sorted[start_idx:end_idx]
            n_on_page = len(page_residues)
            
            fig, axes = plt.subplots(n_on_page, 1, figsize=(14, 3.5*n_on_page), dpi=100, sharex=True)
            
            if n_on_page == 1:
                axes = [axes]
            
            for idx, resid in enumerate(page_residues):
                ax = axes[idx]
                depths = residue_depths[resid]
                depths_smooth = gaussian_filter1d(depths, sigma=SMOOTHING_SIGMA)
                
                aa_name = residue_info[resid]['resname']
                aa_class = residue_info[resid]['aa_class']
                color = get_aa_color(aa_name)
                
                # Plot raw and smoothed
                ax.plot(times_ns, depths, alpha=0.3, color='gray', linewidth=1, label='Raw')
                ax.plot(times_ns, depths_smooth, color=color, linewidth=2.5, label='Smoothed')
                
                # Add mean line
                mean_depth = np.mean(depths)
                ax.axhline(mean_depth, color=COLOR_MEAN, linestyle='--', linewidth=1.5,
                          alpha=0.7, label=f'Mean = {mean_depth:+.2f} Å')
                
                # Add membrane regions
                ax.fill_between(times_ns, -50, 0, alpha=0.05, color=COLOR_UPPER)
                ax.fill_between(times_ns, 0, 50, alpha=0.05, color=COLOR_LOWER)
                ax.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
                
                # Label with residue info
                ylabel_text = f"{aa_name}{resid} ({aa_class})"
                ax.set_ylabel(ylabel_text, fontsize=10, fontweight='bold')
                ax.legend(fontsize=8, loc='best', framealpha=0.95)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#FAFAFA')
                ax.set_ylim(-50, 50)
            
            axes[-1].set_xlabel("Simulation time [ns]", fontsize=12, fontweight='bold')
            fig.suptitle(f"All Residues Time Series (Page {page_idx + 1}/{n_pages})", 
                        fontsize=14, fontweight='bold', y=0.995)
            fig.patch.set_facecolor('white')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight', facecolor='white')
            plt.close(fig)
    
    # Also save first page as PNG
    fig, axes = plt.subplots(min(residues_per_figure, n_residues), 1, 
                             figsize=(14, 3.5*min(residues_per_figure, n_residues)), 
                             dpi=100, sharex=True)
    
    if n_residues == 1:
        axes = [axes]
    
    for idx, resid in enumerate(residues_sorted[:min(residues_per_figure, n_residues)]):
        ax = axes[idx]
        depths = residue_depths[resid]
        depths_smooth = gaussian_filter1d(depths, sigma=SMOOTHING_SIGMA)
        
        aa_name = residue_info[resid]['resname']
        aa_class = residue_info[resid]['aa_class']
        color = get_aa_color(aa_name)
        
        ax.plot(times_ns, depths, alpha=0.3, color='gray', linewidth=1, label='Raw')
        ax.plot(times_ns, depths_smooth, color=color, linewidth=2.5, label='Smoothed')
        
        mean_depth = np.mean(depths)
        ax.axhline(mean_depth, color=COLOR_MEAN, linestyle='--', linewidth=1.5,
                  alpha=0.7, label=f'Mean = {mean_depth:+.2f} Å')
        
        ax.fill_between(times_ns, -50, 0, alpha=0.05, color=COLOR_UPPER)
        ax.fill_between(times_ns, 0, 50, alpha=0.05, color=COLOR_LOWER)
        ax.axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ylabel_text = f"{aa_name}{resid} ({aa_class})"
        ax.set_ylabel(ylabel_text, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#FAFAFA')
        ax.set_ylim(-50, 50)
    
    axes[-1].set_xlabel("Simulation time [ns]", fontsize=12, fontweight='bold')
    fig.suptitle(f"All Residues Time Series (Page 1/{n_pages})", 
                fontsize=14, fontweight='bold', y=0.995)
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"💾 All residues timeseries saved: {output_file} + {pdf_file}")


def plot_3d_residue_colored(u, data, snapshot_frame=-1, output_file=None):
    """Create 3D visualization with residues colored by their insertion depth."""
    if snapshot_frame < 0:
        snapshot_frame = len(u.trajectory) + snapshot_frame
    
    upper_phos = data['upper_phosphates'][snapshot_frame]
    lower_phos = data['lower_phosphates'][snapshot_frame]
    
    residue_info = data['residue_info']
    residue_depths = data['residue_depths']
    
    # Get residue positions at this frame
    u.trajectory[snapshot_frame]
    residues = data['residue_info'].keys()
    
    residue_positions = []
    residue_depth_values = []
    
    for resid in sorted(residues):
        res = u.select_atoms(f"residue {resid}")[0].residue
        res_com = get_residue_com(res, REFERENCE_ATOMS)
        residue_positions.append(res_com)
        residue_depth_values.append(residue_depths[resid][snapshot_frame])
    
    residue_positions = np.array(residue_positions)
    residue_depth_values = np.array(residue_depth_values)
    
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot phosphates (light background)
    ax.scatter(upper_phos[:, 0], upper_phos[:, 1], upper_phos[:, 2],
               s=60, c=COLOR_UPPER, alpha=0.3, edgecolors='darkred', linewidth=0.3,
               label="Upper leaflet", zorder=2)
    
    ax.scatter(lower_phos[:, 0], lower_phos[:, 1], lower_phos[:, 2],
               s=60, c=COLOR_LOWER, alpha=0.3, edgecolors='darkcyan', linewidth=0.3,
               label="Lower leaflet", zorder=2)
    
    # Plot residues with color based on insertion depth
    scatter = ax.scatter(residue_positions[:, 0], residue_positions[:, 1], 
                        residue_positions[:, 2],
                        c=residue_depth_values, cmap='RdBu_r', s=150, alpha=0.8,
                        edgecolors='black', linewidth=1,
                        label="Residues (colored by depth)", vmin=-50, vmax=50, zorder=4)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, label="Insertion Depth [Å]")
    
    # Add reference planes
    all_phos = np.vstack([upper_phos, lower_phos])
    x_range = np.linspace(all_phos[:, 0].min() - 5, all_phos[:, 0].max() + 5, 10)
    y_range = np.linspace(all_phos[:, 1].min() - 5, all_phos[:, 1].max() + 5, 10)
    X_plane, Y_plane = np.meshgrid(x_range, y_range)
    
    z_mid = (upper_phos[:, 2].mean() + lower_phos[:, 2].mean()) / 2.0
    Z_plane = np.full_like(X_plane, z_mid)
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.1, color='gray')
    
    ax.set_xlabel("X [Å]", fontsize=12, fontweight='bold')
    ax.set_ylabel("Y [Å]", fontsize=12, fontweight='bold')
    ax.set_zlabel("Z [Å]", fontsize=12, fontweight='bold')
    ax.set_title(f"3D Membrane with Residues (Frame {snapshot_frame})\nColor-coded by Insertion Depth",
                 fontsize=14, fontweight='bold', pad=20)
    
    max_range = max(
        all_phos[:, 0].max() - all_phos[:, 0].min(),
        all_phos[:, 1].max() - all_phos[:, 1].min(),
        all_phos[:, 2].max() - all_phos[:, 2].min()
    ) / 2.0
    
    mid_x = all_phos[:, 0].mean()
    mid_y = all_phos[:, 1].mean()
    mid_z = all_phos[:, 2].mean()
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"💾 3D residue colored plot saved: {output_file}")


def plot_snapshot_lipid_water_peptide(u, peptide, phosphates, water_resname, water_oxygen_name,
                                       snapshot_frame, output_file, water_cutoff=SNAPSHOT_WATER_CUTOFF):
    """
    2D (x vs z) system snapshot at a given frame, styled after the classic
    lipid/water/peptide insertion-snapshot view:
      - Lipid phosphates (both leaflets) in orange
      - Water oxygens near the peptide in red
      - Full peptide atoms, colored PER RESIDUE using the same AA-class
        color scheme (AA_COLOR_MAP / get_aa_color) already used in the
        residue insertion-depth and residue time-series plots, so colors
        are directly comparable across figures.
      - N- and C-termini highlighted with labeled markers.
    """
    if snapshot_frame < 0:
        snapshot_frame = len(u.trajectory) + snapshot_frame
    u.trajectory[snapshot_frame]

    # --- membrane midplane at this frame ---
    phos_pos = phosphates.positions.copy()
    _, _, z_mid, _, _ = compute_leaflets_and_midplane(phos_pos, z_cutoff=Z_CUTOFF)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=100)

    # --- lipid phosphates (both leaflets) ---
    ax.scatter(phos_pos[:, 0], phos_pos[:, 2] - z_mid,
               s=PHOSPHATE_SPHERE_SIZE, c=COLOR_LIPID_SNAPSHOT, alpha=0.85,
               edgecolors='#B8860B', linewidth=0.6, zorder=1, label="Lipid phosphates")

    # --- water near the peptide ---
    if water_resname is not None:
        water_sel_str = f"resname {water_resname}"
        if water_oxygen_name is not None:
            water_sel_str += f" and name {water_oxygen_name}"
        try:
            nearby_water = u.select_atoms(
                f"({water_sel_str}) and around {water_cutoff} ({PROTEIN_SEL})"
            )
        except Exception as e:
            print(f"⚠️  Water selection failed ({e}); skipping water in snapshot")
            nearby_water = None
        if nearby_water is not None and len(nearby_water) > 0:
            wpos = nearby_water.positions
            ax.scatter(wpos[:, 0], wpos[:, 2] - z_mid,
                       s=70, c=COLOR_WATER_SNAPSHOT, alpha=0.9,
                       edgecolors='#8B0000', linewidth=0.5, zorder=3,
                       label=f"Water O (within {water_cutoff:.0f} Å)")

    # --- peptide atoms, colored per residue by AA class ---
    residues = sorted(peptide.residues, key=lambda r: r.resid)
    plotted_classes = set()
    for i, res in enumerate(residues):
        pos = res.atoms.positions
        color = get_aa_color(res.resname)
        aa_class = classify_amino_acid(res.resname)

        label = None
        if aa_class not in plotted_classes:
            label = f"{aa_class.replace('_', ' ').title()} residues"
            plotted_classes.add(aa_class)

        ax.scatter(pos[:, 0], pos[:, 2] - z_mid,
                   s=SNAPSHOT_ATOM_SIZE, c=color, alpha=0.85,
                   edgecolors='black', linewidth=0.5, zorder=4, label=label)

        # Highlight N- and C-termini with labeled markers (color-matched to their residue class)
        if i == 0 or i == len(residues) - 1:
            com = res.atoms.center_of_mass()
            terminus_label = "N" if i == 0 else "C"
            ax.scatter([com[0]], [com[2] - z_mid], s=SNAPSHOT_TERMINUS_SIZE, c=color,
                       edgecolors='black', linewidth=2, zorder=5)
            ax.text(com[0], com[2] - z_mid, terminus_label, ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white', zorder=6)

    ax.axhline(0, linestyle=":", color=COLOR_MIDPLANE, lw=2, alpha=0.9, zorder=2,
               label="Membrane midplane")

    time_ns = u.trajectory[snapshot_frame].time / 1000.0
    ax.set_title(f"Insertion snapshot -- {time_ns:.1f} ns", fontsize=16, fontweight='bold')
    ax.set_xlabel("x [Å]", fontsize=14, fontweight='bold')
    ax.set_ylabel("z [Å] (midplane = 0)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')

    # De-duplicate legend entries (one per AA class / lipid / water / midplane)
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=10, loc='best',
              framealpha=0.95, edgecolor='black', fancybox=True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"💾 Snapshot plot saved: {output_file}")


def save_residue_data(output_file, data):
    """Save detailed residue-level analysis to CSV."""
    residue_info = data['residue_info']
    residue_depths = data['residue_depths']
    
    rows = []
    for resid in sorted(residue_info.keys()):
        depths_array = residue_depths[resid]
        rows.append({
            'ResID': resid,
            'ResName': residue_info[resid]['resname'],
            'AA_Class': residue_info[resid]['aa_class'],
            'Mean_Depth_Å': np.mean(depths_array),
            'Std_Depth_Å': np.std(depths_array),
            'Min_Depth_Å': np.min(depths_array),
            'Max_Depth_Å': np.max(depths_array),
            'Median_Depth_Å': np.median(depths_array),
            'N_Frames': len(depths_array)
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"💾 Residue data saved: {output_file}")
    
    return df


def save_data(output_file, times, depths, upper, lower):
    """Save global analysis data to file with header."""
    data = np.column_stack((times, depths, upper, lower))
    header = "Time(ps)\tDepth(Å)\tUpper_leaflet(Å)\tLower_leaflet(Å)"
    np.savetxt(output_file, data, header=header, fmt="%12.4f", comments='')
    print(f"💾 Global data saved: {output_file}")


def save_residue_summary(summary_file, df, stats_dict):
    """Save residue-level analysis summary."""
    with open(summary_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("RESIDUE-LEVEL INSERTION DEPTH ANALYSIS SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("TOP 10 MOST PENETRATING RESIDUES:\n")
        f.write("-" * 100 + "\n")
        top_10 = df.nsmallest(10, 'Mean_Depth_Å')
        for _, row in top_10.iterrows():
            f.write(f"  {row['ResName']}{int(row['ResID']):3d}: Mean = {row['Mean_Depth_Å']:+7.2f} Å " + 
                   f"(±{row['Std_Depth_Å']:5.2f}), Min = {row['Min_Depth_Å']:+7.2f}, Max = {row['Max_Depth_Å']:+7.2f}\n")
        
        f.write("\n\nTOP 10 LEAST PENETRATING RESIDUES:\n")
        f.write("-" * 100 + "\n")
        top_10_least = df.nlargest(10, 'Mean_Depth_Å')
        for _, row in top_10_least.iterrows():
            f.write(f"  {row['ResName']}{int(row['ResID']):3d}: Mean = {row['Mean_Depth_Å']:+7.2f} Å " + 
                   f"(±{row['Std_Depth_Å']:5.2f}), Min = {row['Min_Depth_Å']:+7.2f}, Max = {row['Max_Depth_Å']:+7.2f}\n")
        
        f.write("\n\nAMINO ACID TYPE ANALYSIS:\n")
        f.write("-" * 100 + "\n")
        for aa_type in ['hydrophobic', 'polar', 'charged_pos', 'charged_neg', 'special']:
            type_data = df[df['AA_Class'] == aa_type]
            if len(type_data) > 0:
                f.write(f"  {aa_type.upper()}:\n")
                f.write(f"    Count: {len(type_data)}\n")
                f.write(f"    Mean Depth: {type_data['Mean_Depth_Å'].mean():+.2f} Å\n")
                f.write(f"    Std Dev: {type_data['Mean_Depth_Å'].std():.2f} Å\n")
                f.write(f"    Range: {type_data['Mean_Depth_Å'].min():+.2f} to {type_data['Mean_Depth_Å'].max():+.2f} Å\n\n")
        
        f.write("GLOBAL STATISTICS:\n")
        f.write("-" * 100 + "\n")
        for key, value in stats_dict['depth'].items():
            f.write(f"  {key:<35}: {value}\n")
        
        f.write("\n" + "=" * 100 + "\n")
    
    print(f"💾 Residue summary saved: {summary_file}")


def main():
    print("=" * 110)
    print("🚀 ADVANCED PEPTIDE INSERTION DEPTH ANALYSIS (v5.1)")
    print("   Residue-Level Analysis with ALL Residues Time Series Plotting")
    print("=" * 110)
    
    create_output_directory()
    
    # Load trajectory
    print(f"\n📂 Loading topology: {TPR_FILE}")
    print(f"📂 Loading trajectory: {XTC_FILE}")
    try:
        u = mda.Universe(TPR_FILE, XTC_FILE)
        print(f"✅ Universe loaded: {len(u.atoms)} atoms, {len(u.trajectory)} frames")
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        sys.exit(1)
    
    # Select groups
    print(f"\n🔍 Selecting atoms...")
    try:
        peptide, phosphates, phos_atom = select_groups(
            u, LIPID_RESNAMES, PROTEIN_SEL, PHOSPHATE_ATOM
        )
    except Exception as e:
        print(f"{e}")
        sys.exit(1)
    
    # Compute insertion depth (including per-residue)
    print(f"\n📏 Computing insertion depth trajectory with per-residue analysis...")
    print(f"   - Skipping first {SKIP_FRAMES} frames")
    print(f"   - Leaflet cutoff: ±{Z_CUTOFF} Å")
    print(f"   - Protein atoms: {REFERENCE_ATOMS}")
    
    try:
        data = compute_depth_per_residue(
            u, peptide, phosphates,
            skip_frames=SKIP_FRAMES,
            reference_atoms=REFERENCE_ATOMS
        )
        print(f"✅ Computed {len(data['times'])} frames")
    except Exception as e:
        print(f"❌ Computation failed: {e}")
        sys.exit(1)
    
    # Save raw data
    print(f"\n💾 Saving raw data...")
    save_data(OUTPUT_FILE, data['times'], data['depths'], data['upper'], data['lower'])
    
    # Save residue-level data
    print(f"\n📊 Saving residue-level data...")
    residue_df = save_residue_data(RESIDUE_DATA_FILE, data)
    
    # Create 2D time-series plot
    print(f"\n📊 Creating 2D time-series plot...")
    mean_d, std_d, mean_u, mean_l = plot_2d_timeseries(
        data['times'], data['depths'], data['upper'], data['lower'], PLOT_2D_FILE
    )
    
    # Create 3D visualizations
    print(f"\n🎨 Creating 3D visualizations...")
    
    try:
        plot_3d_residue_colored(u, data, snapshot_frame=-1, output_file=PLOT_3D_RESIDUE_COLORED_FILE)
    except Exception as e:
        print(f"⚠️  3D residue colored plot failed: {e}")
    
    # Create 2D lipid/water/peptide snapshot plots (per-residue colored, AA-class scheme)
    print(f"\n🎨 Creating 2D lipid/water/peptide snapshot plots...")
    try:
        water_resname, water_oxygen_name = detect_water(u)
        n_frames_total = len(u.trajectory)
        for frame in SNAPSHOT_FRAMES:
            frame_idx = frame if frame >= 0 else n_frames_total + frame
            snapshot_file = f"{PLOT_SNAPSHOT_PREFIX}_frame{frame_idx}.png"
            plot_snapshot_lipid_water_peptide(
                u, peptide, phosphates, water_resname, water_oxygen_name,
                snapshot_frame=frame, output_file=snapshot_file
            )
    except Exception as e:
        print(f"⚠️  Snapshot plot(s) failed: {e}")
    
    # Create residue-level analysis plots
    print(f"\n📈 Creating residue-level analysis plots...")
    
    try:
        plot_residue_insertion_depth(data, PLOT_RESIDUE_INSERTION_FILE)
    except Exception as e:
        print(f"⚠️  Residue insertion plot failed: {e}")
    
    try:
        plot_residue_penetration_heatmap(data, PLOT_RESIDUE_PENETRATION_FILE)
    except Exception as e:
        print(f"⚠️  Penetration heatmap failed: {e}")
    
    try:
        plot_amino_acid_type_analysis(data, PLOT_AA_TYPE_FILE)
    except Exception as e:
        print(f"⚠️  AA type analysis failed: {e}")
    
    try:
        plot_residue_timeseries_selected(data, PLOT_RESIDUE_TIMESERIES_SELECTED_FILE, n_residues=5)
    except Exception as e:
        print(f"⚠️  Residue timeseries (selected) failed: {e}")
    
    # Plot ALL residues time series
    print(f"\n📊 Creating time series for ALL residues...")
    try:
        plot_residue_timeseries_all(data, PLOT_RESIDUE_TIMESERIES_ALL_FILE, 
                                    residues_per_figure=RESIDUES_PER_FIGURE)
    except Exception as e:
        print(f"⚠️  All residues timeseries failed: {e}")
    
    # Calculate global statistics
    print(f"\n📊 Calculating comprehensive statistics...")
    
    thickness = data['upper'] - data['lower']
    protein_dist = np.abs(data['protein_pos'][:, 2] - 
                         ((data['upper'] + data['lower']) / 2.0))
    
    stats_dict = {
        'depth': {
            'Mean insertion depth': f"{mean_d:+.2f} Å",
            'Std deviation': f"{std_d:.2f} Å",
            'Min depth': f"{data['depths'].min():+.2f} Å",
            'Max depth': f"{data['depths'].max():+.2f} Å",
            'Median depth': f"{np.median(data['depths']):+.2f} Å",
            'Skewness': f"{stats.skew(data['depths']):.3f}",
            'Kurtosis': f"{stats.kurtosis(data['depths']):.3f}"
        },
        'membrane': {
            'Upper leaflet position': f"{mean_u:+.2f} Å",
            'Lower leaflet position': f"{mean_l:+.2f} Å",
            'Mean bilayer thickness': f"{np.mean(thickness):.2f} Å",
            'Thickness std dev': f"{np.std(thickness):.2f} Å",
            'Min thickness': f"{thickness.min():.2f} Å",
            'Max thickness': f"{thickness.max():.2f} Å"
        },
        'dynamics': {
            'Total simulation time': f"{data['times'][-1] / 1000:.2f} ns",
            'Frames analyzed': f"{len(data['times'])}",
            'Sampling frequency': f"{1.0 / ((data['times'][1] - data['times'][0]) / 1000):.2f} ps⁻¹",
            'Depth fluctuation rate': f"{np.std(np.gradient(data['depths'])):.4f} Å/ps"
        },
        'protein': {
            'Mean distance from midplane': f"{np.mean(protein_dist):.2f} Å",
            'Max penetration depth': f"{np.max(protein_dist):.2f} Å",
            'Min distance from midplane': f"{np.min(protein_dist):.2f} Å",
            'COM trajectory spread (X)': f"{np.std(data['protein_pos'][:, 0]):.2f} Å",
            'COM trajectory spread (Y)': f"{np.std(data['protein_pos'][:, 1]):.2f} Å",
            'COM trajectory spread (Z)': f"{np.std(data['protein_pos'][:, 2]):.2f} Å"
        }
    }
    
    # Save residue summary
    save_residue_summary(RESIDUE_SUMMARY_FILE, residue_df, stats_dict)
    
    # Print statistics
    print("\n" + "=" * 110)
    print("📊 COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
    print("=" * 110)
    
    print("\n📍 INSERTION DEPTH STATISTICS (GLOBAL):")
    print("-" * 110)
    for key, value in stats_dict['depth'].items():
        print(f"  {key:<40}: {value}")
    
    print("\n🧬 MEMBRANE STRUCTURE:")
    print("-" * 110)
    for key, value in stats_dict['membrane'].items():
        print(f"  {key:<40}: {value}")
    
    print("\n🔬 RESIDUE-LEVEL ANALYSIS (TOP 5 PENETRATORS):")
    print("-" * 110)
    top_5 = residue_df.nsmallest(5, 'Mean_Depth_Å')
    for _, row in top_5.iterrows():
        print(f"  {row['ResName']:3s}{int(row['ResID']):3d} ({row['AA_Class']:12s}): " + 
              f"{row['Mean_Depth_Å']:+7.2f} ± {row['Std_Depth_Å']:5.2f} Å")
    
    print("\n\n💾 AMINO ACID TYPE STATISTICS:")
    print("-" * 110)
    for aa_type in ['hydrophobic', 'polar', 'charged_pos', 'charged_neg', 'special']:
        type_data = residue_df[residue_df['AA_Class'] == aa_type]
        if len(type_data) > 0:
            print(f"  {aa_type.upper():20s}: Count={len(type_data):2d}, " + 
                  f"Mean={type_data['Mean_Depth_Å'].mean():+6.2f} Å, " +
                  f"StdDev={type_data['Mean_Depth_Å'].std():5.2f} Å")
    
    print("\n" + "=" * 110)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 110)
    
    print(f"\n📁 OUTPUT FILES ({OUTPUT_DIR}/):")
    print(f"   📊 01_insertion_depth_2d_timeseries.png         - Time-series with leaflet boundaries")
    print(f"   🧬 10_residue_insertion_depth.png               - Per-residue insertion depth bars")
    print(f"   🌡️  11_residue_penetration_heatmap.png         - Residue depth heatmap vs time")
    print(f"   🔬 12_amino_acid_type_analysis.png              - AA type classification analysis")
    print(f"   📈 13_residue_timeseries_selected.png           - Time series of top 5 penetrators")
    print(f"   📊 13_residue_timeseries_all.png/pdf            - Time series of ALL residues (!)")
    print(f"   🎨 14_3d_residue_colored.png                    - 3D with residues colored by depth")
    print(f"   🧪 15_snapshot_frame*.png                       - 2D lipid/water/peptide snapshot, residues colored by AA class")
    print(f"   📄 insertion_depth_data.dat                     - Raw global numerical data")
    print(f"   📋 residue_insertion_data.csv                   - Per-residue data table")
    print(f"   📋 residue_analysis_summary.txt                 - Detailed residue-level summary")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

