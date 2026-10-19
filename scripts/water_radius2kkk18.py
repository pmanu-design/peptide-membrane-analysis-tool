#!/usr/bin/env python3
"""
Enhanced Publication-Quality Water and Pore Analysis Script
Comprehensive molecular dynamics analysis with 3D visualizations and professional-grade figures.

Features:
- Advanced water density analysis in shells
- 3D visualization of water molecules in pores
- Peptide/protein-induced pore detection and characterization
- Membrane evolution tracking
- Pore radius calculation and time evolution
- Water permeation quantification through pores
- Ion permeation quantification through pores
- Membrane-normal depth-of-permeation traces for water & ions ("depth of
  insertion"-style plots: raw trace, mean, ±1σ, bilayer-center reference)
- Bilayer-midplane occupancy for water & ions (hydration-defect / pore-
  precursor indicator: species count within a slab centered on Z = 0)
- Combined water permeation & pore radius time-evolution (individual figure/CSV + dashboard panel)
- Publication-quality statistical analysis and visualization
- 3D trajectory animations
- Contour and surface plots
- Phase diagrams and correlations

Author: Enhanced MD Analysis Suite
Date: 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm, colors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults with a clean, pure-white background
# (compatible with older matplotlib versions)
try:
    plt.style.use('seaborn-v0_8-white')
except:
    try:
        plt.style.use('seaborn-white')
    except:
        plt.style.use('default')

plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'patch.linewidth': 1.5,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.color': '#d9d9d9',
    # --- Pure white backgrounds everywhere ---
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
    'legend.facecolor': 'white',
    'legend.edgecolor': '#cccccc',
})

# Color schemes for publication quality
color_palette = {
    'water': '#1f77b4',
    'solute': '#d62728',
    'pore': '#2ca02c',
    'membrane': '#ff7f0e',
    'background': '#ffffff'
}


def whiten_3d_axes(ax):
    """Force a 3D Axes to render on a pure-white background.
    Matplotlib's 3D panes default to a light-gray fill even with a
    white-based style, so they need to be set explicitly."""
    ax.set_facecolor('white')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor('white')
        pane.set_edgecolor('#dddddd')
        pane.set_alpha(1.0)
    ax.xaxis._axinfo['grid']['color'] = (0.85, 0.85, 0.85, 0.5)
    ax.yaxis._axinfo['grid']['color'] = (0.85, 0.85, 0.85, 0.5)
    ax.zaxis._axinfo['grid']['color'] = (0.85, 0.85, 0.85, 0.5)

# ============================================================================
# === USER PARAMETERS ===
# ============================================================================
topol_file = "md_0_300.tpr"
traj_file = "cen10.xtc"

# Selection strings
solute_selection = "not resname TIP3 and not resname ION"  # Peptide/protein
water_selection = "resname TIP3"
ion_selection = "resname ION"  # Edit resnames as needed, e.g. "resname SOD CLA POT"
membrane_selection = "resname OYPE PMPE PMPG POPE PVCL2 PYPG QMPE"
output_prefix = "enhanced_water_pore_analysis_pub"
align_solute = True

# ============================================================================
# === OUTPUT FOLDER ===
# ============================================================================
# All output files (CSVs, figures, report) are written into this folder,
# which is created automatically if it doesn't already exist. The folder is
# named after output_prefix, and output_prefix is then redefined to point
# inside it, so every existing f"{output_prefix}_..." path below is
# automatically saved into the folder without any further changes.
output_dir = output_prefix
os.makedirs(output_dir, exist_ok=True)
output_prefix = os.path.join(output_dir, output_prefix)
print(f"[✓] All output will be saved in: {os.path.abspath(output_dir)}")

# Shell selection (Å)
shell_inner = 0
shell_outer = 15

# Radial cutoff (Å, from the solute's XY center) defining the permeation
# channel used to track how deep water/ions travel along the membrane normal
# (used by the "membrane normal" depth-of-permeation trace plots).
permeation_channel_radius = shell_outer

# Half-thickness (Å) of the slab centered on the bilayer midplane (Z = 0,
# i.e. the hydrophobic core center) used to count water molecules that have
# fully crossed into the membrane center -- a standard hydration-defect /
# pore-precursor indicator.
midplane_half_thickness = 2.0

# Padding (Å) added to the membrane's instantaneous XY bounding box when
# testing whether a water/ion is laterally "inside" the membrane patch, for
# the (protein-independent) bilayer-midplane occupancy calculation. This
# keeps midplane counting meaningful even when no peptide/protein is present
# (e.g. pure-lipid control simulations), instead of relying on a solute-
# centered radial channel.
membrane_footprint_padding = 3.0

# Pore detection parameters
pore_detection_method = "density_based"  # 'density_based' or 'geometric'
pore_radius_threshold = 3.0
water_density_threshold = 0.3
dbscan_eps = 2.5
dbscan_min_samples = 3

# Membrane parameters
membrane_thickness = 4.0
membrane_center_z = 0.0

# KDE smoothing
apply_kde_smoothing = True
kde_sigma = 1.0

# Z-axis binning (Å)
z_bins = np.linspace(-20, 20, 50)
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

# Time binning for evolution analysis
time_bin_size = 10

# 3D Visualization parameters
save_3d_plots = True
frame_indices_3d = [0, -1]  # First and last frames
max_waters_to_plot = 500  # Limit for clarity

# ============================================================================
# === FILE CHECKS ===
# ============================================================================
if not os.path.isfile(topol_file):
    raise FileNotFoundError(f"Missing topology file: {topol_file}")
if not os.path.isfile(traj_file):
    raise FileNotFoundError(f"Missing trajectory file: {traj_file}")

# ============================================================================
# === LOAD TRAJECTORY ===
# ============================================================================
print("[INFO] Loading trajectory...")
u = mda.Universe(topol_file, traj_file)

solute = u.select_atoms(solute_selection)
has_protein = solute.n_atoms > 0
if has_protein:
    print(f"[INFO] Found {solute.n_atoms} peptide/protein atoms — "
          "peptide/protein-induced pore analysis ENABLED.")
else:
    print(f"[WARNING] No atoms matched solute selection '{solute_selection}'. "
          "No peptide/protein is present in this system, so all pore "
          "detection, pore radius, pore lifetime/stability, and "
          "water-permeation-through-pore analyses (which are specifically "
          "about peptide/protein-induced pores) will be SKIPPED and "
          "reported as zero/not-applicable. Membrane, water-distribution, "
          "and bilayer-midplane analyses still run normally.")

waters = u.select_atoms(water_selection)
if waters.n_atoms == 0:
    raise ValueError(f"No atoms found for water selection: '{water_selection}'")

try:
    membrane = u.select_atoms(membrane_selection)
    has_membrane = membrane.n_atoms > 0
    print(f"[INFO] Found {membrane.n_atoms} membrane atoms")
except:
    has_membrane = False
    print("[WARNING] Membrane selection failed. Proceeding without membrane analysis.")

try:
    ions = u.select_atoms(ion_selection)
    has_ions = ions.n_atoms > 0
    if has_ions:
        print(f"[INFO] Found {ions.n_atoms} ion atoms")
    else:
        print(f"[WARNING] No atoms matched ion selection '{ion_selection}'. "
              "Ion permeation panel will be skipped.")
except Exception:
    has_ions = False
    print("[WARNING] Ion selection failed. Proceeding without ion permeation analysis.")

# ============================================================================
# === ALIGNMENT ===
# ============================================================================
if align_solute and has_protein:
    print(f"[INFO] Aligning solute using selection: '{solute_selection}'...")
    ref = u.copy()
    alignto(u, ref, select=solute_selection)
elif align_solute and not has_protein:
    print("[INFO] Skipping alignment (align_solute=True but no peptide/protein "
          "atoms found).")

# ============================================================================
# === LEAFLET ASSIGNMENT & DYNAMIC BILAYER-MIDPLANE SETUP ===
# The bilayer midplane is NOT assumed to sit at a fixed Z; instead it is
# recomputed every frame as the midpoint between the two leaflets' centers
# of mass, and every Z-coordinate used downstream (water, ion, solute,
# pore z-bins) is recentered relative to it. This guarantees the membrane
# midplane is exactly Z = 0 in every frame, regardless of drift/whole-
# system translation over the trajectory.
#
# Leaflet MEMBERSHIP (which residues are "upper" vs "lower") is assigned
# once from the first frame and then tracked for the rest of the
# trajectory -- standard practice, since lipids essentially never flip-flop
# between leaflets on MD timescales.
# ============================================================================
leaflet_upper_ag = None
leaflet_lower_ag = None
if has_membrane:
    u.trajectory[0]
    ref_mem_com_z = membrane.center_of_mass()[2]
    res_com_z = np.array([res.atoms.center_of_mass()[2] for res in membrane.residues])
    upper_res = membrane.residues[res_com_z > ref_mem_com_z]
    lower_res = membrane.residues[res_com_z <= ref_mem_com_z]
    if len(upper_res) > 0 and len(lower_res) > 0:
        leaflet_upper_ag = upper_res.atoms
        leaflet_lower_ag = lower_res.atoms
        print(f"[INFO] Leaflets assigned from frame 0: upper = {leaflet_upper_ag.n_atoms} "
              f"atoms ({len(upper_res)} residues), lower = {leaflet_lower_ag.n_atoms} "
              f"atoms ({len(lower_res)} residues)")
    else:
        print("[WARNING] Could not split the membrane into two leaflets "
              "(all residues fell on one side of the COM); falling back to "
              "the whole-membrane COM for the midplane, and leaflet "
              "reference lines will be omitted from plots.")
else:
    print("[WARNING] No membrane atoms found — the bilayer midplane cannot "
          "be computed from leaflet positions and will fall back to Z = 0 "
          "of the raw input coordinates.")


def get_bilayer_midplane_z():
    """Instantaneous bilayer midplane Z for the CURRENT frame: the midpoint
    between the two leaflets' centers of mass. Falls back to the whole-
    membrane COM, or to 0.0 if no membrane is available at all. Returns
    (midplane_z, leaflet_upper_z, leaflet_lower_z); the latter two are
    np.nan when leaflets could not be split."""
    if leaflet_upper_ag is not None and leaflet_lower_ag is not None:
        z_up = leaflet_upper_ag.center_of_mass()[2]
        z_lo = leaflet_lower_ag.center_of_mass()[2]
        return 0.5 * (z_up + z_lo), z_up, z_lo
    elif has_membrane:
        return membrane.center_of_mass()[2], np.nan, np.nan
    else:
        return 0.0, np.nan, np.nan


def get_membrane_footprint_xy(padding=0.0):
    """(xmin, xmax, ymin, ymax) bounding box of the membrane atoms for the
    CURRENT frame, used to test whether a water/ion sits laterally inside
    the membrane patch — independent of whether a peptide/protein is
    present. Returns None if there is no membrane at all."""
    if not has_membrane:
        return None
    mem_xy = membrane.positions[:, :2]
    return (mem_xy[:, 0].min() - padding, mem_xy[:, 0].max() + padding,
            mem_xy[:, 1].min() - padding, mem_xy[:, 1].max() + padding)


# ============================================================================
# === HELPER FUNCTIONS ===
# ============================================================================

def detect_pores_density_based(water_pos, center_xy, center_z, shell_inner, shell_outer,
                               z_bins, water_density_threshold):
    """Detect pores using water density analysis."""
    dist_xy = np.linalg.norm(water_pos[:, :2] - center_xy, axis=1)
    dist_z = water_pos[:, 2] - center_z
    
    in_shell = (dist_xy >= shell_inner) & (dist_xy <= shell_outer)
    
    pore_regions = []
    water_hist, _ = np.histogram(dist_z[in_shell], bins=z_bins)
    bin_areas = np.pi * (shell_outer**2 - shell_inner**2)
    water_density = water_hist / bin_areas
    
    pore_mask = water_density < water_density_threshold
    
    for i, is_pore in enumerate(pore_mask):
        if is_pore:
            z_min = z_bins[i]
            z_max = z_bins[i + 1]
            pore_regions.append({
                'z_min': z_min,
                'z_max': z_max,
                'z_center': (z_min + z_max) / 2,
                'water_count': int(water_hist[i])
            })
    
    return pore_regions, water_density, water_hist


def detect_pores_geometric(water_pos, center_xy, center_z, shell_inner, shell_outer,
                           dbscan_eps, dbscan_min_samples):
    """Detect pores using geometric clustering of water molecules."""
    dist_xy = np.linalg.norm(water_pos[:, :2] - center_xy, axis=1)
    in_shell = (dist_xy >= shell_inner) & (dist_xy <= shell_outer)
    
    if np.sum(in_shell) < dbscan_min_samples:
        return [], np.zeros((0, 3))
    
    water_shell = water_pos[in_shell]
    
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(water_shell)
    labels = clustering.labels_
    
    pore_regions = []
    if len(np.unique(labels)) > 1:
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster = water_shell[labels == label]
            z_center = np.mean(cluster[:, 2])
            radius = np.std(np.linalg.norm(cluster[:, :2] - center_xy, axis=1))
            pore_regions.append({
                'z_center': z_center,
                'radius': radius,
                'water_count': len(cluster),
                'cluster_coords': cluster
            })
    
    return pore_regions, water_shell


def calculate_pore_radius_from_atoms(atom_pos, center_xy, center_z, z_center, z_tolerance=2.0):
    """Calculate pore radius from atomic positions."""
    z_rel = atom_pos[:, 2] - center_z
    z_mask = np.abs(z_rel - z_center) < z_tolerance
    
    if np.sum(z_mask) == 0:
        return 0.0
    
    atoms_at_z = atom_pos[z_mask]
    dist_xy = np.linalg.norm(atoms_at_z[:, :2] - center_xy, axis=1)
    
    if len(dist_xy) == 0:
        return 0.0
    
    pore_radius = np.mean(dist_xy)
    return pore_radius


def calculate_water_permeation(water_pos, center_xy, center_z, pore_regions,
                               pore_radius_threshold, z_tolerance=3.0):
    """Calculate water permeation through detected pores."""
    permeation_data = {
        'frame': 0,
        'total_pore_waters': 0,
        'pore_count': 0,
        'pore_radii': [],
        'pore_water_counts': []
    }
    
    if len(pore_regions) == 0:
        return permeation_data
    
    total_waters = 0
    valid_pores = 0
    radii = []
    water_counts = []
    
    for pore in pore_regions:
        if 'radius' in pore and pore['radius'] >= pore_radius_threshold:
            total_waters += pore['water_count']
            valid_pores += 1
            radii.append(pore['radius'])
            water_counts.append(pore['water_count'])
        elif 'z_center' in pore:
            total_waters += pore['water_count']
            valid_pores += 1
    
    permeation_data['total_pore_waters'] = total_waters
    permeation_data['pore_count'] = valid_pores
    permeation_data['pore_radii'] = radii
    permeation_data['pore_water_counts'] = water_counts
    
    return permeation_data


# ============================================================================
# === MAIN ANALYSIS LOOP ===
# ============================================================================

num_waters_per_bin = []
radius_per_bin = []
pore_radii_evolution = []
pore_count_evolution = []
water_permeation_evolution = []
frame_indices = []
time_values_ps = []
pore_regions_by_frame = []
membrane_radius_evolution = []
water_histogram_all = []

# --- Membrane-normal permeation-depth traces (water/ion) ---
# For each frame we look at every water/ion atom within a radial cylinder of
# 'permeation_channel_radius' around the solute's XY center (the permeation
# channel), and record the signed Z-position (relative to the solute/membrane
# center) of whichever atom sits CLOSEST to the center plane -- i.e. the
# deepest-penetrating water/ion along the membrane normal at that instant.
# This is the same "depth of insertion"-style quantity, applied to permeants
# instead of the solute itself.
water_permeation_depth_evolution = []   # deepest water Z (signed), per frame
water_channel_mean_z_evolution = []     # mean Z of all channel waters, per frame
water_channel_count_evolution = []      # number of waters in the channel, per frame

ion_permeation_depth_evolution = []     # deepest ion Z (signed), per frame
ion_channel_mean_z_evolution = []       # mean Z of all channel ions, per frame
ion_channel_count_evolution = []        # number of ions in the channel, per frame

# --- Bilayer-midplane occupancy (water/ion count within ±midplane_half_thickness
# of Z = 0, inside the channel) -- a standard hydration-defect / pore-precursor
# indicator: water crossing the hydrophobic core center. ---
water_midplane_count_evolution = []
ion_midplane_count_evolution = []

# --- Leaflet mean Z positions each frame, RECENTERED so the bilayer
# midplane is Z = 0 (i.e. leaflet_upper_z_evolution ~ +half membrane
# thickness, leaflet_lower_z_evolution ~ -half membrane thickness). Used to
# draw upper/lower-leaflet reference bands on the plots. ---
leaflet_upper_z_evolution = []
leaflet_lower_z_evolution = []

print("[INFO] Beginning frame-by-frame analysis...")
print(f"[INFO] Total frames: {len(u.trajectory)}")

for frame_idx, ts in enumerate(u.trajectory):
    if (frame_idx + 1) % max(1, len(u.trajectory) // 10) == 0:
        print(f"[PROGRESS] Frame {frame_idx + 1}/{len(u.trajectory)}")
    
    frame_indices.append(frame_idx)
    time_values_ps.append(ts.time)  # actual simulation time (ps) from the input trajectory

    # --- Bilayer midplane (Z = 0 by construction) and leaflet positions ---
    midplane_z, leaflet_up_z, leaflet_lo_z = get_bilayer_midplane_z()
    leaflet_upper_z_evolution.append(
        float(leaflet_up_z - midplane_z) if np.isfinite(leaflet_up_z) else np.nan)
    leaflet_lower_z_evolution.append(
        float(leaflet_lo_z - midplane_z) if np.isfinite(leaflet_lo_z) else np.nan)
    center_z = midplane_z  # every Z-based quantity below is relative to the true midplane

    if has_protein:
        solute_pos = solute.positions
        center_xy = solute_pos[:, :2].mean(axis=0)
    elif has_membrane:
        solute_pos = None
        center_xy = membrane.center_of_mass()[:2]
    else:
        solute_pos = None
        center_xy = np.array([0.0, 0.0])

    water_pos = waters.positions
    dist_xy = np.linalg.norm(water_pos[:, :2] - center_xy, axis=1)
    dist_z = water_pos[:, 2] - center_z
    
    in_shell = (dist_xy >= shell_inner) & (dist_xy <= shell_outer)
    filtered_z = dist_z[in_shell]
    
    water_hist, _ = np.histogram(filtered_z, bins=z_bins)
    water_histogram_all.append(water_hist)
    num_waters_per_bin.append(water_hist)

    # --- Water permeation depth along the membrane normal (this frame) ---
    channel_mask_water = dist_xy <= permeation_channel_radius
    channel_z_water = dist_z[channel_mask_water]
    water_channel_count_evolution.append(int(channel_mask_water.sum()))
    if channel_z_water.size > 0:
        deepest_idx = np.argmin(np.abs(channel_z_water))
        water_permeation_depth_evolution.append(float(channel_z_water[deepest_idx]))
        water_channel_mean_z_evolution.append(float(np.mean(channel_z_water)))
    else:
        water_permeation_depth_evolution.append(np.nan)
        water_channel_mean_z_evolution.append(np.nan)

    # Water molecules found in the bilayer midplane: within
    # ±midplane_half_thickness of the TRUE midplane (Z = 0 by construction)
    # AND laterally inside the membrane's own XY footprint. This is
    # deliberately independent of the solute channel so it stays meaningful
    # even with no peptide/protein present (e.g. a pure-lipid control).
    footprint = get_membrane_footprint_xy(padding=membrane_footprint_padding)
    if footprint is not None:
        xmin, xmax, ymin, ymax = footprint
        in_footprint_water = (
            (water_pos[:, 0] >= xmin) & (water_pos[:, 0] <= xmax) &
            (water_pos[:, 1] >= ymin) & (water_pos[:, 1] <= ymax)
        )
    else:
        in_footprint_water = np.ones(water_pos.shape[0], dtype=bool)
    water_at_midplane_z = np.abs(dist_z) <= midplane_half_thickness
    water_midplane_count_evolution.append(
        int(np.sum(in_footprint_water & water_at_midplane_z))
    )

    # --- Ion permeation depth along the membrane normal (this frame) ---
    if has_ions:
        ion_pos = ions.positions
        ion_dist_xy = np.linalg.norm(ion_pos[:, :2] - center_xy, axis=1)
        ion_dist_z = ion_pos[:, 2] - center_z
        channel_mask_ion = ion_dist_xy <= permeation_channel_radius
        channel_z_ion = ion_dist_z[channel_mask_ion]
        ion_channel_count_evolution.append(int(channel_mask_ion.sum()))
        if channel_z_ion.size > 0:
            deepest_ion_idx = np.argmin(np.abs(channel_z_ion))
            ion_permeation_depth_evolution.append(float(channel_z_ion[deepest_ion_idx]))
            ion_channel_mean_z_evolution.append(float(np.mean(channel_z_ion)))
        else:
            ion_permeation_depth_evolution.append(np.nan)
            ion_channel_mean_z_evolution.append(np.nan)
        if footprint is not None:
            in_footprint_ion = (
                (ion_pos[:, 0] >= xmin) & (ion_pos[:, 0] <= xmax) &
                (ion_pos[:, 1] >= ymin) & (ion_pos[:, 1] <= ymax)
            )
        else:
            in_footprint_ion = np.ones(ion_pos.shape[0], dtype=bool)
        ion_at_midplane_z = np.abs(ion_dist_z) <= midplane_half_thickness
        ion_midplane_count_evolution.append(
            int(np.sum(in_footprint_ion & ion_at_midplane_z))
        )
    else:
        ion_channel_count_evolution.append(0)
        ion_permeation_depth_evolution.append(np.nan)
        ion_channel_mean_z_evolution.append(np.nan)
        ion_midplane_count_evolution.append(0)
    
    if has_protein:
        # --- Peptide/protein-induced pore detection (only meaningful, and
        # only run, when a peptide/protein is actually present) ---
        xy_dist = np.linalg.norm(solute_pos[:, :2] - center_xy, axis=1)
        solute_z_rel = solute_pos[:, 2] - center_z

        weighted_radius, _ = np.histogram(solute_z_rel, bins=z_bins, weights=xy_dist)
        bin_counts, _ = np.histogram(solute_z_rel, bins=z_bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_radius = np.nan_to_num(weighted_radius / bin_counts)
        radius_per_bin.append(avg_radius)

        if pore_detection_method == "density_based":
            pore_regions, water_density, water_hist_pore = detect_pores_density_based(
                water_pos, center_xy, center_z, shell_inner, shell_outer,
                z_bins, water_density_threshold
            )
        else:
            pore_regions, water_shell = detect_pores_geometric(
                water_pos, center_xy, center_z, shell_inner, shell_outer,
                dbscan_eps, dbscan_min_samples
            )

        for pore in pore_regions:
            if 'z_center' in pore:
                radius = calculate_pore_radius_from_atoms(
                    solute_pos, center_xy, center_z, pore['z_center'], z_tolerance=2.0
                )
                pore['solute_radius'] = radius
            else:
                pore['solute_radius'] = 0.0

        pore_regions_by_frame.append(pore_regions)

        if len(pore_regions) > 0:
            radii = [p.get('solute_radius', p.get('radius', 0)) for p in pore_regions]
            pore_radii_evolution.append(np.mean(radii) if radii else 0.0)
            pore_count_evolution.append(len(pore_regions))
        else:
            pore_radii_evolution.append(0.0)
            pore_count_evolution.append(0)

        permeation_data = calculate_water_permeation(
            water_pos, center_xy, center_z, pore_regions, pore_radius_threshold
        )
        permeation_data['frame'] = frame_idx
        water_permeation_evolution.append(permeation_data)
    else:
        # No peptide/protein present -> there is no peptide/protein-induced
        # pore to detect. Record explicit zeros/empties instead of running
        # pore detection against an empty solute selection.
        radius_per_bin.append(np.zeros(len(z_bins) - 1))
        pore_regions_by_frame.append([])
        pore_radii_evolution.append(0.0)
        pore_count_evolution.append(0)
        water_permeation_evolution.append({
            'frame': frame_idx, 'total_pore_waters': 0, 'pore_count': 0,
            'pore_radii': [], 'pore_water_counts': []
        })

    if has_membrane:
        membrane_pos = membrane.positions
        membrane_z = membrane_pos[:, 2].mean()
        membrane_xy_dist = np.linalg.norm(membrane_pos[:, :2] - center_xy, axis=1)
        membrane_radius_evolution.append(np.mean(membrane_xy_dist))

# ============================================================================
# === CONVERT TO ARRAYS AND STATISTICS ===
# ============================================================================

num_waters_per_bin = np.array(num_waters_per_bin)
radius_per_bin = np.array(radius_per_bin)
pore_radii_evolution = np.array(pore_radii_evolution)
pore_count_evolution = np.array(pore_count_evolution)
frame_indices = np.array(frame_indices)

# --- Leaflet Z positions (recentered so the bilayer midplane is Z = 0) ---
leaflet_upper_z_evolution = np.array(leaflet_upper_z_evolution, dtype=float)
leaflet_lower_z_evolution = np.array(leaflet_lower_z_evolution, dtype=float)
mean_leaflet_upper_z = (float(np.nanmean(leaflet_upper_z_evolution))
                         if np.any(np.isfinite(leaflet_upper_z_evolution)) else np.nan)
mean_leaflet_lower_z = (float(np.nanmean(leaflet_lower_z_evolution))
                         if np.any(np.isfinite(leaflet_lower_z_evolution)) else np.nan)
has_leaflets = np.isfinite(mean_leaflet_upper_z) and np.isfinite(mean_leaflet_lower_z)
if has_leaflets:
    print(f"[INFO] Mean leaflet positions (relative to midplane): "
          f"upper = {mean_leaflet_upper_z:+.2f} Å, lower = {mean_leaflet_lower_z:+.2f} Å "
          f"(bilayer thickness ≈ {mean_leaflet_upper_z - mean_leaflet_lower_z:.2f} Å)")


def add_leaflet_reference(ax, orientation='vertical', legend=True):
    """Draw the bilayer midplane (Z=0) and, if available, shade the region
    between the mean upper/lower leaflet positions on a membrane-normal
    axis. orientation='vertical' -> Z is the x-axis (e.g. Fig 1);
    orientation='horizontal' -> Z is the y-axis (e.g. Figs 13/14)."""
    span = ax.axhspan if orientation == 'horizontal' else ax.axvspan
    line = ax.axhline if orientation == 'horizontal' else ax.axvline
    if has_leaflets:
        span(mean_leaflet_lower_z, mean_leaflet_upper_z, color='#f0c674', alpha=0.22,
             zorder=0, label='Bilayer core (mean leaflet-to-leaflet)')
        line(mean_leaflet_upper_z, color='#8c6d31', linestyle=':', linewidth=1.3,
             alpha=0.8, zorder=1, label='Upper leaflet (mean)')
        line(mean_leaflet_lower_z, color='#4a3b17', linestyle=':', linewidth=1.3,
             alpha=0.8, zorder=1, label='Lower leaflet (mean)')
    line(0, color='black', linestyle='-', linewidth=1.1, alpha=0.55, zorder=1,
         label='Bilayer midplane (Z = 0)')


mean_waters = num_waters_per_bin.mean(axis=0)
std_waters = num_waters_per_bin.std(axis=0)

mean_radius = radius_per_bin.mean(axis=0)
std_radius = radius_per_bin.std(axis=0)

if apply_kde_smoothing:
    mean_waters_smooth = gaussian_filter1d(mean_waters, sigma=kde_sigma)
    mean_radius_smooth = gaussian_filter1d(mean_radius, sigma=kde_sigma)
    pore_radii_smooth = gaussian_filter1d(pore_radii_evolution, sigma=kde_sigma)
    pore_count_smooth = gaussian_filter1d(pore_count_evolution.astype(float), sigma=kde_sigma)
else:
    mean_waters_smooth = mean_waters
    mean_radius_smooth = mean_radius
    pore_radii_smooth = pore_radii_evolution
    pore_count_smooth = pore_count_evolution.astype(float)

time_values_ps = np.array(time_values_ps)

# --- Use the ACTUAL time reported by the input trajectory (MDAnalysis reports
# ts.time in picoseconds, taken directly from the topology/trajectory files'
# own metadata), rather than assuming a fixed frame interval. ---
time_axis = time_values_ps / 1000.0  # ps -> ns, per-frame, exactly as read from the file

if len(time_axis) > 1:
    frame_dt_ns = np.diff(time_axis)
    dt_ns = float(np.median(frame_dt_ns))
    dt_is_uniform = bool(np.allclose(frame_dt_ns, dt_ns, rtol=1e-3, atol=1e-9))
else:
    dt_ns = 0.0
    dt_is_uniform = True

if dt_ns <= 0:
    # Some readers/formats don't populate ts.time (it stays at 0). Fall back to
    # the trajectory's own declared timestep (u.trajectory.dt, in ps) so
    # durations are still computed from file metadata rather than a guess.
    fallback_dt_ps = float(getattr(u.trajectory, 'dt', 1.0) or 1.0)
    dt_ns = fallback_dt_ps / 1000.0
    print(f"[WARNING] Trajectory did not report increasing frame times; "
          f"falling back to trajectory dt = {fallback_dt_ps:.4f} ps/frame "
          f"({dt_ns:.6f} ns/frame) from file metadata.")
    time_axis = frame_indices * dt_ns

time_factor = dt_ns  # ns per frame, derived from the real trajectory (kept as
                      # 'time_factor' so downstream duration math is unchanged)

print(f"[INFO] Actual trajectory time step: {time_factor * 1000:.4f} ps/frame "
      f"({'uniform' if dt_is_uniform else 'NON-UNIFORM - using median spacing'})")
print(f"[INFO] Total simulated time: {time_axis[-1] - time_axis[0]:.4f} ns "
      f"over {len(frame_indices)} frames (start = {time_axis[0]:.4f} ns)")

# --- Water/ion membrane-normal permeation-depth arrays ---
water_permeation_depth_evolution = np.array(water_permeation_depth_evolution)
water_channel_mean_z_evolution = np.array(water_channel_mean_z_evolution)
water_channel_count_evolution = np.array(water_channel_count_evolution)

ion_permeation_depth_evolution = np.array(ion_permeation_depth_evolution)
ion_channel_mean_z_evolution = np.array(ion_channel_mean_z_evolution)
ion_channel_count_evolution = np.array(ion_channel_count_evolution)


def _fill_and_smooth(arr, sigma, apply_smooth):
    """Fill any NaN gaps (frames with zero channel occupancy) by linear
    interpolation over neighboring frames, then optionally apply Gaussian
    smoothing. Returns both the gap-filled raw trace and its smoothed version."""
    arr = np.asarray(arr, dtype=float)
    filled = arr.copy()
    valid = ~np.isnan(filled)
    if valid.sum() >= 2:
        filled[~valid] = np.interp(
            np.flatnonzero(~valid), np.flatnonzero(valid), filled[valid]
        )
    elif valid.sum() == 1:
        filled[~valid] = filled[valid][0]
    else:
        filled[:] = 0.0
    smooth = gaussian_filter1d(filled, sigma=sigma) if apply_smooth else filled
    return filled, smooth


water_permeation_depth_filled, water_permeation_depth_smooth = _fill_and_smooth(
    water_permeation_depth_evolution, kde_sigma, apply_kde_smoothing)
ion_permeation_depth_filled, ion_permeation_depth_smooth = _fill_and_smooth(
    ion_permeation_depth_evolution, kde_sigma, apply_kde_smoothing)

# --- Bilayer-midplane water/ion occupancy arrays ---
water_midplane_count_evolution = np.array(water_midplane_count_evolution, dtype=float)
water_midplane_smooth = (gaussian_filter1d(water_midplane_count_evolution, sigma=kde_sigma)
                          if apply_kde_smoothing else water_midplane_count_evolution)
water_midplane_occupied = water_midplane_count_evolution > 0
water_midplane_occupancy_fraction = float(water_midplane_occupied.mean())

ion_midplane_count_evolution = np.array(ion_midplane_count_evolution, dtype=float)
if has_ions:
    ion_midplane_smooth = (gaussian_filter1d(ion_midplane_count_evolution, sigma=kde_sigma)
                            if apply_kde_smoothing else ion_midplane_count_evolution)
    ion_midplane_occupied = ion_midplane_count_evolution > 0
    ion_midplane_occupancy_fraction = float(ion_midplane_occupied.mean())
else:
    ion_midplane_smooth = ion_midplane_count_evolution
    ion_midplane_occupancy_fraction = 0.0

# ============================================================================
# === PORE STABILITY, LIFETIME, AND TEMPORAL EVOLUTION ANALYSIS ===
# ============================================================================
print("[INFO] Analyzing pore stability and lifetime...")

n_frames = len(frame_indices)
n_zbins = len(z_centers)


def find_true_runs(bool_series):
    """Return list of (start_idx, end_idx_inclusive) for contiguous True runs
    in a 1D boolean array/list."""
    events = []
    in_run = False
    start = None
    for i, val in enumerate(bool_series):
        if val and not in_run:
            in_run, start = True, i
        elif not val and in_run:
            in_run = False
            events.append((start, i - 1))
    if in_run:
        events.append((start, len(bool_series) - 1))
    return events


def autocorrelation(x, max_lag):
    """Normalized autocorrelation function of a 1D time series."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0 or n < 2:
        return np.zeros(max_lag)
    acf = np.empty(max_lag)
    for lag in range(max_lag):
        acf[lag] = np.dot(x[:n - lag], x[lag:]) / (n - lag) / var
    return acf


def relaxation_time(acf, lags_ns, threshold=1.0 / np.e):
    """First lag (ns) at which the ACF decays below 1/e (a standard
    persistence/relaxation-time estimate)."""
    below = np.where(acf < threshold)[0]
    if len(below) > 0:
        return float(lags_ns[below[0]])
    return float(lags_ns[-1]) if len(lags_ns) else np.nan


# --- Build per-(frame, Z-bin) pore occupancy / radius / water-count matrices ---
# Pores are tracked at fixed Z-bin locations across the trajectory. This treats
# a "pore" as a channel opening at a given membrane depth, which is the
# standard way to define pore lifetime/stability in MD pore analyses.
pore_open_matrix = np.zeros((n_frames, n_zbins), dtype=bool)
pore_radius_matrix = np.zeros((n_frames, n_zbins))
pore_water_matrix = np.zeros((n_frames, n_zbins))

for f_idx, frame_pores in enumerate(pore_regions_by_frame):
    for pore in frame_pores:
        z_c = pore.get('z_center', None)
        if z_c is None:
            continue
        bin_idx = int(np.clip(np.searchsorted(z_bins, z_c) - 1, 0, n_zbins - 1))
        pore_open_matrix[f_idx, bin_idx] = True
        r = pore.get('solute_radius', pore.get('radius', 0.0)) or 0.0
        pore_radius_matrix[f_idx, bin_idx] = max(pore_radius_matrix[f_idx, bin_idx], r)
        pore_water_matrix[f_idx, bin_idx] = max(pore_water_matrix[f_idx, bin_idx],
                                                 pore.get('water_count', 0))

# --- Individual pore-opening events (lifetime, per spatial location) ---
pore_lifetime_events = []
for bin_idx in range(n_zbins):
    for (s, e) in find_true_runs(pore_open_matrix[:, bin_idx]):
        duration_frames = e - s + 1
        radii_seg = pore_radius_matrix[s:e + 1, bin_idx]
        water_seg = pore_water_matrix[s:e + 1, bin_idx]
        radii_seg_nz = radii_seg[radii_seg > 0]
        pore_lifetime_events.append({
            'Pore ID': len(pore_lifetime_events) + 1,
            'Z-bin center (Å)': z_centers[bin_idx],
            'Start Frame': int(frame_indices[s]),
            'End Frame': int(frame_indices[e]),
            'Start Time (ns)': time_axis[s],
            'End Time (ns)': time_axis[e],
            'Lifetime (ns)': duration_frames * time_factor,
            'Lifetime (frames)': duration_frames,
            'Mean Radius (Å)': float(np.mean(radii_seg_nz)) if len(radii_seg_nz) else 0.0,
            'Max Radius (Å)': float(np.max(radii_seg_nz)) if len(radii_seg_nz) else 0.0,
            'Radius CV': float(np.std(radii_seg_nz) / np.mean(radii_seg_nz))
                         if len(radii_seg_nz) > 1 and np.mean(radii_seg_nz) > 0 else 0.0,
            'Mean Water Count': float(np.mean(water_seg))
        })

df_pore_lifetimes = pd.DataFrame(pore_lifetime_events)
if len(df_pore_lifetimes) > 0:
    df_pore_lifetimes = df_pore_lifetimes.sort_values('Start Frame').reset_index(drop=True)

# --- Stability by spatial location (occupancy, opening frequency, lifetimes) ---
stability_by_location = []
for bin_idx in range(n_zbins):
    occ = pore_open_matrix[:, bin_idx]
    events = find_true_runs(occ)
    lifetimes_ns = [(e - s + 1) * time_factor for s, e in events]
    stability_by_location.append({
        'Z-bin center (Å)': z_centers[bin_idx],
        'Occupancy Fraction': float(occ.mean()),
        'Num Opening Events': len(events),
        'Mean Lifetime (ns)': float(np.mean(lifetimes_ns)) if lifetimes_ns else 0.0,
        'Max Lifetime (ns)': float(np.max(lifetimes_ns)) if lifetimes_ns else 0.0,
    })
df_pore_stability_location = pd.DataFrame(stability_by_location)

# --- Global (whole-simulation) open/closed state, lifetime & waiting-time stats ---
overall_open = pore_count_evolution > 0
overall_open_events = find_true_runs(overall_open)
overall_closed_events = find_true_runs(~overall_open)

overall_lifetimes_ns = np.array([(e - s + 1) * time_factor for s, e in overall_open_events])
overall_closed_durations_ns = np.array([(e - s + 1) * time_factor for s, e in overall_closed_events])

open_radii = pore_radii_evolution[overall_open]
radius_cv_global = (float(np.std(open_radii) / np.mean(open_radii))
                     if len(open_radii) > 1 and np.mean(open_radii) > 0 else np.nan)
stability_index_global = (1.0 / radius_cv_global
                           if radius_cv_global and radius_cv_global > 0 else np.nan)

# --- Autocorrelation-based persistence/relaxation time (temporal evolution) ---
# max_lag is kept strictly below n_frames to avoid a zero-length overlap window
max_lag = min(200, max(1, n_frames - 1))
acf_pore_count = autocorrelation(pore_count_evolution, max_lag)
acf_pore_radius = autocorrelation(pore_radii_evolution, max_lag)
acf_lags_ns = np.arange(max_lag) * time_factor

tau_pore_count_ns = relaxation_time(acf_pore_count, acf_lags_ns)
tau_pore_radius_ns = relaxation_time(acf_pore_radius, acf_lags_ns)

# --- Rolling (windowed) trends: local temporal evolution of stability ---
rolling_window = max(3, int(n_frames * 0.05))
pore_radius_rolling_mean = pd.Series(pore_radii_evolution).rolling(
    rolling_window, center=True, min_periods=1).mean().values
pore_radius_rolling_std = pd.Series(pore_radii_evolution).rolling(
    rolling_window, center=True, min_periods=1).std().fillna(0).values
pore_count_rolling_mean = pd.Series(pore_count_evolution.astype(float)).rolling(
    rolling_window, center=True, min_periods=1).mean().values

df_temporal_evolution = pd.DataFrame({
    "Frame": frame_indices,
    "Time (ps)": time_values_ps,
    "Time (ns)": time_axis,
    "Pore Open (any)": overall_open.astype(int),
    "Pore Radius Rolling Mean (Å)": pore_radius_rolling_mean,
    "Pore Radius Rolling Std (Å)": pore_radius_rolling_std,
    "Pore Count Rolling Mean": pore_count_rolling_mean,
})

print(f"[INFO] Detected {len(pore_lifetime_events)} pore-opening events "
      f"across {n_zbins} spatial bins")
print(f"[INFO] Mean pore lifetime: "
      f"{np.mean(overall_lifetimes_ns) if len(overall_lifetimes_ns) else 0:.3f} ns | "
      f"Fraction of time open: {100 * overall_open.mean():.1f}%")

# ============================================================================
# === SAVE CSV FILES ===
# ============================================================================

print("[INFO] Saving data to CSV files...")

df_waters = pd.DataFrame({
    "Z (Å)": z_centers,
    "Mean Number of Waters": mean_waters,
    "Smoothed": mean_waters_smooth,
    "Std": std_waters
})
df_waters.to_csv(f"{output_prefix}_waters.csv", index=False)

df_radius = pd.DataFrame({
    "Z (Å)": z_centers,
    "Mean Radius (Å)": mean_radius,
    "Smoothed": mean_radius_smooth,
    "Std": std_radius
})
df_radius.to_csv(f"{output_prefix}_radius.csv", index=False)

df_pore_evolution = pd.DataFrame({
    "Frame": frame_indices,
    "Time (ps)": time_values_ps,
    "Time (ns)": time_axis,
    "Pore Count": pore_count_evolution,
    "Pore Radius (Å)": pore_radii_evolution,
    "Pore Radius Smoothed (Å)": pore_radii_smooth,
    "Pore Count Smoothed": pore_count_smooth
})
df_pore_evolution.to_csv(f"{output_prefix}_pore_evolution.csv", index=False)

permeation_frames = []
for data in water_permeation_evolution:
    permeation_frames.append({
        "Frame": data['frame'],
        "Time (ns)": time_axis[data['frame']],
        "Total Pore Waters": data['total_pore_waters'],
        "Pore Count": data['pore_count']
    })
df_permeation = pd.DataFrame(permeation_frames)
df_permeation.to_csv(f"{output_prefix}_water_permeation.csv", index=False)

# --- Combined water permeation + pore radius, aligned on time (single table) ---
pore_waters_arr = df_permeation['Total Pore Waters'].values.astype(float)
pore_waters_arr_smooth = gaussian_filter1d(pore_waters_arr, sigma=kde_sigma) if apply_kde_smoothing else pore_waters_arr
df_permeation_radius_combined = pd.DataFrame({
    "Frame": frame_indices,
    "Time (ns)": time_axis,
    "Water Permeation (count)": pore_waters_arr,
    "Water Permeation Smoothed": pore_waters_arr_smooth,
    "Pore Radius (Å)": pore_radii_evolution,
    "Pore Radius Smoothed (Å)": pore_radii_smooth,
    "Pore Count": pore_count_evolution
})
df_permeation_radius_combined.to_csv(f"{output_prefix}_permeation_radius_combined.csv", index=False)

# --- Water/ion membrane-normal permeation-depth traces ---
df_water_permeation_depth = pd.DataFrame({
    "Frame": frame_indices,
    "Time (ns)": time_axis,
    "Deepest Water Z (Å, membrane normal)": water_permeation_depth_filled,
    "Deepest Water Z Smoothed (Å)": water_permeation_depth_smooth,
    "Channel Mean Water Z (Å)": water_channel_mean_z_evolution,
    "Waters in Channel (count)": water_channel_count_evolution,
})
df_water_permeation_depth.to_csv(f"{output_prefix}_water_permeation_membrane_normal.csv", index=False)

if has_ions:
    df_ion_permeation_depth = pd.DataFrame({
        "Frame": frame_indices,
        "Time (ns)": time_axis,
        "Deepest Ion Z (Å, membrane normal)": ion_permeation_depth_filled,
        "Deepest Ion Z Smoothed (Å)": ion_permeation_depth_smooth,
        "Channel Mean Ion Z (Å)": ion_channel_mean_z_evolution,
        "Ions in Channel (count)": ion_channel_count_evolution,
    })
    df_ion_permeation_depth.to_csv(f"{output_prefix}_ion_permeation_membrane_normal.csv", index=False)

# --- Bilayer-midplane occupancy (water always; ion if available) ---
df_water_midplane = pd.DataFrame({
    "Frame": frame_indices,
    "Time (ns)": time_axis,
    "Water Count at Midplane": water_midplane_count_evolution,
    "Water Count at Midplane (Smoothed)": water_midplane_smooth,
    "Midplane Occupied": water_midplane_occupied,
})
df_water_midplane.to_csv(f"{output_prefix}_water_midplane_occupancy.csv", index=False)

if has_ions:
    df_ion_midplane = pd.DataFrame({
        "Frame": frame_indices,
        "Time (ns)": time_axis,
        "Ion Count at Midplane": ion_midplane_count_evolution,
        "Ion Count at Midplane (Smoothed)": ion_midplane_smooth,
        "Midplane Occupied": ion_midplane_occupied,
    })
    df_ion_midplane.to_csv(f"{output_prefix}_ion_midplane_occupancy.csv", index=False)

summary_stats = {
    "Metric": [
        "Mean Pore Radius (Å)",
        "Std Pore Radius (Å)",
        "Max Pore Radius (Å)",
        "Mean Pore Count",
        "Max Pore Count",
        "Mean Waters in Pores",
        "Max Waters in Pores",
        "Mean Solute Radius (Å)",
        "Total Frames Analyzed",
        "Trajectory Time Step (ps/frame)",
        "Trajectory Time Step Uniform",
        "Simulation Start Time (ns)",
        "Simulation End Time (ns)",
        "Total Simulated Time (ns)"
    ],
    "Value": [
        f"{np.mean(pore_radii_evolution):.3f}",
        f"{np.std(pore_radii_evolution):.3f}",
        f"{np.max(pore_radii_evolution):.3f}",
        f"{np.mean(pore_count_evolution):.2f}",
        f"{np.max(pore_count_evolution)}",
        f"{np.mean(df_permeation['Total Pore Waters']):.2f}",
        f"{np.max(df_permeation['Total Pore Waters']):.0f}",
        f"{np.mean(mean_radius):.3f}",
        f"{len(frame_indices)}",
        f"{time_factor * 1000:.4f}",
        f"{dt_is_uniform}",
        f"{time_axis[0]:.4f}",
        f"{time_axis[-1]:.4f}",
        f"{time_axis[-1] - time_axis[0]:.4f}"
    ]
}
df_summary = pd.DataFrame(summary_stats)
df_summary.to_csv(f"{output_prefix}_summary.csv", index=False)

# --- Pore lifetime / stability / temporal evolution CSVs ---
df_pore_lifetimes.to_csv(f"{output_prefix}_pore_lifetimes.csv", index=False)
df_pore_stability_location.to_csv(f"{output_prefix}_pore_stability_by_location.csv", index=False)
df_temporal_evolution.to_csv(f"{output_prefix}_temporal_evolution.csv", index=False)

stability_summary = {
    "Metric": [
        "Total Pore-Opening Events",
        "Mean Pore Lifetime (ns)",
        "Std Pore Lifetime (ns)",
        "Median Pore Lifetime (ns)",
        "Max Pore Lifetime (ns)",
        "Min Pore Lifetime (ns)",
        "Mean Closed/Waiting Duration (ns)",
        "Fraction of Simulation Time Open (%)",
        "Global Open-State Radius CV",
        "Global Stability Index (1/CV)",
        "Pore Count ACF Relaxation Time τ (ns)",
        "Pore Radius ACF Relaxation Time τ (ns)",
        "Most Stable Z-location (Å)",
        "Most Frequently Opening Z-location (Å)"
    ],
    "Value": [
        f"{len(pore_lifetime_events)}",
        f"{np.mean(overall_lifetimes_ns):.3f}" if len(overall_lifetimes_ns) else "0.000",
        f"{np.std(overall_lifetimes_ns):.3f}" if len(overall_lifetimes_ns) else "0.000",
        f"{np.median(overall_lifetimes_ns):.3f}" if len(overall_lifetimes_ns) else "0.000",
        f"{np.max(overall_lifetimes_ns):.3f}" if len(overall_lifetimes_ns) else "0.000",
        f"{np.min(overall_lifetimes_ns):.3f}" if len(overall_lifetimes_ns) else "0.000",
        f"{np.mean(overall_closed_durations_ns):.3f}" if len(overall_closed_durations_ns) else "0.000",
        f"{100 * overall_open.mean():.2f}",
        f"{radius_cv_global:.3f}" if not np.isnan(radius_cv_global) else "N/A",
        f"{stability_index_global:.3f}" if not np.isnan(stability_index_global) else "N/A",
        f"{tau_pore_count_ns:.3f}",
        f"{tau_pore_radius_ns:.3f}",
        f"{df_pore_stability_location.loc[df_pore_stability_location['Occupancy Fraction'].idxmax(), 'Z-bin center (Å)']:.2f}"
            if df_pore_stability_location['Occupancy Fraction'].max() > 0 else "N/A",
        f"{df_pore_stability_location.loc[df_pore_stability_location['Num Opening Events'].idxmax(), 'Z-bin center (Å)']:.2f}"
            if df_pore_stability_location['Num Opening Events'].max() > 0 else "N/A",
    ]
}
df_stability_summary = pd.DataFrame(stability_summary)
df_stability_summary.to_csv(f"{output_prefix}_pore_stability_summary.csv", index=False)

print("[✓] CSV files saved")

# ============================================================================
# === FIGURES 13/14: WATER & ION PERMEATION DEPTH ALONG MEMBRANE NORMAL ===
# ("Depth of insertion"-style trace: raw trace + mean (dashed) + ±1σ (dotted)
#  reference lines + shaded region below the bilayer center)
#
# NOTE: These plots (13-16) are generated here, immediately after the CSVs,
# deliberately BEFORE the heavier/more fragile 3D and contour visualizations
# further down (which involve DBSCAN/griddata and can fail on edge-case
# data). Generating them first, each wrapped in its own try/except, means a
# problem later in the pipeline can never prevent these files from existing.
# ============================================================================

def plot_membrane_normal_depth(time_axis, trace, trace_smooth, ylabel, title,
                                save_path, line_color='black', ref_color=None):
    """Reproduce the 'depth of insertion' style plot: a raw black trace over
    time, a dashed horizontal mean line, dotted ±1σ reference lines, and the
    actual bilayer geometry -- the true midplane (Z = 0) plus a shaded band
    showing where the upper and lower leaflets sit (mean position over the
    trajectory) -- for visual reference."""
    ref_color = ref_color or color_palette['membrane']  # orange, matches reference figure

    trace = np.asarray(trace, dtype=float)
    if not np.any(np.isfinite(trace)):
        raise ValueError("Trace contains no finite values - nothing to plot.")
    mean_val = float(np.nanmean(trace))
    std_val = float(np.nanstd(trace))
    if not np.isfinite(std_val):
        std_val = 0.0

    fig, ax = plt.subplots(figsize=(10, 5.5))

    leaflet_span = (mean_leaflet_upper_z - mean_leaflet_lower_z) if has_leaflets else 20.0
    y_min = min(np.nanmin(trace), -abs(mean_val) - 3 * max(std_val, 1.0), -leaflet_span) - 5
    y_max = max(np.nanmax(trace), abs(mean_val) + 3 * max(std_val, 1.0), leaflet_span) + 5

    # Bilayer geometry: true midplane (Z=0) + shaded band between the mean
    # upper/lower leaflet positions (falls back to a generic below-zero
    # shade if leaflets could not be split)
    add_leaflet_reference(ax, orientation='horizontal')
    if not has_leaflets:
        ax.fill_between(time_axis, y_min, 0, color='#e6e6e6', alpha=0.6, zorder=0)

    # Raw trace
    ax.plot(time_axis, trace, color=line_color, linewidth=1.3, zorder=4)

    # Mean (dashed) and ±1σ (dotted) reference lines
    ax.axhline(mean_val, color=ref_color, linestyle='--', linewidth=2.2, zorder=5,
               label=f'Mean = {mean_val:.2f} Å')
    ax.axhline(mean_val + std_val, color=ref_color, linestyle=':', linewidth=1.6, zorder=5,
               label=f'±1σ = {std_val:.2f} Å')
    ax.axhline(mean_val - std_val, color=ref_color, linestyle=':', linewidth=1.6, zorder=5)

    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Simulation time [ns]", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9, ncol=2)
    ax.grid(False)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)


def plot_midplane_occupancy(time_axis, count_raw, count_smooth, occupancy_fraction,
                             species_label, plot_color, save_path):
    """Time series of species count at the bilayer midplane, styled to match
    the rest of the dashboard (lightgray raw dashed trace, colored smoothed
    fill, red dashed mean line), with the overall occupancy fraction called
    out in the title."""
    count_raw = np.asarray(count_raw, dtype=float)
    count_smooth = np.asarray(count_smooth, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(time_axis, count_raw, color='lightgray', linestyle='--', linewidth=1.2,
            alpha=0.8, label='Raw', zorder=2)
    ax.fill_between(time_axis, count_raw, count_smooth, color=plot_color, alpha=0.15, zorder=2)
    ax.plot(time_axis, count_smooth, color=plot_color, linewidth=2.2,
            label=f'Smoothed (σ={kde_sigma})', zorder=3)

    mean_count = float(np.mean(count_raw)) if count_raw.size else 0.0
    ax.axhline(mean_count, color='red', linestyle='--', linewidth=2.0, zorder=4,
               label=f'Mean = {mean_count:.2f}')

    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.set_ylim(bottom=0)
    top = max(1.0, np.nanmax(count_raw) if count_raw.size else 1.0)
    ax.set_ylim(0, top * 1.15)
    ax.set_xlabel("Simulation time [ns]", fontsize=12, fontweight='bold')
    ax.set_ylabel(f"{species_label} count at midplane", fontsize=12, fontweight='bold')
    ax.set_title(
        f"{species_label} Occupancy at the True Bilayer Midplane "
        f"(±{midplane_half_thickness:.1f} Å of Z = 0, within the membrane footprint)\n"
        f"Occupied {100 * occupancy_fraction:.1f}% of simulation time",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close(fig)


# --- Figure 13: Water permeation depth along the membrane normal ---
try:
    plot_membrane_normal_depth(
        time_axis,
        water_permeation_depth_filled,
        water_permeation_depth_smooth,
        ylabel="Depth along membrane normal, Z rel. midplane [Å]",
        title="Water Permeation Depth Along Membrane Normal",
        save_path=f"{output_prefix}_13_water_permeation_membrane_normal.png",
    )
    print(f"[✓] Saved {output_prefix}_13_water_permeation_membrane_normal.png")
except Exception as exc:
    print(f"[ERROR] Failed to generate Figure 13 (water permeation depth): {exc}")

# --- Figure 14: Ion permeation depth along the membrane normal (if ions present) ---
if has_ions:
    try:
        plot_membrane_normal_depth(
            time_axis,
            ion_permeation_depth_filled,
            ion_permeation_depth_smooth,
            ylabel="Depth along membrane normal, Z rel. midplane [Å]",
            title="Ion Permeation Depth Along Membrane Normal",
            save_path=f"{output_prefix}_14_ion_permeation_membrane_normal.png",
        )
        print(f"[✓] Saved {output_prefix}_14_ion_permeation_membrane_normal.png")
    except Exception as exc:
        print(f"[ERROR] Failed to generate Figure 14 (ion permeation depth): {exc}")
else:
    print("[INFO] Skipping ion permeation depth plot (no ions found for "
          f"selection '{ion_selection}').")

# ============================================================================
# === FIGURE 15/16: WATER & ION OCCUPANCY AT THE BILAYER MIDPLANE ===
# Number of water/ion molecules within ±midplane_half_thickness of Z = 0
# (the hydrophobic core center), inside the permeation channel, over time.
# This is a standard hydration-defect / pore-precursor indicator: it flags
# frames where water (or ions) briefly bridge or fully cross the membrane
# center, distinct from the broader-shell pore-water count in Figure 04.
# ============================================================================

# --- Figure 15: Water occupancy at the bilayer midplane ---
try:
    plot_midplane_occupancy(
        time_axis, water_midplane_count_evolution, water_midplane_smooth,
        water_midplane_occupancy_fraction,
        species_label="Water", plot_color=color_palette['water'],
        save_path=f"{output_prefix}_15_water_midplane_occupancy.png",
    )
    print(f"[✓] Saved {output_prefix}_15_water_midplane_occupancy.png")
except Exception as exc:
    print(f"[ERROR] Failed to generate Figure 15 (water midplane occupancy): {exc}")

# --- Figure 16: Ion occupancy at the bilayer midplane (if ions present) ---
if has_ions:
    try:
        plot_midplane_occupancy(
            time_axis, ion_midplane_count_evolution, ion_midplane_smooth,
            ion_midplane_occupancy_fraction,
            species_label="Ion", plot_color=color_palette['solute'],
            save_path=f"{output_prefix}_16_ion_midplane_occupancy.png",
        )
        print(f"[✓] Saved {output_prefix}_16_ion_midplane_occupancy.png")
    except Exception as exc:
        print(f"[ERROR] Failed to generate Figure 16 (ion midplane occupancy): {exc}")
else:
    print("[INFO] Skipping ion midplane occupancy plot (no ions found for "
          f"selection '{ion_selection}').")


# ============================================================================
# === PUBLICATION-QUALITY VISUALIZATIONS ===
# ============================================================================

print("[INFO] Generating publication-quality visualizations...")

# === Figure 1: Water Distribution with Publication Quality ===
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

add_leaflet_reference(ax, orientation='vertical')

ax.plot(z_centers, mean_waters, color='lightgray', linestyle='--', linewidth=1.5,
        alpha=0.7, label='Raw data', zorder=1)
ax.plot(z_centers, mean_waters_smooth, color=color_palette['water'], linewidth=3.0,
        label='Smoothed (σ=1.0)', zorder=3)

ax.fill_between(z_centers, mean_waters - std_waters, mean_waters + std_waters,
                color=color_palette['water'], alpha=0.15, label='±1σ uncertainty', zorder=2)

ax.set_ylabel("Number of Water Molecules", fontsize=13, fontweight='bold')
ax.set_xlabel("Z-coordinate relative to bilayer midplane (Å)", fontsize=13, fontweight='bold')
ax.set_title("Water Density Distribution Along the Membrane Normal (Shell-Filtered, 0-15 Å)\n"
             "Midplane at Z = 0; shaded band = mean bilayer core between leaflets",
            fontsize=13, fontweight='bold', pad=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.legend(fontsize=11, loc='best', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(f"{output_prefix}_01_water_distribution.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.close()

if has_protein:
    # === Figure 2: Solute Radius Profile ===
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    ax.plot(z_centers, mean_radius, color='lightgray', linestyle='--', linewidth=1.5,
            alpha=0.7, label='Raw data', zorder=1)
    ax.plot(z_centers, mean_radius_smooth, color=color_palette['solute'], linewidth=3.0,
            label='Smoothed (σ=1.0)', zorder=3)

    ax.fill_between(z_centers, mean_radius - std_radius, mean_radius + std_radius,
                    color=color_palette['solute'], alpha=0.15, label='±1σ uncertainty', zorder=2)

    ax.set_ylabel("Solute Radius (Å)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Z-coordinate (Å)", fontsize=13, fontweight='bold')
    ax.set_title("Peptide/Protein Radius Profile Along Z-Axis",
                fontsize=14, fontweight='bold', pad=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=11, loc='best', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_02_solute_radius.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close()
    print(f"[\u2713] Saved {output_prefix}_02_solute_radius.png")
else:
    print("[INFO] Skipping Figure 2 (peptide/protein solute radius profile) "
          "(no peptide/protein present in this system).")

if has_protein:
    # === Figure 3: Pore Evolution - Dual Panel ===
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    # Pore count
    ax = axes[0]
    ax.plot(time_axis, pore_count_evolution, color='lightgray', linestyle='--', linewidth=1.5,
            alpha=0.6, label='Raw data', zorder=1)
    ax.plot(time_axis, pore_count_smooth, color=color_palette['pore'], linewidth=3.0,
            label='Smoothed (σ=1.0)', zorder=3)
    ax.fill_between(time_axis, pore_count_evolution, pore_count_smooth,
                    color=color_palette['pore'], alpha=0.2, zorder=2)

    ax.set_ylabel("Number of Pores", fontsize=12, fontweight='bold')
    ax.set_title("Pore Count Evolution During Simulation",
                fontsize=13, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(bottom=0)

    # Pore radius
    ax = axes[1]
    ax.plot(time_axis, pore_radii_evolution, color='lightgray', linestyle='--', linewidth=1.5,
            alpha=0.6, label='Raw data', zorder=1)
    ax.plot(time_axis, pore_radii_smooth, color=color_palette['pore'], linewidth=3.0,
            label='Smoothed (σ=1.0)', zorder=3)
    ax.fill_between(time_axis, pore_radii_evolution, pore_radii_smooth,
                    color=color_palette['pore'], alpha=0.2, zorder=2)
    ax.axhline(y=pore_radius_threshold, color='red', linestyle=':', linewidth=2.5,
              label=f'Detection Threshold ({pore_radius_threshold} Å)', alpha=0.8)

    ax.set_ylabel("Mean Pore Radius (Å)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (ns)", fontsize=12, fontweight='bold')
    ax.set_title("Pore Radius Evolution During Simulation",
                fontsize=13, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_03_pore_evolution.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close()
    print(f"[\u2713] Saved {output_prefix}_03_pore_evolution.png")
else:
    print("[INFO] Skipping Figure 3 (peptide/protein-induced pore evolution) "
          "(no peptide/protein present in this system).")

if has_protein:
    # === Figure 4: Water Permeation ===
    pore_waters = df_permeation['Total Pore Waters'].values
    pore_waters_smooth = gaussian_filter1d(pore_waters.astype(float), sigma=kde_sigma)

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)

    ax.plot(time_axis, pore_waters, color='lightgray', linestyle='--', linewidth=1.5,
            alpha=0.6, label='Raw data', zorder=1)
    ax.plot(time_axis, pore_waters_smooth, color=color_palette['pore'], linewidth=3.0,
            label='Smoothed (σ=1.0)', zorder=3)
    ax.fill_between(time_axis, pore_waters, pore_waters_smooth,
                    color=color_palette['pore'], alpha=0.2, zorder=2)

    ax.set_ylabel("Water Molecules in Pores", fontsize=13, fontweight='bold')
    ax.set_xlabel("Time (ns)", fontsize=13, fontweight='bold')
    ax.set_title("Water Permeation Through Pores Over Time",
                fontsize=14, fontweight='bold', pad=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(fontsize=11, loc='best', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_04_water_permeation.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close()
    print(f"[\u2713] Saved {output_prefix}_04_water_permeation.png")
else:
    print("[INFO] Skipping Figure 4 (water permeation through peptide/protein-induced pores) "
          "(no peptide/protein present in this system).")

if has_protein:
    # === Figure 4b: Combined Water Permeation & Pore Radius Time Evolution ===
    # Overlays both quantities on a shared time axis (dual y-axis) so their
    # relationship over the trajectory can be read directly from one figure.
    fig, ax1 = plt.subplots(figsize=(13, 6.5))
    ax2 = ax1.twinx()

    ax1.plot(time_axis, pore_waters, color=color_palette['water'], linestyle='--',
             linewidth=1.2, alpha=0.35, zorder=1)
    line_water, = ax1.plot(time_axis, pore_waters_smooth, color=color_palette['water'],
                            linewidth=3.0, label='Water Permeation (smoothed)', zorder=3)
    ax1.fill_between(time_axis, pore_waters_smooth, color=color_palette['water'], alpha=0.12, zorder=2)

    ax2.plot(time_axis, pore_radii_evolution, color=color_palette['pore'], linestyle='--',
             linewidth=1.2, alpha=0.35, zorder=1)
    line_radius, = ax2.plot(time_axis, pore_radii_smooth, color=color_palette['pore'],
                             linewidth=3.0, label='Pore Radius (smoothed)', zorder=3)
    line_thresh = ax2.axhline(y=pore_radius_threshold, color='red', linestyle=':', linewidth=2.0,
                               alpha=0.75, label=f'Detection Threshold ({pore_radius_threshold} Å)', zorder=1)

    ax1.set_xlabel("Time (ns)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Water Molecules in Pores", fontsize=13, fontweight='bold', color=color_palette['water'])
    ax2.set_ylabel("Mean Pore Radius (Å)", fontsize=13, fontweight='bold', color=color_palette['pore'])
    ax1.tick_params(axis='y', labelcolor=color_palette['water'])
    ax2.tick_params(axis='y', labelcolor=color_palette['pore'])
    ax1.set_ylim(bottom=0)
    ax1.set_title("Water Permeation and Pore Radius: Combined Temporal Evolution",
                  fontsize=14, fontweight='bold', pad=20)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    ax1.legend([line_water, line_radius, line_thresh],
               [line_water.get_label(), line_radius.get_label(), line_thresh.get_label()],
               fontsize=10, loc='upper right', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_04b_permeation_radius_combined.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close()
    print(f"[\u2713] Saved {output_prefix}_04b_permeation_radius_combined.png")
else:
    print("[INFO] Skipping Figure 4b (combined water permeation & pore radius) "
          "(no peptide/protein present in this system).")

# === Figure 5: Comprehensive Dashboard ===
fig = plt.figure(figsize=(16, 15.5))
gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. Water distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(z_centers, mean_waters_smooth, color=color_palette['water'], linewidth=2.5)
ax1.fill_between(z_centers, mean_waters - std_waters, mean_waters + std_waters,
                 color=color_palette['water'], alpha=0.3)
ax1.set_ylabel("Waters", fontsize=10, fontweight='bold')
ax1.set_title("(A) Water Density", fontsize=11, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, alpha=0.2)

# 2. Solute radius
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(z_centers, mean_radius_smooth, color=color_palette['solute'], linewidth=2.5)
ax2.fill_between(z_centers, mean_radius - std_radius, mean_radius + std_radius,
                 color=color_palette['solute'], alpha=0.3)
ax2.set_ylabel("Radius (Å)", fontsize=10, fontweight='bold')
ax2.set_title("(B) Solute Radius", fontsize=11, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, alpha=0.2)

# 3. Combined Z-distribution: water count as a smoothed line/area (not bars),
#    overlaid with the solute radius profile on a twin axis.
ax3 = fig.add_subplot(gs[0, 2])
water_z_dist = num_waters_per_bin.mean(axis=0)
water_z_dist_smooth = gaussian_filter1d(water_z_dist, sigma=kde_sigma) if apply_kde_smoothing else water_z_dist
radius_z_dist = radius_per_bin.mean(axis=0)
ax3_twin = ax3.twinx()
ax3.plot(z_centers, water_z_dist, color=color_palette['water'], linestyle='--',
         linewidth=1.0, alpha=0.4, zorder=1)
ax3.plot(z_centers, water_z_dist_smooth, color=color_palette['water'], linewidth=2.3,
         label='Waters', zorder=3)
ax3.fill_between(z_centers, 0, water_z_dist_smooth, color=color_palette['water'],
                  alpha=0.22, zorder=2)
ax3_twin.plot(z_centers, radius_z_dist, color=color_palette['solute'], linewidth=2.5,
              marker='o', markersize=4, label='Radius', zorder=3)
ax3.set_ylabel("Water Count", fontsize=10, fontweight='bold', color=color_palette['water'])
ax3_twin.set_ylabel("Radius (Å)", fontsize=10, fontweight='bold', color=color_palette['solute'])
ax3.set_title("(C) Combined Z-Profile", fontsize=11, fontweight='bold')
ax3.spines['top'].set_visible(False)
ax3.tick_params(axis='y', labelcolor=color_palette['water'])
ax3_twin.tick_params(axis='y', labelcolor=color_palette['solute'])
ax3.set_ylim(bottom=0)
ax3.grid(True, alpha=0.2, axis='y')

# 4. Pore count evolution
ax4 = fig.add_subplot(gs[1, :2])
ax4.plot(time_axis, pore_count_evolution, color='lightgray', linestyle='--',
         linewidth=1, alpha=0.5, zorder=1)
ax4.plot(time_axis, pore_count_smooth, color=color_palette['pore'], linewidth=2.5, zorder=3)
ax4.fill_between(time_axis, pore_count_evolution, alpha=0.2, color=color_palette['pore'], zorder=2)
ax4.set_ylabel("Pore Count", fontsize=10, fontweight='bold')
ax4.set_title("(D) Pore Count Evolution", fontsize=11, fontweight='bold')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(True, alpha=0.2)

# 5. Pore radius distribution
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(pore_radii_evolution[pore_radii_evolution > 0], bins=15,
         color=color_palette['pore'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.axvline(np.mean(pore_radii_evolution[pore_radii_evolution > 0]),
            color='red', linestyle='--', linewidth=2, label='Mean')
ax5.set_xlabel("Radius (Å)", fontsize=9, fontweight='bold')
ax5.set_ylabel("Frequency", fontsize=9, fontweight='bold')
ax5.set_title("(E) Radius Distribution", fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.grid(True, alpha=0.2, axis='y')

# 6. Pore radius evolution
ax6 = fig.add_subplot(gs[2, 0])
ax6.plot(time_axis, pore_radii_smooth, color=color_palette['pore'], linewidth=2.5)
ax6.fill_between(time_axis, pore_radii_evolution, alpha=0.2, color=color_palette['pore'])
ax6.set_ylabel("Radius (Å)", fontsize=10, fontweight='bold')
ax6.set_xlabel("Time (ns)", fontsize=10, fontweight='bold')
ax6.set_title("(F) Pore Radius Evolution", fontsize=11, fontweight='bold')
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.grid(True, alpha=0.2)

# 7. Water permeation
ax7 = fig.add_subplot(gs[2, 1])
ax7.plot(time_axis, pore_waters_smooth, color=color_palette['pore'], linewidth=2.5)
ax7.fill_between(time_axis, pore_waters, alpha=0.2, color=color_palette['pore'])
ax7.set_ylabel("Waters", fontsize=10, fontweight='bold')
ax7.set_xlabel("Time (ns)", fontsize=10, fontweight='bold')
ax7.set_title("(G) Water Permeation", fontsize=11, fontweight='bold')
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.grid(True, alpha=0.2)

# 8. Scatter: Radius vs Pore Count
ax8 = fig.add_subplot(gs[2, 2])
scatter = ax8.scatter(pore_count_evolution, pore_radii_evolution,
                     c=time_axis, cmap='twilight_shifted', s=50, alpha=0.7,
                     edgecolor='black', linewidth=0.5)
cb = plt.colorbar(scatter, ax=ax8)
cb.set_label("Time (ns)", fontsize=9, fontweight='bold')
ax8.set_xlabel("Pore Count", fontsize=10, fontweight='bold')
ax8.set_ylabel("Radius (Å)", fontsize=10, fontweight='bold')
ax8.set_title("(H) Count vs Radius", fontsize=11, fontweight='bold')
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)
ax8.grid(True, alpha=0.2)

# 9. Combined water permeation + pore radius, over time (full width)
ax9 = fig.add_subplot(gs[3, :])
ax9_twin = ax9.twinx()
ax9.plot(time_axis, pore_waters, color=color_palette['water'], linestyle='--',
         linewidth=1.0, alpha=0.35, zorder=1)
line_water_d, = ax9.plot(time_axis, pore_waters_smooth, color=color_palette['water'],
                          linewidth=2.5, label='Water Permeation', zorder=3)
ax9.fill_between(time_axis, pore_waters_smooth, color=color_palette['water'], alpha=0.12, zorder=2)
ax9_twin.plot(time_axis, pore_radii_evolution, color=color_palette['pore'], linestyle='--',
              linewidth=1.0, alpha=0.35, zorder=1)
line_radius_d, = ax9_twin.plot(time_axis, pore_radii_smooth, color=color_palette['pore'],
                                linewidth=2.5, label='Pore Radius', zorder=3)
ax9.set_ylabel("Waters", fontsize=10, fontweight='bold', color=color_palette['water'])
ax9_twin.set_ylabel("Radius (Å)", fontsize=10, fontweight='bold', color=color_palette['pore'])
ax9.set_xlabel("Time (ns)", fontsize=10, fontweight='bold')
ax9.set_title("(I) Combined Water Permeation & Pore Radius Evolution", fontsize=11, fontweight='bold')
ax9.tick_params(axis='y', labelcolor=color_palette['water'])
ax9_twin.tick_params(axis='y', labelcolor=color_palette['pore'])
ax9.set_ylim(bottom=0)
ax9.spines['top'].set_visible(False)
ax9_twin.spines['top'].set_visible(False)
ax9.legend([line_water_d, line_radius_d], [line_water_d.get_label(), line_radius_d.get_label()],
           fontsize=9, loc='upper right', frameon=True, fancybox=True, shadow=True)
ax9.grid(True, alpha=0.2)

fig.suptitle("Comprehensive Pore and Water Analysis Dashboard",
            fontsize=15, fontweight='bold', y=0.995)
plt.savefig(f"{output_prefix}_05_dashboard.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.close()

# === Figure 6: Statistical Analysis ===
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Histogram of pore radii
ax = axes[0, 0]
valid_radii = pore_radii_evolution[pore_radii_evolution > 0]
if len(valid_radii) > 0:
    n, bins, patches = ax.hist(valid_radii, bins=20, color=color_palette['pore'],
                               alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axvline(np.mean(valid_radii), color='red', linestyle='--', linewidth=2.5,
              label=f'μ = {np.mean(valid_radii):.2f} Å')
    ax.axvline(np.median(valid_radii), color='green', linestyle='--', linewidth=2.5,
              label=f'σ = {np.std(valid_radii):.2f} Å')
    ax.set_xlabel("Pore Radius (Å)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=11, fontweight='bold')
    ax.set_title("(A) Pore Radius Distribution", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, frameon=True, fancybox=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

# Phase diagram: Count vs Time
ax = axes[0, 1]
scatter = ax.scatter(time_axis, pore_count_evolution, c=pore_radii_evolution,
                    cmap='RdYlGn', s=100, alpha=0.6, edgecolor='black', linewidth=1)
cb = plt.colorbar(scatter, ax=ax)
cb.set_label("Radius (Å)", fontsize=10, fontweight='bold')
ax.set_xlabel("Time (ns)", fontsize=11, fontweight='bold')
ax.set_ylabel("Pore Count", fontsize=11, fontweight='bold')
ax.set_title("(B) Phase Diagram: Count vs Time", fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2)

# Radius vs Water permeation
ax = axes[1, 0]
scatter = ax.scatter(pore_radii_evolution, pore_waters,
                    c=time_axis, cmap='viridis', s=100, alpha=0.6,
                    edgecolor='black', linewidth=1)
cb = plt.colorbar(scatter, ax=ax)
cb.set_label("Time (ns)", fontsize=10, fontweight='bold')
ax.set_xlabel("Pore Radius (Å)", fontsize=11, fontweight='bold')
ax.set_ylabel("Waters in Pores", fontsize=11, fontweight='bold')
ax.set_title("(C) Radius vs Water Permeation", fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2)

# Time series correlation heatmap
ax = axes[1, 1]
corr_matrix = np.corrcoef([pore_count_evolution, pore_radii_evolution, pore_waters])
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Count', 'Radius', 'Waters'], fontsize=10, fontweight='bold')
ax.set_yticklabels(['Count', 'Radius', 'Waters'], fontsize=10, fontweight='bold')
ax.set_title("(D) Correlation Matrix", fontsize=12, fontweight='bold')
cb = plt.colorbar(im, ax=ax)
cb.set_label("Correlation", fontsize=10, fontweight='bold')

# Add correlation values
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontweight='bold',
                      fontsize=11)

plt.tight_layout()
plt.savefig(f"{output_prefix}_06_statistics.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.close()

# === Figure 7: Heatmap of Water Distribution ===
frames_binned = np.arange(0, len(frame_indices), time_bin_size)
if len(frames_binned) > 1:
    heatmap_data = []
    for i in range(len(frames_binned) - 1):
        start_frame = frames_binned[i]
        end_frame = min(frames_binned[i + 1], len(frame_indices))
        avg_waters = num_waters_per_bin[start_frame:end_frame].mean(axis=0)
        heatmap_data.append(avg_waters)
    
    heatmap_data = np.array(heatmap_data)
    
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111)
    
    im = ax.imshow(heatmap_data.T, aspect='auto', origin='lower', cmap='YlOrRd',
                   extent=[0, time_axis[-1], z_bins[0], z_bins[-1]],
                   interpolation='bilinear')
    
    ax.set_xlabel("Time (ns)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Z-coordinate (Å)", fontsize=13, fontweight='bold')
    ax.set_title("Water Density Heatmap: Spatial-Temporal Evolution",
                fontsize=14, fontweight='bold', pad=20)
    
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Water Molecules", fontsize=12, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_07_heatmap_spatiotemporal.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close()

# ============================================================================
# === 3D VISUALIZATIONS ===
# ============================================================================

if save_3d_plots:
    print("[INFO] Generating 3D visualizations...")
    
    # Determine frames to visualize
    frames_to_visualize = []
    for idx in frame_indices_3d:
        if idx < 0:
            actual_idx = len(u.trajectory) + idx
        else:
            actual_idx = idx
        if 0 <= actual_idx < len(u.trajectory):
            frames_to_visualize.append(actual_idx)
    
    # === 3D Plot 1: Water molecules in pore (Initial) ===
    if len(frames_to_visualize) > 0:
        frame_idx = frames_to_visualize[0]
        u.trajectory[frame_idx]
        
        solute_pos = solute.positions
        water_pos = waters.positions
        center_z = solute_pos[:, 2].mean()
        center_xy = solute_pos[:, :2].mean(axis=0)
        
        pore_regions = pore_regions_by_frame[frame_idx]
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        whiten_3d_axes(ax)
        
        solute_pos_centered = solute_pos.copy()
        solute_pos_centered[:, :2] -= center_xy
        solute_pos_centered[:, 2] -= center_z
        ax.scatter(solute_pos_centered[:, 0], solute_pos_centered[:, 1],
                  solute_pos_centered[:, 2], c=color_palette['solute'],
                  s=30, alpha=0.6, label='Peptide/Protein', edgecolors='darkred')
        
        water_pos_centered = water_pos.copy()
        water_pos_centered[:, :2] -= center_xy
        water_pos_centered[:, 2] -= center_z
        
        if len(pore_regions) > 0:
            for pore_idx, pore in enumerate(pore_regions):
                if 'cluster_coords' in pore:
                    cluster = pore['cluster_coords']
                    cluster_centered = cluster.copy()
                    cluster_centered[:, :2] -= center_xy
                    cluster_centered[:, 2] -= center_z
                    ax.scatter(cluster_centered[:, 0], cluster_centered[:, 1],
                              cluster_centered[:, 2],
                              c=[color_palette['pore']], s=80, alpha=0.8,
                              label=f'Pore {pore_idx + 1}', edgecolors='darkgreen',
                              marker='*')
        
        if len(pore_regions) > 0:
            for pore in pore_regions:
                if 'z_center' in pore:
                    z_pore = pore['z_center']
                    radius = pore.get('solute_radius', pore.get('radius', 5))
                    theta = np.linspace(0, 2*np.pi, 30)
                    x_cyl = radius * np.cos(theta)
                    y_cyl = radius * np.sin(theta)
                    z_cyl = np.ones_like(theta) * z_pore
                    ax.plot(x_cyl, y_cyl, z_cyl, 'g--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel("X (Å)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Y (Å)", fontsize=12, fontweight='bold')
        ax.set_zlabel("Z (Å)", fontsize=12, fontweight='bold')
        ax.set_title(f"3D Water Distribution in Pore (Frame {frame_idx})",
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(fontsize=10, loc='upper left')
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_3d_01_pore_structure_frame{frame_idx}.png",
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
        plt.close()
        
        # === 3D Plot 2: Water molecules in pore (Final) ===
        if len(frames_to_visualize) > 1:
            frame_idx = frames_to_visualize[-1]
            u.trajectory[frame_idx]
            
            solute_pos = solute.positions
            water_pos = waters.positions
            center_z = solute_pos[:, 2].mean()
            center_xy = solute_pos[:, :2].mean(axis=0)
            
            pore_regions = pore_regions_by_frame[frame_idx]
            
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            whiten_3d_axes(ax)
            
            solute_pos_centered = solute_pos.copy()
            solute_pos_centered[:, :2] -= center_xy
            solute_pos_centered[:, 2] -= center_z
            ax.scatter(solute_pos_centered[:, 0], solute_pos_centered[:, 1],
                      solute_pos_centered[:, 2], c=color_palette['solute'],
                      s=30, alpha=0.6, label='Peptide/Protein', edgecolors='darkred')
            
            water_pos_centered = water_pos.copy()
            water_pos_centered[:, :2] -= center_xy
            water_pos_centered[:, 2] -= center_z
            
            if len(pore_regions) > 0:
                for pore_idx, pore in enumerate(pore_regions):
                    if 'cluster_coords' in pore:
                        cluster = pore['cluster_coords']
                        cluster_centered = cluster.copy()
                        cluster_centered[:, :2] -= center_xy
                        cluster_centered[:, 2] -= center_z
                        ax.scatter(cluster_centered[:, 0], cluster_centered[:, 1],
                                  cluster_centered[:, 2],
                                  c=[color_palette['pore']], s=80, alpha=0.8,
                                  label=f'Pore {pore_idx + 1}', edgecolors='darkgreen',
                                  marker='*')
            
            if len(pore_regions) > 0:
                for pore in pore_regions:
                    if 'z_center' in pore:
                        z_pore = pore['z_center']
                        radius = pore.get('solute_radius', pore.get('radius', 5))
                        theta = np.linspace(0, 2*np.pi, 30)
                        x_cyl = radius * np.cos(theta)
                        y_cyl = radius * np.sin(theta)
                        z_cyl = np.ones_like(theta) * z_pore
                        ax.plot(x_cyl, y_cyl, z_cyl, 'g--', linewidth=2, alpha=0.7)
            
            ax.set_xlabel("X (Å)", fontsize=12, fontweight='bold')
            ax.set_ylabel("Y (Å)", fontsize=12, fontweight='bold')
            ax.set_zlabel("Z (Å)", fontsize=12, fontweight='bold')
            ax.set_title(f"3D Water Distribution in Pore (Frame {frame_idx})",
                        fontsize=14, fontweight='bold', pad=20)
            
            ax.legend(fontsize=10, loc='upper left')
            ax.view_init(elev=20, azim=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_3d_02_pore_structure_frame{frame_idx}.png",
                       dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
            plt.close()
    
    # === 3D Plot 3: Pore Evolution Surface ===
    if len(frames_to_visualize) > 0:
        frame_idx_start = frames_to_visualize[0]
        frame_idx_end = frames_to_visualize[-1] if len(frames_to_visualize) > 1 else frame_idx_start
        
        frames_range = np.linspace(frame_idx_start, frame_idx_end,
                                   min(10, max(1, frame_idx_end - frame_idx_start + 1)), dtype=int)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        whiten_3d_axes(ax)
        
        colors_range = plt.cm.viridis(np.linspace(0, 1, len(frames_range)))
        
        for color, f_idx in zip(colors_range, frames_range):
            z_vals = z_centers
            waters_vals = num_waters_per_bin[f_idx]
            frame_times = np.ones_like(z_vals) * time_axis[f_idx]
            
            ax.plot(frame_times, z_vals, waters_vals, color=color, linewidth=2.5, alpha=0.8)
            ax.scatter([frame_times[0]], [z_vals[0]], [waters_vals[0]],
                      color=color, s=50, alpha=0.9)
        
        ax.set_xlabel("Time (ns)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Z-coordinate (Å)", fontsize=12, fontweight='bold')
        ax.set_zlabel("Water Count", fontsize=12, fontweight='bold')
        ax.set_title("3D Surface: Water Distribution Evolution",
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.view_init(elev=25, azim=45)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_3d_03_water_evolution_surface.png",
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
        plt.close()
    
    # === 3D Plot 4: Phase Space Trajectory ===
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    whiten_3d_axes(ax)
    
    step = max(1, len(frame_indices) // 100)
    frame_subset = frame_indices[::step]
    time_subset = time_axis[::step]
    count_subset = pore_count_evolution[::step]
    radius_subset = pore_radii_evolution[::step]
    
    scatter = ax.scatter(time_subset, count_subset, radius_subset,
                        c=time_subset, cmap='twilight_shifted', s=80,
                        alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.plot(time_subset, count_subset, radius_subset, 'k-', alpha=0.3, linewidth=1)
    
    ax.set_xlabel("Time (ns)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Pore Count", fontsize=12, fontweight='bold')
    ax.set_zlabel("Pore Radius (Å)", fontsize=12, fontweight='bold')
    ax.set_title("3D Phase Space Trajectory",
                fontsize=14, fontweight='bold', pad=20)
    
    cb = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cb.set_label("Time (ns)", fontsize=11, fontweight='bold')
    
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_3d_04_phase_space.png",
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close()
    
    print("[✓] 3D visualizations generated")

# === Figure 8: Contour Plot ===
fig, ax = plt.subplots(figsize=(13, 8))

frames_for_contour = np.linspace(0, len(frame_indices) - 1, 30, dtype=int)
z_for_contour = z_centers

time_grid, z_grid = np.meshgrid(time_axis[frames_for_contour], z_for_contour)
water_grid = num_waters_per_bin[frames_for_contour].T

vmin, vmax = float(water_grid.min()), float(water_grid.max())
if vmax <= vmin:
    # Degenerate case: water density is uniform (e.g. all-zero) across the
    # sampled frames/bins. np.linspace(vmin, vmax, 15) would be flat/
    # non-increasing and crash contourf, silently killing every plot after
    # it in the script -- pad a tiny epsilon so the levels are strictly
    # increasing and the plot still renders (as a uniform-color field).
    vmax = vmin + 1e-6
levels = np.linspace(vmin, vmax, 15)
cf = ax.contourf(time_grid, z_grid, water_grid, levels=levels, cmap='YlOrRd')
cs = ax.contour(time_grid, z_grid, water_grid, levels=levels, colors='black',
               alpha=0.3, linewidths=0.5)

ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

ax.set_xlabel("Time (ns)", fontsize=13, fontweight='bold')
ax.set_ylabel("Z-coordinate (Å)", fontsize=13, fontweight='bold')
ax.set_title("Water Density Contour Plot: Spatial-Temporal Evolution",
            fontsize=14, fontweight='bold', pad=20)

cb = plt.colorbar(cf, ax=ax, pad=0.02)
cb.set_label("Water Molecules", fontsize=12, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{output_prefix}_08_contour_plot.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.close()

# === Figure 9: Multi-panel Comparison ===
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
ax.plot(z_centers, mean_waters, 'o-', color='lightgray', markersize=4,
        linewidth=1.5, label='Raw', alpha=0.6)
ax.plot(z_centers, mean_waters_smooth, '-', color=color_palette['water'],
        linewidth=3, label='Smoothed')
ax.fill_between(z_centers, mean_waters, mean_waters_smooth,
                alpha=0.2, color=color_palette['water'])
ax.set_ylabel("Water Count", fontsize=11, fontweight='bold')
ax.set_title("(A) Raw vs Smoothed: Water Distribution",
            fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2)

ax = fig.add_subplot(gs[0, 1])
ax.plot(z_centers, mean_radius, 'o-', color='lightgray', markersize=4,
        linewidth=1.5, label='Raw', alpha=0.6)
ax.plot(z_centers, mean_radius_smooth, '-', color=color_palette['solute'],
        linewidth=3, label='Smoothed')
ax.fill_between(z_centers, mean_radius, mean_radius_smooth,
                alpha=0.2, color=color_palette['solute'])
ax.set_ylabel("Radius (Å)", fontsize=11, fontweight='bold')
ax.set_title("(B) Raw vs Smoothed: Solute Radius",
            fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2)

ax = fig.add_subplot(gs[1, 0])
z_quantiles = [z_centers[i] for i in [5, 15, 25, 35, 45] if i < len(z_centers)]
water_quantiles = [num_waters_per_bin[:, i] for i in [5, 15, 25, 35, 45] if i < len(z_centers)]
bp = ax.boxplot(water_quantiles, labels=[f'{z:.1f}' for z in z_quantiles],
               patch_artist=True, widths=0.6)
for patch in bp['boxes']:
    patch.set_facecolor(color_palette['water'])
    patch.set_alpha(0.7)
ax.set_xlabel("Z-coordinate (Å)", fontsize=11, fontweight='bold')
ax.set_ylabel("Water Count Distribution", fontsize=11, fontweight='bold')
ax.set_title("(C) Water Distribution by Z-Bin", fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2, axis='y')

ax = fig.add_subplot(gs[1, 1])
time_blocks = []
time_labels = []
n_blocks = 5
frames_per_block = len(frame_indices) // n_blocks

for i in range(n_blocks):
    start_frame = i * frames_per_block
    end_frame = (i + 1) * frames_per_block if i < n_blocks - 1 else len(frame_indices)
    time_blocks.append(pore_radii_evolution[start_frame:end_frame])
    time_labels.append(f'{time_axis[start_frame]:.1f}-{time_axis[end_frame-1]:.1f}')

parts = ax.violinplot(time_blocks, positions=range(n_blocks), showmeans=True,
                      showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor(color_palette['pore'])
    pc.set_alpha(0.7)

ax.set_xticks(range(n_blocks))
ax.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Pore Radius (Å)", fontsize=11, fontweight='bold')
ax.set_title("(D) Pore Radius Distribution Over Time",
            fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.2, axis='y')

fig.suptitle("Advanced Comparative Analysis",
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f"{output_prefix}_09_advanced_comparison.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
plt.close()

if has_protein:
    # === Figure 10: Pore Opening/Closing Timeline (Lifetime Gantt Chart) ===
    fig, ax = plt.subplots(figsize=(14, 8))

    if len(df_pore_lifetimes) > 0:
        vmin = df_pore_lifetimes['Mean Radius (Å)'].min()
        vmax = df_pore_lifetimes['Mean Radius (Å)'].max()
        norm = colors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
        cmap = plt.cm.viridis
        bar_height = (z_bins[1] - z_bins[0]) * 0.8

        for _, row in df_pore_lifetimes.iterrows():
            bar_color = cmap(norm(row['Mean Radius (Å)']))
            ax.broken_barh(
                [(row['Start Time (ns)'], max(row['Lifetime (ns)'], time_factor * 0.5))],
                (row['Z-bin center (Å)'] - bar_height / 2, bar_height),
                facecolors=bar_color, edgecolors='black', linewidth=0.3, alpha=0.9
            )

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label("Mean Pore Radius (Å)", fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "No pore-opening events detected", ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')

    ax.set_xlabel("Time (ns)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Z-coordinate (Å)", fontsize=13, fontweight='bold')
    ax.set_title("Pore Opening/Closing Timeline (Lifetime Events by Location)",
                fontsize=14, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_10_pore_lifetime_timeline.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close()
    print(f"[\u2713] Saved {output_prefix}_10_pore_lifetime_timeline.png")
else:
    print("[INFO] Skipping Figure 10 (pore opening/closing timeline) "
          "(no peptide/protein present in this system).")

if has_protein:
    # === Figure 11: Pore Lifetime & Stability Distributions ===
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # (A) Lifetime distribution
    ax = fig.add_subplot(gs[0, 0])
    if len(overall_lifetimes_ns) > 0:
        ax.hist(overall_lifetimes_ns, bins=min(20, max(5, len(overall_lifetimes_ns))),
                color=color_palette['pore'], alpha=0.75, edgecolor='black', linewidth=1.2)
        ax.axvline(np.mean(overall_lifetimes_ns), color='red', linestyle='--', linewidth=2,
                  label=f"Mean = {np.mean(overall_lifetimes_ns):.2f} ns")
        ax.legend(fontsize=9)
    ax.set_xlabel("Pore Lifetime (ns)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=11, fontweight='bold')
    ax.set_title("(A) Global Pore-Open Lifetime Distribution", fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

    # (B) Closed / waiting-time distribution
    ax = fig.add_subplot(gs[0, 1])
    if len(overall_closed_durations_ns) > 0:
        ax.hist(overall_closed_durations_ns, bins=min(20, max(5, len(overall_closed_durations_ns))),
                color=color_palette['solute'], alpha=0.75, edgecolor='black', linewidth=1.2)
        ax.axvline(np.mean(overall_closed_durations_ns), color='blue', linestyle='--', linewidth=2,
                  label=f"Mean = {np.mean(overall_closed_durations_ns):.2f} ns")
        ax.legend(fontsize=9)
    ax.set_xlabel("Closed/Waiting Duration (ns)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=11, fontweight='bold')
    ax.set_title("(B) Pore-Closed Waiting-Time Distribution", fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

    # (C) Occupancy fraction by Z-location
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(df_pore_stability_location['Z-bin center (Å)'], df_pore_stability_location['Occupancy Fraction'] * 100,
          width=np.diff(z_centers)[0] * 0.8, color=color_palette['water'], alpha=0.75, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Z-coordinate (Å)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Occupancy (% of time open)", fontsize=11, fontweight='bold')
    ax.set_title("(C) Pore Stability by Location (Occupancy)", fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

    # (D) Opening-event frequency by Z-location
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(df_pore_stability_location['Z-bin center (Å)'], df_pore_stability_location['Num Opening Events'],
          width=np.diff(z_centers)[0] * 0.8, color=color_palette['membrane'], alpha=0.75, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Z-coordinate (Å)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Number of Opening Events", fontsize=11, fontweight='bold')
    ax.set_title("(D) Pore Opening Frequency by Location", fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle("Pore Lifetime and Stability Analysis", fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_11_pore_stability_distributions.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close()
    print(f"[\u2713] Saved {output_prefix}_11_pore_stability_distributions.png")
else:
    print("[INFO] Skipping Figure 11 (pore lifetime & stability distributions) "
          "(no peptide/protein present in this system).")

if has_protein:
    # === Figure 12: Temporal Evolution & Autocorrelation (Persistence) ===
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(3, 1, figure=fig, hspace=0.4, height_ratios=[1, 1, 1.2])

    # (A) Pore radius: rolling mean/std trend over time (local stability trend)
    ax = fig.add_subplot(gs[0])
    ax.plot(time_axis, pore_radii_evolution, color='lightgray', linewidth=1, alpha=0.6, label='Raw', zorder=1)
    ax.plot(time_axis, pore_radius_rolling_mean, color=color_palette['pore'], linewidth=2.5,
           label=f'Rolling mean (window={rolling_window} frames)', zorder=3)
    ax.fill_between(time_axis, pore_radius_rolling_mean - pore_radius_rolling_std,
                    pore_radius_rolling_mean + pore_radius_rolling_std,
                    color=color_palette['pore'], alpha=0.2, label='±1σ (rolling)', zorder=2)
    ax.set_ylabel("Pore Radius (Å)", fontsize=11, fontweight='bold')
    ax.set_title("(A) Temporal Evolution of Pore Radius Stability", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)

    # (B) Pore open/closed state + rolling pore count
    ax = fig.add_subplot(gs[1])
    ax.fill_between(time_axis, 0, overall_open.astype(int), color=color_palette['water'],
                    alpha=0.25, step='mid', label='Pore open (any)')
    ax2 = ax.twinx()
    ax2.plot(time_axis, pore_count_rolling_mean, color=color_palette['solute'], linewidth=2.5,
            label='Rolling mean pore count')
    ax.set_ylabel("Open State", fontsize=11, fontweight='bold', color=color_palette['water'])
    ax2.set_ylabel("Pore Count (rolling mean)", fontsize=11, fontweight='bold', color=color_palette['solute'])
    ax.set_xlabel("Time (ns)", fontsize=11, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Closed', 'Open'])
    ax.set_title("(B) Pore Open/Closed State and Count Evolution", fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(True, alpha=0.2)

    # (C) Autocorrelation function (persistence/relaxation time)
    ax = fig.add_subplot(gs[2])
    ax.axhline(1 / np.e, color='gray', linestyle=':', linewidth=1.2, label='1/e threshold')
    ax.plot(acf_lags_ns, acf_pore_count, color=color_palette['pore'], linewidth=2.2,
           label=f'Pore count ACF (τ = {tau_pore_count_ns:.2f} ns)')
    ax.plot(acf_lags_ns, acf_pore_radius, color=color_palette['solute'], linewidth=2.2,
           label=f'Pore radius ACF (τ = {tau_pore_radius_ns:.2f} ns)')
    ax.axvline(tau_pore_count_ns, color=color_palette['pore'], linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axvline(tau_pore_radius_ns, color=color_palette['solute'], linestyle='--', linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Lag Time (ns)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Autocorrelation", fontsize=11, fontweight='bold')
    ax.set_title("(C) Autocorrelation & Persistence (Relaxation) Time", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Pore Temporal Evolution and Stability Persistence", fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_12_temporal_evolution_stability.png", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='white')
    plt.close()
    print(f"[\u2713] Saved {output_prefix}_12_temporal_evolution_stability.png")
else:
    print("[INFO] Skipping Figure 12 (temporal evolution & autocorrelation of pore state) "
          "(no peptide/protein present in this system).")

print("[✓] Publication-quality visualizations generated")

# ============================================================================
# === GENERATE COMPREHENSIVE REPORT ===
# ============================================================================

report_text = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         ENHANCED PUBLICATION-QUALITY PORE AND WATER ANALYSIS REPORT          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Prefix: {output_prefix}

═══════════════════════════════════════════════════════════════════════════════
 1. ANALYSIS CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════

Input Files:
  • Topology: {topol_file}
  • Trajectory: {traj_file}
  • Total Frames: {len(frame_indices)}
  • Actual Time Step: {time_factor * 1000:.4f} ps/frame ({'uniform' if dt_is_uniform else 'non-uniform, median used'})
  • Simulated Time Window: {time_axis[0]:.4f} ns → {time_axis[-1]:.4f} ns (Δ = {time_axis[-1] - time_axis[0]:.4f} ns)

Selections:
  • Solute (peptide/protein): {solute_selection} ({solute.n_atoms} atoms) — {'PRESENT, pore analysis enabled' if has_protein else 'NOT PRESENT, pore analysis skipped'}
  • Water: {water_selection} ({waters.n_atoms} atoms)
  • Membrane: {'Available' if has_membrane else 'Not available'}

Bilayer Geometry (dynamic, per-frame):
  • Midplane: recomputed every frame as the leaflet-to-leaflet midpoint; recentered to Z = 0
  • Leaflets: {'split from frame 0, ' + str(leaflet_upper_ag.n_atoms) + ' (upper) / ' + str(leaflet_lower_ag.n_atoms) + ' (lower) atoms' if has_leaflets else 'not available (no membrane / could not split)'}
  {f"• Mean leaflet positions (rel. midplane): upper = {mean_leaflet_upper_z:+.2f} Å, lower = {mean_leaflet_lower_z:+.2f} Å (thickness ≈ {mean_leaflet_upper_z - mean_leaflet_lower_z:.2f} Å)" if has_leaflets else ""}

Analysis Parameters:
  • Shell: {shell_inner}-{shell_outer} Å
  • Pore Detection: {pore_detection_method}{' (SKIPPED — no peptide/protein present)' if not has_protein else ''}
  • Z-bins: {len(z_bins)-1}
  • Smoothing: {f'σ={kde_sigma}' if apply_kde_smoothing else 'Off'}

═══════════════════════════════════════════════════════════════════════════════
 2. KEY RESULTS SUMMARY
═══════════════════════════════════════════════════════════════════════════════

Solute Radius:
{f"  ✓ Mean: {np.mean(mean_radius):.3f} ± {np.std(mean_radius):.3f} Å" if has_protein else "  ✗ Not applicable — no peptide/protein present"}
{f"  ✓ Range: {np.max(mean_radius) - np.min(mean_radius):.3f} Å" if has_protein else ""}

Water Distribution:
  ✓ Mean: {np.mean(mean_waters):.2f} ± {np.std(mean_waters):.2f} molecules
  ✓ Peak at: Z = {z_centers[np.argmax(mean_waters)]:.2f} Å (relative to bilayer midplane)

Pore Formation (peptide/protein-induced):
{f"  ✓ Detection Rate: {100 * np.sum(pore_count_evolution > 0) / len(frame_indices):.1f}%" if has_protein else "  ✗ Not applicable — no peptide/protein present, so no peptide/protein-induced pore exists"}
{f"  ✓ Mean Count: {np.mean(pore_count_evolution):.2f}" if has_protein else ""}
{f"  ✓ Mean Radius: {np.mean(pore_radii_evolution):.3f} Å" if has_protein else ""}

Water Permeation:
{f"  ✓ Mean in Pores: {np.mean(df_permeation['Total Pore Waters']):.2f} molecules" if has_protein else "  ✗ Not applicable — no peptide/protein-induced pore present"}
{f"  ✓ Peak: {np.max(df_permeation['Total Pore Waters']):.0f} molecules" if has_protein else ""}

Water Permeation Depth (Membrane Normal, midplane at Z = 0):
  ✓ Mean Deepest Position: {np.nanmean(water_permeation_depth_evolution):.3f} ± {np.nanstd(water_permeation_depth_evolution):.3f} Å
  ✓ Closest Approach to Bilayer Midplane: {np.nanmin(np.abs(water_permeation_depth_evolution)):.3f} Å

Ion Permeation Depth (Membrane Normal, midplane at Z = 0):
{f"  ✓ Mean Deepest Position: {np.nanmean(ion_permeation_depth_evolution):.3f} ± {np.nanstd(ion_permeation_depth_evolution):.3f} Å" if has_ions else "  ✗ Not available (no atoms matched ion selection: '" + ion_selection + "')"}
{f"  ✓ Closest Approach to Bilayer Midplane: {np.nanmin(np.abs(ion_permeation_depth_evolution)):.3f} Å" if has_ions else ""}

Bilayer-Midplane Occupancy (±{midplane_half_thickness:.1f} Å of true Z = 0, within membrane footprint):
  ✓ Water — Mean Count: {np.mean(water_midplane_count_evolution):.3f} | Peak: {np.max(water_midplane_count_evolution):.0f} | Occupied: {100 * water_midplane_occupancy_fraction:.1f}% of simulation time
{f"  ✓ Ion   — Mean Count: {np.mean(ion_midplane_count_evolution):.3f} | Peak: {np.max(ion_midplane_count_evolution):.0f} | Occupied: {100 * ion_midplane_occupancy_fraction:.1f}% of simulation time" if has_ions else "  ✗ Ion — Not available (no atoms matched ion selection: '" + ion_selection + "')"}

═══════════════════════════════════════════════════════════════════════════════
 3. PORE STABILITY, LIFETIME, AND TEMPORAL EVOLUTION
═══════════════════════════════════════════════════════════════════════════════
{'' if has_protein else '''
  ✗ Not applicable — no peptide/protein present in this system, so there is
    no peptide/protein-induced pore to characterize.
'''}
Pore-Opening Events:
  ✓ Total Events Detected: {len(pore_lifetime_events)}
  ✓ Mean Lifetime: {np.mean(overall_lifetimes_ns) if len(overall_lifetimes_ns) else 0:.3f} ns
  ✓ Median Lifetime: {np.median(overall_lifetimes_ns) if len(overall_lifetimes_ns) else 0:.3f} ns
  ✓ Longest-Lived Pore: {np.max(overall_lifetimes_ns) if len(overall_lifetimes_ns) else 0:.3f} ns
  ✓ Mean Closed/Waiting Duration: {np.mean(overall_closed_durations_ns) if len(overall_closed_durations_ns) else 0:.3f} ns

Pore Stability:
  ✓ Fraction of Simulation Time Open: {100 * overall_open.mean():.1f}%
  ✓ Open-State Radius CV: {f'{radius_cv_global:.3f}' if not np.isnan(radius_cv_global) else 'N/A'}
  ✓ Stability Index (1/CV, higher = more stable): {f'{stability_index_global:.3f}' if not np.isnan(stability_index_global) else 'N/A'}

Temporal Persistence (Autocorrelation):
  ✓ Pore Count Relaxation Time τ: {tau_pore_count_ns:.3f} ns
  ✓ Pore Radius Relaxation Time τ: {tau_pore_radius_ns:.3f} ns
  (τ = time for the autocorrelation to decay below 1/e; longer τ means the
   pore state/size at one moment is more predictive of its state later on,
   i.e. the pore behaves more persistently rather than fluctuating randomly)

═══════════════════════════════════════════════════════════════════════════════
 4. OUTPUT FILES
═══════════════════════════════════════════════════════════════════════════════

Data Files:
  • {output_prefix}_waters.csv
  • {output_prefix}_radius.csv
  • {output_prefix}_pore_evolution.csv
  • {output_prefix}_water_permeation.csv
  • {output_prefix}_permeation_radius_combined.csv (water permeation + pore radius, aligned on time)
  • {output_prefix}_water_permeation_membrane_normal.csv (deepest/mean water Z along membrane normal)
  • {output_prefix}_ion_permeation_membrane_normal.csv (deepest/mean ion Z along membrane normal{'' if has_ions else ' — SKIPPED, no ions found'})
  • {output_prefix}_water_midplane_occupancy.csv (water count at bilayer midplane, Z ≈ 0)
  • {output_prefix}_ion_midplane_occupancy.csv (ion count at bilayer midplane{'' if has_ions else ' — SKIPPED, no ions found'})
  • {output_prefix}_summary.csv
  • {output_prefix}_pore_lifetimes.csv (individual pore-opening events)
  • {output_prefix}_pore_stability_by_location.csv (occupancy/stability per Z-bin)
  • {output_prefix}_temporal_evolution.csv (rolling trends over time)
  • {output_prefix}_pore_stability_summary.csv (global lifetime/stability metrics)

Visualizations:
  • {output_prefix}_01_water_distribution.png
  • {output_prefix}_02_solute_radius.png
  • {output_prefix}_03_pore_evolution.png
  • {output_prefix}_04_water_permeation.png
  • {output_prefix}_04b_permeation_radius_combined.png (water permeation & pore radius, combined)
  • {output_prefix}_05_dashboard.png (includes combined permeation/radius panel I)
  • {output_prefix}_06_statistics.png
  • {output_prefix}_07_heatmap_spatiotemporal.png
  • {output_prefix}_08_contour_plot.png
  • {output_prefix}_09_advanced_comparison.png
  • {output_prefix}_10_pore_lifetime_timeline.png
  • {output_prefix}_11_pore_stability_distributions.png
  • {output_prefix}_12_temporal_evolution_stability.png
  • {output_prefix}_13_water_permeation_membrane_normal.png (depth-of-insertion style)
  • {output_prefix}_14_ion_permeation_membrane_normal.png (depth-of-insertion style{'' if has_ions else ' — SKIPPED, no ions found'})
  • {output_prefix}_15_water_midplane_occupancy.png (water occupancy at bilayer midplane)
  • {output_prefix}_16_ion_midplane_occupancy.png (ion occupancy at bilayer midplane{'' if has_ions else ' — SKIPPED, no ions found'})
  • {output_prefix}_3d_*.png (3D visualizations)

═══════════════════════════════════════════════════════════════════════════════

Analysis completed successfully at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
All figures are publication-ready at 300 DPI resolution.

══��════════════════════════════════════════════════════════════════════════════
"""

with open(f"{output_prefix}_analysis_report.txt", 'w') as f:
    f.write(report_text)

print(f"[✓] Report generated")

# ============================================================================
# === FINAL SUMMARY ===
# ============================================================================

print("\n" + "="*80)
print("✅ PUBLICATION-QUALITY ANALYSIS COMPLETE")
print("="*80)
print(f"\n📊 Summary:")
if has_protein:
    print(f"  • Frames: {len(frame_indices)} | Pores/Frame: {np.mean(pore_count_evolution):.2f}")
    print(f"  • Pore Radius: {np.mean(pore_radii_evolution):.3f} Å | Waters: {np.mean(df_permeation['Total Pore Waters']):.2f}")
    print(f"  • Detection Rate: {100 * np.sum(pore_count_evolution > 0) / len(frame_indices):.1f}%")
    print(f"\n⏱️  Pore Lifetime & Stability:")
    print(f"  • Opening Events: {len(pore_lifetime_events)} | Mean Lifetime: "
          f"{np.mean(overall_lifetimes_ns) if len(overall_lifetimes_ns) else 0:.3f} ns")
    print(f"  • Fraction Time Open: {100 * overall_open.mean():.1f}% | "
          f"Stability Index: {f'{stability_index_global:.3f}' if not np.isnan(stability_index_global) else 'N/A'}")
    print(f"  • Relaxation Time τ (count/radius): {tau_pore_count_ns:.3f} ns / {tau_pore_radius_ns:.3f} ns")
else:
    print(f"  • Frames: {len(frame_indices)}")
    print("  • No peptide/protein present — pore detection, pore radius, and "
          "pore lifetime/stability were skipped (nothing to report).")
print(f"\n🌊 Permeation Depth (Membrane Normal, midplane at Z = 0):")
print(f"  • Water — mean deepest position: {np.nanmean(water_permeation_depth_evolution):.3f} Å "
      f"| closest approach: {np.nanmin(np.abs(water_permeation_depth_evolution)):.3f} Å")
if has_ions:
    print(f"  • Ion   — mean deepest position: {np.nanmean(ion_permeation_depth_evolution):.3f} Å "
          f"| closest approach: {np.nanmin(np.abs(ion_permeation_depth_evolution)):.3f} Å")
else:
    print(f"  • Ion   — skipped (no atoms matched ion selection: '{ion_selection}')")
print(f"\n💧 Bilayer-Midplane Occupancy (±{midplane_half_thickness:.1f} Å of Z = 0):")
print(f"  • Water — mean count: {np.mean(water_midplane_count_evolution):.2f} "
      f"| occupied {100 * water_midplane_occupancy_fraction:.1f}% of simulation time")
if has_ions:
    print(f"  • Ion   — mean count: {np.mean(ion_midplane_count_evolution):.2f} "
          f"| occupied {100 * ion_midplane_occupancy_fraction:.1f}% of simulation time")
print(f"\n📁 Output Files: {output_prefix}_*.csv / .png / .txt")
print("="*80 + "\n")

