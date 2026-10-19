# =============================================================================
# ULTRA-ENHANCED MEMBRANE / PORE / PERMEATION ANALYSIS
# FULLY UNIT-CORRECTED VERSION
# UPDATED: unique plot colors + density plots for all analyses
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda

from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from MDAnalysis.analysis.align import alignto
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# MATPLOTLIB STYLE
# =============================================================================

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 700,
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 3,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight"
})

# =============================================================================
# COLOR MAP (NEW)
# =============================================================================

PLOT_COLORS = {
    "water_volume": "royalblue",
    "water_density": "forestgreen",
    "water_occupancy": "darkorange",
    "ion_concentration": "crimson",
    "pore_radius": "purple",
    "pore_area": "teal",
    "water_flux": "dodgerblue",
    "ion_flux": "firebrick",
    "water_ion_ratio": "goldenrod",
    "membrane_thickness": "darkviolet",
    "lipid_disorder": "darkcyan"
}

# =============================================================================
# USER PARAMETERS
# =============================================================================

topol_file = "md_0_1000.tpr"
traj_file = "center1.xtc"
solute_selection = "not resname TIP3 and not resname ION"
water_selection = "resname W"
ion_selection = "resname ION"
lipid_selection = "resname POPC POPG POPE DPPC DOPC"
output_prefix = "ultra_membrane_analysis"
align_solute = True
shell_inner = 0
shell_outer = 15
z_bins = np.linspace(-30, 30, 80)
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
water_molecular_volume = 29.9
AVOGADRO = 6.02214076e23
kde_sigma = 1.5
time_per_frame_ps = 20

# =============================================================================
# FILE CHECKS
# =============================================================================

if not os.path.isfile(topol_file):
    raise FileNotFoundError(f"Missing topology file: {topol_file}")
if not os.path.isfile(traj_file):
    raise FileNotFoundError(f"Missing trajectory file: {traj_file}")

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
# All figures/CSVs used to be saved as f"{output_prefix}_something.png" --
# i.e. output_prefix was just glued onto filenames in the current working
# directory. That silently breaks when a caller (e.g. the launcher GUI)
# patches output_prefix to "." (files become hidden dot-files like
# "._water_volume.png") or to an output folder path (files land one level
# up from that folder instead of inside it). Instead, always create a real,
# clearly-named output folder and save everything inside it.

if not output_prefix or output_prefix.strip() in (".", "./", ""):
    output_dir = os.path.abspath("ultra_membrane_analysis_output")
else:
    output_dir = os.path.abspath(output_prefix)

os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Saving all outputs to: {output_dir}")


def outpath(filename):
    """Build a path for an output file inside output_dir."""
    return os.path.join(output_dir, filename)

# =============================================================================
# LOAD SYSTEM
# =============================================================================

print("[INFO] Loading trajectory...")
u = mda.Universe(topol_file, traj_file)
solute = u.select_atoms(solute_selection)
waters = u.select_atoms(water_selection)
ions = u.select_atoms(ion_selection)
lipids = u.select_atoms(lipid_selection)

if align_solute:
    print("[INFO] Aligning trajectory...")
    ref = u.copy()
    alignto(u, ref, select=solute_selection)

# =============================================================================
# STORAGE
# =============================================================================

water_volume_per_bin = []
water_density_per_bin = []
water_occupancy_per_bin = []
ion_concentration_per_bin = []
pore_radius_per_bin = []
pore_area_per_bin = []
water_flux_per_bin = []
ion_flux_per_bin = []
water_to_ion_ratio_per_bin = []
membrane_thickness_per_frame = []
lipid_disorder_per_frame = []
all_pore_water_xyz = []
all_pore_ion_xyz = []
all_pore_boundary_xyz = []

# =============================================================================
# MAIN ANALYSIS LOOP
# =============================================================================

print("[INFO] Running analysis...")
for ts in u.trajectory:
    solute_pos = solute.positions
    center_xy = solute_pos[:, :2].mean(axis=0)
    center_z = solute_pos[:, 2].mean()

    water_pos = waters.positions
    water_xy_dist = np.linalg.norm(water_pos[:, :2] - center_xy, axis=1)
    water_z_rel = water_pos[:, 2] - center_z

    if ions.n_atoms > 0:
        ion_pos = ions.positions
        ion_xy_dist = np.linalg.norm(ion_pos[:, :2] - center_xy, axis=1)
        ion_z_rel = ion_pos[:, 2] - center_z

    lipid_pos = lipids.positions
    lipid_xy_dist = np.linalg.norm(lipid_pos[:, :2] - center_xy, axis=1)
    lipid_z_rel = lipid_pos[:, 2] - center_z

    bin_heights = np.diff(z_bins)
    shell_volume_A3 = np.pi * (shell_outer**2 - shell_inner**2) * bin_heights
    shell_volume_L = shell_volume_A3 * 1e-27

    water_shell = ((water_xy_dist >= shell_inner) & (water_xy_dist <= shell_outer))
    filtered_water_z = water_z_rel[water_shell]
    water_hist, _ = np.histogram(filtered_water_z, bins=z_bins)

    water_volume = water_hist * water_molecular_volume
    water_density = water_hist / shell_volume_A3
    water_occupancy = water_volume / shell_volume_A3

    water_volume_per_bin.append(water_volume)
    water_density_per_bin.append(water_density)
    water_occupancy_per_bin.append(water_occupancy)

    if ions.n_atoms > 0:
        ion_shell = ((ion_xy_dist >= shell_inner) & (ion_xy_dist <= shell_outer))
        filtered_ion_z = ion_z_rel[ion_shell]
        ion_hist, _ = np.histogram(filtered_ion_z, bins=z_bins)
        ion_concentration = ((ion_hist / AVOGADRO) / shell_volume_L)
        ion_concentration_per_bin.append(ion_concentration)

    pore_radius_frame = []
    pore_area_frame = []
    local_water_flux = []
    local_ion_flux = []
    local_ratio = []

    for i in range(len(z_bins) - 1):
        z_low = z_bins[i]
        z_high = z_bins[i + 1]

        in_z = ((lipid_z_rel >= z_low) & (lipid_z_rel < z_high))
        local_lipid_dist = lipid_xy_dist[in_z]
        pore_r = np.percentile(local_lipid_dist, 10) if len(local_lipid_dist) > 10 else np.nan
        pore_radius_frame.append(pore_r)
        pore_area_frame.append(np.pi * pore_r**2 if not np.isnan(pore_r) else np.nan)

        in_z_water = ((water_z_rel >= z_low) & (water_z_rel < z_high))
        water_inside = (water_xy_dist[in_z_water] <= pore_r)
        water_count = np.sum(water_inside)
        frame_time_ns = time_per_frame_ps / 1000
        local_water_flux.append((water_count * water_molecular_volume) / frame_time_ns)

        if ions.n_atoms > 0:
            in_z_ion = ((ion_z_rel >= z_low) & (ion_z_rel < z_high))
            ion_inside = (ion_xy_dist[in_z_ion] <= pore_r)
            ion_count = np.sum(ion_inside)
            local_ion_flux.append(ion_count / frame_time_ns)
            local_ratio.append(water_count / ion_count if ion_count > 0 else np.nan)

            if not np.isnan(pore_r):
                local_ion_coords = ion_pos[in_z_ion][ion_inside]
                if len(local_ion_coords) > 0:
                    all_pore_ion_xyz.append(local_ion_coords)

        if not np.isnan(pore_r):
            local_water_coords = water_pos[in_z_water][water_inside]
            if len(local_water_coords) > 0:
                all_pore_water_xyz.append(local_water_coords[::max(1, len(local_water_coords)//100)])

            theta = np.linspace(0, 2*np.pi, 50)
            z_mid = (z_low + z_high) / 2
            x = center_xy[0] + pore_r*np.cos(theta)
            y = center_xy[1] + pore_r*np.sin(theta)
            z = np.full_like(theta, z_mid + center_z)
            all_pore_boundary_xyz.append(np.column_stack((x, y, z)))

    pore_radius_per_bin.append(pore_radius_frame)
    pore_area_per_bin.append(pore_area_frame)
    water_flux_per_bin.append(local_water_flux)
    ion_flux_per_bin.append(local_ion_flux)
    water_to_ion_ratio_per_bin.append(local_ratio)

    upper = lipid_pos[lipid_z_rel > 0]
    lower = lipid_pos[lipid_z_rel < 0]
    if len(upper) > 0 and len(lower) > 0:
        membrane_thickness_per_frame.append(upper[:, 2].mean() - lower[:, 2].mean())

    lipid_disorder_per_frame.append(np.std(lipid_z_rel))

# =============================================================================
# ARRAYS + AVERAGES
# =============================================================================

water_volume_per_bin = np.array(water_volume_per_bin)
water_density_per_bin = np.array(water_density_per_bin)
water_occupancy_per_bin = np.array(water_occupancy_per_bin)
ion_concentration_per_bin = np.array(ion_concentration_per_bin)
pore_radius_per_bin = np.array(pore_radius_per_bin)
pore_area_per_bin = np.array(pore_area_per_bin)
water_flux_per_bin = np.array(water_flux_per_bin)
ion_flux_per_bin = np.array(ion_flux_per_bin)
water_to_ion_ratio_per_bin = np.array(water_to_ion_ratio_per_bin)

mean_water_volume = np.mean(water_volume_per_bin, axis=0)
mean_water_density = np.mean(water_density_per_bin, axis=0)
mean_water_occupancy = np.mean(water_occupancy_per_bin, axis=0)
mean_ion_conc = np.mean(ion_concentration_per_bin, axis=0)
mean_pore_radius = np.nanmean(pore_radius_per_bin, axis=0)
mean_pore_area = np.nanmean(pore_area_per_bin, axis=0)
mean_water_flux = np.mean(water_flux_per_bin, axis=0)
mean_ion_flux = np.mean(ion_flux_per_bin, axis=0)
mean_ratio = np.nanmean(water_to_ion_ratio_per_bin, axis=0)

smooth = lambda y: gaussian_filter1d(np.nan_to_num(y), sigma=kde_sigma)

# =============================================================================
# PLOTTING FUNCTIONS (UPDATED)
# =============================================================================

def plot_profile(x, y, y_s, ylabel, title, filename, color):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, y, '--', alpha=0.30, linewidth=2, color='gray', label='Raw')
    ax.plot(x, y_s, linewidth=3.5, color=color, label='Smoothed')
    ax.fill_between(x, y_s, alpha=0.15, color=color)
    ax.axvline(0, linestyle=':', linewidth=1.5, alpha=0.5, color='black')
    ax.set_xlabel('Membrane Normal Z (Å)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.grid(alpha=0.15, linestyle='--')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_density(values, xlabel, title, filename, color):
    values = np.array(values).flatten()
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(values, bins=40, density=True, alpha=0.35, color=color)
    kde = gaussian_kde(values)
    x = np.linspace(values.min(), values.max(), 500)
    ax.plot(x, kde(x), linewidth=3, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# =============================================================================
# PLOTS + DENSITY PLOTS
# =============================================================================

metrics = [
    ('water_volume', mean_water_volume, 'Water Volume (Å³)', 'Water Volume Along Membrane'),
    ('water_density', mean_water_density, 'Water Density (molecules/Å³)', 'Water Density Along Membrane'),
    ('water_occupancy', mean_water_occupancy, 'Occupancy Fraction', 'Water Occupancy Fraction'),
    ('ion_concentration', mean_ion_conc, 'Ion Concentration (mol/L)', 'Ion Concentration Along Membrane'),
    ('pore_radius', mean_pore_radius, 'Pore Radius (Å)', 'Dynamic Pore Radius'),
    ('pore_area', mean_pore_area, 'Pore Area (Å²)', 'Dynamic Pore Area'),
    ('water_flux', mean_water_flux, 'Water Flux (Å³/ns)', 'Water Flux Through Pore'),
    ('ion_flux', mean_ion_flux, 'Ion Flux (ions/ns)', 'Ion Flux Through Pore'),
    ('water_ion_ratio', mean_ratio, 'Water/Ion Ratio', 'Water-to-Ion Permeation Ratio')
]

for key, y, ylabel, title in metrics:
    ys = smooth(y)
    c = PLOT_COLORS[key]
    plot_profile(z_centers, y, ys, ylabel, title, outpath(f'{key}.png'), c)
    plot_density(y, ylabel, f'{title} Density', outpath(f'{key}_density.png'), c)

# thickness
time_ns = np.arange(len(membrane_thickness_per_frame)) * (time_per_frame_ps / 1000)
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(time_ns, membrane_thickness_per_frame, color=PLOT_COLORS['membrane_thickness'])
ax.set_xlabel('Time (ns)'); ax.set_ylabel('Thickness (Å)'); ax.set_title('Membrane Thickness Dynamics'); ax.grid(alpha=0.15)
plt.tight_layout(); plt.savefig(outpath('membrane_thickness.png')); plt.close()
plot_density(membrane_thickness_per_frame, 'Thickness (Å)', 'Membrane Thickness Density', outpath('membrane_thickness_density.png'), PLOT_COLORS['membrane_thickness'])

# disorder
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(time_ns, lipid_disorder_per_frame, color=PLOT_COLORS['lipid_disorder'])
ax.set_xlabel('Time (ns)'); ax.set_ylabel('Disorder (Å)'); ax.set_title('Lipid Disorder Dynamics'); ax.grid(alpha=0.15)
plt.tight_layout(); plt.savefig(outpath('lipid_disorder.png')); plt.close()
plot_density(lipid_disorder_per_frame, 'Disorder (Å)', 'Lipid Disorder Density', outpath('lipid_disorder_density.png'), PLOT_COLORS['lipid_disorder'])

# =============================================================================
# 3D PORE
# =============================================================================

print('[INFO] Generating 3D pore visualization...')
fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111, projection='3d')

if len(all_pore_boundary_xyz) > 0:
    boundary = np.concatenate(all_pore_boundary_xyz)
    ax.scatter(boundary[:,0], boundary[:,1], boundary[:,2], s=0.15, alpha=0.03, color='gray', label='Pore Boundary')
if len(all_pore_water_xyz) > 0:
    waters_xyz = np.concatenate(all_pore_water_xyz)
    ax.scatter(waters_xyz[:,0], waters_xyz[:,1], waters_xyz[:,2], s=1, alpha=0.28, color='royalblue', label='Water')
if len(all_pore_ion_xyz) > 0:
    ions_xyz = np.concatenate(all_pore_ion_xyz)
    ax.scatter(ions_xyz[:,0], ions_xyz[:,1], ions_xyz[:,2], s=25, alpha=0.9, color='darkred', label='Ions')

ax.set_xlabel('X (Å)'); ax.set_ylabel('Y (Å)'); ax.set_zlabel('Z (Å)')
ax.set_title('3D Pore-only Water/Ion Distribution')
ax.view_init(elev=25, azim=55)
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.legend(frameon=False)
plt.tight_layout(); plt.savefig(outpath('3D_pore_distribution.png'), dpi=900); plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

summary = pd.DataFrame({
    'Metric': ['Mean Membrane Thickness (Å)', 'Mean Lipid Disorder (Å)'],
    'Value': [np.mean(membrane_thickness_per_frame), np.mean(lipid_disorder_per_frame)]
})
summary.to_csv(outpath('summary.csv'), index=False)

print('\n[✅] ULTRA ANALYSIS COMPLETE')
print('------------------------------------------------')
print(f'All standard plots + all density plots + 3D pore plot + summary saved to:')
print(f'  {output_dir}')
print('------------------------------------------------')
