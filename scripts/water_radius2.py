import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.align import alignto
from scipy.ndimage import gaussian_filter1d

# === USER PARAMETERS ===
topol_file = "md_0_1000.tpr"
traj_file = "center1.xtc"

# Non-protein solute: exclude water and ions
solute_selection = "not resname TIP3 and resname ION"
water_selection = "resname W"
output_prefix = "nonprotein_water_analysis"
align_solute = True

# Shell selection (Å)
shell_inner = 0
shell_outer = 15

# KDE smoothing
apply_kde_smoothing = True
kde_sigma = 1.0

# Z-axis binning (Å)
z_bins = np.linspace(-20, 20, 50)
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

# === File checks ===
if not os.path.isfile(topol_file):
    raise FileNotFoundError(f"Missing topology file: {topol_file}")
if not os.path.isfile(traj_file):
    raise FileNotFoundError(f"Missing trajectory file: {traj_file}")

# === Load trajectory ===
u = mda.Universe(topol_file, traj_file)

solute = u.select_atoms(solute_selection)
if solute.n_atoms == 0:
    raise ValueError(f"No atoms found for solute selection: '{solute_selection}'")

waters = u.select_atoms(water_selection)
if waters.n_atoms == 0:
    raise ValueError(f"No atoms found for water selection: '{water_selection}'")

# === Alignment ===
if align_solute:
    print(f"[INFO] Aligning solute using selection: '{solute_selection}'...")
    ref = u.copy()
    alignto(u, ref, select=solute_selection)

# === Result containers ===
num_waters_per_bin = []
radius_per_bin = []

print("[INFO] Beginning frame-by-frame analysis...")
for ts in u.trajectory:
    solute_pos = solute.positions
    center_z = solute_pos[:, 2].mean()
    center_xy = solute_pos[:, :2].mean(axis=0)

    # Compute distances
    water_pos = waters.positions
    dist_xy = np.linalg.norm(water_pos[:, :2] - center_xy, axis=1)
    dist_z = water_pos[:, 2] - center_z

    in_shell = (dist_xy >= shell_inner) & (dist_xy <= shell_outer)
    filtered_z = dist_z[in_shell]

    # Water distribution
    water_hist, _ = np.histogram(filtered_z, bins=z_bins)
    num_waters_per_bin.append(water_hist)

    # Solute radius
    xy_dist = np.linalg.norm(solute_pos[:, :2] - center_xy, axis=1)
    solute_z_rel = solute_pos[:, 2] - center_z

    weighted_radius, _ = np.histogram(solute_z_rel, bins=z_bins, weights=xy_dist)
    bin_counts, _ = np.histogram(solute_z_rel, bins=z_bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_radius = np.nan_to_num(weighted_radius / bin_counts)
    radius_per_bin.append(avg_radius)

# === Convert to arrays ===
num_waters_per_bin = np.array(num_waters_per_bin)
radius_per_bin = np.array(radius_per_bin)

mean_waters = num_waters_per_bin.mean(axis=0)
std_waters = num_waters_per_bin.std(axis=0)

mean_radius = radius_per_bin.mean(axis=0)
std_radius = radius_per_bin.std(axis=0)

# === Apply KDE smoothing ===
if apply_kde_smoothing:
    mean_waters_smooth = gaussian_filter1d(mean_waters, sigma=kde_sigma)
    mean_radius_smooth = gaussian_filter1d(mean_radius, sigma=kde_sigma)
else:
    mean_waters_smooth = mean_waters
    mean_radius_smooth = mean_radius

# === Save CSV ===
df_waters = pd.DataFrame({
    "Z (Å)": z_centers,
    "Mean Number of Waters": mean_waters,
    "Smoothed": mean_waters_smooth,
    "Std": std_waters
})
df_radius = pd.DataFrame({
    "Z (Å)": z_centers,
    "Mean Radius (Å)": mean_radius,
    "Smoothed": mean_radius_smooth,
    "Std": std_radius
})

df_waters.to_csv(f"{output_prefix}_waters.csv", index=False)
df_radius.to_csv(f"{output_prefix}_radius.csv", index=False)

# === Plotting ===
plt.figure(figsize=(7, 3))
plt.plot(z_centers, mean_waters, color='gray', linestyle='--', label='Raw')
plt.plot(z_centers, mean_waters_smooth, color='black', label='Smoothed')
plt.fill_between(z_centers, mean_waters - std_waters, mean_waters + std_waters,
                 color='skyblue', alpha=0.4)
plt.ylabel("Number of Waters")
plt.xlabel("Z (Å)")
plt.title("Water Density Along Z (Shell Filtered)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_prefix}_waters.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 3))
plt.plot(z_centers, mean_radius, color='salmon', linestyle='--', label='Raw')
plt.plot(z_centers, mean_radius_smooth, color='darkred', label='Smoothed')
plt.fill_between(z_centers, mean_radius - std_radius, mean_radius + std_radius,
                 color='lightcoral', alpha=0.4)
plt.ylabel("Solute Radius (Å)")
plt.xlabel("Z (Å)")
plt.title("Solute Radius Along Z")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_prefix}_radius.png", dpi=300)
plt.close()

print(f"[✅] Analysis complete. Outputs saved as: {output_prefix}_*.csv / .png")

