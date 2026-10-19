import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Input Files ---
topology_file = "md_0_300.tpr"
trajectory_file = "md_0_300_center.xtc"

# --- Load Universe ---
try:
    u = mda.Universe(topology_file, trajectory_file)
    print("Topology and trajectory files loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# --- Atom Selections ---
protein = u.select_atoms("protein")
water = u.select_atoms("resname SOL")
phosphate = u.select_atoms("name P")

# --- Residue Z-positions ---
residue_z_positions = []
residue_names = []

for residue in protein.residues:
    z_pos = np.mean(residue.atoms.positions[:, 2])  # Already in Å
    residue_z_positions.append(z_pos)
    residue_names.append(f"{residue.resname}{residue.resid}")

# --- Save Peptide Z Positions ---
with open("peptide_z_positions.xvg", "w") as f:
    f.write("# Residue\tZ_Position(Å)\n")
    for name, z in zip(residue_names, residue_z_positions):
        f.write(f"{name}\t{z:.2f}\n")

# --- Z-Axis Parameters ---
zmin, zmax = -60.0, 60.0  # in Å
nbins = 100
z_bins = np.linspace(zmin, zmax, nbins + 1)
z_bin_centers = 0.5 * (z_bins[1:] + z_bins[:-1])

def calculate_density(atoms, z_bins):
    z_pos = atoms.positions[:, 2]  # Å
    density, _ = np.histogram(z_pos, bins=z_bins)
    return density / density.max()

# --- Density Calculations ---
density_protein_z = calculate_density(protein, z_bins)
density_water_z = calculate_density(water, z_bins)
density_P_z = calculate_density(phosphate, z_bins)

# --- Save Densities Separately ---
np.savetxt("protein_density.xvg", np.column_stack([z_bin_centers, density_protein_z]),
           header="Z (Å)\tProtein_Density", comments='')
np.savetxt("water_density.xvg", np.column_stack([z_bin_centers, density_water_z]),
           header="Z (Å)\tWater_Density", comments='')
np.savetxt("phosphate_density.xvg", np.column_stack([z_bin_centers, density_P_z]),
           header="Z (Å)\tPhosphate_Density", comments='')

# --- Combined Plot of Z Positions + Densities ---
plt.figure(figsize=(12, 6))
plt.plot(residue_names, residue_z_positions, 'o-', color="purple", label="Protein Z-Position")
plt.plot(z_bin_centers, density_water_z * max(residue_z_positions), color="blue", label="Water Density (scaled)")
plt.plot(z_bin_centers, density_P_z * max(residue_z_positions), color="green", label="Phosphate Density (scaled)")
plt.xticks(rotation=90)
plt.xlabel("Residue (Amino Acid and Index)")
plt.ylabel("Z-position (Å)")
plt.title("Z-position of Protein Residues with Density Profiles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("protein_residue_z_position_density.png", format='png', dpi=300)
plt.savefig("protein_residue_z_position_density.eps", format='eps', dpi=300)
plt.close()

# --- Leaflet Separation Based on Phosphate Z ---
phosphate_z = phosphate.positions[:, 2]
median_z = np.median(phosphate_z)

upper_leaflet = phosphate.select_atoms(f"prop z > {median_z}")
lower_leaflet = phosphate.select_atoms(f"prop z < {median_z}")

upper_z = np.mean(upper_leaflet.positions[:, 2])
lower_z = np.mean(lower_leaflet.positions[:, 2])

# --- Save Leaflet Z Positions ---
with open("leaflet_z_positions.xvg", "w") as f:
    f.write("# Upper and Lower Leaflet Z-positions (Å)\n")
    f.write(f"Upper_Leaflet\t{upper_z:.2f}\n")
    f.write(f"Lower_Leaflet\t{lower_z:.2f}\n")

# --- Calculate Depth from Nearest Leaflet ---
residue_depths = []
nearest_leaflets = []

for z in residue_z_positions:
    d_upper = z - upper_z
    d_lower = z - lower_z
    if abs(d_upper) < abs(d_lower):
        residue_depths.append(d_upper)
        nearest_leaflets.append("Upper")
    else:
        residue_depths.append(d_lower)
        nearest_leaflets.append("Lower")

# --- Save Depth Data ---
with open("peptide_depth_relative_to_leaflets.xvg", "w") as f:
    f.write("# Residue\tDepth_from_leaflet(Å)\tNearest_leaflet\n")
    for name, depth, leaf in zip(residue_names, residue_depths, nearest_leaflets):
        f.write(f"{name}\t{depth:.2f}\t{leaf}\n")

# --- Plot: Depth from Leaflet ---
plt.figure(figsize=(12, 6))
colors = ['royalblue' if leaf == "Upper" else 'darkorange' for leaf in nearest_leaflets]
plt.bar(residue_names, residue_depths, color=colors)
plt.axhline(0, color='black', linestyle='--', label='Leaflet Surface')
plt.xticks(rotation=90)
plt.xlabel("Residue (Amino Acid and Index)")
plt.ylabel("Depth from Leaflet Surface (Å)")
plt.title("Residue Insertion Depth Relative to Nearest Leaflet")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("residue_depth_relative_to_leaflets.png", format='png', dpi=300)
plt.savefig("residue_depth_relative_to_leaflets.eps", format='eps', dpi=300)
plt.show()

