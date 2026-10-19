import os
# Force Qt to use xcb to avoid Wayland plugin errors
os.environ["QT_QPA_PLATFORM"] = "xcb"

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

# Input files
topology_file = "md_0_1000.tpr"
trajectory_file = "center1.xtc"

# Load simulation
u = mda.Universe(topology_file, trajectory_file)

# Select lipid atoms
lipids = u.select_atoms("resname POPE POPG")

def separate_leaflets(lipids):
    com_z = lipids.center_of_mass()[2]
    upper_leaflet = lipids.select_atoms(f"prop z > {com_z}")
    lower_leaflet = lipids.select_atoms(f"prop z < {com_z}")
    return upper_leaflet, lower_leaflet

def calculate_thickness_map(upper_leaflet, lower_leaflet, grid_size=(50, 50)):
    upper_z_vals = upper_leaflet.positions[:, 2]
    lower_z_vals = lower_leaflet.positions[:, 2]

    x_vals = np.concatenate([upper_leaflet.positions[:, 0], lower_leaflet.positions[:, 0]])
    y_vals = np.concatenate([upper_leaflet.positions[:, 1], lower_leaflet.positions[:, 1]])
    
    x_bins = np.linspace(x_vals.min(), x_vals.max(), grid_size[0])
    y_bins = np.linspace(y_vals.min(), y_vals.max(), grid_size[1])
    
    upper_height_map, _, _ = np.histogram2d(
        upper_leaflet.positions[:, 0], upper_leaflet.positions[:, 1],
        bins=[x_bins, y_bins], weights=upper_z_vals
    )
    lower_height_map, _, _ = np.histogram2d(
        lower_leaflet.positions[:, 0], lower_leaflet.positions[:, 1],
        bins=[x_bins, y_bins], weights=lower_z_vals
    )

    upper_count_map, _, _ = np.histogram2d(
        upper_leaflet.positions[:, 0], upper_leaflet.positions[:, 1],
        bins=[x_bins, y_bins]
    )
    lower_count_map, _, _ = np.histogram2d(
        lower_leaflet.positions[:, 0], lower_leaflet.positions[:, 1],
        bins=[x_bins, y_bins]
    )

    upper_height_map /= (upper_count_map + 1e-6)
    lower_height_map /= (lower_count_map + 1e-6)

    thickness_map = upper_height_map - lower_height_map

    return thickness_map, lower_height_map, upper_height_map

def calculate_average_maps(grid_size=(50, 50)):
    thickness_maps = []
    lower_leaflet_maps = []
    upper_leaflet_maps = []
    
    for ts in u.trajectory:
        upper_leaflet, lower_leaflet = separate_leaflets(lipids)
        thickness_map, lower_map, upper_map = calculate_thickness_map(
            upper_leaflet, lower_leaflet, grid_size
        )
        
        thickness_maps.append(thickness_map)
        lower_leaflet_maps.append(lower_map)
        upper_leaflet_maps.append(upper_map)

    avg_thickness_map = np.mean(thickness_maps, axis=0)
    avg_lower_leaflet_map = np.mean(lower_leaflet_maps, axis=0)
    avg_upper_leaflet_map = np.mean(upper_leaflet_maps, axis=0)
    
    return avg_thickness_map, avg_lower_leaflet_map, avg_upper_leaflet_map

def plot_contour(data, title, filename_prefix, cmap='jet', label="Height (nm)"):
    fig, ax = plt.subplots(figsize=(6, 6))
    contour = ax.contourf(data, cmap=cmap)
    cbar = plt.colorbar(contour)
    cbar.set_label(label)
    ax.set_title(title)
    ax.set_xlabel("X-axis (bin)")
    ax.set_ylabel("Y-axis (bin)")
    
    # Save both PNG and EPS formats
    plt.savefig(f"{filename_prefix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename_prefix}.eps", format='eps', bbox_inches='tight')
    plt.close()

# Run analysis
avg_thickness_map, avg_lower_map, avg_upper_map = calculate_average_maps()

# Plot & Save
plot_contour(avg_thickness_map, "Average Membrane Thickness", "thickness_map", label="Thickness (nm)")
plot_contour(avg_lower_map, "Average Lower Leaflet Height", "lower_leaflet_map", label="Height (nm)")
plot_contour(avg_upper_map, "Average Upper Leaflet Height", "upper_leaflet_map", label="Height (nm)")

