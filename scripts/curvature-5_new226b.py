# Membrane Pore Analysis with Protein Proximity, Temporal Curvature Mapping, and Pore Distribution (with Plot Previews)

import MDAnalysis as mda
from scipy.ndimage import gaussian_filter, gaussian_laplace
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

# Input files
topology_file = "md_0_1000.tpr"
trajectory_file = "center1.xtc"

# Load the MD simulation
u = mda.Universe(topology_file, trajectory_file)
protein = u.select_atoms("protein")
lipids = u.select_atoms("resname POPE POPG")

# Output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Grid parameters
grid_size = 1.0  # in Angstrom

# Extract simulation box
box_dims = u.trajectory.ts.dimensions[:3]
nx = int(box_dims[0] / grid_size)
ny = int(box_dims[1] / grid_size)
x = np.linspace(0, box_dims[0], nx)
y = np.linspace(0, box_dims[1], ny)
X, Y = np.meshgrid(x, y)

# Initialize running sums and global min/max
sum_upper_height = np.zeros((ny, nx))
sum_lower_height = np.zeros((ny, nx))
sum_upper_curv = np.zeros((ny, nx))
sum_lower_curv = np.zeros((ny, nx))
sum_protein_map = np.zeros((ny, nx))
frame_count = 0

global_min_curv = float('inf')
global_max_curv = float('-inf')
global_min_protein = float('inf')
global_max_protein = float('-inf')

def compute_height_map(leaflet):
    zmap = np.zeros((ny, nx))
    counts = np.zeros_like(zmap)
    for pos in leaflet.positions:
        ix = int(pos[0] / grid_size)
        iy = int(pos[1] / grid_size)
        if ix < nx and iy < ny:
            zmap[iy, ix] += pos[2]
            counts[iy, ix] += 1
    zmap[counts > 0] /= counts[counts > 0]
    zmap = gaussian_filter(zmap, sigma=2)
    return zmap

def compute_gaussian_curvature(height_map):
    lap = gaussian_laplace(height_map, sigma=2)
    return lap

for ts in tqdm(u.trajectory, disable=False):
    positions = lipids.positions
    z_center = np.median(positions[:, 2])

    upper_leaflet = lipids.select_atoms(f"prop z > {z_center}")
    lower_leaflet = lipids.select_atoms(f"prop z < {z_center}")

    height_upper = compute_height_map(upper_leaflet)
    height_lower = compute_height_map(lower_leaflet)
    sum_upper_height += height_upper
    sum_lower_height += height_lower

    curv_upper = compute_gaussian_curvature(height_upper)
    curv_lower = compute_gaussian_curvature(height_lower)
    sum_upper_curv += curv_upper
    sum_lower_curv += curv_lower

    global_min_curv = min(global_min_curv, np.min(curv_upper), np.min(curv_lower))
    global_max_curv = max(global_max_curv, np.max(curv_upper), np.max(curv_lower))

    # Compute protein proximity map
    protein_map = np.zeros((ny, nx))
    for pos in protein.positions:
        ix = int(pos[0] / grid_size)
        iy = int(pos[1] / grid_size)
        if 0 <= ix < nx and 0 <= iy < ny:
            protein_map[iy, ix] += 1
    protein_map = gaussian_filter(protein_map, sigma=3)
    sum_protein_map += protein_map

    global_min_protein = min(global_min_protein, np.min(protein_map))
    global_max_protein = max(global_max_protein, np.max(protein_map))

    frame_count += 1

# Averages
avg_upper_height = sum_upper_height / frame_count
avg_lower_height = sum_lower_height / frame_count
avg_upper_curv = sum_upper_curv / frame_count
avg_lower_curv = sum_lower_curv / frame_count
avg_protein_map = sum_protein_map / frame_count

# Save average data
np.save(os.path.join(output_dir, "avg_upper_height.npy"), avg_upper_height)
np.save(os.path.join(output_dir, "avg_lower_height.npy"), avg_lower_height)
np.save(os.path.join(output_dir, "avg_upper_curvature.npy"), avg_upper_curv)
np.save(os.path.join(output_dir, "avg_lower_curvature.npy"), avg_lower_curv)
np.save(os.path.join(output_dir, "avg_protein_map.npy"), avg_protein_map)

# Plot 3D view
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, avg_upper_height + 5,
                facecolors=plt.cm.viridis((avg_upper_height - np.min(avg_upper_height)) / (np.max(avg_upper_height) - np.min(avg_upper_height))),
                rstride=1, cstride=1, antialiased=False, alpha=0.9)

ax.plot_surface(X, Y, avg_lower_height - 5,
                facecolors=plt.cm.plasma((avg_lower_height - np.min(avg_lower_height)) / (np.max(avg_lower_height) - np.min(avg_lower_height))),
                rstride=1, cstride=1, antialiased=False, alpha=0.9)

ax.plot_surface(X, Y, avg_protein_map * 2 + 10,
                facecolors=plt.cm.hot((avg_protein_map - global_min_protein) / (global_max_protein - global_min_protein)),
                rstride=1, cstride=1, antialiased=False, alpha=0.5)

ax.set_title("Average 3D Pore Structure")
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")
ax.view_init(elev=30, azim=45)

# Display first for preview
plt.show()

# Plot 2D contour maps
fig, ax = plt.subplots(1, 3, figsize=(21, 6))
levels = np.linspace(global_min_curv, global_max_curv, 50)

c1 = ax[0].contourf(X, Y, avg_upper_curv, levels=levels, cmap="coolwarm")
ax[0].set_title("Avg Gaussian Curvature - Upper")
fig.colorbar(c1, ax=ax[0])

c2 = ax[1].contourf(X, Y, avg_lower_curv, levels=levels, cmap="coolwarm")
ax[1].set_title("Avg Gaussian Curvature - Lower")
fig.colorbar(c2, ax=ax[1])

c3 = ax[2].contourf(X, Y, avg_protein_map, levels=np.linspace(global_min_protein, global_max_protein, 50), cmap="hot")
ax[2].set_title("Avg Protein Proximity")
fig.colorbar(c3, ax=ax[2])

plt.tight_layout()
# Display first for preview
plt.show()

# Top view (pore view upper leaf)
fig, ax = plt.subplots()
c = ax.contourf(X, Y, avg_upper_height, cmap='terrain')
ax.set_title("Top View - Upper Leaflet")
fig.colorbar(c, ax=ax)
plt.show()

# Lower view (pore view lower leaf)
fig, ax = plt.subplots()
c = ax.contourf(X, Y, avg_lower_height, cmap='terrain')
ax.set_title("Bottom View - Lower Leaflet")
fig.colorbar(c, ax=ax)
plt.show()

# 2D Heatmap of Pore Distribution (without protein or peptide) 
fig, ax = plt.subplots()
c = ax.imshow(avg_upper_height, cmap='hot', origin='lower')
ax.set_title("Pore Distribution - Pure Bilayer")
fig.colorbar(c, ax=ax)
plt.show()

# Gaussian view (Distribution of Pore) 
fig, ax = plt.subplots()
pore_vals = avg_upper_height.flatten()
ax.hist(pore_vals, bins=50, color='darkred')
ax.set_title("Distribution of Pore Heights (Upper Leaflet)")
ax.set_xlabel("Height (Å)")
ax.set_ylabel("Count")
plt.show()

print("Analysis complete.")

