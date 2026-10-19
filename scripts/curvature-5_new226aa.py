# ---------------------------------------------------------------------------
# Membrane Pore Analysis with Protein Proximity, Temporal Curvature Mapping,
# Pure Pore View, Top View, Heatmaps, and 3D Visuals (with water molecules)
# ---------------------------------------------------------------------------
import MDAnalysis as mda
from scipy.ndimage import gaussian_filter, gaussian_laplace
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

# Input files
topology_file = "md_0_1000.tpr"
trajectory_file = "centerB.xtc"

# Load the MD simulation
u = mda.Universe(topology_file, trajectory_file)
protein = u.select_atoms("protein")
lipids = u.select_atoms("resname POPE POPG")

# Output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Grid parameters
grid_size = 1.0  # in Angstrom
box_dims = u.trajectory.ts.dimensions[:3]
nx = int(box_dims[0] / grid_size)
ny = int(box_dims[1] / grid_size)
x = np.linspace(0, box_dims[0], nx)
y = np.linspace(0, box_dims[1], ny)
X, Y = np.meshgrid(x, y)

# Initialize sums and minima, maxima
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

for ts in tqdm(u.trajectory):
    positions = lipids.positions
    z_center = np.median(positions[:, 2])

    upper_leaflet = lipids.select_atoms(f"prop z > {z_center}")
    lower_leaflet = lipids.select_atoms(f"prop z < {z_center}")

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

    height_upper = compute_height_map(upper_leaflet)
    height_lower = compute_height_map(lower_leaflet)
    sum_upper_height += height_upper
    sum_lower_height += height_lower

    def compute_gaussian_curvature(height_map):
        lap = gaussian_laplace(height_map, sigma=2)
        return lap

    curv_upper = compute_gaussian_curvature(height_upper)
    curv_lower = compute_gaussian_curvature(height_lower)
    sum_upper_curv += curv_upper
    sum_lower_curv += curv_lower

    global_min_curv = min(global_min_curv, np.min(curv_upper), np.min(curv_lower))
    global_max_curv = max(global_max_curv, np.max(curv_upper), np.max(curv_lower))

    # Protein proximity
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

# Average maps
avg_upper_height = sum_upper_height / frame_count
avg_lower_height = sum_lower_height / frame_count
avg_upper_curv = sum_upper_curv / frame_count
avg_lower_curv = sum_lower_curv / frame_count
avg_protein_map = sum_protein_map / frame_count

# Save to disk
np.save(os.path.join(output_dir, "avg_upper_height.npy"), avg_upper_height)
np.save(os.path.join(output_dir, "avg_lower_height.npy"), avg_lower_height)
np.save(os.path.join(output_dir, "avg_upper_curvature.npy"), avg_upper_curv)
np.save(os.path.join(output_dir, "avg_lower_curvature.npy"), avg_lower_curv)
np.save(os.path.join(output_dir, "avg_protein_map.npy"), avg_protein_map)

print("Averages successfully saved!")

# ------------------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------------------

# 1. Average 3D structure
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

plt.savefig(os.path.join(output_dir, "avg_3d_structure.png"))
plt.close()
print("Average 3D structure saved!")

# 2. Top view upper, lower, pure pore
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
im1 = axs[0].imshow(avg_upper_height, cmap='terrain', origin='lower')
axs[0].set_title('Top view - upper leaf')
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(avg_lower_height, cmap='terrain', origin='lower')
axs[1].set_title('Top view - lower leaf')
fig.colorbar(im2, ax=axs[1])

plt.savefig(os.path.join(output_dir, "top_bottom_views.png"))
plt.close()
print("Top view of upper and lower leafs saved!")

# Pure pore view (without protein signals)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, avg_upper_height + 5, cmap='terrain')
ax.plot_surface(X, Y, avg_lower_height - 5, cmap='terrain')
ax.set_title("Pure Pore View")
ax.view_init(elev=30, azim=45)

plt.savefig(os.path.join(output_dir, "pure_pore_3d.png"))
plt.close()
print("Pure pore view (without proteins) saved!")

fig, ax = plt.subplots()
im = ax.imshow((avg_upper_height - np.min(avg_upper_height)) / (np.max(avg_upper_height) - np.min(avg_upper_height)),
               cmap='terrain', origin='lower')
ax.set_title('Top view of Pure Pore')
fig.colorbar(im, ax=ax)

plt.savefig(os.path.join(output_dir, "pure_pore_top.png"))
plt.close()
print("Top view of pure pore saved!")

# 3D view upper leaf only
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, avg_upper_height + 5, cmap='terrain')
ax.set_title("3D view upper leaf")
ax.view_init(elev=30, azim=45)

plt.savefig(os.path.join(output_dir, "3d_upper_only.png"))
plt.close()
print("3D view upper leaf only saved!")

# 3D view lower leaf only
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, avg_lower_height - 5, cmap='terrain')
ax.set_title("3D view lower leaf")
ax.view_init(elev=30, azim=45)

plt.savefig(os.path.join(output_dir, "3d_lower_only.png"))
plt.close()
print("3D view lower leaf only saved!")

# 4. Heatmaps for upper and lower
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
im1 = axs[0].imshow(avg_upper_height, cmap='hot', origin='lower')
axs[0].set_title('Heatmap of upper 3D surface')
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(avg_lower_height, cmap='hot', origin='lower')
axs[1].set_title('Heatmap of lower 3D surface')
fig.colorbar(im2, ax=axs[1])

plt.savefig(os.path.join(output_dir, "heatmaps_3d.png"))
plt.close()
print("Heatmaps of 3D structures saved!")

# 5. Side by side preview
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs[0, 0].set_title('Top view upper')
im = axs[0, 0].imshow(avg_upper_height, cmap='terrain', origin='lower')
fig.colorbar(im, ax=axs[0, 0])

axs[0, 1].set_title('Top view lower')
im = axs[0, 1].imshow(avg_lower_height, cmap='terrain', origin='lower')
fig.colorbar(im, ax=axs[0, 1])

axs[0, 2].set_title('Protein map')
im = axs[0, 2].imshow(avg_protein_map, cmap='hot', origin='lower')
fig.colorbar(im, ax=axs[0, 2])

axs[1, 0].set_title('Curvature upper')
im = axs[1, 0].imshow(avg_upper_curv, cmap='coolwarm', origin='lower')
fig.colorbar(im, ax=axs[1, 0])

axs[1, 1].set_title('Curvature lower')
im = axs[1, 1].imshow(avg_lower_curv, cmap='coolwarm', origin='lower')
fig.colorbar(im, ax=axs[1, 1])

axs[1, 2].set_title('Protein + Pore view')
im = axs[1, 2].imshow((avg_protein_map / np.max(avg_protein_map)) + (avg_upper_height / np.max(avg_upper_height)), cmap='hot', origin='lower')
fig.colorbar(im, ax=axs[1, 2])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "side_by_side_report.png"))
plt.close()
print("Side by side preview of all analyses saved.")

# ------------------------------------------------------------------------------
# 6. Water molecules through pore (3D with views)
# ------------------------------------------------------------------------------
waters = u.select_atoms("resname SOL")  # adjust if water has a different resname

# Grab average Z-center for pore
z_center = np.median(lipids.positions[:, 2])
pore_radius = 15.0  # adjust as needed (Å)

# Collect water positions inside pore region
water_positions = []
for ts in u.trajectory[::10]:  # sample every 10th frame
    pore_waters = [pos for pos in waters.positions 
                   if np.linalg.norm(pos[:2] - box_dims[:2]/2) < pore_radius 
                   and (z_center - 15) < pos[2] < (z_center + 15)]
    water_positions.extend(pore_waters)
water_positions = np.array(water_positions)

# Function to save 3D view with waters
def save_pore_with_waters(view, filename, elev, azim):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, avg_upper_height + 5, cmap='terrain', alpha=0.6)
    ax.plot_surface(X, Y, avg_lower_height - 5, cmap='terrain', alpha=0.6)
    if len(water_positions) > 0:
        ax.scatter(water_positions[:,0], water_positions[:,1], water_positions[:,2],
                   c='cyan', s=5, alpha=0.5, label='Waters')
    ax.set_title(f"Pore with waters - {view}")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.view_init(elev=elev, azim=azim)
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Save all requested views
save_pore_with_waters("Top View (Upper Leaflet)", "waters_top.png", elev=90, azim=0)
save_pore_with_waters("Side View", "waters_side.png", elev=0, azim=0)
save_pore_with_waters("Downward View (Lower Leaflet)", "waters_down.png", elev=-90, azim=0)
print("3D pore with water molecules (top, side, downward views) saved!")

# ------------------------------------------------------------------------------
# 7. Water density map (2D heatmap)
# ------------------------------------------------------------------------------
if len(water_positions) > 0:
    H, xedges, yedges = np.histogram2d(water_positions[:,0], water_positions[:,1],
                                       bins=(nx, ny), range=[[0, box_dims[0]], [0, box_dims[1]]])
    H = gaussian_filter(H, sigma=2)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(H.T, cmap='Blues', origin='lower',
                   extent=[0, box_dims[0], 0, box_dims[1]])
    ax.set_title("Water Density Map in Pore Cross-section")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    fig.colorbar(im, ax=ax, label='Density')
    plt.savefig(os.path.join(output_dir, "water_density_map.png"))
    plt.close()
    print("Water density map saved!")
