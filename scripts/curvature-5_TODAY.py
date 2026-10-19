#!/usr/bin/env python3
# =============================================================================
# Advanced Membrane Topology & Peptide Insertion Analysis
# =============================================================================
#
# INPUT SYSTEM
# -----------------------------------------------------------------------------
# Topology   : md_0_1000.tpr
# Trajectory : center.xtc
#
# Peptide    : PROTEIN
# Lipids     : POPE POPG
#
# =============================================================================
#
# FEATURES
# -----------------------------------------------------------------------------
# ✓ Bilayer leaflet reconstruction
# ✓ Membrane thickness mapping
# ✓ Membrane deformation analysis
# ✓ Peptide insertion depth analysis
# ✓ 3D membrane topology rendering
# ✓ 2D thickness heatmap
# ✓ Curvature approximation
# ✓ Time-aware frame labeling (ns)
# ✓ Smoothed membrane interpolation
# ✓ Publication-quality visualization
#
# =============================================================================
#
# REQUIREMENTS
# -----------------------------------------------------------------------------
# pip install MDAnalysis numpy scipy matplotlib tqdm
#
# =============================================================================

import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# USER SETTINGS
# =============================================================================

TPR_FILE = "md_0_1000.tpr"
XTC_FILE = "center.xtc"

PEPTIDE_RESNAME = "PROTEIN"
LIPID_RESNAMES = "POPE POPG"

FRAME = -1                  # last frame
GRID_SIZE = 90
SMOOTH_SIGMA = 2.0

OUTPUT_PREFIX = "membrane_analysis"

MAX_SOLVENT_POINTS = 12000

# =============================================================================
# LOAD UNIVERSE
# =============================================================================

print("\nLoading simulation...")

u = mda.Universe(TPR_FILE, XTC_FILE)

# =============================================================================
# SELECTIONS
# =============================================================================

lipid_selection = "resname " + LIPID_RESNAMES
peptide_selection = f"protein or resname {PEPTIDE_RESNAME}"

# MARTINI water possibilities
solvent_selection = "resname W PW SOL"

lipids = u.select_atoms(lipid_selection)
peptides = u.select_atoms(peptide_selection)
solvent = u.select_atoms(solvent_selection)

print(f"\nLipids selected  : {len(lipids)} atoms")
print(f"Peptide selected : {len(peptides)} atoms")
print(f"Solvent selected : {len(solvent)} atoms")

# =============================================================================
# TRAJECTORY FRAME
# =============================================================================

if FRAME == -1:
    ts = u.trajectory[-1]
else:
    ts = u.trajectory[FRAME]

time_ns = ts.time / 1000.0

print(f"\nAnalyzing frame: {ts.frame}")
print(f"Simulation time: {time_ns:.2f} ns")

# =============================================================================
# MEMBRANE HEADGROUP SELECTION
# =============================================================================

# Common MARTINI phosphate/headgroup names
headgroup_candidates = [
    "name PO4",
    "name GL0",
    "name P",
]

headgroups = None

for sel in headgroup_candidates:

    hg = lipids.select_atoms(sel)

    if len(hg) > 0:
        headgroups = hg
        print(f"\nUsing headgroup selection: {sel}")
        break

if headgroups is None or len(headgroups) == 0:
    raise ValueError(
        "No membrane headgroups found.\n"
        "Try adjusting headgroup selection."
    )

coords = headgroups.positions

# =============================================================================
# LEAFLET SPLITTING
# =============================================================================

x = coords[:, 0]
y = coords[:, 1]
z = coords[:, 2]

z_mid = np.median(z)

upper_leaflet = coords[z >= z_mid]
lower_leaflet = coords[z < z_mid]

print(f"\nUpper leaflet points : {len(upper_leaflet)}")
print(f"Lower leaflet points : {len(lower_leaflet)}")

# =============================================================================
# XY GRID
# =============================================================================

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

grid_x, grid_y = np.meshgrid(
    np.linspace(xmin, xmax, GRID_SIZE),
    np.linspace(ymin, ymax, GRID_SIZE)
)

# =============================================================================
# SURFACE INTERPOLATION
# =============================================================================

print("\nInterpolating membrane surfaces...")

upper_surface = griddata(
    (upper_leaflet[:, 0], upper_leaflet[:, 1]),
    upper_leaflet[:, 2],
    (grid_x, grid_y),
    method="cubic"
)

lower_surface = griddata(
    (lower_leaflet[:, 0], lower_leaflet[:, 1]),
    lower_leaflet[:, 2],
    (grid_x, grid_y),
    method="cubic"
)

# Fill NaNs
upper_surface[np.isnan(upper_surface)] = np.nanmean(upper_surface)
lower_surface[np.isnan(lower_surface)] = np.nanmean(lower_surface)

# Smooth surfaces
upper_surface = gaussian_filter(
    upper_surface,
    sigma=SMOOTH_SIGMA
)

lower_surface = gaussian_filter(
    lower_surface,
    sigma=SMOOTH_SIGMA
)

# =============================================================================
# THICKNESS MAP
# =============================================================================

thickness_map = upper_surface - lower_surface

print("\nMembrane Thickness Statistics")
print("-" * 40)
print(f"Mean thickness : {np.mean(thickness_map):.2f} Å")
print(f"Minimum        : {np.min(thickness_map):.2f} Å")
print(f"Maximum        : {np.max(thickness_map):.2f} Å")
print(f"Std deviation  : {np.std(thickness_map):.2f} Å")

# =============================================================================
# MEMBRANE CURVATURE APPROXIMATION
# =============================================================================

print("\nComputing curvature approximation...")

gx, gy = np.gradient(upper_surface)

gxx, gxy = np.gradient(gx)
gyx, gyy = np.gradient(gy)

curvature = gxx + gyy

curvature = gaussian_filter(curvature, sigma=1.0)

print(f"Curvature range : {curvature.min():.4f} "
      f"to {curvature.max():.4f}")

# =============================================================================
# PEPTIDE COMS
# =============================================================================

print("\nComputing peptide centers of mass...")

pep_coms = []

for residue in peptides.residues:

    com = residue.atoms.center_of_mass()

    pep_coms.append(com)

pep_coms = np.array(pep_coms)

print(f"Detected peptide residues : {len(pep_coms)}")

# =============================================================================
# INSERTION DEPTH ANALYSIS
# =============================================================================

midplane = (upper_surface + lower_surface) / 2

insertion_depths = []

for com in pep_coms:

    px, py, pz = com

    ix = np.abs(grid_x[0] - px).argmin()
    iy = np.abs(grid_y[:, 0] - py).argmin()

    local_mid = midplane[iy, ix]

    depth = pz - local_mid

    insertion_depths.append(depth)

insertion_depths = np.array(insertion_depths)

if len(insertion_depths) > 0:

    print("\nInsertion Depth Statistics")
    print("-" * 40)
    print(f"Mean depth : {np.mean(insertion_depths):.2f} Å")
    print(f"Min depth  : {np.min(insertion_depths):.2f} Å")
    print(f"Max depth  : {np.max(insertion_depths):.2f} Å")

# =============================================================================
# SOLVENT DOWNSAMPLING
# =============================================================================

solvent_coords = solvent.positions

if len(solvent_coords) > MAX_SOLVENT_POINTS:

    idx = np.random.choice(
        len(solvent_coords),
        MAX_SOLVENT_POINTS,
        replace=False
    )

    solvent_coords = solvent_coords[idx]

# =============================================================================
# FIGURE 1 — 3D TOPOLOGY
# =============================================================================

print("\nGenerating 3D topology figure...")

fig = plt.figure(figsize=(14, 11))

ax = fig.add_subplot(111, projection='3d')

# Lower leaflet
surf1 = ax.plot_surface(
    grid_x,
    grid_y,
    lower_surface,
    cmap="coolwarm",
    linewidth=0,
    antialiased=True,
    alpha=0.65
)

# Upper leaflet
surf2 = ax.plot_surface(
    grid_x,
    grid_y,
    upper_surface,
    cmap="viridis",
    linewidth=0,
    antialiased=True,
    alpha=0.55
)

# Peptides
if len(pep_coms) > 0:

    ax.scatter(
        pep_coms[:, 0],
        pep_coms[:, 1],
        pep_coms[:, 2],
        c='lime',
        s=35,
        edgecolors='black',
        alpha=0.95,
        label='Peptides'
    )

# Solvent cloud
ax.scatter(
    solvent_coords[:, 0],
    solvent_coords[:, 1],
    solvent_coords[:, 2],
    c='gray',
    s=1,
    alpha=0.05
)

# Colorbar
mappable = plt.cm.ScalarMappable(cmap='coolwarm')
mappable.set_array(thickness_map)

cbar = fig.colorbar(
    mappable,
    ax=ax,
    shrink=0.65,
    pad=0.08
)

cbar.set_label("Relative Membrane Thickness")

# Labels
ax.set_xlabel("X (Å)", fontsize=12)
ax.set_ylabel("Y (Å)", fontsize=12)
ax.set_zlabel("Z (Å)", fontsize=12)

ax.set_title(
    f"Membrane Topology & Peptide Insertion\n"
    f"Time = {time_ns:.2f} ns",
    fontsize=16,
    pad=20
)

ax.view_init(elev=18, azim=45)

ax.grid(False)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_PREFIX}_3D.png",
    dpi=500,
    bbox_inches='tight'
)

# =============================================================================
# FIGURE 2 — THICKNESS HEATMAP
# =============================================================================

print("Generating thickness heatmap...")

fig2, ax2 = plt.subplots(figsize=(10, 8))

im = ax2.imshow(
    thickness_map,
    cmap='coolwarm',
    origin='lower',
    extent=[xmin, xmax, ymin, ymax]
)

# Overlay peptide positions
if len(pep_coms) > 0:

    ax2.scatter(
        pep_coms[:, 0],
        pep_coms[:, 1],
        c='lime',
        edgecolors='black',
        s=60,
        label='Peptides'
    )

cbar2 = plt.colorbar(im)

cbar2.set_label("Membrane Thickness (Å)")

ax2.set_title(
    f"Membrane Thickness Map\n"
    f"Time = {time_ns:.2f} ns",
    fontsize=15
)

ax2.set_xlabel("X (Å)")
ax2.set_ylabel("Y (Å)")

ax2.legend()

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_PREFIX}_thickness_map.png",
    dpi=500,
    bbox_inches='tight'
)

# =============================================================================
# FIGURE 3 — CURVATURE MAP
# =============================================================================

print("Generating curvature map...")

fig3, ax3 = plt.subplots(figsize=(10, 8))

curv = ax3.imshow(
    curvature,
    cmap='RdBu_r',
    origin='lower',
    extent=[xmin, xmax, ymin, ymax]
)

if len(pep_coms) > 0:

    ax3.scatter(
        pep_coms[:, 0],
        pep_coms[:, 1],
        c='lime',
        edgecolors='black',
        s=60
    )

cbar3 = plt.colorbar(curv)

cbar3.set_label("Approximate Curvature")

ax3.set_title(
    f"Membrane Curvature Map\n"
    f"Time = {time_ns:.2f} ns",
    fontsize=15
)

ax3.set_xlabel("X (Å)")
ax3.set_ylabel("Y (Å)")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_PREFIX}_curvature_map.png",
    dpi=500,
    bbox_inches='tight'
)

# =============================================================================
# FINALIZE
# =============================================================================

print("\nSaved Figures:")
print("-" * 40)
print(f"{OUTPUT_PREFIX}_3D.png")
print(f"{OUTPUT_PREFIX}_thickness_map.png")
print(f"{OUTPUT_PREFIX}_curvature_map.png")

print("\nAnalysis complete.")

plt.show()
