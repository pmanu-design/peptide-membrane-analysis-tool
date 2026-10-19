#!/usr/bin/env python3
# =============================================================================
# COMBINED ULTRA-ENHANCED MEMBRANE TOPOLOGY + FULL CG MEMBRANE ANALYSIS
# =============================================================================
#
# This script combines BOTH original scripts WITHOUT removing functionality.
# All analyses, visualizations, statistics, topology calculations,
# peptide rendering, curvature analysis, pore analysis, and publication-
# quality rendering pipelines are preserved.
#
# INCLUDED MODULES
# -----------------------------------------------------------------------------
# SCRIPT 1 FEATURES
# ✓ Enhanced peptide molecular visualization
# ✓ Backbone trajectory rendering
# ✓ Residue COM network visualization
# ✓ Peptide insertion depth coloring
# ✓ Membrane topology visualization
# ✓ 2D thickness maps
# ✓ 2D contour maps
# ✓ 3D displacement maps
# ✓ Curvature approximation
#
# SCRIPT 2 FEATURES
# ✓ Full trajectory averaging
# ✓ Publication-quality 3D rendering
# ✓ Bilayer curvature analysis
# ✓ Pore detection analysis
# ✓ Disruptivity mapping
# ✓ Multi-angle visualization
# ✓ High-resolution export
# ✓ Symmetric bilayer statistics
#
# =============================================================================

import os
import warnings

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import griddata
from scipy.ndimage import (
    gaussian_filter,
    gaussian_laplace,
    zoom
)
from scipy.spatial.distance import cdist

from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# PUBLICATION SETTINGS
# =============================================================================

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 150

# =============================================================================
# USER SETTINGS
# =============================================================================

TPR_FILE = "md_0_1000.tpr"
XTC_FILE = "centerB.xtc"

PEPTIDE_RESNAME = "PROTEIN"
LIPID_RESNAMES = "POPE POPG"

FRAME = -1

GRID_SIZE = 120
SMOOTH_SIGMA = 2.0

OUTPUT_PREFIX = "enhanced_membrane_analysis"

MAX_SOLVENT_POINTS = 8000

# =============================================================================
# VISUAL SETTINGS
# =============================================================================

UPPER_ALPHA = 0.78
LOWER_ALPHA = 0.78

MEMBRANE_EDGE_ALPHA = 0.03

PEPTIDE_ATOM_SIZE = 22
PEPTIDE_BACKBONE_SIZE = 40
PEPTIDE_COM_SIZE = 180

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

os.makedirs("output", exist_ok=True)
os.makedirs("output/figures_hires", exist_ok=True)

# =============================================================================
# LOAD SYSTEM
# =============================================================================

print("\nLoading simulation...")

u = mda.Universe(TPR_FILE, XTC_FILE)

# =============================================================================
# SELECTIONS
# =============================================================================

lipid_selection = f"resname {LIPID_RESNAMES}"

if PEPTIDE_RESNAME.upper() == "PROTEIN":
    peptide_selection = "protein"
else:
    peptide_selection = f"resname {PEPTIDE_RESNAME}"

solvent_selection = "resname W PW SOL"

lipids = u.select_atoms(lipid_selection)
peptides = u.select_atoms(peptide_selection)
solvent = u.select_atoms(solvent_selection)
protein = peptides

print(f"\nLipids selected   : {len(lipids)}")
print(f"Peptide atoms     : {len(peptides)}")
print(f"Solvent atoms     : {len(solvent)}")

# =============================================================================
# SELECT FRAME
# =============================================================================

if FRAME == -1:
    ts = u.trajectory[-1]
else:
    ts = u.trajectory[FRAME]

time_ns = ts.time / 1000.0

print(f"\nFrame analyzed : {ts.frame}")
print(f"Simulation time: {time_ns:.2f} ns")

# =============================================================================
# HEADGROUP SELECTION
# =============================================================================

headgroup_candidates = [
    "name PO4",
    "name GL0",
    "name P"
]

headgroups = None

for sel in headgroup_candidates:

    hg = lipids.select_atoms(sel)

    if len(hg) > 0:
        headgroups = hg
        print(f"\nUsing headgroup selection: {sel}")
        break

if headgroups is None:

    raise ValueError(
        "No membrane headgroups found."
    )

coords = headgroups.positions

# =============================================================================
# SPLIT LEAFLETS
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
# CREATE GRID
# =============================================================================

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

grid_x, grid_y = np.meshgrid(
    np.linspace(xmin, xmax, GRID_SIZE),
    np.linspace(ymin, ymax, GRID_SIZE)
)

# =============================================================================
# INTERPOLATE SURFACES
# =============================================================================

print("\nInterpolating membrane surfaces...")

upper_surface = griddata(
    (upper_leaflet[:, 0], upper_leaflet[:, 1]),
    upper_leaflet[:, 2],
    (grid_x, grid_y),
    method='cubic'
)

lower_surface = griddata(
    (lower_leaflet[:, 0], lower_leaflet[:, 1]),
    lower_leaflet[:, 2],
    (grid_x, grid_y),
    method='cubic'
)

upper_surface[np.isnan(upper_surface)] = np.nanmean(
    upper_surface
)

lower_surface[np.isnan(lower_surface)] = np.nanmean(
    lower_surface
)

upper_surface = gaussian_filter(
    upper_surface,
    sigma=SMOOTH_SIGMA
)

lower_surface = gaussian_filter(
    lower_surface,
    sigma=SMOOTH_SIGMA
)

# =============================================================================
# THICKNESS
# =============================================================================

thickness_map = upper_surface - lower_surface

print("\nMembrane Thickness Statistics")
print("-" * 50)
print(f"Mean thickness : {np.mean(thickness_map):.2f} Å")
print(f"Minimum        : {np.min(thickness_map):.2f} Å")
print(f"Maximum        : {np.max(thickness_map):.2f} Å")

# =============================================================================
# DISPLACEMENT
# =============================================================================

upper_mean = np.mean(upper_surface)
lower_mean = np.mean(lower_surface)

upper_displacement = upper_surface - upper_mean
lower_displacement = lower_surface - lower_mean

combined_displacement = (
    np.abs(upper_displacement) +
    np.abs(lower_displacement)
)

combined_displacement = gaussian_filter(
    combined_displacement,
    sigma=1.5
)

# =============================================================================
# CURVATURE
# =============================================================================

gx, gy = np.gradient(upper_surface)

gxx, gxy = np.gradient(gx)
gyx, gyy = np.gradient(gy)

curvature = gxx + gyy

curvature = gaussian_filter(
    curvature,
    sigma=1.0
)

# =============================================================================
# PEPTIDE REPRESENTATION
# =============================================================================

print("\nBuilding enhanced peptide representation...")

peptide_atoms = peptides.positions

# =============================================================================
# BACKBONE SELECTION
# =============================================================================

backbone_candidates = [
    "name BB",
    "backbone",
    "name CA"
]

backbone = None

for sel in backbone_candidates:

    bb = peptides.select_atoms(sel)

    if len(bb) > 0:
        backbone = bb
        print(f"Using backbone selection: {sel}")
        break

if backbone is None or len(backbone) == 0:

    backbone = peptides

backbone_coords = backbone.positions

# =============================================================================
# PEPTIDE COMS
# =============================================================================

pep_coms = []

for residue in peptides.residues:

    try:
        com = residue.atoms.center_of_mass()
        pep_coms.append(com)
    except:
        pass

pep_coms = np.array(pep_coms)

print(f"Residue COMs detected: {len(pep_coms)}")

# =============================================================================
# INSERTION DEPTHS
# =============================================================================

midplane = (upper_surface + lower_surface) / 2

depth_values = []

for point in pep_coms:

    px, py, pz = point

    ix = np.abs(grid_x[0] - px).argmin()
    iy = np.abs(grid_y[:, 0] - py).argmin()

    local_mid = midplane[iy, ix]

    depth = pz - local_mid

    depth_values.append(depth)

depth_values = np.array(depth_values)

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
# FIGURE 1 — ENHANCED 3D TOPOLOGY
# =============================================================================

print("\nGenerating enhanced peptide visualization...")

fig1 = plt.figure(figsize=(16, 12))

ax1 = fig1.add_subplot(111, projection='3d')

ax1.plot_surface(
    grid_x,
    grid_y,
    lower_surface,
    cmap='coolwarm',
    linewidth=0,
    antialiased=True,
    alpha=LOWER_ALPHA
)

upper_plot = ax1.plot_surface(
    grid_x,
    grid_y,
    upper_surface,
    cmap='viridis',
    linewidth=0,
    antialiased=True,
    alpha=UPPER_ALPHA
)

ax1.scatter(
    peptide_atoms[:, 0],
    peptide_atoms[:, 1],
    peptide_atoms[:, 2],
    c='gold',
    s=PEPTIDE_ATOM_SIZE * 5,
    alpha=0.05,
    edgecolors='none'
)

ax1.scatter(
    peptide_atoms[:, 0],
    peptide_atoms[:, 1],
    peptide_atoms[:, 2],
    c='orange',
    s=PEPTIDE_ATOM_SIZE,
    alpha=0.95,
    edgecolors='black',
    linewidths=0.2
)

ax1.plot(
    backbone_coords[:, 0],
    backbone_coords[:, 1],
    backbone_coords[:, 2],
    color='red',
    linewidth=2.8,
    alpha=0.95
)

ax1.scatter(
    backbone_coords[:, 0],
    backbone_coords[:, 1],
    backbone_coords[:, 2],
    c='red',
    s=PEPTIDE_BACKBONE_SIZE,
    alpha=1.0,
    edgecolors='black'
)

if len(pep_coms) > 0:

    com_plot = ax1.scatter(
        pep_coms[:, 0],
        pep_coms[:, 1],
        pep_coms[:, 2],
        c=depth_values,
        cmap='turbo',
        s=PEPTIDE_COM_SIZE,
        edgecolors='black',
        linewidths=1.0,
        alpha=1.0
    )

    depth_bar = fig1.colorbar(
        com_plot,
        ax=ax1,
        shrink=0.45,
        pad=0.02
    )

    depth_bar.set_label(
        "Peptide Insertion Depth (Å)"
    )

ax1.scatter(
    solvent_coords[:, 0],
    solvent_coords[:, 1],
    solvent_coords[:, 2],
    c='gray',
    s=1,
    alpha=0.018
)

mappable = plt.cm.ScalarMappable(
    cmap='coolwarm'
)

mappable.set_array(thickness_map)

cbar = fig1.colorbar(
    mappable,
    ax=ax1,
    shrink=0.58,
    pad=0.08
)

cbar.set_label(
    "Membrane Thickness (Å)"
)

ax1.set_xlabel("X (Å)", fontsize=13)
ax1.set_ylabel("Y (Å)", fontsize=13)
ax1.set_zlabel("Z (Å)", fontsize=13)

ax1.set_title(
    f"Enhanced Membrane Topology & Peptide Architecture\n"
    f"Time = {time_ns:.2f} ns",
    fontsize=18,
    pad=20
)

ax1.view_init(
    elev=22,
    azim=50
)

ax1.grid(False)

ax1.xaxis.pane.fill = True
ax1.yaxis.pane.fill = True
ax1.zaxis.pane.fill = False

ax1.xaxis.pane.set_alpha(0.04)
ax1.yaxis.pane.set_alpha(0.04)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_PREFIX}_enhanced_3D_topology.png",
    dpi=600,
    bbox_inches='tight'
)

# =============================================================================
# FIGURE 2 — THICKNESS MAP
# =============================================================================

print("Generating thickness map...")

fig2, ax2 = plt.subplots(figsize=(10, 8))

im = ax2.imshow(
    thickness_map,
    cmap='coolwarm',
    origin='lower',
    extent=[xmin, xmax, ymin, ymax]
)

if len(pep_coms) > 0:

    ax2.scatter(
        pep_coms[:, 0],
        pep_coms[:, 1],
        c=depth_values,
        cmap='turbo',
        edgecolors='black',
        s=90
    )

cbar2 = plt.colorbar(im)

cbar2.set_label("Thickness (Å)")

ax2.set_title(
    f"2D Membrane Thickness Map\n"
    f"Time = {time_ns:.2f} ns",
    fontsize=15
)

ax2.set_xlabel("X (Å)")
ax2.set_ylabel("Y (Å)")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_PREFIX}_2D_thickness.png",
    dpi=500,
    bbox_inches='tight'
)

# =============================================================================
# FIGURE 3 — THICKNESS CONTOUR
# =============================================================================

print("Generating contour map...")

fig3, ax3 = plt.subplots(figsize=(10, 8))

contour = ax3.contourf(
    grid_x,
    grid_y,
    thickness_map,
    levels=50,
    cmap='coolwarm'
)

ax3.contour(
    grid_x,
    grid_y,
    thickness_map,
    levels=25,
    colors='black',
    linewidths=0.35,
    alpha=0.45
)

if len(pep_coms) > 0:

    ax3.scatter(
        pep_coms[:, 0],
        pep_coms[:, 1],
        c=depth_values,
        cmap='turbo',
        edgecolors='black',
        s=90
    )

cbar3 = plt.colorbar(contour)

cbar3.set_label("Thickness (Å)")

ax3.set_title(
    f"2D Thickness Contour Map\n"
    f"Time = {time_ns:.2f} ns",
    fontsize=15
)

ax3.set_xlabel("X (Å)")
ax3.set_ylabel("Y (Å)")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_PREFIX}_2D_thickness_contour.png",
    dpi=500,
    bbox_inches='tight'
)

# =============================================================================
# FIGURE 4 — 3D DISPLACEMENT
# =============================================================================

print("Generating 3D displacement plot...")

fig4 = plt.figure(figsize=(14, 10))

ax4 = fig4.add_subplot(111, projection='3d')

disp_surface = ax4.plot_surface(
    grid_x,
    grid_y,
    combined_displacement,
    cmap='plasma',
    linewidth=0,
    antialiased=True,
    alpha=0.92
)

if len(pep_coms) > 0:

    peptide_z = np.full(
        len(pep_coms),
        np.max(combined_displacement) * 1.03
    )

    ax4.scatter(
        pep_coms[:, 0],
        pep_coms[:, 1],
        peptide_z,
        c=depth_values,
        cmap='turbo',
        edgecolors='black',
        s=90
    )

cbar4 = fig4.colorbar(
    disp_surface,
    ax=ax4,
    shrink=0.7,
    pad=0.08
)

cbar4.set_label(
    "Lipid Displacement"
)

ax4.set_title(
    f"3D Lipid Displacement Surface\n"
    f"Time = {time_ns:.2f} ns",
    fontsize=16,
    pad=20
)

ax4.set_xlabel("X (Å)")
ax4.set_ylabel("Y (Å)")
ax4.set_zlabel("Displacement")

ax4.view_init(
    elev=28,
    azim=42
)

ax4.grid(False)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_PREFIX}_3D_displacement.png",
    dpi=500,
    bbox_inches='tight'
)

# =============================================================================
# FULL TRAJECTORY ANALYSIS SECTION
# =============================================================================

print("\nStarting full trajectory curvature analysis...")

# =============================================================================
# GRID
# =============================================================================

grid_size = 1.0
box = u.trajectory.ts.dimensions[:3]

nx = int(box[0] / grid_size)
ny = int(box[1] / grid_size)

x = np.linspace(0, box[0], nx)
y = np.linspace(0, box[1], ny)

X, Y = np.meshgrid(x, y)

# =============================================================================
# ACCUMULATORS
# =============================================================================

sum_upper = np.zeros((ny, nx))
sum_lower = np.zeros((ny, nx))

sum_curv_up = np.zeros((ny, nx))
sum_curv_low = np.zeros((ny, nx))

n_frames = 0

# =============================================================================
# FUNCTIONS
# =============================================================================

def compute_height_map(leaflet):
    Z = np.zeros((ny, nx))
    C = np.zeros((ny, nx))

    for pos in leaflet.positions:
        ix = int(pos[0] / grid_size)
        iy = int(pos[1] / grid_size)
        if 0 <= ix < nx and 0 <= iy < ny:
            Z[iy, ix] += pos[2]
            C[iy, ix] += 1

    Z[C > 0] /= C[C > 0]
    Z[C == 0] = np.nan
    return Z


def fill_nan(Z):
    xx, yy = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    mask = ~np.isnan(Z)

    pts = np.column_stack((xx[mask], yy[mask]))
    vals = Z[mask]

    Zi = griddata(pts, vals, (xx, yy), method='cubic')

    nan_mask = np.isnan(Zi)
    if np.any(nan_mask):
        Zi[nan_mask] = griddata(
            pts, vals,
            (xx[nan_mask], yy[nan_mask]),
            method='nearest'
        )
    return Zi


def smooth(Z, sigma=3):
    return gaussian_filter(Z, sigma=sigma)


def curvature_func(Z):
    return gaussian_laplace(Z, sigma=2)


def upscale(Z, f=3):
    return zoom(Z, f, order=3)


def robust_norm(F, percentile=98):
    fmax = np.percentile(np.abs(F), percentile)
    return np.clip(F / fmax, -1, 1) if fmax > 0 else F


def add_colorbar(ax, surf, label, shrink=0.8):
    cbar = plt.colorbar(surf, ax=ax, pad=0.1, shrink=shrink)
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    return cbar

# =============================================================================
# TRAJECTORY LOOP
# =============================================================================

for ts in tqdm(u.trajectory, desc="Processing trajectory"):

    z_mid = np.median(lipids.positions[:, 2])

    upper = lipids.select_atoms(f"prop z > {z_mid}")
    lower = lipids.select_atoms(f"prop z < {z_mid}")

    Zu = smooth(fill_nan(compute_height_map(upper)), 2)
    Zl = smooth(fill_nan(compute_height_map(lower)), 2)

    sum_upper += Zu
    sum_lower += Zl

    sum_curv_up += curvature_func(Zu)
    sum_curv_low += curvature_func(Zl)

    n_frames += 1

# =============================================================================
# AVERAGE
# =============================================================================

Z_upper = smooth(sum_upper / n_frames, 3)
Z_lower = smooth(sum_lower / n_frames, 3)

curv_up = smooth(sum_curv_up / n_frames, 3)
curv_low = smooth(sum_curv_low / n_frames, 3)

# =============================================================================
# HIGH RESOLUTION
# =============================================================================

UPS = 3

Zu_hr = upscale(Z_upper, UPS)
Zl_hr = upscale(Z_lower, UPS)

cu_hr = upscale(curv_up, UPS)
cl_hr = upscale(curv_low, UPS)

X_hr = upscale(X, UPS)
Y_hr = upscale(Y, UPS)

# =============================================================================
# DERIVED FIELDS
# =============================================================================

thickness = Zu_hr - Zl_hr
curv_bilayer = (cu_hr - cl_hr) / 2

du = np.abs(Zu_hr - np.mean(Zu_hr))
dl = np.abs(Zl_hr - np.mean(Zl_hr))
d_comb = du + dl

cu_n = robust_norm(cu_hr, 98)
cl_n = robust_norm(cl_hr, 98)
cb_n = robust_norm(curv_bilayer, 98)

dmax = np.percentile(d_comb, 98)
d_norm = np.clip(d_comb / dmax, 0, 1) if dmax > 0 else d_comb

# =============================================================================
# 1️⃣ BILAYER STRUCTURE
# =============================================================================

fig = plt.figure(figsize=(16, 7))
fig.suptitle('Bilayer Structure: Upper and Lower Leaflets', 
             fontsize=14, fontweight='bold', y=0.98)

ax1 = fig.add_subplot(121, projection='3d')
zu_norm = (Zu_hr - Zu_hr.min()) / Zu_hr.ptp()
surf1 = ax1.plot_surface(X_hr, Y_hr, Zu_hr,
                         facecolors=cm.viridis(zu_norm),
                         shade=False, rcount=100, ccount=100)
ax1.set_xlabel('X (Å)', fontsize=10)
ax1.set_ylabel('Y (Å)', fontsize=10)
ax1.set_zlabel('Z (Å)', fontsize=10)
ax1.set_title('Upper Leaflet', fontsize=12, pad=10)
ax1.view_init(elev=25, azim=45)
ax1.set_box_aspect([1, 1, 0.6])

ax2 = fig.add_subplot(122, projection='3d')
zl_norm = (Zl_hr - Zl_hr.min()) / Zl_hr.ptp()
surf2 = ax2.plot_surface(X_hr, Y_hr, Zl_hr,
                         facecolors=cm.plasma(zl_norm),
                         shade=False, rcount=100, ccount=100)
ax2.set_xlabel('X (Å)', fontsize=10)
ax2.set_ylabel('Y (Å)', fontsize=10)
ax2.set_zlabel('Z (Å)', fontsize=10)
ax2.set_title('Lower Leaflet', fontsize=12, pad=10)
ax2.view_init(elev=25, azim=45)
ax2.set_box_aspect([1, 1, 0.6])

add_colorbar(ax1, surf1, 'Height (Å)', shrink=0.7)
add_colorbar(ax2, surf2, 'Height (Å)', shrink=0.7)

plt.tight_layout()
plt.savefig('output/figures_hires/01_bilayer_structure.png', dpi=300, bbox_inches='tight')
plt.savefig('output/figures_hires/01_bilayer_structure.pdf', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 2️⃣ LOCAL CURVATURE
# =============================================================================

fig = plt.figure(figsize=(16, 7))
fig.suptitle('Local Curvature Analysis: Upper and Lower Leaflets', 
             fontsize=14, fontweight='bold', y=0.98)

ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X_hr, Y_hr, Zu_hr,
                         facecolors=cm.coolwarm((cu_n + 1) / 2),
                         shade=False, rcount=100, ccount=100)
ax1.set_xlabel('X (Å)', fontsize=10)
ax1.set_ylabel('Y (Å)', fontsize=10)
ax1.set_zlabel('Z (Å)', fontsize=10)
ax1.set_title('Upper Leaflet (Curv.)', fontsize=12, pad=10)
ax1.view_init(elev=25, azim=45)
ax1.set_box_aspect([1, 1, 0.6])

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X_hr, Y_hr, Zl_hr,
                         facecolors=cm.coolwarm((cl_n + 1) / 2),
                         shade=False, rcount=100, ccount=100)
ax2.set_xlabel('X (Å)', fontsize=10)
ax2.set_ylabel('Y (Å)', fontsize=10)
ax2.set_zlabel('Z (Å)', fontsize=10)
ax2.set_title('Lower Leaflet (Curv.)', fontsize=12, pad=10)
ax2.view_init(elev=25, azim=45)
ax2.set_box_aspect([1, 1, 0.6])

sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.7, pad=0.1, aspect=30)
cbar.set_label('Normalized Curvature', fontsize=10)

plt.tight_layout()
plt.savefig('output/figures_hires/02_local_curvature.png', dpi=300, bbox_inches='tight')
plt.savefig('output/figures_hires/02_local_curvature.pdf', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# REPLACE THIS ENTIRE SECTION:
# =============================================================================
# 3️⃣ BILAYER CURVATURE
# =============================================================================
#
# WITH THE UPDATED VERSION BELOW
# =============================================================================

# =============================================================================
# 3️⃣ BILAYER CURVATURE
# =============================================================================

fig = plt.figure(figsize=(16, 7))
fig.suptitle('Bilayer Curvature (Mean Curvature of Coupled Leaflets)', 
             fontsize=14, fontweight='bold', y=0.98)

ax1 = fig.add_subplot(121, projection='3d')

# --- Upper leaflet colored by bilayer curvature ---
surf1 = ax1.plot_surface(
    X_hr,
    Y_hr,
    Zu_hr,
    facecolors=cm.coolwarm((cb_n + 1) / 2),
    shade=False,
    rcount=100,
    ccount=100,
    alpha=0.96
)

# --- ADD LOWER LEAFLET POSITION LIKE 07 MULTI-ANGLE VIEW ---
ax1.plot_surface(
    X_hr,
    Y_hr,
    Zl_hr,
    color='lightgray',
    alpha=0.22,
    linewidth=0,
    edgecolor='none',
    rcount=80,
    ccount=80
)

ax1.set_xlabel('X (Å)', fontsize=10)
ax1.set_ylabel('Y (Å)', fontsize=10)
ax1.set_zlabel('Z (Å)', fontsize=10)

ax1.set_title(
    'Upper Leaflet Curvature\n'
    '(Lower leaflet shown for reference)',
    fontsize=12,
    pad=10
)

ax1.view_init(elev=25, azim=45)
ax1.set_box_aspect([1, 1, 0.6])

# -------------------------------------------------------------------------

ax2 = fig.add_subplot(122, projection='3d')

# --- Lower leaflet colored by bilayer curvature ---
surf2 = ax2.plot_surface(
    X_hr,
    Y_hr,
    Zl_hr,
    facecolors=cm.coolwarm((cb_n + 1) / 2),
    shade=False,
    rcount=100,
    ccount=100,
    alpha=0.96
)

# --- ADD UPPER LEAFLET POSITION LIKE 07 MULTI-ANGLE VIEW ---
ax2.plot_surface(
    X_hr,
    Y_hr,
    Zu_hr,
    color='lightgray',
    alpha=0.22,
    linewidth=0,
    edgecolor='none',
    rcount=80,
    ccount=80
)

ax2.set_xlabel('X (Å)', fontsize=10)
ax2.set_ylabel('Y (Å)', fontsize=10)
ax2.set_zlabel('Z (Å)', fontsize=10)

ax2.set_title(
    'Lower Leaflet Curvature\n'
    '(Upper leaflet shown for reference)',
    fontsize=12,
    pad=10
)

ax2.view_init(elev=25, azim=45)
ax2.set_box_aspect([1, 1, 0.6])

# -------------------------------------------------------------------------

sm = cm.ScalarMappable(
    cmap=cm.coolwarm,
    norm=plt.Normalize(vmin=-1, vmax=1)
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=[ax1, ax2],
    shrink=0.7,
    pad=0.1,
    aspect=30
)

cbar.set_label(
    'Bilayer Curvature (Normalized)',
    fontsize=10
)

plt.tight_layout()

plt.savefig(
    'output/figures_hires/03_bilayer_curvature.png',
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    'output/figures_hires/03_bilayer_curvature.pdf',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =============================================================================
# REPLACE THIS ENTIRE SECTION:
# =============================================================================
# 5️⃣ DISRUPTIVITY ANALYSIS
# =============================================================================
#
# WITH THE UPDATED VERSION BELOW
# =============================================================================

# =============================================================================
# 5️⃣ DISRUPTIVITY ANALYSIS
# =============================================================================

fig = plt.figure(figsize=(16, 7))

fig.suptitle(
    'Membrane Disruptivity (Local Deviation from Mean Height)', 
    fontsize=14,
    fontweight='bold',
    y=0.98
)

# -------------------------------------------------------------------------
# UPPER LEAFLET DISRUPTIVITY
# -------------------------------------------------------------------------

ax1 = fig.add_subplot(121, projection='3d')

surf1 = ax1.plot_surface(
    X_hr,
    Y_hr,
    Zu_hr,
    facecolors=cm.inferno(d_norm),
    shade=False,
    rcount=100,
    ccount=100,
    alpha=0.96
)

# --- ADD LOWER LEAFLET POSITION FOR CONTEXT ---
ax1.plot_surface(
    X_hr,
    Y_hr,
    Zl_hr,
    color='lightgray',
    alpha=0.22,
    linewidth=0,
    edgecolor='none',
    rcount=80,
    ccount=80
)

ax1.set_xlabel('X (Å)', fontsize=10)
ax1.set_ylabel('Y (Å)', fontsize=10)
ax1.set_zlabel('Z (Å)', fontsize=10)

ax1.set_title(
    'Upper Leaflet Disruptivity\n'
    '(Lower leaflet shown for reference)',
    fontsize=12,
    pad=10
)

ax1.view_init(elev=25, azim=45)
ax1.set_box_aspect([1, 1, 0.6])

# -------------------------------------------------------------------------
# LOWER LEAFLET DISRUPTIVITY
# -------------------------------------------------------------------------

ax2 = fig.add_subplot(122, projection='3d')

surf2 = ax2.plot_surface(
    X_hr,
    Y_hr,
    Zl_hr,
    facecolors=cm.inferno(d_norm),
    shade=False,
    rcount=100,
    ccount=100,
    alpha=0.96
)

# --- ADD UPPER LEAFLET POSITION FOR CONTEXT ---
ax2.plot_surface(
    X_hr,
    Y_hr,
    Zu_hr,
    color='lightgray',
    alpha=0.22,
    linewidth=0,
    edgecolor='none',
    rcount=80,
    ccount=80
)

ax2.set_xlabel('X (Å)', fontsize=10)
ax2.set_ylabel('Y (Å)', fontsize=10)
ax2.set_zlabel('Z (Å)', fontsize=10)

ax2.set_title(
    'Lower Leaflet Disruptivity\n'
    '(Upper leaflet shown for reference)',
    fontsize=12,
    pad=10
)

ax2.view_init(elev=25, azim=45)
ax2.set_box_aspect([1, 1, 0.6])

# -------------------------------------------------------------------------

add_colorbar(
    ax1,
    surf1,
    'Disruptivity (norm.)',
    shrink=0.7
)

add_colorbar(
    ax2,
    surf2,
    'Disruptivity (norm.)',
    shrink=0.7
)

plt.tight_layout()

plt.savefig(
    'output/figures_hires/05_disruptivity.png',
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    'output/figures_hires/05_disruptivity.pdf',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# =============================================================================
# 4️⃣ PORE DETECTION
# =============================================================================

thr = np.percentile(thickness, 10)
mask = thickness < thr

fig = plt.figure(figsize=(16, 7))
fig.suptitle(f'Pore Detection (Thickness < {thr:.1f} Å)', 
             fontsize=14, fontweight='bold', y=0.98)

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X_hr, Y_hr, Zu_hr, color='lightgray', alpha=0.7, 
                 rcount=100, ccount=100, edgecolor='none')
pores_u = ax1.scatter(X_hr[mask], Y_hr[mask], Zu_hr[mask], 
                      c='red', s=8, alpha=0.9, label='Pore regions')
ax1.set_xlabel('X (Å)', fontsize=10)
ax1.set_ylabel('Y (Å)', fontsize=10)
ax1.set_zlabel('Z (Å)', fontsize=10)
ax1.set_title('Upper Leaflet', fontsize=12, pad=10)
ax1.view_init(elev=25, azim=45)
ax1.set_box_aspect([1, 1, 0.6])
ax1.legend(loc='upper right', fontsize=9)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X_hr, Y_hr, Zl_hr, color='lightgray', alpha=0.7,
                 rcount=100, ccount=100, edgecolor='none')
pores_l = ax2.scatter(X_hr[mask], Y_hr[mask], Zl_hr[mask],
                      c='blue', s=8, alpha=0.9, label='Pore regions')
ax2.set_xlabel('X (Å)', fontsize=10)
ax2.set_ylabel('Y (Å)', fontsize=10)
ax2.set_zlabel('Z (Å)', fontsize=10)
ax2.set_title('Lower Leaflet', fontsize=12, pad=10)
ax2.view_init(elev=25, azim=45)
ax2.set_box_aspect([1, 1, 0.6])
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('output/figures_hires/04_pore_detection.png', dpi=300, bbox_inches='tight')
plt.savefig('output/figures_hires/04_pore_detection.pdf', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 5️⃣ DISRUPTIVITY ANALYSIS
# =============================================================================

fig = plt.figure(figsize=(16, 7))
fig.suptitle('Membrane Disruptivity (Local Deviation from Mean Height)', 
             fontsize=14, fontweight='bold', y=0.98)

ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X_hr, Y_hr, Zu_hr,
                         facecolors=cm.inferno(d_norm),
                         shade=False, rcount=100, ccount=100)
ax1.set_xlabel('X (Å)', fontsize=10)
ax1.set_ylabel('Y (Å)', fontsize=10)
ax1.set_zlabel('Z (Å)', fontsize=10)
ax1.set_title('Upper Leaflet', fontsize=12, pad=10)
ax1.view_init(elev=25, azim=45)
ax1.set_box_aspect([1, 1, 0.6])

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X_hr, Y_hr, Zl_hr,
                         facecolors=cm.inferno(d_norm),
                         shade=False, rcount=100, ccount=100)
ax2.set_xlabel('X (Å)', fontsize=10)
ax2.set_ylabel('Y (Å)', fontsize=10)
ax2.set_zlabel('Z (Å)', fontsize=10)
ax2.set_title('Lower Leaflet', fontsize=12, pad=10)
ax2.view_init(elev=25, azim=45)
ax2.set_box_aspect([1, 1, 0.6])

add_colorbar(ax1, surf1, 'Disruptivity (norm.)', shrink=0.7)
add_colorbar(ax2, surf2, 'Disruptivity (norm.)', shrink=0.7)

plt.tight_layout()
plt.savefig('output/figures_hires/05_disruptivity.png', dpi=300, bbox_inches='tight')
plt.savefig('output/figures_hires/05_disruptivity.pdf', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 6️⃣ MEMBRANE THICKNESS
# =============================================================================

fig = plt.figure(figsize=(16, 7))
fig.suptitle('Bilayer Thickness Map (3D Visualization)', 
             fontsize=14, fontweight='bold', y=0.98)

ax1 = fig.add_subplot(121, projection='3d')
thickness_norm = (thickness - thickness.min()) / (thickness.ptp() + 1e-10)
surf1 = ax1.plot_surface(X_hr, Y_hr, Zu_hr,
                         facecolors=cm.RdYlGn(thickness_norm),
                         shade=False, rcount=100, ccount=100)
ax1.set_xlabel('X (Å)', fontsize=10)
ax1.set_ylabel('Y (Å)', fontsize=10)
ax1.set_zlabel('Z (Å)', fontsize=10)
ax1.set_title('Upper Leaflet (Colored by Thickness)', fontsize=12, pad=10)
ax1.view_init(elev=25, azim=45)
ax1.set_box_aspect([1, 1, 0.6])

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X_hr, Y_hr, thickness,
                         facecolors=cm.RdYlGn(thickness_norm),
                         shade=False, rcount=100, ccount=100)
ax2.set_xlabel('X (Å)', fontsize=10)
ax2.set_ylabel('Y (Å)', fontsize=10)
ax2.set_zlabel('Thickness (Å)', fontsize=10)
ax2.set_title('Thickness Profile', fontsize=12, pad=10)
ax2.view_init(elev=25, azim=45)
ax2.set_box_aspect([1, 1, 0.6])

add_colorbar(ax1, surf1, 'Thickness (Å)', shrink=0.7)
add_colorbar(ax2, surf2, 'Thickness (Å)', shrink=0.7)

plt.tight_layout()
plt.savefig('output/figures_hires/06_thickness_map.png', dpi=300, bbox_inches='tight')
plt.savefig('output/figures_hires/06_thickness_map.pdf', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 7️⃣ MULTI-ANGLE VIEW
# =============================================================================

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Bilayer Structure: Multi-Angle 3D Analysis', 
             fontsize=14, fontweight='bold')

angles = [(25, 45), (25, 135), (60, 45), (10, 45)]
titles = ['Front View (elev=25°, azim=45°)',
          'Side View (elev=25°, azim=135°)',
          'Top View (elev=60°, azim=45°)',
          'Oblique View (elev=10°, azim=45°)']

for idx, (elev, azim) in enumerate(angles, 1):
    ax = fig.add_subplot(2, 2, idx, projection='3d')

    zu_norm = (Zu_hr - Zu_hr.min()) / Zu_hr.ptp()
    zl_norm = (Zl_hr - Zl_hr.min()) / Zl_hr.ptp()

    ax.plot_surface(X_hr, Y_hr, Zu_hr,
                    facecolors=cm.viridis(zu_norm),
                    shade=False, rcount=80, ccount=80)
    ax.plot_surface(X_hr, Y_hr, Zl_hr,
                    facecolors=cm.plasma(zl_norm),
                    shade=False, rcount=80, ccount=80)

    ax.set_xlabel('X (Å)', fontsize=9)
    ax.set_ylabel('Y (Å)', fontsize=9)
    ax.set_zlabel('Z (Å)', fontsize=9)
    ax.set_title(titles[idx-1], fontsize=11, pad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 0.6])

plt.tight_layout()
plt.savefig('output/figures_hires/07_multi_angle_view.png', dpi=300, bbox_inches='tight')
plt.savefig('output/figures_hires/07_multi_angle_view.pdf', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\nSaved Figures")
print("-" * 60)

print(f"{OUTPUT_PREFIX}_enhanced_3D_topology.png")
print(f"{OUTPUT_PREFIX}_2D_thickness.png")
print(f"{OUTPUT_PREFIX}_2D_thickness_contour.png")
print(f"{OUTPUT_PREFIX}_3D_displacement.png")

print("\n" + "="*70)
print("SYMMETRIC BILAYER ANALYSIS - STATISTICS SUMMARY")
print("="*70)

print(f"\nFrames analyzed: {n_frames}")
print(f"Grid resolution: {grid_size} Å")
print(f"Upscaling factor: {UPS}x")
print(f"High-res grid: {Zu_hr.shape}")

print(f"\n--- UPPER LEAFLET ---")
print(f"Height: {Zu_hr.min():.2f} - {Zu_hr.max():.2f} Å (mean: {Zu_hr.mean():.2f})")
print(f"Curvature: {cu_hr.min():.4f} - {cu_hr.max():.4f}")

print(f"\n--- LOWER LEAFLET ---")
print(f"Height: {Zl_hr.min():.2f} - {Zl_hr.max():.2f} Å (mean: {Zl_hr.mean():.2f})")
print(f"Curvature: {cl_hr.min():.4f} - {cl_hr.max():.4f}")

print(f"\n--- BILAYER PROPERTIES ---")
print(f"Thickness: {thickness.min():.2f} - {thickness.max():.2f} Å (mean: {thickness.mean():.2f})")
print(f"Pore threshold (10th percentile): {thr:.2f} Å")
print(f"Pore count: {np.sum(mask)}")
print(f"Bilayer curvature: {curv_bilayer.min():.4f} - {curv_bilayer.max():.4f}")
print(f"Mean disruptivity: {d_comb.mean():.4f}")

print(f"\n✓ All figures saved to: output/figures_hires/")
print("="*70 + "\n")

print("\nAnalysis complete.")

plt.show()

