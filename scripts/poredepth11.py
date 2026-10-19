#!/usr/bin/env python3

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# USER INPUT
# ==============================

TPR = "md_0_1000.tpr"
XTC = "center1.xtc"

LIPID_RESNAMES = ["POPE", "POPG"]
WATER_RESNAME = "W"

temperature = 310  # Kelvin

bins_x = 80
bins_y = 80
bins_z = 80

smooth_sigma = 1.5

# ==============================
# CONSTANTS
# ==============================

kB = 0.0083144621  # kJ/mol/K
beta = 1 / (kB * temperature)

# ==============================
# LOAD SYSTEM
# ==============================

print("Loading trajectory...")

u = mda.Universe(TPR, XTC)

lipids = u.select_atoms("resname " + " ".join(LIPID_RESNAMES))
waters = u.select_atoms(f"resname {WATER_RESNAME}")

# ==============================
# COLLECT COORDINATES
# ==============================

print("Collecting water coordinates...")

x_all = []
y_all = []
z_all = []

for ts in u.trajectory:

    pos = waters.positions

    x_all.extend(pos[:,0])
    y_all.extend(pos[:,1])
    z_all.extend(pos[:,2])

x_all = np.array(x_all)
y_all = np.array(y_all)
z_all = np.array(z_all)

# ==============================
# HISTOGRAMS
# ==============================

print("Building density maps...")

H_xz, xedges, zedges = np.histogram2d(
    x_all, z_all, bins=[bins_x, bins_z], density=True
)

H_yz, yedges, zedges2 = np.histogram2d(
    y_all, z_all, bins=[bins_y, bins_z], density=True
)

H_xy, xedges2, yedges2 = np.histogram2d(
    x_all, y_all, bins=[bins_x, bins_y], density=True
)

# transpose for plotting
H_xz = H_xz.T
H_yz = H_yz.T
H_xy = H_xy.T

# ==============================
# SMOOTHING
# ==============================

H_xz = gaussian_filter(H_xz, sigma=smooth_sigma)
H_yz = gaussian_filter(H_yz, sigma=smooth_sigma)
H_xy = gaussian_filter(H_xy, sigma=smooth_sigma)

# ==============================
# FREE ENERGY
# ==============================

print("Converting density to free energy...")

def density_to_FE(H):

    rho = H / np.max(H)
    rho[rho == 0] = 1e-12
    F = -(1/beta) * np.log(rho)

    return F

F_xz = density_to_FE(H_xz)
F_yz = density_to_FE(H_yz)
F_xy = density_to_FE(H_xy)

# ==============================
# GRIDS
# ==============================

xcent = 0.5 * (xedges[:-1] + xedges[1:])
zcent = 0.5 * (zedges[:-1] + zedges[1:])
ycent = 0.5 * (yedges[:-1] + yedges[1:])

X_xz, Z_xz = np.meshgrid(xcent, zcent)
Y_yz, Z_yz = np.meshgrid(ycent, zcent)
X_xy, Y_xy = np.meshgrid(xcent, ycent)

# ==============================
# SIDE VIEW (XZ)
# ==============================

print("Plotting XZ side view...")

plt.figure(figsize=(6,8))

levels = np.linspace(np.min(F_xz), np.max(F_xz), 20)

cont = plt.contourf(
    X_xz,
    Z_xz,
    F_xz,
    levels=levels,
    cmap="jet"
)

plt.xlabel("X (Å)")
plt.ylabel("Z (Å)")
plt.title("Membrane Pore Free Energy Map (Side View XZ)")

cbar = plt.colorbar(cont)
cbar.set_label("Free Energy (kJ/mol)")

plt.tight_layout()
plt.savefig("pore_FES_side_XZ.png", dpi=300)

# ==============================
# FRONT VIEW (YZ)
# ==============================

print("Plotting YZ front view...")

plt.figure(figsize=(6,8))

levels = np.linspace(np.min(F_yz), np.max(F_yz), 20)

cont = plt.contourf(
    Y_yz,
    Z_yz,
    F_yz,
    levels=levels,
    cmap="jet"
)

plt.xlabel("Y (Å)")
plt.ylabel("Z (Å)")
plt.title("Membrane Pore Free Energy Map (Front View YZ)")

cbar = plt.colorbar(cont)
cbar.set_label("Free Energy (kJ/mol)")

plt.tight_layout()
plt.savefig("pore_FES_front_YZ.png", dpi=300)

# ==============================
# TOP VIEW (XY)
# ==============================

print("Plotting XY top view...")

plt.figure(figsize=(7,6))

levels = np.linspace(np.min(F_xy), np.max(F_xy), 20)

cont = plt.contourf(
    X_xy,
    Y_xy,
    F_xy,
    levels=levels,
    cmap="jet"
)

plt.xlabel("X (Å)")
plt.ylabel("Y (Å)")
plt.title("Membrane Pore Free Energy Map (Top View XY)")

cbar = plt.colorbar(cont)
cbar.set_label("Free Energy (kJ/mol)")

plt.tight_layout()
plt.savefig("pore_FES_top_XY.png", dpi=300)

# ==============================
# 3D FREE ENERGY SURFACE
# ==============================

print("Generating 3D free energy surface...")

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X_xz,
    Z_xz,
    F_xz,
    cmap="jet",
    linewidth=0,
    antialiased=True
)

ax.set_xlabel("X (Å)")
ax.set_ylabel("Z (Å)")
ax.set_zlabel("Free Energy (kJ/mol)")
ax.set_title("3D Membrane Pore Free Energy Surface")

fig.colorbar(surf, shrink=0.5, aspect=10)

plt.tight_layout()
plt.savefig("pore_FES_3D_surface.png", dpi=300)

plt.show()

print("Analysis complete.")
print("Saved figures:")
print(" - pore_FES_side_XZ.png")
print(" - pore_FES_front_YZ.png")
print(" - pore_FES_top_XY.png")
print(" - pore_FES_3D_surface.png")
