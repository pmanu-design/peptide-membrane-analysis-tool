#!/usr/bin/env python3
"""
Mass density profile along membrane normal (Z)
with membrane center shifted to Z = 0.

Outputs:
    density_profile_z_centered.csv
    density_profile_z_centered.png
"""

import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER INPUT
# ============================================================

TPR = "md_0_1000.tpr"
XTC = "center1.xtc"

# membrane lipids
LIPIDS = "resname POPE POPG"

# atom selections
HEAD_SELECTION  = f"({LIPIDS}) and name P* N* O1 O2 O3 O4"
INTER_SELECTION = f"({LIPIDS}) and name C1 C2 O11 O12"
TAIL_SELECTION  = f"({LIPIDS}) and name C* H*"

WATER_SELECTION   = "resname W"
PROTEIN_SELECTION = "protein"
MEMBRANE_CENTER_SELECTION = HEAD_SELECTION   # used for centering

NBINS = 200
STEP  = 10

# ============================================================
print("Loading trajectory...")
u = mda.Universe(TPR, XTC)

head = u.select_atoms(HEAD_SELECTION)
inter = u.select_atoms(INTER_SELECTION)
tail = u.select_atoms(TAIL_SELECTION)
water = u.select_atoms(WATER_SELECTION)
protein = u.select_atoms(PROTEIN_SELECTION)
center_atoms = u.select_atoms(MEMBRANE_CENTER_SELECTION)

print(f"Head atoms:    {len(head)}")
print(f"Inter atoms:   {len(inter)}")
print(f"Tail atoms:    {len(tail)}")
print(f"Water atoms:   {len(water)}")
print(f"Protein atoms: {len(protein)}")

# ============================================================
# centered bins: -Lz/2 to +Lz/2
# ============================================================
boxz = u.dimensions[2]

z_edges = np.linspace(-boxz/2, boxz/2, NBINS + 1)
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
dz = z_edges[1] - z_edges[0]

dens_head = np.zeros(NBINS)
dens_inter = np.zeros(NBINS)
dens_tail = np.zeros(NBINS)
dens_water = np.zeros(NBINS)
dens_protein = np.zeros(NBINS)

nframes = 0


# ============================================================
def shift_to_membrane_center(z, center_z, boxz):
    """
    Shift z so membrane center = 0
    wrap to [-Lz/2, +Lz/2]
    """
    z_new = z - center_z
    z_new = (z_new + boxz/2) % boxz - boxz/2
    return z_new


def accumulate_density(atomgroup, accumulator, center_z):
    z = atomgroup.positions[:, 2]
    z = shift_to_membrane_center(z, center_z, boxz)

    masses = atomgroup.masses

    hist, _ = np.histogram(
        z,
        bins=z_edges,
        weights=masses
    )

    accumulator += hist


# ============================================================
print("Calculating centered density profile...")

for ts in u.trajectory[::STEP]:

    nframes += 1

    # membrane center from lipid headgroups
    membrane_center_z = center_atoms.center_of_mass()[2]

    accumulate_density(head, dens_head, membrane_center_z)
    accumulate_density(inter, dens_inter, membrane_center_z)
    accumulate_density(tail, dens_tail, membrane_center_z)
    accumulate_density(water, dens_water, membrane_center_z)
    accumulate_density(protein, dens_protein, membrane_center_z)

# ============================================================
# normalize
# ============================================================
xy_area = u.dimensions[0] * u.dimensions[1]
volume_bin = xy_area * dz

dens_head /= (nframes * volume_bin)
dens_inter /= (nframes * volume_bin)
dens_tail /= (nframes * volume_bin)
dens_water /= (nframes * volume_bin)
dens_protein /= (nframes * volume_bin)

# Å³ → nm³
factor = 1000.0
dens_head *= factor
dens_inter *= factor
dens_tail *= factor
dens_water *= factor
dens_protein *= factor

# ============================================================
# save CSV
# ============================================================
df = pd.DataFrame({
    "Z_nm": z_centers / 10.0,
    "Head": dens_head,
    "Inter": dens_inter,
    "Tail": dens_tail,
    "Sol": dens_water,
    "Protein": dens_protein
})

df.to_csv("density_profile_z_centered.csv", index=False)

print("Saved: density_profile_z_centered.csv")

# ============================================================
# plot
# ============================================================
plt.figure(figsize=(8,6))

plt.plot(z_centers/10, dens_head,
         color='red', lw=2, label='Head')

plt.plot(z_centers/10, dens_inter,
         color='magenta', lw=2, label='Inter')

plt.plot(z_centers/10, dens_tail,
         color='blue', lw=2, label='Tail')

plt.plot(z_centers/10, dens_water,
         color='cyan', lw=2, label='Sol')

plt.fill_between(
    z_centers/10,
    dens_protein,
    color='gray',
    alpha=0.7,
    label='peptide'
)

plt.axvline(0, ls='--', lw=1, color='black')

plt.xlabel("Z (nm)", fontsize=14)
plt.ylabel("Density (kg/m$^3$)", fontsize=14)

plt.xlim(-boxz/20, boxz/20)
plt.ylim(bottom=0)

plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(
    "density_profile_z_centered.png",
    dpi=300
)

plt.show()

print("Saved: density_profile_z_centered.png")
print("Done.")
