#!/usr/bin/env python3
"""
Mass density profile along membrane normal (Z)
with membrane center shifted to Z = 0.

UPDATED:
- Adds individual phospholipid density profiles
- Adds head and tail density profiles for each phospholipid
- Keeps peptide + water
- Exports all profiles to CSV
- Z-axis now in Angstrom (Å)
- Adds leaflet shading (-20 to 0 to +20 Å)
- Enhances membrane boundary visualization

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

LIPIDS = ["POPE", "POPG"]

NBINS = 200
STEP = 10

WATER_SELECTION = "resname W"
PROTEIN_SELECTION = "protein"

CENTER_SELECTION = "(resname POPE POPG) and name P* N* O1 O2 O3 O4"

# leaflet reference range (Å)
LEAFLET_TOP = 20
LEAFLET_BOTTOM = -20

# ============================================================
print("Loading trajectory...")
u = mda.Universe(TPR, XTC)

center_atoms = u.select_atoms(CENTER_SELECTION)
water = u.select_atoms(WATER_SELECTION)
protein = u.select_atoms(PROTEIN_SELECTION)

# ============================================================
# Build selections dynamically
# ============================================================
lipid_groups = {}

for lipid in LIPIDS:
    lipid_groups[f"{lipid}_all"] = u.select_atoms(
        f"resname {lipid}"
    )

    lipid_groups[f"{lipid}_head"] = u.select_atoms(
        f"resname {lipid} and name P* N* O1 O2 O3 O4 C1 C2 O11 O12"
    )

    lipid_groups[f"{lipid}_tail"] = u.select_atoms(
        f"resname {lipid} and name C* H*"
    )

# ============================================================
# BINS
# ============================================================
boxz = u.dimensions[2]

z_edges = np.linspace(-boxz/2, boxz/2, NBINS + 1)
z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
dz = z_edges[1] - z_edges[0]

acc = {}
for k in lipid_groups:
    acc[k] = np.zeros(NBINS)

acc["Water"] = np.zeros(NBINS)
acc["Peptide"] = np.zeros(NBINS)

nframes = 0

# ============================================================
def shift_to_membrane_center(z, center_z, boxz):
    z_new = z - center_z
    z_new = (z_new + boxz/2) % boxz - boxz/2
    return z_new


def accumulate_density(atomgroup, accumulator, center_z):
    if len(atomgroup) == 0:
        return

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
    membrane_center_z = center_atoms.center_of_mass()[2]

    for name, ag in lipid_groups.items():
        accumulate_density(ag, acc[name], membrane_center_z)

    accumulate_density(water, acc["Water"], membrane_center_z)
    accumulate_density(protein, acc["Peptide"], membrane_center_z)

# ============================================================
# normalize
# ============================================================
xy_area = u.dimensions[0] * u.dimensions[1]
volume_bin = xy_area * dz

for k in acc:
    acc[k] /= (nframes * volume_bin)

factor = 1000.0
for k in acc:
    acc[k] *= factor

# ============================================================
# SAVE CSV
# ============================================================
out = {
    "Z_A": z_centers
}

for k in acc:
    out[k] = acc[k]

df = pd.DataFrame(out)
df.to_csv("density_profile_z_centered.csv", index=False)

print("Saved: density_profile_z_centered.csv")

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(10, 7))

# --- leaflet shading ---
plt.axvspan(0, LEAFLET_TOP, color="gray", alpha=0.08)
plt.axvspan(LEAFLET_BOTTOM, 0, color="gray", alpha=0.08)

# --- density curves ---
for k in lipid_groups:
    plt.plot(z_centers, acc[k], lw=2, label=k)

plt.plot(z_centers, acc["Water"], lw=2, label="Water")

plt.fill_between(z_centers, acc["Peptide"], alpha=0.5, label="Peptide")

# --- membrane center + leaflet boundaries ---
plt.axvline(0, lw=2.5, color="black")
plt.axvline(LEAFLET_TOP, lw=2.0, color="black", alpha=0.7)
plt.axvline(LEAFLET_BOTTOM, lw=2.0, color="black", alpha=0.7)

plt.xlabel("Z (Å)", fontsize=14)
plt.ylabel("Density (kg/m$^3$)", fontsize=14)

plt.xlim(-boxz/2, boxz/2)
plt.ylim(bottom=0)

plt.legend(frameon=False, ncol=2)

plt.tight_layout()

plt.savefig("density_profile_z_centered.png", dpi=300)
plt.show()

print("Saved: density_profile_z_centered.png")
print("Done.")
