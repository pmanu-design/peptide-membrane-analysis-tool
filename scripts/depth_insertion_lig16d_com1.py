#!/usr/bin/env python3
"""
Peptide insertion depth (publication-quality plot)
- Uses membrane midplane (true insertion metric)
- Clean aesthetic (like reference image)
- Mean + leaflet boundaries
- Smoothed curve

Author: Prince Manu (refined)
"""

# 🔧 NO GUI (fix Qt issues)
import matplotlib
matplotlib.use('Agg')

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import sys

# === INPUT FILES ===
TPR_FILE = "md_0_1000.tpr"
XTC_FILE = "centerA.xtc"

PEPTIDE_RESNAME = "PROTEIN"
LIPID_RESNAMES = "POPE POPG"

OUTPUT_FILE = "insertion_depth_clean.dat"
PLOT_FILE = "insertion_depth_clean.png"


def detect_phosphate(u):
    for name in ["P", "P31", "PO4", "P1"]:
        if len(u.select_atoms(f"resname {LIPID_RESNAMES} and name {name}")) > 0:
            print(f"✅ Using phosphate: {name}")
            return name
    raise ValueError("❌ No phosphate found")


def select_groups(u):
    peptide = u.select_atoms(f"protein or resname {PEPTIDE_RESNAME}")
    if len(peptide) == 0:
        raise ValueError("❌ Peptide not found")

    pname = detect_phosphate(u)
    phosphates = u.select_atoms(f"resname {LIPID_RESNAMES} and name {pname}")

    return peptide, phosphates


def compute_depth(u, peptide, phosphates):
    times, depths = [], []
    upper_zs, lower_zs = [], []

    for ts in u.trajectory:
        z_all = phosphates.positions[:, 2]

        z_mid = np.mean(z_all)
        upper = z_all[z_all > z_mid]
        lower = z_all[z_all < z_mid]

        pep_z = peptide.center_of_mass()[2]

        depth = pep_z - z_mid

        times.append(ts.time)
        depths.append(depth)
        upper_zs.append(np.mean(upper) - z_mid)
        lower_zs.append(np.mean(lower) - z_mid)

    return (np.array(times),
            np.array(depths),
            np.array(upper_zs),
            np.array(lower_zs))


def plot_clean(times_ps, depths, upper, lower):
    times_ns = times_ps / 1000.0

    # 🔥 smoothing (this is key for your style)
    depths_smooth = gaussian_filter1d(depths, sigma=2)

    mean_depth = np.mean(depths)

    plt.figure(figsize=(8, 4.5))

    # peptide line
    plt.plot(times_ns, depths_smooth, color="black", lw=2)

    # membrane center
    plt.axhline(0, linestyle=":", color="black", lw=1)

    # leaflet boundaries (dotted orange)
    plt.axhline(np.mean(upper), linestyle=":", color="#E69F00", lw=1.5)
    plt.axhline(np.mean(lower), linestyle=":", color="#E69F00", lw=1.5)

    # shaded membrane region
    plt.fill_between(times_ns,
                     np.mean(lower),
                     np.mean(upper),
                     color="gray",
                     alpha=0.2)

    # mean peptide line
    plt.axhline(mean_depth, linestyle="--", color="#E69F00", lw=1.5)

    # labels
    plt.xlabel("Simulation time [ns]", fontsize=12)
    plt.ylabel("Membrane normal [Å]", fontsize=12)

    plt.xlim(times_ns.min(), times_ns.max())
    plt.ylim(-50, 50)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()

    print(f"💾 Saved plot: {PLOT_FILE}")
    print(f"✅ Mean depth: {mean_depth:.2f} Å")


def main():
    print("🚀 Loading system...")
    u = mda.Universe(TPR_FILE, XTC_FILE)

    peptide, phosphates = select_groups(u)

    print("📏 Computing insertion depth...")
    times, depths, upper, lower = compute_depth(u, peptide, phosphates)

    np.savetxt(OUTPUT_FILE,
               np.column_stack((times, depths)),
               header="Time(ps)\tDepth(Å)",
               fmt="%12.4f")

    print(f"💾 Data saved: {OUTPUT_FILE}")

    plot_clean(times, depths, upper, lower)

    print("🎯 Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{e}\n❌ Failed\n")
        sys.exit(1)
