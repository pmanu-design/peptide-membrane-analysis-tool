#!/usr/bin/env python3
"""
analyze_insertion.py
=====================

Analysis / plotting toolkit for a peptide inserting into an asymmetric
POPE/POPG membrane, reproducing three figure types:

  1. plot_snapshot()        -> side-view "cartoon" of the peptide sitting in
                                the bilayer, lipids colored by headgroup,
                                N-/C-termini labeled (cf. snapshot figure).
  2. plot_pmf_profile() /
     plot_barrier_summary()  -> free-energy (PMF) curve along an insertion
                                reaction coordinate with basin/barrier
                                annotations, and a ddG_B summary scatter
                                across lipid compositions.
  3. plot_pore_profile()    -> HOLE-style pore-radius-vs-position profile,
                                averaged over the trajectory with a shaded
                                std-dev band.

Only (1) and (3) can be computed directly from a single TPR/XTC pair, since
they are trajectory observables. (2) is a potential-of-mean-force (PMF),
which is the outcome of an *umbrella-sampling / metadynamics campaign*
(many separate windows/simulations along the reaction coordinate, combined
with WHAM/MBAR) -- it can't be derived from one unbiased trajectory. The
PMF section below is written to consume the usual `gmx wham` output
(profile.xvg) plus a small results table you fill in for the summary panel;
swap in your own umbrella-sampling output paths.

Requirements
------------
    pip install MDAnalysis mdahole2 numpy matplotlib

Pore-radius analysis additionally needs the HOLE2 program itself
(https://www.holeprogram.org/) reachable on $PATH (or pass its path via
`executable=`). This is separate from the `mdahole2` python wrapper.

Usage
-----
    python analyze_insertion.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda


# ----------------------------------------------------------------------
# 1. Snapshot: peptide inserted in the bilayer, lipids colored by type
# ----------------------------------------------------------------------
def plot_snapshot(universe, lipids, peptide_sel="protein", frame=-1, ax=None,
                   headgroup_atom_names=("PO4", "P", "P8"),
                   backbone_atom_names=("BB", "CA"),
                   lipid_colors=None, proj=("x", "z")):
    """
    Draw a side-view snapshot: lipid headgroups as colored beads, peptide
    backbone as a gray trace, N-/C-termini highlighted and labeled.

    Works for both coarse-grained (Martini: PO4/BB) and atomistic
    (CHARMM/AMBER: P/CA) topologies -- it tries each candidate atom name
    in turn and uses whichever one is present.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    lipids   : list[str]   resnames to draw, e.g. ["POPE", "POPG"]
    peptide_sel : str      selection string identifying the peptide
    frame    : int         trajectory frame to render
    proj     : (str, str)  which two cartesian axes to plot (default x vs z,
                            i.e. a side view with z as the membrane normal)
    """
    universe.trajectory[frame]
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 5.5))

    if lipid_colors is None:
        # purple/magenta for POPG, orange for POPE, gray fallback
        lipid_colors = {"POPG": "darkorchid", "POPE": "darkorange"}

    axis_idx = {"x": 0, "y": 1, "z": 2}
    i0, i1 = axis_idx[proj[0]], axis_idx[proj[1]]

    def first_hit(sel_base, names):
        for name in names:
            sel = universe.select_atoms(f"{sel_base} and name {name}")
            if len(sel) > 0:
                return sel
        return universe.select_atoms(f"{sel_base} and name 0")  # empty

    # --- lipids -----------------------------------------------------
    leaflet_z = {}
    for lipid in lipids:
        heads = first_hit(f"resname {lipid}", headgroup_atom_names)
        if len(heads) == 0:
            print(f"[snapshot] warning: no headgroup atoms found for {lipid}")
            continue
        pos = heads.positions
        ax.scatter(pos[:, i0], pos[:, i1], s=35,
                   color=lipid_colors.get(lipid, "gray"),
                   edgecolor="k", linewidth=0.3, zorder=2, label=lipid)
        leaflet_z[lipid] = pos[:, 2].mean()

    # --- peptide backbone --------------------------------------------
    pep = first_hit(peptide_sel, backbone_atom_names)
    if len(pep) > 0:
        order = np.argsort(pep.resindices)
        pos = pep.positions[order]
        ax.plot(pos[:, i0], pos[:, i1], "-", color="dimgray", lw=2.5, zorder=3)
        ax.scatter(pos[:, i0], pos[:, i1], s=70, color="lightgray",
                   edgecolor="k", linewidth=0.6, zorder=4)

        # N-terminus
        ax.scatter(pos[0, i0], pos[0, i1], s=170, color="blue",
                   edgecolor="k", zorder=5)
        ax.text(pos[0, i0], pos[0, i1], "N", color="white", fontsize=11,
               fontweight="bold", ha="center", va="center", zorder=6)
        # C-terminus
        ax.scatter(pos[-1, i0], pos[-1, i1], s=170, color="red",
                   edgecolor="k", zorder=5)
        ax.text(pos[-1, i0], pos[-1, i1], "C", color="white", fontsize=11,
               fontweight="bold", ha="center", va="center", zorder=6)
    else:
        print("[snapshot] warning: no peptide backbone atoms found "
              f"(tried selection '{peptide_sel}')")

    # --- midplane line + leaflet labels --------------------------------
    if len(leaflet_z) == 2:
        zmid = np.mean(list(leaflet_z.values()))
        ax.axhline(zmid, color="k", ls=":", lw=1, zorder=1)
        # whichever lipid sits higher in z gets labeled above the line
        sorted_lipids = sorted(leaflet_z, key=leaflet_z.get, reverse=True)
        xpos = ax.get_xlim()[0] if ax.get_xlim()[0] else 0
        ax.text(0.02, 0.97, sorted_lipids[0], transform=ax.transAxes,
               ha="left", va="top", fontsize=10)
        ax.text(0.02, 0.03, sorted_lipids[1], transform=ax.transAxes,
               ha="left", va="bottom", fontsize=10)

    ax.set_xlabel(f"{proj[0]} [\u00c5]")
    ax.set_ylabel(f"{proj[1]} [\u00c5]")
    ax.set_xticks([])
    return ax


# ----------------------------------------------------------------------
# 2a. PMF profile along the insertion reaction coordinate
# ----------------------------------------------------------------------
def plot_pmf_profile(profile_xvg, basin_c_range, basin_n_range, ax=None):
    """
    Plot a 1D PMF (e.g. produced by `gmx wham -o profile.xvg` from
    umbrella-sampling windows along the insertion coordinate) and annotate
    the C-terminus-insertion basin (ddG_BC) and N-terminus-insertion
    barrier (ddG_BN) relative to the transition-state peak, mirroring the
    reference free-energy plot.

    Parameters
    ----------
    profile_xvg : str
        Path to an .xvg file with columns [reaction_coordinate, G(kJ/mol)].
        This is the standard output of `gmx wham`; swap in your own
        PMF-construction pipeline (WHAM/MBAR/metadynamics) if different.
    basin_c_range, basin_n_range : (float, float)
        Reaction-coordinate windows in which to look for the C-terminus-
        insertion and N-terminus-insertion free-energy minima.
    """
    data = np.loadtxt(profile_xvg, comments=("#", "@"))
    x, g = data[:, 0], data[:, 1]
    g = g - g.min()  # zero the deepest basin like the reference plot does at x=+-8

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    ax.plot(x, g, "k-", lw=2)

    def basin_min(rng):
        mask = (x >= rng[0]) & (x <= rng[1])
        idx = np.argmin(g[mask])
        return x[mask][idx], g[mask][idx]

    xc, gc = basin_min(basin_c_range)
    xn, gn = basin_min(basin_n_range)
    g_ts = g.max()
    x_ts = x[np.argmax(g)]

    ax.hlines(gc, basin_c_range[0], basin_c_range[1], color="red", lw=2, ls=":")
    ax.hlines(gn, basin_n_range[0], basin_n_range[1], color="blue", lw=2, ls=":")
    ax.annotate("", xy=(xc, g_ts), xytext=(xc, gc),
               arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
    ax.annotate("", xy=(xn, g_ts), xytext=(xn, gn),
               arrowprops=dict(arrowstyle="<->", color="blue", lw=1.5))
    ax.text(xc, (g_ts + gc) / 2, r"$\Delta\Delta G_{BC}$", color="red",
           ha="right", va="center", fontstyle="italic")
    ax.text(xn, (g_ts + gn) / 2, r"$\Delta\Delta G_{BN}$", color="blue",
           ha="left", va="center", fontstyle="italic")

    ax.axvline(x_ts, color="gray", lw=0.5, ls="--", zorder=0)
    ax.set_xlabel("distance from the TM state [nm]")
    ax.set_ylabel(r"$\Delta G$ [kJ mol$^{-1}$]")
    return ax, {"ddG_BC": g_ts - gc, "ddG_BN": g_ts - gn}


# ----------------------------------------------------------------------
# 2b. Summary scatter of barrier heights across lipid compositions
# ----------------------------------------------------------------------
def plot_barrier_summary(results, ax=None):
    """
    Scatter plot of ddG_B (insertion barrier) across membrane compositions,
    split by which leaflet/terminus initiates insertion -- reproduces the
    right-hand summary panel of the reference figure.

    Parameters
    ----------
    results : dict
        {composition_label: {"POPE_leaflet": {"N": val, "C": val},
                              "POPG_leaflet": {"N": val, "C": val}}}
        Each composition contributes up to two points per leaflet-of-entry
        group (insertion led by the N-terminus vs. the C-terminus). Only
        the leaflets that are actually relevant for a given composition
        need to be present (e.g. a symmetric POPE-only membrane has no
        "POPG_leaflet" entry).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))

    labels = list(results.keys())
    for xi, label in enumerate(labels):
        for leaflet, marker_set in results[label].items():
            for terminus, val in marker_set.items():
                color = "red" if terminus == "C" else "blue"
                ax.scatter(xi, val, s=90, color=color, edgecolor="k", zorder=3)
                ax.annotate(terminus, (xi, val), textcoords="offset points",
                           xytext=(8, 0), fontsize=10, fontweight="bold",
                           color=color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta\Delta G_B$ [kJ mol$^{-1}$]")
    return ax


# ----------------------------------------------------------------------
# 3. HOLE-style pore-radius profile, averaged over the trajectory
# ----------------------------------------------------------------------
def plot_pore_profile(tpr, xtc, select="protein", executable="hole",
                       n_bins=80, ax=None, start=None, stop=None, step=None):
    """
    Run HOLE2 (via the `mdahole2` MDAnalysis kit) frame-by-frame over the
    trajectory, bin the per-frame radius profiles by reaction coordinate
    (membrane-normal position), and plot the mean radius with a +-1 std
    shaded band -- like the reference pore-radius-vs-position plot.

    Requires the external HOLE2 program on PATH (set `executable=` to its
    full path otherwise).
    """
    try:
        from mdahole2.analysis import HoleAnalysis
    except ImportError:
        print("[pore] mdahole2 is not installed (`pip install mdahole2`); "
              "skipping pore-radius analysis.")
        return None

    u = mda.Universe(tpr, xtc)
    ha = HoleAnalysis(u, select=select, executable=executable)
    ha.run(start=start, stop=stop, step=step)

    binned_radii, edges = ha.bin_radii(bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    means = np.array([np.mean(b) if len(b) else np.nan for b in binned_radii])
    stds = np.array([np.std(b) if len(b) else np.nan for b in binned_radii])

    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 5.5))

    ax.plot(means, centers, "b-", lw=2)
    ax.fill_betweenx(centers, means - stds, means + stds,
                     color="blue", alpha=0.2, linewidth=0)
    ax.set_xlabel("Pore radius (\u00c5)")
    ax.set_ylabel("z (\u00c5)")
    ax.set_xlim(left=0)

    ha.delete_temporary_files()
    return ax, centers, means, stds


# ----------------------------------------------------------------------
# Main: wire everything together for the given TPR/XTC/lipid set
# ----------------------------------------------------------------------
if __name__ == "__main__":
    TPR = "md_0_1000.tpr"
    XTC = "centerbb.xtc"
    LIPIDS = ["POPE", "POPG"]

    os.makedirs("figures", exist_ok=True)

    # ---- 1. Snapshot(s) -------------------------------------------------
    u = mda.Universe(TPR, XTC)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 5.5))
    plot_snapshot(u, LIPIDS, frame=0, ax=axes[0])
    axes[0].set_title("early frame")
    plot_snapshot(u, LIPIDS, frame=-1, ax=axes[1])
    axes[1].set_title("late frame")
    fig.tight_layout()
    fig.savefig("figures/snapshot.png", dpi=300)
    plt.close(fig)

    # ---- 2. PMF profile + barrier summary -------------------------------
    # These require umbrella-sampling output, not the single TPR/XTC above.
    # Point `profile_xvg` at your `gmx wham -o profile.xvg` result and set
    # basin windows (in nm, relative to the TM-state reaction coordinate).
    profile_xvg = "profile.xvg"
    if os.path.exists(profile_xvg):
        fig, ax = plt.subplots(figsize=(5, 4))
        _, barriers = plot_pmf_profile(
            profile_xvg, basin_c_range=(-5, -2), basin_n_range=(2, 5), ax=ax
        )
        fig.tight_layout()
        fig.savefig("figures/pmf_profile.png", dpi=300)
        plt.close(fig)
        print("Computed barriers:", barriers)
    else:
        print(f"[pmf] '{profile_xvg}' not found -- skipping PMF plot. "
              "Generate it with `gmx wham` from your umbrella-sampling "
              "windows, then re-run.")

    # Fill in once you have ddG_B values from each composition's PMF:
    example_results = {
        "POPE": {"POPE_leaflet": {"N": 165}},
        "POPG": {"POPE_leaflet": {"N": 140}},
        "POPE:POPG": {"POPE_leaflet": {"N": 157}},
        "POPE/POPG": {"POPE_leaflet": {"N": 145}, "POPG_leaflet": {"C": 183, "N": 185}},
        "POPG/POPE": {"POPE_leaflet": {"C": 128, "N": 130}, "POPG_leaflet": {"C": 168}},
    }
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_barrier_summary(example_results, ax=ax)
    fig.tight_layout()
    fig.savefig("figures/barrier_summary.png", dpi=300)
    plt.close(fig)

    # ---- 3. Pore-radius profile -----------------------------------------
    result = plot_pore_profile(TPR, XTC, select="protein")
    if result is not None:
        ax = result[0]
        ax.figure.tight_layout()
        ax.figure.savefig("figures/pore_profile.png", dpi=300)
        plt.close(ax.figure)

    print("Done. Figures written to ./figures/")
