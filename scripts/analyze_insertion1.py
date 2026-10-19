#!/usr/bin/env python3
"""
analyze_insertion.py
=====================

Analysis / plotting toolkit for a peptide inserting into an asymmetric
POPE/POPG membrane, reproducing three figure types:

  1. plot_snapshot()          -> 2D side-view "cartoon" of the peptide
                                  sitting in the bilayer, lipids colored by
                                  headgroup, N-/C-termini labeled.
     plot_snapshot_3d()        -> the same idea rendered as a true 3D
                                  scene (x, y, z all shown at once).
     plot_snapshot_3d_panel()  -> a multi-panel "viewing window" figure
                                  that shows the same 3D snapshot from
                                  several camera angles side by side
                                  (e.g. front / side / top-down).
  2. plot_pmf_profile() /
     plot_barrier_summary()    -> free-energy (PMF) curve along an
                                  insertion reaction coordinate with
                                  basin/barrier annotations, and a ddG_B
                                  summary scatter across lipid compositions.
  3. plot_pore_profile()      -> HOLE-style pore-radius-vs-position
                                  profile, averaged over the trajectory
                                  with a shaded std-dev band.

Only (1) and (3) can be computed directly from a single TPR/XTC pair, since
they are trajectory observables. (2) is a potential-of-mean-force (PMF),
which is the outcome of an *umbrella-sampling / metadynamics campaign*
(many separate windows/simulations along the reaction coordinate, combined
with WHAM/MBAR) -- it can't be derived from one unbiased trajectory. The
PMF section below is written to consume the usual `gmx wham` output
(profile.xvg) plus a small results table you fill in for the summary panel;
swap in your own umbrella-sampling output paths.

Note on lipid identity labels
------------------------------
The snapshot plots (2D and 3D) color lipid headgroups by residue name
(e.g. POPE vs. POPG) using `lipid_colors`, but no longer print the lipid
resnames as text/legend on the figure itself -- only the colored beads and
the peptide backbone/termini are annotated. If you want a color key, add a
separate legend/caption outside the axes using `lipid_colors`.

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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
_DEFAULT_LIPID_COLORS = {"POPG": "darkorchid", "POPE": "darkorange"}


def _first_hit(universe, sel_base, names):
    """Return the first non-empty atom selection among candidate atom
    names (tries Martini-style names before atomistic names, or vice
    versa, depending on what's passed in)."""
    for name in names:
        sel = universe.select_atoms(f"{sel_base} and name {name}")
        if len(sel) > 0:
            return sel
    return universe.select_atoms(f"{sel_base} and name 0")  # empty


# ----------------------------------------------------------------------
# 1a. Snapshot (2D): peptide inserted in the bilayer, lipids colored by type
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

    Lipid species are distinguished purely by bead color (see
    `lipid_colors`); no lipid resname text is drawn on the axes.

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
        lipid_colors = _DEFAULT_LIPID_COLORS

    axis_idx = {"x": 0, "y": 1, "z": 2}
    i0, i1 = axis_idx[proj[0]], axis_idx[proj[1]]

    # --- lipids -----------------------------------------------------
    leaflet_z = {}
    for lipid in lipids:
        heads = _first_hit(universe, f"resname {lipid}", headgroup_atom_names)
        if len(heads) == 0:
            print(f"[snapshot] warning: no headgroup atoms found for {lipid}")
            continue
        pos = heads.positions
        ax.scatter(pos[:, i0], pos[:, i1], s=35,
                   color=lipid_colors.get(lipid, "gray"),
                   edgecolor="k", linewidth=0.3, zorder=2)
        leaflet_z[lipid] = pos[:, 2].mean()

    # --- peptide backbone --------------------------------------------
    pep = _first_hit(universe, peptide_sel, backbone_atom_names)
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

    # --- midplane line (no lipid-name text is drawn) -------------------
    if len(leaflet_z) == 2:
        zmid = np.mean(list(leaflet_z.values()))
        ax.axhline(zmid, color="k", ls=":", lw=1, zorder=1)

    ax.set_xlabel(f"{proj[0]} [\u00c5]")
    ax.set_ylabel(f"{proj[1]} [\u00c5]")
    ax.set_xticks([])
    return ax


# ----------------------------------------------------------------------
# 1b. Snapshot (3D): same scene, rendered as a full 3D point cloud
# ----------------------------------------------------------------------
def plot_snapshot_3d(universe, lipids, peptide_sel="protein", frame=-1, ax=None,
                      headgroup_atom_names=("PO4", "P", "P8"),
                      backbone_atom_names=("BB", "CA"),
                      lipid_colors=None, elev=20, azim=-60):
    """
    3D counterpart of `plot_snapshot()`: lipid headgroups as colored beads,
    peptide backbone as a gray trace, N-/C-termini highlighted, all plotted
    in full (x, y, z) instead of a single 2D projection.

    As in the 2D version, lipid species are distinguished only by bead
    color -- no lipid resname text is drawn on the axes.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    lipids   : list[str]   resnames to draw, e.g. ["POPE", "POPG"]
    peptide_sel : str      selection string identifying the peptide
    frame    : int         trajectory frame to render
    ax       : Axes3D, optional   a 3D axes (projection='3d'); created if None
    elev, azim : float     camera elevation / azimuth in degrees, passed to
                            `ax.view_init()` -- these define the "viewing
                            window" onto the 3D scene.
    """
    universe.trajectory[frame]
    if ax is None:
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.add_subplot(111, projection="3d")

    if lipid_colors is None:
        lipid_colors = _DEFAULT_LIPID_COLORS

    # --- lipids -----------------------------------------------------
    for lipid in lipids:
        heads = _first_hit(universe, f"resname {lipid}", headgroup_atom_names)
        if len(heads) == 0:
            print(f"[snapshot_3d] warning: no headgroup atoms found for {lipid}")
            continue
        pos = heads.positions
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=25,
                   color=lipid_colors.get(lipid, "gray"),
                   edgecolor="k", linewidth=0.2, depthshade=True)

    # --- peptide backbone --------------------------------------------
    pep = _first_hit(universe, peptide_sel, backbone_atom_names)
    if len(pep) > 0:
        order = np.argsort(pep.resindices)
        pos = pep.positions[order]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-", color="dimgray", lw=2.5)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=45, color="lightgray",
                   edgecolor="k", linewidth=0.5)

        # N-terminus
        ax.scatter(*pos[0], s=140, color="blue", edgecolor="k")
        ax.text(*pos[0], "N", color="black", fontsize=10, fontweight="bold",
               ha="center", va="center")
        # C-terminus
        ax.scatter(*pos[-1], s=140, color="red", edgecolor="k")
        ax.text(*pos[-1], "C", color="black", fontsize=10, fontweight="bold",
               ha="center", va="center")
    else:
        print("[snapshot_3d] warning: no peptide backbone atoms found "
              f"(tried selection '{peptide_sel}')")

    ax.set_xlabel("x [\u00c5]")
    ax.set_ylabel("y [\u00c5]")
    ax.set_zlabel("z [\u00c5]")
    ax.view_init(elev=elev, azim=azim)
    return ax


# ----------------------------------------------------------------------
# 1c. Multi-angle "viewing window" panel around the same 3D snapshot
# ----------------------------------------------------------------------
def plot_snapshot_3d_panel(universe, lipids, peptide_sel="protein", frame=-1,
                            headgroup_atom_names=("PO4", "P", "P8"),
                            backbone_atom_names=("BB", "CA"),
                            lipid_colors=None,
                            views=(("front", 0, -90),
                                   ("side", 0, 0),
                                   ("top-down", 90, -90)),
                            figsize=None):
    """
    Render the same 3D snapshot multiple times, once per requested camera
    angle, side by side in one figure -- a set of "viewing windows" onto
    the same peptide/bilayer scene so you can inspect it from the front,
    the side, and from above without having to interactively rotate it.

    Parameters
    ----------
    views : sequence of (title, elev, azim)
        One 3D subplot is created per entry, using `plot_snapshot_3d()`
        with that (elev, azim) camera pose. Defaults to a front / side /
        top-down triplet.
    figsize : (float, float), optional
        Defaults to scaling with the number of views.

    Returns
    -------
    fig, axes : the Figure and the list of Axes3D panels created.
    """
    n = len(views)
    if figsize is None:
        figsize = (4.4 * n, 4.6)

    fig = plt.figure(figsize=figsize)
    axes = []
    for i, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        plot_snapshot_3d(universe, lipids, peptide_sel=peptide_sel, frame=frame,
                          ax=ax, headgroup_atom_names=headgroup_atom_names,
                          backbone_atom_names=backbone_atom_names,
                          lipid_colors=lipid_colors, elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)
        axes.append(ax)

    fig.tight_layout()
    return fig, axes


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

    # ---- 1a. 2D snapshot(s) ----------------------------------------------
    u = mda.Universe(TPR, XTC)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 5.5))
    plot_snapshot(u, LIPIDS, frame=0, ax=axes[0])
    axes[0].set_title("early frame")
    plot_snapshot(u, LIPIDS, frame=-1, ax=axes[1])
    axes[1].set_title("late frame")
    fig.tight_layout()
    fig.savefig("figures/snapshot.png", dpi=300)
    plt.close(fig)

    # ---- 1b. Single 3D snapshot (final frame) -----------------------------
    fig = plt.figure(figsize=(5.5, 5.5))
    ax3d = fig.add_subplot(111, projection="3d")
    plot_snapshot_3d(u, LIPIDS, frame=-1, ax=ax3d)
    ax3d.set_title("late frame (3D)")
    fig.tight_layout()
    fig.savefig("figures/snapshot_3d.png", dpi=300)
    plt.close(fig)

    # ---- 1c. Multi-angle 3D viewing-window panel (final frame) ------------
    fig, panel_axes = plot_snapshot_3d_panel(
        u, LIPIDS, frame=-1,
        views=(("front", 0, -90), ("side", 0, 0), ("top-down", 90, -90)),
    )
    fig.suptitle("late frame - multi-angle view", y=1.02)
    fig.savefig("figures/snapshot_3d_panel.png", dpi=300, bbox_inches="tight")
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
