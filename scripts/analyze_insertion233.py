#!/usr/bin/env python3
"""
analyze_insertion.py
=====================

Analysis / plotting toolkit for a peptide inserting into an asymmetric
POPE/POPG membrane, reproducing three figure types:

  1. plot_snapshot()          -> 2D side-view "cartoon" of the peptide
                                  sitting in the bilayer, lipids colored by
                                  headgroup, N-/C-termini labeled.
     plot_snapshot_series()    -> the same 2D cartoon repeated over several
                                  points in time (not just early/late), so
                                  you can watch the insertion progress
                                  frame-by-frame across the trajectory.
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

Membrane midplane convention
-----------------------------
Every plot that shows a membrane-normal (z) coordinate -- the 2D and 3D
snapshots, the multi-angle panel, and the pore-radius profile -- re-centers
z so that the **bilayer midplane sits at z = 0**. The midplane is estimated
per-frame as the midpoint between the mean headgroup z of each lipid
species in `lipids` (this assumes each species is concentrated in one
leaflet, which holds for the asymmetric POPE/POPG system this script
targets). A thin dotted reference line/plane at 0 marks the midplane in
every figure; positive z is "above" it, negative z is "below" it.

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


def _membrane_midplane_z(universe, lipids, headgroup_atom_names=("PO4", "P", "P8")):
    """
    Estimate the z-coordinate of the bilayer midplane at the universe's
    *current* frame, so callers can shift all z-coordinates by this value
    and plot with the midplane fixed at z = 0.

    Defined as the midpoint between the lowest and highest per-species mean
    headgroup z among `lipids`. For a two-leaflet asymmetric membrane where
    each lipid resname lives predominantly in one leaflet (e.g. POPE on one
    side, POPG on the other), this recovers the midpoint between the two
    leaflets. If only one lipid species/leaflet is found, that leaflet's
    mean z is used as a fallback (with a warning) since no midplane can be
    triangulated from a single leaflet.

    Returns 0.0 (no shift) if no headgroup atoms are found at all.
    """
    leaflet_means = []
    for lipid in lipids:
        heads = _first_hit(universe, f"resname {lipid}", headgroup_atom_names)
        if len(heads) == 0:
            continue
        leaflet_means.append(heads.positions[:, 2].mean())

    if not leaflet_means:
        print("[midplane] warning: no lipid headgroup atoms found; "
              "z will not be re-centered (zmid = 0).")
        return 0.0
    if len(leaflet_means) == 1:
        print("[midplane] warning: only one leaflet/lipid species found; "
              "using its mean z as the midplane (no second leaflet to "
              "average against).")
        return leaflet_means[0]

    return 0.5 * (min(leaflet_means) + max(leaflet_means))


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

    The membrane-normal coordinate (z) is re-centered per-frame so the
    bilayer midplane sits at z = 0 (see module docstring); a dotted
    reference line is drawn at 0 whenever `proj` includes "z".

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

    # --- midplane: recenter z so the bilayer midplane is at 0 -----------
    zmid = _membrane_midplane_z(universe, lipids, headgroup_atom_names)

    # --- lipids -----------------------------------------------------
    for lipid in lipids:
        heads = _first_hit(universe, f"resname {lipid}", headgroup_atom_names)
        if len(heads) == 0:
            print(f"[snapshot] warning: no headgroup atoms found for {lipid}")
            continue
        pos = heads.positions.copy()
        pos[:, 2] -= zmid
        ax.scatter(pos[:, i0], pos[:, i1], s=35,
                   color=lipid_colors.get(lipid, "gray"),
                   edgecolor="k", linewidth=0.3, zorder=2)

    # --- peptide backbone --------------------------------------------
    pep = _first_hit(universe, peptide_sel, backbone_atom_names)
    if len(pep) > 0:
        order = np.argsort(pep.resindices)
        pos = pep.positions[order].copy()
        pos[:, 2] -= zmid
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

    # --- midplane reference line at 0 (no lipid-name text is drawn) -----
    if proj[1] == "z":
        ax.axhline(0.0, color="k", ls=":", lw=1, zorder=1)
    elif proj[0] == "z":
        ax.axvline(0.0, color="k", ls=":", lw=1, zorder=1)

    ax.set_xlabel(f"{proj[0]} [\u00c5]" + (" (midplane = 0)" if proj[0] == "z" else ""))
    ax.set_ylabel(f"{proj[1]} [\u00c5]" + (" (midplane = 0)" if proj[1] == "z" else ""))
    ax.set_xticks([])
    return ax


def _frame_time_label(universe, frame, time_units="ns"):
    """Return a short label ('12.0 ns' / '12000 ps' / 'frame 5') for a
    trajectory frame, using the universe's own time metadata when
    possible and falling back to the raw frame index if not."""
    try:
        universe.trajectory[frame]
        t_ps = universe.trajectory.time
        if time_units == "ns":
            return f"{t_ps / 1000.0:.1f} ns"
        if time_units == "ps":
            return f"{t_ps:.0f} ps"
    except Exception:
        pass
    n = len(universe.trajectory)
    idx = frame if frame >= 0 else n + frame
    return f"frame {idx}"


# ----------------------------------------------------------------------
# 1a-bis. Snapshot (2D) over time: many frames, not just early/late
# ----------------------------------------------------------------------
def plot_snapshot_series(universe, lipids, peptide_sel="protein",
                          n_frames=6, frames=None,
                          headgroup_atom_names=("PO4", "P", "P8"),
                          backbone_atom_names=("BB", "CA"),
                          lipid_colors=None, proj=("x", "z"),
                          time_units="ns", share_y=True, figsize=None):
    """
    Repeat `plot_snapshot()` across several points in time, side by side,
    so insertion progress can be watched frame-by-frame instead of only
    comparing a single "early" and "late" frame.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    lipids   : list[str]   resnames to draw, e.g. ["POPE", "POPG"]
    peptide_sel : str      selection string identifying the peptide
    n_frames : int
        Number of evenly spaced frames (including the first and last) to
        draw, if `frames` is not given explicitly. E.g. n_frames=6 gives
        a 6-panel time series across the whole trajectory. Ignored if
        `frames` is provided.
    frames   : sequence[int], optional
        Explicit list of trajectory frame indices to draw instead of an
        evenly spaced series (overrides `n_frames`).
    proj     : (str, str)  which two cartesian axes to plot (default x vs z)
    time_units : {"ns", "ps", "frame"}
        Units used for each panel's title. "frame" (or any value MDAnalysis
        can't interpret as time) falls back to labeling by frame index.
    share_y  : bool        share the z (or y) axis limits/ticks across
                            panels so bilayer position is directly
                            comparable frame-to-frame.
    figsize  : (float, float), optional  defaults to scaling with n panels.

    Returns
    -------
    fig, axes : the Figure and the list/array of Axes created, one per
                frame, each already re-centered so the bilayer midplane
                is at z = 0 (see module docstring).
    """
    n_total = len(universe.trajectory)

    if frames is None:
        if n_frames < 1:
            raise ValueError("n_frames must be >= 1")
        if n_frames == 1:
            frames = [n_total - 1]
        else:
            frames = sorted(set(
                np.linspace(0, n_total - 1, n_frames).round().astype(int).tolist()
            ))
    n = len(frames)

    if figsize is None:
        figsize = (2.6 * n, 5.5)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=share_y)
    axes = np.atleast_1d(axes)

    for i, (ax, frame) in enumerate(zip(axes, frames)):
        plot_snapshot(universe, lipids, peptide_sel=peptide_sel, frame=frame,
                      ax=ax, headgroup_atom_names=headgroup_atom_names,
                      backbone_atom_names=backbone_atom_names,
                      lipid_colors=lipid_colors, proj=proj)
        ax.set_title(_frame_time_label(universe, frame, time_units), fontsize=9)
        if share_y and i > 0:
            ax.set_ylabel("")

    fig.tight_layout()
    return fig, axes


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
    color -- no lipid resname text is drawn on the axes -- and z is
    re-centered per-frame so the bilayer midplane sits at z = 0 (see
    module docstring).

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

    # --- midplane: recenter z so the bilayer midplane is at 0 -----------
    zmid = _membrane_midplane_z(universe, lipids, headgroup_atom_names)

    # --- lipids -----------------------------------------------------
    for lipid in lipids:
        heads = _first_hit(universe, f"resname {lipid}", headgroup_atom_names)
        if len(heads) == 0:
            print(f"[snapshot_3d] warning: no headgroup atoms found for {lipid}")
            continue
        pos = heads.positions.copy()
        pos[:, 2] -= zmid
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=25,
                   color=lipid_colors.get(lipid, "gray"),
                   edgecolor="k", linewidth=0.2, depthshade=True)

    # --- peptide backbone --------------------------------------------
    pep = _first_hit(universe, peptide_sel, backbone_atom_names)
    if len(pep) > 0:
        order = np.argsort(pep.resindices)
        pos = pep.positions[order].copy()
        pos[:, 2] -= zmid
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

    # --- midplane reference plane at z = 0 -------------------------------
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xx, yy = np.meshgrid(xlim, ylim)
    ax.plot_surface(xx, yy, np.zeros_like(xx), color="gray", alpha=0.08,
                     linewidth=0, shade=False, zorder=0)

    ax.set_xlabel("x [\u00c5]")
    ax.set_ylabel("y [\u00c5]")
    ax.set_zlabel("z [\u00c5] (midplane = 0)")
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

    Each panel uses `plot_snapshot_3d()`, so z is re-centered per-frame to
    put the bilayer midplane at z = 0 (see module docstring).

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

    Note: this reaction coordinate is independent of the bilayer-midplane
    z = 0 convention used in the snapshot plots -- it is whatever quantity
    the umbrella-sampling windows were biased along (e.g. peptide
    center-of-mass distance from the bilayer center), and is left as-is.

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
                       n_bins=80, ax=None, start=None, stop=None, step=None,
                       lipids=None, headgroup_atom_names=("PO4", "P", "P8")):
    """
    Run HOLE2 (via the `mdahole2` MDAnalysis kit) frame-by-frame over the
    trajectory, bin the per-frame radius profiles by reaction coordinate
    (membrane-normal position), and plot the mean radius with a +-1 std
    shaded band -- like the reference pore-radius-vs-position plot.

    Requires the external HOLE2 program on PATH (set `executable=` to its
    full path otherwise).

    Parameters
    ----------
    lipids : list[str], optional
        Lipid resnames (e.g. ["POPE", "POPG"]) used to estimate the
        bilayer midplane and shift the plotted z-axis so the midplane
        sits at 0, matching the convention used in the snapshot plots
        (see module docstring). The midplane is estimated once, from the
        first analyzed frame (`start`, or frame 0 if `start` is None); if
        omitted, the raw HOLE reaction-coordinate values are plotted
        unshifted, as before.
    """
    try:
        from mdahole2.analysis import HoleAnalysis
    except ImportError:
        print("[pore] mdahole2 is not installed (`pip install mdahole2`); "
              "skipping pore-radius analysis.")
        return None

    u = mda.Universe(tpr, xtc)

    zmid = 0.0
    if lipids is not None:
        u.trajectory[start if start is not None else 0]
        zmid = _membrane_midplane_z(u, lipids, headgroup_atom_names)

    ha = HoleAnalysis(u, select=select, executable=executable)
    ha.run(start=start, stop=stop, step=step)

    binned_radii, edges = ha.bin_radii(bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:]) - zmid

    means = np.array([np.mean(b) if len(b) else np.nan for b in binned_radii])
    stds = np.array([np.std(b) if len(b) else np.nan for b in binned_radii])

    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 5.5))

    ax.plot(means, centers, "b-", lw=2)
    ax.fill_betweenx(centers, means - stds, means + stds,
                     color="blue", alpha=0.2, linewidth=0)
    if lipids is not None:
        ax.axhline(0.0, color="k", ls=":", lw=1, zorder=1)
    ax.set_xlabel("Pore radius (\u00c5)")
    ax.set_ylabel("z (\u00c5)" + (" (midplane = 0)" if lipids is not None else ""))
    ax.set_xlim(left=0)

    ha.delete_temporary_files()
    return ax, centers, means, stds


# ----------------------------------------------------------------------
# Main: wire everything together for the given TPR/XTC/lipid set
# ----------------------------------------------------------------------
if __name__ == "__main__":
    TPR = "md_0_1000.tpr"
    XTC = "center1.xtc"
    LIPIDS = ["POPE", "POPG"]

    os.makedirs("figures", exist_ok=True)

    # ---- 1a. 2D snapshot over time (not just early/late) ------------------
    u = mda.Universe(TPR, XTC)
    fig, axes = plot_snapshot_series(u, LIPIDS, n_frames=6, time_units="ns")
    fig.suptitle("insertion over time", y=1.02)
    fig.savefig("figures/snapshot_over_time.png", dpi=300, bbox_inches="tight")
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
    result = plot_pore_profile(TPR, XTC, select="protein", lipids=LIPIDS)
    if result is not None:
        ax = result[0]
        ax.figure.tight_layout()
        ax.figure.savefig("figures/pore_profile.png", dpi=300)
        plt.close(ax.figure)

    print("Done. Figures written to ./figures/")
