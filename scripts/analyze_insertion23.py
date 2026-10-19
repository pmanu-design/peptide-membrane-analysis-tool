#!/usr/bin/env python3
"""
analyze_insertion2.py
=====================

Updated version of analyze_insertion.py with the following changes:

  - Removed "midplane = 0" text from all axis labels.
  - Removed the midplane visual references (2D dotted line and 3D plane).
  - Kept internal midplane recentering (zmid computed and subtracted).
  - Added optional 3D peptide surface rendering (trisurf) with graceful
    fallback to scatter if triangulation fails.
  - Added an extended 3D viewer panel with multiple camera presets.
  - Improved 3D visuals: equal-axis scaling helper, larger labels,
    anti-aliasing, orthographic option (if supported), and improved defaults.
  - Preserved all analyses (PMF, barrier summary, HOLE pore analysis) and
    file names / pipeline behavior.

Requirements
------------
    pip install MDAnalysis mdahole2 numpy matplotlib

Usage
-----
    python analyze_insertion2.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)
import matplotlib.tri as mtri

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


def _set_axes_equal(ax):
    """
    Make 3D axes have equal scale so objects aren't distorted. This
    modifies the axes limits in-place.
    """
    try:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
    except AttributeError:
        # Older mpl or unexpected Axes: skip
        return

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


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
    bilayer midplane sits at z = 0 internally; however no textual or
    graphical midplane reference is drawn by this function (per user
    request).

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
                   edgecolor="k", linewidth=0.3, zorder=2, antialiased=True)

    # --- peptide backbone --------------------------------------------
    pep = _first_hit(universe, peptide_sel, backbone_atom_names)
    if len(pep) > 0:
        order = np.argsort(pep.resindices)
        pos = pep.positions[order].copy()
        pos[:, 2] -= zmid
        ax.plot(pos[:, i0], pos[:, i1], "-", color="dimgray", lw=2.5, zorder=3, antialiased=True)
        ax.scatter(pos[:, i0], pos[:, i1], s=70, color="lightgray",
                   edgecolor="k", linewidth=0.6, zorder=4, antialiased=True)

        # N-terminus
        ax.scatter(pos[0, i0], pos[0, i1], s=170, color="blue",
                   edgecolor="k", zorder=5, antialiased=True)
        ax.text(pos[0, i0], pos[0, i1], "N", color="white", fontsize=11,
               fontweight="bold", ha="center", va="center", zorder=6)
        # C-terminus
        ax.scatter(pos[-1, i0], pos[-1, i1], s=170, color="red",
                   edgecolor="k", zorder=5, antialiased=True)
        ax.text(pos[-1, i0], pos[-1, i1], "C", color="white", fontsize=11,
               fontweight="bold", ha="center", va="center", zorder=6)
    else:
        print("[snapshot] warning: no peptide backbone atoms found "
              f"(tried selection '{peptide_sel}')")

    # --- Note: midplane reference lines/planes intentionally omitted -----

    ax.set_xlabel(f"{proj[0]} (\u00c5)")
    ax.set_ylabel(f"{proj[1]} (\u00c5)")
    ax.set_xticks([])
    return ax


# ----------------------------------------------------------------------
# 1b. Snapshot (3D): same scene, rendered as a full 3D point cloud or surface
# ----------------------------------------------------------------------
def plot_snapshot_3d(universe, lipids, peptide_sel="protein", frame=-1, ax=None,
                      headgroup_atom_names=("PO4", "P", "P8"),
                      backbone_atom_names=("BB", "CA"),
                      lipid_colors=None, elev=20, azim=-60,
                      peptide_surface=False, surface_alpha=0.6,
                      hide_axes=False, orthographic=False):
    """
    3D counterpart of `plot_snapshot()`: lipid headgroups as colored beads,
    peptide backbone as a gray trace, N-/C-termini highlighted, all plotted
    in full (x, y, z) instead of a single 2D projection.

    This function supports two peptide rendering modes:
      - scatter/line (default)
      - trisurf surface (set peptide_surface=True)

    The bilayer midplane is still used internally to re-center z per frame,
    but no textual/graphical midplane reference is drawn (per user request).

    Additional visualization niceties are included: equal-axis scaling,
    optional orthographic projection (if supported), anti-aliasing, larger
    labels, and an option to hide axes for publication-style figures.

    Parameters
    ----------
    peptide_surface : bool
        If True, attempt to render a smooth peptide surface using a
        triangulation-based trisurf from the backbone coordinates. If
        the triangulation fails, gracefully fall back to scatter.
    surface_alpha : float
        Alpha/transparency for the trisurf surface (if used).
    hide_axes : bool
        If True, hide axis panes/ticks/labels for a cleaner view (still
        returns the axes for further customization).
    orthographic : bool
        If True, attempt to use orthographic projection (matplotlib 3.2+).
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
                   edgecolor="k", linewidth=0.2, depthshade=True, antialiased=True)

    # --- peptide backbone --------------------------------------------
    pep = _first_hit(universe, peptide_sel, backbone_atom_names)
    if len(pep) > 0:
        order = np.argsort(pep.resindices)
        pos = pep.positions[order].copy()
        pos[:, 2] -= zmid

        if peptide_surface and pos.shape[0] >= 3:
            # Attempt triangulation on the projected XY coordinates
            try:
                tri = mtri.Triangulation(pos[:, 0], pos[:, 1])
                ax.plot_trisurf(tri, pos[:, 2], color="lightgray",
                                linewidth=0, antialiased=True, alpha=surface_alpha,
                                shade=True)
            except Exception as exc:
                # Fall back to line + scatter if trisurf fails
                print(f"[snapshot_3d] trisurf failed ({exc}); falling back to scatter.")
                ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-", color="dimgray", lw=2.5, antialiased=True)
                ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=45, color="lightgray",
                           edgecolor="k", linewidth=0.5, antialiased=True)
        else:
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "-", color="dimgray", lw=2.5, antialiased=True)
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=45, color="lightgray",
                       edgecolor="k", linewidth=0.5, antialiased=True)

        # N-terminus
        ax.scatter(*pos[0], s=140, color="blue", edgecolor="k", antialiased=True)
        ax.text(*pos[0], "N", color="black", fontsize=11, fontweight="bold",
               ha="center", va="center")
        # C-terminus
        ax.scatter(*pos[-1], s=140, color="red", edgecolor="k", antialiased=True)
        ax.text(*pos[-1], "C", color="black", fontsize=11, fontweight="bold",
               ha="center", va="center")
    else:
        print("[snapshot_3d] warning: no peptide backbone atoms found "
              f"(tried selection '{peptide_sel}')")

    # --- midplane reference plane intentionally removed -----------------

    # Labels: simplified, without "(midplane = 0)"
    ax.set_xlabel("x (\u00c5)", fontsize=11)
    ax.set_ylabel("y (\u00c5)", fontsize=11)
    ax.set_zlabel("z (\u00c5)", fontsize=11)

    # View and visual polishing
    ax.view_init(elev=elev, azim=azim)
    try:
        if orthographic and hasattr(ax, "set_proj_type"):
            ax.set_proj_type("ortho")
    except Exception:
        # Older matplotlib doesn't support set_proj_type; ignore.
        pass

    _set_axes_equal(ax)

    if hide_axes:
        ax.set_axis_off()

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
                            figsize=None, peptide_surface=False):
    """
    Render the same 3D snapshot multiple times, once per requested camera
    angle, side by side in one figure -- a set of "viewing windows" onto
    the same peptide/bilayer scene so you can inspect it from the front,
    the side, and from above without having to interactively rotate it.

    Each panel uses `plot_snapshot_3d()`, so z is re-centered per-frame to
    put the bilayer midplane at z = 0 internally, but no midplane visuals
    are drawn.

    Parameters
    ----------
    views : sequence of (title, elev, azim)
        One 3D subplot is created per entry, using `plot_snapshot_3d()` with
        that (elev, azim) camera pose.
    peptide_surface : bool
        Pass-through to `plot_snapshot_3d` to render peptide as a surface.
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
                          lipid_colors=lipid_colors, elev=elev, azim=azim,
                          peptide_surface=peptide_surface)
        ax.set_title(title, fontsize=10)
        axes.append(ax)

    fig.tight_layout()
    return fig, axes


# ----------------------------------------------------------------------
# New: extended 3D viewer panel with more presets (interactive when used
# in an interactive Matplotlib session)
# ----------------------------------------------------------------------
def plot_snapshot_3d_viewer_panel(universe, lipids, peptide_sel="protein", frame=-1,
                                  headgroup_atom_names=("PO4", "P", "P8"),
                                  backbone_atom_names=("BB", "CA"),
                                  lipid_colors=None,
                                  peptide_surface=False,
                                  figsize=None):
    """
    Create a multi-panel viewer with a broad set of camera presets:
    front, back, left, right, top, bottom, and an isometric (isometric/iso).

    The returned figure/axes are fully interactive in an interactive GUI
    Matplotlib backend (you can click-and-drag to rotate the 3D panels).
    """
    views = [
        ("front", 0, -90),
        ("back", 0, 90),
        ("left", 0, 180),
        ("right", 0, 0),
        ("top", 90, -90),
        ("bottom", -90, -90),
        ("iso", 30, -60),
    ]
    n = len(views)
    if figsize is None:
        figsize = (4.0 * n, 4.6) if n <= 4 else (4.4 * 4, 4.6 * 2)
    # arrange as multiple rows if many views:
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=figsize)
    axes = []
    for i, (title, elev, azim) in enumerate(views):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        plot_snapshot_3d(universe, lipids, peptide_sel=peptide_sel, frame=frame,
                          ax=ax, headgroup_atom_names=headgroup_atom_names,
                          backbone_atom_names=backbone_atom_names,
                          lipid_colors=lipid_colors, elev=elev, azim=azim,
                          peptide_surface=peptide_surface, hide_axes=False)
        ax.set_title(title, fontsize=9)
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
    barrier (ddG_BN) relative to the transition-state peak.
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
    split by which leaflet/terminus initiates insertion.
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
    shaded band.
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
    # Midplane visual reference intentionally omitted
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
    XTC = "center50.xtc"
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
    # default: scatter/line peptide rendering; set peptide_surface=True to enable trisurf
    plot_snapshot_3d(u, LIPIDS, frame=-1, ax=ax3d, peptide_surface=False)
    ax3d.set_title("late frame (3D)")
    fig.tight_layout()
    fig.savefig("figures/snapshot_3d.png", dpi=300)
    plt.close(fig)

    # ---- 1c. Multi-angle 3D viewing-window panel (final frame) ------------
    fig, panel_axes = plot_snapshot_3d_panel(
        u, LIPIDS, frame=-1,
        views=(("front", 0, -90), ("side", 0, 0), ("top-down", 90, -90)),
        peptide_surface=False
    )
    fig.suptitle("late frame - multi-angle view", y=1.02)
    fig.savefig("figures/snapshot_3d_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 1d. Extended viewer panel (more presets; interactive in GUI) ----
    # Produces many camera presets; panels are interactive when displayed
    # in an interactive Matplotlib backend (e.g. running in a notebook or
    # with an X11/OSX/Windows GUI and calling plt.show()).
    fig, viewer_axes = plot_snapshot_3d_viewer_panel(
        u, LIPIDS, frame=-1, peptide_surface=False
    )
    fig.suptitle("late frame - extended viewer", y=1.02)
    fig.savefig("figures/snapshot_3d_viewer_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. PMF profile + barrier summary -------------------------------
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