#!/usr/bin/env python3
"""
Pore Analysis Suite
====================

Reproduces the following analyses for a peptide/protein pore embedded in a
lipid bilayer, using MDAnalysis + numpy + matplotlib:

  1. pore_radius_profile()        -> HOLE-style pore radius vs z, with std
                                      shading ("Pore radius (A) vs z" plot)
  2. filter_gate_analysis()       -> (a) pore radius vs z zoomed to the
                                      channel region, (b) scatter of
                                      Filter/Gate residue z-position vs frame
  3. pore_lining_peptides_grid()  -> multi-row x 3-column figure:
                                        col 1: time series of number of
                                               pore-lining peptides, one
                                               color per condition (e.g.
                                               xi = 0.7 / 0.85 / 1.0)
                                        col 2: top-down snapshot with
                                               pore-lining peptide backbones
                                               highlighted (orange tubes)
                                        col 3: side-on snapshot, same
                                      one row per system/replica (a, b, c, ...)
  4. lipid_water_density_maps()   -> 2D histograms of lipid headgroup
                                      phosphate and water oxygen density vs
                                      (radius, z) from the pore center

Each analysis writes PNG(s) to OUTDIR and (where relevant) CSVs of the
underlying data.

NOTES / THINGS YOU WILL LIKELY NEED TO ADJUST
-----------------------------------------------
- FILTER_SEL / GATE_SEL: set these MDAnalysis selection strings to whichever
  residues form the "filter" and "gate" constriction points in your channel.
- SYSTEMS (section 3): describes the systems/replicas (rows a, b, c, ...) and,
  within each, the individual trajectories to overlay (e.g. different
  concentrations/lambda values, colored and labeled separately, exactly like
  "xi = 0.7 / 0.85 / 1.0" in the reference figure). Only one topology/
  trajectory pair was supplied to this script, so SYSTEMS defaults to a
  single row/single trajectory using that pair -- edit SYSTEMS to point at
  your actual set of trajectories to reproduce the full multi-row, multi-
  condition figure.
- PHOSPHATE_ATOMNAME / WATER_OXYGEN_ATOMNAME: change if your force field
  names these atoms differently.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import MDAnalysis as mda

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# USER CONFIG (as supplied)
# ----------------------------------------------------------------------
topol_file = "md_0_1000.tpr"
traj_file = "center1.xtc"
solute_selection = "not resname TIP3 and not resname ION"
water_selection = "resname W"
ion_selection = "resname ION"
lipid_selection = "resname POPC POPG POPE DPPC DOPC"

OUTDIR = "pore_analysis_output"
os.makedirs(OUTDIR, exist_ok=True)

# --- Analysis parameters (tune as needed) ---
FRAME_STEP = 1                 # stride through trajectory (increase to speed up)
N_ZBINS = 60                   # z-bins for pore radius profile
PROBE_RADIUS = 1.4             # A, water probe radius subtracted in HOLE-style calc
PORE_LINING_CUTOFF = 10.0      # A, distance from pore axis to count as "pore lining"
PORE_LINING_MIN_ATOMS = 3      # min atoms within cutoff to count a peptide as lining
REFERENCE_RADII = {            # vertical reference lines for Image-1-style plot
    "grey": 2.0,
    "orange": 3.5,
    "black": 5.0,
}

# Selections you MUST customize for your system:
FILTER_SEL = None   # e.g. "protein and resid 120-130"
GATE_SEL = None      # e.g. "protein and resid 60-70"

# NOTE: water_selection = "resname W" indicates a MARTINI-style coarse-
# grained system (single-bead water), so the defaults below assume CG
# naming. If your system is all-atom, change these to "name P" and
# "name OW" respectively.
PHOSPHATE_ATOMNAME = "name PO4"        # within lipid_selection (MARTINI headgroup bead)
WATER_OXYGEN_ATOMNAME = None           # None = use full water_selection (CG water
                                        # has no separate oxygen atom); for all-atom
                                        # water set e.g. "name OW"

# Candidate atom names to try (in order) if PHOSPHATE_ATOMNAME matches nothing,
# so the script can self-correct across common all-atom / MARTINI naming schemes.
PHOSPHATE_ATOMNAME_CANDIDATES = ["PO4", "P", "P8", "PO1", "P1"]

# --- Section 3 config: systems (rows) x conditions (colored overlays) ---
# Each system becomes one row (a, b, c, ...) in the pore-lining grid figure.
# Each condition within a system becomes one color/legend entry (e.g. xi
# values, replicas, peptide:lipid ratios, etc.) overlaid on that row's
# time-series panel. Only one (topol, traj) pair was given to this script,
# so every entry below points at it by default -- replace `topol`/`traj`
# per condition with your actual files to reproduce the full figure.
SYSTEMS = [
    {
        "label": "a",
        "conditions": [
            {"name": "xi = 0.7",  "color": "#3b7dd8", "topol": topol_file, "traj": traj_file},
            {"name": "xi = 0.85", "color": "#4caf50", "topol": topol_file, "traj": traj_file},
            {"name": "xi = 1.0",  "color": "#e53935", "topol": topol_file, "traj": traj_file},
        ],
    },
    # {
    #     "label": "b",
    #     "conditions": [
    #         {"name": "xi = 0.7",  "color": "#3b7dd8", "topol": "...", "traj": "..."},
    #         {"name": "xi = 0.85", "color": "#4caf50", "topol": "...", "traj": "..."},
    #         {"name": "xi = 1.0",  "color": "#e53935", "topol": "...", "traj": "..."},
    #     ],
    # },
    # {
    #     "label": "c",
    #     "conditions": [ ... ],
    # },
]


def load_universe(topol=None, traj=None):
    topol = topol or topol_file
    traj = traj or traj_file
    print(f"Loading universe: {topol} / {traj}")
    u = mda.Universe(topol, traj)
    print(f"  {len(u.atoms)} atoms, {len(u.trajectory)} frames")
    return u


def guess_vdw_radii(atomgroup):
    """Rough element-based vdW radii (Angstrom) fallback."""
    table = {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
             "P": 1.80, "NA": 2.27, "CL": 1.75}
    elems = []
    for name in atomgroup.names:
        n = name.strip().upper()
        elem = "C"
        for key in table:
            if n.startswith(key):
                elem = key
                break
        elems.append(table.get(elem, 1.70))
    return np.array(elems)


# ----------------------------------------------------------------------
# 1. HOLE-style pore radius profile  (Image 1)
# ----------------------------------------------------------------------
def pore_radius_profile(u, sel=None, n_zbins=N_ZBINS, step=FRAME_STEP,
                         probe_radius=PROBE_RADIUS):
    """
    For each frame: find pore-center (xy COG of the solute selection),
    bin solute atoms by z, and take (min over atoms in bin of
    xy-distance-from-center minus vdw radius minus probe radius) as the
    pore radius in that bin. Average and std over frames.
    Returns z_centers (relative to solute COG), mean_radius, std_radius.
    """
    sel = sel or solute_selection
    solute = u.select_atoms(sel)
    if len(solute) == 0:
        raise ValueError(f"Selection '{sel}' matched 0 atoms")
    vdw = guess_vdw_radii(solute)

    all_z = []
    for ts in u.trajectory[::step]:
        all_z.append(solute.positions[:, 2].copy())
    all_z = np.concatenate(all_z)
    zmin, zmax = np.percentile(all_z, [1, 99])
    z_edges = np.linspace(zmin, zmax, n_zbins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    profiles = []
    for ts in u.trajectory[::step]:
        pos = solute.positions
        center_xy = pos[:, :2].mean(axis=0)
        r_xy = np.linalg.norm(pos[:, :2] - center_xy, axis=1) - vdw - probe_radius

        radius = np.full(n_zbins, np.nan)
        bin_idx = np.digitize(pos[:, 2], z_edges) - 1
        for b in range(n_zbins):
            mask = bin_idx == b
            if mask.any():
                radius[b] = np.nanmin(r_xy[mask])
        profiles.append(radius)

    profiles = np.array(profiles)
    mean_r = np.nanmean(profiles, axis=0)
    std_r = np.nanstd(profiles, axis=0)
    z_rel = z_centers - np.nanmean(all_z)
    return z_rel, mean_r, std_r


def plot_pore_radius_profile(z_rel, mean_r, std_r, outpath, orientation="vertical",
                              zoom=None, title="Pore radius profile"):
    """orientation='vertical' -> radius on x, z on y (Image 1 style).
       orientation='horizontal' -> z on x, radius on y (Image 2-left style)."""
    fig, ax = plt.subplots(figsize=(3.2, 4.2) if orientation == "vertical" else (5, 3.2))

    if orientation == "vertical":
        ax.plot(mean_r, z_rel, color="royalblue", lw=1.5)
        ax.fill_betweenx(z_rel, mean_r - std_r, mean_r + std_r,
                          color="royalblue", alpha=0.25)
        for color, val in REFERENCE_RADII.items():
            style = "--" if color != "black" else ":"
            ax.axvline(val, color=color, linestyle=style, lw=1)
        ax.set_xlabel("Pore radius (\u00c5)")
        ax.set_ylabel("z (\u00c5)")
        ax.set_xlim(0, max(15, np.nanmax(mean_r + std_r) * 1.1))
        if zoom:
            ax.set_ylim(*zoom)
        ax.axhline(0, color="k", ls="--", lw=0.7, alpha=0.5)
    else:
        ax.plot(z_rel, mean_r, color="crimson", lw=1.5)
        ax.fill_between(z_rel, mean_r - std_r, mean_r + std_r,
                         color="crimson", alpha=0.2)
        ax.set_xlabel("z (\u00c5)")
        ax.set_ylabel("Pore radius R (\u00c5)")
        if zoom:
            ax.set_xlim(*zoom)

    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ----------------------------------------------------------------------
# 2. Filter / Gate z-position tracking  (Image 2)
# ----------------------------------------------------------------------
def filter_gate_zscatter(u, filter_sel, gate_sel, step=FRAME_STEP):
    if filter_sel is None or gate_sel is None:
        print("  [skip] FILTER_SEL / GATE_SEL not set -- edit the script config.")
        return None

    filt = u.select_atoms(filter_sel)
    gate = u.select_atoms(gate_sel)
    if len(filt) == 0 or len(gate) == 0:
        print("  [skip] FILTER_SEL or GATE_SEL matched 0 atoms.")
        return None

    frames, filt_z, gate_z = [], [], []
    for ts in u.trajectory[::step]:
        frames.append(ts.frame)
        filt_z.append(filt.positions[:, 2].copy())
        gate_z.append(gate.positions[:, 2].copy())

    return np.array(frames), filt_z, gate_z


def plot_filter_gate(frames, filt_z, gate_z, outpath):
    fig, ax = plt.subplots(figsize=(6, 4))
    for f, zvals in zip(frames, filt_z):
        ax.scatter([f] * len(zvals), zvals, s=4, color="darkred", alpha=0.6,
                   label="Filter" if f == frames[0] else None)
    for f, zvals in zip(frames, gate_z):
        ax.scatter([f] * len(zvals), zvals, s=4, color="seagreen", alpha=0.6,
                   label="Gate" if f == frames[0] else None)
    ax.set_xlabel("Frame")
    ax.set_ylabel("z [\u00c5]")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ----------------------------------------------------------------------
# 3. Pore-lining peptides: time series + top/side snapshots  (new image)
# ----------------------------------------------------------------------
def iter_peptides(u, sel=None):
    """Yield (segid, AtomGroup) for each distinct peptide/segment in sel."""
    sel = sel or solute_selection
    solute = u.select_atoms(sel)
    segids = np.unique(solute.segids) if hasattr(solute, "segids") else [None]
    if len(segids) <= 1:
        # fall back to per-molecule grouping via fragments (bonded connected components)
        for i, frag in enumerate(solute.fragments):
            yield f"frag{i}", frag
    else:
        for sg in segids:
            yield sg, solute.select_atoms(f"segid {sg}")


def pore_lining_peptides(u, sel=None, cutoff=PORE_LINING_CUTOFF, step=FRAME_STEP,
                          min_atoms=PORE_LINING_MIN_ATOMS):
    """Returns (time_ns array, n_lining array, peptides list)."""
    sel = sel or solute_selection
    peptides = list(iter_peptides(u, sel))

    times, counts = [], []
    for ts in u.trajectory[::step]:
        solute_all = u.select_atoms(sel)
        center_xy = solute_all.positions[:, :2].mean(axis=0)
        n_lining = 0
        for segid, ag in peptides:
            r_xy = np.linalg.norm(ag.positions[:, :2] - center_xy, axis=1)
            if (r_xy < cutoff).sum() >= min_atoms:
                n_lining += 1
        times.append(ts.time / 1000.0)  # ps -> ns
        counts.append(n_lining)

    return np.array(times), np.array(counts), peptides


def lining_segments_at_frame(u, peptides, sel=None, cutoff=PORE_LINING_CUTOFF,
                              min_atoms=PORE_LINING_MIN_ATOMS, frame=-1):
    """AtomGroups (per peptide) that are pore-lining at a given frame."""
    sel = sel or solute_selection
    u.trajectory[frame]
    solute_all = u.select_atoms(sel)
    center_xy = solute_all.positions[:, :2].mean(axis=0)

    lining = []
    for segid, ag in peptides:
        r_xy = np.linalg.norm(ag.positions[:, :2] - center_xy, axis=1)
        if (r_xy < cutoff).sum() >= min_atoms:
            lining.append((segid, ag))
    return lining


def draw_top_view(ax, u, lipid_sel, lining_segs, frame=-1):
    u.trajectory[frame]
    lipids = u.select_atoms(lipid_sel)
    ax.scatter(lipids.positions[:, 0], lipids.positions[:, 1],
               s=4, color="dimgray", alpha=0.6, linewidths=0)
    for segid, ag in lining_segs:
        bb = ag.select_atoms("name CA") if len(ag.select_atoms("name CA")) > 1 else ag
        order = np.argsort(bb.positions[:, 2])
        ax.plot(bb.positions[order, 0], bb.positions[order, 1],
                color="darkorange", lw=3, solid_capstyle="round")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def draw_side_view(ax, u, water_sel, lining_segs, frame=-1):
    u.trajectory[frame]
    waters = u.select_atoms(water_sel)
    ax.scatter(waters.positions[:, 0], waters.positions[:, 2],
               s=3, color="mediumblue", alpha=0.6, linewidths=0)
    for segid, ag in lining_segs:
        bb = ag.select_atoms("name CA") if len(ag.select_atoms("name CA")) > 1 else ag
        order = np.argsort(bb.positions[:, 2])
        ax.plot(bb.positions[order, 0], bb.positions[order, 2],
                color="darkorange", lw=3, solid_capstyle="round")
    ax.set_xticks([]); ax.set_yticks([])


def pore_lining_peptides_grid(systems=SYSTEMS, sel=None, lipid_sel=None,
                               water_sel=None, cutoff=PORE_LINING_CUTOFF,
                               step=FRAME_STEP,
                               outpath=os.path.join(OUTDIR, "03_pore_lining_grid.png")):
    """
    Build the multi-row (system a, b, c, ...) x 3-column
    (time series | top view | side view) figure, with one color per
    condition (e.g. xi value) overlaid on each row's time-series panel.
    Also writes one CSV of counts per system/condition.
    """
    sel = sel or solute_selection
    lipid_sel = lipid_sel or lipid_selection
    water_sel = water_sel or water_selection

    n_rows = len(systems)
    fig = plt.figure(figsize=(11, 3.4 * n_rows))
    gs = gridspec.GridSpec(n_rows, 3, width_ratios=[1.3, 1, 1], figure=fig)

    csv_rows = ["system,condition,time_ns,n_pore_lining_peptides"]

    for i, system in enumerate(systems):
        label = system.get("label", chr(ord("a") + i))
        conditions = system["conditions"]

        ax_ts = fig.add_subplot(gs[i, 0])
        ax_top = fig.add_subplot(gs[i, 1])
        ax_side = fig.add_subplot(gs[i, 2])

        best_u, best_peptides, best_lining, best_count = None, None, [], -1

        for cond in conditions:
            u_c = load_universe(cond["topol"], cond["traj"])
            times, counts, peptides = pore_lining_peptides(
                u_c, sel=sel, cutoff=cutoff, step=step)
            ax_ts.scatter(times, counts, s=10, color=cond["color"],
                          label=cond["name"] if i == 0 else None)

            for t, c in zip(times, counts):
                csv_rows.append(f"{label},{cond['name']},{t},{c}")

            # keep the trajectory/frame with the most pore-lining peptides
            # for the representative top/side-view snapshots of this row
            if counts.max(initial=0) > best_count:
                lining = lining_segments_at_frame(u_c, peptides, sel=sel,
                                                    cutoff=cutoff, frame=-1)
                if len(lining) >= counts.max(initial=0):
                    best_u, best_peptides, best_lining = u_c, peptides, lining
                    best_count = counts.max(initial=0)

        ax_ts.set_ylabel("Pore-Lining Peptides")
        ax_ts.set_ylim(-0.3, 6.5)
        if i == n_rows - 1:
            ax_ts.set_xlabel("time (ns)")
        if i == 0:
            ax_ts.legend(loc="upper left", bbox_to_anchor=(0, 1.25), ncol=3,
                         frameon=False, fontsize=9)
        ax_ts.text(-0.18, 1.02, label, transform=ax_ts.transAxes,
                   fontsize=14, fontweight="bold", va="bottom")

        if best_u is not None:
            draw_top_view(ax_top, best_u, lipid_sel, best_lining, frame=-1)
            draw_side_view(ax_side, best_u, water_sel, best_lining, frame=-1)
        else:
            ax_top.axis("off")
            ax_side.axis("off")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")

    with open(os.path.join(OUTDIR, "pore_lining_grid.csv"), "w") as f:
        f.write("\n".join(csv_rows))
    print(f"  wrote {os.path.join(OUTDIR, 'pore_lining_grid.csv')}")


# ----------------------------------------------------------------------
# 4. Lipid headgroup / water density maps  (Image 4)
# ----------------------------------------------------------------------
def lipid_water_density_maps(u, lipid_sel=None, water_sel=None, solute_sel=None,
                              step=FRAME_STEP, r_max_nm=5.0, z_max_nm=3.5,
                              nbins=60, outpath=os.path.join(OUTDIR, "04_density_maps.png")):
    lipid_sel = lipid_sel or lipid_selection
    water_sel = water_sel or water_selection
    solute_sel = solute_sel or solute_selection

    lipids_all = u.select_atoms(lipid_sel)
    if len(lipids_all) == 0:
        raise ValueError(f"No atoms matched lipid selection '{lipid_sel}'")

    phos_sel_str = f"({lipid_sel}) and {PHOSPHATE_ATOMNAME}"
    phosphates = u.select_atoms(phos_sel_str)

    if len(phosphates) == 0:
        print(f"  [warn] '{phos_sel_str}' matched 0 atoms, "
              f"trying fallback candidates {PHOSPHATE_ATOMNAME_CANDIDATES}...")
        for cand in PHOSPHATE_ATOMNAME_CANDIDATES:
            trial = lipids_all.select_atoms(f"name {cand}")
            if len(trial) > 0:
                phosphates = trial
                print(f"  [info] using 'name {cand}' instead "
                      f"({len(trial)} atoms matched)")
                break

    if len(phosphates) == 0:
        available = sorted(set(lipids_all.names))
        raise ValueError(
            f"No phosphate-like atoms found in lipid selection '{lipid_sel}'.\n"
            f"  Tried: {PHOSPHATE_ATOMNAME!r} and candidates {PHOSPHATE_ATOMNAME_CANDIDATES}.\n"
            f"  Atom names actually present in your lipid selection:\n  {available}\n"
            f"  Set PHOSPHATE_ATOMNAME (or add to PHOSPHATE_ATOMNAME_CANDIDATES) "
            f"to one of these."
        )

    if WATER_OXYGEN_ATOMNAME:
        water_ox_str = f"({water_sel}) and {WATER_OXYGEN_ATOMNAME}"
        waters = u.select_atoms(water_ox_str)
        if len(waters) == 0:
            print(f"  [warn] no atoms matched '{water_ox_str}', "
                  f"falling back to full water selection")
            waters = u.select_atoms(water_sel)
    else:
        waters = u.select_atoms(water_sel)

    r_edges = np.linspace(0, r_max_nm, nbins + 1)
    z_edges = np.linspace(-z_max_nm, z_max_nm, nbins + 1)

    hist_p = np.zeros((nbins, nbins))
    hist_w = np.zeros((nbins, nbins))
    n_frames = 0

    for ts in u.trajectory[::step]:
        solute_all = u.select_atoms(solute_sel)
        center = solute_all.positions.mean(axis=0)

        for ag, hist in ((phosphates, hist_p), (waters, hist_w)):
            pos = ag.positions
            r_nm = np.linalg.norm(pos[:, :2] - center[:2], axis=1) / 10.0
            z_nm = (pos[:, 2] - center[2]) / 10.0
            h, _, _ = np.histogram2d(z_nm, r_nm, bins=[z_edges, r_edges])
            hist += h
        n_frames += 1

    hist_p /= n_frames
    hist_w /= n_frames

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, hist, title, cmap in (
        (axes[0], hist_p, "DMPC Head Phosphate", "Reds"),
        (axes[1], hist_w, "Water Oxygen", "Blues"),
    ):
        im = ax.pcolormesh(r_edges, z_edges, hist, cmap=cmap, shading="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Radius from Pore Center (nm)")
        ax.set_ylabel("Z Distance from Pore Center (nm)")
        ax.set_title(title, fontsize=10)
        # reference headgroup lines (approx, refine using your own bilayer)
        p_z = (phosphates.positions[:, 2] - center[2]) / 10.0
        for val in (np.percentile(p_z, 90), np.percentile(p_z, 10)):
            ax.axhline(val, color="k", lw=0.6)
        for val in (np.percentile(p_z, 99), np.percentile(p_z, 1)):
            ax.axhline(val, color="k", lw=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    u = load_universe()

    print("\n[1/4] Pore radius profile (Image 1 style)...")
    z_rel, mean_r, std_r = pore_radius_profile(u, solute_selection)
    np.savetxt(os.path.join(OUTDIR, "pore_radius_profile.csv"),
               np.column_stack([z_rel, mean_r, std_r]),
               header="z_rel,mean_radius,std_radius", delimiter=",", comments="")
    plot_pore_radius_profile(
        z_rel, mean_r, std_r,
        os.path.join(OUTDIR, "01_pore_radius_profile.png"),
        orientation="vertical", title="Pore radius profile")

    print("\n[2/4] Filter/Gate analysis (Image 2 style)...")
    plot_pore_radius_profile(
        z_rel, mean_r, std_r,
        os.path.join(OUTDIR, "02a_pore_radius_zoom.png"),
        orientation="horizontal",
        zoom=(np.nanpercentile(z_rel, 20), np.nanpercentile(z_rel, 90)),
        title="Pore radius vs z (channel region)")

    fg = filter_gate_zscatter(u, FILTER_SEL, GATE_SEL)
    if fg is not None:
        frames, filt_z, gate_z = fg
        plot_filter_gate(frames, filt_z, gate_z,
                          os.path.join(OUTDIR, "02b_filter_gate_zscatter.png"))

    print("\n[3/4] Pore-lining peptides grid (multi-system/condition figure)...")
    pore_lining_peptides_grid()

    print("\n[4/4] Lipid headgroup / water density maps (Image 4 style)...")
    lipid_water_density_maps(u)

    print(f"\nAll done. Outputs written to: {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()
