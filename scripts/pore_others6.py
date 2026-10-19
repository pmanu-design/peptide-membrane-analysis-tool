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
  3. pore_lining_peptides_grid()  -> [DISABLED by default -- not called from
                                      main()] multi-row x 3-column figure of
                                      pore-lining peptide counts + snapshots.
                                      The function is still defined below if
                                      you want to re-enable it; just call
                                      pore_lining_peptides_grid() again.
  4. lipid_water_density_maps()   -> 2D histograms of lipid headgroup
                                      phosphate and water oxygen density vs
                                      radius (\u00c5) and z-distance (\u00c5),
                                      both measured from the membrane
                                      midplane center (0, 0)
  5. residue_pore_radius_profile()-> pore radius vs residue number for one
                                      representative pore-lining chain, with
                                      constriction-site residues auto-labeled
                                      (e.g. Q29/I33/L37/...) and a span arrow
  6. axial_water_ion_profile()    -> pore radius, water occupancy, and ion
                                      occupancy (x10) overlaid vs position
                                      along the pore axis

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
- REFERENCE_PEPTIDE_SEL (section 5): which single chain's residues to plot
  along the x-axis. Defaults to auto-picking the first peptide/segment found;
  set explicitly (e.g. "segid PROA") if that's not the one that best traces
  the pore.
- AXIAL_RADIAL_CUTOFF (section 6): cylinder radius used to count water/ion
  occupancy per z-slice; tune to your pore's typical radius.
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

# --- Section 5 config: residue-level pore radius profile ---
# One representative pore-lining chain/peptide whose per-residue radius
# contribution traces the constriction sites along the pore (Q29/I33/etc.
# style labeling). None = auto-pick the first peptide/segment found in
# solute_selection.
REFERENCE_PEPTIDE_SEL = None   # e.g. "segid PROA" or "chainID A"
N_CONSTRICTION_LABELS = 7      # how many local-minima residues to label

# --- Section 6 config: axial water/ion occupancy profile ---
AXIAL_ZBINS = 60               # z-bins along the pore axis
AXIAL_RADIAL_CUTOFF = 6.0      # A, cylinder radius around the pore axis used
                                # to count water/ion occupancy per z-bin

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
                              zoom=None, title="Pore radius profile",
                              line_color="seagreen", fill_color="royalblue",
                              y_label="Distance along pore (\u00c5)",
                              show_reference_lines=False):
    """orientation='vertical' -> radius on x, z on y (Image 1 style).
       orientation='horizontal' -> z on x, radius on y (Image 2-left style)."""
    fig, ax = plt.subplots(figsize=(3.2, 4.2) if orientation == "vertical" else (5, 3.2))

    if orientation == "vertical":
        ax.plot(mean_r, z_rel, color=line_color, lw=1.8)
        ax.fill_betweenx(z_rel, mean_r - std_r, mean_r + std_r,
                          color=fill_color, alpha=0.35)
        if show_reference_lines:
            for color, val in REFERENCE_RADII.items():
                style = "--" if color != "black" else ":"
                ax.axvline(val, color=color, linestyle=style, lw=1)
        ax.set_xlabel("Pore radius (\u00c5)")
        ax.set_ylabel(y_label)
        ax.set_xlim(0, max(4.0, np.nanmax(mean_r + std_r) * 1.1))
        if zoom:
            ax.set_ylim(*zoom)
    else:
        ax.plot(z_rel, mean_r, color=line_color, lw=1.8)
        ax.fill_between(z_rel, mean_r - std_r, mean_r + std_r,
                         color=fill_color, alpha=0.35)
        ax.set_xlabel(y_label)
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
                              step=FRAME_STEP, r_max_ang=50.0, z_max_ang=35.0,
                              nbins=60, outpath=os.path.join(OUTDIR, "04_density_maps.png")):
    """
    2D histograms of lipid headgroup phosphate and water density vs:
      x: radial distance from the pore center (Angstrom), measured from the
         membrane midplane's xy centroid (mean xy of all lipid headgroup
         phosphate atoms)
      y: z-distance from the membrane midplane (Angstrom), where 0 = the
         bilayer center, defined per-frame as the mean z of the lipid
         headgroup phosphate atoms (both leaflets combined).
    Both axes are therefore centered on the same membrane-defined origin
    (0, 0) = membrane center, per frame.
    """
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

    r_edges = np.linspace(0, r_max_ang, nbins + 1)
    z_edges = np.linspace(-z_max_ang, z_max_ang, nbins + 1)

    hist_p = np.zeros((nbins, nbins))
    hist_w = np.zeros((nbins, nbins))
    n_frames = 0
    last_p_z_ang = None

    for ts in u.trajectory[::step]:
        membrane_center = phosphates.positions.mean(axis=0)
        center_xy = membrane_center[:2]
        membrane_center_z = membrane_center[2]

        for ag, hist in ((phosphates, hist_p), (waters, hist_w)):
            pos = ag.positions
            r_ang = np.linalg.norm(pos[:, :2] - center_xy, axis=1)
            z_ang = pos[:, 2] - membrane_center_z
            h, _, _ = np.histogram2d(z_ang, r_ang, bins=[z_edges, r_edges])
            hist += h
        n_frames += 1
        last_p_z_ang = phosphates.positions[:, 2] - membrane_center_z

    hist_p /= n_frames
    hist_w /= n_frames

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, hist, title, cmap in (
        (axes[0], hist_p, "Phosphate Head Group", "Reds"),
        (axes[1], hist_w, "Water Oxygen", "Blues"),
    ):
        im = ax.pcolormesh(r_edges, z_edges, hist, cmap=cmap, shading="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Radius from Pore Center (\u00c5)")
        ax.set_ylabel("Z Distance from Membrane Center (\u00c5)")
        ax.set_title(title, fontsize=10)
        # reference headgroup lines (approx, refine using your own bilayer),
        # from the last processed frame, relative to the membrane midplane
        for val in (np.percentile(last_p_z_ang, 90), np.percentile(last_p_z_ang, 10)):
            ax.axhline(val, color="k", lw=0.6)
        for val in (np.percentile(last_p_z_ang, 99), np.percentile(last_p_z_ang, 1)):
            ax.axhline(val, color="k", lw=0.6, alpha=0.6)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ----------------------------------------------------------------------
# 5. Residue-level pore radius profile with constriction labeling (Image 2)
# ----------------------------------------------------------------------
def pick_reference_peptide(u, sel=None):
    """Auto-pick the first peptide/segment from solute_selection to use as
    the residue-numbered reference chain lining the pore."""
    sel = sel or solute_selection
    peptides = list(iter_peptides(u, sel))
    if not peptides:
        raise ValueError(f"No peptides found in selection '{sel}'")
    segid, ag = peptides[0]
    print(f"  [info] REFERENCE_PEPTIDE_SEL not set -- auto-picked first "
          f"peptide/segment ({segid}, {len(ag)} atoms) as the reference chain")
    return ag


def residue_pore_radius_profile(u, solute_sel=None, reference_sel=None,
                                 step=FRAME_STEP, probe_radius=PROBE_RADIUS):
    """
    For each residue of a single reference (pore-lining) peptide, compute
    the pore radius contributed by that residue (min over the residue's
    atoms of xy-distance-from-pore-axis minus vdw minus probe), averaged
    (mean/std) over frames. Returns resids, resnames, mean_radius, std_radius.
    """
    solute_sel = solute_sel or solute_selection
    solute = u.select_atoms(solute_sel)

    if reference_sel:
        reference = u.select_atoms(reference_sel)
        if len(reference) == 0:
            raise ValueError(f"REFERENCE_PEPTIDE_SEL '{reference_sel}' matched 0 atoms")
    else:
        reference = pick_reference_peptide(u, solute_sel)

    resids = np.unique(reference.resids)
    resid_to_name = {}
    for r in reference.residues:
        resid_to_name[r.resid] = r.resname

    profiles = {r: [] for r in resids}
    for ts in u.trajectory[::step]:
        center_xy = solute.positions[:, :2].mean(axis=0)
        for rid in resids:
            res_atoms = reference.select_atoms(f"resid {rid}")
            if len(res_atoms) == 0:
                continue
            r_xy = np.linalg.norm(res_atoms.positions[:, :2] - center_xy, axis=1)
            vdw_res = guess_vdw_radii(res_atoms)
            profiles[rid].append(np.min(r_xy - vdw_res - probe_radius))

    resid_arr = np.array(sorted(resids))
    mean_r = np.array([np.mean(profiles[r]) if profiles[r] else np.nan for r in resid_arr])
    std_r = np.array([np.std(profiles[r]) if profiles[r] else np.nan for r in resid_arr])
    resnames = [resid_to_name.get(r, "X") for r in resid_arr]
    return resid_arr, resnames, mean_r, std_r


def find_local_minima(y, min_separation=2):
    """Simple local-minima finder (avoids a scipy dependency)."""
    minima = []
    n = len(y)
    for i in range(1, n - 1):
        if y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            if not minima or (i - minima[-1]) >= min_separation:
                minima.append(i)
            elif y[i] < y[minima[-1]]:
                minima[-1] = i
    return minima


def resname_3to1(resname):
    table = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
             "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
             "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
             "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
    return table.get(resname.upper(), resname[:1])


def plot_residue_pore_radius_profile(resid_arr, resnames, mean_r, std_r, outpath,
                                      n_labels=N_CONSTRICTION_LABELS):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(resid_arr, mean_r, color="blue", lw=1.8)
    ax.fill_between(resid_arr, mean_r - std_r, mean_r + std_r,
                     color="mediumseagreen", alpha=0.3)

    minima_idx = find_local_minima(mean_r, min_separation=max(1, len(resid_arr) // 15))
    # keep the deepest N minima, then re-sort left-to-right for labeling
    minima_idx = sorted(minima_idx, key=lambda i: mean_r[i])[:n_labels]
    minima_idx = sorted(minima_idx)

    for i in minima_idx:
        label = f"{resname_3to1(resnames[i])}{resid_arr[i]}"
        ax.annotate(label, (resid_arr[i], mean_r[i]),
                    textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=9, fontweight="bold")

    if len(minima_idx) >= 2:
        i0, i1 = minima_idx[0], minima_idx[-1]
        span_res = resid_arr[i1] - resid_arr[i0]
        y_arrow = 0.06 * (np.nanmax(mean_r + std_r))
        ax.annotate("", xy=(resid_arr[i1], y_arrow), xytext=(resid_arr[i0], y_arrow),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
        ax.text((resid_arr[i0] + resid_arr[i1]) / 2, y_arrow * 1.3,
                f"{span_res} residues", ha="center", fontsize=9)

    ax.set_xlabel("Residue number")
    ax.set_ylabel("Pore radius (\u00c5)")
    ax.set_ylim(0, max(5, np.nanmax(mean_r + std_r) * 1.15))
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ----------------------------------------------------------------------
# 6. Axial pore radius / water / ion occupancy profile (Image 3)
# ----------------------------------------------------------------------
def axial_water_ion_profile(u, solute_sel=None, water_sel=None, ion_sel=None,
                             n_zbins=AXIAL_ZBINS, step=FRAME_STEP,
                             radial_cutoff=AXIAL_RADIAL_CUTOFF,
                             probe_radius=PROBE_RADIUS):
    """
    Along the pore axis (z), compute per-z-bin:
      - pore radius (as in pore_radius_profile)
      - mean number of water molecules within radial_cutoff of the axis
      - mean number of ions within radial_cutoff of the axis
    Returns z_rel, radius, water_count, ion_count.
    """
    solute_sel = solute_sel or solute_selection
    water_sel = water_sel or water_selection
    ion_sel = ion_sel or ion_selection

    solute = u.select_atoms(solute_sel)
    waters = u.select_atoms(water_sel)
    ions = u.select_atoms(ion_sel)
    vdw = guess_vdw_radii(solute)

    all_z = []
    for ts in u.trajectory[::step]:
        all_z.append(solute.positions[:, 2].copy())
    all_z = np.concatenate(all_z)
    zmin, zmax = np.percentile(all_z, [1, 99])
    z_edges = np.linspace(zmin, zmax, n_zbins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    radius_profiles, water_counts, ion_counts = [], [], []
    for ts in u.trajectory[::step]:
        pos = solute.positions
        center_xy = pos[:, :2].mean(axis=0)
        r_xy_solute = np.linalg.norm(pos[:, :2] - center_xy, axis=1) - vdw - probe_radius
        bin_idx_solute = np.digitize(pos[:, 2], z_edges) - 1

        radius = np.full(n_zbins, np.nan)
        for b in range(n_zbins):
            mask = bin_idx_solute == b
            if mask.any():
                radius[b] = np.nanmin(r_xy_solute[mask])
        radius_profiles.append(radius)

        for ag, counts_list in ((waters, water_counts), (ions, ion_counts)):
            if len(ag) == 0:
                counts_list.append(np.zeros(n_zbins))
                continue
            p = ag.positions
            r_xy = np.linalg.norm(p[:, :2] - center_xy, axis=1)
            in_cyl = r_xy < radial_cutoff
            bin_idx = np.digitize(p[:, 2], z_edges) - 1
            counts = np.zeros(n_zbins)
            for b in range(n_zbins):
                counts[b] = np.sum(in_cyl & (bin_idx == b))
            counts_list.append(counts)

    radius = np.nanmean(radius_profiles, axis=0)
    water_count = np.mean(water_counts, axis=0)
    ion_count = np.mean(ion_counts, axis=0)
    z_rel = z_centers - np.nanmean(all_z)
    return z_rel, radius, water_count, ion_count


def plot_axial_water_ion_profile(z_rel, radius, water_count, ion_count, outpath,
                                  ion_scale=10):
    fig, ax = plt.subplots(figsize=(4, 5))

    l1, = ax.plot(radius, z_rel, color="grey", lw=1.6, label="radius")
    l2, = ax.plot(water_count, z_rel, color="blue", lw=1.4, label="water")
    l3, = ax.plot(ion_count * ion_scale, z_rel, color="red", lw=1.4,
                  label=f"ions\u00d7{ion_scale}")

    ax.set_xlabel("Radius (\u00c5)  /  count")
    ax.set_ylabel("Pore axis (\u00c5)")
    ax.set_xlim(0, max(15, np.nanmax(radius) * 1.2))
    ax.legend(handles=[l1, l2, l3], loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    u = load_universe()

    print("\n[1/5] Pore radius profile (Image 1 style)...")
    z_rel, mean_r, std_r = pore_radius_profile(u, solute_selection)
    np.savetxt(os.path.join(OUTDIR, "pore_radius_profile.csv"),
               np.column_stack([z_rel, mean_r, std_r]),
               header="z_rel,mean_radius,std_radius", delimiter=",", comments="")
    plot_pore_radius_profile(
        z_rel, mean_r, std_r,
        os.path.join(OUTDIR, "01_pore_radius_profile.png"),
        orientation="vertical", title="Pore radius profile")

    print("\n[2/5] Filter/Gate analysis (Image 2 style)...")
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

    print("\n[3/5] Lipid headgroup / water density maps (Image 4 style)...")
    lipid_water_density_maps(u)

    print("\n[4/5] Residue-level pore radius profile with constriction labels "
          "(Image 2 style)...")
    resid_arr, resnames, res_mean_r, res_std_r = residue_pore_radius_profile(
        u, solute_selection, REFERENCE_PEPTIDE_SEL)
    np.savetxt(os.path.join(OUTDIR, "residue_pore_radius_profile.csv"),
               np.column_stack([resid_arr, res_mean_r, res_std_r]),
               header="resid,mean_radius,std_radius", delimiter=",", comments="")
    plot_residue_pore_radius_profile(
        resid_arr, resnames, res_mean_r, res_std_r,
        os.path.join(OUTDIR, "05_residue_pore_radius_profile.png"))

    print("\n[5/5] Axial pore radius / water / ion occupancy profile "
          "(Image 3 style)...")
    z_rel2, radius2, water_count, ion_count = axial_water_ion_profile(u)
    np.savetxt(os.path.join(OUTDIR, "axial_water_ion_profile.csv"),
               np.column_stack([z_rel2, radius2, water_count, ion_count]),
               header="z_rel,radius,water_count,ion_count",
               delimiter=",", comments="")
    plot_axial_water_ion_profile(
        z_rel2, radius2, water_count, ion_count,
        os.path.join(OUTDIR, "06_axial_water_ion_profile.png"))

    print(f"\nAll done. Outputs written to: {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()
