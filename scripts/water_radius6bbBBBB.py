import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.align import AlignTraj, alignto
from MDAnalysis.lib.distances import capped_distance
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binned_statistic, pearsonr

# =====================================================================
# USER PARAMETERS
# =====================================================================
topol_file = "md_0_300.tpr"
traj_file = "cen300.xtc"

solute_selection = "not resname TIP3 and not resname CLA and not resname POT"
water_selection = "resname TIP3"
output_prefix = "nonprotein_water_analysis"

# Every output (csv + png) is written inside this folder, created fresh.
output_dir = f"{output_prefix}_outputs"

align_solute = True

shell_inner = 0
shell_outer = 15

apply_kde_smoothing = True
kde_sigma = 1.0

enable_leaflet_analysis = True

# ---- purely-visual shading boundaries for the two-panel z-profile figures
# (average water-in-pore, pore radius). These do NOT affect any analysis --
# they only draw the gray "Inner leaflet" / "Outer leaflet" bands and the
# black boundary lines on those figures, mimicking typical lipid headgroup
# positions relative to the bilayer midplane (z = 0).
leaflet_shade_inner_z = 10.0   # Angstrom, start of the leaflet headgroup band
leaflet_shade_outer_z = 16.7   # Angstrom, end of the leaflet headgroup band

# Which leaflet-split array feeds panel (b) of each two-panel figure below.
# "upper" uses the mean_upper_* arrays, "lower" uses the mean_lower_* arrays.
water_panel_b_side = "upper"
pore_radius_panel_b_side = "lower"

z_bins = np.linspace(-20, 20, 50)
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

# ---- peptide-insertion / pore / lifetime parameters ----
# A frame is classed as "peptide-inserted" (peptide actively interacting
# with / spanning the bilayer) when BOTH conditions hold:
#   1) the peptide's z-center of mass lies within the instantaneous
#      leaflet-to-leaflet envelope (+/- insertion_z_margin), and
#   2) at least insertion_contact_fraction of protein residues are in
#      contact (< contact cutoff, reused from below) with the membrane.
#
# ALL pore / pore-water / local-thickness / thinning / disruption /
# permeation analysis below is computed ONLY for frames that pass this
# test. Frames where the peptide is not inserted contribute nothing to
# those metrics (stored as NaN) -- there is no "peptide not inserted"
# comparison branch anywhere in this script.
insertion_z_margin = 3.0          # Angstrom, tolerance around leaflet envelope
insertion_contact_fraction = 0.10  # 10% of residues in membrane contact

# Pore-lining/axis definitions
pore_probe_radius = 3.0            # Angstrom, cylinder radius used to count
                                    # "pore waters" and to test for lipid
                                    # exclusion around the pore axis
midplane_half_width = 5.0          # Angstrom, band around the bilayer
                                    # midplane in which the pore is scored
local_thickness_radius = 10.0      # Angstrom, xy radius around the pore
                                    # axis used to compute *local* bilayer
                                    # thickness (for the disruption metric)
bulk_ring_inner = 25.0             # Angstrom, inner radius of the "bulk"
bulk_ring_outer = 35.0             # Angstrom, outer radius of the "bulk"
                                    # annulus used as the undisturbed
                                    # reference thickness

# Pore "open" criteria (frame is counted as having an open pore if EITHER
# is satisfied):
pore_open_water_threshold = 3      # >= this many waters in the probe
                                    # cylinder within the midplane band
pore_open_radius_threshold = 4.0   # OR nearest-lipid radius from axis
                                    # (within midplane band) exceeds this

# Membrane-disruption thresholds, expressed as % local thinning relative
# to the bulk annulus thickness (local vs bulk leaflet-to-leaflet distance)
thinning_minor_pct = 15.0
thinning_moderate_pct = 35.0
# A pore is called "stable" (vs transient) if its lifetime exceeds this
stable_pore_lifetime_ns = 2.0

contact_cutoff = 4.5  # Angstrom, protein-membrane heavy-atom contact cutoff

# ---- snapshot / "insertion over time" figure parameters ----
n_snapshots = 6                 # number of timepoints in the snapshot strip
color_lipid_normal = "orange"   # phospholipids
color_lipid_deflected = "red"   # phospholipids in contact with the peptide
color_pore_water = "cyan"       # midplane-penetrating ("in-pore") waters
color_backbone = "lightgray"
color_nterm = "navy"
color_cterm = "darkred"

# =====================================================================
# File checks
# =====================================================================
if not os.path.isfile(topol_file):
    raise FileNotFoundError(f"Missing topology file: {topol_file}")
if not os.path.isfile(traj_file):
    raise FileNotFoundError(f"Missing trajectory file: {traj_file}")

os.makedirs(output_dir, exist_ok=True)


def outpath(name):
    """Every artifact goes into the single created output folder."""
    return os.path.join(output_dir, name)


def save_fig(fig, name, dpi=300):
    """Save a figure as its own individual file and close it."""
    fig.tight_layout()
    fig.savefig(outpath(name), dpi=dpi)
    plt.close(fig)


def binned_min(x, values, bins):
    """Fast vectorized replacement for a per-bin python loop computing the
    minimum of `values` for points falling in each bin of `x`. Empty bins
    are NaN (matches the behavior of the original manual loop)."""
    result = binned_statistic(x, values, statistic="min", bins=bins).statistic
    return result


def compute_contacts(group_a, group_b, cutoff, box, n_group_a_residues,
                      a_atom_to_res, b_n_atoms, b_atom_to_res, n_group_b_residues):
    """Vectorized protein-membrane contact detection using a cell-list
    neighbor search (capped_distance) instead of one distance_array call
    per residue. Returns:
      residue_contact_bool   : bool array, len n_group_a_residues
      b_atom_contacted_bool  : bool array, len b_n_atoms
      b_residue_contact_bool : bool array, len n_group_b_residues
    """
    residue_contact_bool = np.zeros(n_group_a_residues, dtype=bool)
    b_atom_contacted_bool = np.zeros(b_n_atoms, dtype=bool)
    b_residue_contact_bool = np.zeros(n_group_b_residues, dtype=bool)

    if group_a.n_atoms == 0 or group_b.n_atoms == 0:
        return residue_contact_bool, b_atom_contacted_bool, b_residue_contact_bool

    pairs, _ = capped_distance(
        group_a.positions, group_b.positions, max_cutoff=cutoff, box=box,
        return_distances=True,
    )
    if pairs.size:
        a_idx = np.unique(pairs[:, 0])
        b_idx = np.unique(pairs[:, 1])
        residue_contact_bool[np.unique(a_atom_to_res[a_idx])] = True
        b_atom_contacted_bool[b_idx] = True
        b_residue_contact_bool[np.unique(b_atom_to_res[b_idx])] = True
    return residue_contact_bool, b_atom_contacted_bool, b_residue_contact_bool


# === Load trajectory ===
u = mda.Universe(topol_file, traj_file)

solute = u.select_atoms(solute_selection)
if solute.n_atoms == 0:
    raise ValueError(f"No atoms found for solute selection: '{solute_selection}'")

waters = u.select_atoms(water_selection)
if waters.n_atoms == 0:
    raise ValueError(f"No atoms found for water selection: '{water_selection}'")

protein = u.select_atoms("protein")
membrane = u.select_atoms("not resname TIP3 CLA POT TIP3 and not protein")
if protein.n_atoms == 0:
    raise ValueError("No protein atoms found with selection 'protein'")
if membrane.n_atoms == 0:
    raise ValueError("No membrane atoms found. Check your selection.")

n_residues = len(protein.residues)
lipid_residues = membrane.residues
n_lipid_residues = len(lipid_residues)

# map every membrane atom -> index of its residue within `lipid_residues`
_, mem_atom_to_res = np.unique(membrane.resindices, return_inverse=True)
# map every protein atom -> local index (0..n_residues-1) within protein.residues
_, protein_atom_to_res = np.unique(protein.resindices, return_inverse=True)

# peptide backbone / termini for the snapshot figure
backbone = protein.select_atoms("name CA")
if backbone.n_atoms == 0:
    backbone = protein

nterm_atoms = protein.residues[0].atoms.select_atoms("name CA")
if nterm_atoms.n_atoms == 0:
    nterm_atoms = protein.residues[0].atoms[:1]
cterm_atoms = protein.residues[-1].atoms.select_atoms("name CA")
if cterm_atoms.n_atoms == 0:
    cterm_atoms = protein.residues[-1].atoms[:1]

# representative atom per water molecule (oxygen), used for permeation
# tracking -- much cheaper than following every H atom individually.
water_o = waters.select_atoms("name OH2 or name OW or name O")
if water_o.n_atoms != waters.n_residues:
    # fallback: first atom of every water residue (guaranteed one atom
    # per water molecule regardless of the force field's oxygen naming)
    first_atom_indices = [res.atoms.ix[0] for res in waters.residues]
    water_o = u.atoms[first_atom_indices]
n_water_mol = water_o.n_atoms

# Frame time step (ns), used for lifetime calculations
try:
    dt_ns = u.trajectory.dt / 1000.0
    if dt_ns <= 0:
        dt_ns = 1.0
except Exception:
    dt_ns = 1.0

n_frames = len(u.trajectory)

# =====================================================================
# Alignment
# =====================================================================
# NOTE: the previous version called `alignto(u, ref, select=...)` a single
# time *before* the trajectory loop, which only aligns whatever single
# frame `u` happened to be on to a copy of itself -- a no-op that leaves
# every frame in the trajectory unaligned.
#
# AlignTraj(..., in_memory=True) actually aligns every frame to the
# reference AND loads the aligned coordinates into a RAM array, which
# makes every subsequent `for ts in u.trajectory` pass read from memory
# instead of re-parsing the trajectory file -- a nice speedup for
# trajectories that fit in RAM. But for large systems (many frames x many
# atoms) that array can be tens of GiB and the allocation fails outright.
# So: try the fast in-memory path first, and if it doesn't fit, fall back
# to aligning each frame on-the-fly inside the main loop below (a bit
# slower per frame since it re-reads from disk, but uses negligible
# memory and still fixes the original no-op alignment bug).
aligned_in_memory = False
ref = None
if align_solute:
    print(f"[INFO] Aligning all frames using selection: '{solute_selection}'...")
    t0 = time.time()
    ref = u.copy()
    try:
        AlignTraj(u, ref, select=solute_selection, in_memory=True).run()
        aligned_in_memory = True
        print(f"[INFO] Alignment done in {time.time() - t0:.1f} s "
              f"(trajectory now held in memory for fast re-reads).")
    except MemoryError:
        approx_gib = (u.trajectory.n_frames * u.atoms.n_atoms * 3 * 4) / (1024 ** 3)
        print(f"[WARN] Not enough RAM to hold the full trajectory in memory for "
              f"alignment (~{approx_gib:.1f} GiB needed for "
              f"{u.trajectory.n_frames} frames x {u.atoms.n_atoms} atoms). "
              f"Falling back to on-the-fly per-frame alignment -- slightly slower, "
              f"but uses negligible memory. Frames will still be re-read from disk "
              f"once per pass, same as the original script.")

# === Result containers (existing) ===
num_waters_per_bin = []
radius_per_bin = []
pore_radius_per_bin = []
residue_contacts = {res.resindex: [] for res in protein.residues}

# === Result containers (pore / insertion / disruption) ===
peptide_inserted_flags = []      # bool per frame
pore_axis_xy_series = []         # xy axis used per frame; NaN when not inserted
bilayer_midplane_series = []     # dynamic bilayer midplane z per frame (all frames)
pore_water_count = []            # waters in probe cylinder, midplane band (NaN if not inserted)
pore_lipid_min_radius = []       # nearest-lipid radial distance from axis (NaN if not inserted)
local_thickness = []             # local (near-axis) leaflet-leaflet distance (NaN if not inserted)
bulk_thickness = []              # bulk annulus leaflet-leaflet distance (NaN if not inserted)
thinning_pct_series = []         # % thinning relative to bulk (NaN if not inserted)
contact_fraction_series = []     # fraction of protein residues in contact
n_deflected_lipids_series = []   # count of lipid residues contacting peptide
frac_deflected_lipids_series = []  # fraction of lipid residues contacting peptide

# === Result containers (new analyses) ===
insertion_depth_series = []      # peptide COM z minus bilayer midplane (signed, A)
peptide_tilt_series = []         # angle (deg) between N->C vector and membrane normal (0=spanning, 90=in-plane)
permeation_count_series = np.zeros(n_frames, dtype=int)  # water crossings this frame
permeation_events = []           # list of dicts, one row per crossing event

if enable_leaflet_analysis:
    upper_waters, lower_waters = [], []
    upper_pore_radius, lower_pore_radius = [], []

# permeation state machine (persisted across frames)
_last_side = np.zeros(n_water_mol, dtype=int)

print(f"[INFO] Beginning frame-by-frame analysis over {n_frames} frames...")
t_start = time.time()
progress_every = max(1, n_frames // 10)

for frame_i, ts in enumerate(u.trajectory):
    if align_solute and not aligned_in_memory:
        alignto(u, ref, select=solute_selection)

    solute_pos = solute.positions
    center_z = solute_pos[:, 2].mean()
    center_xy = solute_pos[:, :2].mean(axis=0)

    water_pos = waters.positions
    dist_xy = np.linalg.norm(water_pos[:, :2] - center_xy, axis=1)
    dist_z = water_pos[:, 2] - center_z

    in_shell = (dist_xy >= shell_inner) & (dist_xy <= shell_outer)
    filtered_z = dist_z[in_shell]

    water_hist, _ = np.histogram(filtered_z, bins=z_bins)
    num_waters_per_bin.append(water_hist)

    xy_dist = np.linalg.norm(solute_pos[:, :2] - center_xy, axis=1)
    solute_z_rel = solute_pos[:, 2] - center_z

    weighted_radius, _ = np.histogram(solute_z_rel, bins=z_bins, weights=xy_dist)
    bin_counts, _ = np.histogram(solute_z_rel, bins=z_bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_radius = np.nan_to_num(weighted_radius / bin_counts)
    radius_per_bin.append(avg_radius)

    # vectorized per-bin minimum radius (replaces a 49-iteration python loop)
    pore_radius_per_bin.append(binned_min(solute_z_rel, xy_dist, z_bins))

    # ---- Contact analysis (vectorized cell-list search, replaces one
    # distance_array() call per protein residue with a single capped_
    # distance() call for the whole protein-membrane pair) ----
    contact_counts, membrane_atom_contacted, deflected_res_mask = compute_contacts(
        protein, membrane, contact_cutoff, u.dimensions,
        n_residues, protein_atom_to_res,
        membrane.n_atoms, mem_atom_to_res, n_lipid_residues,
    )
    for i, res in enumerate(protein.residues):
        residue_contacts[res.resindex].append(contact_counts[i])
    contact_fraction = contact_counts.sum() / n_residues if n_residues else 0.0
    contact_fraction_series.append(contact_fraction)

    n_deflected = int(deflected_res_mask.sum())
    n_deflected_lipids_series.append(n_deflected)
    frac_deflected_lipids_series.append(n_deflected / n_lipid_residues if n_lipid_residues else 0.0)

    if enable_leaflet_analysis:
        upper_mask = dist_z > 0
        lower_mask = dist_z < 0
        filtered_z_upper = dist_z[in_shell & upper_mask]
        filtered_z_lower = dist_z[in_shell & lower_mask]

        upper_water_hist, _ = np.histogram(filtered_z_upper, bins=z_bins)
        lower_water_hist, _ = np.histogram(filtered_z_lower, bins=z_bins)
        upper_waters.append(upper_water_hist)
        lower_waters.append(lower_water_hist)

        upper_sel = solute_z_rel > 0
        lower_sel = solute_z_rel < 0
        upper_pore_radius.append(binned_min(solute_z_rel[upper_sel], xy_dist[upper_sel], z_bins))
        lower_pore_radius.append(binned_min(solute_z_rel[lower_sel], xy_dist[lower_sel], z_bins))

    # =================================================================
    # bilayer midplane (dynamic, leaflet-derived) -- computed every frame,
    # this is bilayer geometry, not a peptide-insertion-dependent quantity
    # =================================================================
    mem_pos = membrane.positions
    mem_z_all = mem_pos[:, 2]
    mem_center_estimate = mem_z_all.mean()
    mem_upper_mask = mem_z_all > mem_center_estimate
    mem_lower_mask = ~mem_upper_mask
    upper_leaflet_z = mem_z_all[mem_upper_mask]
    lower_leaflet_z = mem_z_all[mem_lower_mask]
    upper_mean_z = upper_leaflet_z.mean() if upper_leaflet_z.size else mem_center_estimate
    lower_mean_z = lower_leaflet_z.mean() if lower_leaflet_z.size else mem_center_estimate
    bilayer_midplane_z = 0.5 * (upper_mean_z + lower_mean_z)
    bilayer_midplane_series.append(bilayer_midplane_z)

    # is the peptide's z-center within the leaflet envelope?
    peptide_z_in_envelope = (
        (lower_mean_z - insertion_z_margin) <= center_z <= (upper_mean_z + insertion_z_margin)
    )
    peptide_inserted = bool(peptide_z_in_envelope and (contact_fraction >= insertion_contact_fraction))
    peptide_inserted_flags.append(peptide_inserted)

    # ---- peptide insertion depth & tilt (new analyses) ----
    insertion_depth_series.append(center_z - bilayer_midplane_z)
    term_vec = (cterm_atoms.positions.mean(axis=0) - nterm_atoms.positions.mean(axis=0))
    term_norm = np.linalg.norm(term_vec)
    if term_norm > 0:
        cos_ang = np.clip(abs(term_vec[2]) / term_norm, -1.0, 1.0)
        tilt_deg = np.degrees(np.arccos(cos_ang))
    else:
        tilt_deg = np.nan
    peptide_tilt_series.append(tilt_deg)

    # =================================================================
    # Pore / pore-water / thickness / thinning / permeation: ONLY scored
    # for frames where the peptide is actually inserted. Non-inserted
    # frames get NaN placeholders (or zero permeation) and are not
    # otherwise analyzed here.
    # =================================================================
    if peptide_inserted:
        axis_xy = center_xy
        pore_axis_xy_series.append(axis_xy)

        w_xy_d = np.linalg.norm(water_pos[:, :2] - axis_xy, axis=1)
        w_z_rel_mid = water_pos[:, 2] - bilayer_midplane_z
        in_probe = (w_xy_d <= pore_probe_radius) & (np.abs(w_z_rel_mid) <= midplane_half_width)
        pore_water_count.append(int(np.sum(in_probe)))

        mem_xy_d = np.linalg.norm(mem_pos[:, :2] - axis_xy, axis=1)
        mem_z_rel_mid = mem_pos[:, 2] - bilayer_midplane_z
        in_band = np.abs(mem_z_rel_mid) <= midplane_half_width
        lipid_min_r = np.min(mem_xy_d[in_band]) if np.any(in_band) else np.nan
        pore_lipid_min_radius.append(lipid_min_r)

        local_mask = mem_xy_d <= local_thickness_radius
        bulk_mask = (mem_xy_d >= bulk_ring_inner) & (mem_xy_d <= bulk_ring_outer)

        def _leaflet_thickness(mask):
            z_sel = mem_z_all[mask]
            up = z_sel[z_sel > mem_center_estimate]
            lo = z_sel[z_sel <= mem_center_estimate]
            if up.size == 0 or lo.size == 0:
                return np.nan
            return up.mean() - lo.mean()

        th_local = _leaflet_thickness(local_mask)
        th_bulk = _leaflet_thickness(bulk_mask)
        local_thickness.append(th_local)
        bulk_thickness.append(th_bulk)
        if th_bulk and not np.isnan(th_bulk) and th_bulk > 0 and not np.isnan(th_local):
            thinning_pct_series.append(100.0 * (th_bulk - th_local) / th_bulk)
        else:
            thinning_pct_series.append(np.nan)

        # ---- water permeation through the pore (new analysis) ----
        # A "crossing" is counted when a water oxygen, while continuously
        # within the probe cylinder (xy), moves from a confirmed position
        # above the midplane band to a confirmed position below it (or
        # vice versa) between two sampled frames.
        wo_pos = water_o.positions
        wo_xy_d = np.linalg.norm(wo_pos[:, :2] - axis_xy, axis=1)
        wo_z_rel = wo_pos[:, 2] - bilayer_midplane_z
        within_cyl = wo_xy_d <= pore_probe_radius
        side = np.where(wo_z_rel > midplane_half_width, 1,
                         np.where(wo_z_rel < -midplane_half_width, -1, 0))
        confirmed_now = within_cyl & (side != 0)
        crossed = confirmed_now & (_last_side != 0) & (side != _last_side)
        n_cross = int(np.sum(crossed))
        permeation_count_series[frame_i] = n_cross
        if n_cross:
            for mol_idx in np.where(crossed)[0]:
                permeation_events.append({
                    "Frame": frame_i,
                    "Time (ns)": frame_i * dt_ns,
                    "Water molecule index": int(mol_idx),
                    "Direction": "lower->upper" if side[mol_idx] == 1 else "upper->lower",
                })
        _last_side[~within_cyl] = 0
        _last_side[confirmed_now] = side[confirmed_now]
    else:
        pore_axis_xy_series.append((np.nan, np.nan))
        pore_water_count.append(np.nan)
        pore_lipid_min_radius.append(np.nan)
        local_thickness.append(np.nan)
        bulk_thickness.append(np.nan)
        thinning_pct_series.append(np.nan)
        permeation_count_series[frame_i] = 0
        # lost contact with the pore region -- forget in-progress crossings
        _last_side[:] = 0

    if (frame_i + 1) % progress_every == 0 or (frame_i + 1) == n_frames:
        elapsed = time.time() - t_start
        rate = (frame_i + 1) / elapsed if elapsed > 0 else 0.0
        print(f"[INFO]   frame {frame_i + 1}/{n_frames} "
              f"({100 * (frame_i + 1) / n_frames:.0f}%), {rate:.1f} frames/s")

print(f"[INFO] Frame-by-frame analysis finished in {time.time() - t_start:.1f} s.")

# =====================================================================
# Convert existing arrays (unchanged behavior)
# =====================================================================
num_waters_per_bin = np.array(num_waters_per_bin)
radius_per_bin = np.array(radius_per_bin)
pore_radius_per_bin = np.array(pore_radius_per_bin)

mean_waters = num_waters_per_bin.mean(axis=0)
std_waters = num_waters_per_bin.std(axis=0)

mean_radius = radius_per_bin.mean(axis=0)
std_radius = radius_per_bin.std(axis=0)

mean_pore_radius = np.nanmean(pore_radius_per_bin, axis=0)
std_pore_radius = np.nanstd(pore_radius_per_bin, axis=0)

if apply_kde_smoothing:
    mean_waters_smooth = gaussian_filter1d(mean_waters, sigma=kde_sigma)
    mean_radius_smooth = gaussian_filter1d(mean_radius, sigma=kde_sigma)
    mean_pore_radius_smooth = gaussian_filter1d(np.nan_to_num(mean_pore_radius), sigma=kde_sigma)
else:
    mean_waters_smooth = mean_waters
    mean_radius_smooth = mean_radius
    mean_pore_radius_smooth = mean_pore_radius

if enable_leaflet_analysis:
    upper_waters = np.array(upper_waters)
    lower_waters = np.array(lower_waters)
    upper_pore_radius = np.array(upper_pore_radius)
    lower_pore_radius = np.array(lower_pore_radius)

    def process_leaflet(data, is_radius=False):
        mean = np.nanmean(data, axis=0) if is_radius else data.mean(axis=0)
        std = np.nanstd(data, axis=0) if is_radius else data.std(axis=0)
        smooth = gaussian_filter1d(np.nan_to_num(mean), sigma=kde_sigma) if apply_kde_smoothing else mean
        return mean, std, smooth

    mean_upper_waters, std_upper_waters, mean_upper_waters_smooth = process_leaflet(upper_waters)
    mean_lower_waters, std_lower_waters, mean_lower_waters_smooth = process_leaflet(lower_waters)

    mean_upper_pore_radius, std_upper_pore_radius, mean_upper_pore_radius_smooth = process_leaflet(upper_pore_radius, is_radius=True)
    mean_lower_pore_radius, std_lower_pore_radius, mean_lower_pore_radius_smooth = process_leaflet(lower_pore_radius, is_radius=True)

# =====================================================================
# Pore open/closed classification, lifetime, stability, disruption
# (peptide-inserted frames only)
# =====================================================================
peptide_inserted_flags = np.array(peptide_inserted_flags, dtype=bool)
pore_water_count = np.array(pore_water_count, dtype=float)
pore_lipid_min_radius = np.array(pore_lipid_min_radius, dtype=float)
local_thickness = np.array(local_thickness, dtype=float)
bulk_thickness = np.array(bulk_thickness, dtype=float)
thinning_pct_series = np.array(thinning_pct_series, dtype=float)
contact_fraction_series = np.array(contact_fraction_series, dtype=float)
n_deflected_lipids_series = np.array(n_deflected_lipids_series, dtype=float)
frac_deflected_lipids_series = np.array(frac_deflected_lipids_series, dtype=float)
insertion_depth_series = np.array(insertion_depth_series, dtype=float)
peptide_tilt_series = np.array(peptide_tilt_series, dtype=float)
time_ns = np.arange(n_frames) * dt_ns

# pore_open is only meaningful (non-NaN inputs) on inserted frames; on
# non-inserted frames pore_water_count/pore_lipid_min_radius are NaN, so
# nan_to_num(..., nan=0.0) below correctly evaluates to "closed".
pore_open = peptide_inserted_flags & (
    (np.nan_to_num(pore_water_count, nan=0.0) >= pore_open_water_threshold) |
    (np.nan_to_num(pore_lipid_min_radius, nan=0.0) >= pore_open_radius_threshold)
)

with_peptide_mask = peptide_inserted_flags


def disruption_label(thinning, is_open):
    if not is_open or np.isnan(thinning):
        return "No disruption"
    if thinning < thinning_minor_pct:
        return "Minor perturbation (transient defect)"
    elif thinning < thinning_moderate_pct:
        return "Moderate disruption (partial pore)"
    else:
        return "Severe disruption (membrane rupture risk)"


disruption_series = [
    disruption_label(thinning_pct_series[i], pore_open[i]) for i in range(n_frames)
]

# ---- per-frame time series output ----
df_pore_ts = pd.DataFrame({
    "Frame": np.arange(n_frames),
    "Time (ns)": time_ns,
    "Peptide Inserted": peptide_inserted_flags,
    "Contact Fraction": contact_fraction_series,
    "Deflected Lipid Count": n_deflected_lipids_series,
    "Deflected Lipid Fraction": frac_deflected_lipids_series,
    "Peptide Insertion Depth (A)": insertion_depth_series,
    "Peptide Tilt vs Normal (deg)": peptide_tilt_series,
    "Pore Water Count": pore_water_count,
    "Pore Lipid-Excluded Radius (A)": pore_lipid_min_radius,
    "Local Bilayer Thickness (A)": local_thickness,
    "Bulk Bilayer Thickness (A)": bulk_thickness,
    "Local Thinning (%)": thinning_pct_series,
    "Pore Open": pore_open,
    "Disruption Category": disruption_series,
    "Water Permeation Events (this frame)": permeation_count_series,
})
df_pore_ts.to_csv(outpath(f"{output_prefix}_pore_timeseries.csv"), index=False)


# ---- run-length encoding of pore-open events, restricted to frames
# where the peptide is actually inserted in the membrane ----
def find_events(open_mask, group_mask):
    """
    Find contiguous runs of `open_mask & group_mask` frames, but only
    within contiguous runs of group_mask.
    """
    events = []
    n = len(open_mask)
    i = 0
    while i < n:
        if open_mask[i] and group_mask[i]:
            start = i
            while i < n and open_mask[i] and group_mask[i]:
                i += 1
            end = i - 1
            events.append((start, end))
        else:
            i += 1
    return events


def summarize_events(events, label):
    rows = []
    for (start, end) in events:
        duration_ns = (end - start + 1) * dt_ns
        max_radius = np.nanmax(pore_lipid_min_radius[start:end + 1]) if end >= start else np.nan
        max_waters = np.nanmax(pore_water_count[start:end + 1]) if end >= start else np.nan
        mean_thinning = np.nanmean(thinning_pct_series[start:end + 1]) if end >= start else np.nan
        total_permeation = int(np.sum(permeation_count_series[start:end + 1])) if end >= start else 0
        stability = "Stable" if duration_ns >= stable_pore_lifetime_ns else "Transient"
        rows.append({
            "Group": label,
            "Start Frame": start,
            "End Frame": end,
            "Start Time (ns)": start * dt_ns,
            "End Time (ns)": end * dt_ns,
            "Lifetime (ns)": duration_ns,
            "Max Pore Radius (A)": max_radius,
            "Max Pore Water Count": max_waters,
            "Mean Local Thinning (%)": mean_thinning,
            "Water Permeation Events": total_permeation,
            "Stability": stability,
        })
    return rows


events_with_peptide = find_events(pore_open, with_peptide_mask)
event_rows = summarize_events(events_with_peptide, "Peptide-interacting")
df_events = pd.DataFrame(event_rows)
df_events.to_csv(outpath(f"{output_prefix}_pore_lifetime_events.csv"), index=False)


# ---- group-level summary (peptide-inserted frames only) ----
def group_summary(mask, events, label):
    n_group_frames = int(np.sum(mask))
    n_open_frames = int(np.sum(mask & pore_open))
    frac_open = (n_open_frames / n_group_frames) if n_group_frames else np.nan
    lifetimes = [(-e[0] + e[1] + 1) * dt_ns for e in events]
    mean_lifetime = float(np.mean(lifetimes)) if lifetimes else 0.0
    max_lifetime = float(np.max(lifetimes)) if lifetimes else 0.0
    n_events = len(events)
    stable_events = sum(1 for lt in lifetimes if lt >= stable_pore_lifetime_ns)
    mean_water = float(np.nanmean(pore_water_count[mask])) if n_group_frames else np.nan
    mean_pore_radius_grp = float(np.nanmean(pore_lipid_min_radius[mask])) if n_group_frames else np.nan
    mean_thinning_grp = float(np.nanmean(thinning_pct_series[mask & pore_open])) if n_open_frames else np.nan
    mean_deflected_frac = float(np.nanmean(frac_deflected_lipids_series[mask])) if n_group_frames else np.nan
    total_permeation_grp = int(np.sum(permeation_count_series[mask])) if n_group_frames else 0
    worst_disruption = "No disruption"
    order = ["No disruption", "Minor perturbation (transient defect)",
             "Moderate disruption (partial pore)", "Severe disruption (membrane rupture risk)"]
    seen = set(np.array(disruption_series)[mask]) if n_group_frames else set()
    for level in reversed(order):
        if level in seen:
            worst_disruption = level
            break
    return {
        "Group": label,
        "Frames in group": n_group_frames,
        "Frames with pore open": n_open_frames,
        "Fraction of time pore open (stability index)": frac_open,
        "Number of pore-opening events": n_events,
        "Number of stable events (>= threshold lifetime)": stable_events,
        "Mean pore lifetime (ns)": mean_lifetime,
        "Max pore lifetime (ns)": max_lifetime,
        "Mean pore water occupancy": mean_water,
        "Mean pore lipid-excluded radius (A)": mean_pore_radius_grp,
        "Mean local thinning during open pores (%)": mean_thinning_grp,
        "Mean deflected-lipid fraction": mean_deflected_frac,
        "Total water permeation events": total_permeation_grp,
        "Worst observed disruption level": worst_disruption,
    }


summary_rows = [group_summary(with_peptide_mask, events_with_peptide, "Peptide-interacting")]
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(outpath(f"{output_prefix}_pore_summary.csv"), index=False)

# =====================================================================
# Water permeation events (new analysis)
# =====================================================================
df_permeation = pd.DataFrame(permeation_events)
df_permeation.to_csv(outpath(f"{output_prefix}_water_permeation_events.csv"), index=False)

cumulative_permeation = np.cumsum(permeation_count_series)
n_up = sum(1 for e in permeation_events if e["Direction"] == "lower->upper")
n_down = sum(1 for e in permeation_events if e["Direction"] == "upper->lower")
inserted_time_ns = float(np.sum(with_peptide_mask)) * dt_ns
permeation_summary = pd.DataFrame([{
    "Total permeation events": len(permeation_events),
    "Lower->Upper crossings": n_up,
    "Upper->Lower crossings": n_down,
    "Net flux (up - down)": n_up - n_down,
    "Time with peptide inserted (ns)": inserted_time_ns,
    "Permeation rate (events / ns of insertion)": (
        len(permeation_events) / inserted_time_ns if inserted_time_ns > 0 else np.nan
    ),
}])
permeation_summary.to_csv(outpath(f"{output_prefix}_water_permeation_summary.csv"), index=False)

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.plot(time_ns, cumulative_permeation, color="teal", lw=1.4)
ax.set_ylabel("Cumulative water\npermeation events")
ax.set_xlabel("Time (ns)")
ax.set_title(f"Cumulative pore-crossing water permeation events "
             f"(total = {len(permeation_events)}, net = {n_up - n_down})")
save_fig(fig, f"{output_prefix}_water_permeation_cumulative.png")

# =====================================================================
# Peptide insertion depth & tilt (new analysis)
# =====================================================================
fig, ax1 = plt.subplots(figsize=(11, 3.4))
ax1.plot(time_ns, insertion_depth_series, color="darkorange", lw=1.1, label="Insertion depth")
ax1.axhline(0, color="gray", ls=":", lw=1)
ax1.set_ylabel("Insertion depth (Å)\n(peptide COM z - midplane)", color="darkorange")
ax1.set_xlabel("Time (ns)")
ax1.tick_params(axis="y", labelcolor="darkorange")
ax2 = ax1.twinx()
ax2.plot(time_ns, peptide_tilt_series, color="purple", lw=1.1, alpha=0.7, label="Tilt")
ax2.set_ylabel("Tilt vs. membrane normal (deg)", color="purple")
ax2.tick_params(axis="y", labelcolor="purple")
ax1.set_title("Peptide insertion depth and tilt over time")
save_fig(fig, f"{output_prefix}_peptide_depth_tilt_timeseries.png")

# =====================================================================
# Correlation analysis (new analysis) -- purely post-processing of
# already-computed per-frame arrays, restricted to peptide-inserted
# frames where the pore metrics are defined.
# =====================================================================
def safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan, np.nan, int(mask.sum())
    r, p = pearsonr(x[mask], y[mask])
    return r, p, int(mask.sum())


corr_pairs = [
    ("Pore Lipid-Excluded Radius (A)", pore_lipid_min_radius, "Contact Fraction", contact_fraction_series),
    ("Pore Water Count", pore_water_count, "Deflected Lipid Fraction", frac_deflected_lipids_series),
    ("Local Thinning (%)", thinning_pct_series, "Peptide Insertion Depth (A)", insertion_depth_series),
    ("Pore Lipid-Excluded Radius (A)", pore_lipid_min_radius, "Peptide Tilt vs Normal (deg)", peptide_tilt_series),
    ("Pore Water Count", pore_water_count, "Peptide Insertion Depth (A)", insertion_depth_series),
]

corr_rows = []
for name_x, arr_x, name_y, arr_y in corr_pairs:
    r, p, n_used = safe_pearson(np.asarray(arr_x, dtype=float), np.asarray(arr_y, dtype=float))
    corr_rows.append({
        "Variable X": name_x, "Variable Y": name_y,
        "Pearson r": r, "p-value": p, "N frames used": n_used,
    })
df_corr = pd.DataFrame(corr_rows)
df_corr.to_csv(outpath(f"{output_prefix}_correlation_summary.csv"), index=False)

# scatter plots for the two most mechanistically interesting relationships
for name_x, arr_x, name_y, arr_y, fname in [
    ("Peptide Insertion Depth (A)", insertion_depth_series,
     "Pore Lipid-Excluded Radius (A)", pore_lipid_min_radius,
     f"{output_prefix}_corr_depth_vs_poreradius.png"),
    ("Peptide Tilt vs Normal (deg)", peptide_tilt_series,
     "Local Thinning (%)", thinning_pct_series,
     f"{output_prefix}_corr_tilt_vs_thinning.png"),
]:
    x = np.asarray(arr_x, dtype=float)
    y = np.asarray(arr_y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x[mask], y[mask], s=10, alpha=0.5, color="steelblue")
    if mask.sum() >= 3 and np.std(x[mask]) > 0:
        coef = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 50)
        ax.plot(xs, np.polyval(coef, xs), color="crimson", lw=1.5)
        r, p, _ = safe_pearson(x, y)
        ax.text(0.03, 0.95, f"r = {r:.2f}, p = {p:.1e}", transform=ax.transAxes,
                va="top", fontsize=9)
    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.set_title(f"{name_y} vs. {name_x}\n(peptide-inserted frames)")
    save_fig(fig, fname)

# =====================================================================
# Save existing binned outputs (unchanged)
# =====================================================================
pd.DataFrame({"Z (Å)": z_centers, "Mean Number of Waters": mean_waters, "Smoothed": mean_waters_smooth, "Std": std_waters}).to_csv(outpath(f"{output_prefix}_waters.csv"), index=False)
pd.DataFrame({"Z (Å)": z_centers, "Mean Radius (Å)": mean_radius, "Smoothed": mean_radius_smooth, "Std": std_radius}).to_csv(outpath(f"{output_prefix}_radius.csv"), index=False)
pd.DataFrame({"Z (Å)": z_centers, "Mean Pore Radius (Å)": mean_pore_radius, "Smoothed": mean_pore_radius_smooth, "Std": std_pore_radius}).to_csv(outpath(f"{output_prefix}_pore_radius.csv"), index=False)

if enable_leaflet_analysis:
    pd.DataFrame({"Z (Å)": z_centers, "Mean Number of Waters": mean_upper_waters, "Smoothed": mean_upper_waters_smooth, "Std": std_upper_waters}).to_csv(outpath(f"{output_prefix}_upper_leaflet_waters.csv"), index=False)
    pd.DataFrame({"Z (Å)": z_centers, "Mean Number of Waters": mean_lower_waters, "Smoothed": mean_lower_waters_smooth, "Std": std_lower_waters}).to_csv(outpath(f"{output_prefix}_lower_leaflet_waters.csv"), index=False)
    pd.DataFrame({"Z (Å)": z_centers, "Mean Pore Radius (Å)": mean_upper_pore_radius, "Smoothed": mean_upper_pore_radius_smooth, "Std": std_upper_pore_radius}).to_csv(outpath(f"{output_prefix}_upper_leaflet_pore_radius.csv"), index=False)
    pd.DataFrame({"Z (Å)": z_centers, "Mean Pore Radius (Å)": mean_lower_pore_radius, "Smoothed": mean_lower_pore_radius_smooth, "Std": std_lower_pore_radius}).to_csv(outpath(f"{output_prefix}_lower_leaflet_pore_radius.csv"), index=False)


# =====================================================================
# Two-panel z-profile figures: average water-in-pore, and pore radius.
# Panel (a) = whole-system profile, panel (b) = one leaflet-split profile
# (see water_panel_b_side / pore_radius_panel_b_side above). Each panel
# shows the raw curve (dashed), the Gaussian-smoothed curve (solid), a
# shaded std-dev band, and gray "Inner leaflet" / "Outer leaflet" bands.
# =====================================================================
def _add_leaflet_shading(ax):
    lo, hi = leaflet_shade_inner_z, leaflet_shade_outer_z
    ax.axvspan(-hi, -lo, color="0.5", alpha=0.15, zorder=0)
    ax.axvspan(lo, hi, color="0.5", alpha=0.15, zorder=0)
    ax.axvline(-lo, color="black", lw=1, zorder=1)
    ax.axvline(lo, color="black", lw=1, zorder=1)
    ymin, ymax = ax.get_ylim()
    ytext = ymin + 0.06 * (ymax - ymin)
    ax.text(-(lo + hi) / 2, ytext, "Inner leaflet", ha="center", va="bottom", fontsize=8)
    ax.text((lo + hi) / 2, ytext, "Outer leaflet", ha="center", va="bottom", fontsize=8)


def _add_panel_badge(ax, label):
    ax.text(0.02, 0.95, label, transform=ax.transAxes, ha="center", va="center",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", edgecolor="black"))


def plot_two_panel_profile(z, top, bottom, ylabel, fill_color, line_color, fname):
    """top / bottom are dicts with keys: raw, smooth, std (each 1D array over z)."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 7.5), sharex=False)

    for ax, data, badge in zip(axes, (top, bottom), ("a", "b")):
        raw, smooth, std = data["raw"], data["smooth"], data["std"]
        ax.plot(z, raw, "--", color="0.5", lw=1.3, label="Raw")
        ax.plot(z, smooth, "-", color=line_color, lw=1.8, label="Smoothed")
        ax.fill_between(z, smooth - std, smooth + std, color=fill_color, alpha=0.35, linewidth=0)
        ax.set_ylabel(ylabel)
        ax.set_xlim(z.min(), z.max())
        _add_leaflet_shading(ax)
        _add_panel_badge(ax, badge)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[-1].set_xlabel("Z (Å)")
    save_fig(fig, fname)


# ---- 1) average water-in-pore z-profile ----
_water_side_mean = mean_upper_waters if water_panel_b_side == "upper" else mean_lower_waters
_water_side_std = std_upper_waters if water_panel_b_side == "upper" else std_lower_waters
_water_side_smooth = mean_upper_waters_smooth if water_panel_b_side == "upper" else mean_lower_waters_smooth

if enable_leaflet_analysis:
    plot_two_panel_profile(
        z_centers,
        top={"raw": mean_waters, "smooth": mean_waters_smooth, "std": std_waters},
        bottom={"raw": _water_side_mean, "smooth": _water_side_smooth, "std": _water_side_std},
        ylabel="Number of Waters",
        fill_color="lightblue",
        line_color="black",
        fname=f"{output_prefix}_water_profile_panels.png",
    )
else:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(z_centers, mean_waters, "--", color="0.5", lw=1.3, label="Raw")
    ax.plot(z_centers, mean_waters_smooth, "-", color="black", lw=1.8, label="Smoothed")
    ax.fill_between(z_centers, mean_waters_smooth - std_waters, mean_waters_smooth + std_waters,
                     color="lightblue", alpha=0.35, linewidth=0)
    ax.set_ylabel("Number of Waters")
    ax.set_xlabel("Z (Å)")
    _add_leaflet_shading(ax)
    ax.legend(loc="upper right", fontsize=8)
    save_fig(fig, f"{output_prefix}_water_profile_panels.png")

# ---- 2) pore radius z-profile ----
_radius_side_mean = mean_upper_pore_radius if pore_radius_panel_b_side == "upper" else mean_lower_pore_radius
_radius_side_std = std_upper_pore_radius if pore_radius_panel_b_side == "upper" else std_lower_pore_radius
_radius_side_smooth = mean_upper_pore_radius_smooth if pore_radius_panel_b_side == "upper" else mean_lower_pore_radius_smooth

if enable_leaflet_analysis:
    plot_two_panel_profile(
        z_centers,
        top={"raw": mean_pore_radius, "smooth": mean_pore_radius_smooth, "std": std_pore_radius},
        bottom={"raw": _radius_side_mean, "smooth": _radius_side_smooth, "std": _radius_side_std},
        ylabel="Radius (Å)",
        fill_color="lightcoral",
        line_color="darkred",
        fname=f"{output_prefix}_pore_radius_profile_panels.png",
    )
else:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(z_centers, mean_pore_radius, "--", color="0.5", lw=1.3, label="Raw")
    ax.plot(z_centers, mean_pore_radius_smooth, "-", color="darkred", lw=1.8, label="Smoothed")
    ax.fill_between(z_centers, mean_pore_radius_smooth - std_pore_radius, mean_pore_radius_smooth + std_pore_radius,
                     color="lightcoral", alpha=0.35, linewidth=0)
    ax.set_ylabel("Radius (Å)")
    ax.set_xlabel("Z (Å)")
    _add_leaflet_shading(ax)
    ax.legend(loc="upper right", fontsize=8)
    save_fig(fig, f"{output_prefix}_pore_radius_profile_panels.png")

# === Save per-residue contact profile ===
residue_contact_profile = {"Residue": [], "Resname": [], "Chain": [], "Contact Frequency (%)": []}
for res in protein.residues:
    contacts = residue_contacts[res.resindex]
    contact_freq = 100 * np.sum(contacts) / n_frames
    residue_contact_profile["Residue"].append(res.resid)
    residue_contact_profile["Resname"].append(res.resname)
    residue_contact_profile["Chain"].append(res.segid)
    residue_contact_profile["Contact Frequency (%)"].append(contact_freq)

df_contacts = pd.DataFrame(residue_contact_profile)
df_contacts.to_csv(outpath(f"{output_prefix}_protein_membrane_contacts.csv"), index=False)

# === Plot per-residue contact (individual file) ===
fig = plt.figure(figsize=(10, 3))
plt.bar(df_contacts["Residue"], df_contacts["Contact Frequency (%)"], color='darkblue')
plt.xlabel("Residue")
plt.ylabel("Contact Frequency (%)")
plt.title("Per-Residue Contact Frequency with Membrane")
save_fig(fig, f"{output_prefix}_protein_membrane_contacts.png")

# =====================================================================
# Pore plots -- each saved as its own individual file.
# These metrics only exist on peptide-inserted frames (NaN elsewhere),
# so a single color is used and non-inserted frames simply don't plot.
# =====================================================================
pore_point_color = "crimson"

# 1a) Pore radius over time
fig = plt.figure(figsize=(11, 3.2))
plt.scatter(time_ns, pore_lipid_min_radius, c=pore_point_color, s=6)
plt.axhline(pore_open_radius_threshold, color="gray", ls="--", lw=1, label="Open threshold")
plt.ylabel("Pore radius (Å)")
plt.xlabel("Time (ns)")
plt.title("Pore radius over time (peptide-inserted frames only)")
plt.legend(loc="upper right", fontsize=8)
save_fig(fig, f"{output_prefix}_pore_radius_timeseries.png")

# 1b) Pore water occupancy over time
fig = plt.figure(figsize=(11, 3.2))
plt.scatter(time_ns, pore_water_count, c=pore_point_color, s=6)
plt.axhline(pore_open_water_threshold, color="gray", ls="--", lw=1, label="Open threshold")
plt.ylabel("Waters in pore")
plt.xlabel("Time (ns)")
plt.title("Pore water occupancy over time (peptide-inserted frames only)")
plt.legend(loc="upper right", fontsize=8)
save_fig(fig, f"{output_prefix}_pore_water_timeseries.png")

# 1c) Local membrane thinning over time (shaded severity bands)
fig, ax = plt.subplots(figsize=(11, 3.6))
ymax = np.nanmax(thinning_pct_series) if np.any(~np.isnan(thinning_pct_series)) else thinning_moderate_pct * 1.5
ymax = max(ymax, thinning_moderate_pct * 1.2)
ax.axhspan(-5, thinning_minor_pct, color="green", alpha=0.08, label="No / negligible disruption")
ax.axhspan(thinning_minor_pct, thinning_moderate_pct, color="orange", alpha=0.10, label=f"Minor (>{thinning_minor_pct:.0f}%)")
ax.axhspan(thinning_moderate_pct, ymax + 5, color="red", alpha=0.10, label=f"Moderate/severe (>{thinning_moderate_pct:.0f}%)")
ax.scatter(time_ns, thinning_pct_series, c=pore_point_color, s=8, zorder=3)
ax.axhline(thinning_minor_pct, color="darkorange", ls="--", lw=1)
ax.axhline(thinning_moderate_pct, color="darkred", ls="--", lw=1)
ax.set_ylim(-5, ymax + 5)
ax.set_ylabel("Local thinning (%)")
ax.set_xlabel("Time (ns)")
ax.set_title("Local membrane thinning relative to bulk (peptide-inserted frames only)")
ax.legend(loc="upper right", fontsize=7, ncol=3)
save_fig(fig, f"{output_prefix}_membrane_thinning_timeseries.png")

# 1d) combined 3-panel overview (kept in addition to the individual plots)
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
axes[0].scatter(time_ns, pore_lipid_min_radius, c=pore_point_color, s=6)
axes[0].axhline(pore_open_radius_threshold, color="gray", ls="--", lw=1, label="Open threshold")
axes[0].set_ylabel("Pore radius (Å)")
axes[0].set_title("Pore radius over time (peptide-inserted frames only)")
axes[0].legend(loc="upper right", fontsize=8)
axes[1].scatter(time_ns, pore_water_count, c=pore_point_color, s=6)
axes[1].axhline(pore_open_water_threshold, color="gray", ls="--", lw=1)
axes[1].set_ylabel("Waters in pore")
axes[1].set_title("Pore water occupancy over time")
axes[2].scatter(time_ns, thinning_pct_series, c=pore_point_color, s=6)
axes[2].axhline(thinning_minor_pct, color="orange", ls="--", lw=1, label=f">{thinning_minor_pct}% minor")
axes[2].axhline(thinning_moderate_pct, color="red", ls="--", lw=1, label=f">{thinning_moderate_pct}% moderate/severe")
axes[2].set_ylabel("Local thinning (%)")
axes[2].set_xlabel("Time (ns)")
axes[2].set_title("Local membrane thinning relative to bulk")
axes[2].legend(loc="upper right", fontsize=8)
save_fig(fig, f"{output_prefix}_pore_overview_combined.png")

# 2) Pore lifetime histogram (peptide-inserted events only)
fig = plt.figure(figsize=(7, 4))
lt_with = [(-e[0] + e[1] + 1) * dt_ns for e in events_with_peptide]
bins = np.linspace(0, max([1.0] + lt_with), 20)
plt.hist(lt_with, bins=bins, alpha=0.75, color="crimson")
plt.xlabel("Pore lifetime (ns)")
plt.ylabel("Number of events")
plt.title("Pore lifetime distribution (peptide-inserted frames)")
save_fig(fig, f"{output_prefix}_pore_lifetime_histogram.png")

# 3) Deflected-lipid fraction over time
fig = plt.figure(figsize=(11, 3.2))
plt.plot(time_ns, 100 * frac_deflected_lipids_series, color=color_lipid_deflected, lw=1.2)
plt.fill_between(time_ns, 0, 100 * frac_deflected_lipids_series, color=color_lipid_deflected, alpha=0.15)
plt.ylabel("Deflected lipids (%)")
plt.xlabel("Time (ns)")
plt.title("Fraction of phospholipids in contact with the peptide over time")
save_fig(fig, f"{output_prefix}_deflected_lipid_fraction_timeseries.png")

# =====================================================================
# Membrane thickness analysis (local vs. bulk) -- peptide-inserted frames
# =====================================================================
# 4a) local vs bulk thickness over time (only defined on inserted frames;
# non-inserted frames are NaN and simply leave a gap in the line)
fig, ax = plt.subplots(figsize=(11, 3.6))
insert_idx = np.where(peptide_inserted_flags)[0]
if insert_idx.size:
    edges = np.where(np.diff(insert_idx) > 1)[0]
    starts = np.r_[insert_idx[0], insert_idx[edges + 1]] if edges.size else np.r_[insert_idx[0]]
    ends = np.r_[insert_idx[edges], insert_idx[-1]] if edges.size else np.r_[insert_idx[-1]]
    for s, e in zip(starts, ends):
        ax.axvspan(time_ns[s], time_ns[e], color="crimson", alpha=0.08)
ax.plot(time_ns, bulk_thickness, color="steelblue", lw=1.3, label="Bulk thickness (undisturbed annulus)")
ax.plot(time_ns, local_thickness, color="darkorange", lw=1.3, label="Local thickness (near pore axis)")
ax.set_ylabel("Bilayer thickness (Å)")
ax.set_xlabel("Time (ns)")
ax.set_title("Local vs. bulk bilayer thickness (computed during peptide-inserted spans, shaded)")
ax.legend(loc="upper right", fontsize=8)
save_fig(fig, f"{output_prefix}_membrane_thickness_timeseries.png")

# 4b) thickness distributions (histogram, inserted frames only)
finite_bulk = bulk_thickness[np.isfinite(bulk_thickness)]
finite_local = local_thickness[np.isfinite(local_thickness)]
if finite_bulk.size and finite_local.size:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins_th = np.linspace(
        min(finite_bulk.min(), finite_local.min()),
        max(finite_bulk.max(), finite_local.max()),
        30,
    )
    ax.hist(finite_bulk, bins=bins_th, alpha=0.6, color="steelblue", label="Bulk")
    ax.hist(finite_local, bins=bins_th, alpha=0.6, color="darkorange", label="Local (near pore axis)")
    ax.set_xlabel("Bilayer thickness (Å)")
    ax.set_ylabel("Frame count")
    ax.set_title("Distribution of local vs. bulk bilayer thickness (peptide-inserted frames)")
    ax.legend()
    save_fig(fig, f"{output_prefix}_membrane_thickness_histogram.png")

# 4c) thinning distribution (peptide-inserted frames only -- no
# "not inserted" comparison group)
finite_thinning = thinning_pct_series[np.isfinite(thinning_pct_series)]
if finite_thinning.size:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins_thin = np.linspace(finite_thinning.min(), finite_thinning.max(), 30)
    ax.hist(finite_thinning, bins=bins_thin, alpha=0.75, color="crimson", label="Peptide inserted")
    ax.axvline(thinning_minor_pct, color="darkorange", ls="--", lw=1)
    ax.axvline(thinning_moderate_pct, color="darkred", ls="--", lw=1)
    ax.set_xlabel("Local thinning (%)")
    ax.set_ylabel("Frame count")
    ax.set_title("Local thinning distribution during peptide insertion")
    ax.legend()
    save_fig(fig, f"{output_prefix}_membrane_thinning_histogram.png")

# 4d) thickness/thinning summary CSV
thickness_summary = pd.DataFrame([{
    "Mean bulk thickness (A)": np.nanmean(bulk_thickness),
    "Std bulk thickness (A)": np.nanstd(bulk_thickness),
    "Mean local thickness (A)": np.nanmean(local_thickness),
    "Std local thickness (A)": np.nanstd(local_thickness),
    "Mean thinning, peptide-inserted frames (%)": np.nanmean(thinning_pct_series),
    "Fraction of inserted frames with minor+ thinning (%)": (
        100 * np.nanmean(thinning_pct_series[with_peptide_mask] >= thinning_minor_pct)
        if np.any(with_peptide_mask) else np.nan
    ),
    "Fraction of inserted frames with moderate+ thinning (%)": (
        100 * np.nanmean(thinning_pct_series[with_peptide_mask] >= thinning_moderate_pct)
        if np.any(with_peptide_mask) else np.nan
    ),
}])
thickness_summary.to_csv(outpath(f"{output_prefix}_membrane_thickness_summary.csv"), index=False)

# =====================================================================
# Snapshot-over-time figure(s): "insertion over time (midplane-penetrating
# water)" -- phospholipids in orange, peptide-deflected lipids in red,
# midplane-penetrating (pore) waters in cyan, peptide backbone with
# labeled N/C termini. (Visualization only -- for frames where the
# peptide isn't classed as "inserted", the membrane's own geometric
# center is used purely to place the panel, no pore metrics are implied.)
# =====================================================================
snapshot_frames = np.unique(np.linspace(0, n_frames - 1, n_snapshots).astype(int))
n_panels = len(snapshot_frames)
fig_combined, axes_combined = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 5), sharey=True)
if n_panels == 1:
    axes_combined = [axes_combined]

for panel_i, frame_idx in enumerate(snapshot_frames):
    u.trajectory[frame_idx]
    t_ns = frame_idx * dt_ns

    mem_pos_snap = membrane.positions
    mem_center_xy_snap = mem_pos_snap[:, :2].mean(axis=0)
    midplane_z = bilayer_midplane_series[frame_idx]
    if peptide_inserted_flags[frame_idx]:
        axis_xy = pore_axis_xy_series[frame_idx]
    else:
        axis_xy = mem_center_xy_snap

    # phospholipid representative points: per-residue center of mass
    lipid_xy = np.array([res.atoms.center_of_mass()[:2] for res in lipid_residues])
    lipid_z = np.array([res.atoms.center_of_mass()[2] for res in lipid_residues])

    # which lipid residues are in contact with the peptide right now?
    _, _, deflected_mask_snap = compute_contacts(
        protein, membrane, contact_cutoff, u.dimensions,
        n_residues, protein_atom_to_res,
        membrane.n_atoms, mem_atom_to_res, n_lipid_residues,
    )

    lipid_x_plot = lipid_xy[:, 0] - mem_center_xy_snap[0]
    lipid_z_plot = lipid_z - midplane_z

    # midplane-penetrating ("pore") waters at this frame
    water_pos_snap = waters.positions
    w_xy_d_snap = np.linalg.norm(water_pos_snap[:, :2] - axis_xy, axis=1)
    w_z_rel_snap = water_pos_snap[:, 2] - midplane_z
    in_probe_snap = (w_xy_d_snap <= pore_probe_radius) & (np.abs(w_z_rel_snap) <= midplane_half_width)
    pore_water_xy_snap = water_pos_snap[in_probe_snap][:, 0] - mem_center_xy_snap[0]
    pore_water_z_snap = water_pos_snap[in_probe_snap][:, 2] - midplane_z

    # peptide backbone trace + termini
    bb_x = backbone.positions[:, 0] - mem_center_xy_snap[0]
    bb_z = backbone.positions[:, 2] - midplane_z
    nterm_x = nterm_atoms.positions[:, 0].mean() - mem_center_xy_snap[0]
    nterm_z = nterm_atoms.positions[:, 2].mean() - midplane_z
    cterm_x = cterm_atoms.positions[:, 0].mean() - mem_center_xy_snap[0]
    cterm_z = cterm_atoms.positions[:, 2].mean() - midplane_z

    def draw_panel(ax):
        ax.axhline(0, color="black", ls=":", lw=1, zorder=1)
        ax.scatter(lipid_x_plot[~deflected_mask_snap], lipid_z_plot[~deflected_mask_snap],
                   c=color_lipid_normal, s=45, edgecolor="white", linewidth=0.3, zorder=2)
        ax.scatter(lipid_x_plot[deflected_mask_snap], lipid_z_plot[deflected_mask_snap],
                   c=color_lipid_deflected, s=45, edgecolor="white", linewidth=0.3, zorder=2)
        ax.scatter(pore_water_xy_snap, pore_water_z_snap,
                   c=color_pore_water, s=35, edgecolor="black", linewidth=0.3, zorder=3)
        ax.scatter(bb_x, bb_z, c=color_backbone, s=90, edgecolor="black", linewidth=0.5, zorder=4)
        ax.scatter([nterm_x], [nterm_z], c=color_nterm, s=260, edgecolor="black", linewidth=0.8, zorder=5)
        ax.text(nterm_x, nterm_z, "N", ha="center", va="center", color="white", fontweight="bold", fontsize=9, zorder=6)
        ax.scatter([cterm_x], [cterm_z], c=color_cterm, s=260, edgecolor="black", linewidth=0.8, zorder=5)
        ax.text(cterm_x, cterm_z, "C", ha="center", va="center", color="white", fontweight="bold", fontsize=9, zorder=6)
        ax.set_xlabel("x [Å]")
        ax.set_title(f"{t_ns:.1f} ns")

    draw_panel(axes_combined[panel_i])

    # individual per-frame snapshot file
    fig_i, ax_i = plt.subplots(figsize=(5, 5))
    draw_panel(ax_i)
    ax_i.set_ylabel("z [Å] (midplane = 0)")
    ax_i.set_title(f"Insertion snapshot -- {t_ns:.1f} ns")
    save_fig(fig_i, f"{output_prefix}_snapshot_{t_ns:.0f}ns.png")

axes_combined[0].set_ylabel("z [Å] (midplane = 0)")
fig_combined.suptitle("Insertion over time (midplane-penetrating water)")
save_fig(fig_combined, f"{output_prefix}_snapshot_over_time.png")

print(f"[OK] Analysis complete. All outputs saved in: {output_dir}/")
print("[INFO] Key outputs:")
print(f"   {output_prefix}_pore_timeseries.csv")
print(f"   {output_prefix}_pore_lifetime_events.csv")
print(f"   {output_prefix}_pore_summary.csv")
print(f"   {output_prefix}_membrane_thickness_summary.csv")
print(f"   {output_prefix}_water_permeation_events.csv / _water_permeation_summary.csv / _water_permeation_cumulative.png")
print(f"   {output_prefix}_peptide_depth_tilt_timeseries.png")
print(f"   {output_prefix}_correlation_summary.csv / _corr_depth_vs_poreradius.png / _corr_tilt_vs_thinning.png")
print(f"   {output_prefix}_pore_radius_timeseries.png / _pore_water_timeseries.png / _membrane_thinning_timeseries.png")
print(f"   {output_prefix}_membrane_thickness_timeseries.png / _membrane_thickness_histogram.png / _membrane_thinning_histogram.png")
print(f"   {output_prefix}_deflected_lipid_fraction_timeseries.png")
print(f"   {output_prefix}_snapshot_over_time.png  (+ one _snapshot_<t>ns.png per timepoint)")
print(f"   {output_prefix}_water_profile_panels.png  (avg. water-in-pore vs Z, whole-system + leaflet)")
print(f"   {output_prefix}_pore_radius_profile_panels.png  (pore radius vs Z, whole-system + leaflet)")
