#!/usr/bin/env python3
"""
Peptide insertion depth + pore size + water permeation + free-energy analysis
================================================================================

Computes, in a SINGLE pass over the trajectory:

  1. Membrane midplane insertion depth of the peptide (unchanged logic from
     the original script).
  2. Time-resolved pore radius induced by the peptide (nearest-lipid-phosphate
     proxy method).
  3. THREE distinct water-occupancy metrics (these are easy to conflate, so
     they are kept separate on purpose):
       a. "loose" cylinder water  - waters within the per-frame pore radius
          (xy) AND within a fixed z-slab around the peptide (this is the
          original script's definition; kept for backward comparison).
       b. "pore_water_exclusive"  - waters within the per-frame pore radius
          (xy) AND within the HYDROPHOBIC CORE z-range only (i.e. genuinely
          inside the lipid tail region intersected by the peptide's pore,
          not just "near" the peptide in a generic slab). This is the more
          defensible definition of "water in the pore".
       c. "hydrophobic_core_water_total" - ALL waters within the hydrophobic
          core z-range and within a local xy patch around the peptide
          (--core_xy_radius), independent of the estimated pore radius. This
          captures general bilayer hydration defects (e.g. water fingers)
          that are not necessarily inside the peptide-defined pore column.
  4. Water permeation events, counted with a hysteresis sign-crossing state
     machine (a water only "counts" as crossing once it has been clearly on
     one side, beyond --hysteresis_margin, and is later clearly on the other
     side, while continuously satisfying the pore-cylinder xy criterion).
     This is more robust to thermal jitter at z ~ 0 than a bare sign flip.
  5. Two approximate (Boltzmann-inversion) free-energy profiles:
       a. G(r) for the pore radius, from the observed radius distribution.
       b. G(z) for water along the membrane normal within the pore column,
          from the observed water z-density, referenced to the bulk-water
          density far from the membrane.

METHOD CAVEATS (read before trusting numbers on a real system)
----------------------------------------------------------------
- Pore radius is a lightweight nearest-phosphate proxy, NOT a HOLE/CHAP-grade
  profile. For publication-grade pore geometry, cross-check with HOLE, CHAP,
  or MDAnalysis' own pore-finding tools.
- The free-energy profiles here are plain Boltzmann inversions of observed
  equilibrium(-ish) distributions (G = -kB*T*ln(P)), NOT umbrella-sampling /
  ABF / metadynamics PMFs. They are only meaningful if the trajectory
  actually visits the relevant range of pore radii / water z-positions many
  times and is reasonably well-equilibrated. Sparse sampling will produce
  noisy, misleading "free energy" curves - always check the underlying
  histograms (saved alongside the profiles) before interpreting them.
- Permeation events are pore-column crossings (xy within the pore radius),
  not full bulk-to-bulk residence-time-filtered permeation events as used in
  rigorous permeation-free-energy studies. Treat as a first-pass metric.
- PBC handling: peptide coordinates are unwrapped (if bond information is
  available) so its center of mass is not corrupted by wrapping. All xy
  distances (peptide-to-lipid, peptide-to-water) use the minimum-image
  convention assuming an ORTHORHOMBIC box. Triclinic boxes are not handled.

Author: Prince Manu (refined)
"""

# NO GUI (fix Qt issues)
import matplotlib
matplotlib.use('Agg')

import argparse
import os
import sys
import traceback

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import MDAnalysis as mda
import MDAnalysis.transformations as trans

# Boltzmann constant in kcal/mol/K
KB_KCAL = 0.0019872041


# =============================================================================
# CLI / CONFIG
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Peptide insertion depth, pore size, water permeation and "
                    "free-energy analysis.")

    # --- input files ---
    p.add_argument("--tpr", default="md_0_500.tpr", help="Topology (.tpr)")
    p.add_argument("--xtc", default="cen300.xtc", help="Trajectory (.xtc)")

    # --- selections ---
    p.add_argument("--peptide_resname", default="PROTEIN")
    p.add_argument("--lipid_resnames", default="OYPE PMPE PMPG POPE PVCL2 PYPG QMPE")
    p.add_argument("--water_resnames", default="TIP3")

    # --- pore geometry ---
    p.add_argument("--slab_half_thickness", type=float, default=10.0,
                    help="A, half-thickness of z-slab around peptide z used "
                         "for the legacy loose water-in-cylinder metric")
    p.add_argument("--n_nearest_phosphates", type=int, default=5)
    p.add_argument("--probe_radius", type=float, default=1.4,
                    help="A, water probe radius subtracted from lipid distance")
    p.add_argument("--use_fixed_cylinder_radius", action="store_true",
                    help="Use --water_cylinder_radius instead of the "
                         "per-frame pore radius estimate")
    p.add_argument("--water_cylinder_radius", type=float, default=8.0)

    # --- hydrophobic core / water breakdown ---
    p.add_argument("--hydrophobic_core_margin", type=float, default=8.0,
                    help="A, inward shift from each leaflet's mean phosphate "
                         "z-plane used to define the hydrophobic core "
                         "(excludes headgroup/interfacial region)")
    p.add_argument("--core_xy_radius", type=float, default=15.0,
                    help="A, xy radius of the local membrane patch around "
                         "the peptide used for hydrophobic_core_water_total "
                         "and for the water z-density used in the PMF")

    # --- permeation ---
    p.add_argument("--hysteresis_margin", type=float, default=2.0,
                    help="A, a water must be this far past the peptide z "
                         "before a side is 'committed', to reject jitter "
                         "at z~0 as spurious crossings")

    # --- midplane water penetration ---
    p.add_argument("--midplane_water_margin", type=float, default=3.0,
                    help="A, waters within this |z - z_midplane| are "
                         "counted/plotted as having 'penetrated into the "
                         "midplane'")
    p.add_argument("--midplane_water_xy_radius", type=float, default=None,
                    help="A, xy radius (around the peptide) used to decide "
                         "which midplane waters to count/draw. Defaults to "
                         "the per-frame pore cylinder radius (same cutoff "
                         "used for the loose-cylinder metric) if not set.")

    # --- structural snapshots (peptide + lipids + midplane water) ---
    p.add_argument("--snapshot_times", default="",
                    help="Comma-separated list of times in ns at which to "
                         "render a structural snapshot (peptide backbone, "
                         "phosphates, midplane-penetrating water), e.g. "
                         "'495,496,497,498,499,500'. If left empty, 6 "
                         "evenly spaced snapshots spanning the trajectory "
                         "are chosen automatically.")
    p.add_argument("--n_auto_snapshots", type=int, default=6,
                    help="Number of automatically-chosen snapshot frames "
                         "used when --snapshot_times is not given")
    p.add_argument("--snapshot_near_lipid_cutoff", type=float, default=12.0,
                    help="A, xy distance from the peptide within which a "
                         "phosphate is drawn in orange (pore-lining) "
                         "instead of gray in the snapshot plots")
    p.add_argument("--snapshot_water_xy_radius", type=float, default=20.0,
                    help="A, xy distance from the peptide within which "
                         "midplane-penetrating waters are drawn (cyan) in "
                         "the snapshot plots")
    p.add_argument("--snapshot_dir", default=None,
                    help="Directory to save individual snapshot PNGs into "
                         "(default: '<out_prefix>snapshots/')")
    p.add_argument("--pore_snapshot_xy_range", type=float, default=40.0,
                    help="A, half-width of the x/y axes in the top-down "
                         "pore-size snapshot (centered on the peptide)")

    # --- free energy profiles ---
    p.add_argument("--temperature", type=float, default=310.0, help="K")
    p.add_argument("--pore_radius_bins", type=int, default=30)
    p.add_argument("--z_bin_width", type=float, default=1.0, help="A")
    p.add_argument("--z_range", type=float, default=40.0,
                    help="A, half-range of the z-PMF histogram around the "
                         "peptide z position")
    p.add_argument("--bulk_z_cutoff", type=float, default=25.0,
                    help="A, |z_rel| beyond this is treated as bulk water "
                         "for normalizing the water free-energy profile")

    # --- output ---
    p.add_argument("--out_prefix", default="",
                    help="Output folder for all result files (created if it "
                         "doesn't exist). Also accepted as a plain filename "
                         "prefix; either way, everything is saved into a real "
                         "dedicated folder -- see the output-directory setup "
                         "at the top of main().")
    p.add_argument("--progress_every", type=int, default=100,
                    help="Print a progress line every N frames (0 disables)")

    return p


# =============================================================================
# SELECTIONS
# =============================================================================

def detect_phosphate(u, lipid_resnames):
    for name in ["P", "P31", "PO4", "P1"]:
        if len(u.select_atoms(f"resname {lipid_resnames} and name {name}")) > 0:
            print(f"Using phosphate: {name}")
            return name
    raise ValueError("No phosphate found")


def detect_water_oxygen(u, water_resnames):
    for name in ["OW", "OH2", "O", "OW1", "W"]:
        sel = u.select_atoms(f"resname {water_resnames} and name {name}")
        if len(sel) > 0:
            print(f"Using water oxygen: {name} ({len(sel)} waters)")
            return sel
    raise ValueError("No water oxygens found - check --water_resnames / atom names")


def select_backbone_trace(peptide):
    """
    Backbone trace used for the structural snapshot plots. Falls back from
    CA atoms to the full peptide selection if no CA atoms are found (e.g.
    coarse-grained models), and picks the first/last atom along the trace
    as the N-/C-terminal markers.
    """
    ca = peptide.select_atoms("name CA")
    if len(ca) == 0:
        ca = peptide.select_atoms("name BB")  # common CG (Martini) backbone name
    if len(ca) == 0:
        print("WARNING: no CA/BB atoms found for backbone trace; using all "
              "peptide atoms (snapshot trace may look noisy).")
        ca = peptide
    return ca


def select_groups(u, args):
    peptide = u.select_atoms(f"protein or resname {args.peptide_resname}")
    if len(peptide) == 0:
        raise ValueError("Peptide not found")

    pname = detect_phosphate(u, args.lipid_resnames)
    phosphates = u.select_atoms(f"resname {args.lipid_resnames} and name {pname}")
    water_o = detect_water_oxygen(u, args.water_resnames)
    backbone = select_backbone_trace(peptide)

    return peptide, phosphates, water_o, backbone


def setup_unwrap(u, peptide):
    """
    Unwrap the peptide so its center of mass isn't corrupted if it straddles
    a periodic boundary. Falls back gracefully (with a warning) if bond
    information isn't available for the unwrap transformation.
    """
    try:
        u.trajectory.add_transformations(trans.unwrap(peptide))
        print("Peptide unwrap transformation enabled (PBC-safe center of mass).")
    except Exception as e:
        print(f"WARNING: could not enable peptide unwrap ({e}). "
              f"Peptide center of mass may be wrong if it straddles a "
              f"periodic boundary in the raw trajectory.")


# =============================================================================
# PBC-AWARE XY DISTANCE (minimum image, orthorhombic box assumed)
# =============================================================================

def xy_min_image_dist(xy_positions, ref_xy, box):
    """
    xy distance from each row of xy_positions to ref_xy, using the minimum
    image convention in x and y. `box` is the (6,) MDAnalysis dimensions
    array [lx, ly, lz, alpha, beta, gamma]; only lx, ly are used, so this is
    only strictly correct for orthorhombic boxes.
    """
    delta = xy_positions - ref_xy
    if box is not None:
        box_xy = box[:2]
        safe = np.where(box_xy > 0, box_xy, np.inf)
        delta = delta - safe * np.round(delta / safe)
    return np.linalg.norm(delta, axis=1)


def xy_min_image_vector(xy_positions, ref_xy, box):
    """
    Same minimum-image correction as xy_min_image_dist, but returns the
    (dx, dy) delta vector for each row instead of collapsing to a distance.
    Used for top-down (pore-size) snapshots where we need signed relative
    x/y, not just a radius.
    """
    delta = xy_positions - ref_xy
    if box is not None:
        box_xy = box[:2]
        safe = np.where(box_xy > 0, box_xy, np.inf)
        delta = delta - safe * np.round(delta / safe)
    return delta


# =============================================================================
# MAIN SINGLE-PASS ANALYSIS
# =============================================================================

def resolve_snapshot_frames(u, args):
    """
    Decide which frame indices get a structural snapshot rendered, based on
    --snapshot_times (ns) or, if that's empty, --n_auto_snapshots evenly
    spaced frames spanning the trajectory. Returns {frame_index: time_ns}.
    """
    n_frames = len(u.trajectory)
    frame_times_ps = np.array([ts.time for ts in u.trajectory])
    u.trajectory.rewind()

    if args.snapshot_times.strip():
        wanted_ns = [float(t) for t in args.snapshot_times.split(",") if t.strip()]
    else:
        n_auto = max(1, args.n_auto_snapshots)
        if n_frames <= n_auto:
            idx = np.arange(n_frames)
        else:
            idx = np.linspace(0, n_frames - 1, n_auto).round().astype(int)
        return {int(i): frame_times_ps[i] / 1000.0 for i in idx}

    frame_map = {}
    for t_ns in wanted_ns:
        t_ps = t_ns * 1000.0
        fi = int(np.argmin(np.abs(frame_times_ps - t_ps)))
        frame_map[fi] = frame_times_ps[fi] / 1000.0
    return frame_map


def run_analysis(u, peptide, phosphates, water_o, backbone, args):
    n_water = len(water_o)

    snapshot_frame_map = resolve_snapshot_frames(u, args)
    snapshot_data = {}

    times = []
    depths, upper_rel, lower_rel = [], [], []

    pore_radii = []
    loose_water_counts = []
    pore_water_exclusive_counts = []
    hydrophobic_core_water_counts = []
    midplane_water_counts = []
    permeation_events = []

    # Boltzmann-inversion accumulators
    radius_samples = []  # valid per-frame pore radii, for G(r)
    z_bins = np.arange(-args.z_range, args.z_range + args.z_bin_width, args.z_bin_width)
    z_hist_counts = np.zeros(len(z_bins) - 1)
    n_frames_for_z_hist = 0

    # permeation state: last "committed" side (+1/-1) for each water, only
    # while continuously tracked inside the pore cylinder. NaN = untracked.
    committed_sign = np.full(n_water, np.nan)

    n_frames = len(u.trajectory)

    for fi, ts in enumerate(u.trajectory):
        if args.progress_every and fi % args.progress_every == 0:
            print(f"  frame {fi}/{n_frames} (t = {ts.time:.1f} ps)")

        box = ts.dimensions

        pep_pos = peptide.center_of_mass()
        pep_xy = pep_pos[:2]
        pep_z = pep_pos[2]

        # --- leaflets / midplane / hydrophobic core bounds ---
        z_phos_all = phosphates.positions[:, 2]
        z_mid = np.mean(z_phos_all)
        upper_mask = z_phos_all > z_mid
        lower_mask = ~upper_mask
        upper_mean = np.mean(z_phos_all[upper_mask]) if upper_mask.any() else np.nan
        lower_mean = np.mean(z_phos_all[lower_mask]) if lower_mask.any() else np.nan

        core_lower = lower_mean + args.hydrophobic_core_margin
        core_upper = upper_mean - args.hydrophobic_core_margin

        depth = pep_z - z_mid
        times.append(ts.time)
        depths.append(depth)
        upper_rel.append(upper_mean - z_mid)
        lower_rel.append(lower_mean - z_mid)

        # --- pore radius: N nearest phosphates in a z-slab around peptide ---
        slab_mask = np.abs(z_phos_all - pep_z) < args.slab_half_thickness
        if slab_mask.sum() < args.n_nearest_phosphates:
            radius = np.nan
        else:
            xy_phos = phosphates.positions[slab_mask][:, :2]
            dists = xy_min_image_dist(xy_phos, pep_xy, box)
            nearest = np.sort(dists)[:args.n_nearest_phosphates]
            radius = max(nearest.mean() - args.probe_radius, 0.0)

        pore_radii.append(radius)
        if not np.isnan(radius):
            radius_samples.append(radius)

        cutoff = (args.water_cylinder_radius
                  if (args.use_fixed_cylinder_radius or np.isnan(radius))
                  else radius)

        # --- water positions / distances (computed once per frame) ---
        w_pos = water_o.positions
        w_xy = w_pos[:, :2]
        w_z = w_pos[:, 2]
        dist_xy = xy_min_image_dist(w_xy, pep_xy, box)

        # (a) legacy "loose" cylinder water: xy < cutoff AND generic z-slab
        in_slab = np.abs(w_z - pep_z) < args.slab_half_thickness
        in_loose_cylinder = (dist_xy < cutoff) & in_slab
        loose_water_counts.append(int(in_loose_cylinder.sum()))

        # (b) pore water, exclusive: xy < cutoff AND within hydrophobic core
        in_core_z = (w_z > core_lower) & (w_z < core_upper)
        in_pore_exclusive = (dist_xy < cutoff) & in_core_z
        pore_water_exclusive_counts.append(int(in_pore_exclusive.sum()))

        # (c) hydrophobic core water, total: within core z-range AND within
        #     a local xy patch around the peptide (independent of pore radius)
        in_core_patch = (dist_xy < args.core_xy_radius) & in_core_z
        hydrophobic_core_water_counts.append(int(in_core_patch.sum()))

        # (d) midplane-penetrating water: waters sitting right at the
        #     bilayer midplane (|z - z_mid| < margin), within an xy radius
        #     of the peptide (defaults to the same pore cylinder cutoff)
        midplane_xy_cutoff = (args.midplane_water_xy_radius
                               if args.midplane_water_xy_radius is not None
                               else cutoff)
        w_z_abs = w_pos[:, 2]
        dist_xy_from_pep = dist_xy  # already computed relative to peptide xy
        in_midplane_z = np.abs(w_z_abs - z_mid) < args.midplane_water_margin
        in_midplane = (dist_xy_from_pep < midplane_xy_cutoff) & in_midplane_z
        midplane_water_counts.append(int(in_midplane.sum()))

        # --- structural snapshot capture (peptide + lipids + midplane water,
        #     loose-cylinder water, and pore-size geometry) ---
        if fi in snapshot_frame_map:
            bb_pos = backbone.positions
            pep_x0 = pep_xy[0]

            phos_pos = phosphates.positions
            phos_dist_xy = xy_min_image_dist(phos_pos[:, :2], pep_xy, box)
            phos_near = phos_dist_xy < args.snapshot_near_lipid_cutoff
            # phosphates actually used for the pore-radius estimate this frame
            if slab_mask.sum() >= args.n_nearest_phosphates:
                slab_idx = np.where(slab_mask)[0]
                slab_dists = xy_min_image_dist(
                    phosphates.positions[slab_mask][:, :2], pep_xy, box)
                order = np.argsort(slab_dists)[:args.n_nearest_phosphates]
                pore_forming_global_idx = slab_idx[order]
            else:
                pore_forming_global_idx = np.array([], dtype=int)
            phos_pore_forming = np.zeros(len(phos_pos), dtype=bool)
            phos_pore_forming[pore_forming_global_idx] = True

            # side-view (x-z) midplane-penetrating water, as before
            snap_water_mask = ((dist_xy < args.snapshot_water_xy_radius) &
                                in_midplane_z)

            # side-view (x-z) loose-cylinder water: EXACT same criterion as
            # the loose_water_counts time series (per-frame pore/fixed
            # cylinder radius, z-slab around peptide z)
            loose_snap_mask = in_loose_cylinder

            # side-view (x-z) pore water, exclusive: EXACT same criterion as
            # the pore_water_exclusive_counts time series (xy < cutoff AND
            # within hydrophobic core z-range)
            pore_excl_snap_mask = in_pore_exclusive

            # side-view (x-z) hydrophobic core water, total: EXACT same
            # criterion as the hydrophobic_core_water_counts time series
            # (local xy patch around peptide AND within core z-range)
            core_snap_mask = in_core_patch

            # top-down (x-y) relative positions, PBC-corrected, for the
            # pore-size snapshot
            phos_xy_rel = xy_min_image_vector(phos_pos[:, :2], pep_xy, box)
            loose_water_xy_rel = xy_min_image_vector(
                w_pos[loose_snap_mask][:, :2], pep_xy, box)

            snapshot_data[fi] = dict(
                time_ns=snapshot_frame_map[fi],
                bb_x=bb_pos[:, 0] - pep_x0,
                bb_z=bb_pos[:, 2] - z_mid,
                phos_x=phos_pos[:, 0] - pep_x0,
                phos_z=phos_pos[:, 2] - z_mid,
                phos_near=phos_near,
                water_x=w_pos[snap_water_mask, 0] - pep_x0,
                water_z=w_pos[snap_water_mask, 2] - z_mid,
                loose_water_x=w_pos[loose_snap_mask, 0] - pep_x0,
                loose_water_z=w_pos[loose_snap_mask, 2] - z_mid,
                pore_excl_water_x=w_pos[pore_excl_snap_mask, 0] - pep_x0,
                pore_excl_water_z=w_pos[pore_excl_snap_mask, 2] - z_mid,
                core_water_x=w_pos[core_snap_mask, 0] - pep_x0,
                core_water_z=w_pos[core_snap_mask, 2] - z_mid,
                pore_radius=radius,
                phos_xy_rel=phos_xy_rel,
                phos_pore_forming=phos_pore_forming,
                loose_water_xy_rel=loose_water_xy_rel,
            )

        # --- z-density histogram for the water free-energy profile ---
        in_patch_xy = dist_xy < args.core_xy_radius
        z_rel = w_z[in_patch_xy] - pep_z
        counts, _ = np.histogram(z_rel, bins=z_bins)
        z_hist_counts += counts
        n_frames_for_z_hist += 1

        # --- permeation events: hysteresis sign-crossing state machine ---
        tracked = in_loose_cylinder
        z_diff = w_z - pep_z
        side = np.where(z_diff > args.hysteresis_margin, 1.0,
                np.where(z_diff < -args.hysteresis_margin, -1.0, 0.0))

        has_committed = ~np.isnan(committed_sign)
        decisive = tracked & (side != 0.0)

        crossed = decisive & has_committed & (side != committed_sign)
        events_this_frame = int(crossed.sum())
        permeation_events.append(events_this_frame)

        update_mask = decisive
        committed_sign = np.where(update_mask, side, committed_sign)
        committed_sign = np.where(tracked, committed_sign, np.nan)

    times = np.array(times)
    depths = np.array(depths)
    upper_rel = np.array(upper_rel)
    lower_rel = np.array(lower_rel)
    pore_radii = np.array(pore_radii)
    loose_water_counts = np.array(loose_water_counts)
    pore_water_exclusive_counts = np.array(pore_water_exclusive_counts)
    hydrophobic_core_water_counts = np.array(hydrophobic_core_water_counts)
    midplane_water_counts = np.array(midplane_water_counts)
    permeation_events = np.array(permeation_events)
    cum_permeations = np.cumsum(permeation_events)

    results = dict(
        times=times,
        depths=depths, upper_rel=upper_rel, lower_rel=lower_rel,
        pore_radii=pore_radii,
        loose_water_counts=loose_water_counts,
        pore_water_exclusive_counts=pore_water_exclusive_counts,
        hydrophobic_core_water_counts=hydrophobic_core_water_counts,
        midplane_water_counts=midplane_water_counts,
        permeation_events=permeation_events,
        cum_permeations=cum_permeations,
        radius_samples=np.array(radius_samples),
        z_bins=z_bins,
        z_hist_counts=z_hist_counts,
        n_frames_for_z_hist=n_frames_for_z_hist,
        snapshot_data=snapshot_data,
    )
    return results


# =============================================================================
# FREE ENERGY PROFILES (Boltzmann inversion)
# =============================================================================

def free_energy_from_samples(samples, n_bins, temperature):
    """G(r) = -kB T ln(P(r)/P_max), referenced to 0 at the most-populated bin."""
    samples = samples[np.isfinite(samples)]
    if len(samples) < n_bins:
        return None, None, None
    counts, edges = np.histogram(samples, bins=n_bins, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide="ignore"):
        prob = counts / counts.sum()
        g = -KB_KCAL * temperature * np.log(prob / prob.max())
    g[counts == 0] = np.nan
    return centers, g, counts


def free_energy_from_z_histogram(z_bins, z_hist_counts, bulk_z_cutoff, temperature):
    """
    G(z) = -kB T ln(rho(z)/rho_bulk), where rho_bulk is the mean count in
    bins with |z_center| > bulk_z_cutoff (i.e. far from the membrane).
    """
    centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    bulk_mask = np.abs(centers) > bulk_z_cutoff
    if not bulk_mask.any() or z_hist_counts[bulk_mask].sum() == 0:
        print("WARNING: no bulk-water reference region found for the z-PMF "
              "(check --z_range / --bulk_z_cutoff against your box size). "
              "Skipping water free-energy profile.")
        return centers, np.full_like(centers, np.nan), z_hist_counts
    rho_bulk = z_hist_counts[bulk_mask].mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        g = -KB_KCAL * temperature * np.log(z_hist_counts / rho_bulk)
    g[z_hist_counts == 0] = np.nan
    return centers, g, z_hist_counts


# =============================================================================
# OUTPUT: DATA FILES
# =============================================================================

def save_outputs(results, args):
    pfx = args.out_prefix

    np.savetxt(f"{pfx}insertion_depth_clean.dat",
               np.column_stack((results["times"], results["depths"])),
               header="Time(ps)\tDepth(A)", fmt="%12.4f")

    np.savetxt(f"{pfx}pore_radius_water.dat",
               np.column_stack((results["times"], results["pore_radii"],
                                 results["loose_water_counts"],
                                 results["permeation_events"],
                                 results["cum_permeations"])),
               header="Time(ps)\tPoreRadius(A)\tLooseWatersInCylinder\t"
                      "PermeationEvents(frame)\tCumulativePermeations",
               fmt="%14.4f")

    np.savetxt(f"{pfx}water_breakdown.dat",
               np.column_stack((results["times"],
                                 results["loose_water_counts"],
                                 results["pore_water_exclusive_counts"],
                                 results["hydrophobic_core_water_counts"],
                                 results["midplane_water_counts"])),
               header="Time(ps)\tLooseCylinderWater\tPoreWaterExclusive\t"
                      "HydrophobicCoreWaterTotal\tMidplaneWater",
               fmt="%14.4f")

    print("Data saved: insertion_depth_clean.dat, pore_radius_water.dat, "
          "water_breakdown.dat")


def save_and_plot_free_energy(results, args):
    pfx = args.out_prefix

    r_centers, g_r, r_counts = free_energy_from_samples(
        results["radius_samples"], args.pore_radius_bins, args.temperature)

    z_centers, g_z, z_counts = free_energy_from_z_histogram(
        results["z_bins"], results["z_hist_counts"], args.bulk_z_cutoff,
        args.temperature)

    if r_centers is not None:
        np.savetxt(f"{pfx}free_energy_pore_radius.dat",
                   np.column_stack((r_centers, g_r, r_counts)),
                   header="PoreRadius(A)\tG(kcal/mol)\tCounts", fmt="%14.6f")

    np.savetxt(f"{pfx}free_energy_water_z.dat",
               np.column_stack((z_centers, g_z, z_counts)),
               header="Z_relative_to_peptide(A)\tG(kcal/mol)\tCounts",
               fmt="%14.6f")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    if r_centers is not None:
        axes[0].plot(r_centers, g_r, "o-", color="black", lw=1.5, ms=4)
        axes[0].set_xlabel("Pore radius [A]", fontsize=12)
        axes[0].set_ylabel("G [kcal/mol]", fontsize=12)
        axes[0].set_title("Pore-radius free energy\n(Boltzmann inversion)", fontsize=11)
    else:
        axes[0].text(0.5, 0.5, "Not enough valid\npore-radius samples",
                     ha="center", va="center", transform=axes[0].transAxes)

    axes[1].plot(z_centers, g_z, "-", color="#0072B2", lw=2)
    axes[1].set_xlabel("Z relative to peptide [A]", fontsize=12)
    axes[1].set_ylabel("G [kcal/mol]", fontsize=12)
    axes[1].set_title("Water z-density free energy\n(pore-column, vs. bulk)", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{pfx}free_energy_profiles.png", dpi=300)
    plt.close()
    print(f"Saved plot: {pfx}free_energy_profiles.png")


# =============================================================================
# OUTPUT: PLOTS
# =============================================================================

def plot_depth(results, args):
    pfx = args.out_prefix
    times_ns = results["times"] / 1000.0
    depths = results["depths"]
    upper = results["upper_rel"]
    lower = results["lower_rel"]

    depths_smooth = gaussian_filter1d(depths, sigma=2)
    mean_depth = np.mean(depths)

    plt.figure(figsize=(8, 4.5))
    plt.plot(times_ns, depths_smooth, color="black", lw=2)
    plt.axhline(0, linestyle=":", color="black", lw=1)
    plt.axhline(np.mean(upper), linestyle=":", color="#E69F00", lw=1.5)
    plt.axhline(np.mean(lower), linestyle=":", color="#E69F00", lw=1.5)
    plt.fill_between(times_ns, np.mean(lower), np.mean(upper), color="gray", alpha=0.2)
    plt.axhline(mean_depth, linestyle="--", color="#E69F00", lw=1.5)
    plt.xlabel("Simulation time [ns]", fontsize=12)
    plt.ylabel("Membrane normal [A]", fontsize=12)
    plt.xlim(times_ns.min(), times_ns.max())
    plt.ylim(-50, 50)
    plt.tight_layout()
    plt.savefig(f"{pfx}insertion_depth_clean.png", dpi=300)
    plt.close()

    print(f"Saved plot: {pfx}insertion_depth_clean.png")
    print(f"Mean depth: {mean_depth:.2f} A")


def plot_pore_and_permeation(results, args):
    pfx = args.out_prefix
    times_ns = results["times"] / 1000.0
    pore_radii = results["pore_radii"]
    water_counts = results["loose_water_counts"]
    cum_permeations = results["cum_permeations"]

    valid = ~np.isnan(pore_radii)
    pore_smooth = np.copy(pore_radii)
    if valid.sum() > 3:
        pore_smooth[valid] = gaussian_filter1d(pore_radii[valid], sigma=2)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(times_ns, pore_smooth, color="black", lw=2)
    axes[0].axhline(np.nanmean(pore_radii), linestyle="--", color="#E69F00", lw=1.5)
    axes[0].set_ylabel("Pore radius [A]", fontsize=12)

    water_smooth = gaussian_filter1d(water_counts.astype(float), sigma=2)
    axes[1].plot(times_ns, water_smooth, color="#0072B2", lw=1.8)
    axes[1].set_ylabel("Loose cylinder\nwater (count)", fontsize=12)

    axes[2].plot(times_ns, cum_permeations, color="#D55E00", lw=2)
    axes[2].set_ylabel("Cumulative\npermeation events", fontsize=12)
    axes[2].set_xlabel("Simulation time [ns]", fontsize=12)
    axes[2].set_xlim(times_ns.min(), times_ns.max())

    plt.tight_layout()
    plt.savefig(f"{pfx}pore_and_water.png", dpi=300)
    plt.close()

    print(f"Saved plot: {pfx}pore_and_water.png")
    print(f"Mean pore radius: {np.nanmean(pore_radii):.2f} A")
    print(f"Mean loose cylinder water: {np.mean(water_counts):.2f}")
    print(f"Total permeation events: {int(cum_permeations[-1]) if len(cum_permeations) else 0}")


def plot_water_breakdown(results, args):
    pfx = args.out_prefix
    times_ns = results["times"] / 1000.0

    loose = gaussian_filter1d(results["loose_water_counts"].astype(float), sigma=2)
    excl = gaussian_filter1d(results["pore_water_exclusive_counts"].astype(float), sigma=2)
    core = gaussian_filter1d(results["hydrophobic_core_water_counts"].astype(float), sigma=2)

    plt.figure(figsize=(8, 4.5))
    plt.plot(times_ns, loose, label="Loose cylinder water (legacy)", color="#999999", lw=1.5)
    plt.plot(times_ns, excl, label="Pore water (exclusive)", color="#0072B2", lw=2)
    plt.plot(times_ns, core, label="Hydrophobic core water (local patch)", color="#D55E00", lw=2)
    plt.xlabel("Simulation time [ns]", fontsize=12)
    plt.ylabel("Water count", fontsize=12)
    plt.xlim(times_ns.min(), times_ns.max())
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{pfx}water_breakdown.png", dpi=300)
    plt.close()

    print(f"Saved plot: {pfx}water_breakdown.png")
    print(f"Mean pore water (exclusive): {np.mean(results['pore_water_exclusive_counts']):.2f}")
    print(f"Mean hydrophobic core water (local patch): "
          f"{np.mean(results['hydrophobic_core_water_counts']):.2f}")


def plot_pore_water_exclusive(results, args):
    """Pore water (exclusive) count over time, saved as its own file."""
    pfx = args.out_prefix
    times_ns = results["times"] / 1000.0
    water_smooth = gaussian_filter1d(
        results["pore_water_exclusive_counts"].astype(float), sigma=2)

    plt.figure(figsize=(8, 4.5))
    plt.plot(times_ns, water_smooth, color="#0072B2", lw=2)
    plt.axhline(np.mean(results["pore_water_exclusive_counts"]), linestyle="--",
                color="#E69F00", lw=1.5)
    plt.xlabel("Simulation time [ns]", fontsize=12)
    plt.ylabel("Pore water, exclusive (count)", fontsize=12)
    plt.xlim(times_ns.min(), times_ns.max())
    plt.tight_layout()
    plt.savefig(f"{pfx}pore_water_exclusive.png", dpi=300)
    plt.close()

    print(f"Saved plot: {pfx}pore_water_exclusive.png")
    print(f"Mean pore water (exclusive): "
          f"{np.mean(results['pore_water_exclusive_counts']):.2f}")


def plot_hydrophobic_core_water(results, args):
    """Hydrophobic core water (local patch) count over time, saved as its
    own file."""
    pfx = args.out_prefix
    times_ns = results["times"] / 1000.0
    water_smooth = gaussian_filter1d(
        results["hydrophobic_core_water_counts"].astype(float), sigma=2)

    plt.figure(figsize=(8, 4.5))
    plt.plot(times_ns, water_smooth, color="#D55E00", lw=2)
    plt.axhline(np.mean(results["hydrophobic_core_water_counts"]), linestyle="--",
                color="#E69F00", lw=1.5)
    plt.xlabel("Simulation time [ns]", fontsize=12)
    plt.ylabel("Hydrophobic core water (count)", fontsize=12)
    plt.xlim(times_ns.min(), times_ns.max())
    plt.tight_layout()
    plt.savefig(f"{pfx}hydrophobic_core_water.png", dpi=300)
    plt.close()

    print(f"Saved plot: {pfx}hydrophobic_core_water.png")
    print(f"Mean hydrophobic core water (local patch): "
          f"{np.mean(results['hydrophobic_core_water_counts']):.2f}")


def plot_loose_cylinder_water(results, args):
    """Loose-cylinder water count over time, saved as its own file."""
    pfx = args.out_prefix
    times_ns = results["times"] / 1000.0
    water_smooth = gaussian_filter1d(results["loose_water_counts"].astype(float), sigma=2)

    plt.figure(figsize=(8, 4.5))
    plt.plot(times_ns, water_smooth, color="#0072B2", lw=2)
    plt.axhline(np.mean(results["loose_water_counts"]), linestyle="--",
                color="#E69F00", lw=1.5)
    plt.xlabel("Simulation time [ns]", fontsize=12)
    plt.ylabel("Loose cylinder water (count)", fontsize=12)
    plt.xlim(times_ns.min(), times_ns.max())
    plt.tight_layout()
    plt.savefig(f"{pfx}loose_cylinder_water.png", dpi=300)
    plt.close()

    print(f"Saved plot: {pfx}loose_cylinder_water.png")
    print(f"Mean loose cylinder water: {np.mean(results['loose_water_counts']):.2f}")


def plot_midplane_water(results, args):
    """Midplane-penetrating water count over time, saved as its own file."""
    pfx = args.out_prefix
    times_ns = results["times"] / 1000.0
    water_smooth = gaussian_filter1d(results["midplane_water_counts"].astype(float), sigma=2)

    plt.figure(figsize=(8, 4.5))
    plt.plot(times_ns, water_smooth, color="#00B8B8", lw=2)  # cyan, readable on white
    plt.axhline(np.mean(results["midplane_water_counts"]), linestyle="--",
                color="#E69F00", lw=1.5)
    plt.xlabel("Simulation time [ns]", fontsize=12)
    plt.ylabel("Midplane water (count)", fontsize=12)
    plt.xlim(times_ns.min(), times_ns.max())
    plt.tight_layout()
    plt.savefig(f"{pfx}midplane_water.png", dpi=300)
    plt.close()

    print(f"Saved plot: {pfx}midplane_water.png")
    print(f"Mean midplane water: {np.mean(results['midplane_water_counts']):.2f}")


def plot_water_counts_combined(results, args):
    """
    Loose-cylinder water and midplane water counts, side by side in one
    figure for easy visual comparison (in addition to the two separate
    single-panel files saved by plot_loose_cylinder_water/plot_midplane_water).
    """
    pfx = args.out_prefix
    times_ns = results["times"] / 1000.0

    loose_smooth = gaussian_filter1d(results["loose_water_counts"].astype(float), sigma=2)
    mid_smooth = gaussian_filter1d(results["midplane_water_counts"].astype(float), sigma=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    axes[0].plot(times_ns, loose_smooth, color="#0072B2", lw=2)
    axes[0].axhline(np.mean(results["loose_water_counts"]), linestyle="--",
                     color="#E69F00", lw=1.5)
    axes[0].set_xlabel("Simulation time [ns]", fontsize=12)
    axes[0].set_ylabel("Loose cylinder water (count)", fontsize=12)
    axes[0].set_title("Loose cylinder water", fontsize=11)

    axes[1].plot(times_ns, mid_smooth, color="#00B0B0", lw=2)
    axes[1].axhline(np.mean(results["midplane_water_counts"]), linestyle="--",
                     color="#E69F00", lw=1.5)
    axes[1].set_xlabel("Simulation time [ns]", fontsize=12)
    axes[1].set_ylabel("Midplane water (count)", fontsize=12)
    axes[1].set_title("Midplane-penetrating water", fontsize=11)

    for ax in axes:
        ax.set_xlim(times_ns.min(), times_ns.max())

    plt.tight_layout()
    plt.savefig(f"{pfx}water_counts_combined.png", dpi=300)
    plt.close()

    print(f"Saved plot: {pfx}water_counts_combined.png")


def _draw_snapshot_panel(ax, snap, water_x_key="water_x", water_z_key="water_z",
                          water_color="cyan", water_edge="#008080"):
    """Draw one structural snapshot (peptide trace + phosphates + water)
    onto a matplotlib axis, styled after the reference figure. Which water
    subset is drawn is controlled by water_x_key/water_z_key so the same
    routine can render either the midplane-penetrating water snapshot or
    the loose-cylinder water snapshot."""
    bb_x, bb_z = snap["bb_x"], snap["bb_z"]
    n_pts = len(bb_x)

    # phosphates: gray by default, orange if within the near-lipid cutoff
    near = snap["phos_near"]
    ax.scatter(snap["phos_x"][~near], snap["phos_z"][~near],
               s=90, color="#808080", edgecolor="white", linewidth=0.5,
               zorder=2)
    ax.scatter(snap["phos_x"][near], snap["phos_z"][near],
               s=90, color="#FF8C00", edgecolor="white", linewidth=0.5,
               zorder=2)

    # water subset (midplane-penetrating OR loose-cylinder, per caller)
    w_x, w_z = snap[water_x_key], snap[water_z_key]
    if len(w_x) > 0:
        ax.scatter(w_x, w_z, s=90, color=water_color, edgecolor=water_edge,
                   linewidth=0.8, zorder=4)

    # peptide backbone trace: light gray markers, size grows toward termini
    sizes = np.linspace(40, 140, n_pts)
    ax.plot(bb_x, bb_z, color="gray", lw=1.5, zorder=3)
    ax.scatter(bb_x, bb_z, s=sizes, color="#D3D3D3", edgecolor="black",
               linewidth=0.6, zorder=5)

    # N-/C-terminal markers
    ax.scatter([bb_x[0]], [bb_z[0]], s=260, color="#0000CD",
               edgecolor="black", linewidth=1.0, zorder=6)
    ax.annotate("N", (bb_x[0], bb_z[0]), color="white", fontsize=11,
                fontweight="bold", ha="center", va="center", zorder=7)
    ax.scatter([bb_x[-1]], [bb_z[-1]], s=260, color="#CD0000",
               edgecolor="black", linewidth=1.0, zorder=6)
    ax.annotate("C", (bb_x[-1], bb_z[-1]), color="white", fontsize=11,
                fontweight="bold", ha="center", va="center", zorder=7)

    ax.axhline(0, linestyle=":", color="black", lw=1, zorder=1)
    ax.set_xlabel("x [A]", fontsize=11)


def _render_snapshot_series(results, args, out_subdir, combined_filename,
                             suptitle, water_x_key, water_z_key, water_color,
                             water_edge, count_key=None):
    """
    Shared driver for the x-z side-view snapshot series (used for both the
    midplane-water and loose-cylinder-water snapshot variants): saves one
    PNG per captured frame, plus one combined multi-panel overview.
    """
    pfx = args.out_prefix
    snapshot_data = results["snapshot_data"]
    if not snapshot_data:
        print("No snapshot frames captured; skipping snapshot plots.")
        return

    import os
    snap_dir = os.path.join(args.snapshot_dir or f"{pfx}snapshots", out_subdir)
    os.makedirs(snap_dir, exist_ok=True)

    ordered = sorted(snapshot_data.items(), key=lambda kv: kv[1]["time_ns"])

    for fi, snap in ordered:
        fig, ax = plt.subplots(figsize=(6, 7))
        _draw_snapshot_panel(ax, snap, water_x_key, water_z_key,
                              water_color, water_edge)
        ax.set_ylabel("z [A] (midplane = 0)", fontsize=11)
        title = f"{snap['time_ns']:.1f} ns"
        if count_key is not None:
            title += f"  (n={len(snap[water_x_key])})"
        ax.set_title(title, fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(snap_dir, f"{out_subdir}_{snap['time_ns']:.1f}ns.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved plot: {out_path}")

    n_panels = len(ordered)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 7), sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, (fi, snap) in zip(axes, ordered):
        _draw_snapshot_panel(ax, snap, water_x_key, water_z_key,
                              water_color, water_edge)
        ax.set_title(f"{snap['time_ns']:.1f} ns", fontsize=12)
    axes[0].set_ylabel("z [A] (midplane = 0)", fontsize=12)
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{pfx}{combined_filename}", dpi=300)
    plt.close()
    print(f"Saved plot: {pfx}{combined_filename}")


def plot_snapshots(results, args):
    """
    Structural snapshots of peptide insertion at selected times: backbone
    trace (N/C termini highlighted), lipid phosphates (orange = near the
    peptide, gray = far), and water molecules that have penetrated into
    the bilayer midplane (cyan). Criterion: |z - z_mid| < midplane_water_margin
    AND xy distance to peptide < snapshot_water_xy_radius. Each snapshot is
    saved as its own PNG, plus one combined multi-panel overview.
    """
    _render_snapshot_series(
        results, args, out_subdir="midplane_water",
        combined_filename="snapshot_over_time.png",
        suptitle="insertion over time (midplane-penetrating water)",
        water_x_key="water_x", water_z_key="water_z",
        water_color="cyan", water_edge="#008080")


def plot_loose_cylinder_snapshots(results, args):
    """
    Same structural snapshot as plot_snapshots, but the highlighted water
    uses the EXACT loose-cylinder criterion from the loose_water_counts
    time series (per-frame pore/fixed cylinder xy-radius AND a z-slab of
    +/- slab_half_thickness around the peptide's z), instead of the
    midplane-only criterion. Saved separately from the midplane variant.
    """
    _render_snapshot_series(
        results, args, out_subdir="loose_cylinder_water",
        combined_filename="loose_cylinder_snapshot_over_time.png",
        suptitle="insertion over time (loose-cylinder water)",
        water_x_key="loose_water_x", water_z_key="loose_water_z",
        water_color="#1E90FF", water_edge="#00008B", count_key="loose")


def plot_pore_water_exclusive_snapshots(results, args):
    """
    Same structural snapshot as plot_snapshots, but the highlighted water
    uses the EXACT pore-water-exclusive criterion from the
    pore_water_exclusive_counts time series (xy < cutoff AND within the
    hydrophobic core z-range), instead of the midplane-only criterion.
    Saved separately from the midplane/loose-cylinder variants.
    """
    _render_snapshot_series(
        results, args, out_subdir="pore_water_exclusive",
        combined_filename="pore_water_exclusive_snapshot_over_time.png",
        suptitle="insertion over time (pore water, exclusive)",
        water_x_key="pore_excl_water_x", water_z_key="pore_excl_water_z",
        water_color="#0072B2", water_edge="#003D5C", count_key="pore_excl")


def plot_hydrophobic_core_water_snapshots(results, args):
    """
    Same structural snapshot as plot_snapshots, but the highlighted water
    uses the EXACT hydrophobic-core-water criterion from the
    hydrophobic_core_water_counts time series (local xy patch around the
    peptide AND within the hydrophobic core z-range), instead of the
    midplane-only criterion. Saved separately from the other variants.
    """
    _render_snapshot_series(
        results, args, out_subdir="hydrophobic_core_water",
        combined_filename="hydrophobic_core_water_snapshot_over_time.png",
        suptitle="insertion over time (hydrophobic core water)",
        water_x_key="core_water_x", water_z_key="core_water_z",
        water_color="#D55E00", water_edge="#8B3A00", count_key="core")


def _draw_pore_size_panel(ax, snap, xy_range):
    """Top-down (x-y) view of the estimated pore at one snapshot frame:
    nearby phosphates (gray), the phosphates actually used to estimate the
    pore radius this frame (purple), loose-cylinder water inside the pore
    column (blue), the peptide center (black cross), and a dashed circle
    of the estimated pore radius."""
    phos_xy = snap["phos_xy_rel"]
    pore_forming = snap["phos_pore_forming"]
    radius = snap["pore_radius"]

    in_view = (np.abs(phos_xy[:, 0]) < xy_range) & (np.abs(phos_xy[:, 1]) < xy_range)

    other_mask = in_view & ~pore_forming
    ax.scatter(phos_xy[other_mask, 0], phos_xy[other_mask, 1],
               s=90, color="#808080", edgecolor="white", linewidth=0.5, zorder=2)

    form_mask = in_view & pore_forming
    ax.scatter(phos_xy[form_mask, 0], phos_xy[form_mask, 1],
               s=110, color="#8B008B", edgecolor="white", linewidth=0.6, zorder=3,
               label="Pore-lining phosphate (used for radius)")

    lw_xy = snap["loose_water_xy_rel"]
    if len(lw_xy) > 0:
        in_view_w = (np.abs(lw_xy[:, 0]) < xy_range) & (np.abs(lw_xy[:, 1]) < xy_range)
        ax.scatter(lw_xy[in_view_w, 0], lw_xy[in_view_w, 1],
                   s=70, color="#1E90FF", edgecolor="#00008B", linewidth=0.7,
                   zorder=4, label="Water in cylinder")

    ax.scatter([0], [0], s=140, marker="+", color="black", linewidth=2.5, zorder=6)

    if not np.isnan(radius):
        circle = plt.Circle((0, 0), radius, fill=False, linestyle="--",
                             edgecolor="black", linewidth=1.8, zorder=5)
        ax.add_patch(circle)
        label = f"r = {radius:.1f} A"
    else:
        label = "r = n/a"

    ax.set_xlim(-xy_range, xy_range)
    ax.set_ylim(-xy_range, xy_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [A]", fontsize=11)
    ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=10,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8))


def plot_pore_size_snapshots(results, args):
    """
    Top-down (x-y) snapshots of the estimated pore at selected times:
    nearby phosphates (gray), the N nearest phosphates actually used to
    estimate the pore radius (purple), water inside the loose cylinder
    (blue), and a dashed circle showing the estimated pore radius. Each
    snapshot is saved as its own PNG, plus one combined multi-panel
    overview.
    """
    pfx = args.out_prefix
    snapshot_data = results["snapshot_data"]
    if not snapshot_data:
        print("No snapshot frames captured; skipping pore-size snapshot plots.")
        return

    import os
    snap_dir = os.path.join(args.snapshot_dir or f"{pfx}snapshots", "pore_size")
    os.makedirs(snap_dir, exist_ok=True)

    xy_range = args.pore_snapshot_xy_range
    ordered = sorted(snapshot_data.items(), key=lambda kv: kv[1]["time_ns"])

    for fi, snap in ordered:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        _draw_pore_size_panel(ax, snap, xy_range)
        ax.set_ylabel("y [A]", fontsize=11)
        ax.set_title(f"{snap['time_ns']:.1f} ns", fontsize=12)
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        out_path = os.path.join(snap_dir, f"pore_size_{snap['time_ns']:.1f}ns.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved plot: {out_path}")

    n_panels = len(ordered)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.6), sharey=True)
    if n_panels == 1:
        axes = [axes]
    for ax, (fi, snap) in zip(axes, ordered):
        _draw_pore_size_panel(ax, snap, xy_range)
        ax.set_title(f"{snap['time_ns']:.1f} ns", fontsize=12)
    axes[0].set_ylabel("y [A]", fontsize=12)
    fig.suptitle("pore size over time", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{pfx}pore_size_snapshot_over_time.png", dpi=300)
    plt.close()
    print(f"Saved plot: {pfx}pore_size_snapshot_over_time.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = build_parser().parse_args()

    # =========================================================================
    # OUTPUT DIRECTORY
    # =========================================================================
    # Every plot/data file in this script is built as f"{pfx}some_file.png",
    # i.e. args.out_prefix is glued directly onto filenames. That silently
    # breaks when out_prefix is a bare directory path with no trailing
    # separator (e.g. passed by the launcher GUI as --out_prefix
    # "/path/to/output_folder") -- the files then land one level *above*
    # that folder, with garbled concatenated names, instead of inside it.
    # Normalize out_prefix here into a real, guaranteed-to-exist output
    # folder, and make sure it ends with a path separator so every
    # downstream f"{pfx}filename" still works exactly as written.
    out_dir = (args.out_prefix or "").strip()
    out_dir = out_dir.rstrip("/\\")
    if not out_dir or out_dir == ".":
        out_dir = "depth_pore_permeation_output"
    os.makedirs(out_dir, exist_ok=True)
    args.out_prefix = out_dir + os.sep
    print(f"[INFO] Saving all outputs to: {os.path.abspath(out_dir)}")

    print("Loading system...")
    u = mda.Universe(args.tpr, args.xtc)

    peptide, phosphates, water_o, backbone = select_groups(u, args)
    setup_unwrap(u, peptide)

    print("Running single-pass analysis "
          "(depth, pore radius, water breakdown, permeation, snapshots)...")
    results = run_analysis(u, peptide, phosphates, water_o, backbone, args)

    save_outputs(results, args)
    plot_depth(results, args)
    plot_pore_and_permeation(results, args)
    plot_water_breakdown(results, args)
    plot_loose_cylinder_water(results, args)
    plot_midplane_water(results, args)
    plot_pore_water_exclusive(results, args)
    plot_hydrophobic_core_water(results, args)
    plot_water_counts_combined(results, args)
    plot_snapshots(results, args)
    plot_loose_cylinder_snapshots(results, args)
    plot_pore_water_exclusive_snapshots(results, args)
    plot_hydrophobic_core_water_snapshots(results, args)
    plot_pore_size_snapshots(results, args)

    print("Computing free-energy profiles (Boltzmann inversion)...")
    save_and_plot_free_energy(results, args)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nFailed with an exception:\n")
        traceback.print_exc()
        sys.exit(1)
