#!/usr/bin/env python3
"""
MD Membrane Analysis Suite — Launcher GUI
==========================================

A single, professional point of entry for the lab's collection of
MDAnalysis-based membrane/curvature/pore/density analysis scripts. Pick an
analysis family from the catalog sidebar, pick the specific script variant,
fill in your topology/trajectory files (and a couple of analysis
parameters), and hit Run. Output (logs + figures) is streamed live to an
in-app console and saved into the output folder you choose.

Run with:   python launcher.py
Requires:   Python 3.8+ (Tkinter is part of the standard library — the
            launcher itself has no third-party dependencies, including for
            the animated GIF preview, which uses Tk's native multi-frame
            GIF support).
            The underlying analysis scripts additionally require
            MDAnalysis, numpy, scipy, matplotlib, tqdm, etc. — whatever
            each individual script already imports.

Folder layout expected:
    launcher.py
    porous.gif                 (optional — shown as an animated preview
                                 in the header banner and About dialog)
    scripts/
        advanced_membrane_analysis_Version34.py
        ... (all the analysis scripts, unmodified)

License:    MIT License — see LICENSE for the full text.
"""

import os
import re
import sys
import shlex
import queue
import shutil
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from fractions import Fraction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")


# =============================================================================
# CATALOG
# =============================================================================
# Each top-level key is an "analysis family" (main option button). Inside it,
# "variants" lists the individual scripts that perform that family of
# analysis but differ in some way (sub-option buttons). "patch" tells the
# launcher how to inject the user's chosen TPR/XTC/lipid/peptide values into
# that particular script before running it (since these scripts hard-code
# those values at the top of the file rather than taking them as CLI args).
#
# patch fields:
#   tpr_var / xtc_var   : name of the variable holding the topology/trajectory
#                          path in that script (string-valued).
#   lipid_var           : name of the variable holding lipid residue names.
#   lipid_format        : "list"  -> LIPIDS = ["POPE", "POPG"]
#                          "space" -> LIPID_RESNAMES = "POPE POPG"
#                          "space_with_resname" -> "resname POPE POPG"
#   peptide_var         : name of the variable holding the peptide/protein
#                          residue name (string), if the script exposes one.
#   cli                 : if set, this script is launched via its own
#                          argparse CLI instead of source patching.

CATALOG = {
    "Membrane Surface & Topology Analysis": {
        "description": (
            "Builds upper/lower leaflet surfaces from the bilayer, then maps "
            "membrane thickness, 2D/3D topology, deformation, and "
            "peptide-insertion depth relative to the membrane."
        ),
        "variants": [
            {
                "file": "advanced_membrane_analysis_Version34.py",
                "label": "Version 34 (baseline)",
                "desc": "Original baseline surface/topology analyzer.",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "advanced_membrane_analysis_Version34_UPDATED_WITH_LEAFLET_STRUCTURE.py",
                "label": "Version 34 + Leaflet Structure (a)",
                "desc": "Adds explicit leaflet-structure reconstruction on top of v34.",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "advanced_membrane_analysis_Version34_UPDATED_WITH_LEAFLET_STRUCTURE1.py",
                "label": "Version 34 + Leaflet Structure (b)",
                "desc": "Second iteration of the leaflet-structure update to v34.",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "advanced_membrane_analysis_Version34_UPDATED_WITH_LEAFLET_STRUCTURE2.py",
                "label": "Version 34 + Leaflet Structure (c)",
                "desc": "Third iteration of the leaflet-structure update to v34.",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "advanced_membrane_analysis_Version35.py",
                "label": "Version 35 (baseline)",
                "desc": "Next-gen analyzer after v34, refined surface/grid handling.",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "advanced_membrane_analysis_Version35_UPDATED2.py",
                "label": "Version 35 — Updated 2",
                "desc": "v35 with additional refinements/bugfixes.",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "advanced_membrane_analysis_Version35_UPDATED3.py",
                "label": "Version 35 — Updated 3 (recommended, fastest)",
                "desc": ("Latest version: fixes an O(N^2) bottleneck in the water-"
                         "density calculation, adds numerical safety guards, and "
                         "exposes a real --tpr/--xtc/--lipids/--water/--grid-points/"
                         "--sigma/--output-dir CLI."),
                "patch": {"cli": True},
            },
            # Thickness / height map scripts added to this family
            {
                "file": "thickness-1_selected1.py",
                "label": "Thickness map (selected1)",
                "desc": "Average bilayer thickness map (grid-based).",
                "patch": {"tpr_var": "topology_file", "xtc_var": "trajectory_file"},
            },
            {
                "file": "thickness-1_selected11.py",
                "label": "Thickness map (selected11)",
                "desc": "Average bilayer thickness map (alternate variant).",
                "patch": {"tpr_var": "topology_file", "xtc_var": "trajectory_file"},
            },
            # Ultra-enhanced variant (new)
            {
                "file": "updated_ultra_membrane_analysis_script1.py",
                "label": "Ultra-enhanced analysis (unit-corrected)",
                "desc": "Comprehensive unit-corrected ultra analysis: density plots, 3D pore rendering, and summary CSV.",
                "patch": {"tpr_var": "topol_file", "xtc_var": "traj_file",
                          "lipid_var": "lipid_selection", "lipid_format": "space_with_resname",
                          "output_var": "output_prefix"},
            },
        ],
    },

    "Curvature & Pore Analysis": {
        "description": (
            "Computes per-leaflet curvature fields, detects membrane pores, "
            "tracks peptide proximity to pores over time, and renders "
            "heatmaps / 3D visuals (some variants also do full molecular "
            "rendering of the peptide, SCD order parameters, area-per-lipid, "
            "Voronoi packing defects, etc.)."
        ),
        "variants": [
            {
                "file": "curvature-5_TODAY.py",
                "label": "Curvature-5 (today build)",
                "desc": "Lightweight bilayer topology + deformation + 3D rendering.",
                "patch": {"tpr_var": "TPR_FILE", "xtc_var": "XTC_FILE",
                          "lipid_var": "LIPID_RESNAMES", "lipid_format": "space",
                          "peptide_var": "PEPTIDE_RESNAME"},
            },
            {
                "file": "curvature-5_new226aa.py",
                "label": "Curvature-5 226 (a) — with water molecules",
                "desc": ("Pore analysis with protein proximity, temporal curvature "
                         "mapping, pure-pore/top views, heatmaps, 3D visuals "
                         "INCLUDING water molecules."),
                "patch": {"tpr_var": "topology_file", "xtc_var": "trajectory_file",
                          "output_var": "output_dir"},
            },
            {
                "file": "curvature-5_new226b.py",
                "label": "Curvature-5 226 (b) — plot previews",
                "desc": ("Pore analysis with protein proximity, temporal curvature "
                         "mapping, and pore-distribution plots with previews "
                         "(no water rendering, lighter/faster than 226aa)."),
                "patch": {"tpr_var": "topology_file", "xtc_var": "trajectory_file",
                          "output_var": "output_dir"},
            },
            {
                "file": "curvature-combined-ultra-v10.py",
                "label": "Combined Ultra v10 (most comprehensive)",
                "desc": ("Everything: peptide rendering, backbone/COM network, "
                         "insertion depth, thickness/contour/3D maps, full-"
                         "trajectory curvature, pore detection, Gaussian/principal "
                         "curvature, tilt angle, SCD order, area-per-lipid, "
                         "Voronoi packing defects, roughness, radial density, "
                         "asymmetry/leaflet coupling, slices."),
                "patch": {"tpr_var": "TPR_FILE", "xtc_var": "XTC_FILE",
                          "lipid_var": "LIPID_RESNAMES", "lipid_format": "space",
                          "peptide_var": "PEPTIDE_RESNAME"},
            },
            {
                "file": "curvature-combined.py",
                "label": "Combined (baseline)",
                "desc": "Combined topology + full CG membrane analysis, baseline build.",
                "patch": {"tpr_var": "TPR_FILE", "xtc_var": "XTC_FILE",
                          "lipid_var": "LIPID_RESNAMES", "lipid_format": "space",
                          "peptide_var": "PEPTIDE_RESNAME"},
            },
            {
                "file": "curvature-combined1.py",
                "label": "Combined — rev 1",
                "desc": "Combined analysis, revision 1 (different default trajectory naming).",
                "patch": {"tpr_var": "TPR_FILE", "xtc_var": "XTC_FILE",
                          "lipid_var": "LIPID_RESNAMES", "lipid_format": "space",
                          "peptide_var": "PEPTIDE_RESNAME"},
            },
            {
                "file": "curvature-combined3.py",
                "label": "Combined — rev 3 (enhanced secondary structure)",
                "desc": ("Combined analysis with enhanced peptide visualization: "
                         "VDW atom spheres, smoothed backbone trace, alpha-"
                         "helix/beta-sheet/coil ribbons, residue COM network."),
                "patch": {"tpr_var": "TPR_FILE", "xtc_var": "XTC_FILE",
                          "lipid_var": "LIPID_RESNAMES", "lipid_format": "space",
                          "peptide_var": "PEPTIDE_RESNAME"},
            },
            # Curvature-frustration "Ultra" family (Marsh 2007 model + full
            # leaflet/APL/bending-modulus/pore/water-permeation suite)
            {
                "file": "advanced_membrane_analysis_Version3445a.py",
                "label": "Curvature-Frustration Ultra (3445a — baseline)",
                "desc": ("Full Marsh (2007) curvature-frustration model: leaflet "
                         "curvature/topology, spontaneous-curvature frustration "
                         "energy, pore + water-permeation single-frame analysis, "
                         "publication-quality static + interactive (Plotly) figures."),
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "advanced_membrane_analysis_Version3445b.py",
                "label": "Curvature-Frustration Ultra (3445b — accuracy fixes)",
                "desc": ("Adds PBC-unwrap, a fixed once-per-trajectory leaflet "
                         "assignment (no more midplane 'flicker'), anisotropic "
                         "pore-radius gridding, area-per-lipid per leaflet, a "
                         "Helfrich-spectrum bending-modulus estimate, and "
                         "vectorized/time-resolved pore radius + water permeation."),
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "advanced_membrane_analysis_Version3445c.py",
                "label": "Curvature-Frustration Ultra (3445c — + Voronoi APL, recommended)",
                "desc": ("Everything in 3445b plus a spatially-resolved area-per-"
                         "lipid map from 2D Voronoi tessellation of each leaflet, "
                         "with the peptide overlaid on the map."),
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            # Pore depth / free-energy map variant(s)
            {
                "file": "poredepth11.py",
                "label": "Pore free-energy maps (poredepth11)",
                "desc": "Water density → free-energy maps (XZ / YZ / XY) and 3D surface.",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPID_RESNAMES", "lipid_format": "list"},
            },
        ],
    },

    "Pore Analysis Suite (legacy & extended)": {
        "description": (
            "Standalone pore analysis scripts (legacy and extended). "
            "Includes pore radius profiles, filter/gate tracking, "
            "pore-lining peptide grids, and density maps."
        ),
        "variants": [
            {
                "file": "pore_others3.py",
                "label": "Pore Analysis — legacy (pore_others3)",
                "desc": "Legacy pore analysis suite: radius profiles, filter/gate, pore-lining grid, density maps.",
                "patch": {"tpr_var": "topol_file", "xtc_var": "traj_file",
                          "lipid_var": "lipid_selection", "lipid_format": "space_with_resname",
                          "output_var": "OUTDIR"},
            },
            {
                "file": "pore_others6.py",
                "label": "Pore Analysis — extended (pore_others6)",
                "desc": "Extended pore analysis: residue-level profiles, axial occupancy, density maps.",
                "patch": {"tpr_var": "topol_file", "xtc_var": "traj_file",
                          "lipid_var": "lipid_selection", "lipid_format": "space_with_resname",
                          "output_var": "OUTDIR"},
            },
        ],
    },

    "Peptide Depth Insertion": {
        "description": (
            "Tracks how deeply a peptide/ligand inserts into the bilayer "
            "relative to the lipid phosphate plane over the trajectory."
        ),
        "variants": [
            {
                "file": "depth_insertion_lig16d_com1.py",
                "label": "Depth Insertion (ligand COM)",
                "desc": "Peptide/ligand center-of-mass insertion depth vs. time.",
                "patch": {"tpr_var": "TPR_FILE", "xtc_var": "XTC_FILE",
                          "lipid_var": "LIPID_RESNAMES", "lipid_format": "space",
                          "peptide_var": "PEPTIDE_RESNAME"},
            },
            {
                "file": "depth_insertion_lig16d_com1_Version6.py",
                "label": "Depth Insertion — Residue-level (v6)",
                "desc": "Residue-level detailed insertion analysis (per-residue CSV + plots).",
                "patch": {"tpr_var": "TPR_FILE", "xtc_var": "XTC_FILE",
                          "lipid_var": "LIPID_RESNAMES", "lipid_format": "space",
                          "peptide_var": "PEPTIDE_RESNAME"},
            },
            {
                "file": "depth_insertion_lig16d_com1_Version7.py",
                "label": "Depth Insertion — Residue-level + 2D Snapshot (v7, recommended)",
                "desc": ("Everything in v6, plus auto water detection (SOL/TIP3/"
                         "TIP4P/etc.) and a 2D lipid/water/peptide system snapshot "
                         "at chosen frames, colored per-residue with N-/C-termini "
                         "highlighted."),
                "patch": {"tpr_var": "TPR_FILE", "xtc_var": "XTC_FILE",
                          "lipid_var": "LIPID_RESNAMES", "lipid_format": "space",
                          "peptide_var": "PEPTIDE_RESNAME"},
            },
            {
                "file": "depth_insertion_lig16d_com1_Version7_4.py",
                "label": "Depth Insertion — Cartoon Ribbon Snapshot (v7.4, recommended)",
                "desc": ("Everything in v7, but the 2D system snapshot now renders "
                         "the peptide as a real cartoon-style secondary-structure "
                         "ribbon (PyMOL/VMD-like): helical stretches drawn as a "
                         "twisted, blue/red-shaded ribbon and coil/loop as a "
                         "thinner AA-class-colored tube, with side-chain sticks "
                         "branching off the backbone, instead of plain atom dots."),
                "patch": {"tpr_var": "TPR_FILE", "xtc_var": "XTC_FILE",
                          "lipid_var": "LIPID_RESNAMES", "lipid_format": "space",
                          "peptide_var": "PEPTIDE_RESNAME"},
            },
            {
                "file": "finalized_depth_insertion-new2.py",
                "label": "Depth Insertion — Z-position + Density (finalized baseline)",
                "desc": ("Simpler baseline build: per-residue Z-position and "
                         "protein/water/phosphate density profiles along Z, plus "
                         "leaflet-relative insertion depth from a median-phosphate "
                         "leaflet split."),
                "patch": {"tpr_var": "topology_file", "xtc_var": "trajectory_file"},
            },
            {
                "file": "depth_pore_permeation_lig16d_com22B1.py",
                "label": "Depth + Pore Radius + Water Permeation + Free Energy",
                "desc": ("Single-pass analysis combining insertion depth, time-"
                         "resolved peptide-induced pore radius, three distinct "
                         "water-occupancy metrics, hysteresis-gated permeation "
                         "event counting, and Boltzmann-inversion free-energy "
                         "profiles G(r) and G(z). Has its own argparse CLI."),
                "patch": {"cli_v2": {
                    "tpr": "--tpr", "xtc": "--xtc",
                    "lipids": "--lipid_resnames", "peptide": "--peptide_resname",
                    "water": "--water_resnames", "output": "--out_prefix",
                }},
            },
            # The analyze_insertion* family of snapshot/pore/pmf scripts:
            {
                "file": "analyze_insertion3.py",
                "label": "analyze_insertion (variant 3)",
                "desc": "Snapshot + PMF summary + HOLE pore profile toolkit (variant 3).",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "analyze_insertion23.py",
                "label": "analyze_insertion2 (variant 23)",
                "desc": "Updated snapshot visuals + trisurf + viewer panel (variant 23).",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "analyze_insertion233.py",
                "label": "analyze_insertion (variant 233)",
                "desc": "Snapshot + 2D time series + PMF wrapper (variant 233).",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "analyze_insertion234.py",
                "label": "analyze_insertion (variant 234)",
                "desc": ("2D snapshot time-series + extended 3D viewer + HOLE pore "
                         "analysis; sampling interval override available."),
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "analyze_insertion233.py",
                "label": "analyze_insertion (variant 233 duplicate)",
                "desc": "Alternate copy of variant 233 (kept for provenance).",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
        ],
    },

    "MD Density Analysis": {
        "description": (
            "Computes 1D mass-density profiles along the membrane normal "
            "(z-axis) for lipids/protein/water, centered on the bilayer."
        ),
        "variants": [
            {
                "file": "md_density_analysis3.py",
                "label": "Density Profile (v3)",
                "desc": "Per-lipid + protein density profile along z, baseline build.",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "md_density_analysis32.py",
                "label": "Density Profile (v3.2)",
                "desc": "Refined version of the density-profile script (v3 -> v3.2).",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            {
                "file": "md_density_analysis1.py",
                "label": "Density Profile (v1)",
                "desc": "Early density-profile script (selection-string based).",
                # This script uses LIPIDS = "resname POPE POPG" — keep patching
                # such that the injected string matches that pattern.
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "space_with_resname"},
            },
            {
                "file": "md_density_analysis11.py",
                "label": "Density Profile (v1.1)",
                "desc": "Variant with Å units saved in CSV (v1.1).",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "space_with_resname"},
            },
            {
                "file": "md_density_analysis33.py",
                "label": "Density Profile (v3.3)",
                "desc": "Variant that outputs selected whole-lipid profiles (v3.3).",
                "patch": {"tpr_var": "TPR", "xtc_var": "XTC",
                          "lipid_var": "LIPIDS", "lipid_format": "list"},
            },
            # New addition: water_radius2 (non-protein water / radius analysis)
            {
                "file": "water_radius2.py",
                "label": "Non-protein Water + Radius Analysis",
                "desc": "Non-protein water distribution and solute radius along Z (shell-filtered).",
                "patch": {"tpr_var": "topol_file", "xtc_var": "traj_file",
                          "lipid_var": "", "lipid_format": "space", "output_var": "output_prefix",
                          "output_mode": "prefix",
                          "water_var": "water_selection", "water_format": "resname"},
            },
            # Enhanced publication-quality water/pore/permeation family
            {
                "file": "water_radius2kkk18.py",
                "label": "Enhanced Water/Pore + Permeation Analysis (v18)",
                "desc": ("Shell-based water density, 3D pore visualization, water "
                         "AND ion permeation counts, depth-of-permeation traces, "
                         "midplane occupancy, and a pore-radius/permeation "
                         "dashboard — full publication-figure suite."),
                "patch": {"tpr_var": "topol_file", "xtc_var": "traj_file",
                          "lipid_var": "", "lipid_format": "space", "output_var": "output_prefix",
                          "output_mode": "prefix",
                          "water_var": "water_selection", "water_format": "resname"},
            },
            {
                "file": "water_radius2kkk19.py",
                "label": "Enhanced Water/Pore + Permeation Analysis (v19, recommended)",
                "desc": ("Same as v18, plus peptide-insertion gating: pore/water-"
                         "permeation events are only attributed to the peptide on "
                         "frames where it is actually embedded in the bilayer "
                         "core, not just floating nearby in bulk solvent."),
                "patch": {"tpr_var": "topol_file", "xtc_var": "traj_file",
                          "lipid_var": "", "lipid_format": "space", "output_var": "output_prefix",
                          "output_mode": "prefix",
                          "water_var": "water_selection", "water_format": "resname"},
            },
            # Non-protein water + pore-radius family ("6bb")
            {
                "file": "water_radius6bbBBBB.py",
                "label": "Non-protein Water + Pore Radius (6bb — two-panel figures)",
                "desc": ("Non-protein water shell density and pore-radius z-"
                         "profiles, each rendered as a two-panel figure (whole-"
                         "system + one leaflet-split panel), with optional "
                         "leaflet-region shading."),
                "patch": {"tpr_var": "topol_file", "xtc_var": "traj_file",
                          "lipid_var": "", "lipid_format": "space", "output_var": "output_prefix",
                          "output_mode": "prefix",
                          "water_var": "water_selection", "water_format": "resname"},
            },
            {
                "file": "water_radius6bbBBBBB.py",
                "label": "Non-protein Water + Pore Radius (6bb — single-panel)",
                "desc": ("Simplified single-panel version of the 6bb water/pore-"
                         "radius z-profiles (whole-system only), plus a peptide-"
                         "inserted boolean flag tracked per frame."),
                "patch": {"tpr_var": "topol_file", "xtc_var": "traj_file",
                          "lipid_var": "", "lipid_format": "space", "output_var": "output_prefix",
                          "output_mode": "prefix",
                          "water_var": "water_selection", "water_format": "resname"},
            },
            {
                "file": "water_radius6bbBBBBBB.py",
                "label": "Non-protein Water + Pore Radius (6bb — refined, recommended)",
                "desc": ("Latest single-panel 6bb build: refined leaflet-boundary "
                         "shading/labels and cleaned-up plotting helpers."),
                "patch": {"tpr_var": "topol_file", "xtc_var": "traj_file",
                          "lipid_var": "", "lipid_format": "space", "output_var": "output_prefix",
                          "output_mode": "prefix",
                          "water_var": "water_selection", "water_format": "resname"},
            },
        ],
    },

    "Peptide Tilt & Rotation Analysis": {
        "description": (
            "Computes peptide helix tilt (τ) and azimuthal rotation (ρ) angles "
            "per chain/segment over the trajectory, with 2D density heatmaps, "
            "residue-level tilt/rotation profiles, and (in the newer variant) "
            "tilt-over-time trends and automatic MARTINI/CG backbone-atom "
            "fallback."
        ),
        "variants": [
            {
                "file": "tilt_rotation_analysis5.py",
                "label": "Tilt & Rotation (v5 — baseline)",
                "desc": ("Per-chain tilt/rotation density heatmap (with contours "
                         "and mean marker) plus residue-level tilt and rotation "
                         "line plots. Assumes an all-atom 'CA' reference atom."),
                "patch": {"tpr_var": "topology", "xtc_var": "trajectory"},
            },
            {
                "file": "tilt_rotation_analysis55.py",
                "label": "Tilt & Rotation (v55 — CG-aware, recommended)",
                "desc": ("Adds automatic fallback through CA/BB/CA1 reference-"
                         "atom names (so MARTINI coarse-grained 'BB' backbone "
                         "beads resolve correctly), a tilt-angle-over-time plot "
                         "with rolling average, clearer per-frame warnings, and "
                         "a friendlier error if the peptide selection matches 0 "
                         "atoms."),
                "patch": {"tpr_var": "topology", "xtc_var": "trajectory"},
            },
        ],
    },
}


# Common CG Martini / atomistic phospholipid (and sterol) residue names,
# offered in the GUI as one-click additions to the lipid field so people
# don't have to remember exact spelling/casing.
COMMON_LIPIDS = [
    "POPC", "POPE", "POPG", "POPS", "POPA",
    "DOPC", "DOPE", "DOPG", "DOPS",
    "DPPC", "DPPE", "DPPG",
    "DLPC", "DMPC",
    "CHOL",
]


# =============================================================================
# Script patching helpers
# =============================================================================

def _py_str(value: str) -> str:
    return repr(value)


def _var_pattern(varname: str) -> "re.Pattern":
    # Matches: VARNAME = "..."   or   VARNAME = [...]   (at the start of a
    # line, allowing leading whitespace/indentation). This intentionally
    # supports simple top-level literal assignments only.
    return re.compile(
        r'(?m)^(\s*' + re.escape(varname) + r'\s*=\s*)(?:".*?"|\'.*?\'|\[[^\]\n]*\])'
    )


def find_current_var_literal(src_text: str, varname: str):
    """Return the current right-hand-side literal text for `varname` as it
    appears (unpatched) in the script source, or None if not found. Used to
    show the person what the script's built-in default is before they run."""
    if not varname:
        return None
    m = _var_pattern(varname).search(src_text)
    if not m:
        return None
    line = m.group(0)
    return line.split("=", 1)[1].strip()


def format_lipid_literal(lipids: str, fmt: str):
    """Turn a free-text lipid field (space/comma separated) into the Python
    literal text used to patch the script, plus the cleaned name list.

    Supported formats:
      - "list"                -> Python list literal: ["POPE", "POPG"]
      - "space"               -> simple space-joined string: "POPE POPG"
      - "space_with_resname"  -> string prefixed with 'resname ': "resname POPE POPG"
      - "space_comma_cli"     -> comma-joined (for CLI --lipids style) "POPE,POPG"
    """
    names = [s.strip().upper() for s in lipids.replace(",", " ").split() if s.strip()]
    if fmt == "list":
        literal = "[" + ", ".join(_py_str(n) for n in names) + "]"
    elif fmt == "space_with_resname":
        # Many older scripts expect the literal to be exactly "resname POPE POPG"
        literal = _py_str("resname " + " ".join(names)) if names else _py_str("")
    else:  # "space" or default
        literal = _py_str(" ".join(names))
    return literal, names


# =============================================================================
# Water residue / system-type (all-atom vs coarse-grained) auto-detection
# =============================================================================
# Different force fields name the water residue differently, and the whole
# system's "resolution" (all-atom vs coarse-grained, e.g. MARTINI) changes
# which name is even possible. Rather than making the person look this up,
# the launcher reads the topology file directly and figures it out.

# All-atom water residue-name conventions (TIP3P/TIP4P/SPC/etc. force fields).
AA_WATER_CANDIDATES = ["TIP3", "TIP3P", "TIP4P", "TIP4", "TIP5P", "SOL",
                        "WAT", "HOH", "SPC", "SPCE", "T3P", "T4P", "OPC"]

# Coarse-grained water residue-name conventions (MARTINI 2/3, polarizable
# water, and similar CG mappings).
CG_WATER_CANDIDATES = ["W", "WN", "PW", "SW", "WF"]


def detect_water_and_system_type(tpr_path: str) -> dict:
    """Inspect a topology file and determine (a) whether the system looks
    all-atom or coarse-grained (e.g. MARTINI), and (b) which residue name
    holds the water in this specific file, so the launcher can pre-fill the
    'Water residue name' field automatically.

    Returns {"resname": str|None, "system_type": "all-atom"|"coarse-grained"|
    "unknown", "detail": str}. Never raises -- any failure is reported back
    in "detail" instead, so the GUI can show it rather than crash.
    """
    try:
        import MDAnalysis as mda
    except ImportError:
        return {"resname": None, "system_type": "unknown",
                "detail": "MDAnalysis is not available in the launcher's Python "
                          "environment, so auto-detection can't run here -- "
                          "enter the water residue name manually."}

    try:
        u = mda.Universe(tpr_path)
    except Exception as e:
        return {"resname": None, "system_type": "unknown",
                "detail": f"Could not read the topology for auto-detection: {e}"}

    try:
        resnames_upper = {r.upper(): r for r in set(u.residues.resnames)}
    except Exception as e:
        return {"resname": None, "system_type": "unknown",
                "detail": f"Could not read residue names for auto-detection: {e}"}

    cg_hits = [resnames_upper[c] for c in CG_WATER_CANDIDATES if c in resnames_upper]
    aa_hits = [resnames_upper[c] for c in AA_WATER_CANDIDATES if c in resnames_upper]

    # Overall system granularity, as a tie-breaker: all-atom systems run
    # well above ~3 atoms/residue even just from water; MARTINI-style CG
    # systems sit close to 1 atom (bead) per residue on average.
    try:
        avg_atoms_per_residue = u.atoms.n_atoms / max(1, u.residues.n_residues)
    except Exception:
        avg_atoms_per_residue = None
    likely_cg = avg_atoms_per_residue is not None and avg_atoms_per_residue < 2.5

    if cg_hits and (likely_cg or not aa_hits):
        resname = cg_hits[0]
        gran = f"~{avg_atoms_per_residue:.1f} atoms/residue" if avg_atoms_per_residue else "bead-level topology"
        return {"resname": resname, "system_type": "coarse-grained",
                "detail": f"Coarse-grained system detected ({gran}); water residue '{resname}'."}

    if aa_hits:
        # If several all-atom water-like names are present, prefer the one
        # covering the most residues (most likely the actual solvent).
        best = max(aa_hits, key=lambda r: len(u.select_atoms(f"resname {r}").residues))
        return {"resname": best, "system_type": "all-atom",
                "detail": f"All-atom system detected; water residue '{best}'."}

    if cg_hits:
        resname = cg_hits[0]
        return {"resname": resname, "system_type": "coarse-grained",
                "detail": f"Coarse-grained water residue '{resname}' detected."}

    return {"resname": None, "system_type": "unknown",
            "detail": "No recognized water residue name found in this topology "
                      "-- enter it manually (common: SOL/TIP3 for all-atom, "
                      "W/WN for MARTINI coarse-grained)."}


def patch_script_source(src_text: str, patch: dict, tpr_path: str, xtc_path: str,
                         lipids: str, peptide: str, output_dir: str, water: str = ""):
    """Return (patched_text, report) where report is a list of human-readable
    strings describing exactly which variables were changed and to what —
    so the GUI can show the person proof that their phospholipid names (and
    everything else) actually made it into the script that will run.

    Raises RuntimeError with a specific, actionable message if a value the
    person supplied (lipids, peptide, tpr, xtc) could not be located and
    patched in the target script — this is deliberate: silently keeping the
    script's original hard-coded lipids while pretending the person's input
    was applied is worse than failing loudly.
    """
    text = src_text
    report = []

    def replace_var(text, varname, new_literal):
        new_text, n = _var_pattern(varname).subn(
            lambda m: m.group(1) + new_literal, text, count=1)
        return new_text, n

    if patch.get("tpr_var"):
        text, n = replace_var(text, patch["tpr_var"], _py_str(tpr_path))
        if n == 0:
            raise RuntimeError(
                f"Could not find variable '{patch['tpr_var']}' to patch (topology file). "
                "The script's source may have changed — check that this variable still "
                "exists as a plain top-level assignment.")
        report.append(f"✓ {patch['tpr_var']} → {tpr_path}")

    if patch.get("xtc_var"):
        text, n = replace_var(text, patch["xtc_var"], _py_str(xtc_path))
        if n == 0:
            raise RuntimeError(
                f"Could not find variable '{patch['xtc_var']}' to patch (trajectory file). "
                "The script's source may have changed — check that this variable still "
                "exists as a plain top-level assignment.")
        report.append(f"✓ {patch['xtc_var']} → {xtc_path}")

    if patch.get("lipid_var"):
        if lipids.strip():
            fmt = patch.get("lipid_format", "space")
            literal, names = format_lipid_literal(lipids, fmt)
            text, n = replace_var(text, patch["lipid_var"], literal)
            if n == 0:
                raise RuntimeError(
                    f"Could not find variable '{patch['lipid_var']}' to patch (lipid "
                    f"residue names). Your phospholipid names ({', '.join(names)}) were "
                    "NOT applied — the script would have run with its original hard-coded "
                    "lipids. Check that this variable still exists as a plain top-level "
                    "assignment in the script.")
            report.append(f"✓ {patch['lipid_var']} → {literal}   (your phospholipid selection was applied)")
        else:
            current = find_current_var_literal(src_text, patch["lipid_var"])
            report.append(
                f"⚠ No lipid names entered — script default kept for "
                f"{patch['lipid_var']}"
                + (f" = {current}" if current else "") + ".")

    if patch.get("peptide_var"):
        if peptide.strip():
            text, n = replace_var(text, patch["peptide_var"], _py_str(peptide.strip()))
            if n == 0:
                raise RuntimeError(
                    f"Could not find variable '{patch['peptide_var']}' to patch "
                    "(peptide/protein residue name). Check that this variable still "
                    "exists as a plain top-level assignment in the script.")
            report.append(f"✓ {patch['peptide_var']} → {_py_str(peptide.strip())}")
        else:
            current = find_current_var_literal(src_text, patch["peptide_var"])
            report.append(
                f"⚠ No peptide residue entered — script default kept for "
                f"{patch['peptide_var']}" + (f" = {current}" if current else "") + ".")

    if patch.get("water_var"):
        if water.strip():
            wname = water.strip().split()[0]  # a single resname is expected here
            fmt = patch.get("water_format", "plain")
            literal = _py_str("resname " + wname) if fmt == "resname" else _py_str(wname)
            text, n = replace_var(text, patch["water_var"], literal)
            if n == 0:
                raise RuntimeError(
                    f"Could not find variable '{patch['water_var']}' to patch (water "
                    "residue name). Check that this variable still exists as a plain "
                    "top-level assignment in the script.")
            report.append(f"✓ {patch['water_var']} → {literal}   (auto-detected/entered water residue applied)")
        else:
            current = find_current_var_literal(src_text, patch["water_var"])
            report.append(
                f"⚠ No water residue entered — script default kept for "
                f"{patch['water_var']}" + (f" = {current}" if current else "") + ".")

    if patch.get("output_var") and output_dir:
        mode = patch.get("output_mode", "dir")
        if mode == "prefix":
            # This variable is concatenated directly into filenames/folder
            # names by the script itself (e.g. f"{output_prefix}_waters.csv"
            # or f"{output_prefix}_outputs"), not passed through
            # os.path.join(). Patching it with "." (used for real dir-style
            # variables below) would silently produce hidden dot-files/dot-
            # folders here instead, so use a plain, visible basename.
            literal_value = "analysis_output"
        else:
            literal_value = output_dir
        text, n = replace_var(text, patch["output_var"], _py_str(literal_value))
        if n > 0:
            report.append(f"✓ {patch['output_var']} → {literal_value}")
        else:
            report.append(
                f"⚠ Could not find variable '{patch['output_var']}' to set the output "
                "location -- outputs may land in an unexpected place.")

    return text, report


def build_run_command(variant: dict, tpr_path: str, xtc_path: str,
                       lipids: str, peptide: str, water: str,
                       output_dir: str, work_dir: str):
    """Prepares the script to run (patched copy or CLI invocation) and
    returns (cmd, run_cwd, report) where report is a list of human-readable
    strings confirming exactly what was applied (including phospholipid
    residue names) so the caller can show the person proof before/while
    the run happens."""

    src_path = os.path.join(SCRIPTS_DIR, variant["file"])
    patch = variant.get("patch", {})

    os.makedirs(output_dir, exist_ok=True)

    if patch.get("cli_v2"):
        # Script exposes its own argparse CLI with non-standard flag names
        # (differs per script) -- use the flag map given in this variant's
        # patch dict directly, no source patching needed.
        flags = patch["cli_v2"]
        cmd = [sys.executable, src_path]
        report = []

        cmd += [flags["tpr"], tpr_path]
        report.append(f"✓ {flags['tpr']} {tpr_path}")
        cmd += [flags["xtc"], xtc_path]
        report.append(f"✓ {flags['xtc']} {xtc_path}")

        if flags.get("output"):
            cmd += [flags["output"], output_dir]
            report.append(f"✓ {flags['output']} {output_dir}")

        if flags.get("lipids"):
            if lipids.strip():
                _, names = format_lipid_literal(lipids, "space")
                joined = " ".join(names)
                cmd += [flags["lipids"], joined]
                report.append(f"✓ {flags['lipids']} {joined}   (your phospholipid selection was applied)")
            else:
                report.append(f"⚠ No lipid names entered — script's built-in {flags['lipids']} default will be used.")

        if flags.get("peptide"):
            if peptide.strip():
                cmd += [flags["peptide"], peptide.strip()]
                report.append(f"✓ {flags['peptide']} {peptide.strip()}")
            else:
                report.append(f"⚠ No peptide residue entered — script's built-in {flags['peptide']} default will be used.")

        if flags.get("water") and water.strip():
            cmd += [flags["water"], water.strip()]
            report.append(f"✓ {flags['water']} {water.strip()}")

        return cmd, output_dir, report

    if patch.get("cli"):
        # Script already exposes a real argparse CLI -- use it directly,
        # no source patching needed.
        cmd = [
            sys.executable, src_path, "--nogui",
            "--tpr", tpr_path,
            "--xtc", xtc_path,
            "--output-dir", output_dir,
        ]
        report = [
            f"✓ --tpr {tpr_path}",
            f"✓ --xtc {xtc_path}",
            f"✓ --output-dir {output_dir}",
        ]
        if lipids.strip():
            _, names = format_lipid_literal(lipids, "space")
            cli_lipids = ",".join(names)
            cmd += ["--lipids", cli_lipids]
            report.append(f"✓ --lipids {cli_lipids}   (your phospholipid selection was applied)")
        else:
            report.append("⚠ No lipid names entered — script's built-in --lipids default will be used.")
        if water.strip():
            cmd += ["--water", water.strip()]
            report.append(f"✓ --water {water.strip()}")
        return cmd, output_dir, report

    # Otherwise: patch a temp copy of the script with the chosen values,
    # then run that copy with its working directory set to the output
    # folder (since most of these scripts write outputs to relative paths
    # like "output/..." or the cwd directly).
    with open(src_path, "r", encoding="utf-8") as f:
        src_text = f.read()

    patched_text, report = patch_script_source(
        src_text, patch, tpr_path, xtc_path, lipids, peptide, output_dir=".", water=water
    )

    os.makedirs(work_dir, exist_ok=True)
    patched_path = os.path.join(work_dir, "_run_" + variant["file"])
    with open(patched_path, "w", encoding="utf-8") as f:
        f.write(patched_text)

    return [sys.executable, patched_path], output_dir, report
# =============================================================================
# THEME / STYLE
# =============================================================================
# A single, consistent professional palette used throughout the app. Keeping
# it centralized here makes it trivial to re-skin the whole interface later.

class Theme:
    # Single dark "ink" family for header/sidebar/status bar so they read as
    # one continuous surface instead of three slightly different blacks —
    # HEADER is the darkest plane (top banner), SIDEBAR one step lighter,
    # SIDEBAR_HOVER/BORDER_DARK step up again. All four share the same hue
    # so nothing clashes when they sit edge to edge.
    BG            = "#ffffff"   # main content background
    PANEL         = "#ffffff"   # cards / panels
    PARAM_BG      = "#eef3fd"   # Input Parameters card — soft blue tint, so it
                                 # stands out from the plain white cards around it
    PARAM_BORDER  = "#c9dbf7"   # matching border for the Input Parameters card
    HEADER        = "#0a0f1c"   # top banner (darkest plane)
    SIDEBAR       = "#0f1626"   # left navigation rail (one step up from HEADER)
    SIDEBAR_HOVER = "#1a2338"   # hover state (two steps up)
    BORDER_DARK   = "#232e48"   # borders/dividers on dark surfaces
    BORDER        = "#e3e7ee"   # borders/dividers on light surfaces
    TEXT          = "#1a2233"
    TEXT_MUTED    = "#5b6878"
    TEXT_ON_DARK  = "#eef1f6"
    TEXT_ON_DARK_MUTED = "#93a1ba"
    ACCENT        = "#2563eb"   # primary action blue
    ACCENT_DARK   = "#1d4ed8"
    ACCENT_SOFT   = "#e8effd"
    ACCENT_SOFT_DARK = "#182444"  # accent tint that reads correctly on dark surfaces
    SUCCESS       = "#16a34a"
    DANGER        = "#dc2626"
    DANGER_DARK   = "#b91c1c"
    CONSOLE_BG    = "#0a0f1c"
    CONSOLE_FG    = "#d7e1f0"
    CONSOLE_MUTED = "#5d6b85"

    FONT_FAMILY   = "Segoe UI" if sys.platform.startswith("win") else \
                     ("Helvetica Neue" if sys.platform == "darwin" else "DejaVu Sans")
    MONO_FAMILY   = "Consolas" if sys.platform.startswith("win") else \
                     ("Menlo" if sys.platform == "darwin" else "DejaVu Sans Mono")


APP_NAME = "NICS LAB PEPTIDE-MEMBRANE ANALYSIS TOOL"
APP_TAGLINE = "A unified workbench for membrane, curvature & pore analysis from MD trajectories"
APP_VERSION = "2.0.0"
LICENSE_NAME = "MIT License"
ASSET_GIF = os.path.join(BASE_DIR, "porous.gif")
ASSET_GIF2 = os.path.join(BASE_DIR, "PGLa-folding.gif")
ASSET_LOGO = os.path.join(BASE_DIR, "nics_lab_logo.png")
ASSET_HEADER_IMG1 = os.path.join(BASE_DIR, "header_image1.png")
ASSET_HEADER_IMG2 = os.path.join(BASE_DIR, "header_image2.png")

# Every thumbnail in the header media strip (and the About dialog) is fit to
# ONE shared box, so a native-portrait GIF, a native-landscape PNG and the
# square logo all end up the same on-screen size — "same size" means the
# same bounding box, not the same file, so aspect ratio is still preserved
# inside it. Defined once here so header/about can never drift apart again.
HEADER_MEDIA_BOX = (148, 148)
ABOUT_MEDIA_BOX = (168, 168)


def configure_styles(root: tk.Tk) -> ttk.Style:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    base_font = (Theme.FONT_FAMILY, 10)
    root.option_add("*Font", base_font)

    style.configure(".", background=Theme.BG, foreground=Theme.TEXT, font=base_font)

    # Frames
    style.configure("App.TFrame", background=Theme.BG)
    style.configure("Card.TFrame", background=Theme.PANEL)
    style.configure("Header.TFrame", background=Theme.HEADER)
    style.configure("Sidebar.TFrame", background=Theme.SIDEBAR)
    style.configure("StatusBar.TFrame", background=Theme.SIDEBAR)

    # Labels
    style.configure("App.TLabel", background=Theme.BG, foreground=Theme.TEXT)
    style.configure("Card.TLabel", background=Theme.PANEL, foreground=Theme.TEXT)
    style.configure("CardMuted.TLabel", background=Theme.PANEL, foreground=Theme.TEXT_MUTED)
    style.configure("CardTitle.TLabel", background=Theme.PANEL, foreground=Theme.TEXT,
                     font=(Theme.FONT_FAMILY, 14, "bold"))

    # Input Parameters card — same shapes as the Card.* styles above, just on
    # the PARAM_BG tint so this one section reads as distinct from the plain
    # white cards around it.
    style.configure("ParamCard.TFrame", background=Theme.PARAM_BG)
    style.configure("ParamCard.TLabel", background=Theme.PARAM_BG, foreground=Theme.TEXT)
    style.configure("ParamCardMuted.TLabel", background=Theme.PARAM_BG, foreground=Theme.TEXT_MUTED)
    style.configure("ParamCardTitle.TLabel", background=Theme.PARAM_BG, foreground=Theme.TEXT,
                     font=(Theme.FONT_FAMILY, 14, "bold"))
    style.configure("SectionTitle.TLabel", background=Theme.BG, foreground=Theme.TEXT,
                     font=(Theme.FONT_FAMILY, 11, "bold"))
    style.configure("HeaderTitle.TLabel", background=Theme.HEADER, foreground=Theme.TEXT_ON_DARK,
                     font=(Theme.FONT_FAMILY, 19, "bold"))
    style.configure("HeaderTagline.TLabel", background=Theme.HEADER, foreground=Theme.TEXT_ON_DARK_MUTED,
                     font=(Theme.FONT_FAMILY, 10))
    style.configure("HeaderBadge.TLabel", background=Theme.HEADER, foreground=Theme.ACCENT_SOFT,
                     font=(Theme.MONO_FAMILY, 9, "bold"))
    style.configure("Sidebar.TLabel", background=Theme.SIDEBAR, foreground=Theme.TEXT_ON_DARK,
                     font=(Theme.FONT_FAMILY, 11, "bold"))
    style.configure("SidebarMuted.TLabel", background=Theme.SIDEBAR, foreground=Theme.TEXT_ON_DARK_MUTED,
                     font=(Theme.FONT_FAMILY, 8))
    style.configure("Status.TLabel", background=Theme.SIDEBAR, foreground=Theme.TEXT_ON_DARK_MUTED,
                     font=(Theme.FONT_FAMILY, 9))
    style.configure("StatusValue.TLabel", background=Theme.SIDEBAR, foreground=Theme.TEXT_ON_DARK,
                     font=(Theme.FONT_FAMILY, 9, "bold"))

    # Buttons
    style.configure("TButton", padding=(10, 6), font=(Theme.FONT_FAMILY, 9))

    style.configure("Accent.TButton", padding=(14, 8), font=(Theme.FONT_FAMILY, 10, "bold"),
                     background=Theme.ACCENT, foreground="white", borderwidth=0)
    style.map("Accent.TButton",
              background=[("disabled", "#a9bdee"), ("active", Theme.ACCENT_DARK)],
              foreground=[("disabled", "#eef2fb")])

    style.configure("Danger.TButton", padding=(14, 8), font=(Theme.FONT_FAMILY, 10, "bold"),
                     background=Theme.DANGER, foreground="white", borderwidth=0)
    style.map("Danger.TButton",
              background=[("disabled", "#eeb3b0"), ("active", Theme.DANGER_DARK)],
              foreground=[("disabled", "#fbecec")])

    style.configure("Ghost.TButton", padding=(12, 7), font=(Theme.FONT_FAMILY, 9),
                     background=Theme.PANEL, foreground=Theme.TEXT, borderwidth=1)
    style.map("Ghost.TButton", background=[("active", Theme.ACCENT_SOFT)])

    style.configure("Variant.TButton", padding=(8, 6), font=(Theme.FONT_FAMILY, 9),
                     background=Theme.SIDEBAR, foreground=Theme.TEXT_ON_DARK_MUTED,
                     borderwidth=0, anchor="w")
    style.map("Variant.TButton",
              background=[("active", Theme.SIDEBAR_HOVER)],
              foreground=[("active", Theme.TEXT_ON_DARK)])

    style.configure("VariantMissing.TButton", padding=(8, 6), font=(Theme.FONT_FAMILY, 9),
                     background=Theme.SIDEBAR, foreground="#c47a4a",
                     borderwidth=0, anchor="w")
    style.map("VariantMissing.TButton",
              background=[("active", Theme.SIDEBAR_HOVER)],
              foreground=[("active", "#e0a074")])

    style.configure("VariantSelected.TButton", padding=(8, 6), font=(Theme.FONT_FAMILY, 9, "bold"),
                     background=Theme.ACCENT, foreground="white", borderwidth=0, anchor="w")
    style.map("VariantSelected.TButton", background=[("active", Theme.ACCENT_DARK)])

    style.configure("Family.TButton", padding=(8, 8), font=(Theme.FONT_FAMILY, 10, "bold"),
                     background=Theme.SIDEBAR, foreground=Theme.TEXT_ON_DARK, borderwidth=0, anchor="w")
    style.map("Family.TButton", background=[("active", Theme.SIDEBAR_HOVER)])

    # Entries
    style.configure("TEntry", padding=6, fieldbackground="white", borderwidth=1)
    style.configure("TCombobox", padding=6)

    # Labelframe (legacy, unused but kept consistent if referenced elsewhere)
    style.configure("Card.TLabelframe", background=Theme.PANEL, bordercolor=Theme.BORDER)
    style.configure("Card.TLabelframe.Label", background=Theme.PANEL, foreground=Theme.TEXT,
                     font=(Theme.FONT_FAMILY, 10, "bold"))

    # Progressbar
    style.configure("Accent.Horizontal.TProgressbar", troughcolor=Theme.BORDER,
                     background=Theme.ACCENT, bordercolor=Theme.BORDER, lightcolor=Theme.ACCENT,
                     darkcolor=Theme.ACCENT)

    # Scrollbar
    style.configure("TScrollbar", background=Theme.PANEL, troughcolor=Theme.BG, borderwidth=0)

    # Separator
    style.configure("TSeparator", background=Theme.BORDER)

    return style


def card(parent, style="Card.TFrame", border=None, **kwargs):
    """A flat 'card' container with a thin border, matching the rest of the
    professional theme (ttk.Frame alone can't draw a clean border in the
    'clam' theme, so we use a plain tk.Frame as the border + ttk inside).
    Pass style="ParamCard.TFrame" (and border=Theme.PARAM_BORDER) to get the
    tinted Input Parameters variant instead of the plain white default."""
    outer = tk.Frame(parent, background=border or Theme.BORDER, **kwargs)
    inner = ttk.Frame(outer, style=style, padding=16)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    return outer, inner


# =============================================================================
# Image scaling helper (stdlib-only; uses Tk's native zoom()/subsample() so
# no Pillow dependency is required for the launcher itself)
# =============================================================================

def _fit_photo_to_box(photo, max_w, max_h):
    """Scale a tk.PhotoImage to fit within (max_w, max_h) while preserving
    its aspect ratio, using an integer zoom()/subsample() pair to approximate
    the needed scale factor (Tk has no arbitrary-ratio resize built in).
    Never upscales. Two images given the SAME (max_w, max_h) box always end
    up the same on-screen size along their limiting dimension, regardless of
    how different their native resolutions are -- that's what makes gifs or
    photos of different native sizes appear uniformly sized side by side."""
    w, h = photo.width(), photo.height()
    if not max_w and not max_h:
        return photo
    if w <= 0 or h <= 0:
        return photo
    limits = [r for r in ((max_w / w) if max_w else None,
                          (max_h / h) if max_h else None) if r is not None]
    scale = min(limits + [1.0])  # never upscale
    if scale >= 0.999:
        return photo
    frac = Fraction(scale).limit_denominator(64)
    num, den = max(1, frac.numerator), max(1, frac.denominator)
    if num > 1:
        photo = photo.zoom(num, num)
    if den > 1:
        photo = photo.subsample(den, den)
    return photo


# =============================================================================
# Animated GIF widget (stdlib-only; uses Tk's native multi-frame GIF support
# so no Pillow dependency is required for the launcher itself)
# =============================================================================

class AnimatedGifLabel(tk.Label):
    def __init__(self, master, path, max_width=None, max_height=None, background=None, **kwargs):
        self._bg = background or Theme.HEADER
        super().__init__(master, background=self._bg, borderwidth=0, highlightthickness=0, **kwargs)
        self.frames = []
        self.delay_ms = 90
        self._job = None
        self._index = 0
        self._load(path, max_width, max_height)
        if self.frames:
            self.configure(image=self.frames[0])
            self._index = 0
            self._schedule_next()
        else:
            self.configure(text="(preview unavailable)", fg=Theme.TEXT_ON_DARK_MUTED,
                            background=self._bg, font=(Theme.FONT_FAMILY, 9, "italic"))

    def _load(self, path, max_width, max_height):
        if not path or not os.path.isfile(path):
            return
        frames = []
        i = 0
        try:
            while True:
                frame = tk.PhotoImage(master=self, file=path, format=f"gif -index {i}")
                frame = _fit_photo_to_box(frame, max_width, max_height)
                frames.append(frame)
                i += 1
        except tk.TclError:
            pass  # ran out of frames
        self.frames = frames

    def _schedule_next(self):
        if len(self.frames) <= 1:
            return
        self._job = self.after(self.delay_ms, self._advance)

    def _advance(self):
        self._index = (self._index + 1) % len(self.frames)
        self.configure(image=self.frames[self._index])
        self._schedule_next()

    def destroy(self):
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except Exception:
                pass
        super().destroy()


class StaticImageLabel(tk.Label):
    """Simple non-animated image label (PNG/GIF/etc. via Tk's native
    PhotoImage -- Tk 8.6+ reads PNG natively, no Pillow needed), fit within
    an optional (max_width, max_height) box with aspect ratio preserved.
    Degrades to blank/placeholder if the file is missing or this Tk build
    can't decode the format."""

    def __init__(self, master, path, max_width=None, max_height=None, background=None, **kwargs):
        self._bg = background or Theme.HEADER
        super().__init__(master, background=self._bg, borderwidth=0, highlightthickness=0, **kwargs)
        self.image_ref = None
        if path and os.path.isfile(path):
            try:
                img = tk.PhotoImage(master=self, file=path)
                img = _fit_photo_to_box(img, max_width, max_height)
                self.image_ref = img  # keep a reference so it isn't garbage-collected
                self.configure(image=img)
            except tk.TclError:
                self.configure(text="(image unavailable)", fg=Theme.TEXT_ON_DARK_MUTED,
                                background=self._bg, font=(Theme.FONT_FAMILY, 8, "italic"))
        else:
            self.configure(background=self._bg)


# =============================================================================
# Media strip helper — a uniformly sized, bordered "thumbnail card" used for
# every logo/reference-image/gif in the header and About dialog. Built in
# two steps: the card (frame + border + caption) is created immediately —
# that's just Tk frames/labels and is essentially free — while the actual
# image/GIF decoding happens later via populate(). Splitting it this way is
# what lets the launcher paint its window right away instead of blocking on
# every GIF's frame-by-frame decode before the first pixel appears.
# =============================================================================

class MediaThumb(tk.Frame):
    """A bordered, captioned thumbnail card with a fixed box and lazy content."""

    def __init__(self, master, caption, box, background=None, dark=True):
        self._bg = background or (Theme.HEADER if dark else Theme.PANEL)
        border = Theme.BORDER_DARK if dark else Theme.BORDER
        super().__init__(master, background=border, highlightthickness=0, bd=0)
        self._box = box
        muted = Theme.TEXT_ON_DARK_MUTED if dark else Theme.TEXT_MUTED

        inner = tk.Frame(self, background=self._bg, width=box[0], height=box[1])
        inner.pack(padx=1, pady=(1, 0))
        inner.pack_propagate(False)
        self._inner = inner

        self._placeholder = tk.Label(inner, text="···", background=self._bg, fg=muted,
                                      font=(Theme.FONT_FAMILY, 11))
        self._placeholder.place(relx=0.5, rely=0.5, anchor="center")

        if caption:
            tk.Label(self, text=caption, background=border, fg=muted,
                     font=(Theme.FONT_FAMILY, 8, "bold")).pack(pady=(4, 3), padx=6)

    def populate(self, kind, path):
        """kind: 'image' or 'gif'. Decodes the asset and swaps it into the card."""
        if self._placeholder is not None:
            self._placeholder.destroy()
            self._placeholder = None
        max_w, max_h = self._box
        if kind == "gif":
            widget = AnimatedGifLabel(self._inner, path, max_width=max_w, max_height=max_h,
                                       background=self._bg)
        else:
            widget = StaticImageLabel(self._inner, path, max_width=max_w, max_height=max_h,
                                       background=self._bg)
        widget.place(relx=0.5, rely=0.5, anchor="center")
        return widget


# =============================================================================
# GUI — Run panel (parameters, execution controls, live console)
# =============================================================================

class RunPanel(ttk.Frame):
    """Right-hand content panel: selected-analysis summary, input parameters,
    run controls and a live execution console."""

    def __init__(self, master, status_callback=None):
        super().__init__(master, style="App.TFrame", padding=(18, 16, 18, 16))
        self.process = None
        self.log_queue = queue.Queue()
        self.current_variant = None
        self.current_family = None
        self.status_callback = status_callback or (lambda *a, **k: None)

        # --- Selected analysis summary card -------------------------------
        sel_outer, sel = card(self)
        sel_outer.pack(fill="x", pady=(0, 12))

        head_row = ttk.Frame(sel, style="Card.TFrame")
        head_row.pack(fill="x")
        self.title_lbl = ttk.Label(head_row, text="Select an analysis to begin",
                                    style="CardTitle.TLabel")
        self.title_lbl.pack(side="left", anchor="w")
        self.file_badge = ttk.Label(head_row, text="", style="CardMuted.TLabel",
                                     font=(Theme.MONO_FAMILY, 8))
        self.file_badge.pack(side="right", anchor="e")

        self.desc_lbl = ttk.Label(sel, text=(
            "Choose an analysis family from the left, then pick a specific script "
            "variant. Required input fields will appear below."),
            style="CardMuted.TLabel", wraplength=760, justify="left")
        self.desc_lbl.pack(anchor="w", pady=(4, 0))

        self.missing_banner_lbl = ttk.Label(
            sel, text="", style="CardMuted.TLabel", wraplength=760, justify="left",
            foreground=Theme.DANGER, font=(Theme.FONT_FAMILY, 9, "bold"))
        # Not packed initially -- _set_missing_banner() packs/unpacks it.

        # --- Parameters card ---------------------------------------------
        # Distinct tinted card (see Theme.PARAM_BG) so this section — the one
        # the user actually has to fill in — stands out from the plain white
        # summary/console cards around it.
        param_outer, param = card(self, style="ParamCard.TFrame", border=Theme.PARAM_BORDER)
        param_outer.pack(fill="x", pady=(0, 12))

        ttk.Label(param, text="Input Parameters", style="ParamCardTitle.TLabel",
                  font=(Theme.FONT_FAMILY, 11, "bold")).pack(anchor="w", pady=(0, 10))

        form = ttk.Frame(param, style="ParamCard.TFrame")
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        self.tpr_var = tk.StringVar()
        self.xtc_var = tk.StringVar()
        self.lipid_var = tk.StringVar(value="POPE POPG")
        self.peptide_var = tk.StringVar(value="UNK")
        self.ligand_var = tk.StringVar()
        self.water_var = tk.StringVar(value="WAT")
        self.outdir_var = tk.StringVar(value=os.path.join(BASE_DIR, "analysis_output"))
        self.detect_queue = queue.Queue()

        self._row(form, 0, "Topology file", "(.tpr / .gro / .pdb / .psf)", self.tpr_var, self._browse_tpr)
        self._row(form, 1, "Trajectory file", "(.xtc / .trr / .dcd)", self.xtc_var, self._browse_xtc)
        self.lipid_entry = self._row(form, 2, "Lipid residue names", "(space or comma separated)", self.lipid_var)
        self.peptide_row = self._row(form, 3, "Peptide / protein residue", "", self.peptide_var)
        self.ligand_row = self._row(
            form, 4, "Ligand / small-molecule residue",
            "(use this instead of Peptide/Protein for ligand-only systems — "
            "e.g. a drug or small molecule embedded in the membrane, no protein present)",
            self.ligand_var)
        self.water_row = self._row(
            form, 5, "Water residue name",
            "(auto-detected from the topology once loaded — edit if needed)",
            self.water_var, self._autodetect_water, button_text="Detect")
        self._row(form, 6, "Output folder", "", self.outdir_var, self._browse_outdir)

        self.water_status_lbl = ttk.Label(
            param, text="Load a topology file to auto-detect the water residue "
                         "and whether the system is all-atom or coarse-grained.",
            style="ParamCardMuted.TLabel", font=(Theme.FONT_FAMILY, 8, "italic"), wraplength=760)
        self.water_status_lbl.pack(anchor="w", pady=(2, 0))

        # -- Phospholipid helper strip: shows what the *script* currently
        # hard-codes for this variant, plus one-click chips to add common
        # lipid/sterol residue names to the field above without retyping.
        lipid_help = ttk.Frame(param, style="ParamCard.TFrame")
        lipid_help.pack(fill="x", pady=(2, 0))

        self.lipid_default_lbl = ttk.Label(
            lipid_help, text="", style="ParamCardMuted.TLabel", font=(Theme.FONT_FAMILY, 8, "italic"))
        self.lipid_default_lbl.pack(anchor="w")

        chip_row = ttk.Frame(lipid_help, style="ParamCard.TFrame")
        chip_row.pack(anchor="w", pady=(4, 0))
        ttk.Label(chip_row, text="Quick add:", style="ParamCardMuted.TLabel",
                  font=(Theme.FONT_FAMILY, 8)).pack(side="left", padx=(0, 6))
        for name in COMMON_LIPIDS:
            ttk.Button(chip_row, text=name, style="Ghost.TButton",
                       command=lambda n=name: self._add_lipid(n)).pack(side="left", padx=2, pady=2)
        ttk.Button(chip_row, text="Clear", style="Ghost.TButton",
                   command=lambda: self.lipid_var.set("")).pack(side="left", padx=(10, 0))

        # --- Actions -----------------------------------------------------
        btn_row = ttk.Frame(self, style="App.TFrame")
        btn_row.pack(fill="x", pady=(2, 10))

        self.run_btn = ttk.Button(btn_row, text="▶  Run Analysis", style="Accent.TButton",
                                   command=self.run)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(btn_row, text="■  Stop", style="Danger.TButton",
                                    command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))
        self.open_out_btn = ttk.Button(btn_row, text="Open Output Folder", style="Ghost.TButton",
                                        command=self._open_output)
        self.open_out_btn.pack(side="left", padx=(8, 0))
        self.clear_log_btn = ttk.Button(btn_row, text="Clear Log", style="Ghost.TButton",
                                         command=lambda: self.log_text.delete("1.0", "end"))
        self.clear_log_btn.pack(side="left", padx=(8, 0))

        self.progress = ttk.Progressbar(btn_row, mode="indeterminate", length=160,
                                         style="Accent.Horizontal.TProgressbar")
        self.progress.pack(side="right")

        # --- Console -------------------------------------------------------
        console_outer, console = card(self)
        console_outer.pack(fill="both", expand=True)

        crow = ttk.Frame(console, style="Card.TFrame")
        crow.pack(fill="x", pady=(0, 8))
        ttk.Label(crow, text="Execution Log", style="CardTitle.TLabel",
                  font=(Theme.FONT_FAMILY, 11, "bold")).pack(side="left")
        self.run_state_lbl = ttk.Label(crow, text="● Idle", style="CardMuted.TLabel",
                                        font=(Theme.FONT_FAMILY, 9, "bold"))
        self.run_state_lbl.pack(side="right")

        log_frame = tk.Frame(console, background=Theme.CONSOLE_BG, highlightthickness=1,
                              highlightbackground=Theme.BORDER)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=16, bg=Theme.CONSOLE_BG, fg=Theme.CONSOLE_FG,
                                 insertbackground=Theme.CONSOLE_FG, relief="flat", borderwidth=0,
                                 padx=10, pady=8, font=(Theme.MONO_FAMILY, 9))
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scroll.set)
        self.log_text.tag_configure("dim", foreground=Theme.CONSOLE_MUTED)
        self.log_text.tag_configure("ok", foreground="#7ee787")
        self.log_text.tag_configure("warn", foreground="#f2c14e")
        self._log("Ready. Select an analysis family and variant from the left panel to begin.\n",
                   tag="dim")
        self._log_scripts_availability_summary()

        self.after(150, self._poll_log_queue)
        self.after(300, self._poll_detect_queue)

    def _log_scripts_availability_summary(self):
        total = sum(len(f["variants"]) for f in CATALOG.values())
        missing = []
        for fam, data in CATALOG.items():
            for v in data["variants"]:
                if not os.path.isfile(os.path.join(SCRIPTS_DIR, v["file"])):
                    missing.append(v["file"])
        found = total - len(missing)
        if missing:
            self._log(f"⚠ {found} of {total} catalog scripts found in scripts/ "
                       f"({len(missing)} missing — marked with ⚠ in the sidebar).\n",
                       tag="warn")
            self._log("  Missing files won't run until copied into the 'scripts' "
                       "folder next to launcher.py:\n", tag="dim")
            for f in missing:
                self._log(f"    - {f}\n", tag="dim")
            self._log("\n", tag="dim")
        else:
            self._log(f"✓ All {total} catalog scripts found in scripts/.\n\n", tag="ok")

    # -- form helpers --------------------------------------------------------
    def _row(self, parent, r, label, hint, var, browse_cmd=None, button_text="Browse…"):
        lbl_frame = ttk.Frame(parent, style="ParamCard.TFrame")
        lbl_frame.grid(row=r, column=0, sticky="w", pady=4, padx=(0, 8))
        ttk.Label(lbl_frame, text=label, style="ParamCard.TLabel").pack(anchor="w")
        if hint:
            ttk.Label(lbl_frame, text=hint, style="ParamCardMuted.TLabel",
                      font=(Theme.FONT_FAMILY, 8)).pack(anchor="w")
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=r, column=1, sticky="ew", padx=6, pady=4)
        if browse_cmd:
            ttk.Button(parent, text=button_text, style="Ghost.TButton",
                       command=browse_cmd).grid(row=r, column=2, pady=4)
        return entry

    def _add_lipid(self, name):
        """Append a phospholipid/sterol residue name to the lipid field
        (from a quick-pick chip), avoiding duplicates."""
        current = [s.strip().upper() for s in self.lipid_var.get().replace(",", " ").split() if s.strip()]
        if name.upper() not in current:
            current.append(name.upper())
        self.lipid_var.set(" ".join(current))

    def _browse_tpr(self):
        path = filedialog.askopenfilename(
            title="Select topology file",
            filetypes=[("Topology files", "*.tpr *.gro *.pdb *.psf"), ("All files", "*.*")])
        if path:
            self.tpr_var.set(path)
            self._autodetect_water()

    def _autodetect_water(self):
        """Kick off background detection of the water residue name and
        all-atom vs coarse-grained system type from the currently selected
        topology file. Runs off the UI thread since loading a topology can
        take a moment; results are picked up by _poll_detect_queue()."""
        tpr = self.tpr_var.get().strip()
        if not tpr or not os.path.exists(tpr):
            messagebox.showinfo("Auto-detect water", "Select a valid topology file first.")
            return
        self.water_status_lbl.config(text="Detecting water residue and system type…")
        threading.Thread(target=self._autodetect_water_worker, args=(tpr,), daemon=True).start()

    def _autodetect_water_worker(self, tpr):
        self.detect_queue.put(detect_water_and_system_type(tpr))

    def _poll_detect_queue(self):
        try:
            while True:
                result = self.detect_queue.get_nowait()
                if result.get("resname"):
                    self.water_var.set(result["resname"])
                badge = {"all-atom": "All-atom", "coarse-grained": "Coarse-grained (CG)",
                         "unknown": "Unknown"}.get(result.get("system_type"), "Unknown")
                self.water_status_lbl.config(text=f"[{badge}]  {result.get('detail', '')}")
        except queue.Empty:
            pass
        self.after(300, self._poll_detect_queue)

    def _browse_xtc(self):
        path = filedialog.askopenfilename(
            title="Select trajectory file",
            filetypes=[("Trajectory files", "*.xtc *.trr *.dcd"), ("All files", "*.*")])
        if path:
            self.xtc_var.set(path)

    def _browse_outdir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.outdir_var.set(path)

    def _open_output(self):
        out = self.outdir_var.get()
        if not out or not os.path.isdir(out):
            messagebox.showinfo("Output folder", "Output folder doesn't exist yet — run an analysis first.")
            return
        if sys.platform.startswith("win"):
            os.startfile(out)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", out])
        else:
            subprocess.Popen(["xdg-open", out])

    # -- selection / logging --------------------------------------------------
    def load_variant(self, family_name: str, variant: dict):
        self.current_family = family_name
        self.current_variant = variant
        self.title_lbl.config(text=variant["label"])
        self.file_badge.config(text=variant["file"])
        self.desc_lbl.config(text=f"{family_name}  ·  " + variant.get("desc", ""))

        script_path = os.path.join(SCRIPTS_DIR, variant["file"])
        script_missing = not os.path.isfile(script_path)
        self._set_missing_banner(script_missing, variant["file"])
        self.run_btn.config(state="disabled" if script_missing else "normal")

        patch = variant.get("patch", {})
        has_peptide = (bool(patch.get("peptide_var")) or bool(patch.get("cli"))
                       or bool(patch.get("cli_v2", {}).get("peptide")))
        has_water = (bool(patch.get("water_var")) or bool(patch.get("cli"))
                     or bool(patch.get("cli_v2", {}).get("water")))
        self._set_row_enabled(self.peptide_row, has_peptide)
        # The Ligand field is an alternative way to fill the same
        # peptide/solute-of-interest variable, so it's relevant whenever
        # the Peptide field is.
        self._set_row_enabled(self.ligand_row, has_peptide)
        self._set_row_enabled(self.water_row, has_water)

        # Preview the script's current hard-coded lipid value, read straight
        # from the source file, so the person can see exactly what their
        # phospholipid entry will overwrite.
        self._update_lipid_default_preview(variant)

        self.log_text.delete("1.0", "end")
        self._log(f"Loaded:  {family_name} → {variant['label']}\n", tag="dim")
        self._log(f"Script:  {variant['file']}\n\n", tag="dim")
        if script_missing:
            self._log(
                f"⚠ This script's file is not in your scripts/ folder:\n"
                f"    {script_path}\n"
                f"  Copy '{variant['file']}' into the 'scripts' folder next to launcher.py, "
                f"then re-select this analysis. Run is disabled until it's found.\n\n",
                tag="warn")
        self.status_callback(family=family_name, variant=variant["label"])

    def _set_missing_banner(self, missing: bool, filename: str):
        if missing:
            self.missing_banner_lbl.config(
                text=f"⚠  Script file not found in scripts/: {filename}  "
                     "— place it in the scripts folder next to launcher.py.")
            self.missing_banner_lbl.pack(anchor="w", pady=(4, 0))
        else:
            self.missing_banner_lbl.pack_forget()

    def _update_lipid_default_preview(self, variant):
        patch = variant.get("patch", {})
        lipid_var_name = patch.get("lipid_var")
        if not lipid_var_name:
            if patch.get("cli"):
                self.lipid_default_lbl.config(
                    text="This script takes lipids via a --lipids command-line flag "
                         "(no fixed script default).")
            else:
                self.lipid_default_lbl.config(text="This script variant does not use a lipid selection.")
            return
        src_path = os.path.join(SCRIPTS_DIR, variant["file"])
        current = None
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                current = find_current_var_literal(f.read(), lipid_var_name)
        except OSError:
            pass
        if current:
            self.lipid_default_lbl.config(
                text=f"Script default ({lipid_var_name}): {current}  "
                     "— entering names below will replace this for the run.")
        else:
            self.lipid_default_lbl.config(
                text=f"Patches variable '{lipid_var_name}' in this script; "
                     "current value could not be previewed.")

    def _set_row_enabled(self, entry_widget, enabled):
        entry_widget.config(state="normal" if enabled else "disabled")

    def _log(self, msg, tag=None):
        if tag:
            self.log_text.insert("end", msg, tag)
        else:
            self.log_text.insert("end", msg)
        self.log_text.see("end")

    # -- run / stop --------------------------------------------------------
    def run(self):
        if self.current_variant is None:
            messagebox.showwarning("No analysis selected", "Pick an analysis family and variant first.")
            return
        tpr = self.tpr_var.get().strip()
        xtc = self.xtc_var.get().strip()
        outdir = self.outdir_var.get().strip()
        lipids = self.lipid_var.get().strip()
        if not tpr or not os.path.exists(tpr):
            messagebox.showerror("Missing input", "Please select a valid topology file.")
            return
        if not xtc or not os.path.exists(xtc):
            messagebox.showerror("Missing input", "Please select a valid trajectory file.")
            return
        if not outdir:
            messagebox.showerror("Missing input", "Please choose an output folder.")
            return

        patch = self.current_variant.get("patch", {})
        if patch.get("lipid_var") or patch.get("cli"):
            if not lipids:
                proceed = messagebox.askyesno(
                    "No phospholipid names entered",
                    "You haven't entered any lipid residue names, so this run will use "
                    "the script's built-in default lipid selection instead of your own.\n\n"
                    "Continue anyway?")
                if not proceed:
                    return

        # The Ligand field is an alternate way to fill the same "solute of
        # interest" variable these scripts call PEPTIDE_RESNAME/peptide_var --
        # it exists so people analyzing a ligand-only system (no protein)
        # don't have to type a small molecule's name into a field labeled
        # "Peptide". Whichever one is filled in wins; Peptide takes priority
        # if both are (rare, but avoids silently ignoring an entry).
        peptide_val = self.peptide_var.get().strip()
        ligand_val = self.ligand_var.get().strip()
        solute = peptide_val or ligand_val

        try:
            work_dir = os.path.join(BASE_DIR, "_patched_scripts")
            cmd, run_cwd, report = build_run_command(
                self.current_variant,
                tpr_path=os.path.abspath(tpr),
                xtc_path=os.path.abspath(xtc),
                lipids=lipids,
                peptide=solute,
                water=self.water_var.get().strip(),
                output_dir=os.path.abspath(outdir),
                work_dir=work_dir,
            )
        except Exception as e:
            messagebox.showerror("Could not prepare script", str(e))
            return

        self.log_text.delete("1.0", "end")
        if ligand_val and not peptide_val:
            self._log(f"  ℹ Using Ligand field ('{ligand_val}') for this script's "
                      "peptide/solute-of-interest variable (no peptide entered).\n", tag="dim")
        self._log("Parameters applied to this run:\n", tag="dim")
        for line in report:
            tag = "ok" if line.strip().startswith("✓") else ("warn" if line.strip().startswith("⚠") else None)
            self._log(f"  {line}\n", tag=tag)
        self._log("\n", tag="dim")
        self._log(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n", tag="dim")
        self._log(f"(working directory: {run_cwd})\n\n", tag="dim")

        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.run_state_lbl.config(text="● Running", foreground=Theme.SUCCESS)
        self.progress.start(12)
        self.status_callback(running=True)

        threading.Thread(target=self._run_subprocess, args=(cmd, run_cwd), daemon=True).start()

    def _run_subprocess(self, cmd, cwd):
        try:
            self.process = subprocess.Popen(
                cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True,
            )
            for line in self.process.stdout:
                self.log_queue.put(line)
            self.process.wait()
            self.log_queue.put(f"\n[process exited with code {self.process.returncode}]\n")
        except Exception as e:
            self.log_queue.put(f"\n[ERROR launching process: {e}]\n")
        finally:
            self.process = None
            self.log_queue.put("__DONE__")

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            self._log("\n[Stop requested — terminating process…]\n", tag="dim")

    def _poll_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item == "__DONE__":
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.run_state_lbl.config(text="● Idle", foreground=Theme.TEXT_MUTED)
                    self.progress.stop()
                    self.status_callback(running=False)
                else:
                    self._log(item)
        except queue.Empty:
            pass
        self.after(150, self._poll_log_queue)


# =============================================================================
# GUI — Main application window
# =============================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1320x820")
        self.minsize(1080, 680)
        self.configure(background=Theme.BG)

        self.style = configure_styles(self)
        self._selected_btn = None

        self._build_menu()
        self._build_header()

        body = ttk.Frame(self, style="App.TFrame")
        body.pack(fill="both", expand=True)

        paned = ttk.PanedWindow(body, orient="horizontal")
        paned.pack(fill="both", expand=True)

        self._build_sidebar(paned)

        self.run_panel = RunPanel(paned, status_callback=self._on_run_status)
        paned.add(self.run_panel, weight=3)

        self._build_status_bar()

        if not os.path.isdir(SCRIPTS_DIR):
            messagebox.showwarning(
                "Scripts folder not found",
                f"Expected a 'scripts' folder next to this launcher at:\n{SCRIPTS_DIR}\n\n"
                "Make sure launcher.py and the scripts/ folder are kept together.")

        # The whole window is now laid out with lightweight placeholders and
        # is ready to paint. The two genuinely slow parts -- decoding the
        # header's GIFs/PNGs, and building 100+ catalog buttons with a
        # file-existence check each -- are deferred to run right after the
        # first paint instead of inline here, so the launcher window appears
        # immediately and fills in a beat later instead of sitting blank.
        self.after(1, self._load_deferred_content)

    # -- menu --------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Scripts Folder", command=self._open_scripts_folder)
        file_menu.add_command(label="Open Output Folder", command=lambda: self.run_panel._open_output())
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _open_scripts_folder(self):
        if not os.path.isdir(SCRIPTS_DIR):
            messagebox.showinfo("Scripts folder", f"Not found:\n{SCRIPTS_DIR}")
            return
        if sys.platform.startswith("win"):
            os.startfile(SCRIPTS_DIR)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", SCRIPTS_DIR])
        else:
            subprocess.Popen(["xdg-open", SCRIPTS_DIR])

    def _show_about(self):
        win = tk.Toplevel(self)
        win.title(f"About {APP_NAME}")
        win.configure(background=Theme.PANEL)
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        pad = ttk.Frame(win, style="Card.TFrame", padding=20)
        pad.pack(fill="both", expand=True)

        # Logo + both GIFs, in that order, all in the same ABOUT_MEDIA_BOX so
        # they line up as a single matched row regardless of native size.
        media_row = tk.Frame(pad, background=Theme.PANEL)
        media_row.pack(pady=(0, 14))
        logo_thumb = MediaThumb(media_row, "NICS LAB", ABOUT_MEDIA_BOX, background=Theme.PANEL, dark=False)
        logo_thumb.pack(side="left", padx=5)
        gif1_thumb = MediaThumb(media_row, "Pore formation", ABOUT_MEDIA_BOX, background=Theme.PANEL, dark=False)
        gif1_thumb.pack(side="left", padx=5)
        gif2_thumb = MediaThumb(media_row, "PGLa folding", ABOUT_MEDIA_BOX, background=Theme.PANEL, dark=False)
        gif2_thumb.pack(side="left", padx=5)
        # The About dialog is opened on demand (not at startup), so there's
        # no need to defer this decode — it only runs once the user asks.
        logo_thumb.populate("image", ASSET_LOGO)
        gif1_thumb.populate("gif", ASSET_GIF)
        gif2_thumb.populate("gif", ASSET_GIF2)

        ttk.Label(pad, text=APP_NAME, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(pad, text=f"Version {APP_VERSION}", style="CardMuted.TLabel").pack(anchor="w", pady=(0, 8))
        ttk.Label(pad, text=APP_TAGLINE, style="Card.TLabel", wraplength=420,
                  justify="left").pack(anchor="w", pady=(0, 10))
        ttk.Label(pad, text=(
            "A unified Tkinter front end for the lab's MDAnalysis-based membrane, "
            "curvature, pore, and density analysis scripts. Select an analysis family "
            "and variant, supply your topology/trajectory files, and run — all output "
            "and logs are captured into the folder you choose."),
            style="CardMuted.TLabel", wraplength=420, justify="left").pack(anchor="w", pady=(0, 12))
        ttk.Separator(pad).pack(fill="x", pady=8)
        ttk.Label(pad, text=f"Released under the {LICENSE_NAME}.", style="CardMuted.TLabel",
                  font=(Theme.FONT_FAMILY, 8, "italic")).pack(anchor="w")
        ttk.Button(pad, text="Close", style="Accent.TButton", command=win.destroy).pack(anchor="e", pady=(14, 0))

    # -- header / banner -----------------------------------------------------
    def _build_header(self):
        header = ttk.Frame(self, style="Header.TFrame", padding=(20, 16, 20, 16))
        header.pack(fill="x", side="top")

        left = ttk.Frame(header, style="Header.TFrame")
        left.pack(side="left", fill="both", expand=True)

        title_row = ttk.Frame(left, style="Header.TFrame")
        title_row.pack(anchor="w")
        ttk.Label(title_row, text=APP_NAME, style="HeaderTitle.TLabel").pack(side="left")
        ttk.Label(title_row, text=f"  v{APP_VERSION}", style="HeaderBadge.TLabel").pack(side="left", anchor="s",
                                                                                          pady=(0, 3))
        ttk.Label(left, text=APP_TAGLINE, style="HeaderTagline.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Label(left, text=f"Distributed under the {LICENSE_NAME}  ·  Python {sys.version_info.major}."
                              f"{sys.version_info.minor}  ·  MDAnalysis-based pipelines",
                  style="HeaderTagline.TLabel").pack(anchor="w", pady=(4, 0))

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=(14, 0))

        # A single ordered media strip along the bottom of the header: the
        # lab logo, both reference figures, then both animated previews --
        # left to right in the order a reader would want to make sense of
        # them (brand -> what the data looks like -> the analyses in motion).
        # Every card is the exact same HEADER_MEDIA_BOX, bordered and
        # captioned, so the whole row reads as one deliberate, uniform strip
        # instead of a jumble of differently-sized images.
        strip = tk.Frame(header, background=Theme.HEADER)
        strip.pack(side="right", anchor="s", padx=(24, 0))
        self._header_media = []
        for caption, path, kind in (
            ("NICS LAB",        ASSET_LOGO,        "image"),
            ("Reference A",     ASSET_HEADER_IMG1,  "image"),
            ("Reference B",     ASSET_HEADER_IMG2,  "image"),
            ("Pore formation",  ASSET_GIF,          "gif"),
            ("PGLa folding",    ASSET_GIF2,         "gif"),
        ):
            thumb = MediaThumb(strip, caption, HEADER_MEDIA_BOX, background=Theme.HEADER, dark=True)
            thumb.pack(side="left", padx=5, pady=(0, 2))
            self._header_media.append((thumb, kind, path))

    def _load_header_media(self):
        """Decode and drop the real logo/reference-image/GIF content into the
        header media strip. Called shortly after the window first paints
        (see __init__) rather than during _build_header, so the launcher is
        visible before the slower GIF decoding starts."""
        for thumb, kind, path in getattr(self, "_header_media", []):
            thumb.populate(kind, path)

    def _load_deferred_content(self):
        """Runs once, a beat after the window first paints: fills in the
        header media strip and builds the sidebar's analysis catalog. Both
        are cheap to *look* instant this way even though the underlying work
        (GIF decoding, 100+ button creation with file checks) takes the same
        wall-clock time as before -- the difference is the user sees the
        window immediately instead of staring at nothing while it happens."""
        self._load_header_media()
        self._build_family_tree()

    # -- sidebar -------------------------------------------------------------
    def _build_sidebar(self, paned):
        sidebar = ttk.Frame(paned, style="Sidebar.TFrame")
        paned.add(sidebar, weight=1)

        # Must exist before the search Entry below is created, since
        # inserting its placeholder text fires the write-trace callback.
        self._family_blocks = []  # (family_name_lower, box_frame, [(variant_label_lower, button)])

        top = ttk.Frame(sidebar, style="Sidebar.TFrame", padding=(16, 16, 16, 8))
        top.pack(fill="x")
        ttk.Label(top, text="ANALYSIS CATALOG", style="Sidebar.TLabel").pack(anchor="w")
        ttk.Label(top, text="Pick a family, then a script variant", style="SidebarMuted.TLabel").pack(
            anchor="w", pady=(2, 0))

        search_row = ttk.Frame(sidebar, style="Sidebar.TFrame", padding=(16, 4, 16, 8))
        search_row.pack(fill="x")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *a: self._filter_tree())
        search_entry = ttk.Entry(search_row, textvariable=self.search_var)
        search_entry.pack(fill="x")
        search_entry.insert(0, "")
        self._add_placeholder(search_entry, "Filter analyses…")

        canvas = tk.Canvas(sidebar, highlightthickness=0, background=Theme.SIDEBAR, bd=0)
        vscroll = ttk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
        self.family_container = ttk.Frame(canvas, style="Sidebar.TFrame")
        self.family_container.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.family_container, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(0, 0))
        vscroll.pack(side="right", fill="y")

        def _wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _wheel)

        # Building the full catalog means one ttk.Button per script variant
        # (there are well over a hundred across all families) plus an
        # os.path.isfile() check for each -- by far the single biggest chunk
        # of startup work. It's deferred the same way the header media is
        # (see __init__ / _load_deferred_content), with a lightweight
        # placeholder shown in the meantime, so it never blocks the window
        # from appearing.
        self._catalog_loading_lbl = ttk.Label(
            self.family_container, text="Loading analysis catalog…",
            style="SidebarMuted.TLabel", padding=(14, 10))
        self._catalog_loading_lbl.pack(anchor="w")

    @staticmethod
    def _add_placeholder(entry, text):
        entry.insert(0, text)
        entry.config(foreground=Theme.TEXT_MUTED)

        def on_focus_in(_e):
            if entry.get() == text:
                entry.delete(0, "end")
                entry.config(foreground=Theme.TEXT)

        def on_focus_out(_e):
            if not entry.get():
                entry.insert(0, text)
                entry.config(foreground=Theme.TEXT_MUTED)

        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

    def _build_family_tree(self):
        loading_lbl = getattr(self, "_catalog_loading_lbl", None)
        if loading_lbl is not None:
            loading_lbl.destroy()
            self._catalog_loading_lbl = None
        for family_name, family in CATALOG.items():
            box = ttk.Frame(self.family_container, style="Sidebar.TFrame", padding=(14, 10, 14, 12))
            box.pack(fill="x")

            ttk.Label(box, text=family_name.upper(), style="Sidebar.TLabel",
                      font=(Theme.FONT_FAMILY, 9, "bold")).pack(anchor="w")
            ttk.Label(box, text=family["description"], wraplength=300,
                      style="SidebarMuted.TLabel", justify="left").pack(anchor="w", pady=(3, 8))

            variant_buttons = []
            for variant in family["variants"]:
                script_present = os.path.isfile(os.path.join(SCRIPTS_DIR, variant["file"]))
                icon = "›" if script_present else "⚠"
                label_text = f"{icon}  {variant['label']}" + ("" if script_present else "  (file missing)")
                b = ttk.Button(
                    box, text=label_text,
                    style="Variant.TButton" if script_present else "VariantMissing.TButton",
                    command=lambda f=family_name, v=variant, btn_holder=[]: self._select_variant(f, v))
                b.pack(fill="x", pady=2)
                # rebind with the actual button reference for selection styling
                b.configure(command=lambda f=family_name, v=variant, b=b: self._select_variant(f, v, b))
                variant_buttons.append((variant["label"].lower(), b))

            ttk.Separator(self.family_container).pack(fill="x", pady=(4, 4))
            self._family_blocks.append((family_name.lower(), box, variant_buttons))

    def _select_variant(self, family_name, variant, button):
        if self._selected_btn is not None:
            self._selected_btn.configure(style="Variant.TButton")
        button.configure(style="VariantSelected.TButton")
        self._selected_btn = button
        self.run_panel.load_variant(family_name, variant)

    def _filter_tree(self):
        query = self.search_var.get().strip().lower()
        if query == "filter analyses…":
            query = ""
        for family_lower, box, variant_buttons in self._family_blocks:
            if not query:
                box.pack(fill="x")
                continue
            match = query in family_lower or any(query in lbl for lbl, _ in variant_buttons)
            if match:
                box.pack(fill="x")
            else:
                box.pack_forget()

    # -- status bar -----------------------------------------------------------
    def _build_status_bar(self):
        bar = ttk.Frame(self, style="StatusBar.TFrame", padding=(16, 6, 16, 6))
        bar.pack(fill="x", side="bottom")

        ttk.Label(bar, text="STATUS", style="Status.TLabel").pack(side="left")
        self.status_state_lbl = ttk.Label(bar, text="Idle", style="StatusValue.TLabel")
        self.status_state_lbl.pack(side="left", padx=(6, 18))

        ttk.Label(bar, text="SELECTED", style="Status.TLabel").pack(side="left")
        self.status_sel_lbl = ttk.Label(bar, text="None", style="StatusValue.TLabel")
        self.status_sel_lbl.pack(side="left", padx=(6, 18))

        ttk.Label(bar, text=f"{APP_NAME}  ·  v{APP_VERSION}  ·  {LICENSE_NAME}",
                  style="Status.TLabel").pack(side="right")

    def _on_run_status(self, family=None, variant=None, running=None):
        if family is not None and variant is not None:
            self.status_sel_lbl.config(text=f"{family} → {variant}")
        if running is True:
            self.status_state_lbl.config(text="Running…", foreground=Theme.SUCCESS)
        elif running is False:
            self.status_state_lbl.config(text="Idle", foreground=Theme.TEXT_ON_DARK)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
