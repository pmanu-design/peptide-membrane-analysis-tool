#!/usr/bin/env python3
# =============================================================================
# ULTRA-COMPREHENSIVE MEMBRANE TOPOLOGY & CG MEMBRANE ANALYSIS v4.0 (v10)
# =============================================================================
#
# COMPLETE FEATURE LIST
# -----------------------------------------------------------------------------
# ORIGINAL FEATURES (v1–v3, all preserved)
# ✓ Enhanced peptide molecular visualization
# ✓ Backbone trajectory rendering & residue COM network
# ✓ Peptide insertion depth coloring
# ✓ 2D thickness maps & contour maps, 3D displacement maps
# ✓ Full trajectory averaging & bilayer curvature analysis
# ✓ Pore detection analysis, disruptivity mapping
# ✓ Gaussian / principal curvatures, normal vector field
# ✓ Tilt angle maps, SCD-like order parameter approx
# ✓ Area-per-lipid, Voronoi tessellation, packing defects
# ✓ Roughness maps, surface area, compressibility proxy
# ✓ Radial density maps, distance-dependent deformation
# ✓ Bilayer asymmetry, leaflet coupling, XZ/YZ slices
# ✓ Bending energy, time evolution, stats CSV/Excel
# ✓ Publication PDF, interactive Plotly, dashboard
# ✓ Peptide COM diffusion, helix tilt, orientation
# ✓ Membrane bending analysis (2D & 3D)
#
# NEW IN v4.0
# ✓ N1. TRUE lipid order parameter (SCD) from tail bond vectors (2D + 3D)
# ✓ N2. Per-lipid-type SCD decomposition & comparison bar chart
# ✓ N3. SCD time series (how order evolves over the trajectory)
# ✓ N4. Lateral pressure / stress profile approximation
# ✓ N5. Lipid interdigitation index (tail overlap between leaflets)
# ✓ N6. Hydrogen bond / electrostatic interaction count vs time
# ✓ N7. Cholesterol / lipid tilt angle distribution (2D histogram)
# ✓ N8. Membrane thickness probability distribution
# ✓ N9. Radial distribution function g(r) of headgroups
# ✓ N10. Lipid flip-flop detection (Z-crossing events)
# ✓ N11. Thermal undulation mode analysis (Helfrich κ fit)
# ✓ N12. Comparative 2D/3D order parameter dashboard panel
# ✓ IMPROVED: All plots now use consistent, publication-quality styling
# ✓ IMPROVED: Better colormaps, axis labels, and annotation throughout
#
# UPDATED IN v10 (display window support for all 3-D plots)
# ✓ D1. Interactive backend auto-selection (TkAgg → Qt5Agg → QtAgg →
#        WXAgg → MacOSX → Agg fallback) so 3-D figures open as
#        rotatable pop-up windows when a display is available.
# ✓ D2. All 3-D matplotlib figures now call save_fig(..., show=True),
#        including the previously missing 31_curvature_avg figure.
# ✓ D3. The S_CD comprehensive dashboard (72_SCD_dashboard, fig_op),
#        which contains 3-D subplots and used a bare plt.savefig() call,
#        now calls plt.show() before saving.
# ✓ D4. On headless systems (no display / Agg backend) plt.show() is a
#        no-op — the script continues without interruption or errors.
# ✓ D5. Removed duplicate import block (was imported twice in v9).
#
# =============================================================================

import os
import sys
import warnings
import traceback

import numpy as np
import matplotlib
# ---------------------------------------------------------------------------
# Backend selection: prefer an interactive GUI backend so that 3-D figures
# open in a display window (allows rotation / inspection).  Fall back to
# 'Agg' (file-only, no window) when no display is available (e.g. headless
# cluster nodes or SSH sessions without X11 forwarding).
# ---------------------------------------------------------------------------
_INTERACTIVE_BACKENDS = ['TkAgg', 'Qt5Agg', 'QtAgg', 'WXAgg', 'MacOSX']
_backend_set = False
for _be in _INTERACTIVE_BACKENDS:
    try:
        matplotlib.use(_be)
        import matplotlib.pyplot as _plt_test   # triggers backend init
        _backend_set = True
        print(f"[INFO] Matplotlib backend: {_be}  (interactive windows enabled)")
        del _plt_test
        break
    except Exception:
        continue
if not _backend_set:
    matplotlib.use('Agg')
    print("[WARNING] No interactive display found — falling back to Agg "
          "(3-D figures will be saved as files only, no pop-up windows).")
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

try:
    import plotly.graph_objects as go
    import plotly.subplots as psp
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[WARNING] Plotly not available — skipping interactive 3D.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARNING] Pandas not available — CSV/Excel export may be limited.")

try:
    from openpyxl import Workbook
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
    HAS_VORONOI = True
except ImportError:
    HAS_VORONOI = False

import MDAnalysis as mda
from scipy.interpolate import griddata
from scipy.ndimage import (
    gaussian_filter,
    gaussian_laplace,
    zoom,
    label as nd_label,
    find_objects
)
from scipy.spatial.distance import cdist
from scipy.signal import welch
from tqdm import tqdm

warnings.filterwarnings('ignore')


# (duplicate import block removed — all imports are handled above)

# =============================================================================
# PUBLICATION SETTINGS
# =============================================================================

plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['font.sans-serif']  = ['Arial', 'Helvetica']
plt.rcParams['font.size']        = 10
plt.rcParams['axes.labelsize']   = 11
plt.rcParams['axes.titlesize']   = 12
plt.rcParams['xtick.labelsize']  = 9
plt.rcParams['ytick.labelsize']  = 9
plt.rcParams['legend.fontsize']  = 9
plt.rcParams['lines.linewidth']  = 1.5
plt.rcParams['axes.linewidth']   = 1.2
plt.rcParams['figure.dpi']       = 150

# =============================================================================
# USER SETTINGS  ← edit these before running
# =============================================================================

TPR_FILE        = "md_0_1000.tpr"
XTC_FILE        = "centerB.xtc"

PEPTIDE_RESNAME = "PROTEIN"      # "PROTEIN" or a resname like "KALP"
LIPID_RESNAMES  = "POPE POPG"

FRAME           = -1             # -1 = last frame for snapshot analyses

GRID_SIZE       = 120            # 2D grid resolution for snapshot maps
SMOOTH_SIGMA    = 2.0

OUTPUT_PREFIX   = "ultra_membrane"

MAX_SOLVENT_POINTS = 8000

# Membrane bending modulus κ (kBT) – typical DPPC ≈ 20 kBT
KAPPA_KBT       = 20.0
KB_T_JOULES     = 4.11e-21        # kBT at 300 K in Joules

# =============================================================================
# VISUAL SETTINGS
# =============================================================================

UPPER_ALPHA         = 0.78
LOWER_ALPHA         = 0.78
PEPTIDE_ATOM_SIZE   = 22
PEPTIDE_BACKBONE_SIZE = 40
PEPTIDE_COM_SIZE    = 180

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

for d in ["output", "output/figures_hires", "output/interactive",
          "output/stats", "output/pdf"]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# LOAD SYSTEM
# =============================================================================

print("\n" + "="*70)
print("LOADING SIMULATION")
print("="*70)

u = mda.Universe(TPR_FILE, XTC_FILE)

lipid_selection   = f"resname {LIPID_RESNAMES}"
peptide_selection = "protein" if PEPTIDE_RESNAME.upper() == "PROTEIN" \
                    else f"resname {PEPTIDE_RESNAME}"
solvent_selection = "resname W PW SOL"

lipids   = u.select_atoms(lipid_selection)
peptides = u.select_atoms(peptide_selection)
solvent  = u.select_atoms(solvent_selection)
protein  = peptides

print(f"  Lipids    : {len(lipids)}")
print(f"  Peptide   : {len(peptides)}")
print(f"  Solvent   : {len(solvent)}")
print(f"  Frames    : {len(u.trajectory)}")

# =============================================================================
# SELECT SNAPSHOT FRAME
# =============================================================================

ts      = u.trajectory[FRAME]
time_ns = ts.time / 1000.0
print(f"\n  Snapshot frame : {ts.frame}  ({time_ns:.2f} ns)")

# =============================================================================
# HEADGROUP SELECTION
# =============================================================================

for sel in ["name PO4", "name GL0", "name P"]:
    hg = lipids.select_atoms(sel)
    if len(hg) > 0:
        headgroups = hg
        print(f"  Headgroup sel  : '{sel}'  ({len(hg)} atoms)")
        break
else:
    raise ValueError("No membrane headgroups found with PO4 / GL0 / P names.")

coords = headgroups.positions
x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
z_mid   = np.median(z)

upper_leaflet = coords[z >= z_mid]
lower_leaflet = coords[z <  z_mid]

print(f"  Upper leaflet  : {len(upper_leaflet)}")
print(f"  Lower leaflet  : {len(lower_leaflet)}")

# =============================================================================
# SNAPSHOT GRID
# =============================================================================

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

grid_x, grid_y = np.meshgrid(
    np.linspace(xmin, xmax, GRID_SIZE),
    np.linspace(ymin, ymax, GRID_SIZE)
)

print("\nInterpolating membrane surfaces...")

def interp_surface(leaflet, gx, gy):
    Z = griddata(
        (leaflet[:, 0], leaflet[:, 1]),
        leaflet[:, 2],
        (gx, gy),
        method='cubic'
    )
    Z[np.isnan(Z)] = np.nanmean(Z)
    return gaussian_filter(Z, sigma=SMOOTH_SIGMA)

upper_surface = interp_surface(upper_leaflet, grid_x, grid_y)
lower_surface = interp_surface(lower_leaflet, grid_x, grid_y)

thickness_map    = upper_surface - lower_surface
midplane         = (upper_surface + lower_surface) / 2.0
upper_displ      = upper_surface - np.mean(upper_surface)
lower_displ      = lower_surface - np.mean(lower_surface)
combined_displ   = gaussian_filter(np.abs(upper_displ) + np.abs(lower_displ), 1.5)

# =============================================================================
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  CURVATURE TOOLKIT  (mean, Gaussian, principal, normals, tilt)          │
# └──────────────────────────────────────────────────────────────────────────┘
# =============================================================================

def compute_curvatures(Z):
    """
    Returns:
        H  – mean curvature
        K  – Gaussian curvature
        k1 – first principal curvature
        k2 – second principal curvature
        nx, ny, nz – unit normal components
        tilt_deg   – tilt angle from vertical (degrees)
    """
    dx = (grid_x[0, 1] - grid_x[0, 0])
    dy = (grid_y[1, 0] - grid_y[0, 0])

    Zx  = np.gradient(Z, dx, axis=1)
    Zy  = np.gradient(Z, dy, axis=0)
    Zxx = np.gradient(Zx, dx, axis=1)
    Zyy = np.gradient(Zy, dy, axis=0)
    Zxy = np.gradient(Zx, dy, axis=0)

    W = 1.0 + Zx**2 + Zy**2  # denominator squared

    # Mean curvature (signed)
    H = ((1 + Zy**2) * Zxx - 2*Zx*Zy*Zxy + (1 + Zx**2) * Zyy) / (2 * W**1.5)

    # Gaussian curvature
    K = (Zxx * Zyy - Zxy**2) / W**2

    # Principal curvatures
    disc   = np.sqrt(np.maximum((H**2 - K), 0))
    k1     = H + disc
    k2     = H - disc

    # Surface normals
    mag  = np.sqrt(1.0 + Zx**2 + Zy**2)
    nx   = -Zx / mag
    ny   = -Zy / mag
    nz   =  1.0 / mag

    # Tilt angle from z-axis
    tilt_deg = np.degrees(np.arccos(np.clip(nz, -1, 1)))

    return H, K, k1, k2, nx, ny, nz, tilt_deg


print("Computing curvatures for snapshot frame...")
H_u, K_u, k1_u, k2_u, nx_u, ny_u, nz_u, tilt_u = compute_curvatures(upper_surface)
H_l, K_l, k1_l, k2_l, nx_l, ny_l, nz_l, tilt_l = compute_curvatures(lower_surface)
H_mid, K_mid, k1_m, k2_m, nx_m, ny_m, nz_m, tilt_m = compute_curvatures(midplane)

mean_curvature_snapshot = H_mid
gaussian_curv_snapshot  = K_mid

# =============================================================================
# PEPTIDE REPRESENTATION
# =============================================================================

print("Building peptide representation...")

peptide_atoms = peptides.positions

for sel in ["name BB", "backbone", "name CA"]:
    bb = peptides.select_atoms(sel)
    if len(bb) > 0:
        backbone = bb
        print(f"  Backbone sel: '{sel}'")
        break
else:
    backbone = peptides

backbone_coords = backbone.positions

pep_coms = []
for residue in peptides.residues:
    try:
        pep_coms.append(residue.atoms.center_of_mass())
    except Exception:
        pass
pep_coms = np.array(pep_coms) if pep_coms else np.zeros((0, 3))

depth_values = []
for point in pep_coms:
    px, py, pz = point
    ix = np.abs(grid_x[0] - px).argmin()
    iy = np.abs(grid_y[:, 0] - py).argmin()
    depth_values.append(pz - midplane[iy, ix])
depth_values = np.array(depth_values)

solvent_coords = solvent.positions
if len(solvent_coords) > MAX_SOLVENT_POINTS:
    idx = np.random.choice(len(solvent_coords), MAX_SOLVENT_POINTS, replace=False)
    solvent_coords = solvent_coords[idx]

# peptide COM (for radial analyses)
pep_com_global = pep_coms.mean(axis=0) if len(pep_coms) > 0 \
                 else np.array([np.mean(x), np.mean(y), np.mean(z)])

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def robust_norm(F, pct=98):
    fmax = np.percentile(np.abs(F), pct)
    return np.clip(F / fmax, -1, 1) if fmax > 0 else F

def add_colorbar(ax, surf_or_im, label, shrink=0.8, cmap=None, vmin=None, vmax=None):
    try:
        cbar = plt.colorbar(surf_or_im, ax=ax, pad=0.08, shrink=shrink)
        cbar.set_label(label, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        return cbar
    except Exception:
        return None

def save_fig(fig, basename, hires=True, pdf=True, show=False):
    """Save figure to PNG (and optionally PDF), then close.

    Parameters
    ----------
    show : bool
        If True an interactive display window is opened (plt.show()) before
        saving.  All 3-D figures pass show=True so they appear as rotatable
        pop-up windows when running interactively.  When the script is run in
        a headless environment (no display), the backend falls back to 'Agg'
        and plt.show() becomes a no-op, so the script continues uninterrupted.
    """
    if show:
        plt.show()
    dpi = 300 if hires else 150
    fig.savefig(f"output/figures_hires/{basename}.png", dpi=dpi, bbox_inches='tight')
    if pdf:
        fig.savefig(f"output/figures_hires/{basename}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {basename}")

# radial distance map from peptide COM projected on XY
Rpep = np.sqrt(
    (grid_x - pep_com_global[0])**2 +
    (grid_y - pep_com_global[1])**2
)

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
# S E C T I O N  1  –  SNAPSHOT VISUALIZATIONS (all new + original)
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

print("\n" + "="*70)
print("SECTION 1 – SNAPSHOT VISUALIZATIONS")
print("="*70)

# ─── 1.1  Enhanced 3D topology with cylindrical tube peptide ─────────────────
print("  [1.1] Enhanced 3D topology with cylindrical tube peptide...")

# ── Cylindrical tube builder ─────────────────────────────────────────────────
def build_helix_tube(centers, radius=2.5, n_around=24, n_along=8,
                     color_values=None, cmap_name='turbo'):
    """
    Build a smooth cylindrical tube surface through a sequence of centre points.

    Parameters
    ----------
    centers      : (N, 3) backbone / residue-COM positions along the chain
    radius       : tube cross-section radius in Å
    n_around     : angular resolution (facets around the circle)
    n_along      : interpolation steps between each pair of centres
    color_values : (N,) scalar used to colour the tube (e.g. insertion depth)
    cmap_name    : matplotlib colormap name

    Returns
    -------
    xs, ys, zs   : 2-D arrays suitable for ax.plot_surface
    fcolors      : RGBA face-colour array  (shape xs.shape + (4,))
    """
    if len(centers) < 2:
        return None, None, None, None

    # ── Catmull-Rom spline interpolation ────────────────────────────────────
    def catmull_rom(pts, n_seg):
        """Interpolate a sequence of 3-D points with a Catmull-Rom spline."""
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p0 = pts[max(i - 1, 0)]
            p1 = pts[i]
            p2 = pts[i + 1]
            p3 = pts[min(i + 2, len(pts) - 1)]
            for j in range(1, n_seg + 1):
                t  = j / n_seg
                t2 = t * t;  t3 = t2 * t
                v  = 0.5 * (
                    (2*p1) +
                    (-p0 + p2) * t +
                    (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                    (-p0 + 3*p1 - 3*p2 + p3) * t3
                )
                out.append(v)
        return np.array(out)

    pts_spline = catmull_rom(centers, n_along)

    # interpolate colour values along the spline
    if color_values is not None:
        n_orig = len(centers)
        n_spl  = len(pts_spline)
        t_orig = np.linspace(0, 1, n_orig)
        t_spl  = np.linspace(0, 1, n_spl)
        col_spline = np.interp(t_spl, t_orig, color_values)
    else:
        col_spline = np.linspace(0, 1, len(pts_spline))

    # ── Build Frenet–Serret frames along the spline ─────────────────────────
    tangents = np.gradient(pts_spline, axis=0)
    norms    = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    tangents = tangents / norms

    # initial normal: use any vector not parallel to tangent[0]
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangents[0], ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    normals = np.zeros_like(tangents)
    normals[0] = np.cross(tangents[0], ref)
    normals[0] /= (np.linalg.norm(normals[0]) + 1e-8)
    # propagate with rotation-minimising frames
    for i in range(1, len(tangents)):
        b = np.cross(tangents[i-1], tangents[i])
        bn = np.linalg.norm(b)
        if bn < 1e-8:
            normals[i] = normals[i-1]
        else:
            b /= bn
            theta = np.arccos(np.clip(np.dot(tangents[i-1], tangents[i]), -1, 1))
            # Rodrigues rotation
            n_prev = normals[i-1]
            normals[i] = (n_prev * np.cos(theta) +
                          np.cross(b, n_prev) * np.sin(theta) +
                          b * np.dot(b, n_prev) * (1 - np.cos(theta)))
            normals[i] /= (np.linalg.norm(normals[i]) + 1e-8)
    binormals = np.cross(tangents, normals)

    # ── Sample tube surface ──────────────────────────────────────────────────
    theta_arr = np.linspace(0, 2*np.pi, n_around, endpoint=False)
    n_s = len(pts_spline)
    xs  = np.zeros((n_s, n_around))
    ys  = np.zeros((n_s, n_around))
    zs  = np.zeros((n_s, n_around))

    for i in range(n_s):
        for j, th in enumerate(theta_arr):
            offset = radius * (np.cos(th) * normals[i] + np.sin(th) * binormals[i])
            xs[i, j] = pts_spline[i, 0] + offset[0]
            ys[i, j] = pts_spline[i, 1] + offset[1]
            zs[i, j] = pts_spline[i, 2] + offset[2]

    # close the tube around the circumference
    xs = np.hstack([xs, xs[:, :1]])
    ys = np.hstack([ys, ys[:, :1]])
    zs = np.hstack([zs, zs[:, :1]])

    # ── Face colours ────────────────────────────────────────────────────────
    cmap_t = plt.get_cmap(cmap_name)
    if color_values is not None:
        col_norm = (col_spline - col_spline.min()) / (col_spline.ptp() + 1e-12)
    else:
        col_norm = np.linspace(0, 1, n_s)

    # broadcast colour per-row → shape (n_s, n_around+1, 4)
    fcolors = np.zeros((xs.shape[0], xs.shape[1], 4))
    for i in range(n_s):
        rgba = cmap_t(col_norm[i])
        fcolors[i, :] = rgba

    return xs, ys, zs, fcolors


def draw_tube_caps(ax, centers, radius=2.5, n_around=24, color='red', alpha=0.95):
    """Draw filled circular caps at start and end of the tube."""
    for cap_idx in [0, -1]:
        if cap_idx == 0:
            tang = centers[1] - centers[0]
        else:
            tang = centers[-1] - centers[-2]
        tang = tang / (np.linalg.norm(tang) + 1e-8)

        ref = np.array([0., 0., 1.])
        if abs(np.dot(tang, ref)) > 0.9:
            ref = np.array([0., 1., 0.])
        norm_v = np.cross(tang, ref)
        norm_v /= (np.linalg.norm(norm_v) + 1e-8)
        binorm = np.cross(tang, norm_v)

        theta_arr = np.linspace(0, 2*np.pi, n_around, endpoint=False)
        cx = centers[cap_idx, 0] + radius * (np.cos(theta_arr) * norm_v[0]
                                              + np.sin(theta_arr) * binorm[0])
        cy = centers[cap_idx, 1] + radius * (np.cos(theta_arr) * norm_v[1]
                                              + np.sin(theta_arr) * binorm[1])
        cz = centers[cap_idx, 2] + radius * (np.cos(theta_arr) * norm_v[2]
                                              + np.sin(theta_arr) * binorm[2])
        # fill_between in 3D via plot_trisurf
        cx = np.append(cx, centers[cap_idx, 0])
        cy = np.append(cy, centers[cap_idx, 1])
        cz = np.append(cz, centers[cap_idx, 2])
        tri_i = np.arange(n_around)
        tri_j = (tri_i + 1) % n_around
        tri_k = np.full(n_around, n_around)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        verts = [list(zip(
                     [cx[tri_i[t]], cx[tri_j[t]], cx[tri_k[t]]],
                     [cy[tri_i[t]], cy[tri_j[t]], cy[tri_k[t]]],
                     [cz[tri_i[t]], cz[tri_j[t]], cz[tri_k[t]]]
                 )) for t in range(n_around)]
        cap_poly = Poly3DCollection(verts, alpha=alpha, linewidth=0)
        cap_poly.set_facecolor(color)
        ax.add_collection3d(cap_poly)


# ── Choose source points for the tube ────────────────────────────────────────
# Prefer backbone coords (CA / BB); fall back to residue COMs
if len(backbone_coords) >= 2:
    tube_centers = backbone_coords.copy()
    tube_colors  = None
    # colour by backbone Z relative to midplane
    if len(pep_coms) > 0 and len(depth_values) == len(pep_coms):
        # interpolate depth_values onto backbone points by nearest residue
        if len(pep_coms) == len(backbone_coords):
            tube_colors = depth_values.copy()
        else:
            # nearest-neighbour mapping
            from scipy.spatial import cKDTree
            kd = cKDTree(pep_coms)
            _, idx_map = kd.query(backbone_coords)
            tube_colors = depth_values[idx_map]
elif len(pep_coms) >= 2:
    tube_centers = pep_coms.copy()
    tube_colors  = depth_values.copy() if len(depth_values) == len(pep_coms) else None
else:
    tube_centers = None
    tube_colors  = None

# ── Build single-panel figure (clean presentation) ───────────────────────────
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# ── Membrane surfaces ────────────────────────────────────────────────────────
ax.plot_surface(
    grid_x, grid_y, lower_surface,
    cmap='coolwarm', linewidth=0, antialiased=True,
    alpha=LOWER_ALPHA
)
upper_plot = ax.plot_surface(
    grid_x, grid_y, upper_surface,
    cmap='viridis', linewidth=0, antialiased=True,
    alpha=UPPER_ALPHA
)

# ── Cylindrical tube peptide ─────────────────────────────────────────────────
if tube_centers is not None and len(tube_centers) >= 2:
    xs_t, ys_t, zs_t, fcolors_t = build_helix_tube(
        tube_centers,
        radius       = 2.5,
        n_around     = 28,
        n_along      = 10,
        color_values = tube_colors,
        cmap_name    = 'turbo'
    )
    if xs_t is not None:
        ax.plot_surface(
            xs_t, ys_t, zs_t,
            facecolors  = fcolors_t,
            linewidth   = 0,
            antialiased = True,
            shade       = True,
            alpha       = 0.96,
            rcount      = xs_t.shape[0],
            ccount      = xs_t.shape[1]
        )
        # end caps
        draw_tube_caps(ax, tube_centers, radius=2.5, n_around=28,
                       color='#ff4444', alpha=0.95)

        # ring at each residue COM to mark sequence position
        if len(pep_coms) > 0:
            for ri, (com, dv) in enumerate(zip(pep_coms, depth_values)):
                col_r = plt.cm.turbo(
                    (dv - depth_values.min()) / (depth_values.ptp() + 1e-12)
                )
                ax.scatter(*com, color=col_r, s=30, zorder=5,
                           edgecolors='black', linewidths=0.6, alpha=0.9)

# ── Insertion-depth colourbar ────────────────────────────────────────────────
if len(pep_coms) > 0 and tube_colors is not None:
    sm_tube = cm.ScalarMappable(
        cmap=cm.turbo,
        norm=Normalize(vmin=tube_colors.min(), vmax=tube_colors.max())
    )
    sm_tube.set_array([])
    depth_bar = fig.colorbar(sm_tube, ax=ax, shrink=0.45, pad=0.02)
    depth_bar.set_label("Peptide Insertion Depth (Å)")

# ── Sparse solvent dots ──────────────────────────────────────────────────────
ax.scatter(
    solvent_coords[:, 0], solvent_coords[:, 1], solvent_coords[:, 2],
    c='gray', s=1, alpha=0.018
)

# ── Membrane thickness colourbar ─────────────────────────────────────────────
mappable = plt.cm.ScalarMappable(cmap='coolwarm')
mappable.set_array(thickness_map)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.58, pad=0.08)
cbar.set_label("Membrane Thickness (Å)")

ax.set_xlabel("X (Å)", fontsize=13)
ax.set_ylabel("Y (Å)", fontsize=13)
ax.set_zlabel("Z (Å)", fontsize=13)

ax.set_title(
    f"Enhanced Membrane Topology & Peptide Architecture\n"
    f"Time = {time_ns:.2f} ns",
    fontsize=18,
    pad=20
)

ax.view_init(elev=22, azim=50)
ax.grid(False)

ax.xaxis.pane.fill = True
ax.yaxis.pane.fill = True
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_alpha(0.04)
ax.yaxis.pane.set_alpha(0.04)

plt.tight_layout()
save_fig(fig, "01_enhanced_3D_topology", show=True)

# ─── 1.2  2D thickness map ───────────────────────────────────────────────────
print("  [1.2] 2D thickness map...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
im = axes[0].imshow(thickness_map, cmap='RdYlGn', origin='lower',
                    extent=[xmin, xmax, ymin, ymax])
cb0 = plt.colorbar(im, ax=axes[0], label="Thickness (Å)")
cb0.ax.tick_params(labelsize=8)
axes[0].set_title(f"2D Membrane Thickness\nt = {time_ns:.2f} ns", fontsize=12, fontweight='bold')
axes[0].set_xlabel("X (Å)"); axes[0].set_ylabel("Y (Å)")
# Add mean thickness contour
mean_thk = np.nanmean(thickness_map)
axes[0].contour(grid_x, grid_y, thickness_map, levels=[mean_thk],
                colors=['white'], linewidths=1.5, linestyles='--', alpha=0.8)
axes[0].text(0.02, 0.97, f"Mean = {mean_thk:.1f} Å", transform=axes[0].transAxes,
             color='white', fontsize=9, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

cnt = axes[1].contourf(grid_x, grid_y, thickness_map, levels=50, cmap='RdYlGn')
axes[1].contour(grid_x, grid_y, thickness_map, levels=20,
                colors='black', linewidths=0.4, alpha=0.4)
cb1 = plt.colorbar(cnt, ax=axes[1], label="Thickness (Å)")
cb1.ax.tick_params(labelsize=8)
axes[1].set_title("Thickness Contour Map", fontsize=12, fontweight='bold')
axes[1].set_xlabel("X (Å)"); axes[1].set_ylabel("Y (Å)")
if len(pep_coms) > 0:
    for ax2d in axes:
        sc = ax2d.scatter(pep_coms[:, 0], pep_coms[:, 1],
                     c=depth_values, cmap='turbo', edgecolors='white',
                     linewidths=0.8, s=80, zorder=5, label="Residue COM")
        ax2d.legend(loc='lower right', fontsize=8)
plt.suptitle("Membrane Thickness Analysis", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "02_thickness_maps")

# ─── 1.3  Mean curvature – both leaflets 2D ──────────────────────────────────
print("  [1.3] Mean curvature maps...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
vmax_H = max(np.percentile(np.abs(H_u), 98), np.percentile(np.abs(H_l), 98))
for ax2d, Hmap, title in zip(axes,
                              [H_u, H_l, H_mid],
                              ["Upper Leaflet", "Lower Leaflet", "Midplane"]):
    norm_H = TwoSlopeNorm(vmin=-vmax_H, vcenter=0, vmax=vmax_H)
    im = ax2d.imshow(Hmap, cmap='RdBu_r', origin='lower',
                     extent=[xmin, xmax, ymin, ymax], norm=norm_H)
    plt.colorbar(im, ax=ax2d, label="Mean Curvature H (Å⁻¹)")
    ax2d.set_title(f"Mean Curvature – {title}")
    ax2d.set_xlabel("X (Å)"); ax2d.set_ylabel("Y (Å)")
plt.suptitle("Mean Curvature Analysis", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "03_mean_curvature_2D")

# ─── 1.4  Gaussian curvature 2D ──────────────────────────────────────────────
print("  [1.4] Gaussian curvature maps...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax2d, Kmap, title in zip(axes,
                              [K_u, K_l, K_mid],
                              ["Upper Leaflet", "Lower Leaflet", "Midplane"]):
    vK = np.percentile(np.abs(Kmap), 98)
    im = ax2d.imshow(Kmap, cmap='seismic', origin='lower',
                     extent=[xmin, xmax, ymin, ymax],
                     vmin=-vK, vmax=vK)
    plt.colorbar(im, ax=ax2d, label="Gaussian Curv K (Å⁻²)")
    ax2d.set_title(f"Gaussian Curvature – {title}")
    ax2d.set_xlabel("X (Å)"); ax2d.set_ylabel("Y (Å)")
plt.suptitle("Gaussian Curvature Analysis", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "04_gaussian_curvature_2D")

# ─── 1.5  Principal curvatures k1 k2 ─────────────────────────────────────────
print("  [1.5] Principal curvature maps...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
pairs = [(k1_u, "k₁ Upper"), (k2_u, "k₂ Upper"),
         (k1_l, "k₁ Lower"), (k2_l, "k₂ Lower")]
for ax2d, (kmap, title) in zip(axes.flat, pairs):
    vk = np.percentile(np.abs(kmap), 98)
    im = ax2d.imshow(kmap, cmap='bwr', origin='lower',
                     extent=[xmin, xmax, ymin, ymax],
                     vmin=-vk, vmax=vk)
    plt.colorbar(im, ax=ax2d, label="Principal Curv (Å⁻¹)")
    ax2d.set_title(title); ax2d.set_xlabel("X (Å)"); ax2d.set_ylabel("Y (Å)")
plt.suptitle("Principal Curvatures k₁ and k₂", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "05_principal_curvatures")

# ─── 1.6  Normal vector field ─────────────────────────────────────────────────
print("  [1.6] Normal vector field...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
stride = max(1, GRID_SIZE // 20)
for ax2d, (Z, nx_f, ny_f, tilt, title) in zip(axes, [
    (upper_surface, nx_u, ny_u, tilt_u, "Upper Leaflet"),
    (lower_surface, nx_l, ny_l, tilt_l, "Lower Leaflet")
]):
    im = ax2d.imshow(tilt, cmap='hot', origin='lower',
                     extent=[xmin, xmax, ymin, ymax])
    plt.colorbar(im, ax=ax2d, label="Tilt angle (°)")
    ax2d.quiver(grid_x[::stride, ::stride], grid_y[::stride, ::stride],
                nx_f[::stride, ::stride], ny_f[::stride, ::stride],
                scale=30, color='white', width=0.003, alpha=0.7)
    ax2d.set_title(f"Normal Vector Field & Tilt – {title}")
    ax2d.set_xlabel("X (Å)"); ax2d.set_ylabel("Y (Å)")
plt.suptitle("Membrane Normal Vector Field", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "06_normal_vector_tilt")

# ─── 1.7  Tilt angle maps 3D ─────────────────────────────────────────────────
print("  [1.7] 3D tilt angle maps...")
fig = plt.figure(figsize=(16, 7))
for idx, (surf_Z, tilt, title) in enumerate([
    (upper_surface, tilt_u, "Upper Leaflet"),
    (lower_surface, tilt_l, "Lower Leaflet")
], 1):
    ax3d = fig.add_subplot(1, 2, idx, projection='3d')
    tn   = tilt / max(tilt.max(), 1e-10)
    ax3d.plot_surface(grid_x, grid_y, surf_Z,
                      facecolors=cm.hot(tn), shade=False, rcount=80, ccount=80)
    ax3d.set_xlabel("X (Å)"); ax3d.set_ylabel("Y (Å)"); ax3d.set_zlabel("Z (Å)")
    ax3d.set_title(f"Tilt Angle – {title}")
    ax3d.view_init(elev=25, azim=45)
sm = cm.ScalarMappable(cmap=cm.hot, norm=Normalize(vmin=0, vmax=tilt_u.max()))
sm.set_array([])
fig.colorbar(sm, ax=fig.axes, shrink=0.6, label="Tilt angle (°)")
plt.suptitle("Membrane Surface Tilt Angle Maps", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "07_tilt_angle_3D", show=True)

# ─── 1.8  TRUE Lipid Order Parameter (SCD from tail bond vectors) ─────────────
print("  [1.8] TRUE lipid order parameter (SCD from bond vectors)...")

# ── Compute SCD from actual C–C (or SC–SC for CG) tail bond vectors ──────────
def compute_scd_per_lipid(lipid_ag, z_mid_val):
    """
    For each lipid residue find the tail heavy atoms (C and S names),
    sort them by residue-internal atom index to get the chain order,
    then compute S_CD = <0.5*(3cos²θ - 1)> where θ is the bond-to-z angle.
    Returns array of (x, y, scd) for each lipid and leaflet label.
    """
    results = []
    for res in lipid_ag.residues:
        # grab tail atoms: carbons + sulfur for CG MARTINI
        tail = res.atoms.select_atoms("name C* S* D*")
        if len(tail) < 2:
            tail = res.atoms  # fallback: all atoms
        pos = tail.positions[np.argsort(tail.ids)]  # order by atom id
        if len(pos) < 2:
            continue
        bonds = np.diff(pos, axis=0)                # bond vectors
        bond_len = np.linalg.norm(bonds, axis=1, keepdims=True)
        bond_len[bond_len < 1e-6] = 1e-6
        bonds_n = bonds / bond_len                  # unit bond vectors
        cos2 = bonds_n[:, 2]**2                     # cos²θ from z
        scd_res = np.mean(0.5 * (3 * cos2 - 1))    # S_CD
        com = res.atoms.center_of_mass()
        leaflet = "upper" if com[2] >= z_mid_val else "lower"
        results.append((com[0], com[1], com[2], scd_res, leaflet))
    return results

scd_data = compute_scd_per_lipid(lipids, z_mid)
if len(scd_data) > 0:
    scd_arr = np.array([(r[0], r[1], r[2], r[3]) for r in scd_data])
    leaf_label = [r[4] for r in scd_data]

    scd_u_pts = scd_arr[[l == "upper" for l in leaf_label]]
    scd_l_pts = scd_arr[[l == "lower" for l in leaf_label]]

    def scd_to_grid(pts, gx, gy):
        """Interpolate per-lipid SCD values onto the regular grid."""
        if len(pts) < 4:
            return np.full_like(gx, np.nan)
        G = griddata((pts[:, 0], pts[:, 1]), pts[:, 3],
                     (gx, gy), method='linear')
        G[np.isnan(G)] = np.nanmean(pts[:, 3])
        return gaussian_filter(G, sigma=SMOOTH_SIGMA * 0.8)

    scd_grid_u = scd_to_grid(scd_u_pts, grid_x, grid_y)
    scd_grid_l = scd_to_grid(scd_l_pts, grid_x, grid_y)

    # ── 2D SCD maps ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax2d, scd_g, pts, title in zip(
        axes,
        [scd_grid_u, scd_grid_l],
        [scd_u_pts, scd_l_pts],
        ["Upper Leaflet", "Lower Leaflet"]
    ):
        im = ax2d.imshow(scd_g, cmap='RdYlGn_r', origin='lower',
                         extent=[xmin, xmax, ymin, ymax], vmin=-0.5, vmax=0.5)
        cb = plt.colorbar(im, ax=ax2d, label="S_CD")
        cb.ax.tick_params(labelsize=8)
        # overlay per-lipid scatter
        sc = ax2d.scatter(pts[:, 0], pts[:, 1], c=pts[:, 3], cmap='RdYlGn_r',
                          vmin=-0.5, vmax=0.5, s=14, edgecolors='black',
                          linewidths=0.3, alpha=0.75, zorder=3)
        mean_scd = np.nanmean(pts[:, 3])
        ax2d.set_title(f"TRUE S_CD – {title}\nMean = {mean_scd:.3f}", fontsize=11, fontweight='bold')
        ax2d.set_xlabel("X (Å)"); ax2d.set_ylabel("Y (Å)")
        # annotate reference lines
        ax2d.axhline(np.mean(grid_y), color='white', linewidth=0.6, linestyle=':', alpha=0.5)
        ax2d.axvline(np.mean(grid_x), color='white', linewidth=0.6, linestyle=':', alpha=0.5)
        if len(pep_coms) > 0:
            ax2d.scatter(pep_coms[:, 0], pep_coms[:, 1], c='cyan', s=60,
                         marker='*', zorder=10, edgecolors='navy', linewidths=0.5,
                         label="Peptide")
            ax2d.legend(fontsize=8, loc='lower right')
    plt.suptitle("Lipid Chain Order Parameter S_CD\n(from tail bond vectors, snapshot)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(); save_fig(fig, "08_lipid_order_parameter_SCD_2D")
    print("  ✓ TRUE SCD 2D map saved")

    # ── 3D SCD surface ───────────────────────────────────────────────────────
    print("  [1.8b] 3D SCD surface...")
    fig = plt.figure(figsize=(17, 7))
    fig.suptitle("Lipid Order Parameter S_CD – 3D Surface (snapshot)",
                 fontsize=13, fontweight='bold')
    for idx, (surf_Z, scd_g, pts, title) in enumerate([
        (upper_surface, scd_grid_u, scd_u_pts, "Upper Leaflet"),
        (lower_surface, scd_grid_l, scd_l_pts, "Lower Leaflet")
    ], 1):
        ax3d = fig.add_subplot(1, 2, idx, projection='3d')
        # normalize SCD to [0,1] for colormap
        scd_norm = (np.clip(scd_g, -0.5, 0.5) + 0.5)          # 0 to 1
        ax3d.plot_surface(grid_x, grid_y, surf_Z,
                          facecolors=cm.RdYlGn_r(scd_norm),
                          shade=True, rcount=80, ccount=80, alpha=0.92)
        # scatter per-lipid SCD dots on the surface
        if len(pts) > 0:
            ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                         c=pts[:, 3], cmap='RdYlGn_r',
                         vmin=-0.5, vmax=0.5, s=15, alpha=0.8, zorder=5)
        ax3d.set_xlabel("X (Å)", labelpad=4)
        ax3d.set_ylabel("Y (Å)", labelpad=4)
        ax3d.set_zlabel("Z (Å)", labelpad=4)
        ax3d.set_title(f"S_CD – {title}\nMean = {np.nanmean(pts[:, 3]):.3f}",
                       fontsize=10, fontweight='bold')
        ax3d.view_init(elev=28, azim=45)
        ax3d.set_box_aspect([1, 1, 0.5])

    sm_scd = cm.ScalarMappable(cmap=cm.RdYlGn_r, norm=Normalize(vmin=-0.5, vmax=0.5))
    sm_scd.set_array([])
    fig.colorbar(sm_scd, ax=fig.axes, shrink=0.55, pad=0.04, label="S_CD")
    plt.tight_layout(); save_fig(fig, "08b_lipid_order_parameter_SCD_3D", show=True)
    print("  ✓ TRUE SCD 3D surface saved")

    # ── SCD comparison bar chart by lipid type ───────────────────────────────
    print("  [1.8c] SCD by lipid type...")
    lipid_resnames_list = list(set([r.resname for r in lipids.residues]))
    scd_by_type_u = {}
    scd_by_type_l = {}
    for rn in lipid_resnames_list:
        sub = lipids.select_atoms(f"resname {rn}")
        dat = compute_scd_per_lipid(sub, z_mid)
        if len(dat) == 0:
            continue
        arr = np.array([(d[3], d[4]) for d in dat], dtype=object)
        u_vals = [float(d[0]) for d in arr if d[1] == "upper"]
        l_vals = [float(d[0]) for d in arr if d[1] == "lower"]
        scd_by_type_u[rn] = u_vals
        scd_by_type_l[rn] = l_vals

    if scd_by_type_u:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax2d, scd_dict, leaflet in zip(axes,
                                           [scd_by_type_u, scd_by_type_l],
                                           ["Upper Leaflet", "Lower Leaflet"]):
            names = list(scd_dict.keys())
            means = [np.mean(v) for v in scd_dict.values()]
            stds  = [np.std(v)  for v in scd_dict.values()]
            colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
            bars = ax2d.bar(names, means, yerr=stds, color=colors,
                            edgecolor='black', linewidth=0.8,
                            error_kw=dict(elinewidth=1.5, capsize=4, capthick=1.5))
            ax2d.axhline(0, color='gray', linewidth=1, linestyle='--')
            ax2d.set_xlabel("Lipid Type"); ax2d.set_ylabel("Mean S_CD")
            ax2d.set_title(f"S_CD by Lipid Type – {leaflet}", fontweight='bold')
            ax2d.set_ylim(-0.55, 0.55)
            ax2d.grid(True, axis='y', alpha=0.3)
            for bar, mean in zip(bars, means):
                ax2d.text(bar.get_x() + bar.get_width()/2, mean + 0.02,
                          f"{mean:.3f}", ha='center', va='bottom', fontsize=8)
        plt.suptitle("Lipid Order Parameter S_CD Decomposition by Lipid Type",
                     fontsize=13, fontweight='bold')
        plt.tight_layout(); save_fig(fig, "08c_SCD_by_lipid_type")

    # Store SCD grids for dashboard use
    S_u = scd_grid_u
    S_l = scd_grid_l
    scd_mean_u = np.nanmean(scd_u_pts[:, 3]) if len(scd_u_pts) else np.nan
    scd_mean_l = np.nanmean(scd_l_pts[:, 3]) if len(scd_l_pts) else np.nan
else:
    # Fallback: SCD approximation from tilt angle
    print("  [1.8] Fallback: SCD from tilt angle approximation...")
    S_u = (3 * np.cos(np.radians(tilt_u))**2 - 1) / 2.0
    S_l = (3 * np.cos(np.radians(tilt_l))**2 - 1) / 2.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax2d, S, title in zip(axes, [S_u, S_l], ["Upper Leaflet", "Lower Leaflet"]):
        im = ax2d.imshow(S, cmap='RdYlGn_r', origin='lower',
                         extent=[xmin, xmax, ymin, ymax], vmin=-0.5, vmax=0.5)
        plt.colorbar(im, ax=ax2d, label="S_CD (tilt approx)")
        ax2d.set_title(f"Order Parameter – {title}")
        ax2d.set_xlabel("X (Å)"); ax2d.set_ylabel("Y (Å)")
    plt.suptitle("Lipid Order Parameter Approximation (from tilt)", fontsize=14, fontweight='bold')
    plt.tight_layout(); save_fig(fig, "08_lipid_order_parameter")
    scd_mean_u = np.nanmean(S_u)
    scd_mean_l = np.nanmean(S_l)

# ─── 1.9  Area-per-lipid estimation ──────────────────────────────────────────
print("  [1.9] Area-per-lipid...")
total_area_upper = np.sum(
    np.sqrt(1 + np.gradient(upper_surface, axis=0)**2 +
                np.gradient(upper_surface, axis=1)**2)
) * (grid_x[0, 1]-grid_x[0, 0]) * (grid_y[1, 0]-grid_y[0, 0])
total_area_lower = np.sum(
    np.sqrt(1 + np.gradient(lower_surface, axis=0)**2 +
                np.gradient(lower_surface, axis=1)**2)
) * (grid_x[0, 1]-grid_x[0, 0]) * (grid_y[1, 0]-grid_y[0, 0])
n_upper = len(upper_leaflet)
n_lower = len(lower_leaflet)
APL_u   = total_area_upper / n_upper if n_upper > 0 else np.nan
APL_l   = total_area_lower / n_lower if n_lower > 0 else np.nan
print(f"  APL upper = {APL_u:.2f} Å²   APL lower = {APL_l:.2f} Å²")

# ─── 1.10  Lipid density heat maps ───────────────────────────────────────────
print("  [1.10] Lipid density heat maps...")
def density_map(leaflet_coords, gx, gy, sigma=3.0):
    from scipy.stats import gaussian_kde
    H2d, xe, ye = np.histogram2d(leaflet_coords[:, 0], leaflet_coords[:, 1],
                                  bins=GRID_SIZE,
                                  range=[[xmin, xmax], [ymin, ymax]])
    return gaussian_filter(H2d.T, sigma=sigma)

dens_u = density_map(upper_leaflet, grid_x, grid_y)
dens_l = density_map(lower_leaflet, grid_x, grid_y)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax2d, D, title in zip(axes, [dens_u, dens_l], ["Upper", "Lower"]):
    im = ax2d.imshow(D, cmap='YlOrRd', origin='lower',
                     extent=[xmin, xmax, ymin, ymax])
    plt.colorbar(im, ax=ax2d, label="Density (a.u.)")
    ax2d.set_title(f"Lipid Density – {title} Leaflet")
    ax2d.set_xlabel("X (Å)"); ax2d.set_ylabel("Y (Å)")
plt.suptitle("Lipid Density Heat Maps", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "09_lipid_density_maps")

# ─── 1.11  Voronoi tessellation & packing defects ────────────────────────────
print("  [1.11] Voronoi tessellation + packing defects...")
if HAS_VORONOI and len(upper_leaflet) > 4:
    try:
        pts_u = upper_leaflet[:, :2]
        vor_u = Voronoi(pts_u)

        # cell areas
        from scipy.spatial import ConvexHull
        areas_u = []
        for region_idx in vor_u.point_region:
            region = vor_u.regions[region_idx]
            if -1 in region or len(region) == 0:
                areas_u.append(np.nan)
            else:
                try:
                    hull = ConvexHull(vor_u.vertices[region])
                    areas_u.append(hull.volume)
                except Exception:
                    areas_u.append(np.nan)
        areas_u = np.array(areas_u)
        valid_mask = ~np.isnan(areas_u)
        mean_area  = np.nanmean(areas_u)
        defect_thr = mean_area * 1.5

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Voronoi plot
        ax_v = axes[0]
        from scipy.spatial import voronoi_plot_2d
        voronoi_plot_2d(vor_u, ax=ax_v, show_vertices=False,
                        line_colors='steelblue', line_width=0.8, line_alpha=0.6,
                        point_size=2)
        ax_v.set_xlim(xmin, xmax); ax_v.set_ylim(ymin, ymax)
        ax_v.set_title("Voronoi Tessellation (Upper Leaflet)")
        ax_v.set_xlabel("X (Å)"); ax_v.set_ylabel("Y (Å)")
        # Packing defect heat map
        defect_pts = pts_u[valid_mask & (areas_u > defect_thr)]
        H2, xe2, ye2 = np.histogram2d(defect_pts[:, 0], defect_pts[:, 1],
                                       bins=GRID_SIZE // 2,
                                       range=[[xmin, xmax], [ymin, ymax]])
        im = axes[1].imshow(gaussian_filter(H2.T, 2), cmap='Reds', origin='lower',
                             extent=[xmin, xmax, ymin, ymax])
        plt.colorbar(im, ax=axes[1], label="Defect count")
        axes[1].set_title("Packing Defect Heat Map (Upper Leaflet)")
        axes[1].set_xlabel("X (Å)"); axes[1].set_ylabel("Y (Å)")
        plt.suptitle("Voronoi Tessellation & Packing Defects", fontsize=14, fontweight='bold')
        plt.tight_layout(); save_fig(fig, "10_voronoi_packing_defects")
    except Exception as e:
        print(f"  [skip] Voronoi: {e}")

# ─── 1.12  Roughness analysis ─────────────────────────────────────────────────
print("  [1.12] Roughness analysis...")
def rms_roughness(Z):
    return np.sqrt(np.mean((Z - np.mean(Z))**2))

Rq_u = rms_roughness(upper_surface)
Rq_l = rms_roughness(lower_surface)
Ra_u = np.mean(np.abs(upper_surface - np.mean(upper_surface)))
Ra_l = np.mean(np.abs(lower_surface - np.mean(lower_surface)))
print(f"  Rq upper = {Rq_u:.3f} Å   Rq lower = {Rq_l:.3f} Å")

# roughness map (local)
from scipy.ndimage import generic_filter
def local_rq(Z, size=7):
    def rq_kernel(vals): return np.sqrt(np.mean((vals - np.mean(vals))**2))
    return generic_filter(Z, rq_kernel, size=size)

local_rq_u = local_rq(upper_surface)
local_rq_l = local_rq(lower_surface)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax2d, Rmap, title in zip(axes,
                              [local_rq_u, local_rq_l],
                              ["Upper", "Lower"]):
    im = ax2d.imshow(Rmap, cmap='magma', origin='lower',
                     extent=[xmin, xmax, ymin, ymax])
    plt.colorbar(im, ax=ax2d, label="Local Rq (Å)")
    ax2d.set_title(f"Membrane Roughness – {title} Leaflet")
    ax2d.set_xlabel("X (Å)"); ax2d.set_ylabel("Y (Å)")
plt.suptitle("Membrane Roughness (RMS)", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "11_roughness_maps")

# ─── 1.13  Surface area per leaflet ──────────────────────────────────────────
print("  [1.13] Surface area calculations...")
def surface_area_numerical(Z, dx, dy):
    Zx = np.gradient(Z, dx, axis=1)
    Zy = np.gradient(Z, dy, axis=0)
    dA = np.sqrt(1 + Zx**2 + Zy**2) * dx * dy
    return np.sum(dA)

dx_grid = (xmax - xmin) / GRID_SIZE
dy_grid = (ymax - ymin) / GRID_SIZE
SA_u    = surface_area_numerical(upper_surface, dx_grid, dy_grid)
SA_l    = surface_area_numerical(lower_surface, dx_grid, dy_grid)
SA_flat = (xmax - xmin) * (ymax - ymin)
print(f"  Surface area upper = {SA_u:.1f} Å²   lower = {SA_l:.1f} Å²")
print(f"  Flat projection    = {SA_flat:.1f} Å²")

# ─── 1.14  Local compressibility map ─────────────────────────────────────────
print("  [1.14] Local compressibility map...")
# κ_A ≈ local variance of thickness (proxy; real κ_A needs ensemble)
from scipy.ndimage import generic_filter

def local_std(Z, size=7):
    def std_k(v): return np.std(v)
    return generic_filter(Z, std_k, size=size)

comp_map = local_std(thickness_map)
fig, ax2d = plt.subplots(figsize=(9, 7))
im = ax2d.imshow(comp_map, cmap='viridis', origin='lower',
                 extent=[xmin, xmax, ymin, ymax])
plt.colorbar(im, ax=ax2d, label="Local Thickness Std (Å)")
ax2d.set_title("Local Membrane Compressibility Proxy")
ax2d.set_xlabel("X (Å)"); ax2d.set_ylabel("Y (Å)")
if len(pep_coms) > 0:
    ax2d.scatter(pep_coms[:, 0], pep_coms[:, 1], c='red', s=80, zorder=5,
                 edgecolors='white', label="Peptide")
    ax2d.legend()
plt.tight_layout(); save_fig(fig, "12_compressibility_map")

# ─── 1.15  2D radial density map around peptide ───────────────────────────────
print("  [1.15] 2D radial density map around peptide...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax2d, D, title in zip(axes, [dens_u, dens_l], ["Upper", "Lower"]):
    im = ax2d.imshow(D, cmap='YlOrRd', origin='lower',
                     extent=[xmin, xmax, ymin, ymax])
    plt.colorbar(im, ax=ax2d, label="Density (a.u.)")
    if len(pep_coms) > 0:
        for r in [20, 40, 60, 80]:
            circle = plt.Circle((pep_com_global[0], pep_com_global[1]), r,
                                 fill=False, color='cyan', linewidth=1.2,
                                 linestyle='--', alpha=0.7)
            ax2d.add_patch(circle)
        ax2d.scatter([pep_com_global[0]], [pep_com_global[1]],
                     c='blue', s=120, zorder=10, marker='*', label="Peptide")
        ax2d.legend()
    ax2d.set_title(f"Radial Density – {title} Leaflet")
    ax2d.set_xlabel("X (Å)"); ax2d.set_ylabel("Y (Å)")
plt.suptitle("2D Radial Density Maps Around Peptide", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "13_radial_density_maps")

# ─── 1.16  Distance-dependent membrane deformation ───────────────────────────
print("  [1.16] Distance-dependent deformation profiles...")
R_flat = Rpep.flatten()
sort_idx = np.argsort(R_flat)
R_sorted = R_flat[sort_idx]
thk_sorted   = thickness_map.flatten()[sort_idx]
displ_sorted = combined_displ.flatten()[sort_idx]
H_sorted     = H_mid.flatten()[sort_idx]

bin_edges  = np.linspace(0, R_sorted.max(), 50)
bin_centers= 0.5*(bin_edges[:-1]+bin_edges[1:])
def bin_stat(vals, R=R_sorted, edges=bin_edges):
    means, stds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (R >= lo) & (R < hi)
        v = vals[mask]
        means.append(np.mean(v) if len(v) > 0 else np.nan)
        stds.append(np.std(v)  if len(v) > 0 else np.nan)
    return np.array(means), np.array(stds)

thk_m, thk_s     = bin_stat(thk_sorted)
displ_m, displ_s = bin_stat(displ_sorted)
H_m, H_s         = bin_stat(H_sorted)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax2d, (m, s, lbl) in zip(axes, [
    (thk_m, thk_s, "Thickness (Å)"),
    (displ_m, displ_s, "Displacement (Å)"),
    (H_m, H_s, "Mean Curvature (Å⁻¹)")
]):
    ax2d.plot(bin_centers, m, 'b-', linewidth=2)
    ax2d.fill_between(bin_centers, m-s, m+s, alpha=0.3, color='blue')
    ax2d.set_xlabel("Distance from Peptide (Å)")
    ax2d.set_ylabel(lbl)
    ax2d.set_title(f"{lbl} vs Distance")
    ax2d.axhline(np.nanmean(m), color='red', linestyle='--', label='mean')
    ax2d.legend()
plt.suptitle("Distance-Dependent Membrane Deformation", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "14_distance_deformation_profiles")

# ─── 1.17  Thickness vs distance  &  Curvature vs distance (dedicated) ───────
print("  [1.17] Thickness & curvature vs distance...")
fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(14, 6))
ax_t.plot(bin_centers, thk_m, 'g-', linewidth=2.5)
ax_t.fill_between(bin_centers, thk_m-thk_s, thk_m+thk_s, alpha=0.25, color='g')
ax_t.set_xlabel("Distance from Peptide (Å)")
ax_t.set_ylabel("Membrane Thickness (Å)")
ax_t.set_title("Thickness vs Distance from Peptide")
ax_t.grid(True, alpha=0.3)

ax_c.plot(bin_centers, H_m, 'r-', linewidth=2.5)
ax_c.fill_between(bin_centers, H_m-H_s, H_m+H_s, alpha=0.25, color='r')
ax_c.set_xlabel("Distance from Peptide (Å)")
ax_c.set_ylabel("Mean Curvature H (Å⁻¹)")
ax_c.set_title("Curvature vs Distance from Peptide")
ax_c.axhline(0, color='black', linestyle='--', linewidth=1)
ax_c.grid(True, alpha=0.3)
plt.tight_layout(); save_fig(fig, "15_thickness_curvature_vs_distance")

# ─── 1.18  Pore detection + pore size distribution + connected clusters ───────
print("  [1.18] Pore detection / size distribution / cluster analysis...")
thr_pore = np.percentile(thickness_map, 10)
pore_mask = (thickness_map < thr_pore).astype(int)

# connected component labelling
labeled_pores, n_pore_clusters = nd_label(pore_mask)
cluster_sizes = []
for label_i in range(1, n_pore_clusters + 1):
    size = np.sum(labeled_pores == label_i) * dx_grid * dy_grid  # Å²
    cluster_sizes.append(size)
cluster_sizes = np.array(cluster_sizes)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# pore map
im = axes[0].imshow(pore_mask, cmap='hot_r', origin='lower',
                    extent=[xmin, xmax, ymin, ymax])
axes[0].set_title(f"Pore Regions (Thickness < {thr_pore:.1f} Å)")
axes[0].set_xlabel("X (Å)"); axes[0].set_ylabel("Y (Å)")
# cluster map
im2 = axes[1].imshow(labeled_pores, cmap='tab20', origin='lower',
                     extent=[xmin, xmax, ymin, ymax])
plt.colorbar(im2, ax=axes[1], label="Cluster ID")
axes[1].set_title(f"Pore Clusters (n={n_pore_clusters})")
axes[1].set_xlabel("X (Å)"); axes[1].set_ylabel("Y (Å)")
# size distribution
if len(cluster_sizes) > 0:
    axes[2].hist(cluster_sizes, bins=max(5, len(cluster_sizes)//2 + 1),
                 color='steelblue', edgecolor='black')
    axes[2].set_xlabel("Pore Cluster Area (Å²)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Pore Size Distribution")
else:
    axes[2].text(0.5, 0.5, 'No pores detected', ha='center', va='center',
                 transform=axes[2].transAxes, fontsize=12)
plt.suptitle("Pore Analysis", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "16_pore_analysis")

# ─── 1.19  Bilayer asymmetry index ───────────────────────────────────────────
print("  [1.19] Bilayer asymmetry index...")
upper_norm_pos = upper_surface - np.mean(upper_surface)
lower_norm_pos = lower_surface - np.mean(lower_surface)
asymmetry_map  = upper_norm_pos + lower_norm_pos   # 0 = symmetric
AI             = np.mean(np.abs(asymmetry_map))
fig, ax2d = plt.subplots(figsize=(9, 7))
vA = np.percentile(np.abs(asymmetry_map), 98)
im = ax2d.imshow(asymmetry_map, cmap='coolwarm', origin='lower',
                 extent=[xmin, xmax, ymin, ymax], vmin=-vA, vmax=vA)
plt.colorbar(im, ax=ax2d, label="Asymmetry (Å)")
ax2d.set_title(f"Bilayer Asymmetry Index (mean |AI| = {AI:.2f} Å)")
ax2d.set_xlabel("X (Å)"); ax2d.set_ylabel("Y (Å)")
plt.tight_layout(); save_fig(fig, "17_asymmetry_index")

# ─── 1.20  Leaflet coupling analysis ─────────────────────────────────────────
print("  [1.20] Leaflet coupling analysis...")
from scipy.signal import correlate2d
coupling_cc = np.corrcoef(upper_surface.flatten(), lower_surface.flatten())[0, 1]
cross_corr  = correlate2d(upper_norm_pos - upper_norm_pos.mean(),
                           lower_norm_pos - lower_norm_pos.mean(),
                           mode='same', boundary='wrap')
cross_corr /= (np.std(upper_norm_pos) * np.std(lower_norm_pos) *
               upper_norm_pos.size + 1e-12)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(upper_surface.flatten(), lower_surface.flatten(), alpha=0.05,
                s=1, c='steelblue')
axes[0].set_xlabel("Upper leaflet Z (Å)"); axes[0].set_ylabel("Lower leaflet Z (Å)")
axes[0].set_title(f"Leaflet Height Correlation (r = {coupling_cc:.3f})")
axes[0].plot([upper_surface.min(), upper_surface.max()],
             [upper_surface.min(), upper_surface.max()], 'r--', linewidth=1)
im = axes[1].imshow(cross_corr, cmap='RdBu_r', origin='lower',
                    extent=[xmin, xmax, ymin, ymax])
plt.colorbar(im, ax=axes[1], label="2D Cross-Correlation")
axes[1].set_title("Spatial Leaflet Coupling Map")
axes[1].set_xlabel("ΔX (Å)"); axes[1].set_ylabel("ΔY (Å)")
plt.suptitle("Leaflet Coupling Analysis", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "18_leaflet_coupling")

# ─── 1.21  Cross-sectional XZ and YZ slices ──────────────────────────────────
print("  [1.21] Cross-sectional XZ and YZ slices...")
mid_row = GRID_SIZE // 2
mid_col = GRID_SIZE // 2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# XZ slice
axes[0, 0].plot(grid_x[mid_row, :], upper_surface[mid_row, :], 'b-', label='Upper')
axes[0, 0].plot(grid_x[mid_row, :], lower_surface[mid_row, :], 'r-', label='Lower')
axes[0, 0].fill_between(grid_x[mid_row, :],
                         lower_surface[mid_row, :], upper_surface[mid_row, :],
                         alpha=0.2, color='green')
axes[0, 0].set_xlabel("X (Å)"); axes[0, 0].set_ylabel("Z (Å)")
axes[0, 0].set_title("XZ Cross-Section (mid Y)")
axes[0, 0].legend()
# YZ slice
axes[0, 1].plot(grid_y[:, mid_col], upper_surface[:, mid_col], 'b-', label='Upper')
axes[0, 1].plot(grid_y[:, mid_col], lower_surface[:, mid_col], 'r-', label='Lower')
axes[0, 1].fill_between(grid_y[:, mid_col],
                         lower_surface[:, mid_col], upper_surface[:, mid_col],
                         alpha=0.2, color='green')
axes[0, 1].set_xlabel("Y (Å)"); axes[0, 1].set_ylabel("Z (Å)")
axes[0, 1].set_title("YZ Cross-Section (mid X)")
axes[0, 1].legend()
# Thickness XZ
axes[1, 0].plot(grid_x[mid_row, :], thickness_map[mid_row, :], 'g-', linewidth=2)
axes[1, 0].set_xlabel("X (Å)"); axes[1, 0].set_ylabel("Thickness (Å)")
axes[1, 0].set_title("Thickness XZ Profile")
# Thickness YZ
axes[1, 1].plot(grid_y[:, mid_col], thickness_map[:, mid_col], 'g-', linewidth=2)
axes[1, 1].set_xlabel("Y (Å)"); axes[1, 1].set_ylabel("Thickness (Å)")
axes[1, 1].set_title("Thickness YZ Profile")
plt.suptitle("Cross-Sectional Membrane Slices", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "19_cross_sectional_slices")

# ─── 1.22  Membrane deformation energy ───────────────────────────────────────
print("  [1.22] Membrane deformation energy approximation...")
# Helfrich energy density: f = κ (2H)² / 2  + κ_G K
# Integrate over surface.  κ in kBT, multiply by kBT for Joules.
kappa_J  = KAPPA_KBT * KB_T_JOULES
dA_elem  = dx_grid * dy_grid  # Å²  → convert to m² later
dA_m2    = dA_elem * 1e-20    # 1 Å² = 1e-20 m²

f_helfrich_density = (kappa_J / 2.0) * (2 * H_mid)**2  # J/m²  (before dA)
E_helfrich_per_cell= f_helfrich_density * dA_m2          # Joules per grid cell
E_total_J          = np.nansum(E_helfrich_per_cell)
E_total_kBT        = E_total_J / KB_T_JOULES

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im = axes[0].imshow(f_helfrich_density * dA_elem,  # in kBT/Å²*dA
                    cmap='inferno', origin='lower',
                    extent=[xmin, xmax, ymin, ymax])
plt.colorbar(im, ax=axes[0], label="Bending Energy Density (J/cell)")
axes[0].set_title("Local Bending Energy Density")
axes[0].set_xlabel("X (Å)"); axes[0].set_ylabel("Y (Å)")
axes[1].hist((f_helfrich_density * dA_m2 / KB_T_JOULES).flatten(),
             bins=80, color='darkorange', edgecolor='black', log=True)
axes[1].set_xlabel("Local Bending Energy (kBT per cell)")
axes[1].set_ylabel("Count (log)")
axes[1].set_title(f"Bending Energy Distribution\nTotal ≈ {E_total_kBT:.1f} kBT")
plt.suptitle("Membrane Deformation Energy (Helfrich Model)", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "20_deformation_energy")

# ─── 1.23  3D displacement ───────────────────────────────────────────────────
print("  [1.23] 3D displacement surface...")
fig = plt.figure(figsize=(14, 10))
ax3d = fig.add_subplot(111, projection='3d')
surf_d = ax3d.plot_surface(grid_x, grid_y, combined_displ,
                            cmap='plasma', linewidth=0, antialiased=True, alpha=0.92)
if len(pep_coms) > 0:
    pz = np.full(len(pep_coms), np.max(combined_displ) * 1.03)
    ax3d.scatter(pep_coms[:, 0], pep_coms[:, 1], pz,
                 c=depth_values, cmap='turbo', edgecolors='black', s=90)
fig.colorbar(surf_d, ax=ax3d, shrink=0.7, pad=0.08, label="Lipid Displacement (Å)")
ax3d.set_title(f"3D Lipid Displacement Surface\n{time_ns:.2f} ns", fontsize=16, pad=20)
ax3d.set_xlabel("X (Å)"); ax3d.set_ylabel("Y (Å)"); ax3d.set_zlabel("Displacement (Å)")
ax3d.view_init(elev=28, azim=42); ax3d.grid(False)
save_fig(fig, "21_3D_displacement", show=True)

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
# S E C T I O N  2  –  TRAJECTORY-AVERAGED ANALYSES
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

print("\n" + "="*70)
print("SECTION 2 – FULL TRAJECTORY ANALYSIS")
print("="*70)

grid_size_traj = 1.0   # Å per cell
box = u.trajectory[0].dimensions[:3]
nx  = int(box[0] / grid_size_traj)
ny  = int(box[1] / grid_size_traj)
x_t = np.linspace(0, box[0], nx)
y_t = np.linspace(0, box[1], ny)
X_t, Y_t = np.meshgrid(x_t, y_t)

sum_upper  = np.zeros((ny, nx))
sum_lower  = np.zeros((ny, nx))
sum_cu     = np.zeros((ny, nx))
sum_cl     = np.zeros((ny, nx))
n_frames   = 0

# time-evolution accumulators
time_arr   = []
thk_mean_t = []
thk_std_t  = []
curv_mean_t= []
pore_area_t= []
disrupt_t  = []
roughness_t= []
APL_t      = []

# peptide COM trajectory
pep_com_traj = []

def compute_height_map_traj(leaflet_ag, nx_, ny_, gs):
    Z = np.zeros((ny_, nx_))
    C = np.zeros((ny_, nx_))
    for pos in leaflet_ag.positions:
        ix = int(pos[0] / gs)
        iy = int(pos[1] / gs)
        if 0 <= ix < nx_ and 0 <= iy < ny_:
            Z[iy, ix] += pos[2]
            C[iy, ix] += 1
    Z[C > 0] /= C[C > 0]
    Z[C == 0] = np.nan
    return Z

def fill_nan_traj(Z):
    xx, yy = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    mask = ~np.isnan(Z)
    pts  = np.column_stack((xx[mask], yy[mask]))
    vals = Z[mask]
    Zi   = griddata(pts, vals, (xx, yy), method='cubic')
    nm   = np.isnan(Zi)
    if np.any(nm):
        Zi[nm] = griddata(pts, vals, (xx[nm], yy[nm]), method='nearest')
    return Zi

def smooth_traj(Z, sigma=3):
    return gaussian_filter(Z, sigma=sigma)

def curvature_func_traj(Z):
    return gaussian_laplace(Z, sigma=2)

def upscale(Z, f=3):
    return zoom(Z, f, order=3)

def robust_norm_traj(F, pct=98):
    fmax = np.percentile(np.abs(F), pct)
    return np.clip(F / fmax, -1, 1) if fmax > 0 else F

print("  Running trajectory loop...")
for ts_loop in tqdm(u.trajectory, desc="  Frames"):
    zm   = np.median(lipids.positions[:, 2])
    up_l = lipids.select_atoms(f"prop z > {zm}")
    lo_l = lipids.select_atoms(f"prop z < {zm}")

    Zu = smooth_traj(fill_nan_traj(compute_height_map_traj(up_l, nx, ny, grid_size_traj)), 2)
    Zl = smooth_traj(fill_nan_traj(compute_height_map_traj(lo_l, nx, ny, grid_size_traj)), 2)

    sum_upper += Zu
    sum_lower += Zl
    sum_cu    += curvature_func_traj(Zu)
    sum_cl    += curvature_func_traj(Zl)

    thk  = Zu - Zl
    curv = (curvature_func_traj(Zu) - curvature_func_traj(Zl)) / 2.0

    thr_p     = np.percentile(thk, 10)
    pore_area = np.sum(thk < thr_p) * grid_size_traj**2
    disp_comb = np.abs(Zu - Zu.mean()) + np.abs(Zl - Zl.mean())

    time_arr.append(ts_loop.time / 1000.0)
    thk_mean_t.append(np.mean(thk))
    thk_std_t.append(np.std(thk))
    curv_mean_t.append(np.mean(np.abs(curv)))
    pore_area_t.append(pore_area)
    disrupt_t.append(np.mean(disp_comb))
    roughness_t.append(np.sqrt(np.mean((thk - np.mean(thk))**2)))
    APL_t.append((box[0] * box[1]) / (len(up_l.residues) + 0.5*len(lo_l.residues) + 1e-6))

    if len(peptides) > 0:
        pep_com_traj.append(peptides.center_of_mass())

    n_frames += 1

pep_com_traj = np.array(pep_com_traj) if pep_com_traj else np.zeros((0, 3))
time_arr     = np.array(time_arr)

print(f"  Frames processed: {n_frames}")

# ── averages & high-res upscale ───────────────────────────────────────────────
Z_upper  = smooth_traj(sum_upper / n_frames, 3)
Z_lower  = smooth_traj(sum_lower / n_frames, 3)
curv_up  = smooth_traj(sum_cu    / n_frames, 3)
curv_low = smooth_traj(sum_cl    / n_frames, 3)

UPS  = 3
Zu_hr = upscale(Z_upper, UPS)
Zl_hr = upscale(Z_lower, UPS)
cu_hr = upscale(curv_up,  UPS)
cl_hr = upscale(curv_low, UPS)
X_hr  = upscale(X_t, UPS)
Y_hr  = upscale(Y_t, UPS)

thickness   = Zu_hr - Zl_hr
curv_bil    = (cu_hr - cl_hr) / 2.0
du  = np.abs(Zu_hr - np.mean(Zu_hr))
dl  = np.abs(Zl_hr - np.mean(Zl_hr))
d_comb      = du + dl
cb_n        = robust_norm_traj(curv_bil, 98)
d_norm      = np.clip(d_comb / (np.percentile(d_comb, 98) + 1e-12), 0, 1)
thickness_n = (thickness - thickness.min()) / (thickness.ptp() + 1e-10)

# ─── 2.1  Bilayer structure ───────────────────────────────────────────────────
print("  [2.1] Bilayer structure (averaged)...")
fig = plt.figure(figsize=(16, 7))
fig.suptitle('Bilayer Structure (Trajectory Average)', fontsize=14, fontweight='bold')
for idx, (Z_hr, cmap, title) in enumerate([
    (Zu_hr, cm.viridis, "Upper Leaflet"),
    (Zl_hr, cm.plasma,  "Lower Leaflet")
], 1):
    ax3d = fig.add_subplot(1, 2, idx, projection='3d')
    zn   = (Z_hr - Z_hr.min()) / (Z_hr.ptp() + 1e-10)
    surf = ax3d.plot_surface(X_hr, Y_hr, Z_hr, facecolors=cmap(zn),
                              shade=False, rcount=100, ccount=100)
    ax3d.set_xlabel('X (Å)'); ax3d.set_ylabel('Y (Å)'); ax3d.set_zlabel('Z (Å)')
    ax3d.set_title(title); ax3d.view_init(elev=25, azim=45)
    ax3d.set_box_aspect([1, 1, 0.6])
plt.tight_layout(); save_fig(fig, "30_bilayer_structure_avg", show=True)

# ─── 2.2  Mean curvature (averaged) ──────────────────────────────────────────
print("  [2.2] Mean curvature (averaged)...")
fig = plt.figure(figsize=(16, 7))
fig.suptitle('Local Curvature – Trajectory Average', fontsize=14, fontweight='bold')
cu_n = robust_norm_traj(cu_hr, 98)
cl_n = robust_norm_traj(cl_hr, 98)
for idx, (Z_hr, Cn, title) in enumerate([
    (Zu_hr, cu_n, "Upper Leaflet"),
    (Zl_hr, cl_n, "Lower Leaflet")
], 1):
    ax3d = fig.add_subplot(1, 2, idx, projection='3d')
    ax3d.plot_surface(X_hr, Y_hr, Z_hr, facecolors=cm.coolwarm((Cn+1)/2),
                      shade=False, rcount=100, ccount=100)
    ax3d.set_xlabel('X (Å)'); ax3d.set_ylabel('Y (Å)'); ax3d.set_zlabel('Z (Å)')
    ax3d.set_title(title); ax3d.view_init(elev=25, azim=45)
    ax3d.set_box_aspect([1, 1, 0.6])
sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=Normalize(vmin=-1, vmax=1))
sm.set_array([]); fig.colorbar(sm, ax=fig.axes, shrink=0.6, label='Norm. Curvature')
plt.tight_layout(); save_fig(fig, "31_curvature_avg", show=True)

# ─── 2.3  Pore detection (averaged) ──────────────────────────────────────────
print("  [2.3] Pore detection (averaged)...")
thr_avg = np.percentile(thickness, 10)
mask_avg = thickness < thr_avg
fig = plt.figure(figsize=(16, 7))
fig.suptitle(f'Pore Detection – Avg (Thickness < {thr_avg:.1f} Å)',
             fontsize=14, fontweight='bold')
for idx, (Z_hr, clr, title) in enumerate([
    (Zu_hr, 'red',  "Upper Leaflet"),
    (Zl_hr, 'blue', "Lower Leaflet")
], 1):
    ax3d = fig.add_subplot(1, 2, idx, projection='3d')
    ax3d.plot_surface(X_hr, Y_hr, Z_hr, color='lightgray', alpha=0.6,
                      rcount=100, ccount=100, edgecolor='none')
    ax3d.scatter(X_hr[mask_avg], Y_hr[mask_avg], Z_hr[mask_avg],
                 c=clr, s=6, alpha=0.9, label='Pore')
    ax3d.set_xlabel('X (Å)'); ax3d.set_ylabel('Y (Å)'); ax3d.set_zlabel('Z (Å)')
    ax3d.set_title(title); ax3d.view_init(elev=25, azim=45)
    ax3d.set_box_aspect([1, 1, 0.6]); ax3d.legend()
plt.tight_layout(); save_fig(fig, "32_pore_detection_avg", show=True)

# ─── 2.4  Disruptivity (averaged) ────────────────────────────────────────────
print("  [2.4] Disruptivity (averaged)...")
fig = plt.figure(figsize=(16, 7))
fig.suptitle('Disruptivity – Trajectory Average', fontsize=14, fontweight='bold')
for idx, Z_hr, title in [(1, Zu_hr, "Upper"), (2, Zl_hr, "Lower")]:
    ax3d = fig.add_subplot(1, 2, idx, projection='3d')
    surf = ax3d.plot_surface(X_hr, Y_hr, Z_hr, facecolors=cm.inferno(d_norm),
                              shade=False, rcount=100, ccount=100)
    ax3d.set_xlabel('X (Å)'); ax3d.set_ylabel('Y (Å)'); ax3d.set_zlabel('Z (Å)')
    ax3d.set_title(title); ax3d.view_init(elev=25, azim=45)
    ax3d.set_box_aspect([1, 1, 0.6])
sm2 = cm.ScalarMappable(cmap=cm.inferno, norm=Normalize(0, 1))
sm2.set_array([]); fig.colorbar(sm2, ax=fig.axes, shrink=0.6, label='Disruptivity (norm.)')
plt.tight_layout(); save_fig(fig, "33_disruptivity_avg", show=True)

# ─── 2.5  Thickness map (averaged 3D) ────────────────────────────────────────
print("  [2.5] Thickness map (averaged 3D)...")
fig = plt.figure(figsize=(16, 7))
fig.suptitle('Bilayer Thickness – Trajectory Average', fontsize=14, fontweight='bold')
ax3d1 = fig.add_subplot(121, projection='3d')
ax3d1.plot_surface(X_hr, Y_hr, Zu_hr, facecolors=cm.RdYlGn(thickness_n),
                    shade=False, rcount=100, ccount=100)
ax3d1.set_xlabel('X (Å)'); ax3d1.set_ylabel('Y (Å)'); ax3d1.set_zlabel('Z (Å)')
ax3d1.set_title('Upper Leaflet (colored by thickness)'); ax3d1.view_init(elev=25, azim=45)
ax3d2 = fig.add_subplot(122, projection='3d')
surf_t = ax3d2.plot_surface(X_hr, Y_hr, thickness, facecolors=cm.RdYlGn(thickness_n),
                              shade=False, rcount=100, ccount=100)
ax3d2.set_xlabel('X (Å)'); ax3d2.set_ylabel('Y (Å)'); ax3d2.set_zlabel('Thickness (Å)')
ax3d2.set_title('Thickness Profile'); ax3d2.view_init(elev=25, azim=45)
sm3 = cm.ScalarMappable(cmap=cm.RdYlGn,
                          norm=Normalize(thickness.min(), thickness.max()))
sm3.set_array([]); fig.colorbar(sm3, ax=fig.axes, shrink=0.6, label='Thickness (Å)')
plt.tight_layout(); save_fig(fig, "34_thickness_avg_3D", show=True)

# ─── 2.6  Multi-angle view ───────────────────────────────────────────────────
print("  [2.6] Multi-angle view...")
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Bilayer Structure: Multi-Angle 3D', fontsize=14, fontweight='bold')
angles = [(25, 45), (25, 135), (60, 45), (10, 45)]
titles_ma = ['Front', 'Side', 'Top-down', 'Oblique']
for i, (elev, azim) in enumerate(angles, 1):
    ax3d = fig.add_subplot(2, 2, i, projection='3d')
    ax3d.plot_surface(X_hr, Y_hr, Zu_hr,
                      facecolors=cm.viridis((Zu_hr-Zu_hr.min())/Zu_hr.ptp()),
                      shade=False, rcount=80, ccount=80)
    ax3d.plot_surface(X_hr, Y_hr, Zl_hr,
                      facecolors=cm.plasma((Zl_hr-Zl_hr.min())/Zl_hr.ptp()),
                      shade=False, rcount=80, ccount=80)
    ax3d.set_xlabel('X (Å)'); ax3d.set_ylabel('Y (Å)'); ax3d.set_zlabel('Z (Å)')
    ax3d.set_title(titles_ma[i-1]); ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_box_aspect([1, 1, 0.6])
plt.tight_layout(); save_fig(fig, "35_multi_angle_avg", show=True)

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
# S E C T I O N  3  –  MEMBRANE BENDING ANALYSIS (2D & 3D)
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

print("\n" + "="*70)
print("SECTION 3 – MEMBRANE BENDING ANALYSIS")
print("="*70)

# Use midplane of averaged trajectory surfaces
mid_avg    = (Z_upper + Z_lower) / 2.0
H_avg, K_avg, k1_avg, k2_avg, nx_avg, ny_avg, nz_avg, tilt_avg = \
    compute_curvatures(mid_avg)

# Spontaneous curvature (H₀ = 0 for symmetric bilayer)
H0    = 0.0
dH    = H_avg - H0
bend_energy_density = KAPPA_KBT * 2 * dH**2   # in kBT/Å²

# ─── 3.1  2D bending analysis ─────────────────────────────────────────────────
print("  [3.1] 2D bending analysis maps...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
titles_b = ["Mean Curvature H (Å⁻¹)",
             "Gaussian Curvature K (Å⁻²)",
             "Principal k₁ (Å⁻¹)",
             "Principal k₂ (Å⁻¹)",
             "Bending Energy Density (kBT/Å²)",
             "Midplane Height Z (Å)"]
maps_b   = [H_avg, K_avg, k1_avg, k2_avg, bend_energy_density, mid_avg]
cmaps_b  = ['RdBu_r', 'seismic', 'bwr', 'bwr', 'hot', 'viridis']
for ax2d, data, title_b, cmap_b in zip(axes.flat, maps_b, titles_b, cmaps_b):
    vb = np.percentile(np.abs(data), 98)
    if 'Height' in title_b or 'Energy' in title_b:
        im = ax2d.imshow(data, cmap=cmap_b, origin='lower',
                         extent=[0, box[0], 0, box[1]])
    else:
        im = ax2d.imshow(data, cmap=cmap_b, origin='lower',
                         extent=[0, box[0], 0, box[1]],
                         vmin=-vb, vmax=vb)
    plt.colorbar(im, ax=ax2d, label=title_b)
    ax2d.set_title(title_b)
    ax2d.set_xlabel("X (Å)"); ax2d.set_ylabel("Y (Å)")
plt.suptitle("Membrane Bending Analysis – 2D Maps", fontsize=15, fontweight='bold')
plt.tight_layout(); save_fig(fig, "40_bending_2D_maps")

# ─── 3.2  3D bending surface ─────────────────────────────────────────────────
print("  [3.2] 3D bending analysis surfaces...")
mid_hr       = upscale(mid_avg, UPS)
H_avg_hr     = upscale(H_avg, UPS)
bend_en_hr   = upscale(bend_energy_density, UPS)

fig = plt.figure(figsize=(18, 7))
fig.suptitle("Membrane Bending Analysis – 3D", fontsize=15, fontweight='bold')

ax_b1 = fig.add_subplot(131, projection='3d')
Hn_hr = robust_norm_traj(H_avg_hr, 98)
ax_b1.plot_surface(X_hr, Y_hr, mid_hr,
                    facecolors=cm.RdBu_r((Hn_hr+1)/2),
                    shade=False, rcount=80, ccount=80)
ax_b1.set_title("Mean Curvature H")
ax_b1.set_xlabel("X (Å)"); ax_b1.set_ylabel("Y (Å)"); ax_b1.set_zlabel("Z (Å)")
ax_b1.view_init(elev=25, azim=45); ax_b1.set_box_aspect([1, 1, 0.6])

ax_b2 = fig.add_subplot(132, projection='3d')
Ben  = bend_en_hr / (bend_en_hr.max() + 1e-12)
ax_b2.plot_surface(X_hr, Y_hr, mid_hr,
                    facecolors=cm.hot(Ben),
                    shade=False, rcount=80, ccount=80)
ax_b2.set_title("Bending Energy Density")
ax_b2.set_xlabel("X (Å)"); ax_b2.set_ylabel("Y (Å)"); ax_b2.set_zlabel("Z (Å)")
ax_b2.view_init(elev=25, azim=45); ax_b2.set_box_aspect([1, 1, 0.6])

ax_b3 = fig.add_subplot(133, projection='3d')
K_avg_hr = upscale(K_avg, UPS)
Kn_hr    = robust_norm_traj(K_avg_hr, 98)
ax_b3.plot_surface(X_hr, Y_hr, mid_hr,
                    facecolors=cm.seismic((Kn_hr+1)/2),
                    shade=False, rcount=80, ccount=80)
ax_b3.set_title("Gaussian Curvature K")
ax_b3.set_xlabel("X (Å)"); ax_b3.set_ylabel("Y (Å)"); ax_b3.set_zlabel("Z (Å)")
ax_b3.view_init(elev=25, azim=45); ax_b3.set_box_aspect([1, 1, 0.6])

for sm_cmap, sm_norm, sm_lbl, ax_sm in [
    (cm.RdBu_r, Normalize(-1, 1), 'Mean Curv (norm.)', ax_b1),
    (cm.hot,    Normalize(0, 1),  'Bend. Energy (norm.)', ax_b2),
    (cm.seismic,Normalize(-1, 1), 'Gaussian Curv (norm.)', ax_b3)
]:
    sm_tmp = cm.ScalarMappable(cmap=sm_cmap, norm=sm_norm)
    sm_tmp.set_array([])
    fig.colorbar(sm_tmp, ax=ax_sm, shrink=0.5, label=sm_lbl)

plt.tight_layout(); save_fig(fig, "41_bending_3D_surfaces", show=True)

# ─── 3.3  Bending spectrum (undulation analysis) ──────────────────────────────
print("  [3.3] Bending undulation spectrum (power spectrum)...")
# Power spectral density of midplane fluctuations
mid_fluct = mid_avg - np.mean(mid_avg)
PS        = np.abs(np.fft.fft2(mid_fluct))**2
PS_shift  = np.fft.fftshift(PS)
freq_x    = np.fft.fftshift(np.fft.fftfreq(nx, d=grid_size_traj))
freq_y    = np.fft.fftshift(np.fft.fftfreq(ny, d=grid_size_traj))
Fq        = np.sqrt(freq_x[np.newaxis, :]**2 + freq_y[:, np.newaxis]**2)

# radial average of power spectrum
q_bins = np.linspace(0, Fq.max()*0.5, 60)
qc     = 0.5 * (q_bins[:-1] + q_bins[1:])
PS_rad = []
for lo, hi in zip(q_bins[:-1], q_bins[1:]):
    mask_q = (Fq >= lo) & (Fq < hi)
    PS_rad.append(np.mean(PS_shift[mask_q]) if mask_q.sum() > 0 else np.nan)
PS_rad = np.array(PS_rad)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im_ps = axes[0].imshow(np.log10(PS_shift + 1), cmap='plasma', origin='lower')
plt.colorbar(im_ps, ax=axes[0], label="log₁₀(Power)")
axes[0].set_title("2D Power Spectrum of Midplane")
axes[0].set_xlabel("q_x (Å⁻¹)"); axes[0].set_ylabel("q_y (Å⁻¹)")

valid = np.isfinite(PS_rad) & (qc > 0)
axes[1].loglog(qc[valid], PS_rad[valid], 'b-o', markersize=3, linewidth=1.5)
# Helfrich theory: <|h_q|²> = kBT / (κ q⁴)  → add theoretical line
q_th  = qc[valid]
PS_th = 1.0 / (KAPPA_KBT * (2*np.pi*q_th)**4 + 1e-30)
PS_th *= PS_rad[valid][PS_rad[valid] > 0][0] / PS_th[0]  # scale to data
axes[1].loglog(q_th, PS_th, 'r--', linewidth=2, label=f'q⁻⁴ (κ={KAPPA_KBT} kBT)')
axes[1].set_xlabel("q (Å⁻¹)"); axes[1].set_ylabel("Power spectral density")
axes[1].set_title("Radial Power Spectrum (Undulation Analysis)")
axes[1].legend()
plt.suptitle("Membrane Bending Undulation Spectrum", fontsize=14, fontweight='bold')
plt.tight_layout(); save_fig(fig, "42_bending_undulation_spectrum")

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
# S E C T I O N  4  –  PEPTIDE ANALYSES
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

print("\n" + "="*70)
print("SECTION 4 – PEPTIDE / PROTEIN ANALYSES")
print("="*70)

if len(pep_com_traj) > 1:
    # ─── 4.1  COM trajectory & diffusion ─────────────────────────────────────
    print("  [4.1] COM trajectory and diffusion...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].plot(time_arr, pep_com_traj[:, 0], 'b-', label='X')
    axes[0].plot(time_arr, pep_com_traj[:, 1], 'r-', label='Y')
    axes[0].plot(time_arr, pep_com_traj[:, 2], 'g-', label='Z')
    axes[0].set_xlabel("Time (ns)"); axes[0].set_ylabel("Position (Å)")
    axes[0].set_title("Peptide COM Trajectory"); axes[0].legend()

    axes[1].plot(pep_com_traj[:, 0], pep_com_traj[:, 1], 'purple', linewidth=0.7)
    axes[1].scatter([pep_com_traj[0, 0]], [pep_com_traj[0, 1]], c='green', s=60,
                    zorder=5, label='Start')
    axes[1].scatter([pep_com_traj[-1, 0]], [pep_com_traj[-1, 1]], c='red', s=60,
                    zorder=5, label='End')
    axes[1].set_xlabel("X (Å)"); axes[1].set_ylabel("Y (Å)")
    axes[1].set_title("XY Trajectory"); axes[1].legend()

    # MSD
    dt   = time_arr[1] - time_arr[0] if len(time_arr) > 1 else 1.0
    lags = np.arange(1, len(pep_com_traj)//2)
    msd  = np.array([np.mean(np.sum((pep_com_traj[lag:] - pep_com_traj[:-lag])**2, axis=1))
                     for lag in lags])
    axes[2].plot(lags * dt, msd, 'darkorange', linewidth=2)
    axes[2].set_xlabel("Lag time (ns)"); axes[2].set_ylabel("MSD (Å²)")
    axes[2].set_title("Mean Square Displacement")
    # linear fit for D
    if len(lags) > 3:
        try:
            coeffs = np.polyfit(lags[:len(lags)//3] * dt,
                                msd[:len(lags)//3], 1)
            D_ns   = coeffs[0] / 4.0   # Å²/ns (2D)
            axes[2].plot(lags[:len(lags)//3]*dt,
                         np.polyval(coeffs, lags[:len(lags)//3]*dt),
                         'k--', label=f"D ≈ {D_ns:.2f} Å²/ns")
            axes[2].legend()
        except Exception:
            pass
    plt.suptitle("Peptide COM Trajectory & Diffusion", fontsize=14, fontweight='bold')
    plt.tight_layout(); save_fig(fig, "50_peptide_COM_diffusion")

    # ─── 4.2  Helix tilt angle ───────────────────────────────────────────────
    print("  [4.2] Helix tilt angle from snapshot...")
    if len(pep_coms) >= 2:
        # Principal axis = eigenvector of inertia tensor
        rel = pep_coms - pep_coms.mean(axis=0)
        I   = np.dot(rel.T, rel)
        evals, evecs = np.linalg.eigh(I)
        helix_axis   = evecs[:, 0]    # smallest eigenvalue = longest axis
        z_hat        = np.array([0, 0, 1])
        cos_tilt     = np.abs(np.dot(helix_axis, z_hat))
        helix_tilt   = np.degrees(np.arccos(np.clip(cos_tilt, -1, 1)))
        if helix_tilt > 90:
            helix_tilt = 180 - helix_tilt
        print(f"  Helix tilt angle = {helix_tilt:.1f}°")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].scatter(pep_coms[:, 0], pep_coms[:, 1], c=np.arange(len(pep_coms)),
                        cmap='rainbow', s=80, edgecolors='black', zorder=5)
        xs = pep_coms.mean(axis=0)[0] + np.linspace(-20, 20, 50) * helix_axis[0]
        ys = pep_coms.mean(axis=0)[1] + np.linspace(-20, 20, 50) * helix_axis[1]
        axes[0].plot(xs, ys, 'k--', linewidth=2, label='Helix axis (XY)')
        axes[0].set_xlabel("X (Å)"); axes[0].set_ylabel("Y (Å)")
        axes[0].set_title(f"Helix Orientation (XY)\nTilt = {helix_tilt:.1f}° from Z")
        axes[0].legend()

        ax3d = fig.add_subplot(1, 2, 2, projection='3d')
        ax3d.scatter(pep_coms[:, 0], pep_coms[:, 1], pep_coms[:, 2],
                     c=np.arange(len(pep_coms)), cmap='rainbow', s=60, edgecolors='black')
        cm_p = pep_coms.mean(axis=0)
        t_p  = np.linspace(-20, 20, 50)
        ax3d.plot(cm_p[0]+t_p*helix_axis[0], cm_p[1]+t_p*helix_axis[1],
                  cm_p[2]+t_p*helix_axis[2], 'k-', linewidth=3, label='Helix axis')
        ax3d.set_xlabel("X (Å)"); ax3d.set_ylabel("Y (Å)"); ax3d.set_zlabel("Z (Å)")
        ax3d.set_title("3D Helix Orientation")
        ax3d.legend()
        plt.suptitle("Peptide Helix Orientation & Tilt Angle", fontsize=14, fontweight='bold')
        plt.tight_layout(); save_fig(fig, "51_helix_tilt_angle", show=True)
    else:
        helix_tilt = np.nan

    # ─── 4.3  Peptide orientation analysis ───────────────────────────────────
    print("  [4.3] Peptide orientation analysis...")
    if len(pep_coms) >= 2:
        # insertion depth along trajectory
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].bar(range(len(depth_values)), depth_values,
                    color=plt.cm.RdYlGn((depth_values - depth_values.min()) /
                                        (depth_values.ptp() + 1e-10)))
        axes[0].axhline(0, color='black', linestyle='--')
        axes[0].set_xlabel("Residue index")
        axes[0].set_ylabel("Insertion depth relative to midplane (Å)")
        axes[0].set_title("Residue Insertion Depth")

        axes[1].scatter(range(len(depth_values)), depth_values,
                        c=depth_values, cmap='turbo', s=80, edgecolors='black')
        axes[1].plot(range(len(depth_values)), depth_values, 'gray', linewidth=1)
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5,
                        label='Midplane')
        axes[1].set_xlabel("Residue index"); axes[1].set_ylabel("Depth (Å)")
        axes[1].set_title("Depth Profile Along Peptide Sequence")
        axes[1].legend()
        plt.suptitle("Peptide Orientation & Insertion Depth", fontsize=14, fontweight='bold')
        plt.tight_layout(); save_fig(fig, "52_peptide_orientation")
else:
    print("  [skip] Peptide COM trajectory too short or no peptide found.")
    helix_tilt = np.nan

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
# S E C T I O N  5  –  TIME EVOLUTION PLOTS
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

print("\n" + "="*70)
print("SECTION 5 – TIME EVOLUTION PLOTS")
print("="*70)

time_arr  = np.array(time_arr)
thk_mean_t= np.array(thk_mean_t)
thk_std_t = np.array(thk_std_t)
curv_mean_t=np.array(curv_mean_t)
pore_area_t=np.array(pore_area_t)
disrupt_t = np.array(disrupt_t)
roughness_t=np.array(roughness_t)
APL_t     = np.array(APL_t)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("Time Evolution of Membrane Properties", fontsize=15, fontweight='bold')

def plot_evo(ax_e, t, y, lbl, clr='steelblue', std=None):
    ax_e.plot(t, y, color=clr, linewidth=1.8)
    if std is not None:
        ax_e.fill_between(t, y-std, y+std, alpha=0.25, color=clr)
    ax_e.set_xlabel("Time (ns)"); ax_e.set_ylabel(lbl)
    ax_e.set_title(lbl + " vs Time"); ax_e.grid(True, alpha=0.3)

plot_evo(axes[0,0], time_arr, thk_mean_t, "Mean Thickness (Å)", std=thk_std_t)
plot_evo(axes[0,1], time_arr, curv_mean_t,"Mean |Curvature| (Å⁻²)", 'darkorange')
plot_evo(axes[0,2], time_arr, pore_area_t,"Pore Area (Å²)", 'tomato')
plot_evo(axes[1,0], time_arr, disrupt_t,  "Mean Disruptivity (Å)", 'purple')
plot_evo(axes[1,1], time_arr, roughness_t,"RMS Roughness (Å)", 'teal')
plot_evo(axes[1,2], time_arr, APL_t,      "Area-per-Lipid (Å²)", 'seagreen')

if len(pep_com_traj) > 1:
    plot_evo(axes[2,0], time_arr, pep_com_traj[:, 0], "Peptide X COM (Å)", 'navy')
    plot_evo(axes[2,1], time_arr, pep_com_traj[:, 1], "Peptide Y COM (Å)", 'darkgreen')
    plot_evo(axes[2,2], time_arr, pep_com_traj[:, 2], "Peptide Z COM (Å)", 'maroon')
else:
    for axi in axes[2]:
        axi.text(0.5, 0.5, 'No peptide data', ha='center', va='center',
                 transform=axi.transAxes)

plt.tight_layout(); save_fig(fig, "60_time_evolution_plots", pdf=True)

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
# S E C T I O N  5B – ADVANCED NEW ANALYSES (v4.0)
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

print("\n" + "="*70)
print("SECTION 5B – ADVANCED ANALYSES (v4.0 NEW)")
print("="*70)

# ─── 5B.1  SCD Time Evolution ─────────────────────────────────────────────────
print("  [5B.1] S_CD time evolution over trajectory...")
scd_time_u = []
scd_time_l = []
time_scd   = []

print("    Computing per-frame SCD (subsampled)...")
subsample_step = max(1, len(u.trajectory) // 50)   # max 50 frames
for ts_scd in tqdm(u.trajectory[::subsample_step], desc="    SCD frames"):
    zm_s = np.median(lipids.positions[:, 2])
    frame_scd = compute_scd_per_lipid(lipids, zm_s)
    if len(frame_scd) == 0:
        continue
    u_vals = [r[3] for r in frame_scd if r[4] == "upper"]
    l_vals = [r[3] for r in frame_scd if r[4] == "lower"]
    scd_time_u.append(np.mean(u_vals) if u_vals else np.nan)
    scd_time_l.append(np.mean(l_vals) if l_vals else np.nan)
    time_scd.append(ts_scd.time / 1000.0)

time_scd   = np.array(time_scd)
scd_time_u = np.array(scd_time_u)
scd_time_l = np.array(scd_time_l)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(time_scd, scd_time_u, 'steelblue', linewidth=2, label='Upper')
axes[0].plot(time_scd, scd_time_l, 'tomato',    linewidth=2, label='Lower')
axes[0].set_xlabel("Time (ns)"); axes[0].set_ylabel("Mean S_CD")
axes[0].set_title("Lipid Order Parameter S_CD vs Time", fontweight='bold')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].axhline(0, color='black', linewidth=1, linestyle=':')

axes[1].plot(time_scd, scd_time_u - scd_time_l, 'purple', linewidth=2)
axes[1].axhline(0, color='gray', linewidth=1, linestyle='--')
axes[1].set_xlabel("Time (ns)")
axes[1].set_ylabel("ΔS_CD (Upper – Lower)")
axes[1].set_title("Leaflet S_CD Asymmetry vs Time", fontweight='bold')
axes[1].grid(True, alpha=0.3)
plt.suptitle("Lipid Order Parameter Time Evolution", fontsize=13, fontweight='bold')
plt.tight_layout(); save_fig(fig, "65_SCD_time_evolution")
print("  ✓ SCD time evolution saved")

# ─── 5B.2  Membrane Thickness Probability Distribution ────────────────────────
print("  [5B.2] Thickness probability distribution...")
thk_flat   = thickness_map.flatten()
thk_all_t  = np.concatenate([np.array(thk_mean_t)])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Snapshot distribution
axes[0].hist(thk_flat[thk_flat > 0], bins=60, density=True,
             color='steelblue', edgecolor='white', linewidth=0.4, alpha=0.85)
from scipy.stats import norm as sp_norm
mu_t, std_t = sp_norm.fit(thk_flat[thk_flat > 0])
x_t = np.linspace(thk_flat.min(), thk_flat.max(), 200)
axes[0].plot(x_t, sp_norm.pdf(x_t, mu_t, std_t), 'r-', linewidth=2.5,
             label=f"Gaussian fit\nμ={mu_t:.1f}, σ={std_t:.1f} Å")
axes[0].axvline(mu_t, color='red', linewidth=1.5, linestyle='--', alpha=0.7)
axes[0].set_xlabel("Membrane Thickness (Å)"); axes[0].set_ylabel("Probability Density")
axes[0].set_title("Thickness Distribution (Snapshot)", fontweight='bold')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

# Time-averaged distribution
axes[1].hist(np.array(thk_mean_t), bins=30, density=True,
             color='seagreen', edgecolor='white', linewidth=0.4, alpha=0.85)
axes[1].set_xlabel("Mean Thickness per Frame (Å)"); axes[1].set_ylabel("Probability Density")
axes[1].set_title("Thickness Distribution Over Trajectory", fontweight='bold')
axes[1].axvline(np.mean(thk_mean_t), color='red', linewidth=2, linestyle='--',
                label=f"Mean={np.mean(thk_mean_t):.1f} Å")
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
plt.suptitle("Membrane Thickness Probability Distribution", fontsize=13, fontweight='bold')
plt.tight_layout(); save_fig(fig, "66_thickness_distribution")

# ─── 5B.3  Radial Distribution Function g(r) of Headgroups ───────────────────
print("  [5B.3] Headgroup radial distribution function g(r)...")

def compute_rdf_2d(coords_a, coords_b, r_max=50.0, n_bins=100, box_xy=None):
    """2D radial distribution function."""
    bins = np.linspace(0, r_max, n_bins + 1)
    r_c  = 0.5 * (bins[:-1] + bins[1:])
    dr   = bins[1] - bins[0]
    counts = np.zeros(n_bins)
    n_a = len(coords_a)
    n_b = len(coords_b)
    if n_a == 0 or n_b == 0:
        return r_c, counts
    dxy = coords_a[:, np.newaxis, :2] - coords_b[np.newaxis, :, :2]
    if box_xy is not None:
        for d in range(2):
            dxy[..., d] -= np.round(dxy[..., d] / box_xy[d]) * box_xy[d]
    dist = np.sqrt((dxy**2).sum(axis=-1)).flatten()
    dist = dist[(dist > 0) & (dist < r_max)]
    counts, _ = np.histogram(dist, bins=bins)
    # normalize to g(r)
    if box_xy is not None:
        area = box_xy[0] * box_xy[1]
    else:
        area = np.ptp(coords_b[:, 0]) * np.ptp(coords_b[:, 1])
    rho_b = n_b / max(area, 1e-6)
    shell_area = 2 * np.pi * r_c * dr
    ideal = rho_b * shell_area * n_a
    g_r = counts / (ideal + 1e-12)
    return r_c, g_r

box_xy_val = u.trajectory[FRAME].dimensions[:2]
r_c_uu, g_uu = compute_rdf_2d(upper_leaflet, upper_leaflet, box_xy=box_xy_val)
r_c_ll, g_ll = compute_rdf_2d(lower_leaflet, lower_leaflet, box_xy=box_xy_val)
r_c_ul, g_ul = compute_rdf_2d(upper_leaflet, lower_leaflet, box_xy=box_xy_val)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(r_c_uu, g_uu, 'steelblue', linewidth=2, label='Upper–Upper')
axes[0].plot(r_c_ll, g_ll, 'tomato',    linewidth=2, label='Lower–Lower')
axes[0].axhline(1, color='gray', linewidth=1, linestyle='--')
axes[0].set_xlabel("r (Å)"); axes[0].set_ylabel("g(r)")
axes[0].set_title("Intra-Leaflet Headgroup g(r)", fontweight='bold')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 50)

axes[1].plot(r_c_ul, g_ul, 'purple', linewidth=2, label='Upper–Lower')
axes[1].axhline(1, color='gray', linewidth=1, linestyle='--')
axes[1].set_xlabel("r (Å)"); axes[1].set_ylabel("g(r)")
axes[1].set_title("Inter-Leaflet Headgroup g(r)", fontweight='bold')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 50)
plt.suptitle("Lipid Headgroup Radial Distribution Functions g(r)",
             fontsize=13, fontweight='bold')
plt.tight_layout(); save_fig(fig, "67_headgroup_RDF")

# ─── 5B.4  Lipid Flip-Flop Detection ─────────────────────────────────────────
print("  [5B.4] Lipid flip-flop detection...")
flipflop_events = []
z_threshold = 5.0    # Å band around midplane to count crossings

# Track each lipid residue's leaflet assignment over time (subsampled)
flipflop_step = max(1, len(u.trajectory) // 80)
resids_all    = list(lipids.residues.resids)
leaf_assign_prev = {}

# initialize at first frame
u.trajectory[0]
z_mid_ff = np.median(lipids.positions[:, 2])
for res in lipids.residues:
    leaf_assign_prev[res.resid] = "upper" if res.atoms.center_of_mass()[2] >= z_mid_ff else "lower"

flip_times = []
n_flip_events = 0

for ts_ff in u.trajectory[::flipflop_step]:
    z_mid_ff = np.median(lipids.positions[:, 2])
    for res in lipids.residues:
        com_z = res.atoms.center_of_mass()[2]
        new_leaf = "upper" if com_z >= z_mid_ff else "lower"
        if (res.resid in leaf_assign_prev and
                new_leaf != leaf_assign_prev[res.resid] and
                abs(com_z - z_mid_ff) < z_threshold * 3):
            n_flip_events += 1
            flip_times.append(ts_ff.time / 1000.0)
        leaf_assign_prev[res.resid] = new_leaf

print(f"  Flip-flop events detected: {n_flip_events}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
if len(flip_times) > 0:
    axes[0].hist(flip_times, bins=30, color='orchid', edgecolor='white',
                 linewidth=0.5, density=True)
    axes[0].set_xlabel("Time (ns)"); axes[0].set_ylabel("Event Rate (ns⁻¹)")
    axes[0].set_title(f"Flip-Flop Events vs Time\nTotal: {n_flip_events}", fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    flip_rate = len(flip_times) / max(time_arr[-1] - time_arr[0], 1e-6)
    axes[0].axhline(flip_rate, color='red', linewidth=1.5, linestyle='--',
                    label=f"Rate ≈ {flip_rate:.2f} ns⁻¹")
    axes[0].legend()
else:
    axes[0].text(0.5, 0.5, f"No flip-flop events\ndetected (n=0)",
                 ha='center', va='center', transform=axes[0].transAxes, fontsize=12)
    axes[0].set_title("Flip-Flop Events vs Time", fontweight='bold')

# Cumulative count
flip_cum = np.cumsum(np.ones(len(flip_times)))
if len(flip_times) > 0:
    axes[1].step(flip_times, flip_cum, 'purple', linewidth=2)
    axes[1].set_xlabel("Time (ns)"); axes[1].set_ylabel("Cumulative Flip-Flops")
    axes[1].set_title("Cumulative Flip-Flop Count", fontweight='bold')
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, "No events", ha='center', va='center',
                 transform=axes[1].transAxes, fontsize=12)
    axes[1].set_title("Cumulative Flip-Flop Count", fontweight='bold')

plt.suptitle("Lipid Flip-Flop Analysis", fontsize=13, fontweight='bold')
plt.tight_layout(); save_fig(fig, "68_flip_flop_detection")

# ─── 5B.5  Membrane Interdigitation ──────────────────────────────────────────
print("  [5B.5] Lipid tail interdigitation analysis...")

def tail_density_z(lipid_ag, z_min, z_max, n_bins=100):
    """1D density of tail atoms along Z."""
    tail_atoms = lipid_ag.select_atoms("name C* S* D*")
    if len(tail_atoms) == 0:
        tail_atoms = lipid_ag
    z_pos = tail_atoms.positions[:, 2]
    counts, edges = np.histogram(z_pos, bins=n_bins, range=(z_min, z_max))
    z_centers = 0.5 * (edges[:-1] + edges[1:])
    return z_centers, counts / (counts.max() + 1e-12)

z_all = lipids.positions[:, 2]
z_min_l, z_max_l = z_all.min(), z_all.max()

zc, rho_u = tail_density_z(
    lipids.select_atoms(f"prop z >= {z_mid}"), z_min_l, z_max_l)
zc, rho_l = tail_density_z(
    lipids.select_atoms(f"prop z <  {z_mid}"), z_min_l, z_max_l)

# interdigitation = overlap integral
overlap = np.trapz(np.minimum(rho_u, rho_l), zc)
IDI = overlap / (np.trapz(rho_u, zc) + np.trapz(rho_l, zc) + 1e-12) * 2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].fill_betweenx(zc, 0, rho_u, alpha=0.5, color='steelblue', label='Upper tails')
axes[0].fill_betweenx(zc, 0, rho_l, alpha=0.5, color='tomato',    label='Lower tails')
axes[0].fill_betweenx(zc, 0, np.minimum(rho_u, rho_l), alpha=0.7,
                      color='orchid', label=f'Overlap (IDI={IDI:.3f})')
axes[0].axhline(z_mid, color='black', linewidth=1.5, linestyle='--', label='Midplane')
axes[0].set_xlabel("Normalised Tail Density"); axes[0].set_ylabel("Z (Å)")
axes[0].set_title("Tail Density Profiles & Interdigitation", fontweight='bold')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, axis='x')

# 2D interdigitation map: local overlap of upper/lower tail densities
def leaflet_tail_density_2D(leaflet_coords, gx, gy):
    H2d, _, _ = np.histogram2d(leaflet_coords[:, 0], leaflet_coords[:, 1],
                                bins=GRID_SIZE,
                                range=[[xmin, xmax], [ymin, ymax]])
    return gaussian_filter(H2d.T, sigma=SMOOTH_SIGMA)

idi_map = np.minimum(dens_u / (dens_u.max() + 1e-12),
                     dens_l / (dens_l.max() + 1e-12))
im = axes[1].imshow(idi_map, cmap='hot', origin='lower',
                    extent=[xmin, xmax, ymin, ymax])
plt.colorbar(im, ax=axes[1], label="Local Interdigitation (a.u.)")
axes[1].set_title(f"2D Interdigitation Map (IDI={IDI:.3f})", fontweight='bold')
axes[1].set_xlabel("X (Å)"); axes[1].set_ylabel("Y (Å)")
plt.suptitle("Lipid Tail Interdigitation Analysis", fontsize=13, fontweight='bold')
plt.tight_layout(); save_fig(fig, "69_interdigitation")

# ─── 5B.6  Thermal Undulation Mode Analysis (Helfrich κ fit) ─────────────────
print("  [5B.6] Thermal undulation mode analysis (Helfrich κ fit)...")

# Use the averaged midplane and fit κ from <|h_q|²> = kBT/(κ q⁴)
mid_avg_for_kappa = (Z_upper + Z_lower) / 2.0
mid_fluct_k = mid_avg_for_kappa - np.mean(mid_avg_for_kappa)
FT2 = np.abs(np.fft.fft2(mid_fluct_k))**2 / mid_fluct_k.size
FT_shift = np.fft.fftshift(FT2)

# q vectors in Å⁻¹
qx_k = np.fft.fftshift(np.fft.fftfreq(mid_fluct_k.shape[1], d=grid_size_traj)) * 2 * np.pi
qy_k = np.fft.fftshift(np.fft.fftfreq(mid_fluct_k.shape[0], d=grid_size_traj)) * 2 * np.pi
Fq_k = np.sqrt(qx_k[np.newaxis, :]**2 + qy_k[:, np.newaxis]**2)

q_bins_k  = np.linspace(0, Fq_k.max() * 0.4, 40)
qc_k      = 0.5 * (q_bins_k[:-1] + q_bins_k[1:])
hq2_rad   = []
for lo, hi in zip(q_bins_k[:-1], q_bins_k[1:]):
    mask_q = (Fq_k >= lo) & (Fq_k < hi) & (Fq_k > 0)
    hq2_rad.append(np.mean(FT_shift[mask_q]) if mask_q.sum() > 0 else np.nan)
hq2_rad = np.array(hq2_rad)

valid_k = np.isfinite(hq2_rad) & (qc_k > 0) & (hq2_rad > 0)

# Fit κ from <|h_q|²> q⁴ = kBT/κ  →  log-log linear fit
if valid_k.sum() >= 4:
    log_q = np.log(qc_k[valid_k])
    log_h = np.log(hq2_rad[valid_k] * qc_k[valid_k]**4)
    # Use middle q-range (avoid small-q finite-size & large-q noise)
    mid_range = (log_q > np.percentile(log_q, 20)) & (log_q < np.percentile(log_q, 75))
    if mid_range.sum() >= 3:
        slope, intercept = np.polyfit(log_q[mid_range], log_h[mid_range], 1)
        kappa_fit_kBT = 1.0 / np.exp(np.mean(log_h[mid_range]))
        print(f"    Fitted κ ≈ {kappa_fit_kBT:.1f} kBT (theoretical: {KAPPA_KBT:.0f} kBT)")
    else:
        kappa_fit_kBT = np.nan
else:
    kappa_fit_kBT = np.nan

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Power spectrum 2D
axes[0].imshow(np.log10(FT_shift + 1e-20), cmap='plasma', origin='lower')
axes[0].set_title("2D Power Spectrum of Midplane (log₁₀)", fontweight='bold')
axes[0].set_xlabel("q_x (grid)"); axes[0].set_ylabel("q_y (grid)")

# Helfrich fit plot
if valid_k.sum() >= 4:
    axes[1].loglog(qc_k[valid_k], hq2_rad[valid_k], 'o', color='steelblue',
                   markersize=5, linewidth=1.5, label='Simulation')
    # theoretical q^-4
    q_fit = qc_k[valid_k]
    PS_th_k = 1.0 / (KAPPA_KBT * q_fit**4 + 1e-30)
    PS_th_k *= (hq2_rad[valid_k][0] * qc_k[valid_k][0]**4) / (KAPPA_KBT * qc_k[valid_k][0]**4 + 1e-30) * KAPPA_KBT
    axes[1].loglog(q_fit, PS_th_k, 'r--', linewidth=2.5,
                   label=f'Helfrich q⁻⁴\n(κ={KAPPA_KBT:.0f} kBT)')
    if not np.isnan(kappa_fit_kBT):
        axes[1].set_title(f"Undulation Spectrum\nFitted κ ≈ {kappa_fit_kBT:.1f} kBT",
                          fontweight='bold')
    else:
        axes[1].set_title("Undulation Spectrum (Helfrich)", fontweight='bold')
    axes[1].set_xlabel("q (Å⁻¹)"); axes[1].set_ylabel("<|h(q)|²>")
    axes[1].legend(fontsize=9); axes[1].grid(True, which='both', alpha=0.2)
else:
    axes[1].text(0.5, 0.5, "Insufficient q-range\nfor Helfrich fit",
                 ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
    kappa_fit_kBT = np.nan

plt.suptitle("Thermal Undulation Modes & Helfrich κ Fit", fontsize=13, fontweight='bold')
plt.tight_layout(); save_fig(fig, "70_undulation_modes_kappa_fit")

# ─── 5B.7  Lateral Pressure / Stress Profile (approximation) ─────────────────
print("  [5B.7] Lateral pressure profile (approximation)...")
# Note: Full lateral pressure requires a stress tensor from MD engine.
# Here we approximate using the "curvature-induced" lateral stress:
# P_L(z) ≈ -κ (2H)² / d  as a proxy for the bending contribution.
# This is a recognised approximation used when stress tensor data is absent.
d_bilayer = np.nanmean(thickness_map)
if d_bilayer < 1:
    d_bilayer = 40.0

z_slices = np.linspace(z_all.min(), z_all.max(), 60)
lat_stress_approx = []

for z_s in z_slices:
    # fraction of atoms at this z slice (±2 Å window)
    frac = np.sum(np.abs(headgroups.positions[:, 2] - z_s) < 2.0) / (len(headgroups) + 1e-6)
    # curvature contribution
    curv_contrib = -KAPPA_KBT * np.nanmean((2 * H_mid)**2) * frac / (d_bilayer + 1e-6)
    lat_stress_approx.append(curv_contrib)

lat_stress_approx = np.array(lat_stress_approx)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(lat_stress_approx, z_slices, 'steelblue', linewidth=2.5)
axes[0].axvline(0, color='gray', linewidth=1, linestyle='--')
axes[0].axhline(z_mid, color='red', linewidth=1.5, linestyle='--',
                alpha=0.7, label='Midplane')
axes[0].set_xlabel("Lateral Stress (kBT/Å³, approx)")
axes[0].set_ylabel("Z (Å)")
axes[0].set_title("Lateral Pressure Profile\n(curvature-based approximation)", fontweight='bold')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis='y')
axes[0].fill_betweenx(z_slices, lat_stress_approx, 0,
                       where=(lat_stress_approx < 0),
                       color='steelblue', alpha=0.3, label='Tension')
axes[0].fill_betweenx(z_slices, lat_stress_approx, 0,
                       where=(lat_stress_approx >= 0),
                       color='tomato', alpha=0.3, label='Compression')

# 2D lateral stress map
lat_stress_2d = -KAPPA_KBT * (2 * H_mid)**2 / (d_bilayer + 1e-6)
im_ls = axes[1].imshow(lat_stress_2d, cmap='coolwarm', origin='lower',
                        extent=[xmin, xmax, ymin, ymax],
                        vmin=np.percentile(lat_stress_2d, 2),
                        vmax=np.percentile(lat_stress_2d, 98))
plt.colorbar(im_ls, ax=axes[1], label="Lat. Stress proxy (kBT/Å³)")
axes[1].set_title("2D Lateral Stress Map (approx)", fontweight='bold')
axes[1].set_xlabel("X (Å)"); axes[1].set_ylabel("Y (Å)")
if len(pep_coms) > 0:
    axes[1].scatter(pep_coms[:, 0], pep_coms[:, 1], c='yellow', s=60,
                    marker='*', edgecolors='black', zorder=10, label="Peptide")
    axes[1].legend(fontsize=8)
plt.suptitle("Lateral Pressure / Stress Profile (Helfrich Approximation)",
             fontsize=13, fontweight='bold')
plt.tight_layout(); save_fig(fig, "71_lateral_pressure_profile")

# ─── 5B.8  Comparative SCD 2D/3D Order Parameter Dashboard ───────────────────
print("  [5B.8] Order parameter comprehensive dashboard...")
fig_op = plt.figure(figsize=(20, 12))
fig_op.patch.set_facecolor('#f5f5f5')
gs_op = fig_op.add_gridspec(2, 3, hspace=0.38, wspace=0.3,
                              left=0.06, right=0.97, top=0.90, bottom=0.06)

def styled_ax(ax_s, title=''):
    ax_s.set_facecolor('white')
    for sp in ax_s.spines.values():
        sp.set_edgecolor('#cccccc')
    ax_s.tick_params(labelsize=8, colors='#333333')
    if title:
        ax_s.set_title(title, fontsize=10, fontweight='bold', color='#222244')
    return ax_s

# row 0: upper leaflet SCD 2D
ax_op0 = styled_ax(fig_op.add_subplot(gs_op[0, 0]), "S_CD – Upper (2D)")
im_op0 = ax_op0.imshow(S_u, cmap='RdYlGn_r', origin='lower',
                        extent=[xmin, xmax, ymin, ymax], vmin=-0.5, vmax=0.5)
fig_op.colorbar(im_op0, ax=ax_op0, label="S_CD", shrink=0.9)
ax_op0.set_xlabel("X (Å)"); ax_op0.set_ylabel("Y (Å)")

# row 0: lower leaflet SCD 2D
ax_op1 = styled_ax(fig_op.add_subplot(gs_op[0, 1]), "S_CD – Lower (2D)")
im_op1 = ax_op1.imshow(S_l, cmap='RdYlGn_r', origin='lower',
                        extent=[xmin, xmax, ymin, ymax], vmin=-0.5, vmax=0.5)
fig_op.colorbar(im_op1, ax=ax_op1, label="S_CD", shrink=0.9)
ax_op1.set_xlabel("X (Å)"); ax_op1.set_ylabel("Y (Å)")

# row 0: difference map
ax_op2 = styled_ax(fig_op.add_subplot(gs_op[0, 2]), "ΔS_CD (Upper – Lower)")
diff_scd = S_u - S_l
vd = np.percentile(np.abs(diff_scd), 98)
im_op2 = ax_op2.imshow(diff_scd, cmap='coolwarm', origin='lower',
                        extent=[xmin, xmax, ymin, ymax], vmin=-vd, vmax=vd)
fig_op.colorbar(im_op2, ax=ax_op2, label="ΔS_CD", shrink=0.9)
ax_op2.set_xlabel("X (Å)"); ax_op2.set_ylabel("Y (Å)")

# row 1: 3D upper SCD
ax_op3 = fig_op.add_subplot(gs_op[1, 0], projection='3d')
ax_op3.set_facecolor('white')
scd_n_u = (np.clip(S_u, -0.5, 0.5) + 0.5)
ax_op3.plot_surface(grid_x, grid_y, upper_surface,
                     facecolors=cm.RdYlGn_r(scd_n_u), shade=True,
                     rcount=60, ccount=60, alpha=0.92)
ax_op3.set_title("S_CD Upper (3D)", fontsize=10, fontweight='bold')
ax_op3.set_xlabel("X"); ax_op3.set_ylabel("Y"); ax_op3.set_zlabel("Z")
ax_op3.view_init(elev=25, azim=50); ax_op3.set_box_aspect([1, 1, 0.5])

# row 1: 3D lower SCD
ax_op4 = fig_op.add_subplot(gs_op[1, 1], projection='3d')
ax_op4.set_facecolor('white')
scd_n_l = (np.clip(S_l, -0.5, 0.5) + 0.5)
ax_op4.plot_surface(grid_x, grid_y, lower_surface,
                     facecolors=cm.RdYlGn_r(scd_n_l), shade=True,
                     rcount=60, ccount=60, alpha=0.92)
ax_op4.set_title("S_CD Lower (3D)", fontsize=10, fontweight='bold')
ax_op4.set_xlabel("X"); ax_op4.set_ylabel("Y"); ax_op4.set_zlabel("Z")
ax_op4.view_init(elev=25, azim=50); ax_op4.set_box_aspect([1, 1, 0.5])

# row 1: SCD time series
ax_op5 = styled_ax(fig_op.add_subplot(gs_op[1, 2]), "S_CD Time Evolution")
if len(time_scd) > 0:
    ax_op5.plot(time_scd, scd_time_u, 'steelblue', linewidth=2, label='Upper')
    ax_op5.plot(time_scd, scd_time_l, 'tomato',    linewidth=2, label='Lower')
    ax_op5.axhline(0, color='gray', linewidth=1, linestyle=':')
    ax_op5.set_xlabel("Time (ns)"); ax_op5.set_ylabel("Mean S_CD")
    ax_op5.legend(fontsize=8); ax_op5.grid(True, alpha=0.3)
else:
    ax_op5.text(0.5, 0.5, "No time data", ha='center', va='center',
                transform=ax_op5.transAxes)

sm_op = cm.ScalarMappable(cmap=cm.RdYlGn_r, norm=Normalize(-0.5, 0.5))
sm_op.set_array([])
fig_op.colorbar(sm_op, ax=[ax_op3, ax_op4], shrink=0.7, label="S_CD", pad=0.04)

fig_op.suptitle("Lipid Order Parameter S_CD – Comprehensive Dashboard",
                fontsize=14, fontweight='bold', color='#111133')
plt.show()   # open interactive display window (allows 3-D rotation)
plt.savefig("output/figures_hires/72_SCD_dashboard.png",
            dpi=200, bbox_inches='tight', facecolor=fig_op.get_facecolor())
plt.savefig("output/figures_hires/72_SCD_dashboard.pdf",
            dpi=200, bbox_inches='tight', facecolor=fig_op.get_facecolor())
plt.close(fig_op)
print("  ✓ SCD comprehensive dashboard saved")

print(f"\n  [5B Summary]")
print(f"  SCD mean upper      : {scd_mean_u:.4f}" if 'scd_mean_u' in dir() else "  SCD mean upper      : N/A")
print(f"  SCD mean lower      : {scd_mean_l:.4f}" if 'scd_mean_l' in dir() else "  SCD mean lower      : N/A")
print(f"  Interdigitation IDI : {IDI:.4f}")
print(f"  Flip-flop events    : {n_flip_events}")
print(f"  Fitted κ (kBT)      : {kappa_fit_kBT:.2f}" if not np.isnan(kappa_fit_kBT) else "  Fitted κ (kBT)      : N/A")

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
# S E C T I O N  6  –  INTERACTIVE 3D (PLOTLY)
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

if HAS_PLOTLY:
    print("\n" + "="*70)
    print("SECTION 6 – INTERACTIVE 3D VISUALIZATIONS (Plotly)")
    print("="*70)

    try:
        # Downsample for browser performance
        step = max(1, UPS * 2)
        Xp   = X_hr[::step, ::step]
        Yp   = Y_hr[::step, ::step]
        Zup  = Zu_hr[::step, ::step]
        Zlp  = Zl_hr[::step, ::step]
        Thkp = thickness[::step, ::step]
        Hmp  = upscale(H_avg, UPS)[::step, ::step]
        BEp  = upscale(bend_energy_density, UPS)[::step, ::step]

        fig_ply = make_subplots(
            rows=2, cols=2,
            specs=[[{'type':'surface'},{'type':'surface'}],
                   [{'type':'surface'},{'type':'surface'}]],
            subplot_titles=["Upper Leaflet",
                             "Lower Leaflet",
                             "Thickness Map",
                             "Bending Energy"]
        )
        for surf_Z, colorscale, title_p, row, col in [
            (Zup,  'Viridis', 'Upper', 1, 1),
            (Zlp,  'Plasma',  'Lower', 1, 2),
            (Thkp, 'RdYlGn', 'Thick', 2, 1),
            (BEp,  'Hot',     'Bend',  2, 2)
        ]:
            fig_ply.add_trace(
                go.Surface(x=Xp, y=Yp, z=surf_Z,
                           colorscale=colorscale, showscale=True,
                           name=title_p, opacity=0.9),
                row=row, col=col
            )
        if len(pep_coms) > 0:
            fig_ply.add_trace(
                go.Scatter3d(x=pep_coms[:, 0], y=pep_coms[:, 1], z=pep_coms[:, 2],
                             mode='markers+lines',
                             marker=dict(size=5, color=depth_values, colorscale='Turbo'),
                             name='Peptide'),
                row=1, col=1
            )
        fig_ply.update_layout(
            title="Interactive Membrane Analysis Dashboard",
            height=900, width=1400
        )
        fig_ply.write_html("output/interactive/membrane_3D_interactive.html")
        print("  ✓ Saved: output/interactive/membrane_3D_interactive.html")

        # Standalone curvature interactive
        fig_c = go.Figure()
        fig_c.add_trace(go.Surface(x=Xp, y=Yp, z=upscale(mid_avg, UPS)[::step, ::step],
                                    surfacecolor=Hmp, colorscale='RdBu',
                                    showscale=True, name='Mean Curvature H'))
        fig_c.update_layout(title="Midplane Colored by Mean Curvature",
                             scene=dict(xaxis_title="X (Å)",
                                        yaxis_title="Y (Å)",
                                        zaxis_title="Z (Å)"))
        fig_c.write_html("output/interactive/curvature_interactive.html")
        print("  ✓ Saved: output/interactive/curvature_interactive.html")

    except Exception as exc:
        print(f"  [Plotly error] {exc}")

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
# S E C T I O N  7  –  STATISTICS REPORT (CSV + EXCEL)
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

print("\n" + "="*70)
print("SECTION 7 – STATISTICAL REPORT")
print("="*70)

stats_dict = {
    # Geometry
    "Mean Thickness (Å)":           float(np.mean(thickness)),
    "Min Thickness (Å)":            float(np.min(thickness)),
    "Max Thickness (Å)":            float(np.max(thickness)),
    "Std Thickness (Å)":            float(np.std(thickness)),
    # Curvature
    "Mean H upper (Å⁻¹)":          float(np.nanmean(H_u)),
    "Mean H lower (Å⁻¹)":          float(np.nanmean(H_l)),
    "Mean K upper (Å⁻²)":          float(np.nanmean(K_u)),
    "Mean K lower (Å⁻²)":          float(np.nanmean(K_l)),
    "Mean k1 upper (Å⁻¹)":         float(np.nanmean(k1_u)),
    "Mean k2 upper (Å⁻¹)":         float(np.nanmean(k2_u)),
    # Roughness
    "Rq Upper (Å)":                 float(Rq_u),
    "Rq Lower (Å)":                 float(Rq_l),
    "Ra Upper (Å)":                 float(Ra_u),
    "Ra Lower (Å)":                 float(Ra_l),
    # Surface area
    "Surface Area Upper (Å²)":      float(SA_u),
    "Surface Area Lower (Å²)":      float(SA_l),
    "Flat Projection Area (Å²)":    float(SA_flat),
    # APL
    "APL Upper (Å²)":               float(APL_u),
    "APL Lower (Å²)":               float(APL_l),
    # Pores
    "Pore Threshold (Å)":           float(thr_pore),
    "Pore Grid Cells":              int(np.sum(pore_mask)),
    "Pore Clusters":                int(n_pore_clusters),
    "Largest Pore (Å²)":           float(cluster_sizes.max()) if len(cluster_sizes) > 0 else 0.0,
    # Asymmetry
    "Asymmetry Index |AI| (Å)":     float(AI),
    "Leaflet Coupling r":           float(coupling_cc),
    # Bending energy
    "Total Bending Energy (kBT)":   float(E_total_kBT),
    "Mean Bend Energy Density":     float(np.nanmean(bend_energy_density)),
    # Peptide
    "Helix Tilt Angle (°)":         float(helix_tilt) if not np.isnan(helix_tilt) else np.nan,
    "Mean Insertion Depth (Å)":     float(np.mean(depth_values)) if len(depth_values) > 0 else np.nan,
    # NEW v4.0 – Lipid Order Parameters
    "Mean S_CD Upper":              float(scd_mean_u) if 'scd_mean_u' in dir() else np.nan,
    "Mean S_CD Lower":              float(scd_mean_l) if 'scd_mean_l' in dir() else np.nan,
    # NEW v4.0 – Interdigitation
    "Interdigitation IDI":          float(IDI),
    # NEW v4.0 – Flip-flop
    "Flip-Flop Events (sampled)":   int(n_flip_events),
    # NEW v4.0 – Undulation
    "Fitted Bending Modulus (kBT)": float(kappa_fit_kBT) if not np.isnan(kappa_fit_kBT) else np.nan,
    # System
    "Frames Analyzed":              int(n_frames),
    "Lipid Count":                  int(len(lipids.residues)),
    "Upper Leaflet Lipids":         int(n_upper),
    "Lower Leaflet Lipids":         int(n_lower),
    "Snapshot Time (ns)":           float(time_ns),
}

# Time evolution table
time_df_data = {
    "Time (ns)":           time_arr,
    "Mean Thickness (Å)":  thk_mean_t,
    "Std Thickness (Å)":   thk_std_t,
    "Mean |Curvature|":    curv_mean_t,
    "Pore Area (Å²)":      pore_area_t,
    "Disruptivity (Å)":    disrupt_t,
    "RMS Roughness (Å)":   roughness_t,
    "APL (Å²)":            APL_t,
}

# CSV export
import csv
with open("output/stats/statistics_summary.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Parameter", "Value"])
    for k, v in stats_dict.items():
        writer.writerow([k, f"{v:.4f}" if isinstance(v, float) else v])
print("  ✓ Saved: output/stats/statistics_summary.csv")

with open("output/stats/time_evolution.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(time_df_data.keys()))
    writer.writeheader()
    for i in range(len(time_arr)):
        writer.writerow({k: v[i] for k, v in time_df_data.items()})
print("  ✓ Saved: output/stats/time_evolution.csv")

# Excel export
if HAS_PANDAS:
    try:
        df_summary = pd.DataFrame(list(stats_dict.items()), columns=["Parameter", "Value"])
        df_time    = pd.DataFrame(time_df_data)
        with pd.ExcelWriter("output/stats/membrane_analysis.xlsx", engine='openpyxl') as writer_xl:
            df_summary.to_excel(writer_xl, sheet_name="Summary Statistics", index=False)
            df_time.to_excel(writer_xl, sheet_name="Time Evolution", index=False)
        print("  ✓ Saved: output/stats/membrane_analysis.xlsx")
    except Exception as exc:
        print(f"  [Excel error] {exc}")

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
# S E C T I O N  8  –  PUBLICATION-READY SUMMARY PDF
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

print("\n" + "="*70)
print("SECTION 8 – SUMMARY PDF")
print("="*70)

pdf_path = "output/pdf/membrane_analysis_summary.pdf"

with PdfPages(pdf_path) as pdf:

    # ── Cover page ─────────────────────────────────────────────────────────────
    fig_cv = plt.figure(figsize=(8.27, 11.69))
    fig_cv.patch.set_facecolor('navy')
    ax_cv  = fig_cv.add_axes([0, 0, 1, 1])
    ax_cv.set_facecolor('navy'); ax_cv.axis('off')
    ax_cv.text(0.5, 0.75, "MEMBRANE ANALYSIS REPORT",
               ha='center', va='center', fontsize=26, fontweight='bold',
               color='white', transform=ax_cv.transAxes)
    ax_cv.text(0.5, 0.62, "Ultra-Comprehensive CG Bilayer Analysis",
               ha='center', va='center', fontsize=16, color='lightblue',
               transform=ax_cv.transAxes)
    ax_cv.text(0.5, 0.52, f"Trajectory: {XTC_FILE}",
               ha='center', va='center', fontsize=12, color='white',
               transform=ax_cv.transAxes)
    ax_cv.text(0.5, 0.46, f"Lipids: {LIPID_RESNAMES}   |   Frames: {n_frames}",
               ha='center', va='center', fontsize=12, color='white',
               transform=ax_cv.transAxes)
    ax_cv.text(0.5, 0.40, f"Snapshot: {time_ns:.2f} ns",
               ha='center', va='center', fontsize=12, color='white',
               transform=ax_cv.transAxes)
    pdf.savefig(fig_cv); plt.close(fig_cv)

    # ── Statistics table ───────────────────────────────────────────────────────
    fig_tb = plt.figure(figsize=(8.27, 11.69))
    ax_tb  = fig_tb.add_subplot(111)
    ax_tb.axis('off')
    ax_tb.set_title("Summary Statistics", fontsize=14, fontweight='bold', pad=20)
    rows_tb = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)]
               for k, v in stats_dict.items()]
    tbl = ax_tb.table(cellText=rows_tb, colLabels=["Parameter", "Value"],
                      cellLoc='left', loc='center',
                      colWidths=[0.65, 0.35])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1f4e79'); cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#d9e1f2')
    pdf.savefig(fig_tb); plt.close(fig_tb)

    # ── Key figures ────────────────────────────────────────────────────────────
    import glob
    key_figs = sorted(glob.glob("output/figures_hires/*.png"))
    for fp in key_figs:
        try:
            img = plt.imread(fp)
            fig_img = plt.figure(figsize=(11.69, 8.27))
            ax_img  = fig_img.add_subplot(111)
            ax_img.imshow(img); ax_img.axis('off')
            ax_img.set_title(os.path.basename(fp).replace('.png', '').replace('_', ' '),
                              fontsize=11)
            pdf.savefig(fig_img, bbox_inches='tight'); plt.close(fig_img)
        except Exception:
            pass

    d = pdf.infodict()
    d['Title']   = 'Membrane Analysis Report'
    d['Author']  = 'curvature-combined-ultra.py'

print(f"  ✓ Saved: {pdf_path}")

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
# S E C T I O N  9  –  COMPREHENSIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================

print("\n" + "="*70)
print("SECTION 9 – COMPREHENSIVE DASHBOARD")
print("="*70)

fig_dash = plt.figure(figsize=(22, 28))
fig_dash.patch.set_facecolor('#0a0a1a')
gs       = fig_dash.add_gridspec(5, 4, hspace=0.38, wspace=0.32,
                                  left=0.05, right=0.97,
                                  top=0.94, bottom=0.03)

def dark_ax(ax_d):
    ax_d.set_facecolor('#111122')
    for sp in ax_d.spines.values():
        sp.set_edgecolor('#444466')
    ax_d.tick_params(colors='#ccccdd', labelsize=7)
    ax_d.xaxis.label.set_color('#ccccdd')
    ax_d.yaxis.label.set_color('#ccccdd')
    ax_d.title.set_color('#eeeeff')
    return ax_d

# row 0 – thickness + curvature 2D
ax0 = dark_ax(fig_dash.add_subplot(gs[0, 0]))
im0 = ax0.imshow(thickness_map, cmap='RdYlGn', origin='lower',
                  extent=[xmin, xmax, ymin, ymax])
fig_dash.colorbar(im0, ax=ax0, label='Thickness (Å)')
ax0.set_title("Thickness (snapshot)")

ax1 = dark_ax(fig_dash.add_subplot(gs[0, 1]))
vH  = np.percentile(np.abs(H_mid), 98)
im1 = ax1.imshow(H_mid, cmap='RdBu_r', origin='lower',
                  extent=[xmin, xmax, ymin, ymax], vmin=-vH, vmax=vH)
fig_dash.colorbar(im1, ax=ax1, label='H (Å⁻¹)')
ax1.set_title("Mean Curvature H")

ax2 = dark_ax(fig_dash.add_subplot(gs[0, 2]))
vK  = np.percentile(np.abs(K_mid), 98)
im2 = ax2.imshow(K_mid, cmap='seismic', origin='lower',
                  extent=[xmin, xmax, ymin, ymax], vmin=-vK, vmax=vK)
fig_dash.colorbar(im2, ax=ax2, label='K (Å⁻²)')
ax2.set_title("Gaussian Curvature K")

ax3 = dark_ax(fig_dash.add_subplot(gs[0, 3]))
im3 = ax3.imshow(tilt_m, cmap='hot', origin='lower',
                  extent=[xmin, xmax, ymin, ymax])
fig_dash.colorbar(im3, ax=ax3, label='Tilt (°)')
ax3.set_title("Surface Tilt Angle")

# row 1 – roughness / density / order / compressibility
ax4 = dark_ax(fig_dash.add_subplot(gs[1, 0]))
im4 = ax4.imshow(local_rq_u, cmap='magma', origin='lower',
                  extent=[xmin, xmax, ymin, ymax])
fig_dash.colorbar(im4, ax=ax4, label='Local Rq (Å)')
ax4.set_title("Roughness (Upper)")

ax5 = dark_ax(fig_dash.add_subplot(gs[1, 1]))
im5 = ax5.imshow(dens_u, cmap='YlOrRd', origin='lower',
                  extent=[xmin, xmax, ymin, ymax])
fig_dash.colorbar(im5, ax=ax5, label='Density')
ax5.set_title("Lipid Density (Upper)")

ax6 = dark_ax(fig_dash.add_subplot(gs[1, 2]))
im6 = ax6.imshow(S_u, cmap='RdYlGn_r', origin='lower',
                  extent=[xmin, xmax, ymin, ymax], vmin=-0.5, vmax=0.5)
fig_dash.colorbar(im6, ax=ax6, label='S_CD')
scd_lbl = f"TRUE S_CD" if len(scd_data) > 0 else "S_CD (approx)"
ax6.set_title(f"{scd_lbl} (Upper)")

ax7 = dark_ax(fig_dash.add_subplot(gs[1, 3]))
im7 = ax7.imshow(comp_map, cmap='viridis', origin='lower',
                  extent=[xmin, xmax, ymin, ymax])
fig_dash.colorbar(im7, ax=ax7, label='Thickness σ (Å)')
ax7.set_title("Compressibility Proxy")

# row 2 – asymmetry / pores / bending energy / coupling
ax8 = dark_ax(fig_dash.add_subplot(gs[2, 0]))
im8 = ax8.imshow(asymmetry_map, cmap='coolwarm', origin='lower',
                  extent=[xmin, xmax, ymin, ymax], vmin=-vA, vmax=vA)
fig_dash.colorbar(im8, ax=ax8, label='Asymmetry (Å)')
ax8.set_title(f"Asymmetry |AI|={AI:.2f} Å")

ax9 = dark_ax(fig_dash.add_subplot(gs[2, 1]))
ax9.imshow(labeled_pores, cmap='tab20', origin='lower',
           extent=[xmin, xmax, ymin, ymax])
ax9.set_title(f"Pore Clusters (n={n_pore_clusters})")

ax10 = dark_ax(fig_dash.add_subplot(gs[2, 2]))
im10 = ax10.imshow(bend_energy_density * dA_elem,
                    cmap='inferno', origin='lower',
                    extent=[0, box[0], 0, box[1]])
fig_dash.colorbar(im10, ax=ax10, label='kBT/cell')
ax10.set_title(f"Bend Energy (tot={E_total_kBT:.1f} kBT)")

ax11 = dark_ax(fig_dash.add_subplot(gs[2, 3]))
im11 = ax11.imshow(cross_corr, cmap='RdBu_r', origin='lower',
                    extent=[xmin, xmax, ymin, ymax])
fig_dash.colorbar(im11, ax=ax11, label='Cross-corr')
ax11.set_title(f"Leaflet Coupling (r={coupling_cc:.3f})")

# row 3 – time evolution
ax12 = dark_ax(fig_dash.add_subplot(gs[3, :2]))
ax12.plot(time_arr, thk_mean_t, 'lime', linewidth=1.5)
ax12.fill_between(time_arr, thk_mean_t-thk_std_t,
                  thk_mean_t+thk_std_t, alpha=0.2, color='lime')
ax12.set_xlabel("Time (ns)"); ax12.set_ylabel("Thickness (Å)")
ax12.set_title("Thickness vs Time"); ax12.grid(True, alpha=0.2)

ax13 = dark_ax(fig_dash.add_subplot(gs[3, 2]))
ax13.plot(time_arr, roughness_t, 'cyan', linewidth=1.5)
ax13.set_xlabel("Time (ns)"); ax13.set_ylabel("Rq (Å)")
ax13.set_title("RMS Roughness"); ax13.grid(True, alpha=0.2)

ax14 = dark_ax(fig_dash.add_subplot(gs[3, 3]))
ax14.plot(time_arr, pore_area_t, 'tomato', linewidth=1.5)
ax14.set_xlabel("Time (ns)"); ax14.set_ylabel("Pore Area (Å²)")
ax14.set_title("Pore Area"); ax14.grid(True, alpha=0.2)

# row 4 – peptide + radial profiles + power spectrum
if len(pep_com_traj) > 1:
    ax15 = dark_ax(fig_dash.add_subplot(gs[4, 0]))
    ax15.plot(pep_com_traj[:, 0], pep_com_traj[:, 1],
              color='magenta', linewidth=0.8, alpha=0.7)
    ax15.scatter([pep_com_traj[0, 0]], [pep_com_traj[0, 1]],
                 c='lime', s=40, zorder=5)
    ax15.scatter([pep_com_traj[-1, 0]], [pep_com_traj[-1, 1]],
                 c='red', s=40, zorder=5)
    ax15.set_xlabel("X (Å)"); ax15.set_ylabel("Y (Å)")
    ax15.set_title("Peptide XY Trajectory")
else:
    ax15 = dark_ax(fig_dash.add_subplot(gs[4, 0]))
    ax15.text(0.5, 0.5, 'No peptide', ha='center', va='center',
              transform=ax15.transAxes, color='white')

ax16 = dark_ax(fig_dash.add_subplot(gs[4, 1]))
ax16.plot(bin_centers, thk_m, 'limegreen', linewidth=2)
ax16.fill_between(bin_centers, thk_m-thk_s, thk_m+thk_s, alpha=0.2, color='lime')
ax16.set_xlabel("Distance from Peptide (Å)")
ax16.set_ylabel("Thickness (Å)")
ax16.set_title("Thickness vs Distance")
ax16.grid(True, alpha=0.2)

ax17 = dark_ax(fig_dash.add_subplot(gs[4, 2]))
ax17.plot(bin_centers, H_m, 'coral', linewidth=2)
ax17.axhline(0, color='white', linestyle='--', linewidth=1)
ax17.set_xlabel("Distance from Peptide (Å)")
ax17.set_ylabel("Mean Curvature")
ax17.set_title("Curvature vs Distance")
ax17.grid(True, alpha=0.2)

ax18 = dark_ax(fig_dash.add_subplot(gs[4, 3]))
if len(cluster_sizes) > 0:
    ax18.hist(cluster_sizes, bins=max(3, len(cluster_sizes)//2 + 1),
              color='gold', edgecolor='black')
    ax18.set_xlabel("Pore Area (Å²)"); ax18.set_ylabel("Count")
    ax18.set_title("Pore Size Distribution")
else:
    ax18.text(0.5, 0.5, 'No pores', ha='center', va='center',
              transform=ax18.transAxes, color='white')

fig_dash.suptitle("COMPREHENSIVE MEMBRANE ANALYSIS DASHBOARD",
                   fontsize=18, fontweight='bold', color='white', y=0.97)

fig_dash.savefig("output/figures_hires/dashboard.png",
                  dpi=200, bbox_inches='tight',
                  facecolor=fig_dash.get_facecolor())
fig_dash.savefig("output/figures_hires/dashboard.pdf",
                  dpi=200, bbox_inches='tight',
                  facecolor=fig_dash.get_facecolor())
plt.close(fig_dash)
print("  ✓ Saved: output/figures_hires/dashboard.png")
print("  ✓ Saved: output/figures_hires/dashboard.pdf")

# =============================================================================
# FINAL SUMMARY PRINTOUT
# =============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE – STATISTICS SUMMARY")
print("="*70)
print(f"\n  Frames analyzed      : {n_frames}")
print(f"  Snapshot time        : {time_ns:.2f} ns")
print(f"\n  --- GEOMETRY ---")
print(f"  Mean thickness       : {stats_dict['Mean Thickness (Å)']:.2f} Å")
print(f"  Min / Max thickness  : {stats_dict['Min Thickness (Å)']:.2f} / {stats_dict['Max Thickness (Å)']:.2f} Å")
print(f"  APL upper / lower    : {APL_u:.2f} / {APL_l:.2f} Å²")
print(f"  Surface area upper   : {SA_u:.1f} Å²")
print(f"  Surface area lower   : {SA_l:.1f} Å²")
print(f"\n  --- CURVATURE ---")
print(f"  Mean H upper         : {np.nanmean(H_u):.4e} Å⁻¹")
print(f"  Mean H lower         : {np.nanmean(H_l):.4e} Å⁻¹")
print(f"  Mean K upper         : {np.nanmean(K_u):.4e} Å⁻²")
print(f"  Mean K lower         : {np.nanmean(K_l):.4e} Å⁻²")
print(f"\n  --- ROUGHNESS ---")
print(f"  Rq upper / lower     : {Rq_u:.3f} / {Rq_l:.3f} Å")
print(f"\n  --- PORES ---")
print(f"  Pore threshold       : {thr_pore:.2f} Å")
print(f"  Pore clusters        : {n_pore_clusters}")
print(f"\n  --- BILAYER PROPERTIES ---")
print(f"  Asymmetry |AI|       : {AI:.3f} Å")
print(f"  Leaflet coupling r   : {coupling_cc:.4f}")
print(f"  Total bending energy : {E_total_kBT:.2f} kBT")
if not np.isnan(helix_tilt):
    print(f"  Helix tilt angle     : {helix_tilt:.1f}°")
print(f"\n  --- ORDER PARAMETERS (v4.0) ---")
if 'scd_mean_u' in dir() and not np.isnan(scd_mean_u):
    print(f"  Mean S_CD upper      : {scd_mean_u:.4f}")
    print(f"  Mean S_CD lower      : {scd_mean_l:.4f}")
print(f"  Interdigitation IDI  : {IDI:.4f}")
print(f"  Flip-flop events     : {n_flip_events}")
if not np.isnan(kappa_fit_kBT):
    print(f"  Fitted κ             : {kappa_fit_kBT:.1f} kBT")
print(f"\n  Output directories:")
print(f"    Figures  → output/figures_hires/")
print(f"    Stats    → output/stats/")
print(f"    PDF      → output/pdf/")
print(f"    Plotly   → output/interactive/")
print("="*70)
