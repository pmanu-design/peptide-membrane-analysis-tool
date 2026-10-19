#!/usr/bin/env python3
"""
Advanced Membrane Surface Analysis - v36
=========================================

Professional membrane surface / curvature / thickness / pore / water-permeation
analysis from MD trajectories, with a desktop GUI control panel.

Improvements over the previous version:
  - Fixed a quadratic O(N^2) python loop in water-density calculation -> now
    vectorized with np.histogram2d (orders of magnitude faster on big systems).
  - All outputs are written to a user-selectable output directory instead of
    the current working directory.
  - Robust error handling: missing files, empty selections, single-frame
    trajectories, and selection typos no longer crash the whole run -- they
    produce a clear message in the log panel.
  - Centralized, leveled logging (INFO / WARNING / ERROR) instead of bare
    print() calls, mirrored both to stdout and to the GUI log console.
  - New Tkinter GUI ("control panel") to pick input files, lipid/water
    selections, grid resolution, smoothing, choose which figure groups to
    generate, and run the whole pipeline in a background thread with a
    progress bar -- so the window never freezes.
  - The original command-line / scripted usage still works unchanged
    (see `run_headless()` at the bottom), for users who don't want the GUI.
  - Minor numerical robustness fixes (guard against all-NaN grids, divide by
    zero in curvature normalization, single-frame trajectories, etc).

Author: improved & refactored version
"""

import os
import sys
import time
import queue
import logging
import threading
import traceback
from datetime import datetime

import numpy as np

# Matplotlib must use a non-interactive backend when running inside threads
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, distance_transform_edt

import warnings
warnings.filterwarnings("ignore")

try:
    import MDAnalysis as mda
    MDANALYSIS_AVAILABLE = True
except ImportError:
    MDANALYSIS_AVAILABLE = False

# ----------------------------------------------------------------------------
# Publication-quality matplotlib defaults
# ----------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Sans',   # Arial may not exist on all systems
    'axes.linewidth': 1.5,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'grid.linewidth': 0.8,
    'figure.dpi': 150,      # screen-friendly; savefig.dpi controls file output
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 6,
})


# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
def make_logger(name="membrane_analysis"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOG = make_logger()


class QueueLogHandler(logging.Handler):
    """A logging handler that pushes formatted records into a thread-safe
    queue, so the Tkinter main loop can poll it and update the GUI safely."""

    def __init__(self, q: "queue.Queue"):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            msg = self.format(record)
            self.q.put(msg)
        except Exception:
            pass


class AnalysisError(Exception):
    """Raised for expected, user-facing analysis failures."""
    pass


# ----------------------------------------------------------------------------
# Core analyzer
# ----------------------------------------------------------------------------
class ProfessionalMembraneSurfaceAnalyzer:
    """Professional membrane analyzer: surfaces, curvature, thickness,
    pores, water permeation, and publication-quality 2D/3D visualizations."""

    def __init__(self, tpr, xtc, lipids, water_mol='WAT', grid_points=200,
                 smoothing_sigma=2, output_dir='.', logger=None,
                 progress_cb=None):
        """
        Parameters
        ----------
        tpr, xtc : str
            Topology and trajectory file paths.
        lipids : list[str]
            Lipid residue names defining the membrane (e.g. ['POPE', 'POPG']).
        water_mol : str
            Water residue name.
        grid_points : int
            Resolution of the interpolated surface grid (NxN).
        smoothing_sigma : float
            Gaussian smoothing sigma applied to interpolated surfaces.
        output_dir : str
            Directory where all figures/files are written.
        logger : logging.Logger, optional
            Logger to use; defaults to the module-level LOG.
        progress_cb : callable(str, float), optional
            Callback invoked as progress_cb(stage_name, fraction_0_to_1).
        """
        if not MDANALYSIS_AVAILABLE:
            raise AnalysisError(
                "MDAnalysis is not installed. Install it with "
                "'pip install MDAnalysis' before running this analysis."
            )

        self.log = logger or LOG
        self.progress_cb = progress_cb or (lambda stage, frac: None)
        self.output_dir = output_dir or '.'
        os.makedirs(self.output_dir, exist_ok=True)

        self.log.info("=" * 80)
        self.log.info("PROFESSIONAL MEMBRANE SURFACE ANALYSIS WITH DYNAMICS")
        self.log.info("=" * 80)

        if not os.path.isfile(tpr):
            raise AnalysisError(f"Topology file not found: {tpr}")
        if not os.path.isfile(xtc):
            raise AnalysisError(f"Trajectory file not found: {xtc}")
        if not lipids:
            raise AnalysisError("No lipid residue names were provided.")

        self.log.info("Loading system...")
        try:
            self.u = mda.Universe(tpr, xtc)
        except Exception as exc:
            raise AnalysisError(f"Failed to load MDAnalysis Universe: {exc}") from exc

        self.lipids = " ".join(lipids)
        self.lipid_atoms = self.u.select_atoms(f"resname {self.lipids}")
        if len(self.lipid_atoms) == 0:
            raise AnalysisError(
                f"No atoms matched resname(s) '{self.lipids}'. Check the "
                f"lipid residue names against your topology."
            )

        self.water = self.u.select_atoms(f"resname {water_mol}")
        if len(self.water) == 0:
            self.log.warning(f"No atoms matched water resname '{water_mol}'. "
                              f"Water permeation analysis will be skipped.")

        self.protein = self.u.select_atoms("protein")

        self.grid_points = int(grid_points)
        self.smooth_sigma = float(smoothing_sigma)
        self.ts = self.u.trajectory[-1]

        # Time-series storage
        self.curvature_over_time = []
        self.thickness_over_time = []
        self.deflection_over_time = []
        self.radius_over_time = []
        self.time_points = []

        self.log.info(f"✓ Loaded {len(self.u.atoms)} atoms, "
                       f"{len(self.u.trajectory)} frames")
        self.log.info(f"✓ Lipid atoms selected: {len(self.lipid_atoms)} "
                       f"(resname {self.lipids})")

    def _save_fig(self, filename):
        """Helper: returns full output path and logs the save."""
        path = os.path.join(self.output_dir, filename)
        return path

    # ------------------------------------------------------------------
    # Core surface extraction / interpolation
    # ------------------------------------------------------------------
    def extract_surface_data(self):
        """Extract membrane surface coordinates (per-residue COM)."""
        coords = self.lipid_atoms.center_of_mass(compound="residues")
        if coords.shape[0] < 4:
            raise AnalysisError(
                "Too few lipid residues found to build a surface "
                "(need at least 4)."
            )

        self.x = coords[:, 0]
        self.y = coords[:, 1]
        self.z = coords[:, 2]

        mid = np.median(self.z)
        self.upper = coords[self.z > mid]
        self.lower = coords[self.z <= mid]

        if len(self.upper) < 4 or len(self.lower) < 4:
            raise AnalysisError(
                "Could not separate upper/lower leaflets reliably "
                "(too few points on one side of the membrane midplane)."
            )

        self.log.info(f"✓ Upper leaflet points: {len(self.upper)}")
        self.log.info(f"✓ Lower leaflet points: {len(self.lower)}")

    def create_surface_grid(self):
        """Create interpolated surface grids for whole/upper/lower leaflets."""
        xi = np.linspace(self.x.min(), self.x.max(), self.grid_points)
        yi = np.linspace(self.y.min(), self.y.max(), self.grid_points)
        self.xi, self.yi = np.meshgrid(xi, yi)

        self.zi = griddata((self.x, self.y), self.z, (self.xi, self.yi), method='cubic')
        self.zu = griddata((self.upper[:, 0], self.upper[:, 1]), self.upper[:, 2],
                            (self.xi, self.yi), method='cubic')
        self.zl = griddata((self.lower[:, 0], self.lower[:, 1]), self.lower[:, 2],
                            (self.xi, self.yi), method='cubic')

        for name in ('zi', 'zu', 'zl'):
            arr = getattr(self, name)
            if np.all(np.isnan(arr)):
                raise AnalysisError(
                    f"Surface interpolation for '{name}' produced only NaNs; "
                    f"check that the lipid selection spans a continuous patch."
                )
            arr[np.isnan(arr)] = np.nanmean(arr)
            setattr(self, name, arr)

        self.zi = gaussian_filter(self.zi, self.smooth_sigma)
        self.zu = gaussian_filter(self.zu, self.smooth_sigma)
        self.zl = gaussian_filter(self.zl, self.smooth_sigma)

        self.thickness = self.zu - self.zl

    def calculate_curvature(self):
        """Calculate mean-curvature-like Laplacian maps for full/upper/lower."""
        def laplacian(z):
            dzdx, dzdy = np.gradient(z)
            d2zdx2, _ = np.gradient(dzdx)
            _, d2zdy2 = np.gradient(dzdy)
            return d2zdx2 + d2zdy2

        self.curvature = laplacian(self.zi)
        max_abs = np.max(np.abs(self.curvature))
        self.curvature_norm = self.curvature / max_abs if max_abs > 1e-12 else np.zeros_like(self.curvature)

        self.curvature_u = laplacian(self.zu)
        self.curvature_l = laplacian(self.zl)

        self.calc_radius_of_curvature()

    def calc_radius_of_curvature(self):
        """Calculate radius of curvature from curvature values (in grid units)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            self.radius_of_curvature = np.where(
                np.abs(self.curvature) > 1e-6,
                1.0 / np.abs(self.curvature),
                np.nan
            )
        self.radius_of_curvature = np.nan_to_num(self.radius_of_curvature, nan=1000.0)

    def calculate_lipid_deflection(self):
        """Calculate phospholipid deflection relative to the local mid-plane."""
        z_avg = (self.zu + self.zl) / 2
        self.deflection_u = self.zu - z_avg
        self.deflection_l = self.zl - z_avg
        self.deflection_total = np.abs(self.deflection_u) + np.abs(self.deflection_l)

        self.log.info(f"✓ Upper leaflet max deflection: {np.max(np.abs(self.deflection_u)):.3f} Å")
        self.log.info(f"✓ Lower leaflet max deflection: {np.max(np.abs(self.deflection_l)):.3f} Å")

    def calculate_electrostatic_potential(self):
        """Heuristic electrostatic-potential-like map combining curvature and
        radial distance from the membrane patch centroid (Gaussian weighting).
        NOTE: this is a geometric proxy, not a true Poisson-Boltzmann solve."""
        center_x = np.mean(self.x)
        center_y = np.mean(self.y)
        dist_from_center = np.sqrt((self.xi - center_x) ** 2 + (self.yi - center_y) ** 2)
        denom = max(np.max(dist_from_center) / 2, 1e-12)
        self.potential_map = -np.abs(self.curvature_norm) * np.exp(-dist_from_center / denom)

    def analyze_pore_radius(self, threshold=0.5):
        """Detect thin (pore-like) regions and estimate maximum pore radius."""
        std = self.thickness.std()
        if std < 1e-12:
            self.pore_mask = np.zeros_like(self.thickness, dtype=bool)
        else:
            thickness_norm = (self.thickness - self.thickness.mean()) / std
            self.pore_mask = thickness_norm < -threshold

        self.pore_thickness = np.where(self.pore_mask, self.thickness, np.nan)

        if np.any(self.pore_mask):
            distance = distance_transform_edt(self.pore_mask)
            grid_spacing = np.mean([
                np.abs(self.xi[1, 0] - self.xi[0, 0]),
                np.abs(self.yi[0, 1] - self.yi[0, 0])
            ])
            self.max_pore_radius = np.max(distance) * grid_spacing
            self.pore_radius_map = distance * grid_spacing
        else:
            self.max_pore_radius = 0.0
            self.pore_radius_map = np.zeros_like(self.pore_mask, dtype=float)

        self.log.info(f"✓ Maximum pore radius: {self.max_pore_radius:.2f} Å")

    def analyze_water_permeation(self, z_min=None, z_max=None):
        """Analyze water permeation through the membrane region.

        Uses a vectorized 2D histogram instead of a per-cell python loop
        (this was previously O(grid_points^2 * n_waters), now O(n_waters))."""
        if len(self.water) == 0:
            self.log.warning("⚠ No water molecules found")
            self.water_influx = 0
            self.water_density = np.zeros_like(self.zi)
            return

        water_coords = self.water.positions

        if z_min is None:
            z_min = self.zl.min() - 5
        if z_max is None:
            z_max = self.zu.max() + 5

        in_membrane = (water_coords[:, 2] > z_min) & (water_coords[:, 2] < z_max)
        water_in_membrane_xyz = water_coords[in_membrane]

        self.water_influx = len(water_in_membrane_xyz)
        self.log.info(f"✓ Water molecules in membrane region: {self.water_influx}")

        water_density = np.zeros_like(self.zi)
        if len(water_in_membrane_xyz) > 0:
            x_edges = np.linspace(self.xi.min(), self.xi.max(), self.grid_points + 1)
            y_edges = np.linspace(self.yi.min(), self.yi.max(), self.grid_points + 1)
            hist, _, _ = np.histogram2d(
                water_in_membrane_xyz[:, 0], water_in_membrane_xyz[:, 1],
                bins=[x_edges, y_edges]
            )
            # histogram2d returns (x_bins, y_bins); our grids are meshgrid(xi,yi)
            # with shape (grid_points, grid_points) indexed [row=y, col=x] via
            # np.meshgrid default 'xy' indexing -> transpose to match.
            water_density = hist.T

        self.water_density = gaussian_filter(water_density, self.smooth_sigma)

    # ------------------------------------------------------------------
    # Time series
    # ------------------------------------------------------------------
    def collect_time_series_data(self, max_frames=20):
        """Collect curvature/thickness/deflection data across the trajectory."""
        self.log.info("\n" + "=" * 80)
        self.log.info("COLLECTING TIME-SERIES DATA")
        self.log.info("=" * 80)

        n_frames = len(self.u.trajectory)
        stride = max(1, n_frames // max_frames)
        frames = list(self.u.trajectory[::stride])
        n_to_process = len(frames)
        self.log.info(f"Processing {n_to_process} of {n_frames} frames (stride={stride})...")

        for frame_idx, ts in enumerate(frames):
            try:
                self.extract_surface_data()
                self.create_surface_grid()
                self.calculate_curvature()
                self.calculate_lipid_deflection()
            except AnalysisError as exc:
                self.log.warning(f"  ⚠ Skipping frame at t={ts.time:.1f} ps: {exc}")
                continue

            self.time_points.append(ts.time)
            self.curvature_over_time.append(np.mean(np.abs(self.curvature)))
            self.thickness_over_time.append(np.mean(self.thickness))
            self.deflection_over_time.append(np.mean(self.deflection_total))

            rad_curv_clean = np.nan_to_num(self.radius_of_curvature, nan=1000.0)
            valid = rad_curv_clean[rad_curv_clean < 500]
            self.radius_over_time.append(np.nanmean(valid) if valid.size else np.nan)

            self.progress_cb("Time-series", (frame_idx + 1) / max(1, n_to_process))
            if (frame_idx + 1) % max(1, n_to_process // 5) == 0:
                self.log.info(f"  ✓ Processed frame {frame_idx + 1}/{n_to_process}")

        if not self.time_points:
            raise AnalysisError("Time-series collection produced no usable frames.")

        self.time_points = np.array(self.time_points)
        self.curvature_over_time = np.array(self.curvature_over_time)
        self.thickness_over_time = np.array(self.thickness_over_time)
        self.deflection_over_time = np.array(self.deflection_over_time)
        self.radius_over_time = np.array(self.radius_over_time)

        # Re-run a final single-frame analysis (on the last loaded frame) so
        # downstream "snapshot" plots use consistent, freshly computed grids.
        self.u.trajectory[-1]
        self.extract_surface_data()
        self.create_surface_grid()
        self.calculate_curvature()
        self.calculate_lipid_deflection()
        self.analyze_pore_radius()
        self.analyze_water_permeation()

    # ==================== A: RADIUS OF CURVATURE TIME-SERIES (2D & 3D) ====================
    def plot_radius_of_curvature_timeseries_enhanced(self):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        time_us = self.time_points / 1000

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time_us, self.radius_over_time, 'b-', linewidth=2.5, label='Upper leaflet')
        ax1.fill_between(time_us, self.radius_over_time, alpha=0.3, color='blue')

        mean_radius = np.nanmean(self.radius_over_time)
        ax1.axhline(y=mean_radius, color='r', linestyle='--', linewidth=2.5,
                    label=f'Mean: {mean_radius:.2f} nm')

        ax1.set_xlabel('Time (μs)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Radius of curvature (nm)', fontsize=13, fontweight='bold')
        ax1.set_title('A: Radius of Curvature Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12, loc='upper right', framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        ymax = np.nanmax(self.radius_over_time) if np.isfinite(np.nanmax(self.radius_over_time)) else 1
        ax1.set_ylim([0, ymax * 1.1 if ymax > 0 else 1])

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.radius_over_time[~np.isnan(self.radius_over_time)], bins=30,
                 color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(mean_radius, color='r', linestyle='--', linewidth=2.5, label=f'Mean: {mean_radius:.2f}')
        ax2.axvline(np.nanmedian(self.radius_over_time), color='g', linestyle=':', linewidth=2.5,
                    label=f'Median: {np.nanmedian(self.radius_over_time):.2f}')
        ax2.set_xlabel('Radius of curvature (nm)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution', fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.2, axis='y')

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        stats_text = (
            f"RADIUS OF CURVATURE ANALYSIS\n\n"
            f"Mean: {mean_radius:.2f} nm\n"
            f"Median: {np.nanmedian(self.radius_over_time):.2f} nm\n"
            f"Std Dev: {np.nanstd(self.radius_over_time):.2f} nm\n"
            f"Min: {np.nanmin(self.radius_over_time):.2f} nm\n"
            f"Max: {np.nanmax(self.radius_over_time):.2f} nm\n\n"
            f"Time Range: {time_us[0]:.2f} - {time_us[-1]:.2f} μs\n"
            f"Total Frames: {len(self.time_points)}\n"
        )
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2))

        plt.suptitle('A: Radius of Curvature Time-Series Analysis', fontsize=16, fontweight='bold', y=0.995)
        path = self._save_fig('A_radius_curvature_timeseries.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        self.log.info(f"✓ Saved: {path}")
        plt.close(fig)

    def plot_radius_of_curvature_3d_interactive(self):
        time_us = self.time_points / 1000
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_us, y=self.radius_over_time, mode='lines+markers',
            name='Radius of curvature', line=dict(color='blue', width=3),
            marker=dict(size=6, color='blue'), fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)',
            hovertemplate='Time: %{x:.2f} μs<br>Radius: %{y:.2f} nm<extra></extra>'
        ))
        mean_radius = np.nanmean(self.radius_over_time)
        fig.add_hline(y=mean_radius, line_dash='dash', line_color='red',
                      annotation_text=f'Mean: {mean_radius:.2f} nm', annotation_position='right')
        fig.update_layout(
            title=dict(text='<b>A: 3D Radius of Curvature Time-Series</b>', x=0.5, xanchor='center', font=dict(size=18)),
            xaxis_title='Time (μs)', yaxis_title='Radius of Curvature (nm)',
            width=1200, height=600, hovermode='closest', font=dict(family='Arial', size=12),
            template='plotly_white'
        )
        path = self._save_fig('A_radius_curvature_3d_interactive.html')
        fig.write_html(path)
        self.log.info(f"✓ Saved: {path}")

    # ==================== C: ELECTROSTATIC POTENTIAL / HEATMAP (2D & 3D) ====================
    def plot_electrostatic_potential_2d(self):
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        self.calculate_electrostatic_potential()

        ax1 = fig.add_subplot(gs[0, :])
        im = ax1.imshow(self.potential_map, cmap='RdBu_r', origin='lower',
                        extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()], aspect='auto')
        contours = ax1.contour(self.xi, self.yi, self.potential_map, levels=12,
                               colors='white', alpha=0.3, linewidths=0.8)
        ax1.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        plt.colorbar(im, ax=ax1, label='Potential (a.u.)', pad=0.02)
        ax1.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
        ax1.set_title('C: Electrostatic Potential Map', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.1)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.potential_map.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(self.potential_map), color='r', linestyle='--', linewidth=2.5)
        ax2.set_xlabel('Potential (a.u.)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.2, axis='y')

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        stats_text = (
            f"ELECTROSTATIC POTENTIAL\n\n"
            f"Mean: {np.mean(self.potential_map):.3f}\n"
            f"Median: {np.median(self.potential_map):.3f}\n"
            f"Std Dev: {np.std(self.potential_map):.3f}\n"
            f"Min: {np.min(self.potential_map):.3f}\n"
            f"Max: {np.max(self.potential_map):.3f}\n\n"
            f"Combines:\n- Membrane curvature\n- Distance from center\n- Gaussian weighting\n"
        )
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=2))

        plt.suptitle('C: Electrostatic Potential / Heatmap Analysis', fontsize=16, fontweight='bold', y=0.995)
        path = self._save_fig('C_electrostatic_potential_2d.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        self.log.info(f"✓ Saved: {path}")
        plt.close(fig)

    def plot_electrostatic_potential_3d(self):
        self.calculate_electrostatic_potential()
        fig = go.Figure(data=[go.Surface(
            x=self.xi, y=self.yi, z=self.potential_map, colorscale='RdBu', showscale=True,
            colorbar=dict(title='Potential (a.u.)', thickness=20),
            contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True))), opacity=0.95
        )])
        fig.update_layout(
            title=dict(text='<b>C: 3D Electrostatic Potential Surface</b>', x=0.5, xanchor='center', font=dict(size=18)),
            scene=dict(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Potential (a.u.)',
                      camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))),
            width=1000, height=800, font=dict(family='Arial', size=12)
        )
        path = self._save_fig('C_electrostatic_potential_3d.html')
        fig.write_html(path)
        self.log.info(f"✓ Saved: {path}")

    # ==================== D: 3D STRUCTURE WITH COLORED LEAFLETS (2D & 3D) ====================
    def plot_3d_structure_with_leaflets_2d(self):
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

        def panel(ax, pts, color, edge, label, title):
            ax.set_facecolor('white')
            span = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
            circle_bg = plt.Circle((np.mean(pts[:, 0]), np.mean(pts[:, 1])), span / 2 + 5,
                                   color='lightgray', alpha=0.2, zorder=0)
            ax.add_patch(circle_bg)
            ax.scatter(pts[:, 0], pts[:, 1], s=250, c=color, alpha=0.75,
                      edgecolors=edge, linewidth=1.5, zorder=2)
            ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=edge, linewidth=1.5))
            ax.set_xlabel('X (Å)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Y (Å)', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(False)
            ax.set_xlim([pts[:, 0].min() - 10, pts[:, 0].max() + 10])
            ax.set_ylim([pts[:, 1].min() - 10, pts[:, 1].max() + 10])
            ax.set_aspect('equal')

        panel(fig.add_subplot(gs[0, 0]), self.upper, 'dodgerblue', 'darkblue', '● G1', 'Upper Leaflet (Extracellular)')
        panel(fig.add_subplot(gs[0, 1]), self.upper[::2], 'dodgerblue', 'darkblue', '● G2', 'Upper Leaflet (Rotated View)')
        panel(fig.add_subplot(gs[1, 0]), self.lower, 'orange', 'darkorange', '● H1', 'Lower Leaflet (Cytoplasm)')
        panel(fig.add_subplot(gs[1, 1]), self.lower[::2], 'orange', 'darkorange', '● H2', 'Lower Leaflet (Rotated View)')

        fig.suptitle('D: Membrane Leaflets - Top-Down View\n'
                    'Blue (G1, G2) = Upper Leaflet | Orange (H1, H2) = Lower Leaflet',
                    fontsize=16, fontweight='bold', y=0.98)
        path = self._save_fig('D_3d_structure_leaflets_2d.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        self.log.info(f"✓ Saved: {path}")
        plt.close(fig)

    def plot_membrane_side_view_3d(self):
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

        # Panel 1: X-Z view
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        y_central = np.linspace(self.yi.min(), self.yi.max(), 20)
        z_upper_profile = np.mean(self.zu, axis=0)
        z_lower_profile = np.mean(self.zl, axis=0)
        x_profile = self.xi[0, :]

        x_3d = np.tile(x_profile, (len(y_central), 1))
        y_3d = np.tile(y_central[:, np.newaxis], (1, len(x_profile)))
        z_upper_3d = np.tile(z_upper_profile, (len(y_central), 1))
        z_lower_3d = np.tile(z_lower_profile, (len(y_central), 1))

        ax1.plot_surface(x_3d, y_3d, z_upper_3d, alpha=0.3, color='lightblue', edgecolor='none')
        ax1.plot_surface(x_3d, y_3d, z_lower_3d, alpha=0.3, color='lightyellow', edgecolor='none')
        ax1.scatter(self.upper[:, 0], self.upper[:, 1], self.upper[:, 2],
                   c='dodgerblue', s=100, alpha=0.65, edgecolors='darkblue', linewidth=0.8, label='Upper leaflet')
        ax1.scatter(self.lower[:, 0], self.lower[:, 1], self.lower[:, 2],
                   c='orange', s=100, alpha=0.65, edgecolors='darkorange', linewidth=0.8, label='Lower leaflet')
        ax1.set_xlabel('X (Å)', fontweight='bold', fontsize=10)
        ax1.set_ylabel('Y (Å)', fontweight='bold', fontsize=10)
        ax1.set_zlabel('Z (Å)', fontweight='bold', fontsize=10)
        ax1.set_title('Side View 1 (X-Z Plane)', fontsize=12, fontweight='bold')
        ax1.view_init(elev=20, azim=0)
        ax1.legend(loc='upper right', fontsize=9)

        # Panel 2: Y-Z view
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        x_central = np.linspace(self.xi.min(), self.xi.max(), 20)
        z_upper_profile_yz = np.mean(self.zu, axis=1)
        z_lower_profile_yz = np.mean(self.zl, axis=1)
        y_profile = self.yi[:, 0]

        x_3d_yz = np.tile(x_central[:, np.newaxis], (1, len(y_profile)))
        y_3d_yz = np.tile(y_profile, (len(x_central), 1))
        z_upper_3d_yz = np.tile(z_upper_profile_yz, (len(x_central), 1))
        z_lower_3d_yz = np.tile(z_lower_profile_yz, (len(x_central), 1))

        ax2.plot_surface(x_3d_yz, y_3d_yz, z_upper_3d_yz, alpha=0.3, color='lightblue', edgecolor='none')
        ax2.plot_surface(x_3d_yz, y_3d_yz, z_lower_3d_yz, alpha=0.3, color='lightyellow', edgecolor='none')
        ax2.scatter(self.upper[:, 0], self.upper[:, 1], self.upper[:, 2],
                   c='dodgerblue', s=100, alpha=0.65, edgecolors='darkblue', linewidth=0.8, label='Upper leaflet')
        ax2.scatter(self.lower[:, 0], self.lower[:, 1], self.lower[:, 2],
                   c='orange', s=100, alpha=0.65, edgecolors='darkorange', linewidth=0.8, label='Lower leaflet')
        ax2.set_xlabel('X (Å)', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Y (Å)', fontweight='bold', fontsize=10)
        ax2.set_zlabel('Z (Å)', fontweight='bold', fontsize=10)
        ax2.set_title('Side View 2 (Y-Z Plane)', fontsize=12, fontweight='bold')
        ax2.view_init(elev=20, azim=90)
        ax2.legend(loc='upper right', fontsize=9)

        fig.suptitle('E: Membrane Side-View Analysis\nBlue = Upper Leaflet | Orange = Lower Leaflet',
                    fontsize=15, fontweight='bold', y=0.98)
        path = self._save_fig('E_membrane_side_view_3d.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        self.log.info(f"✓ Saved: {path}")
        plt.close(fig)

    def plot_3d_structure_with_leaflets_3d(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=self.upper[:, 0], y=self.upper[:, 1], z=self.upper[:, 2], mode='markers',
            marker=dict(size=4, color='dodgerblue', opacity=0.7), name='Upper leaflet (extracellular)',
            hovertemplate='<b>Upper Leaflet</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter3d(
            x=self.lower[:, 0], y=self.lower[:, 1], z=self.lower[:, 2], mode='markers',
            marker=dict(size=4, color='orange', opacity=0.7), name='Lower leaflet (cytoplasm)',
            hovertemplate='<b>Lower Leaflet</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
        ))
        fig.add_trace(go.Surface(x=self.xi, y=self.yi, z=self.zu, colorscale='Blues',
                                 showscale=False, opacity=0.2, name='Upper surface', hoverinfo='skip'))
        fig.add_trace(go.Surface(x=self.xi, y=self.yi, z=self.zl, colorscale='Oranges',
                                 showscale=False, opacity=0.2, name='Lower surface', hoverinfo='skip'))
        fig.update_layout(
            title=dict(text='<b>D: 3D Membrane Structure - Both Leaflets</b><br>'
                           '<sub>Blue = Upper leaflet (extracellular) | Orange = Lower leaflet (cytoplasm)</sub>',
                      x=0.5, xanchor='center', font=dict(size=16)),
            scene=dict(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)',
                      camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)), aspectmode='data'),
            width=1200, height=900, showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='black', borderwidth=1),
            font=dict(family='Arial', size=12), hovermode='closest'
        )
        path = self._save_fig('D_3d_structure_leaflets_3d.html')
        fig.write_html(path)
        self.log.info(f"✓ Saved: {path}")

    def plot_3d_structure_with_heatmap_overlay(self):
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                            subplot_titles=('<b>Curvature Heatmap</b>', '<b>Electrostatic Potential</b>'),
                            horizontal_spacing=0.12)
        self.calculate_electrostatic_potential()
        fig.add_trace(go.Surface(x=self.xi, y=self.yi, z=self.zu, surfacecolor=self.curvature_norm,
                                 colorscale='RdBu', showscale=True,
                                 colorbar=dict(title='Curvature', x=0.46, len=0.6), opacity=0.9), row=1, col=1)
        fig.add_trace(go.Surface(x=self.xi, y=self.yi, z=self.zu, surfacecolor=self.potential_map,
                                 colorscale='RdYlGn', showscale=True,
                                 colorbar=dict(title='Potential', x=1.02, len=0.6), opacity=0.9), row=1, col=2)
        fig.update_scenes(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)',
                          camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)))
        fig.update_layout(
            title=dict(text='<b>D: 3D Heatmap Overlays on Membrane Structure</b>', x=0.5, xanchor='center', font=dict(size=18)),
            width=1400, height=700, font=dict(family='Arial', size=12)
        )
        path = self._save_fig('D_3d_structure_heatmap_overlay.html')
        fig.write_html(path)
        self.log.info(f"✓ Saved: {path}")

    # ==================== STANDARD PLOTS ====================
    def plot_2d_standard_all(self):
        analyses = [
            (self.thickness, 'RdYlGn_r', 'Thickness', 'thickness_2d_analysis.png'),
            (self.curvature_norm, 'RdBu_r', 'Curvature', 'curvature_2d_analysis.png'),
            (self.water_density, 'Blues', 'Water Density', 'water_2d_analysis.png'),
        ]

        for data, cmap, title, filename in analyses:
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

            ax0 = fig.add_subplot(gs[0, :])
            im = ax0.imshow(data, cmap=cmap, origin='lower',
                           extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()])
            contours = ax0.contour(self.xi, self.yi, data, levels=12, colors='white', alpha=0.2)
            ax0.clabel(contours, inline=True, fontsize=8)
            plt.colorbar(im, ax=ax0, label=title, pad=0.02)
            ax0.set_xlabel('X (Å)', fontweight='bold')
            ax0.set_ylabel('Y (Å)', fontweight='bold')
            ax0.set_title(f'{title} Heatmap', fontweight='bold', fontsize=13)
            ax0.grid(True, alpha=0.1)

            ax1 = fig.add_subplot(gs[1, 0])
            ax1.hist(data.flatten(), bins=60, color='steelblue', edgecolor='black', alpha=0.7)
            ax1.axvline(np.mean(data), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(data):.2f}')
            ax1.set_xlabel(title, fontweight='bold')
            ax1.set_ylabel('Frequency', fontweight='bold')
            ax1.set_title('Distribution', fontweight='bold')
            ax1.legend(fontsize=10, framealpha=0.9)
            ax1.grid(True, alpha=0.2, axis='y')

            ax2 = fig.add_subplot(gs[1, 1])
            ax2.axis('off')
            stats_text = (
                f"{title.upper()} STATISTICS\n\n"
                f"Mean: {np.mean(data):.3f}\n"
                f"Median: {np.median(data):.3f}\n"
                f"Std Dev: {np.std(data):.3f}\n"
                f"Min: {np.min(data):.3f}\n"
                f"Max: {np.max(data):.3f}"
            )
            ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=11,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))

            plt.suptitle(f'{title} Analysis', fontsize=16, fontweight='bold', y=0.995)
            path = self._save_fig(filename)
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            self.log.info(f"✓ Saved: {path}")
            plt.close(fig)

    def plot_3d_standard_all(self):
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                            subplot_titles=('<b>Upper Leaflet</b>', '<b>Lower Leaflet</b>'),
                            horizontal_spacing=0.1)
        fig.add_trace(go.Surface(x=self.xi, y=self.yi, z=self.zu, colorscale='Blues', showscale=False, opacity=0.95), row=1, col=1)
        fig.add_trace(go.Surface(x=self.xi, y=self.yi, z=self.zl, colorscale='Oranges', showscale=False, opacity=0.95), row=1, col=2)
        fig.update_layout(
            title=dict(text='<b>3D Membrane Leaflet Surfaces</b>', x=0.5, xanchor='center', font=dict(size=18)),
            width=1500, height=700, font=dict(family='Arial', size=12)
        )
        path = self._save_fig('thickness_3d_analysis.html')
        fig.write_html(path)
        self.log.info(f"✓ Saved: {path}")

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                            subplot_titles=('<b>Upper Curvature</b>', '<b>Lower Curvature</b>'),
                            horizontal_spacing=0.1)
        curv_min = min(self.curvature_u.min(), self.curvature_l.min())
        curv_max = max(self.curvature_u.max(), self.curvature_l.max())
        fig.add_trace(go.Surface(x=self.xi, y=self.yi, z=self.zu, surfacecolor=self.curvature_u,
                                 colorscale='RdBu', showscale=True, colorbar=dict(title='Curvature', len=0.6, x=0.46),
                                 cmin=curv_min, cmax=curv_max, opacity=0.95), row=1, col=1)
        fig.add_trace(go.Surface(x=self.xi, y=self.yi, z=self.zl, surfacecolor=self.curvature_l,
                                 colorscale='RdBu', showscale=False, cmin=curv_min, cmax=curv_max, opacity=0.95), row=1, col=2)
        fig.update_layout(
            title=dict(text='<b>3D Membrane Curvature</b>', x=0.5, xanchor='center', font=dict(size=18)),
            width=1500, height=700, font=dict(family='Arial', size=12)
        )
        path = self._save_fig('curvature_3d_analysis.html')
        fig.write_html(path)
        self.log.info(f"✓ Saved: {path}")

    # ==================== MAIN EXECUTION ====================
    def run_analysis(self, groups=None):
        """Execute complete analysis.

        Parameters
        ----------
        groups : set[str] or None
            Which figure groups to generate. Options:
            {'A', 'C', 'D', 'E', 'standard'}. None means all.
        """
        all_groups = {'A', 'C', 'D', 'E', 'standard'}
        groups = set(groups) if groups else all_groups

        self.progress_cb("Starting", 0.0)
        self.extract_surface_data()
        self.create_surface_grid()
        self.calculate_curvature()
        self.calculate_lipid_deflection()
        self.analyze_pore_radius()
        self.analyze_water_permeation()

        self.collect_time_series_data()
        self.progress_cb("Time-series complete", 0.3)

        if 'A' in groups:
            self.log.info("\n" + "=" * 80)
            self.log.info("GENERATING A: RADIUS OF CURVATURE TIME-SERIES (2D & 3D)")
            self.log.info("=" * 80)
            self.plot_radius_of_curvature_timeseries_enhanced()
            self.plot_radius_of_curvature_3d_interactive()
        self.progress_cb("Group A complete", 0.45)

        if 'C' in groups:
            self.log.info("\n" + "=" * 80)
            self.log.info("GENERATING C: ELECTROSTATIC POTENTIAL / HEATMAP (2D & 3D)")
            self.log.info("=" * 80)
            self.plot_electrostatic_potential_2d()
            self.plot_electrostatic_potential_3d()
        self.progress_cb("Group C complete", 0.6)

        if 'D' in groups:
            self.log.info("\n" + "=" * 80)
            self.log.info("GENERATING D: 3D STRUCTURE WITH COLORED LEAFLETS (2D & 3D)")
            self.log.info("=" * 80)
            self.plot_3d_structure_with_leaflets_2d()
            self.plot_3d_structure_with_leaflets_3d()
            self.plot_3d_structure_with_heatmap_overlay()
        self.progress_cb("Group D complete", 0.75)

        if 'E' in groups:
            self.log.info("\n" + "=" * 80)
            self.log.info("GENERATING E: MEMBRANE SIDE-VIEW ANALYSIS")
            self.log.info("=" * 80)
            self.plot_membrane_side_view_3d()
        self.progress_cb("Group E complete", 0.85)

        if 'standard' in groups:
            self.log.info("\n" + "=" * 80)
            self.log.info("GENERATING STANDARD 2D/3D ANALYSIS PLOTS")
            self.log.info("=" * 80)
            self.plot_2d_standard_all()
            self.plot_3d_standard_all()
        self.progress_cb("Done", 1.0)

        self.log.info("\n" + "=" * 80)
        self.log.info("✓ COMPLETE ANALYSIS FINISHED!")
        self.log.info(f"  Output directory: {os.path.abspath(self.output_dir)}")
        self.log.info("=" * 80 + "\n")


# ----------------------------------------------------------------------------
# Headless / scripted entry point (no GUI)
# ----------------------------------------------------------------------------
def run_headless(tpr, xtc, lipids, water_mol='WAT', grid_points=200,
                  smoothing_sigma=2, output_dir='.', groups=None):
    analyzer = ProfessionalMembraneSurfaceAnalyzer(
        tpr, xtc, lipids, water_mol=water_mol, grid_points=grid_points,
        smoothing_sigma=smoothing_sigma, output_dir=output_dir
    )
    analyzer.run_analysis(groups=groups)
    return analyzer


# ----------------------------------------------------------------------------
# GUI Control Panel (Tkinter)
# ----------------------------------------------------------------------------
def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext

    class MembraneAnalysisGUI(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Membrane Surface Analysis - Control Panel")
            self.geometry("880x720")
            self.minsize(820, 650)

            self.log_queue: "queue.Queue[str]" = queue.Queue()
            self.worker_thread = None
            self.running = False

            self._build_style()
            self._build_widgets()
            self.after(150, self._poll_log_queue)  # full impl defined later in class

        # ---------------- style ----------------
        def _build_style(self):
            style = ttk.Style(self)
            try:
                style.theme_use('clam')
            except tk.TclError:
                pass
            style.configure('Header.TLabel', font=('TkDefaultFont', 14, 'bold'))
            style.configure('Section.TLabelframe.Label', font=('TkDefaultFont', 10, 'bold'))
            style.configure('Run.TButton', font=('TkDefaultFont', 11, 'bold'))

        # ---------------- widgets ----------------
        def _build_widgets(self):
            header = ttk.Label(self, text="🧬 Membrane Surface Analysis", style='Header.TLabel')
            header.pack(anchor='w', padx=15, pady=(12, 4))
            sub = ttk.Label(self, text="Configure inputs, choose figures, and run the full pipeline.")
            sub.pack(anchor='w', padx=15, pady=(0, 10))

            container = ttk.Frame(self)
            container.pack(fill='both', expand=True, padx=15, pady=5)
            container.columnconfigure(0, weight=1)
            container.columnconfigure(1, weight=1)

            # ---- Files panel ----
            files_box = ttk.Labelframe(container, text="Input Files", style='Section.TLabelframe')
            files_box.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 8))
            files_box.columnconfigure(1, weight=1)

            self.tpr_var = tk.StringVar()
            self.xtc_var = tk.StringVar()
            self.out_var = tk.StringVar(value=os.path.abspath("membrane_analysis_output"))

            self._file_row(files_box, 0, "Topology (.tpr/.gro/.pdb):", self.tpr_var, self._browse_tpr)
            self._file_row(files_box, 1, "Trajectory (.xtc/.trr/.dcd):", self.xtc_var, self._browse_xtc)
            self._file_row(files_box, 2, "Output directory:", self.out_var, self._browse_outdir)

            # ---- Selection panel ----
            sel_box = ttk.Labelframe(container, text="Selections", style='Section.TLabelframe')
            sel_box.grid(row=1, column=0, sticky='nsew', padx=(0, 6), pady=8)

            ttk.Label(sel_box, text="Lipid residues (comma-separated):").grid(row=0, column=0, sticky='w', padx=8, pady=(8, 2))
            self.lipids_var = tk.StringVar(value="POPE, POPG")
            ttk.Entry(sel_box, textvariable=self.lipids_var, width=30).grid(row=1, column=0, sticky='ew', padx=8)

            ttk.Label(sel_box, text="Water residue name:").grid(row=2, column=0, sticky='w', padx=8, pady=(8, 2))
            self.water_var = tk.StringVar(value="WAT")
            ttk.Entry(sel_box, textvariable=self.water_var, width=30).grid(row=3, column=0, sticky='ew', padx=8, pady=(0, 8))

            sel_box.columnconfigure(0, weight=1)

            # ---- Parameters panel ----
            param_box = ttk.Labelframe(container, text="Grid Parameters", style='Section.TLabelframe')
            param_box.grid(row=1, column=1, sticky='nsew', padx=(6, 0), pady=8)

            ttk.Label(param_box, text="Grid resolution (points per axis):").grid(row=0, column=0, sticky='w', padx=8, pady=(8, 2))
            self.grid_var = tk.IntVar(value=200)
            ttk.Spinbox(param_box, from_=50, to=600, increment=10, textvariable=self.grid_var, width=10).grid(row=1, column=0, sticky='w', padx=8)

            ttk.Label(param_box, text="Gaussian smoothing sigma:").grid(row=2, column=0, sticky='w', padx=8, pady=(8, 2))
            self.sigma_var = tk.DoubleVar(value=2.0)
            ttk.Spinbox(param_box, from_=0.0, to=10.0, increment=0.5, textvariable=self.sigma_var, width=10).grid(row=3, column=0, sticky='w', padx=8, pady=(0, 8))

            # ---- Figure groups panel ----
            groups_box = ttk.Labelframe(container, text="Figure Groups to Generate", style='Section.TLabelframe')
            groups_box.grid(row=2, column=0, columnspan=2, sticky='ew', pady=8)

            self.group_vars = {
                'A': tk.BooleanVar(value=True),
                'C': tk.BooleanVar(value=True),
                'D': tk.BooleanVar(value=True),
                'E': tk.BooleanVar(value=True),
                'standard': tk.BooleanVar(value=True),
            }
            labels = {
                'A': 'A: Radius of curvature time-series',
                'C': 'C: Electrostatic potential heatmap',
                'D': 'D: 3D structure with colored leaflets',
                'E': 'E: Membrane side-view analysis',
                'standard': 'Standard thickness/curvature/water plots',
            }
            for i, key in enumerate(['A', 'C', 'D', 'E', 'standard']):
                ttk.Checkbutton(groups_box, text=labels[key], variable=self.group_vars[key]).grid(
                    row=i // 3, column=i % 3, sticky='w', padx=10, pady=4)

            # ---- Run controls ----
            run_box = ttk.Frame(container)
            run_box.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(8, 4))
            run_box.columnconfigure(0, weight=1)

            self.run_button = ttk.Button(run_box, text="▶ Run Analysis", style='Run.TButton', command=self._on_run)
            self.run_button.grid(row=0, column=0, sticky='w')

            self.progress = ttk.Progressbar(run_box, mode='determinate', maximum=100)
            self.progress.grid(row=0, column=1, sticky='ew', padx=10)
            run_box.columnconfigure(1, weight=3)

            self.status_var = tk.StringVar(value="Idle")
            ttk.Label(run_box, textvariable=self.status_var).grid(row=0, column=2, sticky='e')

            # ---- Log panel ----
            log_box = ttk.Labelframe(container, text="Log", style='Section.TLabelframe')
            log_box.grid(row=4, column=0, columnspan=2, sticky='nsew', pady=(8, 0))
            container.rowconfigure(4, weight=1)

            self.log_widget = scrolledtext.ScrolledText(log_box, height=14, font=('Courier New', 9), bg='#101418', fg='#d6d6d6')
            self.log_widget.pack(fill='both', expand=True, padx=6, pady=6)
            self.log_widget.configure(state='disabled')

        def _file_row(self, parent, row, label, var, browse_cmd):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=8, pady=4)
            ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky='ew', padx=4, pady=4)
            ttk.Button(parent, text="Browse...", command=browse_cmd).grid(row=row, column=2, padx=8, pady=4)

        # ---------------- browse handlers ----------------
        def _browse_tpr(self):
            path = filedialog.askopenfilename(title="Select topology file",
                filetypes=[("Topology files", "*.tpr *.gro *.pdb *.psf"), ("All files", "*.*")])
            if path:
                self.tpr_var.set(path)

        def _browse_xtc(self):
            path = filedialog.askopenfilename(title="Select trajectory file",
                filetypes=[("Trajectory files", "*.xtc *.trr *.dcd *.nc"), ("All files", "*.*")])
            if path:
                self.xtc_var.set(path)

        def _browse_outdir(self):
            path = filedialog.askdirectory(title="Select output directory")
            if path:
                self.out_var.set(path)

        # ---------------- logging ----------------
        def _append_log(self, msg):
            self.log_widget.configure(state='normal')
            self.log_widget.insert('end', msg + '\n')
            self.log_widget.see('end')
            self.log_widget.configure(state='disabled')

        # ---------------- run ----------------
        def _on_run(self):
            if self.running:
                messagebox.showinfo("Already running", "An analysis is already in progress.")
                return

            tpr = self.tpr_var.get().strip()
            xtc = self.xtc_var.get().strip()
            out_dir = self.out_var.get().strip() or "."
            lipids = [s.strip() for s in self.lipids_var.get().split(',') if s.strip()]
            water_mol = self.water_var.get().strip() or "WAT"

            if not tpr or not os.path.isfile(tpr):
                messagebox.showerror("Missing topology", "Please select a valid topology file.")
                return
            if not xtc or not os.path.isfile(xtc):
                messagebox.showerror("Missing trajectory", "Please select a valid trajectory file.")
                return
            if not lipids:
                messagebox.showerror("Missing lipids", "Please enter at least one lipid residue name.")
                return

            try:
                grid_points = int(self.grid_var.get())
                sigma = float(self.sigma_var.get())
            except (tk.TclError, ValueError):
                messagebox.showerror("Invalid parameters", "Grid resolution and sigma must be numeric.")
                return

            selected_groups = {k for k, v in self.group_vars.items() if v.get()}
            if not selected_groups:
                messagebox.showerror("No figure groups", "Select at least one figure group to generate.")
                return

            self.log_widget.configure(state='normal')
            self.log_widget.delete('1.0', 'end')
            self.log_widget.configure(state='disabled')
            self.progress['value'] = 0
            self.status_var.set("Running...")
            self.run_button.configure(state='disabled')
            self.running = True

            self.worker_thread = threading.Thread(
                target=self._run_worker,
                args=(tpr, xtc, lipids, water_mol, grid_points, sigma, out_dir, selected_groups),
                daemon=True
            )
            self.worker_thread.start()

        def _run_worker(self, tpr, xtc, lipids, water_mol, grid_points, sigma, out_dir, groups):
            logger = make_logger("membrane_gui")
            handler = QueueLogHandler(self.log_queue)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)

            def progress_cb(stage, frac):
                self.log_queue.put(f"__PROGRESS__{frac:.4f}__{stage}")

            try:
                analyzer = ProfessionalMembraneSurfaceAnalyzer(
                    tpr, xtc, lipids, water_mol=water_mol, grid_points=grid_points,
                    smoothing_sigma=sigma, output_dir=out_dir, logger=logger,
                    progress_cb=progress_cb
                )
                analyzer.run_analysis(groups=groups)
                self.log_queue.put(f"__DONE__{out_dir}")
            except AnalysisError as exc:
                logger.error(f"\n✗ ANALYSIS ERROR: {exc}")
                self.log_queue.put("__FAILED__")
            except Exception:
                logger.error("\n✗ UNEXPECTED ERROR:\n" + traceback.format_exc())
                self.log_queue.put("__FAILED__")

        def _poll_log_queue(self):
            try:
                while True:
                    msg = self.log_queue.get_nowait()
                    if msg.startswith("__PROGRESS__"):
                        parts = msg[len("__PROGRESS__"):].split("__", 1)
                        frac = float(parts[0])
                        stage = parts[1] if len(parts) > 1 else ""
                        self.progress['value'] = max(0, min(100, frac * 100))
                        self.status_var.set(stage)
                    elif msg.startswith("__DONE__"):
                        out_dir = msg[len("__DONE__"):]
                        self.status_var.set("Finished ✓")
                        self.progress['value'] = 100
                        self.running = False
                        self.run_button.configure(state='normal')
                        self._append_log(f"\nAll done. Files written to: {os.path.abspath(out_dir)}")
                        messagebox.showinfo("Analysis complete",
                                             f"Analysis finished successfully.\n\nOutput directory:\n{os.path.abspath(out_dir)}")
                    elif msg.startswith("__FAILED__"):
                        self.status_var.set("Failed ✗")
                        self.running = False
                        self.run_button.configure(state='normal')
                        messagebox.showerror("Analysis failed",
                                              "The analysis failed. See the log panel for details.")
                    else:
                        self._append_log(msg)
            except queue.Empty:
                pass
            self.after(150, self._poll_log_queue)

    app = MembraneAnalysisGUI()
    app.mainloop()


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Membrane Surface Analysis")
    parser.add_argument('--nogui', action='store_true', help="Run headless instead of launching the GUI control panel.")
    parser.add_argument('--tpr', default="md_0_1000.tpr")
    parser.add_argument('--xtc', default="center1.xtc")
    parser.add_argument('--lipids', default="POPE,POPG")
    parser.add_argument('--water', default="WAT")
    parser.add_argument('--grid-points', type=int, default=250)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--output-dir', default=".")
    args = parser.parse_args()

    if args.nogui:
        lipids = [s.strip() for s in args.lipids.split(',') if s.strip()]
        try:
            run_headless(args.tpr, args.xtc, lipids, water_mol=args.water,
                         grid_points=args.grid_points, smoothing_sigma=args.sigma,
                         output_dir=args.output_dir)
        except AnalysisError as exc:
            LOG.error(f"✗ ANALYSIS ERROR: {exc}")
            sys.exit(1)
    else:
        launch_gui()
