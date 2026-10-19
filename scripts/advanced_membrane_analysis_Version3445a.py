import os
import csv
import json
from datetime import datetime
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Arc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.optimize import curve_fit, least_squares
import warnings
warnings.filterwarnings('ignore')

# Enhanced publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'Arial',
    'axes.linewidth': 1.5,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'grid.linewidth': 0.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

# ============================================================================
# CURVATURE-FRUSTRATION MODEL (Marsh, D. Biophys J. 2007;93:3884-3899)
# ============================================================================
# Marsh shows that the effect of the lateral pressure profile, pi(z), on
# protein conformation/insertion is equivalent to the elastic response of
# the lipid monolayer to its "frustrated" spontaneous curvature, c0 (his J0):
#
#   kc * c0  =  INTEGRAL (z - z0) * pi(z) dz     [first moment of pi(z)]
#
# and the Helfrich elastic bending energy density of a monolayer whose
# actual (total) curvature at the neutral surface is c1 = c1x + c1y = 2H is:
#
#   gc(c1) = (kc/2) * (c1 - c0)^2
#          = (kc/2)*c1^2  -  kc*c0*c1  +  (kc/2)*c0^2
#
# The cross term, -kc*c0*c1, is the "spontaneous curvature frustration"
# energy: the penalty (or relief) incurred when the monolayer is held at
# some actual curvature c1 that differs from its unconstrained preference
# c0. In a planar bilayer each leaflet is forced flat (c1 ~ 0) even though
# c0 =/= 0 for non-lamellar-forming lipids (e.g., PE, CL), so the leaflet
# is permanently "frustrated," storing energy (kc/2)*c0^2 per unit area.
# Local curvature induced by an inserted protein (c1 =/= 0) partially
# relieves or adds to this frustration depending on its sign relative
# to c0 -- this is what couples protein conformation to lipid composition.
#
# This script computes c1 directly from the fitted leaflet surfaces via
# proper differential-geometry mean curvature (not a small-slope Laplacian
# approximation), and combines it with literature/user-supplied c0 values
# per lipid species (composition-weighted per leaflet) to map the local
# elastic bending energy density gc(x,y) and the total stored frustration
# energy, following Marsh (2007), Eqs. 1-3 and 10-13.
#
# NOTE: A full reconstruction of pi(z) itself requires the local stress
# tensor (virial per z-slab), which needs force information not available
# from a plain coordinate trajectory in MDAnalysis. This script therefore
# implements the curvature/elastic side of the theory (c1, c0, kc, gc),
# which is what is needed to evaluate curvature frustration and its
# coupling to protein conformation/insertion as derived in the paper.
# ============================================================================

# Literature/estimated monolayer spontaneous curvatures, c0 (Å^-1), by
# headgroup class (Marsh, D. Chem Phys Lipids. 2006;144:146-159; Marsh 2007;
# Kollmitzer et al. Soft Matter 2013;9:10877 -- original literature values
# are commonly quoted in nm^-1 and are converted here by dividing by 10).
# Negative c0 = curves toward water (favors inverted/H_II-type curvature);
# values near 0 = ~lamellar. These are DEFAULTS classified by headgroup
# substring in the resname and should be overridden with experimental/
# literature values where available.
DEFAULT_C0_BY_HEADGROUP = {
    'PE':  -0.035,   # phosphatidylethanolamines (non-lamellar-forming); -0.35 nm^-1
    'PG':  -0.005,   # phosphatidylglycerols (near-lamellar, mildly negative); -0.05 nm^-1
    'PC':  -0.002,   # phosphatidylcholines (~lamellar); -0.02 nm^-1
    'PS':  -0.003,   # phosphatidylserines; -0.03 nm^-1
    'CL':  -0.040,   # cardiolipins (VCL, PVCL variants; strongly non-lamellar); -0.40 nm^-1
    'VCL': -0.040,
}

DEFAULT_KC_KBT = 9.9   # monolayer bending modulus, kc, in units of kBT
                        # (mean value for DOPE/DOPC monolayers, Marsh 2007)


def classify_lipid_c0(resname, c0_overrides=None):
    """Return spontaneous curvature c0 (Å^-1) for a lipid resname.

    Looks up an explicit override first, then falls back to a
    headgroup-substring classification (see DEFAULT_C0_BY_HEADGROUP).
    """
    if c0_overrides and resname in c0_overrides:
        return c0_overrides[resname]
    name = resname.upper()
    # Longer/more specific keys first so e.g. 'VCL' beats a stray 'PC' match
    for key in sorted(DEFAULT_C0_BY_HEADGROUP, key=len, reverse=True):
        if key in name:
            return DEFAULT_C0_BY_HEADGROUP[key]
    return 0.0  # unknown lipid: assume ~lamellar as a conservative default


class ProfessionalMembraneSurfaceAnalyzer:
    def __init__(self, tpr, xtc, lipids, water_mol='WAT', grid_points=200, smoothing_sigma=2,
                 kc_kbt=DEFAULT_KC_KBT, c0_overrides=None, temperature=310.0,
                 output_dir=None, n_timeseries_frames=20, protein_probe_radius=15.0,
                 arc_axis='x', n_normal_vectors=7):
        """Initialize professional membrane analyzer

        Additional parameters (Marsh 2007 curvature-frustration model):
            kc_kbt      : monolayer bending modulus, kc, in units of kBT (default 9.9)
            c0_overrides: dict mapping resname -> spontaneous curvature c0 (Å^-1);
                          entries not given fall back to headgroup-class defaults
                          (see DEFAULT_C0_BY_HEADGROUP)
            temperature : temperature in K, used only for reporting kc/energies
                          in absolute units (kJ/mol) alongside kBT units

        Additional parameters (Pacsin1/F-BAR-style curvature induction &
        sensing analysis, after Mahmood, Noguchi & Okazaki, curvature
        induction/sensing of F-BAR protein Pacsin1 via MD simulations):
            output_dir          : folder to save ALL output into (created if it
                                   doesn't exist). Defaults to a fresh
                                   timestamped folder so repeated runs never
                                   overwrite each other.
            n_timeseries_frames : approx. number of trajectory frames to sample
                                   for every time-series analysis (default 20)
            protein_probe_radius: radius (Å), around the protein's XY center of
                                   mass, used to define the protein "footprint"
                                   on the membrane surface for curvature-sensing
                                   analysis (default 15 Å)
            arc_axis             : 'x' or 'y' -- which cross-section direction to
                                   use for the normal-vector arc/Δφ curvature
                                   schematic (default 'x', through the grid center)
            n_normal_vectors     : number of evenly-spaced surface normal
                                   vectors S(x) to draw on the Pacsin1-style
                                   curvature schematic (default 7, as in the
                                   reference figure)
        """
        print("=" * 80)
        print("PROFESSIONAL MEMBRANE SURFACE ANALYSIS WITH DYNAMICS")
        print("=" * 80)

        # ---------------- Output folder tree ----------------
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"membrane_analysis_output_{timestamp}"
        self.output_dir = output_dir
        self.subdirs = {
            '2d': os.path.join(self.output_dir, '01_2D_analysis'),
            '3d': os.path.join(self.output_dir, '02_3D_analysis'),
            'timeseries': os.path.join(self.output_dir, '03_time_series'),
            'frustration': os.path.join(self.output_dir, '04_curvature_frustration'),
            'pacsin1': os.path.join(self.output_dir, '05_pacsin1_curvature_sensing'),
            'data': os.path.join(self.output_dir, '06_data_exports'),
        }
        for path in [self.output_dir] + list(self.subdirs.values()):
            os.makedirs(path, exist_ok=True)
        print(f"✓ All output will be saved under: {os.path.abspath(self.output_dir)}")

        print("Loading system...")
        self.u = mda.Universe(tpr, xtc)
        self.lipids = " ".join(lipids)
        self.lipid_atoms = self.u.select_atoms(f"resname {self.lipids}")
        self.water = self.u.select_atoms(f"resname {water_mol}")
        self.protein = self.u.select_atoms("protein")
        
        self.grid_points = grid_points
        self.smooth_sigma = smoothing_sigma
        self.ts = self.u.trajectory[-1]

        # Marsh (2007) curvature-frustration model parameters
        self.kc_kbt = kc_kbt                    # monolayer bending modulus (kBT)
        self.c0_overrides = c0_overrides or {}   # resname -> c0 (Å^-1) overrides
        self.temperature = temperature           # K
        R_GAS = 8.314462618e-3                   # kJ mol^-1 K^-1
        self.kbt_kjmol = R_GAS * self.temperature  # kBT in kJ/mol at this T
        self.kc_kjmol = self.kc_kbt * self.kbt_kjmol

        # Pacsin1/F-BAR curvature-sensing parameters
        self.n_timeseries_frames = n_timeseries_frames
        self.protein_probe_radius = protein_probe_radius
        self.arc_axis = arc_axis
        self.n_normal_vectors = n_normal_vectors
        self.has_protein = len(self.protein) > 0
        if not self.has_protein:
            print("⚠ No 'protein' selection found -- curvature-sensing analysis "
                  "relative to protein position will be skipped.")

        # For time-series analysis
        self.curvature_over_time = []
        self.curvature_u_over_time = []
        self.curvature_l_over_time = []
        self.thickness_over_time = []
        self.deflection_over_time = []
        self.time_points = []
        self.frustration_energy_over_time = []
        self.arc_radius_over_time = []          # circle-fit R(t), Pacsin1-style
        self.protein_local_curvature_over_time = []
        self.protein_bulk_curvature_over_time = []
        self.protein_insertion_depth_over_time = []
        self.protein_com_z_over_time = []
        
    def extract_surface_data(self):
        """Extract membrane surface coordinates"""
        coords = self.lipid_atoms.center_of_mass(compound="residues")
        resnames = self.lipid_atoms.residues.resnames

        self.x = coords[:, 0]
        self.y = coords[:, 1]
        self.z = coords[:, 2]

        mid = np.median(self.z)
        upper_mask = self.z > mid
        lower_mask = self.z < mid
        self.upper = coords[upper_mask]
        self.lower = coords[lower_mask]
        self.upper_resnames = resnames[upper_mask]
        self.lower_resnames = resnames[lower_mask]

        print(f"✓ Upper leaflet points: {len(self.upper)}")
        print(f"✓ Lower leaflet points: {len(self.lower)}")
        
    def create_surface_grid(self):
        """Create interpolated surface grids"""
        xi = np.linspace(self.x.min(), self.x.max(), self.grid_points)
        yi = np.linspace(self.y.min(), self.y.max(), self.grid_points)
        self.xi, self.yi = np.meshgrid(xi, yi)

        # Real physical grid spacing (Angstrom), needed for correctly-scaled
        # derivatives/curvatures rather than assuming unit grid spacing
        self.dx = xi[1] - xi[0]
        self.dy = yi[1] - yi[0]

        self.zi = griddata((self.x, self.y), self.z, (self.xi, self.yi), method='cubic')
        self.zu = griddata((self.upper[:, 0], self.upper[:, 1]), self.upper[:, 2], 
                          (self.xi, self.yi), method='cubic')
        self.zl = griddata((self.lower[:, 0], self.lower[:, 1]), self.lower[:, 2], 
                          (self.xi, self.yi), method='cubic')
        
        for arr in [self.zi, self.zu, self.zl]:
            arr[np.isnan(arr)] = np.nanmean(arr)
        
        self.zi = gaussian_filter(self.zi, self.smooth_sigma)
        self.zu = gaussian_filter(self.zu, self.smooth_sigma)
        self.zl = gaussian_filter(self.zl, self.smooth_sigma)
        
        self.thickness = self.zu - self.zl
        
    @staticmethod
    def _mean_and_gaussian_curvature(z, dy, dx, method='gaussian_derivative', deriv_sigma=1.5):
        """Differential-geometry mean (H) and Gaussian (K) curvature of a
        Monge-patch surface z(x,y), using the exact (not small-slope) formulas:

            H = [(1+zx^2)*zyy - 2*zx*zy*zxy + (1+zy^2)*zxx] / [2*(1+zx^2+zy^2)^1.5]
            K = [zxx*zyy - zxy^2] / (1+zx^2+zy^2)^2

        Returns (H, K, c1) where c1 = 2H is the "total curvature"
        (c1 = c1_x + c1_y, sum of principal curvatures) in the same
        convention as the spontaneous curvature c0 used by Marsh (2007)
        and standard monolayer elasticity literature (c0 = 2*H0).
        Units follow the input grid spacing (Angstrom in this script).

        method:
            'gaussian_derivative' (default, IMPROVED/"actual" curvature):
                first/second partial derivatives are estimated as
                scale-space (Gaussian-derivative) filters of the surface,
                i.e. convolving z(x,y) with derivatives of a Gaussian
                kernel (scipy.ndimage.gaussian_filter(..., order=...)).
                This is the standard noise-robust way to differentiate a
                noisy sampled surface twice: naive np.gradient-of-gradient
                amplifies high-frequency lipid-position noise on the SECOND
                derivative, which can distort the fitted curvature; a
                Gaussian-derivative filter differentiates and smooths in a
                single, mathematically consistent step (differentiation of
                a mollified surface), giving a more reliable estimate of
                the true underlying membrane curvature at a chosen length
                scale (deriv_sigma grid cells).
            'finite_difference' (legacy): plain np.gradient applied twice,
                kept only for backward comparison/validation.
        """
        if method == 'gaussian_derivative':
            zy = gaussian_filter(z, sigma=deriv_sigma, order=(1, 0)) / dy
            zx = gaussian_filter(z, sigma=deriv_sigma, order=(0, 1)) / dx
            zyy = gaussian_filter(z, sigma=deriv_sigma, order=(2, 0)) / dy**2
            zxx = gaussian_filter(z, sigma=deriv_sigma, order=(0, 2)) / dx**2
            zxy = gaussian_filter(z, sigma=deriv_sigma, order=(1, 1)) / (dx * dy)
        else:
            # np.gradient(z, dy, dx): first return axis=0 (rows -> y), second axis=1 (cols -> x)
            zy, zx = np.gradient(z, dy, dx)
            zyy, zyx = np.gradient(zy, dy, dx)
            zxy, zxx = np.gradient(zx, dy, dx)
            zxy = 0.5 * (zxy + zyx)  # symmetrize mixed partials (numerical cross-check)

        denom = (1.0 + zx**2 + zy**2)
        H = ((1 + zx**2) * zyy - 2 * zx * zy * zxy + (1 + zy**2) * zxx) / (2 * denom**1.5)
        K = (zxx * zyy - zxy**2) / (denom**2)
        c1 = 2.0 * H
        return H, K, c1

    def calculate_curvature(self, method='gaussian_derivative'):
        """Calculate membrane curvature using exact differential geometry
        (mean curvature H, Gaussian curvature K) at the neutral surface of
        each leaflet, rather than a small-slope Laplacian approximation.

        c1 = 2H is the total (sum of principal) curvature, in the same
        sign/scale convention as the lipid spontaneous curvature c0 used
        in the Marsh (2007) elastic frustration model.

        method: 'gaussian_derivative' (default) -- noise-robust scale-space
                derivatives, the recommended/"actual" curvature estimate
                (see _mean_and_gaussian_curvature docstring); or
                'finite_difference' for the legacy double-np.gradient estimate.
        """
        dy, dx = self.dy, self.dx
        self.curvature_method = method

        # Bilayer midplane (kept for backward-compatible plots)
        H_mid, K_mid, c1_mid = self._mean_and_gaussian_curvature(self.zi, dy, dx, method=method)
        self.curvature = c1_mid                    # total curvature, Angstrom^-1
        self.gaussian_curvature = K_mid
        self.curvature_norm = self.curvature / np.max(np.abs(self.curvature))

        # Upper leaflet (neutral surface approximated by fitted headgroup surface)
        H_u, K_u, c1_u = self._mean_and_gaussian_curvature(self.zu, dy, dx, method=method)
        self.mean_curvature_u = H_u
        self.gaussian_curvature_u = K_u
        self.curvature_u = c1_u                     # total curvature c1, Angstrom^-1

        # Lower leaflet
        H_l, K_l, c1_l = self._mean_and_gaussian_curvature(self.zl, dy, dx, method=method)
        self.mean_curvature_l = H_l
        self.gaussian_curvature_l = K_l
        self.curvature_l = c1_l                      # Angstrom^-1

        # Calculate radius of curvature (midplane + per-leaflet maps)
        self.calc_radius_of_curvature()

    def calc_radius_of_curvature(self, cap=1000.0):
        """Calculate the ACTUAL radius-of-curvature maps (Å) from the total
        curvature c1 = 2H, at the midplane and for each leaflet
        independently. For a sphere, c1 = 2/R, so R = 2/|c1|; this reduces
        to the familiar R = 1/|c| only in the small-curvature limit.

        Near-flat regions (|c1| -> 0) give a formally divergent R; these
        are capped at `cap` Å so the maps/plots stay finite and readable,
        rather than silently reporting fictitious near-infinite radii.
        """
        def _r_map(curv):
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(np.abs(curv) > 1e-6, 2.0 / np.abs(curv), np.nan)
            return np.nan_to_num(r, nan=cap)

        self.radius_of_curvature = _r_map(self.curvature)          # midplane, Å
        self.radius_of_curvature_u = _r_map(self.curvature_u)      # upper leaflet, Å
        self.radius_of_curvature_l = _r_map(self.curvature_l)      # lower leaflet, Å

    def calculate_arc_radius_of_curvature(self):
        """Second, independent estimate of the (global) radius of curvature:
        a direct circle fit to the leaflet-midplane cross-section arc, in
        the spirit of the F-BAR/Pacsin1 curvature-sensing papers, which
        characterize the membrane's bend by fitting a circular arc to the
        bead/lipid backbone positions along the curved profile, rather than
        a local differential-geometry estimate.

        Fits (x - xc)^2 + (z - zc)^2 = R^2 to the mid-surface profile along
        self.arc_axis through the grid center, via least-squares (algebraic
        circle fit), and stores:
            self.arc_fit_radius   : fitted R (Å)
            self.arc_fit_center   : (xc, zc)
            self.arc_profile_xy   : (s_profile, z_profile) used for the fit
        """
        mid = self.grid_points // 2
        if self.arc_axis == 'y':
            s_profile = self.yi[:, mid]
            z_profile = self.zi[:, mid]
        else:
            s_profile = self.xi[mid, :]
            z_profile = self.zi[mid, :]

        def residuals(params, s, z):
            sc, zc, R = params
            return np.sqrt((s - sc) ** 2 + (z - zc) ** 2) - R

        s0 = np.nanmean(s_profile)
        z0 = np.nanmean(z_profile)
        R0 = max(np.ptp(s_profile), 1.0)
        try:
            valid = ~(np.isnan(s_profile) | np.isnan(z_profile))
            result = least_squares(residuals, x0=[s0, z0, R0],
                                    args=(s_profile[valid], z_profile[valid]))
            sc_fit, zc_fit, R_fit = result.x
            self.arc_fit_radius = float(abs(R_fit))
            self.arc_fit_center = (float(sc_fit), float(zc_fit))
        except Exception:
            self.arc_fit_radius = np.nan
            self.arc_fit_center = (np.nan, np.nan)

        self.arc_profile_s = s_profile
        self.arc_profile_z = z_profile

    def calculate_normal_vector_curvature(self):
        """Discrete surface-normal / Δφ curvature analysis, matching the
        reference schematic: place self.n_normal_vectors evenly-spaced
        points S(x) along the leaflet mid-surface cross-section arc, build
        the local outward surface-normal vector at each point, and compute
        the angle Δφ between successive normal vectors. For a circular
        arc, R = arc_length_between_points / Δφ, giving a purely
        geometric, local estimate of the radius of curvature that is
        independent of the differential-geometry (Laplacian) estimate
        above -- useful as a cross-check, and to reproduce a Pacsin1/F-BAR
        -style normal-vector curvature-sensing figure.
        """
        mid = self.grid_points // 2
        if self.arc_axis == 'y':
            s_full = self.yi[:, mid]
            z_full = self.zi[:, mid]
        else:
            s_full = self.xi[mid, :]
            z_full = self.zi[mid, :]

        order = np.argsort(s_full)
        s_full = s_full[order]
        z_full = z_full[order]

        n = self.n_normal_vectors
        idx = np.linspace(0, len(s_full) - 1, n).astype(int)
        s_pts = s_full[idx]
        z_pts = z_full[idx]

        # Local tangent via central differences on the sampled arc, then
        # normal = tangent rotated 90 deg (pointing away from the membrane,
        # i.e. +z-ward for the upper leaflet convention used here)
        ds = np.gradient(s_full)
        dz = np.gradient(z_full)
        tangent = np.stack([ds, dz], axis=1)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        normal[normal[:, 1] < 0] *= -1  # keep normals pointing "up" (+z)

        normal_pts = normal[idx]

        # angle of each normal vector from +z axis
        phi = np.arctan2(normal_pts[:, 0], normal_pts[:, 1])
        dphi = np.diff(phi)

        # arc length between consecutive S(x) points along the sampled curve
        arc_len_full = np.concatenate([[0], np.cumsum(np.hypot(np.diff(s_full), np.diff(z_full)))])
        arc_len_pts = arc_len_full[idx]
        ds_between = np.diff(arc_len_pts)

        with np.errstate(divide='ignore', invalid='ignore'):
            local_R = np.where(np.abs(dphi) > 1e-8, ds_between / np.abs(dphi), np.nan)

        self.normal_vector_s = s_pts
        self.normal_vector_z = z_pts
        self.normal_vector_dirs = normal_pts
        self.normal_vector_dphi = dphi          # Δφ between successive S(x), radians
        self.normal_vector_local_R = local_R    # local radius of curvature between successive S(x)

    def calculate_lipid_deflection(self):
        """Calculate phospholipid deflection"""
        z_avg = (self.zu + self.zl) / 2
        
        self.deflection_u = self.zu - z_avg
        self.deflection_l = self.zl - z_avg
        self.deflection_total = np.abs(self.deflection_u) + np.abs(self.deflection_l)
        
        print(f"✓ Upper leaflet max deflection: {np.max(np.abs(self.deflection_u)):.3f} Å")
        print(f"✓ Lower leaflet max deflection: {np.max(np.abs(self.deflection_l)):.3f} Å")

    # ==================== SPONTANEOUS CURVATURE FRUSTRATION (Marsh 2007) ====================

    def _leaflet_c0(self, resnames):
        """Composition (mole-fraction) weighted spontaneous curvature c0
        (Å^-1) for a leaflet, from its residue-name inventory."""
        resnames = np.asarray(resnames)
        if len(resnames) == 0:
            return 0.0, {}
        unique, counts = np.unique(resnames, return_counts=True)
        fractions = counts / counts.sum()
        c0_by_lipid = {name: classify_lipid_c0(name, self.c0_overrides) for name in unique}
        c0_avg = float(np.sum(fractions * np.array([c0_by_lipid[n] for n in unique])))
        composition = {name: (frac, c0_by_lipid[name]) for name, frac in zip(unique, fractions)}
        return c0_avg, composition

    def calculate_curvature_frustration(self):
        """Compute the Helfrich elastic bending energy density and the
        spontaneous-curvature-frustration energy of each leaflet, following
        Marsh (2007), Biophys J 93:3884-3899:

            gc(x,y)     = (kc/2) * (c1(x,y) - c0)^2        [total elastic energy density]
            g_frust(x,y)= -kc * c0 * c1(x,y)                [curvature-frustration cross term]

        where c1 = 2H is the actual (fitted) leaflet curvature computed in
        calculate_curvature(), and c0 is the composition-weighted lipid
        spontaneous curvature of that leaflet. kc is the monolayer bending
        modulus (self.kc_kbt, units kBT).

        Energies are reported both as densities (kBT/Å^2) and integrated
        over the leaflet area (kBT, and kJ/mol).
        """
        # Composition-weighted spontaneous curvature per leaflet
        self.c0_upper, self.composition_upper = self._leaflet_c0(self.upper_resnames)
        self.c0_lower, self.composition_lower = self._leaflet_c0(self.lower_resnames)

        kc = self.kc_kbt

        # gc = (kc/2)*(c1-c0)^2 ; g_frust = -kc*c0*c1  (both per unit area, kBT/Å^2)
        c1_u = self.curvature_u
        c1_l = self.curvature_l

        self.gc_upper = 0.5 * kc * (c1_u - self.c0_upper) ** 2
        self.gc_lower = 0.5 * kc * (c1_l - self.c0_lower) ** 2
        self.g_frustration_upper = -kc * self.c0_upper * c1_u
        self.g_frustration_lower = -kc * self.c0_lower * c1_l

        # Baseline: energy if the leaflet were perfectly flat (c1=0), i.e.
        # the "stored" frustration energy of an unperturbed planar bilayer
        self.gc_flat_upper = 0.5 * kc * self.c0_upper ** 2
        self.gc_flat_lower = 0.5 * kc * self.c0_lower ** 2

        # Integrate energy density over leaflet area -> total energy (kBT)
        cell_area_ang2 = self.dx * self.dy   # Angstrom^2
        self.total_gc_upper_kbt = float(np.nansum(self.gc_upper) * cell_area_ang2)
        self.total_gc_lower_kbt = float(np.nansum(self.gc_lower) * cell_area_ang2)
        self.total_frustration_kbt = float(
            np.nansum(self.g_frustration_upper + self.g_frustration_lower) * cell_area_ang2
        )
        self.total_gc_kbt = self.total_gc_upper_kbt + self.total_gc_lower_kbt

        print(f"✓ Leaflet bending modulus kc = {kc:.2f} kBT ({self.kc_kjmol:.2f} kJ/mol)")
        print(f"✓ Upper leaflet: c0 = {self.c0_upper:+.4f} Å⁻¹  "
              f"(composition: {self._format_composition(self.composition_upper)})")
        print(f"✓ Lower leaflet: c0 = {self.c0_lower:+.4f} Å⁻¹  "
              f"(composition: {self._format_composition(self.composition_lower)})")
        print(f"✓ Mean actual curvature c1: upper = {np.nanmean(c1_u):+.5f} Å⁻¹, "
              f"lower = {np.nanmean(c1_l):+.5f} Å⁻¹")
        print(f"✓ Total elastic bending energy: {self.total_gc_kbt:.1f} kBT "
              f"({self.total_gc_kbt * self.kbt_kjmol:.1f} kJ/mol)")
        print(f"✓ Net curvature-frustration energy (-kc·c0·c1, integrated): "
              f"{self.total_frustration_kbt:.1f} kBT")

    @staticmethod
    def _format_composition(composition):
        return ", ".join(f"{name} {frac*100:.0f}%(c0={c0:+.2f})"
                          for name, (frac, c0) in composition.items())

    def analyze_pore_radius(self, threshold=0.5):
        """Analyze pore regions and radius"""
        thickness_norm = (self.thickness - self.thickness.mean()) / self.thickness.std()
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
            self.max_pore_radius = 0
            self.pore_radius_map = np.zeros_like(self.pore_mask, dtype=float)
            
        print(f"✓ Maximum pore radius: {self.max_pore_radius:.2f} Å")
        
    def analyze_water_permeation(self, z_min=None, z_max=None):
        """Analyze water permeation"""
        if len(self.water) == 0:
            print("⚠ No water molecules found")
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
        print(f"✓ Water molecules in membrane region: {self.water_influx}")
        
        water_density = np.zeros_like(self.zi)
        if len(water_in_membrane_xyz) > 0:
            water_xy = water_in_membrane_xyz[:, :2]
            
            for i in range(len(self.xi)):
                for j in range(len(self.yi[0])):
                    x_range = [self.xi[i, j] - np.abs(self.xi[1, 0] - self.xi[0, 0])/2,
                              self.xi[i, j] + np.abs(self.xi[1, 0] - self.xi[0, 0])/2]
                    y_range = [self.yi[i, j] - np.abs(self.yi[0, 1] - self.yi[0, 0])/2,
                              self.yi[i, j] + np.abs(self.yi[0, 1] - self.yi[0, 0])/2]
                    
                    in_cell = ((water_xy[:, 0] > x_range[0]) & (water_xy[:, 0] < x_range[1]) &
                              (water_xy[:, 1] > y_range[0]) & (water_xy[:, 1] < y_range[1]))
                    water_density[i, j] = np.sum(in_cell)
        
        self.water_density = gaussian_filter(water_density, self.smooth_sigma)
        
    def _protein_footprint_curvature(self):
        """Curvature-SENSING metric (Pacsin1/F-BAR style): compare the mean
        actual curvature c1 of the grid cells under the protein's XY
        footprint (within protein_probe_radius of its center of mass)
        against the mean c1 of the rest of the leaflet ("bulk"). A protein
        that senses/prefers curved regions will show local >> bulk.
        Also returns the protein's insertion depth relative to the local
        bilayer midplane (z_protein_com - z_local_midplane)."""
        if not self.has_protein:
            return np.nan, np.nan, np.nan, np.nan

        prot_com = self.protein.center_of_mass()
        px, py, pz = prot_com

        dist2 = (self.xi - px) ** 2 + (self.yi - py) ** 2
        footprint = dist2 <= self.protein_probe_radius ** 2

        if not np.any(footprint):
            # protein footprint smaller than grid resolution -> nearest cell
            idx = np.unravel_index(np.argmin(dist2), dist2.shape)
            footprint = np.zeros_like(dist2, dtype=bool)
            footprint[idx] = True

        local_curv = np.nanmean(self.curvature[footprint])
        bulk_curv = np.nanmean(self.curvature[~footprint])

        # local bilayer midplane z under the footprint, for insertion depth
        local_mid_z = np.nanmean(self.zi[footprint])
        insertion_depth = pz - local_mid_z

        return local_curv, bulk_curv, insertion_depth, pz

    def collect_time_series_data(self):
        """Collect curvature, thickness, deflection, frustration-energy,
        arc-fit radius-of-curvature, and (if a protein is present)
        curvature-sensing/insertion-depth data across multiple frames."""
        print("\n" + "="*80)
        print("COLLECTING TIME-SERIES DATA")
        print("="*80)
        
        n_frames = len(self.u.trajectory)
        step = max(1, n_frames // self.n_timeseries_frames)
        print(f"Processing {n_frames} frames (sampling every {step} frames)...")
        
        for frame_idx, ts in enumerate(self.u.trajectory[::step]):
            self.extract_surface_data()
            self.create_surface_grid()
            self.calculate_curvature()
            self.calculate_lipid_deflection()
            self.calculate_curvature_frustration()
            self.calculate_arc_radius_of_curvature()

            # Store time-series data
            time_ps = ts.time
            self.time_points.append(time_ps)
            self.curvature_over_time.append(np.mean(np.abs(self.curvature)))
            self.curvature_u_over_time.append(np.nanmean(self.curvature_u))
            self.curvature_l_over_time.append(np.nanmean(self.curvature_l))
            self.thickness_over_time.append(np.mean(self.thickness))
            self.deflection_over_time.append(np.mean(self.deflection_total))
            self.frustration_energy_over_time.append(self.total_gc_kbt)
            self.arc_radius_over_time.append(self.arc_fit_radius)

            local_curv, bulk_curv, ins_depth, pcom_z = self._protein_footprint_curvature()
            self.protein_local_curvature_over_time.append(local_curv)
            self.protein_bulk_curvature_over_time.append(bulk_curv)
            self.protein_insertion_depth_over_time.append(ins_depth)
            self.protein_com_z_over_time.append(pcom_z)
            
            if (frame_idx + 1) % max(1, self.n_timeseries_frames // 5) == 0:
                print(f"  ✓ Processed frame {frame_idx + 1} (t={time_ps/1000:.2f} ns)")
        
        self.time_points = np.array(self.time_points)
        self.curvature_over_time = np.array(self.curvature_over_time)
        self.curvature_u_over_time = np.array(self.curvature_u_over_time)
        self.curvature_l_over_time = np.array(self.curvature_l_over_time)
        self.thickness_over_time = np.array(self.thickness_over_time)
        self.deflection_over_time = np.array(self.deflection_over_time)
        self.frustration_energy_over_time = np.array(self.frustration_energy_over_time)
        self.arc_radius_over_time = np.array(self.arc_radius_over_time)
        self.protein_local_curvature_over_time = np.array(self.protein_local_curvature_over_time)
        self.protein_bulk_curvature_over_time = np.array(self.protein_bulk_curvature_over_time)
        self.protein_insertion_depth_over_time = np.array(self.protein_insertion_depth_over_time)
        self.protein_com_z_over_time = np.array(self.protein_com_z_over_time)

        # Actual radius-of-curvature over time, R = 2/|c1|, midplane + per-leaflet
        with np.errstate(divide='ignore', invalid='ignore'):
            self.radius_mid_over_time = np.where(self.curvature_over_time > 1e-6,
                                                  2.0 / self.curvature_over_time, np.nan)
            self.radius_u_over_time = np.where(np.abs(self.curvature_u_over_time) > 1e-6,
                                                2.0 / np.abs(self.curvature_u_over_time), np.nan)
            self.radius_l_over_time = np.where(np.abs(self.curvature_l_over_time) > 1e-6,
                                                2.0 / np.abs(self.curvature_l_over_time), np.nan)

        self._export_timeseries_csv()
        
    # ==================== TOP & SIDE VIEW PLOTS ====================
    
    def plot_top_view(self):
        """Top view: lipid scatter + leaflet contours (individual file)."""
        fig, ax_top = plt.subplots(figsize=(9, 8))

        lipid_xy = self.lipid_atoms.positions[:, :2]
        ax_top.scatter(lipid_xy[:, 0], lipid_xy[:, 1], c='red', s=1, alpha=0.4, label='Lipid atoms')
        ax_top.contour(self.xi, self.yi, self.zu, levels=8, colors='blue', linewidths=2.5)
        ax_top.contour(self.xi, self.yi, self.zl, levels=8, colors='red', linewidths=1.5, linestyles='--', alpha=0.7)

        ax_top.set_xlabel('X (Å)', fontsize=13, fontweight='bold')
        ax_top.set_ylabel('Y (Å)', fontsize=13, fontweight='bold')
        ax_top.set_title('Top View', fontsize=15, fontweight='bold', pad=15)
        ax_top.legend(['Lipid atoms', 'Upper leaflet', 'Lower leaflet'], fontsize=11, loc='upper right')
        ax_top.grid(True, alpha=0.2)
        ax_top.set_aspect('equal')

        plt.tight_layout()
        outpath = os.path.join(self.subdirs['2d'], 'top_view.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_side_view(self):
        """Side view: leaflet cross-section with the actual radius of
        curvature (from the differential-geometry calculation) annotated
        (individual file)."""
        fig, ax_side = plt.subplots(figsize=(9, 8))

        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :]
        zu_profile = self.zu[y_idx, :]
        zl_profile = self.zl[y_idx, :]

        ax_side.fill_between(x_profile, zu_profile, zu_profile + 10, color='blue', alpha=0.4, label='Upper leaflet')
        ax_side.plot(x_profile, zu_profile, 'b-', linewidth=3)
        ax_side.fill_between(x_profile, zl_profile - 10, zl_profile, color='red', alpha=0.4, label='Lower leaflet')
        ax_side.plot(x_profile, zl_profile, 'r-', linewidth=3)

        rad_curv = np.nanmean(self.radius_of_curvature[~np.isinf(self.radius_of_curvature)])
        ax_side.text(0.5, 0.95, f'Radius of curvature = {rad_curv:.1f} Å',
                    transform=ax_side.transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    verticalalignment='top', horizontalalignment='center')

        ax_side.set_xlabel('X (Å)', fontsize=13, fontweight='bold')
        ax_side.set_ylabel('Z (Å)', fontsize=13, fontweight='bold')
        ax_side.set_title('Side View', fontsize=15, fontweight='bold', pad=15)
        ax_side.legend(fontsize=11, loc='upper right')
        ax_side.grid(True, alpha=0.2)

        plt.tight_layout()
        outpath = os.path.join(self.subdirs['2d'], 'side_view.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    # ==================== CROSS-SECTION PROFILES (individual files) ====================

    def plot_z_profile(self):
        """Membrane Z cross-section profile (individual file)."""
        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :]
        zu_profile = self.zu[y_idx, :]
        zl_profile = self.zl[y_idx, :]

        fig, ax1 = plt.subplots(figsize=(11, 6.5))
        ax1.fill_between(x_profile, zu_profile, zu_profile + 10, color='blue', alpha=0.3, label='Upper leaflet')
        ax1.plot(x_profile, zu_profile, 'b-', linewidth=2.5, label='Upper surface')
        ax1.fill_between(x_profile, zl_profile - 10, zl_profile, color='red', alpha=0.3, label='Lower leaflet')
        ax1.plot(x_profile, zl_profile, 'r-', linewidth=2.5, label='Lower surface')
        ax1.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Z (Å)', fontsize=12, fontweight='bold')
        ax1.set_title('Membrane Z-Profile (Cross-Section)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper right', framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['2d'], 'z_profile.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_thickness_profile(self):
        """Membrane thickness cross-section profile (individual file)."""
        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :]
        thickness_profile = self.thickness[y_idx, :]

        fig, ax2 = plt.subplots(figsize=(11, 6.5))
        ax2.plot(x_profile, thickness_profile, 'g-', linewidth=2.5)
        ax2.fill_between(x_profile, thickness_profile, alpha=0.3, color='green')
        ax2.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Thickness (Å)', fontsize=12, fontweight='bold')
        ax2.set_title('Membrane Thickness Profile', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['2d'], 'thickness_profile.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_curvature_profile(self):
        """Membrane curvature (c1) cross-section profile (individual file)."""
        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :]
        curvature_profile = self.curvature[y_idx, :]

        fig, ax3 = plt.subplots(figsize=(11, 6.5))
        ax3.plot(x_profile, curvature_profile, 'purple', linewidth=2.5)
        ax3.fill_between(x_profile, curvature_profile, alpha=0.3, color='purple')
        ax3.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Curvature (Å⁻¹)', fontsize=12, fontweight='bold')
        ax3.set_title('Membrane Curvature Profile', fontsize=13, fontweight='bold')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['2d'], 'curvature_profile.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_radius_of_curvature_profile(self):
        """ACTUAL radius-of-curvature (R = 2/|c1|) cross-section profile
        (individual file) -- the direct real-space companion to the
        curvature profile above."""
        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :]
        r_profile = self.radius_of_curvature[y_idx, :]

        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(x_profile, r_profile, color='darkslateblue', linewidth=2.5)
        ax.fill_between(x_profile, r_profile, alpha=0.3, color='darkslateblue')
        ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Radius of Curvature (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Membrane Radius-of-Curvature Profile (Midplane)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['2d'], 'radius_of_curvature_profile.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    # ==================== SPONTANEOUS CURVATURE FRUSTRATION PLOTS (Marsh 2007) ====================

    def plot_curvature_frustration(self):
        """Plot the actual leaflet curvature c1, the composition-weighted
        spontaneous curvature c0, and the resulting elastic bending /
        frustration energy density maps, following Marsh (2007).
        Each panel is saved as its own individual file."""
        extent = [self.xi.min(), self.xi.max(), self.yi.min(), self.yi.max()]

        # c1 maps (actual curvature, Å^-1)
        c1_min = min(np.nanmin(self.curvature_u), np.nanmin(self.curvature_l))
        c1_max = max(np.nanmax(self.curvature_u), np.nanmax(self.curvature_l))
        c1_lim = max(abs(c1_min), abs(c1_max))

        fig, ax1 = plt.subplots(figsize=(8, 7))
        im1 = ax1.imshow(self.curvature_u, extent=extent, origin='lower', cmap='RdBu_r',
                          vmin=-c1_lim, vmax=c1_lim, aspect='equal')
        ax1.set_title(f'Upper leaflet c1  (mean {np.nanmean(self.curvature_u):+.4f} Å⁻¹)',
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (Å)'); ax1.set_ylabel('Y (Å)')
        plt.colorbar(im1, ax=ax1, label='c1 (Å⁻¹)', shrink=0.85)
        plt.tight_layout()
        outpath = os.path.join(self.subdirs['frustration'], 'c1_map_upper.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

        fig, ax2 = plt.subplots(figsize=(8, 7))
        im2 = ax2.imshow(self.curvature_l, extent=extent, origin='lower', cmap='RdBu_r',
                          vmin=-c1_lim, vmax=c1_lim, aspect='equal')
        ax2.set_title(f'Lower leaflet c1  (mean {np.nanmean(self.curvature_l):+.4f} Å⁻¹)',
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (Å)'); ax2.set_ylabel('Y (Å)')
        plt.colorbar(im2, ax=ax2, label='c1 (Å⁻¹)', shrink=0.85)
        plt.tight_layout()
        outpath = os.path.join(self.subdirs['frustration'], 'c1_map_lower.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

        # Summary text panel
        fig, ax3 = plt.subplots(figsize=(6, 7))
        ax3.axis('off')
        summary = (
            "Marsh (2007) Curvature Frustration\n"
            "gc = (kc/2)(c1 - c0)^2\n"
            "g_frust = -kc·c0·c1\n\n"
            f"kc = {self.kc_kbt:.2f} kBT ({self.kc_kjmol:.2f} kJ/mol)\n\n"
            f"Upper leaflet c0 = {self.c0_upper:+.4f} Å⁻¹\n"
            f"Lower leaflet c0 = {self.c0_lower:+.4f} Å⁻¹\n\n"
            f"Total elastic energy gc:\n"
            f"  upper = {self.total_gc_upper_kbt:.1f} kBT\n"
            f"  lower = {self.total_gc_lower_kbt:.1f} kBT\n"
            f"  total = {self.total_gc_kbt:.1f} kBT\n\n"
            f"Net frustration energy:\n"
            f"  {self.total_frustration_kbt:.1f} kBT\n\n"
            f"Flat-bilayer baseline gc:\n"
            f"  upper = {self.gc_flat_upper * np.ptp(self.xi) * np.ptp(self.yi):.1f} kBT\n"
            f"  lower = {self.gc_flat_lower * np.ptp(self.xi) * np.ptp(self.yi):.1f} kBT"
        )
        ax3.text(0.02, 0.98, summary, transform=ax3.transAxes, fontsize=10.5,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                           edgecolor='black', linewidth=1.5))
        plt.tight_layout()
        outpath = os.path.join(self.subdirs['frustration'], 'frustration_summary.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

        # gc energy density maps
        gc_max = max(np.nanmax(self.gc_upper), np.nanmax(self.gc_lower))

        fig, ax4 = plt.subplots(figsize=(8, 7))
        im4 = ax4.imshow(self.gc_upper, extent=extent, origin='lower', cmap='inferno',
                          vmin=0, vmax=gc_max, aspect='equal')
        ax4.set_title('Upper leaflet gc (elastic energy density)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('X (Å)'); ax4.set_ylabel('Y (Å)')
        plt.colorbar(im4, ax=ax4, label='gc (kBT/Å²)', shrink=0.85)
        plt.tight_layout()
        outpath = os.path.join(self.subdirs['frustration'], 'gc_map_upper.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

        fig, ax5 = plt.subplots(figsize=(8, 7))
        im5 = ax5.imshow(self.gc_lower, extent=extent, origin='lower', cmap='inferno',
                          vmin=0, vmax=gc_max, aspect='equal')
        ax5.set_title('Lower leaflet gc (elastic energy density)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('X (Å)'); ax5.set_ylabel('Y (Å)')
        plt.colorbar(im5, ax=ax5, label='gc (kBT/Å²)', shrink=0.85)
        plt.tight_layout()
        outpath = os.path.join(self.subdirs['frustration'], 'gc_map_lower.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

        # Cross-section of c1 vs c0 (upper/lower leaflet, middle row)
        fig, ax6 = plt.subplots(figsize=(9, 7))
        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :]
        ax6.plot(x_profile, self.curvature_u[y_idx, :], 'b-', linewidth=2.5, label='c1 upper (actual)')
        ax6.plot(x_profile, self.curvature_l[y_idx, :], 'r-', linewidth=2.5, label='c1 lower (actual)')
        ax6.axhline(self.c0_upper, color='b', linestyle='--', linewidth=1.8, alpha=0.7, label='c0 upper')
        ax6.axhline(self.c0_lower, color='r', linestyle='--', linewidth=1.8, alpha=0.7, label='c0 lower')
        ax6.axhline(0, color='k', linestyle=':', alpha=0.4)
        ax6.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Curvature (Å⁻¹)', fontsize=12, fontweight='bold')
        ax6.set_title('Actual curvature c1 vs spontaneous curvature c0', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9, loc='best')
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        outpath = os.path.join(self.subdirs['frustration'], 'c1_vs_c0_profile.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_frustration_energy_timeseries(self):
        """Plot the total elastic bending/frustration energy over time"""
        if len(self.time_points) == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        time_ns = self.time_points / 1000.0  # ps -> ns
        ax.plot(time_ns, self.frustration_energy_over_time, 'o-', color='darkorange', linewidth=2.5)
        ax.fill_between(time_ns, self.frustration_energy_over_time, alpha=0.25, color='darkorange')
        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total elastic bending energy, gc (kBT)', fontsize=12, fontweight='bold')
        ax.set_title('Membrane Curvature-Frustration Energy Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        outpath = os.path.join(self.subdirs['frustration'], 'frustration_energy_timeseries.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    # ==================== TIME-SERIES ANALYSIS (individual files) ====================

    def plot_radius_of_curvature_over_time_smoothed(self):
        """Radius of curvature over time (midplane, differential-geometry
        estimate) with a 5-frame moving average overlaid (individual file)."""
        if len(self.time_points) == 0:
            return
        time_ns = self.time_points / 1000.0

        fig, ax = plt.subplots(figsize=(11, 6.5))
        rad_curv_list = self.radius_mid_over_time

        ax.plot(time_ns, rad_curv_list, 'b-', linewidth=2, label='Instantaneous')
        ax.fill_between(time_ns, rad_curv_list, alpha=0.3)

        if len(rad_curv_list) > 5:
            rad_curv_clean = np.nan_to_num(rad_curv_list, nan=np.nanmean(rad_curv_list))
            moving_avg = np.convolve(rad_curv_clean, np.ones(5) / 5, mode='same')
            ax.plot(time_ns, moving_avg, 'r--', linewidth=2.5, label='Moving average (5-frame)')

        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Radius of Curvature (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Radius of Curvature Over Time (Midplane, smoothed)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['timeseries'], 'radius_of_curvature_over_time_smoothed.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_radius_of_curvature_upper_lower_over_time(self):
        """Per-leaflet radius of curvature over time (individual file)."""
        if len(self.time_points) == 0:
            return
        time_ns = self.time_points / 1000.0

        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(time_ns, self.radius_u_over_time, 'o-', color='steelblue',
                linewidth=2, markersize=5, label='Upper leaflet')
        ax.plot(time_ns, self.radius_l_over_time, 's-', color='firebrick',
                linewidth=2, markersize=5, label='Lower leaflet')
        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Radius of Curvature (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Per-Leaflet Radius of Curvature Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['timeseries'], 'radius_of_curvature_upper_lower_over_time.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_thickness_over_time(self):
        """Membrane thickness over time (individual file)."""
        if len(self.time_points) == 0:
            return
        time_ns = self.time_points / 1000.0

        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(time_ns, self.thickness_over_time, 'g-', linewidth=2)
        ax.fill_between(time_ns, self.thickness_over_time, alpha=0.3, color='green')
        ax.axhline(y=np.mean(self.thickness_over_time), color='r', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(self.thickness_over_time):.2f}')
        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Thickness (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Membrane Thickness Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['timeseries'], 'thickness_over_time.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_deflection_over_time(self):
        """Lipid deflection over time (individual file)."""
        if len(self.time_points) == 0:
            return
        time_ns = self.time_points / 1000.0

        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(time_ns, self.deflection_over_time, 'orange', linewidth=2)
        ax.fill_between(time_ns, self.deflection_over_time, alpha=0.3, color='orange')
        ax.axhline(y=np.mean(self.deflection_over_time), color='r', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(self.deflection_over_time):.2f}')
        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Deflection (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Lipid Deflection Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['timeseries'], 'deflection_over_time.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_timeseries_stats_summary(self):
        """Time-series summary statistics panel (individual file)."""
        if len(self.time_points) == 0:
            return
        time_ns = self.time_points / 1000.0

        rad_curv_clean = np.nan_to_num(self.radius_mid_over_time, nan=np.nanmean(self.radius_mid_over_time))

        fig, ax = plt.subplots(figsize=(8, 6.5))
        ax.axis('off')
        stats_text = f"""TIME-SERIES STATISTICS

Radius of Curvature (midplane):
  Mean: {np.nanmean(rad_curv_clean):.1f} Å
  Min: {np.nanmin(rad_curv_clean):.1f} Å
  Max: {np.nanmax(rad_curv_clean):.1f} Å
  Std: {np.nanstd(rad_curv_clean):.1f} Å

Thickness:
  Mean: {np.mean(self.thickness_over_time):.2f} Å
  Std: {np.std(self.thickness_over_time):.2f} Å

Deflection:
  Mean: {np.mean(self.deflection_over_time):.2f} Å
  Std: {np.std(self.deflection_over_time):.2f} Å

Time Range: {time_ns[0]:.2f} - {time_ns[-1]:.2f} ns
Total Frames: {len(self.time_points)}
"""
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2))
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['timeseries'], 'timeseries_stats_summary.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_3d_interactive_topside(self):
        """Create interactive 3D top and side views"""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]],
            subplot_titles=('<b>Top View (XY)</b>', '<b>Side View (XZ)</b>'),
            horizontal_spacing=0.15
        )
        
        # Get middle cross-section
        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :]
        zu_profile = self.zu[y_idx, :]
        zl_profile = self.zl[y_idx, :]
        
        # TOP VIEW
        # Upper leaflet contours
        for level in np.linspace(self.zu.min(), self.zu.max(), 8):
            contour = plt.contour(self.xi, self.yi, self.zu, levels=[level], colors='blue', linewidths=1)
            # matplotlib >=3.8 removed QuadContourSet.collections; get_paths()
            # on the contour set itself (or allsegs) works across versions.
            if hasattr(contour, 'get_paths'):
                paths = contour.get_paths()
            else:  # pragma: no cover - older matplotlib fallback
                paths = [p for coll in contour.collections for p in coll.get_paths()]
            for path in paths:
                vertices = path.vertices
                if len(vertices) > 0:
                    fig.add_trace(go.Scatter(
                        x=vertices[:, 0], y=vertices[:, 1],
                        mode='lines', line=dict(color='blue', width=2),
                        showlegend=False, name='Upper'
                    ), row=1, col=1)
        plt.close()
        
        # SIDE VIEW
        fig.add_trace(go.Scatter(
            x=x_profile, y=zu_profile,
            fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.3)',
            line=dict(color='blue', width=3),
            name='Upper leaflet',
            showlegend=True
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=x_profile, y=zl_profile,
            fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=3),
            name='Lower leaflet',
            showlegend=True
        ), row=1, col=2)
        
        fig.update_xaxes(title_text="X (Å)", row=1, col=1)
        fig.update_yaxes(title_text="Y (Å)", row=1, col=1)
        fig.update_xaxes(title_text="X (Å)", row=1, col=2)
        fig.update_yaxes(title_text="Z (Å)", row=1, col=2)
        
        fig.update_layout(
            title=dict(text='<b>3D Interactive Top & Side Views</b>', x=0.5, xanchor='center', font=dict(size=18)),
            width=1400, height=600,
            hovermode='closest'
        )
        
        outpath = os.path.join(self.subdirs['3d'], 'interactive_top_side_view.html')
        fig.write_html(outpath)
        print(f"✓ Saved: {outpath}")
        
    # ==================== 2D/3D STANDARD PLOTS ====================
    
    def _plot_heatmap_single(self, data, cmap, title, filename, vmin=None, vmax=None, cbar_label=None):
        """Save a single heatmap panel as its own file."""
        fig, ax0 = plt.subplots(figsize=(9, 8))
        im = ax0.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax,
                       extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()])
        contours = ax0.contour(self.xi, self.yi, data, levels=12, colors='white', alpha=0.2)
        ax0.clabel(contours, inline=True, fontsize=8)
        plt.colorbar(im, ax=ax0, label=cbar_label or title, pad=0.02)
        ax0.set_xlabel('X (Å)', fontweight='bold')
        ax0.set_ylabel('Y (Å)', fontweight='bold')
        ax0.set_title(f'{title} Heatmap', fontweight='bold', fontsize=14)
        ax0.grid(True, alpha=0.1)
        plt.tight_layout()
        outpath = os.path.join(self.subdirs['2d'], filename)
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def _plot_histogram_single(self, data, title, filename):
        """Save a single distribution-histogram panel as its own file."""
        fig, ax1 = plt.subplots(figsize=(9, 6.5))
        ax1.hist(data.flatten(), bins=60, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(data), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(data):.2f}')
        ax1.set_xlabel(title, fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title(f'{title} Distribution', fontweight='bold', fontsize=14)
        ax1.legend(fontsize=10, framealpha=0.9)
        ax1.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        outpath = os.path.join(self.subdirs['2d'], filename)
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def _plot_stats_single(self, data, title, filename):
        """Save a single statistics-summary panel as its own file."""
        fig, ax2 = plt.subplots(figsize=(6, 5))
        ax2.axis('off')
        stats_text = f"""{title.upper()} STATISTICS

Mean: {np.mean(data):.3f}
Median: {np.median(data):.3f}
Std Dev: {np.std(data):.3f}
Min: {np.min(data):.3f}
Max: {np.max(data):.3f}"""
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))
        plt.tight_layout()
        outpath = os.path.join(self.subdirs['2d'], filename)
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_2d_standard_all(self):
        """Create standard 2D analysis plots -- heatmap, histogram, and
        statistics for each quantity, each saved as its own individual file
        (no combined panels)."""
        analyses = [
            (self.thickness, 'RdYlGn_r', 'Thickness', 'thickness', None, None),
            (self.curvature_norm, 'RdBu_r', 'Curvature', 'curvature', None, None),
            (self.water_density, 'Blues', 'Water Density', 'water_density', None, None),
        ]

        for data, cmap, title, stem, vmin, vmax in analyses:
            self._plot_heatmap_single(data, cmap, title, f'{stem}_heatmap.png', vmin, vmax)
            self._plot_histogram_single(data, title, f'{stem}_histogram.png')
            self._plot_stats_single(data, title, f'{stem}_stats.png')

    # ==================== ACTUAL RADIUS-OF-CURVATURE PLOTS (2D + 3D) ====================

    def plot_radius_of_curvature_2d(self):
        """2D radius-of-curvature (R = 2/|c1|) heatmap, histogram, and
        statistics -- the ACTUAL, real-space radius-of-curvature analysis,
        each saved individually. Display is clipped at the 97th percentile
        of the (uncapped) finite values so near-flat regions -- which
        formally diverge -- don't wash out the colorbar; the underlying
        data used for histogram/statistics is the raw capped R map."""
        finite_vals = self.radius_of_curvature[self.radius_of_curvature < np.nanmax(self.radius_of_curvature)]
        vmax = np.nanpercentile(finite_vals, 97) if finite_vals.size else np.nanmax(self.radius_of_curvature)

        self._plot_heatmap_single(self.radius_of_curvature, 'viridis', 'Radius of Curvature',
                                   'radius_of_curvature_heatmap.png', vmin=0, vmax=vmax,
                                   cbar_label='Radius of Curvature (Å)')
        self._plot_histogram_single(self.radius_of_curvature, 'Radius of Curvature (Å)',
                                     'radius_of_curvature_histogram.png')
        self._plot_stats_single(self.radius_of_curvature, 'Radius of Curvature',
                                 'radius_of_curvature_stats.png')

    def plot_radius_of_curvature_3d(self):
        """3D surfaces (midplane, upper leaflet, lower leaflet) colored by
        the ACTUAL local radius of curvature R = 2/|c1|, each saved as its
        own individual interactive HTML file."""
        finite_vals = self.radius_of_curvature[self.radius_of_curvature < np.nanmax(self.radius_of_curvature)]
        rmax = np.nanpercentile(finite_vals, 97) if finite_vals.size else np.nanmax(self.radius_of_curvature)

        surfaces = [
            (self.zi, self.radius_of_curvature, 'Midplane', 'radius_of_curvature_3d_midplane.html'),
            (self.zu, self.radius_of_curvature_u, 'Upper Leaflet', 'radius_of_curvature_3d_upper.html'),
            (self.zl, self.radius_of_curvature_l, 'Lower Leaflet', 'radius_of_curvature_3d_lower.html'),
        ]
        for z_surf, r_surf, label, fname in surfaces:
            fig = go.Figure(data=[go.Surface(
                x=self.xi, y=self.yi, z=z_surf, surfacecolor=r_surf,
                colorscale='Viridis', cmin=0, cmax=rmax, opacity=0.95,
                colorbar=dict(title='R (Å)')
            )])
            fig.update_layout(
                title=dict(text=f'<b>3D Radius of Curvature -- {label}</b>', x=0.5, xanchor='center', font=dict(size=18)),
                scene=dict(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)',
                           camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))),
                width=1000, height=800
            )
            outpath = os.path.join(self.subdirs['3d'], fname)
            fig.write_html(outpath)
            print(f"✓ Saved: {outpath}")

    def plot_3d_standard_all(self):
        """Create standard 3D plots -- each leaflet/quantity saved as its
        own individual interactive HTML file (no combined subplot panels)."""
        # ---------------- Thickness / leaflet surfaces ----------------
        fig = go.Figure(data=[go.Surface(x=self.xi, y=self.yi, z=self.zu, colorscale='Blues', opacity=0.95)])
        fig.update_layout(title=dict(text='<b>3D Membrane Surface -- Upper Leaflet</b>', x=0.5, xanchor='center', font=dict(size=18)),
                           scene=dict(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)'),
                           width=1000, height=800)
        outpath = os.path.join(self.subdirs['3d'], 'thickness_3d_upper.html')
        fig.write_html(outpath)
        print(f"✓ Saved: {outpath}")

        fig = go.Figure(data=[go.Surface(x=self.xi, y=self.yi, z=self.zl, colorscale='Reds', opacity=0.95)])
        fig.update_layout(title=dict(text='<b>3D Membrane Surface -- Lower Leaflet</b>', x=0.5, xanchor='center', font=dict(size=18)),
                           scene=dict(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)'),
                           width=1000, height=800)
        outpath = os.path.join(self.subdirs['3d'], 'thickness_3d_lower.html')
        fig.write_html(outpath)
        print(f"✓ Saved: {outpath}")

        # ---------------- Curvature (c1) ----------------
        curv_min = min(self.curvature_u.min(), self.curvature_l.min())
        curv_max = max(self.curvature_u.max(), self.curvature_l.max())

        fig = go.Figure(data=[go.Surface(
            x=self.xi, y=self.yi, z=self.zu, surfacecolor=self.curvature_u,
            colorscale='RdBu', cmin=curv_min, cmax=curv_max, opacity=0.95,
            colorbar=dict(title='c1 (Å⁻¹)'))])
        fig.update_layout(title=dict(text='<b>3D Membrane Curvature -- Upper Leaflet</b>', x=0.5, xanchor='center', font=dict(size=18)),
                           scene=dict(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)'),
                           width=1000, height=800)
        outpath = os.path.join(self.subdirs['3d'], 'curvature_3d_upper.html')
        fig.write_html(outpath)
        print(f"✓ Saved: {outpath}")

        fig = go.Figure(data=[go.Surface(
            x=self.xi, y=self.yi, z=self.zl, surfacecolor=self.curvature_l,
            colorscale='RdBu', cmin=curv_min, cmax=curv_max, opacity=0.95,
            colorbar=dict(title='c1 (Å⁻¹)'))])
        fig.update_layout(title=dict(text='<b>3D Membrane Curvature -- Lower Leaflet</b>', x=0.5, xanchor='center', font=dict(size=18)),
                           scene=dict(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)'),
                           width=1000, height=800)
        outpath = os.path.join(self.subdirs['3d'], 'curvature_3d_lower.html')
        fig.write_html(outpath)
        print(f"✓ Saved: {outpath}")

        # ---------------- Deflection ----------------
        defl_min = min(self.deflection_u.min(), self.deflection_l.min())
        defl_max = max(self.deflection_u.max(), self.deflection_l.max())

        fig = go.Figure(data=[go.Surface(
            x=self.xi, y=self.yi, z=self.zu, surfacecolor=self.deflection_u,
            colorscale='Viridis', cmin=defl_min, cmax=defl_max, opacity=0.95,
            colorbar=dict(title='Deflection (Å)'))])
        fig.update_layout(title=dict(text='<b>3D Phospholipid Deflection -- Upper Leaflet</b>', x=0.5, xanchor='center', font=dict(size=18)),
                           scene=dict(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)'),
                           width=1000, height=800)
        outpath = os.path.join(self.subdirs['3d'], 'deflection_3d_upper.html')
        fig.write_html(outpath)
        print(f"✓ Saved: {outpath}")

        fig = go.Figure(data=[go.Surface(
            x=self.xi, y=self.yi, z=self.zl, surfacecolor=self.deflection_l,
            colorscale='Viridis', cmin=defl_min, cmax=defl_max, opacity=0.95,
            colorbar=dict(title='Deflection (Å)'))])
        fig.update_layout(title=dict(text='<b>3D Phospholipid Deflection -- Lower Leaflet</b>', x=0.5, xanchor='center', font=dict(size=18)),
                           scene=dict(xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)'),
                           width=1000, height=800)
        outpath = os.path.join(self.subdirs['3d'], 'deflection_3d_lower.html')
        fig.write_html(outpath)
        print(f"✓ Saved: {outpath}")

        # ---------------- Pore ----------------
        fig = go.Figure()
        pore_color = np.where(self.pore_mask, self.pore_radius_map, np.nan)
        fig.add_trace(go.Surface(
            x=self.xi, y=self.yi, z=self.thickness,
            surfacecolor=pore_color,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title='Pore Radius (Å)'),
            opacity=0.95
        ))
        fig.update_layout(
            title=dict(text='<b>3D Pore Radius Analysis</b>', x=0.5, xanchor='center', font=dict(size=18)),
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Thickness (Å)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=1000, height=800
        )
        outpath = os.path.join(self.subdirs['3d'], 'pore_3d_analysis.html')
        fig.write_html(outpath)
        print(f"✓ Saved: {outpath}")

        # ---------------- ACTUAL radius of curvature (new) ----------------
        self.plot_radius_of_curvature_3d()
        
    # ==================== PACSIN1 / F-BAR STYLE CURVATURE ANALYSIS ====================
    # Reproduces the core analyses of Mahmood, Noguchi & Okazaki -- curvature
    # induction and sensing of the F-BAR protein Pacsin1 on lipid membranes
    # via MD simulations: (1) a discrete surface-normal-vector / Δφ picture
    # of local membrane curvature (matching the reference schematic with
    # S(x) vectors and angle Δφ), (2) an independent circle-fit radius of
    # curvature, (3) curvature INDUCTION over time, and (4) curvature
    # SENSING -- whether the protein preferentially sits on/moves toward
    # more curved membrane regions than the leaflet bulk.

    def _export_timeseries_csv(self):
        """Export every collected time-series quantity to a single CSV and
        a JSON summary in the data-exports subfolder."""
        csv_path = os.path.join(self.subdirs['data'], 'timeseries_data.csv')
        n = len(self.time_points)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'time_ps', 'time_ns', 'mean_abs_curvature_A-1',
                'mean_curvature_upper_A-1', 'mean_curvature_lower_A-1',
                'radius_of_curvature_midplane_A', 'radius_of_curvature_upper_A',
                'radius_of_curvature_lower_A', 'arc_fit_radius_A',
                'mean_thickness_A', 'mean_deflection_A',
                'total_elastic_energy_kBT',
                'protein_local_curvature_A-1', 'protein_bulk_curvature_A-1',
                'protein_insertion_depth_A', 'protein_com_z_A',
            ])
            for i in range(n):
                writer.writerow([
                    self.time_points[i], self.time_points[i] / 1000.0,
                    self.curvature_over_time[i], self.curvature_u_over_time[i],
                    self.curvature_l_over_time[i],
                    self.radius_mid_over_time[i], self.radius_u_over_time[i],
                    self.radius_l_over_time[i], self.arc_radius_over_time[i],
                    self.thickness_over_time[i],
                    self.deflection_over_time[i], self.frustration_energy_over_time[i],
                    self.protein_local_curvature_over_time[i],
                    self.protein_bulk_curvature_over_time[i],
                    self.protein_insertion_depth_over_time[i], self.protein_com_z_over_time[i],
                ])
        print(f"✓ Saved: {csv_path}")

        summary = {
            'kc_kbt': self.kc_kbt,
            'c0_upper': getattr(self, 'c0_upper', None),
            'c0_lower': getattr(self, 'c0_lower', None),
            'mean_arc_fit_radius_A': float(np.nanmean(self.arc_radius_over_time)) if n else None,
            'has_protein': self.has_protein,
        }
        json_path = os.path.join(self.subdirs['data'], 'analysis_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved: {json_path}")

    def plot_normal_vector_curvature_schematic(self):
        """Recreate a Pacsin1/F-BAR-style curvature schematic: the leaflet
        mid-surface cross-section arc (dashed), evenly-spaced points S(x)
        along it, their local outward surface-normal vectors, and the
        angle Δφ marked between adjacent normals -- the same discrete
        geometric picture used to characterize curvature induction/sensing
        in the reference figure."""
        self.calculate_normal_vector_curvature()

        fig, ax = plt.subplots(figsize=(11, 7))

        s_full = self.arc_profile_s
        z_full = self.arc_profile_z
        order = np.argsort(s_full)
        ax.plot(s_full[order], z_full[order], 'k--', linewidth=1.8, alpha=0.8,
                label='Leaflet mid-surface (arc)')

        s_pts = self.normal_vector_s
        z_pts = self.normal_vector_z
        normals = self.normal_vector_dirs
        arrow_len = 0.12 * (np.nanmax(s_full) - np.nanmin(s_full))

        colors = cm.coolwarm(np.linspace(0, 1, len(s_pts)))
        for i in range(len(s_pts)):
            ax.plot(s_pts[i], z_pts[i], 'o', color='goldenrod', markersize=11,
                     markeredgecolor='black', zorder=5)
            ax.annotate('', xy=(s_pts[i] + normals[i, 0] * arrow_len,
                                 z_pts[i] + normals[i, 1] * arrow_len),
                         xytext=(s_pts[i], z_pts[i]),
                         arrowprops=dict(arrowstyle='-|>', color=colors[i], lw=2.2),
                         zorder=6)
            if i < len(s_pts) - 1:
                ax.annotate('S(x)', xy=(s_pts[i] + normals[i, 0] * arrow_len * 1.08,
                                         z_pts[i] + normals[i, 1] * arrow_len * 1.08),
                            fontsize=9, color=colors[i], fontweight='bold', ha='center')

        # Mark Δφ between the two central vectors
        if len(s_pts) >= 3:
            c = len(s_pts) // 2
            v1 = normals[c - 1] * arrow_len * 0.55
            v2 = normals[c] * arrow_len * 0.55
            origin = ((s_pts[c - 1] + s_pts[c]) / 2, (z_pts[c - 1] + z_pts[c]) / 2)
            ang1 = np.degrees(np.arctan2(v1[1], v1[0]))
            ang2 = np.degrees(np.arctan2(v2[1], v2[0]))
            arc = Arc(origin, arrow_len * 0.9, arrow_len * 0.9,
                      angle=0, theta1=min(ang1, ang2), theta2=max(ang1, ang2),
                      color='black', linewidth=1.8)
            ax.add_patch(arc)
            ax.text(origin[0], origin[1] - arrow_len * 0.55, 'Δφ', fontsize=13,
                    fontweight='bold', ha='center')

        mean_local_R = np.nanmean(self.normal_vector_local_R)
        ax.text(0.02, 0.03,
                f"Circle-fit R = {self.arc_fit_radius:.1f} Å\n"
                f"Mean S(x)-to-S(x) local R = {mean_local_R:.1f} Å\n"
                f"Mean |Δφ| = {np.nanmean(np.abs(self.normal_vector_dphi)):.4f} rad",
                transform=ax.transAxes, fontsize=10.5, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                          edgecolor='black'), va='bottom')

        ax.set_xlabel(f'{self.arc_axis.upper()} (Å)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Z (Å)', fontsize=13, fontweight='bold')
        ax.set_title('Membrane Curvature: Surface Normal Vectors S(x) and Δφ\n'
                      '(Pacsin1/F-BAR-style curvature-sensing schematic)',
                      fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('auto')

        outpath = os.path.join(self.subdirs['pacsin1'], 'normal_vector_curvature_schematic.png')
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_curvature_over_time_standalone(self):
        """Standalone 'curvature induction over time' plot: mean |c1|
        (midplane), plus upper- and lower-leaflet curvature, vs time."""
        if len(self.time_points) == 0:
            return
        time_ns = self.time_points / 1000.0

        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(time_ns, self.curvature_over_time, 'k-', linewidth=2.2,
                label='Midplane |c1| (mean)')
        ax.plot(time_ns, np.abs(self.curvature_u_over_time), 'b-', linewidth=2,
                alpha=0.85, label='Upper leaflet |c1| (mean)')
        ax.plot(time_ns, np.abs(self.curvature_l_over_time), 'r-', linewidth=2,
                alpha=0.85, label='Lower leaflet |c1| (mean)')
        ax.fill_between(time_ns, self.curvature_over_time, alpha=0.15, color='gray')

        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Curvature, c1 = 2H (Å⁻¹)', fontsize=12, fontweight='bold')
        ax.set_title('Membrane Curvature Induction Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['timeseries'], 'curvature_over_time.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_radius_of_curvature_over_time_standalone(self):
        """Standalone radius-of-curvature-over-time plot, comparing the
        differential-geometry estimate (R = 2/|c1|) against the
        independent circle-fit estimate (arc_fit_radius)."""
        if len(self.time_points) == 0:
            return
        time_ns = self.time_points / 1000.0

        with np.errstate(divide='ignore', invalid='ignore'):
            R_diffgeo = np.where(self.curvature_over_time > 1e-6,
                                  2.0 / self.curvature_over_time, np.nan)

        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(time_ns, R_diffgeo, 'o-', color='steelblue', linewidth=2,
                markersize=5, label='R from differential geometry (2/|c1|)')
        ax.plot(time_ns, self.arc_radius_over_time, 's--', color='darkorange',
                linewidth=2, markersize=5, label='R from circle fit (Pacsin1-style)')
        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Radius of Curvature (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Radius of Curvature Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['timeseries'], 'radius_of_curvature_over_time.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_bead_scatter_fit(self):
        """Raw lipid bead X-Z scatter for a cross-section slab, with the
        fitted mid-surface arc overlaid (individual file), in the style of
        the F-BAR/Pacsin1-type literature figure. Lengths in Angstrom (Å)."""
        axis_idx = 1 if self.arc_axis == 'y' else 0
        coords_all = self.lipid_atoms.positions
        s_all = coords_all[:, axis_idx]
        z_all = coords_all[:, 2]

        # Keep a slab centered on the arc line (perpendicular coordinate
        # within a few grid cells of the center) so the scatter reflects a
        # genuine cross-section, as in the reference figure, rather than
        # projecting the full 3D system onto one axis.
        perp_idx = 0 if axis_idx == 1 else 1
        perp_all = coords_all[:, perp_idx]
        perp_center = np.median(perp_all)
        slab_halfwidth = max(self.dx, self.dy) * 3
        in_slab = np.abs(perp_all - perp_center) <= slab_halfwidth

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(s_all[in_slab], z_all[in_slab], s=4, c='royalblue', alpha=0.45,
                   linewidths=0, label='Lipid beads (raw)')

        s_fit = self.arc_profile_s
        z_fit = self.arc_profile_z
        order = np.argsort(s_fit)
        ax.plot(s_fit[order], z_fit[order], color='seagreen', linewidth=3.5,
                label='Fitted mid-surface (arc)')

        ax.set_xlabel(f'{self.arc_axis.upper()} (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Membrane Cross-Section: Beads + Fitted Curvature', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9.5, loc='upper right')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['pacsin1'], 'bead_scatter_fit.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_radius_of_curvature_raw_timeseries(self):
        """Raw (unsmoothed) circle-fit radius of curvature vs time
        (individual file) -- matches the noisy R(t) trace style of the
        reference literature figure. Time in ns, R in Å."""
        if len(self.time_points) == 0:
            print("⚠ Skipping raw radius-of-curvature timeseries (no time-series data).")
            return
        time_ns = self.time_points / 1000.0

        fig, ax2 = plt.subplots(figsize=(10, 6.5))
        ax2.plot(time_ns, self.arc_radius_over_time, '-', color='blue', linewidth=1.3)
        ax2.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Radius of curvature (Å)', fontsize=12, fontweight='bold')
        ax2.set_title('Radius of Curvature Over Time (circle fit, raw)', fontsize=13, fontweight='bold')
        ax2.set_xlim(time_ns.min(), time_ns.max())
        ax2.grid(True, alpha=0.25)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['pacsin1'], 'radius_of_curvature_raw_timeseries.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_protein_local_vs_bulk_curvature(self):
        """Curvature-SENSING analysis: local (under-protein-footprint) vs
        bulk (rest-of-leaflet) curvature over time (individual file). If
        local curvature consistently tracks above bulk curvature, the
        protein is preferentially associated with more curved membrane --
        the MD signature of curvature sensing reported for Pacsin1/F-BAR
        domains."""
        if not self.has_protein or len(self.time_points) == 0:
            print("⚠ Skipping local-vs-bulk curvature plot (no protein selected).")
            return
        time_ns = self.time_points / 1000.0

        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(time_ns, self.protein_local_curvature_over_time, 'o-',
                color='crimson', linewidth=2, markersize=5,
                label=f'Local (under protein, r={self.protein_probe_radius:.0f} Å)')
        ax.plot(time_ns, self.protein_bulk_curvature_over_time, 's--',
                color='gray', linewidth=2, markersize=5, label='Bulk (rest of leaflet)')
        ax.axhline(0, color='k', linestyle=':', alpha=0.4)
        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Curvature, c1 (Å⁻¹)', fontsize=12, fontweight='bold')
        ax.set_title('Curvature Sensing: Local (Protein Footprint) vs Bulk Membrane Curvature',
                      fontsize=13, fontweight='bold')
        ax.legend(fontsize=10.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['pacsin1'], 'protein_local_vs_bulk_curvature.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_protein_insertion_depth_over_time(self):
        """Protein center-of-mass position relative to the local bilayer
        midplane, over time (individual file)."""
        if not self.has_protein or len(self.time_points) == 0:
            print("⚠ Skipping protein insertion-depth plot (no protein selected).")
            return
        time_ns = self.time_points / 1000.0

        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(time_ns, self.protein_insertion_depth_over_time, '^-',
                color='darkgreen', linewidth=2, markersize=5)
        ax.fill_between(time_ns, self.protein_insertion_depth_over_time, alpha=0.2, color='darkgreen')
        ax.axhline(0, color='k', linestyle=':', alpha=0.4, label='Local bilayer midplane')
        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Protein Insertion Depth (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Protein Center-of-Mass Position Relative to Local Bilayer Midplane',
                      fontsize=13, fontweight='bold')
        ax.legend(fontsize=10.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        outpath = os.path.join(self.subdirs['pacsin1'], 'protein_insertion_depth_over_time.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    def plot_protein_footprint_map(self):
        """Show the final-frame curvature map with the protein footprint
        circle overlaid, so the curvature-sensing region is visible
        directly on the membrane surface."""
        if not self.has_protein:
            print("⚠ Skipping protein footprint map (no protein selected).")
            return

        fig, ax = plt.subplots(figsize=(9, 8))
        extent = [self.xi.min(), self.xi.max(), self.yi.min(), self.yi.max()]
        c_lim = np.nanmax(np.abs(self.curvature))
        im = ax.imshow(self.curvature, extent=extent, origin='lower', cmap='RdBu_r',
                        vmin=-c_lim, vmax=c_lim, aspect='equal')
        plt.colorbar(im, ax=ax, label='Curvature, c1 (Å⁻¹)', shrink=0.85)

        prot_com = self.protein.center_of_mass()
        circle = plt.Circle((prot_com[0], prot_com[1]), self.protein_probe_radius,
                             fill=False, edgecolor='lime', linewidth=2.5,
                             label='Protein footprint')
        ax.add_patch(circle)
        ax.plot(prot_com[0], prot_com[1], '*', color='lime', markersize=18,
                markeredgecolor='black', label='Protein COM (XY)')

        ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Membrane Curvature Map with Protein Footprint\n(final frame)',
                      fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')

        outpath = os.path.join(self.subdirs['pacsin1'], 'protein_footprint_curvature_map.png')
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {outpath}")
        plt.close()

    # ==================== MAIN EXECUTION ====================
    
    def run_analysis(self):
        """Execute complete analysis"""
        # Single frame analysis (uses the LAST frame of the trajectory,
        # self.ts, set in __init__)
        self.u.trajectory[-1]
        self.extract_surface_data()
        self.create_surface_grid()
        self.calculate_curvature()
        self.calculate_lipid_deflection()
        self.calculate_arc_radius_of_curvature()

        print("\n" + "="*80)
        print("SPONTANEOUS CURVATURE FRUSTRATION (Marsh, Biophys J 2007;93:3884-3899)")
        print("="*80)
        self.calculate_curvature_frustration()

        self.analyze_pore_radius()
        self.analyze_water_permeation()
        
        # Time-series analysis (also computes arc-fit R(t) and, if a
        # protein is present, curvature-sensing / insertion-depth data)
        self.collect_time_series_data()

        # Restore final-frame state for all the single-frame plots below
        self.u.trajectory[-1]
        self.extract_surface_data()
        self.create_surface_grid()
        self.calculate_curvature()
        self.calculate_lipid_deflection()
        self.calculate_curvature_frustration()
        self.analyze_pore_radius()
        self.analyze_water_permeation()
        self.calculate_arc_radius_of_curvature()
        
        print("\n" + "="*80)
        print("GENERATING TOP & SIDE VIEW ANALYSIS")
        print("="*80)
        self.plot_top_view()
        self.plot_side_view()
        self.plot_3d_interactive_topside()
        
        print("\n" + "="*80)
        print("GENERATING CROSS-SECTION PROFILES")
        print("="*80)
        self.plot_z_profile()
        self.plot_thickness_profile()
        self.plot_curvature_profile()
        self.plot_radius_of_curvature_profile()

        print("\n" + "="*80)
        print("GENERATING CURVATURE FRUSTRATION ANALYSIS")
        print("="*80)
        self.plot_curvature_frustration()
        self.plot_frustration_energy_timeseries()
        
        print("\n" + "="*80)
        print("GENERATING TIME-SERIES ANALYSIS")
        print("="*80)
        self.plot_radius_of_curvature_over_time_smoothed()
        self.plot_radius_of_curvature_upper_lower_over_time()
        self.plot_thickness_over_time()
        self.plot_deflection_over_time()
        self.plot_timeseries_stats_summary()
        self.plot_curvature_over_time_standalone()
        self.plot_radius_of_curvature_over_time_standalone()

        print("\n" + "="*80)
        print("GENERATING PACSIN1/F-BAR-STYLE CURVATURE INDUCTION & SENSING ANALYSIS")
        print("="*80)
        self.plot_normal_vector_curvature_schematic()
        self.plot_bead_scatter_fit()
        self.plot_radius_of_curvature_raw_timeseries()
        self.plot_protein_local_vs_bulk_curvature()
        self.plot_protein_insertion_depth_over_time()
        self.plot_protein_footprint_map()
        
        print("\n" + "="*80)
        print("GENERATING STANDARD 2D ANALYSIS PLOTS")
        print("="*80)
        self.plot_2d_standard_all()
        self.plot_radius_of_curvature_2d()
        
        print("\n" + "="*80)
        print("GENERATING STANDARD 3D ANALYSIS PLOTS")
        print("="*80)
        self.plot_3d_standard_all()
        
        print("\n" + "="*80)
        print("✓ COMPLETE ANALYSIS FINISHED!")
        print("="*80)
        print(f"\n📁 ALL FILES SAVED UNDER: {os.path.abspath(self.output_dir)}\n")
        print("📊 GENERATED FILES SUMMARY (every plot saved individually -- no combined panels):")
        print(f"\n01_2D_analysis/  ({self.subdirs['2d']})")
        print("  ✓ top_view.png, side_view.png")
        print("  ✓ z_profile.png, thickness_profile.png, curvature_profile.png, radius_of_curvature_profile.png")
        print("  ✓ thickness_heatmap.png, thickness_histogram.png, thickness_stats.png")
        print("  ✓ curvature_heatmap.png, curvature_histogram.png, curvature_stats.png")
        print("  ✓ water_density_heatmap.png, water_density_histogram.png, water_density_stats.png")
        print("  ✓ radius_of_curvature_heatmap.png, radius_of_curvature_histogram.png, radius_of_curvature_stats.png  <- ACTUAL R map")
        print(f"\n02_3D_analysis/  ({self.subdirs['3d']})")
        print("  ✓ interactive_top_side_view.html")
        print("  ✓ thickness_3d_upper.html, thickness_3d_lower.html")
        print("  ✓ curvature_3d_upper.html, curvature_3d_lower.html")
        print("  ✓ deflection_3d_upper.html, deflection_3d_lower.html")
        print("  ✓ radius_of_curvature_3d_midplane.html, radius_of_curvature_3d_upper.html, radius_of_curvature_3d_lower.html  <- ACTUAL R, 3D")
        print("  ✓ pore_3d_analysis.html")
        print(f"\n03_time_series/  ({self.subdirs['timeseries']})")
        print("  ✓ radius_of_curvature_over_time_smoothed.png  <- midplane R(t), diff-geom")
        print("  ✓ radius_of_curvature_upper_lower_over_time.png  <- per-leaflet R(t)")
        print("  ✓ radius_of_curvature_over_time.png  <- diff-geom R(t) vs circle-fit R(t)")
        print("  ✓ curvature_over_time.png  <- curvature induction over time")
        print("  ✓ thickness_over_time.png, deflection_over_time.png, timeseries_stats_summary.png")
        print(f"\n04_curvature_frustration/  ({self.subdirs['frustration']})")
        print("  ✓ c1_map_upper.png, c1_map_lower.png, frustration_summary.png")
        print("  ✓ gc_map_upper.png, gc_map_lower.png, c1_vs_c0_profile.png")
        print("  ✓ frustration_energy_timeseries.png")
        print(f"\n05_pacsin1_curvature_sensing/  ({self.subdirs['pacsin1']})")
        print("  ✓ normal_vector_curvature_schematic.png  <- S(x)/Δφ reference-style figure")
        print("  ✓ bead_scatter_fit.png, radius_of_curvature_raw_timeseries.png  <- raw beads+fit & R(t), Å/ns")
        print("  ✓ protein_local_vs_bulk_curvature.png, protein_insertion_depth_over_time.png")
        print("  ✓ protein_footprint_curvature_map.png")
        print(f"\n06_data_exports/  ({self.subdirs['data']})")
        print("  ✓ timeseries_data.csv  (includes midplane/upper/lower R(t) columns)")
        print("  ✓ analysis_summary.json")
        print("\n" + "="*80)
        n_files = 17 + 11 + 7 + 6 + 5 + 2
        print(f"TOTAL: {n_files} Individually-Saved, Publication-Quality Visualizations / Data Files Generated!")
        print("="*80 + "\n")

# Usage
if __name__ == "__main__":
    TPR = "md_0_300.tpr"
    XTC = "cen20.xtc"
    LIPIDS = ["PMPG", "PMPE", "POPE", "OYPE", "PVCL", "PYPG", "QMPE"]

    # Spontaneous curvature c0 (Å^-1) overrides for the Marsh (2007) model.
    # Unlisted resnames fall back to a headgroup-substring classification
    # (see DEFAULT_C0_BY_HEADGROUP at the top of this file). Replace these
    # with experimental/literature values for your specific lipids where
    # available (e.g., from Marsh, Chem Phys Lipids 2006;144:146-159, or
    # Kollmitzer et al., Soft Matter 2013;9:10877 -- those sources typically
    # quote values in nm^-1; divide by 10 to convert to Å^-1 as done below).
    C0_OVERRIDES = {
        "PMPG": -0.005,   # PG headgroup: near-lamellar (-0.05 nm^-1)
        "PMPE": -0.035,   # PE headgroup: non-lamellar-forming (-0.35 nm^-1)
        "POPE": -0.035,
        "OYPE": -0.035,
        "PVCL": -0.040,   # cardiolipin variant: strongly non-lamellar (-0.40 nm^-1)
        "PYPG": -0.005,
        "QMPE": -0.035,
    }

    # Leave as None to auto-create a fresh timestamped folder each run
    # (e.g. "membrane_analysis_output_20260727_153000"), or set an explicit
    # path/name here to control where everything gets saved.
    OUTPUT_DIR = None

    analyzer = ProfessionalMembraneSurfaceAnalyzer(
        TPR, XTC, LIPIDS, grid_points=250, smoothing_sigma=2,
        kc_kbt=9.9,                 # monolayer bending modulus (kBT), Marsh 2007
        c0_overrides=C0_OVERRIDES,  # per-lipid spontaneous curvature (Å^-1)
        temperature=310.0,          # K, for kBT <-> kJ/mol conversion
        output_dir=OUTPUT_DIR,      # all output saved here, in organized subfolders
        n_timeseries_frames=20,     # number of frames sampled for every time-series plot
        protein_probe_radius=15.0,  # Å, protein "footprint" radius for curvature sensing
        arc_axis='x',               # cross-section direction for the S(x)/Δφ schematic
        n_normal_vectors=7,         # number of S(x) normal vectors drawn (as in ref. figure)
    )
    analyzer.run_analysis()
