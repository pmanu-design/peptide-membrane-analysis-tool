# ============================================================================
# CHANGELOG (this revision)
# ----------------------------------------------------------------------------
# Accuracy fixes:
#   - PBC unwrap transformation added so molecules split across a periodic
#     boundary get correct centers of mass (affects every downstream metric).
#   - Leaflet assignment (upper/lower) is now made ONCE per lipid residue at
#     a reference frame and kept fixed, instead of re-splitting by a global
#     z-median every frame (which caused leaflet "flicker" for lipids near
#     the midplane and corrupted leaflet-resolved composition/curvature).
#   - Pore-radius distance transform now uses the true anisotropic grid
#     spacing (dx, dy) instead of averaging them into one isotropic value.
#   - curvature_norm now guards against divide-by-zero on a perfectly flat frame.
#
# New analyses:
#   - Area per lipid (APL) per leaflet, single-frame and over time.
#   - Bending modulus kc estimated directly from the trajectory via the
#     Helfrich undulation spectrum <|h(q)|^2>/A = kBT/(kc*q^4), an
#     independent cross-check on the literature kc_kbt fed into the Marsh
#     (2007) curvature-frustration model elsewhere in this script.
#   - Pore radius and water-permeation time series (previously single-frame only).
#   - Water-density gridding vectorized (np.histogram2d) -- was an
#     O(grid_points^2 * n_water) nested loop; now O(n_water + grid_points^2).
# ============================================================================

import os
from datetime import datetime
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree, Voronoi
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

# Per-lipid acyl-chain atom names, in bond order from the glycerol/headgroup
# end down to the terminal methyl, one list per tail ("A"/"B" = sn-1/sn-2,
# or however your force field labels them). Used by calculate_order_parameters()
# to build the C(i-1)->C(i+1) vector at each interior carbon and estimate the
# S_CH-proxy order parameter. Leave a resname OUT of this dict to fall back to
# _auto_detect_tail_atoms(), which groups atom names by a trailing chain
# letter and sorts by the embedded chain-position number (works for common
# CHARMM-style "C22 C23 ... C216"/"C32...C316" and MARTINI-style
# "C1A C2A C3A C4A"/"C1B..." naming) -- but auto-detection is a heuristic,
# so for production figures verify (or just fill in) the exact atom names
# for your force field here, e.g.:
#   TAIL_CARBON_ATOMS_BY_RESNAME = {
#       'POPE': {'A': ['C22','C23','C24','C25','C26','C27','C28','C29',
#                       'C210','C211','C212','C213','C214','C215','C216'],
#                'B': ['C32','C33','C34','C35','C36','C37','C38','C39',
#                       'C310','C311','C312','C313','C314','C315','C316']},
#   }
TAIL_CARBON_ATOMS_BY_RESNAME = {}

# Full-bilayer bending rigidity used in the Pacsin1 tube-formation elastic
# cost estimate (Mahmood, Noguchi & Okazaki, Sci Rep 2019, Eq. 5), which is
# an independent quantity from the monolayer kc used in the Marsh model
# above; the paper cites ~20 kBT (Zimmerberg & McLaughlin, Curr Biol 2004).
DEFAULT_BILAYER_KC_KBT = 20.0


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
                 output_dir=None, tail_atom_names=None):
        """Initialize professional membrane analyzer

        Additional parameters (Marsh 2007 curvature-frustration model):
            kc_kbt      : monolayer bending modulus, kc, in units of kBT (default 9.9)
            c0_overrides: dict mapping resname -> spontaneous curvature c0 (Å^-1);
                          entries not given fall back to headgroup-class defaults
                          (see DEFAULT_C0_BY_HEADGROUP)
            temperature : temperature in K, used only for reporting kc/energies
                          in absolute units (kJ/mol) alongside kBT units
            output_dir  : folder where ALL generated figures/HTML files are
                          saved. If None, an auto-timestamped folder
                          ("membrane_analysis_output_<timestamp>") is created
                          in the current working directory.
            tail_atom_names : optional dict overriding/extending
                          TAIL_CARBON_ATOMS_BY_RESNAME (see top of file) for
                          the acyl-tail order-parameter calculation, e.g.
                          {'POPE': {'A': [...], 'B': [...]}}. Resnames not
                          listed fall back to auto-detection.
        """
        print("=" * 80)
        print("PROFESSIONAL MEMBRANE SURFACE ANALYSIS WITH DYNAMICS")
        print("=" * 80)

        # ---- Output folder: every figure this analyzer produces is saved here ----
        if output_dir is None:
            output_dir = f"membrane_analysis_output_{datetime.now():%Y%m%d_%H%M%S}"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✓ Output folder created: {os.path.abspath(self.output_dir)}")

        print("Loading system...")
        self.u = mda.Universe(tpr, xtc)

        # ---- PBC correctness: make molecules whole across periodic images ----
        # Without this, a lipid/protein straddling a box edge gets its
        # center-of-mass computed from a mix of wrapped and unwrapped atom
        # coordinates, which silently corrupts COM positions (and therefore
        # every downstream surface, curvature, and leaflet calculation) for
        # any residue that happens to sit on a boundary in a given frame.
        # MDAnalysis' `unwrap` transformation uses bond connectivity (present
        # in a .tpr) to make every molecule whole, on the fly, every frame.
        try:
            from MDAnalysis import transformations
            self.u.trajectory.add_transformations(transformations.unwrap(self.u.atoms))
            print("✓ PBC unwrap transformation active (molecules made whole every frame)")
        except Exception as exc:
            print(f"⚠ Could not apply PBC unwrap transformation ({exc}). Proceeding without "
                  "it -- center-of-mass positions for molecules split across a periodic "
                  "boundary may be inaccurate. Ensure the topology (.tpr) contains bonds.")

        self.lipids = " ".join(lipids)
        self.lipid_atoms = self.u.select_atoms(f"resname {self.lipids}")
        self.water = self.u.select_atoms(f"resname {water_mol}")
        self.protein = self.u.select_atoms("protein")
        if len(self.protein) == 0:
            print("⚠ No atoms matched selection 'protein' -- protein-specific "
                  "curvature-sensing analyses (intrinsic curvature, s_norm/θ "
                  "sensing, tube areal density) will be skipped.")
        else:
            print(f"✓ Protein atoms found: {len(self.protein)} "
                  f"({len(self.protein.residues)} residues)")

        self.grid_points = grid_points
        self.smooth_sigma = smoothing_sigma
        self.ts = self.u.trajectory[-1]

        # resname -> {'A': [ordered tail-1 atom names], 'B': [...]} override
        # for calculate_order_parameters(); merged over TAIL_CARBON_ATOMS_BY_RESNAME
        # so a caller only needs to pass the resnames that need overriding.
        self.tail_atom_overrides = {**TAIL_CARBON_ATOMS_BY_RESNAME, **(tail_atom_names or {})}

        # ---- Persistent leaflet assignment ----
        # The original approach re-split lipids into "upper"/"lower" by a
        # global z-median on EVERY frame independently. For lipids whose
        # headgroups sit near the bilayer midplane (thermal fluctuation,
        # local bending, or a lipid near a pore/protein), this causes them
        # to flicker between leaflets from frame to frame, which corrupts
        # both the fitted leaflet surfaces (a "lower" lipid briefly
        # contributing to the "upper" grid) and the composition-weighted
        # c0 used in the Marsh (2007) frustration model (leaflet lipid
        # counts should not change over a trajectory with no flip-flop).
        # Instead, assign each lipid residue to a leaflet ONCE (last frame)
        # and keep that assignment fixed for the rest of the analysis --
        # standard practice (cf. MDAnalysis LeafletFinder, LiPyphilic).
        self._assign_leaflets_once()

        # Marsh (2007) curvature-frustration model parameters
        self.kc_kbt = kc_kbt                    # monolayer bending modulus (kBT)
        self.c0_overrides = c0_overrides or {}   # resname -> c0 (Å^-1) overrides
        self.temperature = temperature           # K
        R_GAS = 8.314462618e-3                   # kJ mol^-1 K^-1
        self.kbt_kjmol = R_GAS * self.temperature  # kBT in kJ/mol at this T
        self.kc_kjmol = self.kc_kbt * self.kbt_kjmol

        # For time-series analysis
        self.curvature_over_time = []
        self.thickness_over_time = []
        self.deflection_over_time = []
        self.time_points = []
        self.frustration_energy_over_time = []

        # New: midplane height fields (needed for the undulation-spectrum /
        # bending-modulus estimate), pore radius, water permeation, and
        # area-per-lipid, all sampled at the same frames as the rest of the
        # time series so everything lines up on the same time axis.
        self.height_fields_over_time = []
        self.protein_pore_radius_over_time = []
        self.pore_radius_over_time = []
        self.water_influx_over_time = []
        self.water_core_count_over_time = []
        self.apl_upper_over_time = []
        self.apl_lower_over_time = []

        # Pacsin1-paper-style time series (Mahmood, Noguchi & Okazaki, Sci Rep 2019)
        self.circlefit_radius_over_time = []       # Fig. 5b-style membrane circle-fit R(t)
        self.protein_radius_over_time = []         # Figs. 2b/3b/S1-style protein banana R(t)
        self.protein_snorm_over_time = []           # Fig. 6a/b-style protein arc position
        self.protein_theta_over_time = []            # Fig. 6a/b-style protein orientation
        self.protein_local_curvature_over_time = []  # C(s) at the protein's position

    # ==================== OUTPUT HELPERS ====================

    def _save_fig(self, filename):
        """Save the current matplotlib figure into self.output_dir and close it."""
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {path}")
        plt.close()

    def _save_html(self, fig, filename):
        """Save a plotly figure into self.output_dir."""
        path = os.path.join(self.output_dir, filename)
        fig.write_html(path)
        print(f"✓ Saved: {path}")

    def _time_ns(self, time_points_ps=None):
        """Convert stored trajectory time (self.time_points, in ps, as
        reported by MDAnalysis ts.time) to nanoseconds. All time-series
        plots in this script use nanoseconds for consistency."""
        if time_points_ps is None:
            time_points_ps = self.time_points
        return np.asarray(time_points_ps) / 1000.0   # ps -> ns

    def _assign_leaflets_once(self):
        """Classify every lipid residue as upper/lower leaflet ONE time
        (using the reference frame self.ts, i.e. the last frame), and cache
        the result by residue index. See the note in __init__ for why a
        fixed, persistent assignment is used instead of a per-frame
        z-median split. If a genuine lipid flip-flop analysis is desired,
        that is a distinct, explicit calculation and is not what the
        leaflet-resolved surface/curvature/composition analyses below need.
        """
        coords = self.lipid_atoms.center_of_mass(compound="residues")
        z = coords[:, 2]
        mid = np.median(z)
        upper_mask = z > mid

        resindices = self.lipid_atoms.residues.resindices
        # Map resindex -> leaflet (True=upper) with a dict, rather than
        # assuming resindices come back in the same sorted order every call
        # (true in practice since the selection's topology is static, but a
        # dict lookup removes that assumption entirely).
        self._leaflet_upper_by_resindex = {int(ri): bool(u) for ri, u in zip(resindices, upper_mask)}

        n_upper, n_lower = int(upper_mask.sum()), int((~upper_mask).sum())
        print(f"✓ Leaflet assignment fixed at reference frame: "
              f"{n_upper} upper-leaflet / {n_lower} lower-leaflet lipids")

    def extract_surface_data(self):
        """Extract membrane surface coordinates for the CURRENT frame, using
        the persistent per-residue leaflet assignment computed once in
        _assign_leaflets_once() (called from __init__) rather than
        re-splitting by z-median on every frame."""
        coords = self.lipid_atoms.center_of_mass(compound="residues")
        resnames = self.lipid_atoms.residues.resnames
        resindices = self.lipid_atoms.residues.resindices

        self.x = coords[:, 0]
        self.y = coords[:, 1]
        self.z = coords[:, 2]

        # resindices are stable per-residue identifiers; look up each
        # residue's fixed leaflet assignment regardless of ordering.
        upper_mask = np.array([self._leaflet_upper_by_resindex[int(ri)] for ri in resindices],
                               dtype=bool)
        lower_mask = ~upper_mask

        self.upper = coords[upper_mask]
        self.lower = coords[lower_mask]
        self.upper_resnames = resnames[upper_mask]
        self.lower_resnames = resnames[lower_mask]

        print(f"✓ Upper leaflet points: {len(self.upper)}")
        print(f"✓ Lower leaflet points: {len(self.lower)}")

    def create_surface_grid(self):
        """Create interpolated surface grids, then RECENTER z so that the
        bilayer midplane sits at z = 0.

        self.z_offset is the average local midplane height, (zu+zl)/2,
        over the whole grid for THIS frame; it is subtracted from zu, zl,
        and zi so that z=0 consistently means "the bilayer center" for
        every downstream calculation and plot (thickness and curvature are
        unaffected, since both are invariant to a constant z-shift). Any
        raw MDAnalysis coordinate (water, protein) compared against these
        grids elsewhere must also be shifted by self.z_offset -- see
        analyze_water_permeation() and analyze_protein_induced_pore().

        Also builds RegularGridInterpolator objects for the LOCAL upper/
        lower leaflet height at any (x,y), used so that "is this water
        molecule inside the bilayer" is evaluated against the true local
        membrane surface rather than a single global min/max bounding box
        -- important wherever the membrane is curved, buckled, thinned by
        a pore, or locally deformed by a bound protein.
        """
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

        # ---- Recenter: bilayer midplane -> z = 0 ----
        self.z_offset = float(np.nanmean(0.5 * (self.zu + self.zl)))
        self.zu -= self.z_offset
        self.zl -= self.z_offset
        self.zi -= self.z_offset

        self.thickness = self.zu - self.zl

        # ---- Local leaflet-height interpolators (for water/protein queries) ----
        x_centers, y_centers = self.xi[0, :], self.yi[:, 0]
        self._zu_interp = RegularGridInterpolator((y_centers, x_centers), self.zu,
                                                    bounds_error=False, fill_value=None)
        self._zl_interp = RegularGridInterpolator((y_centers, x_centers), self.zl,
                                                    bounds_error=False, fill_value=None)

    def _local_leaflet_z(self, xy):
        """Interpolate the local upper/lower leaflet height (already
        midplane-centered, z=0 = bilayer center) at arbitrary lateral
        (x, y) points, e.g. water or protein atom positions. xy: (N,2)
        array of (x,y). Returns (zu_local, zl_local), each shape (N,)."""
        xy = np.atleast_2d(xy)
        pts = np.column_stack([xy[:, 1], xy[:, 0]])   # interpolator axes are (y, x)
        return self._zu_interp(pts), self._zl_interp(pts)

    @staticmethod
    def _mean_and_gaussian_curvature(z, dy, dx):
        """Differential-geometry mean (H) and Gaussian (K) curvature of a
        Monge-patch surface z(x,y), using the exact (not small-slope) formulas:

            H = [(1+zx^2)*zyy - 2*zx*zy*zxy + (1+zy^2)*zxx] / [2*(1+zx^2+zy^2)^1.5]
            K = [zxx*zyy - zxy^2] / (1+zx^2+zy^2)^2

        Returns (H, K, c1) where c1 = 2H is the "total curvature"
        (c1 = c1_x + c1_y, sum of principal curvatures) in the same
        convention as the spontaneous curvature c0 used by Marsh (2007)
        and standard monolayer elasticity literature (c0 = 2*H0).
        Units follow the input grid spacing (Angstrom in this script).
        """
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

    def calculate_curvature(self):
        """Calculate membrane curvature using exact differential geometry
        (mean curvature H, Gaussian curvature K) at the neutral surface of
        each leaflet, rather than a small-slope Laplacian approximation.

        c1 = 2H is the total (sum of principal) curvature, in the same
        sign/scale convention as the lipid spontaneous curvature c0 used
        in the Marsh (2007) elastic frustration model.
        """
        dy, dx = self.dy, self.dx

        # Bilayer midplane (kept for backward-compatible plots)
        H_mid, K_mid, c1_mid = self._mean_and_gaussian_curvature(self.zi, dy, dx)
        self.curvature = c1_mid                    # total curvature, Angstrom^-1
        self.gaussian_curvature = K_mid
        curvature_absmax = np.nanmax(np.abs(self.curvature))
        self.curvature_norm = self.curvature / curvature_absmax if curvature_absmax > 1e-12 \
            else np.zeros_like(self.curvature)

        # Upper leaflet (neutral surface approximated by fitted headgroup surface)
        H_u, K_u, c1_u = self._mean_and_gaussian_curvature(self.zu, dy, dx)
        self.mean_curvature_u = H_u
        self.gaussian_curvature_u = K_u
        self.curvature_u = c1_u                     # total curvature c1, Angstrom^-1

        # Lower leaflet
        H_l, K_l, c1_l = self._mean_and_gaussian_curvature(self.zl, dy, dx)
        self.mean_curvature_l = H_l
        self.gaussian_curvature_l = K_l
        self.curvature_l = c1_l                      # Angstrom^-1

        # Calculate radius of curvature
        self.calc_radius_of_curvature()

    def calc_radius_of_curvature(self):
        """Calculate radius of curvature from the total curvature c1 = 2H.
        For a sphere, c1 = 2/R, so R = 2/|c1|; this reduces to the familiar
        R = 1/|c| only in the small-curvature limit used previously."""
        with np.errstate(divide='ignore', invalid='ignore'):
            self.radius_of_curvature = np.where(
                np.abs(self.curvature) > 1e-6,
                2.0 / np.abs(self.curvature),
                np.nan
            )
        self.radius_of_curvature = np.nan_to_num(self.radius_of_curvature, nan=1000.0)

    def calculate_lipid_deflection(self):
        """Calculate phospholipid deflection"""
        z_avg = (self.zu + self.zl) / 2

        self.deflection_u = self.zu - z_avg
        self.deflection_l = self.zl - z_avg
        self.deflection_total = np.abs(self.deflection_u) + np.abs(self.deflection_l)

        print(f"✓ Upper leaflet max deflection: {np.max(np.abs(self.deflection_u)):.3f} Å")
        print(f"✓ Lower leaflet max deflection: {np.max(np.abs(self.deflection_l)):.3f} Å")

    # ==================== AREA PER LIPID ====================

    def calculate_area_per_lipid(self):
        """Area per lipid (APL) for each leaflet: the projected (x,y) box
        area divided by the number of lipids assigned to that leaflet --
        the standard estimator for a laterally homogeneous bilayer
        (Kucerka, Nieh & Katsaras, BBA 2011;1808:2761). Uses the CURRENT
        simulation box (self.u.dimensions), so this reflects the true
        instantaneous lateral area rather than assuming a fixed box.

        Caveat: for a strongly curved or tubulated membrane this projected-
        area estimate systematically UNDERESTIMATES the true per-lipid area
        (the flat x,y projection is smaller than the actual curved surface
        area), so treat it as an approximate global average, not a locally
        curvature-corrected value.
        """
        box = self.u.dimensions[:2]  # Lx, Ly (Å); dimensions = [Lx,Ly,Lz,alpha,beta,gamma]
        projected_area = float(box[0] * box[1])

        n_upper = len(self.upper)
        n_lower = len(self.lower)
        self.apl_upper = projected_area / n_upper if n_upper else np.nan
        self.apl_lower = projected_area / n_lower if n_lower else np.nan
        self.apl_mean = np.nanmean([self.apl_upper, self.apl_lower])

        print(f"✓ Area per lipid: upper = {self.apl_upper:.2f} Å² "
              f"({self.apl_upper/100:.3f} nm², n={n_upper}), "
              f"lower = {self.apl_lower:.2f} Å² "
              f"({self.apl_lower/100:.3f} nm², n={n_lower})")
        return self.apl_upper, self.apl_lower

    # ==================== SPATIAL AREA-PER-LIPID MAP (Voronoi) ====================

    def calculate_area_per_lipid_map(self, leaflet='upper'):
        """Spatially-resolved area per lipid via 2D Voronoi tessellation of
        lipid center-of-mass (x,y) positions in the chosen leaflet, at the
        CURRENT frame -- the standard local-APL estimator (cf. Lukat, Krüger
        & Sommer, J Chem Inf Model 2013, "VMD Voro"; Gapsys et al. 2013).

        Each lipid's local area is the area of its Voronoi cell; a lipid
        squeezed by a nearby inserted peptide/protein gets a smaller cell
        (tighter packing) while lipids far from any perturbation relax
        toward the bulk APL from calculate_area_per_lipid(). The per-lipid
        values are then interpolated onto the same (self.xi, self.yi) grid
        used everywhere else in this script, so the map lines up with the
        thickness/curvature/water grids for overlay plots.

        The tessellation is built with the 8 periodic images of the leaflet
        added around the real points (minimum-image padding) so that cells
        near the box edge are correctly bounded by their true periodic
        neighbors instead of running off to infinity.
        """
        coords = self.upper[:, :2] if leaflet == 'upper' else self.lower[:, :2]
        n_real = len(coords)
        if n_real < 4:
            print(f"⚠ Too few lipids in {leaflet} leaflet for a Voronoi tessellation "
                  "-- skipping area-per-lipid map.")
            self.apl_map = None
            return None

        box = self.u.dimensions[:2]  # Lx, Ly (Å)
        offsets = [(dx, dy) for dx in (-box[0], 0, box[0]) for dy in (-box[1], 0, box[1])]
        pts = np.vstack([coords + np.array(off) for off in offsets])  # real points = pts[:n_real]

        vor = Voronoi(pts)
        areas = np.full(n_real, np.nan)
        for i in range(n_real):
            region = vor.regions[vor.point_region[i]]
            if not region or -1 in region:
                continue  # unbounded cell (shouldn't happen with periodic padding, but guard anyway)
            poly = vor.vertices[region]
            # Shoelace formula
            x, y = poly[:, 0], poly[:, 1]
            areas[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        # Drop any lipid whose cell wasn't resolved before interpolating
        valid = np.isfinite(areas)
        apl_map = griddata((coords[valid, 0], coords[valid, 1]), areas[valid],
                            (self.xi, self.yi), method='linear')
        nan_mask = np.isnan(apl_map)
        if nan_mask.any():
            apl_map_nearest = griddata((coords[valid, 0], coords[valid, 1]), areas[valid],
                                        (self.xi, self.yi), method='nearest')
            apl_map[nan_mask] = apl_map_nearest[nan_mask]
        apl_map = gaussian_filter(apl_map, self.smooth_sigma)

        self.apl_map = apl_map            # Å², shape (grid_points, grid_points)
        self.apl_map_leaflet = leaflet
        self.apl_map_per_lipid = areas    # Å², one value per lipid in this leaflet (NaN where unresolved)

        print(f"✓ Area-per-lipid map ({leaflet} leaflet, n={n_real} lipids): "
              f"{np.nanmean(areas):.2f} ± {np.nanstd(areas):.2f} Å² "
              f"({np.nanmean(areas)/100:.3f} nm²)")
        return self.apl_map

    def plot_area_per_lipid_map(self, leaflet='upper', vmin=None, vmax=None):
        """Filled-contour area-per-lipid map (nm²) over the box footprint,
        matching the standard "APL heatmap with the peptide overlaid on
        top" figure style: a jet colormap, axes in nm, and the peptide/
        protein backbone traced over the region it occupies so the local
        APL depression it causes is visible directly under it.
        """
        if not hasattr(self, 'apl_map') or self.apl_map is None or \
                getattr(self, 'apl_map_leaflet', None) != leaflet:
            self.calculate_area_per_lipid_map(leaflet=leaflet)
        if self.apl_map is None:
            return

        apl_nm2 = self.apl_map / 100.0          # Å² -> nm²
        x_nm, y_nm = self.xi / 10.0, self.yi / 10.0

        fig, ax = plt.subplots(figsize=(8, 7))
        levels = np.linspace(np.nanmin(apl_nm2) if vmin is None else vmin,
                              np.nanmax(apl_nm2) if vmax is None else vmax, 13)
        cf = ax.contourf(x_nm, y_nm, apl_nm2, levels=levels, cmap='jet', extend='both')
        cbar = plt.colorbar(cf, ax=ax, pad=0.02)
        cbar.set_label('Area per lipid (nm²)', fontweight='bold')

        # Overlay the peptide/protein backbone (CA trace) over the map so the
        # local packing perturbation it causes is visible in context.
        if len(self.protein) > 0:
            ca = self.protein.select_atoms("name CA")
            trace_atoms = ca if len(ca) > 0 else self.protein
            for seg in trace_atoms.segments:
                seg_pos = seg.atoms.positions
                if len(seg_pos) > 1:
                    ax.plot(seg_pos[:, 0] / 10.0, seg_pos[:, 1] / 10.0,
                            '-', color='indigo', linewidth=1.5, alpha=0.85, zorder=5)
            ax.scatter(trace_atoms.positions[:, 0] / 10.0, trace_atoms.positions[:, 1] / 10.0,
                       s=18, color='darkviolet', alpha=0.7, zorder=6, label='Peptide/protein')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.85)

        ax.set_xlabel('Box X (nm)', fontweight='bold')
        ax.set_ylabel('Box Y (nm)', fontweight='bold')
        ax.set_title('Area per lipid', fontweight='bold', fontsize=13)
        ax.set_xlim(x_nm.min(), x_nm.max())
        ax.set_ylim(y_nm.min(), y_nm.max())
        self._save_fig(f'area_per_lipid_map_{leaflet}.png')

    # ==================== ACYL-CHAIN ORDER PARAMETERS ====================

    @staticmethod
    def _auto_detect_tail_atoms(residue):
        """Heuristic fallback when a resname isn't in TAIL_CARBON_ATOMS_BY_RESNAME:
        group this residue's carbon atom names by a trailing chain-letter
        suffix (A/B, or the last non-digit character) and sort each group by
        the embedded chain-position integer. Returns {chain_id: [ordered
        atom names]} for chains with >= 4 atoms (need >=3 for an interior
        C(i-1)->C(i+1) vector); returns {} if nothing matches.
        """
        import re
        pattern = re.compile(r'^C(\d+)([A-Za-z]?)$')
        groups = {}
        for name in residue.atoms.names:
            m = pattern.match(name)
            if not m:
                continue
            idx, chain = int(m.group(1)), (m.group(2) or '0')
            groups.setdefault(chain, []).append((idx, name))
        return {chain: [n for _, n in sorted(pairs)]
                for chain, pairs in groups.items() if len(pairs) >= 4}

    def _tail_atoms_for_resname(self, resname, sample_residue):
        if resname in self.tail_atom_overrides:
            return self.tail_atom_overrides[resname]
        return self._auto_detect_tail_atoms(sample_residue)

    def calculate_order_parameters(self, core_radius_angstrom=15.0, rim_radius_angstrom=30.0):
        """Acyl-chain order parameter S_CC, per carbon position, for the
        CURRENT frame, resolved into three regions relative to the
        peptide/protein's lateral (x,y) position:

          Core : lipid COM within core_radius_angstrom of the protein COM
                 (annular ring immediately lining/near the peptide)
          Rim  : lipid COM between core_radius_angstrom and rim_radius_angstrom
          Bilayer : all lipids (unrestricted bulk average)

        For each interior chain carbon i (excluding the two chain ends,
        which have no well-defined C(i-1)->C(i+1) vector), this uses the
        common C-H-free proxy for the deuterium order parameter (Vermeer et
        al., Eur Biophys J 2007;36:919; Piggot et al., J Chem Theory Comput
        2012;8:4593):

            S_i = 0.5 * (3<cos^2(theta_i)> - 1)

        where theta_i is the angle between the C(i-1)->C(i+1) vector and the
        bilayer normal (z-axis; surfaces are already midplane-recentered so
        this is just the lab-frame z-axis). <...> averages over all lipids
        of that region present at this frame, and |S_i| is what's reported/
        plotted (sign is a tail-orientation convention, magnitude is what's
        physically meaningful and comparable to experiment).

        Requires no protein/peptide present -> Core/Rim are skipped and only
        the Bilayer curve is computed. Requires per-resname tail atom names
        (TAIL_CARBON_ATOMS_BY_RESNAME / tail_atom_names, or auto-detection --
        see top of file); resnames that resolve to no tail atoms are skipped
        with a warning.
        """
        if len(self.protein) > 0:
            protein_com_xy = self.protein.center_of_mass()[:2]
        else:
            protein_com_xy = None

        # region -> {carbon_index: [S values across lipids/chains this frame]}
        region_carbon_values = {'core': {}, 'rim': {}, 'bilayer': {}}
        skipped_resnames = set()

        for residue in self.lipid_atoms.residues:
            resname = residue.resname
            tails = self._tail_atoms_for_resname(resname, residue)
            if not tails:
                skipped_resnames.add(resname)
                continue

            com_xy = residue.atoms.center_of_mass()[:2]
            if protein_com_xy is not None:
                dist = float(np.linalg.norm(com_xy - protein_com_xy))
                region = 'core' if dist <= core_radius_angstrom else \
                         ('rim' if dist <= rim_radius_angstrom else None)
            else:
                region = None

            for chain_atoms in tails.values():
                present = [a for a in chain_atoms if a in residue.atoms.names]
                if len(present) < 3:
                    continue
                pos = np.array([residue.atoms.select_atoms(f"name {a}").positions[0]
                                 for a in present])
                for i in range(1, len(pos) - 1):
                    vec = pos[i + 1] - pos[i - 1]
                    norm = np.linalg.norm(vec)
                    if norm == 0:
                        continue
                    cos_theta = vec[2] / norm  # z-axis = bilayer normal
                    s = 0.5 * (3 * cos_theta ** 2 - 1)
                    region_carbon_values['bilayer'].setdefault(i, []).append(s)
                    if region is not None:
                        region_carbon_values[region].setdefault(i, []).append(s)

        if skipped_resnames:
            print(f"⚠ No tail-atom mapping resolved for resname(s) {sorted(skipped_resnames)} "
                  "-- skipped in order-parameter calculation (add to "
                  "TAIL_CARBON_ATOMS_BY_RESNAME or tail_atom_names).")

        def _finalize(region_dict):
            if not region_dict:
                return np.array([]), np.array([])
            carbons = np.array(sorted(region_dict.keys()))
            s_vals = np.array([np.abs(np.mean(region_dict[c])) for c in carbons])
            return carbons, s_vals

        self.op_carbons_core, self.op_values_core = _finalize(region_carbon_values['core'])
        self.op_carbons_rim, self.op_values_rim = _finalize(region_carbon_values['rim'])
        self.op_carbons_bilayer, self.op_values_bilayer = _finalize(region_carbon_values['bilayer'])
        self.op_core_radius = core_radius_angstrom
        self.op_rim_radius = rim_radius_angstrom
        self.op_protein_com_xy = protein_com_xy

        if len(self.op_carbons_bilayer):
            print(f"✓ Order parameters (bilayer avg over #carbon "
                  f"{int(self.op_carbons_bilayer.min())}-{int(self.op_carbons_bilayer.max())}): "
                  f"peak |S| = {self.op_values_bilayer.max():.3f}")
        else:
            print("⚠ No order parameters computed -- no tail atoms resolved for any lipid.")

        # ---- Spatially-gridded per-lipid mean |S| map (for the map panel) ----
        xy, s_mean = [], []
        for residue in self.lipid_atoms.residues:
            tails = self._tail_atoms_for_resname(residue.resname, residue)
            if not tails:
                continue
            per_lipid_s = []
            for chain_atoms in tails.values():
                present = [a for a in chain_atoms if a in residue.atoms.names]
                if len(present) < 3:
                    continue
                pos = np.array([residue.atoms.select_atoms(f"name {a}").positions[0]
                                 for a in present])
                for i in range(1, len(pos) - 1):
                    vec = pos[i + 1] - pos[i - 1]
                    norm = np.linalg.norm(vec)
                    if norm == 0:
                        continue
                    cos_theta = vec[2] / norm
                    per_lipid_s.append(abs(0.5 * (3 * cos_theta ** 2 - 1)))
            if per_lipid_s:
                xy.append(residue.atoms.center_of_mass()[:2])
                s_mean.append(np.mean(per_lipid_s))

        if len(xy) >= 4:
            xy = np.array(xy)
            self.order_param_map = griddata(xy, np.array(s_mean), (self.xi, self.yi), method='linear')
            nan_mask = np.isnan(self.order_param_map)
            if nan_mask.any():
                nearest = griddata(xy, np.array(s_mean), (self.xi, self.yi), method='nearest')
                self.order_param_map[nan_mask] = nearest[nan_mask]
            self.order_param_map = gaussian_filter(self.order_param_map, self.smooth_sigma)
        else:
            self.order_param_map = None

        return self.op_carbons_bilayer, self.op_values_bilayer

    def plot_thickness_order_parameter_analysis(self):
        """Three-panel figure: (A) thickness map, (B) acyl-chain order
        parameter map with the Core/Rim sampling regions marked, (C) order
        parameter vs. carbon position for Rim / Core / Bilayer -- same
        layout as the standard "thickness + order parameter" summary figure
        used for peptide/AMP-in-membrane papers.
        """
        if not hasattr(self, 'order_param_map'):
            self.calculate_order_parameters()
        if self.order_param_map is None:
            print("⚠ No order-parameter map available -- skipping "
                  "plot_thickness_order_parameter_analysis().")
            return

        extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]
        fig = plt.figure(figsize=(7, 13))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.35)

        # Panel A: thickness
        axA = fig.add_subplot(gs[0])
        imA = axA.contourf(self.xi, self.yi, self.thickness, levels=14, cmap='jet')
        plt.colorbar(imA, ax=axA, label='Thickness (Å)')
        axA.set_title('A   Thickness', loc='left', fontweight='bold', fontsize=13)
        axA.set_xlabel('x-axis (Å)', fontweight='bold')
        axA.set_ylabel('y-axis (Å)', fontweight='bold')

        # Panel B: order-parameter map with Core/Rim markers
        axB = fig.add_subplot(gs[1])
        imB = axB.contourf(self.xi, self.yi, self.order_param_map, levels=14, cmap='jet')
        plt.colorbar(imB, ax=axB, label=r'$S_{CH}$')
        axB.set_title('B   Order parameters', loc='left', fontweight='bold', fontsize=13)
        axB.set_xlabel('x-axis (Å)', fontweight='bold')
        axB.set_ylabel('y-axis (Å)', fontweight='bold')
        if getattr(self, 'op_protein_com_xy', None) is not None:
            cx, cy = self.op_protein_com_xy
            axB.plot(cx, cy, 'w+', markersize=10, markeredgewidth=2)
            axB.annotate('Core', (cx, cy), color='white', fontweight='bold', ha='center', va='center')
            rim_xy = (cx + self.op_rim_radius, cy)
            axB.plot([cx, rim_xy[0]], [cy, rim_xy[1]], 'w--', linewidth=1.2)
            axB.annotate('Rim', rim_xy, color='white', fontweight='bold', ha='left', va='center')

        # Panel C: order parameter vs carbon number, Rim/Core/Bilayer
        axC = fig.add_subplot(gs[2])
        if len(self.op_carbons_rim):
            axC.plot(self.op_carbons_rim, self.op_values_rim, color='tab:blue', linewidth=2, label='Rim')
        if len(self.op_carbons_core):
            axC.plot(self.op_carbons_core, self.op_values_core, color='tab:orange', linewidth=2, label='Core')
        if len(self.op_carbons_bilayer):
            axC.plot(self.op_carbons_bilayer, self.op_values_bilayer, color='tab:green', linewidth=2, label='Bilayer')
        axC.set_xlabel('#Carbon', fontweight='bold')
        axC.set_ylabel('Order parameter', fontweight='bold')
        axC.set_title('C', loc='left', fontweight='bold', fontsize=13)
        axC.grid(True, alpha=0.3)
        axC.legend(fontsize=10, framealpha=0.9)

        self._save_fig('thickness_order_parameter_analysis.png')

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

    # ==================== UNDULATION SPECTRUM -> SIMULATION-DERIVED kc ====================

    def compute_undulation_spectrum(self, height_fields=None, n_qbins=25):
        """Estimate the bilayer bending rigidity kc DIRECTLY from this
        trajectory's thermal height fluctuations, via the standard Helfrich
        undulation-spectrum method (Brannigan & Brown, J Chem Phys 2004;
        Marrink & Mark, Biophys J 2001; Levine et al., J Chem Theory
        Comput 2011): for the tensionless, small-fluctuation regime,

            <|h(q)|^2> / A  =  kBT / (kc * q^4)      (q != 0)

        where h(q) is the Fourier transform of the midplane height field
        and A is the projected membrane area. This is independent of, and
        a physically meaningful check on, the literature/default kc_kbt
        value supplied to __init__ and used throughout the Marsh (2007)
        frustration-energy calculation elsewhere in this script -- if the
        two disagree substantially, the assumed literature kc may not be
        appropriate for this particular lipid composition/force field.

        Requires several sampled frames (height_fields, defaulting to
        self.height_fields_over_time collected in collect_time_series_data)
        to average out single-frame noise in |h(q)|^2.

        Limitations: this simple form neglects lateral tension (sigma) and
        protrusion-mode contributions, both of which curve the spectrum
        away from pure q^-4 scaling at low and high q respectively; the fit
        below therefore restricts to an intermediate q-range where the
        pure-bending term dominates, which is the standard mitigation but
        not a complete substitute for a tension-inclusive fit.
        """
        if height_fields is None:
            height_fields = self.height_fields_over_time
        if not height_fields or len(height_fields) < 3:
            print("⚠ Fewer than 3 sampled frames available -- skipping undulation-spectrum "
                  "bending-modulus estimate (need several frames to average |h(q)|^2).")
            self.kc_undulation_kbt = np.nan
            self.kc_undulation_err_kbt = np.nan
            self.q_bins, self.Sq_bins = None, None
            return None

        ny, nx = height_fields[0].shape
        Lx, Ly = nx * self.dx, ny * self.dy
        A = Lx * Ly

        qx = 2 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        qy = 2 * np.pi * np.fft.fftfreq(ny, d=self.dy)
        QX, QY = np.meshgrid(qx, qy)
        Q = np.sqrt(QX ** 2 + QY ** 2)

        # <|h(q)|^2>/A, averaged over frames. h(q) approximated as a Riemann
        # sum of the continuum Fourier transform: h(q) = dx*dy * FFT(h_grid).
        S_frames = []
        for h in height_fields:
            h0 = np.nan_to_num(h - np.nanmean(h))
            hq = (self.dx * self.dy) * np.fft.fft2(h0)
            S_frames.append((np.abs(hq) ** 2) / A)
        S_avg = np.mean(S_frames, axis=0)

        # Azimuthal average into |q| bins
        q_flat, S_flat = Q.ravel(), S_avg.ravel()
        q_max = q_flat.max()
        edges = np.linspace(0, q_max, n_qbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        S_binned = np.full(n_qbins, np.nan)
        for k in range(n_qbins):
            sel = (q_flat >= edges[k]) & (q_flat < edges[k + 1]) & (q_flat > 1e-8)
            if np.any(sel):
                S_binned[k] = np.mean(S_flat[sel])

        self.q_bins, self.Sq_bins = centers, S_binned

        valid = np.isfinite(S_binned) & (S_binned > 0)
        q_valid, S_valid = centers[valid], S_binned[valid]
        if len(q_valid) < 6:
            print("⚠ Not enough valid q-bins to fit an undulation spectrum; skipping.")
            self.kc_undulation_kbt = np.nan
            self.kc_undulation_err_kbt = np.nan
            return None

        # Fit only the intermediate q-range: low-q is noisy/finite-size-
        # limited, high-q is contaminated by discretization & protrusions.
        lo = max(1, int(0.15 * len(q_valid)))
        hi = max(lo + 4, int(0.65 * len(q_valid)))
        hi = min(hi, len(q_valid))
        q_fit, S_fit = q_valid[lo:hi], S_valid[lo:hi]

        def helfrich_model(q, kc_kbt_fit):
            # In units where kBT = 1 (i.e. reporting kc directly in kBT):
            # S(q) = 1 / (kc_kbt * q^4)
            return 1.0 / (kc_kbt_fit * q ** 4)

        try:
            popt, pcov = curve_fit(helfrich_model, q_fit, S_fit,
                                    p0=[self.kc_kbt], maxfev=5000)
            self.kc_undulation_kbt = float(popt[0])
            self.kc_undulation_err_kbt = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else np.nan
        except Exception as exc:
            print(f"⚠ Undulation-spectrum fit failed ({exc}); skipping.")
            self.kc_undulation_kbt = np.nan
            self.kc_undulation_err_kbt = np.nan
            return None

        print(f"✓ Undulation-spectrum bending modulus: kc = {self.kc_undulation_kbt:.2f} "
              f"± {self.kc_undulation_err_kbt:.2f} kBT (from {len(height_fields)} frames, "
              f"fit over q ∈ [{q_fit[0]:.4f}, {q_fit[-1]:.4f}] Å⁻¹)")
        print(f"  Compare to the literature/default kc used in the Marsh (2007) model: "
              f"{self.kc_kbt:.2f} kBT")
        return self.kc_undulation_kbt, self.kc_undulation_err_kbt

    def analyze_pore_radius(self, threshold=0.5):
        """Analyze pore regions and radius"""
        thickness_norm = (self.thickness - self.thickness.mean()) / self.thickness.std()
        self.pore_mask = thickness_norm < -threshold
        self.pore_thickness = np.where(self.pore_mask, self.thickness, np.nan)

        if np.any(self.pore_mask):
            # `sampling=(dy, dx)` tells distance_transform_edt the true
            # physical spacing along each axis, instead of averaging dx/dy
            # into one isotropic spacing after the fact -- this matters
            # whenever the grid is non-square (grid_points fixed but the
            # box's X and Y extents differ, which is the common case).
            distance = distance_transform_edt(self.pore_mask, sampling=(self.dy, self.dx))
            self.max_pore_radius = np.max(distance)
            self.pore_radius_map = distance
        else:
            self.max_pore_radius = 0
            self.pore_radius_map = np.zeros_like(self.pore_mask, dtype=float)

        print(f"✓ Maximum pore radius: {self.max_pore_radius:.2f} Å")

    def _grid_histogram(self, xy_points):
        """2D histogram of lateral (x,y) points onto this frame's (xi,yi)
        grid cells, vectorized with np.histogram2d (bin edges built to
        exactly match each grid cell's footprint: center +/- half
        spacing). Returns an array shaped like self.zi. Shared helper used
        by both the water-density and core-water-density maps."""
        density = np.zeros_like(self.zi)
        if len(xy_points) == 0:
            return density
        x_centers, y_centers = self.xi[0, :], self.yi[:, 0]
        x_edges = np.concatenate([x_centers - self.dx / 2, [x_centers[-1] + self.dx / 2]])
        y_edges = np.concatenate([y_centers - self.dy / 2, [y_centers[-1] + self.dy / 2]])
        hist, _, _ = np.histogram2d(xy_points[:, 0], xy_points[:, 1], bins=[x_edges, y_edges])
        # meshgrid(xi, yi) default 'xy' indexing -> array[i,j] <-> (x_j, y_i);
        # histogram2d returns hist[a,b] <-> (x_bin_a, y_bin_b) -> transpose to match.
        return hist.T

    def analyze_water_permeation(self, core_fraction=0.5, halo_angstrom=5.0):
        """Water permeation / hydration analysis in a MIDPLANE-CENTERED
        frame (z = 0 at the bilayer center, set in create_surface_grid),
        using the LOCAL (per lateral position) leaflet surfaces rather
        than a single global z_min/z_max bounding box. This matters
        wherever the membrane is curved, buckled, locally thinned by a
        pore, or deformed by a bound protein: a global box either over-
        or under-counts water relative to the true local envelope at that
        (x, y).

        Reports two quantities:
          self.water_influx      : water within the local leaflet envelope
                                    plus a `halo_angstrom` margin outside it
                                    (broad "in/near the bilayer" count,
                                    analogous to the previous behaviour).
          self.water_core_count  : water within the innermost
                                    `core_fraction` of the LOCAL thickness,
                                    centered on the local midplane -- i.e.
                                    water that has actually penetrated the
                                    hydrophobic core near z=0, as opposed
                                    to sitting in the headgroup region.
                                    This is the physically relevant count
                                    for pore/permeation events.

        self.water_depths / self.water_core_depths store each counted
        water molecule's SIGNED depth relative to the LOCAL bilayer
        midplane (Å; 0 = exactly at the local midplane, + = upper-leaflet
        side, - = lower-leaflet side), for the depth-profile plot.
        """
        if len(self.water) == 0:
            print("⚠ No water molecules found")
            self.water_influx = 0
            self.water_core_count = 0
            self.water_density = np.zeros_like(self.zi)
            self.water_core_density = np.zeros_like(self.zi)
            self.water_depths = np.array([])
            self.water_core_depths = np.array([])
            return

        water_coords = self.water.positions.copy()
        water_coords[:, 2] -= self.z_offset   # into the midplane-centered (z=0) frame

        zu_local, zl_local = self._local_leaflet_z(water_coords[:, :2])
        local_mid = 0.5 * (zu_local + zl_local)
        local_thick = zu_local - zl_local
        z = water_coords[:, 2]
        depth = z - local_mid   # signed depth relative to the LOCAL midplane

        # Broad "in/near bilayer" region: local leaflet envelope + halo margin
        in_membrane = (z > zl_local - halo_angstrom) & (z < zu_local + halo_angstrom)
        self.water_influx = int(np.sum(in_membrane))
        self.water_depths = depth[in_membrane]
        print(f"✓ Water molecules in bilayer envelope (local leaflets ± {halo_angstrom:.0f} Å halo): "
              f"{self.water_influx}")

        # Narrow "midplane / hydrophobic core" region: innermost
        # core_fraction of the LOCAL thickness, centered on the local
        # midplane -- this is "water found in the bilayer midplane".
        core_halfwidth = core_fraction * local_thick / 2.0
        in_core = np.abs(depth) < core_halfwidth
        self.water_core_count = int(np.sum(in_core))
        self.water_core_depths = depth[in_core]
        print(f"✓ Water molecules in midplane/hydrophobic-core region "
              f"(innermost {core_fraction*100:.0f}% of local thickness, "
              f"|depth from local midplane| < {core_fraction*100:.0f}%·t/2): "
              f"{self.water_core_count}")

        self.water_density = gaussian_filter(
            self._grid_histogram(water_coords[in_membrane][:, :2]), self.smooth_sigma)
        self.water_core_density = gaussian_filter(
            self._grid_histogram(water_coords[in_core][:, :2]), self.smooth_sigma)

    # ==================== PROTEIN/PEPTIDE-INDUCED PORE ====================

    def analyze_protein_induced_pore(self, lateral_cutoff_angstrom=15.0, thickness_threshold=0.5):
        """Pore formation specifically ASSOCIATED WITH the bound protein/
        peptide, as distinct from generic thermal thinning fluctuations
        that analyze_pore_radius() detects anywhere in the membrane. Only
        runs if a protein/peptide selection is present -- there is nothing
        to attribute an "induced" pore to otherwise, so this is skipped
        entirely (not just zeroed out) when self.protein is empty.

        Method:
          1. Project every protein atom onto the lateral (x,y) grid and
             mark all grid cells within `lateral_cutoff_angstrom` of ANY
             protein atom as the protein's lateral "footprint" (nearest-
             neighbor query via a KD-tree).
          2. Within that footprint only, apply the same thickness z-score
             criterion used in analyze_pore_radius() to flag locally
             thinned/pore-like cells, then measure the resulting induced
             pore's area and maximum radius (anisotropic Euclidean
             distance transform).
          3. Report thinning (mean thickness inside vs. outside the
             footprint) and the ratio of midplane/hydrophobic-core water
             density inside vs. outside the footprint -- elevated core-
             water density colocalized with thinning under the protein is
             the classic hydration-defect / water-wire signature of a
             peptide-induced (e.g. toroidal) pore.
        """
        if len(self.protein) == 0:
            print("⚠ No protein/peptide selected -- skipping protein-induced pore analysis "
                  "(there is no protein to attribute an induced pore to).")
            self.protein_pore_radius = np.nan
            self.protein_pore_area = np.nan
            self.protein_pore_mask = None
            self.protein_footprint_mask = None
            self.thinning_fraction = np.nan
            self.water_core_density_ratio_protein = np.nan
            return None

        protein_xy = self.protein.positions[:, :2]
        grid_xy = np.column_stack([self.xi.ravel(), self.yi.ravel()])
        tree = cKDTree(protein_xy)
        dist_to_protein, _ = tree.query(grid_xy)
        near_protein = (dist_to_protein <= lateral_cutoff_angstrom).reshape(self.xi.shape)
        self.protein_footprint_mask = near_protein

        if not np.any(near_protein):
            print(f"⚠ No grid cells within {lateral_cutoff_angstrom:.0f} Å of the protein "
                  "(check units/selection); skipping protein-induced pore analysis.")
            self.protein_pore_radius = np.nan
            self.protein_pore_area = np.nan
            self.protein_pore_mask = None
            self.thinning_fraction = np.nan
            self.water_core_density_ratio_protein = np.nan
            return None

        thickness_std = self.thickness.std()
        thickness_norm = (self.thickness - self.thickness.mean()) / thickness_std if thickness_std > 0 \
            else np.zeros_like(self.thickness)
        protein_pore_mask = (thickness_norm < -thickness_threshold) & near_protein
        self.protein_pore_mask = protein_pore_mask

        if np.any(protein_pore_mask):
            distance = distance_transform_edt(protein_pore_mask, sampling=(self.dy, self.dx))
            self.protein_pore_radius = float(np.max(distance))
            self.protein_pore_area = float(np.sum(protein_pore_mask) * self.dx * self.dy)
        else:
            self.protein_pore_radius = 0.0
            self.protein_pore_area = 0.0

        thickness_in = float(np.mean(self.thickness[near_protein]))
        thickness_out = float(np.mean(self.thickness[~near_protein])) if np.any(~near_protein) else np.nan
        self.thickness_in_footprint = thickness_in
        self.thickness_outside_footprint = thickness_out
        self.thinning_fraction = (1.0 - thickness_in / thickness_out) if thickness_out else np.nan

        if hasattr(self, 'water_core_density') and self.water_core_density is not None:
            wd_in = float(np.mean(self.water_core_density[near_protein]))
            wd_out = float(np.mean(self.water_core_density[~near_protein])) if np.any(~near_protein) else np.nan
            self.water_core_density_ratio_protein = (wd_in / wd_out) if wd_out and wd_out > 0 else np.nan
        else:
            self.water_core_density_ratio_protein = np.nan

        print(f"✓ Protein lateral footprint: {int(np.sum(near_protein))} grid cells within "
              f"{lateral_cutoff_angstrom:.0f} Å of the protein")
        print(f"✓ Protein-induced pore radius: {self.protein_pore_radius:.2f} Å "
              f"(area {self.protein_pore_area:.1f} Å²)")
        print(f"✓ Thickness under protein = {thickness_in:.2f} Å vs. bulk membrane = "
              f"{thickness_out:.2f} Å (thinning = {self.thinning_fraction*100:.1f}%)")
        if np.isfinite(self.water_core_density_ratio_protein):
            print(f"✓ Core-water density ratio (under protein / bulk membrane): "
                  f"{self.water_core_density_ratio_protein:.2f}×")

        return self.protein_pore_radius

    # ==================== PACSIN1-PAPER-STYLE CURVATURE ANALYSES ====================
    # Following Mahmood, M.I., Noguchi, H. & Okazaki, K. "Curvature induction
    # and sensing of the F-BAR protein Pacsin1 on lipid membranes via
    # molecular dynamics simulations." Sci Rep 9, 14557 (2019).
    #
    # The paper's Fig. 5b estimates the radius of curvature of a Pacsin1-
    # bent membrane by least-squares circle-fitting the x-z projection of
    # the membrane after centering its peak position. Its Fig. 6 / Methods
    # (Eqs 1-4) locate a bound protein's position (arc-length coordinate
    # s_norm) and orientation (angle theta from the buckling direction) on
    # an analytically-known buckled-membrane shape, and use the tangent-
    # angle derivative C(s) = dpsi/ds together with C(s,theta)=C(s)cos^2(theta)
    # to get the local curvature sensed by the protein. Because a general
    # simulation trajectory does not have an idealized analytic buckled
    # shape, this script reproduces the same *geometric* logic numerically
    # from the fitted membrane midline itself (arc-length parametrization,
    # discrete tangent angle, discrete dpsi/ds), which reduces to the same
    # quantities for a buckled system and generalizes to any membrane shape.
    # ==================================================================

    @staticmethod
    def _fit_circle_2d(x, z):
        """Algebraic (Kasa) least-squares circle fit to 2D points (x, z).
        Returns (cx, cz, R): centre and radius of the best-fit circle."""
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        mask = np.isfinite(x) & np.isfinite(z)
        x, z = x[mask], z[mask]
        A = np.column_stack([x, z, np.ones_like(x)])
        b = x**2 + z**2
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx = sol[0] / 2.0
        cz = sol[1] / 2.0
        R = np.sqrt(max(sol[2] + cx**2 + cz**2, 0.0))
        return cx, cz, R

    def analyze_membrane_circlefit_curvature(self):
        """Least-squares circle fit to the membrane midplane cross-section
        (x, z), following Mahmood, Noguchi & Okazaki (2019), Fig. 5b: the
        membrane radius of curvature induced by (e.g.) a bound protein
        assembly is estimated by fitting a circle to the x-z projection of
        the membrane midline after centering on its peak position.

        Returns (x_centered, z_centered, cx, cz, R) where cx, cz are the
        fitted circle centre (in centered coordinates) and R the radius
        of curvature (Å) of the current frame's membrane midline.
        """
        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :]
        z_profile = self.zi[y_idx, :]
        z_centered = z_profile - z_profile.mean()
        peak_idx = np.argmax(np.abs(z_centered))
        x_centered = x_profile - x_profile[peak_idx]
        cx, cz, R = self._fit_circle_2d(x_centered, z_centered)
        return x_centered, z_centered, cx, cz, R

    def _protein_radius_of_curvature(self):
        """Radius of curvature of the protein's own banana/crescent-shaped
        backbone, following Mahmood, Noguchi & Okazaki (2019), Figs 2b, 3b
        and S1, where the intrinsic curvature of the Pacsin1 F-BAR dimer
        was tracked over time (and shown to relax from ~25 nm in the
        crystal structure to ~50 nm and larger once bound to a membrane).

        The protein coordinates are PCA-projected onto their own bending
        plane (spanned by the long axis and the out-of-plane "banana"
        direction), and a circle is fit to that 2D projection.
        """
        if len(self.protein) == 0:
            return np.nan
        pos = self.protein.positions - self.protein.positions.mean(axis=0)
        cov = np.cov(pos.T)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        u = pos @ evecs[:, 0]   # long axis
        v = pos @ evecs[:, 2]   # smallest-variance (bending-plane normal) axis
        _, _, R = self._fit_circle_2d(u, v)
        return R

    def _protein_position_orientation_on_arc(self):
        """Numerical generalization of Mahmood, Noguchi & Okazaki (2019)
        Fig. 6a / Methods Eqs 1-4: locate the protein's centre of mass on
        the membrane midline arc length s (normalized s_norm = s/L), its
        orientation theta relative to the local tangent (buckling)
        direction, and the local curvature C(s) = dpsi/ds sensed there
        (psi = tangent angle). Returns (s_norm, theta_deg, C_local).
        """
        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :]
        z_profile = self.zi[y_idx, :]

        ds = np.sqrt(np.diff(x_profile) ** 2 + np.diff(z_profile) ** 2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        L = s[-1]

        dxds = np.gradient(x_profile, s)
        dzds = np.gradient(z_profile, s)
        psi = np.unwrap(np.arctan2(dzds, dxds))
        C_s = np.gradient(psi, s)   # Eq. 3 analogue: C(s) = dpsi/ds

        com = self.protein.center_of_mass()
        com = com.copy()
        com[2] -= self.z_offset   # into the same midplane-centered (z=0) frame as self.zi
        dist2 = (x_profile - com[0]) ** 2 + (z_profile - com[2]) ** 2
        idx = int(np.argmin(dist2))
        s_norm = s[idx] / L if L > 0 else np.nan
        C_here = C_s[idx]

        pos = self.protein.positions - self.protein.center_of_mass()
        cov = np.cov(pos.T)
        evals, evecs = np.linalg.eigh(cov)
        long_axis = evecs[:, np.argmax(evals)]
        axis_xz = np.array([long_axis[0], long_axis[2]])
        if np.linalg.norm(axis_xz) < 1e-9:
            theta = np.nan
        else:
            axis_xz = axis_xz / np.linalg.norm(axis_xz)
            tangent = np.array([np.cos(psi[idx]), np.sin(psi[idx])])
            cos_theta = np.clip(abs(np.dot(axis_xz, tangent)), -1.0, 1.0)
            theta = np.degrees(np.arccos(cos_theta))  # 0-90 deg, as in the paper's convention

        return s_norm, theta, C_here

    def estimate_tube_areal_density(self, R_bind_angstrom, dG_bind_kbt,
                                     kc_bilayer_kbt=DEFAULT_BILAYER_KC_KBT):
        """Estimate the protein areal number density needed to generate a
        membrane tube of radius R_bind, following Mahmood, Noguchi &
        Okazaki (2019), Eqs 5-6:

            f_tube(R) = (kc/2) * (1/R^2)          [elastic cost per unit area]
            rho       = f_tube(R_bind) / |dG_bind|

        kc_bilayer_kbt : full-bilayer bending rigidity (paper: ~20 kBT);
                         independent of the monolayer kc used elsewhere.
        R_bind_angstrom: preferred/target radius of curvature (Å), e.g.
                         from plot_protein_curvature_sensing().
        dG_bind_kbt    : curvature-dependent protein binding free energy
                         at R_bind relative to a flat membrane (kBT;
                         negative = favourable). This is a PMF/umbrella-
                         sampling result that cannot be derived from
                         geometry alone -- supply a literature or
                         computed value.
        """
        f_tube = 0.5 * kc_bilayer_kbt / (R_bind_angstrom ** 2)   # kBT / Å^2
        rho_ang2 = f_tube / abs(dG_bind_kbt)                      # proteins / Å^2
        rho_nm2 = rho_ang2 * 100.0                                # proteins / nm^2
        area_per_protein_nm2 = 1.0 / rho_nm2 if rho_nm2 > 0 else np.nan

        fig, ax = plt.subplots(figsize=(8, 6))
        R_range = np.linspace(max(10.0, R_bind_angstrom * 0.3), R_bind_angstrom * 3, 200)
        f_range = 0.5 * kc_bilayer_kbt / (R_range ** 2)
        ax.plot(R_range / 10.0, f_range, 'b-', linewidth=2.5,
                label=r'$f_{tube}(R)=\frac{k_c}{2R^2}$')
        ax.axvline(R_bind_angstrom / 10.0, color='red', linestyle='--',
                   label=f'$R_{{bind}}$ = {R_bind_angstrom/10:.1f} nm')
        ax.axhline(f_tube, color='gray', linestyle=':', alpha=0.7)
        ax.set_xlabel('Tube radius, R (nm)', fontweight='bold')
        ax.set_ylabel(r'Elastic cost, $f_{tube}$ (k$_B$T/Å$^2$)', fontweight='bold')
        ax.set_title('Tube-Formation Elastic Cost & Required Protein Density\n'
                      '(cf. Mahmood, Noguchi & Okazaki, Sci Rep 2019, Eqs 5-6)',
                      fontweight='bold', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        txt = (f"kc (bilayer) = {kc_bilayer_kbt:.1f} kBT\n"
               f"R_bind = {R_bind_angstrom/10:.1f} nm\n"
               f"ΔG_bind = {dG_bind_kbt:+.2f} kBT\n"
               f"f_tube(R_bind) = {f_tube:.4f} kBT/Å²\n"
               f"ρ_protein ≈ {rho_nm2:.4f} nm⁻²\n"
               f"Area/protein ≈ {area_per_protein_nm2:.1f} nm²")
        ax.text(0.98, 0.95, txt, transform=ax.transAxes, fontsize=9.5, va='top', ha='right',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        self._save_fig('tube_areal_density_estimate.png')
        print(f"✓ Estimated protein areal density for R_bind={R_bind_angstrom/10:.1f} nm: "
              f"{rho_nm2:.4f} nm⁻² ({area_per_protein_nm2:.1f} nm² per protein)")
        return rho_nm2, area_per_protein_nm2

    def collect_time_series_data(self):
        """Collect curvature (and Pacsin1-paper-style) data from multiple frames"""
        print("\n" + "="*80)
        print("COLLECTING TIME-SERIES DATA")
        print("="*80)

        n_frames = len(self.u.trajectory)
        print(f"Processing {n_frames} frames...")

        self.circlefit_radius_over_time = []
        self.protein_radius_over_time = []
        self.protein_snorm_over_time = []
        self.protein_theta_over_time = []
        self.protein_local_curvature_over_time = []

        for frame_idx, ts in enumerate(self.u.trajectory[::max(1, n_frames//20)]):  # Sample ~20 frames
            self.extract_surface_data()
            self.create_surface_grid()
            self.calculate_curvature()
            self.calculate_lipid_deflection()
            self.calculate_curvature_frustration()

            # Store time-series data
            time_ps = ts.time
            self.time_points.append(time_ps)
            self.curvature_over_time.append(np.mean(np.abs(self.curvature)))
            self.thickness_over_time.append(np.mean(self.thickness))
            self.deflection_over_time.append(np.mean(self.deflection_total))
            self.frustration_energy_over_time.append(self.total_gc_kbt)

            # Midplane height field, needed for the undulation-spectrum
            # bending-modulus estimate (compute_undulation_spectrum)
            self.height_fields_over_time.append(self.zi.copy())

            # Pore radius and water permeation, sampled at the same frames
            # as everything else so they share one consistent time axis
            self.analyze_pore_radius()
            self.pore_radius_over_time.append(self.max_pore_radius)
            self.analyze_water_permeation()
            self.water_influx_over_time.append(self.water_influx)
            self.water_core_count_over_time.append(self.water_core_count)

            # Protein/peptide-induced pore -- only tracked if a protein is
            # present; the list simply stays empty otherwise (nothing to plot).
            if len(self.protein) > 0:
                self.analyze_protein_induced_pore()
                self.protein_pore_radius_over_time.append(self.protein_pore_radius)

            # Area per lipid (uses the current frame's box, so this tracks
            # any lateral-area fluctuation e.g. under NPT conditions)
            self.calculate_area_per_lipid()
            self.apl_upper_over_time.append(self.apl_upper)
            self.apl_lower_over_time.append(self.apl_lower)

            # Mahmood, Noguchi & Okazaki (2019) Fig. 5b -- circle-fit radius
            _, _, _, _, R_cf = self.analyze_membrane_circlefit_curvature()
            self.circlefit_radius_over_time.append(R_cf)

            # Fig. 2b/3b/S1 -- protein's own intrinsic (banana) curvature
            self.protein_radius_over_time.append(self._protein_radius_of_curvature())

            # Fig. 6a/b -- protein position (s_norm) & orientation (theta) on the arc
            if len(self.protein) > 0:
                s_norm, theta, C_local = self._protein_position_orientation_on_arc()
                self.protein_snorm_over_time.append(s_norm)
                self.protein_theta_over_time.append(theta)
                self.protein_local_curvature_over_time.append(C_local)
            else:
                self.protein_snorm_over_time.append(np.nan)
                self.protein_theta_over_time.append(np.nan)
                self.protein_local_curvature_over_time.append(np.nan)

            if (frame_idx + 1) % max(1, n_frames//20 // 5) == 0:
                print(f"  ✓ Processed frame {frame_idx + 1}/{n_frames}")

        self.time_points = np.array(self.time_points)
        self.curvature_over_time = np.array(self.curvature_over_time)
        self.thickness_over_time = np.array(self.thickness_over_time)
        self.deflection_over_time = np.array(self.deflection_over_time)
        self.frustration_energy_over_time = np.array(self.frustration_energy_over_time)
        self.circlefit_radius_over_time = np.array(self.circlefit_radius_over_time)
        self.protein_radius_over_time = np.array(self.protein_radius_over_time)
        self.protein_snorm_over_time = np.array(self.protein_snorm_over_time)
        self.protein_theta_over_time = np.array(self.protein_theta_over_time)
        self.protein_local_curvature_over_time = np.array(self.protein_local_curvature_over_time)
        self.pore_radius_over_time = np.array(self.pore_radius_over_time)
        self.water_influx_over_time = np.array(self.water_influx_over_time)
        self.water_core_count_over_time = np.array(self.water_core_count_over_time)
        self.apl_upper_over_time = np.array(self.apl_upper_over_time)
        self.apl_lower_over_time = np.array(self.apl_lower_over_time)
        self.protein_pore_radius_over_time = np.array(self.protein_pore_radius_over_time)

        # Simulation-derived bending modulus from the collected height
        # fields, as an independent cross-check on the literature kc_kbt
        # used throughout the Marsh (2007) frustration-energy analysis.
        print("\n" + "="*80)
        print("UNDULATION SPECTRUM -> SIMULATION-DERIVED BENDING MODULUS")
        print("="*80)
        self.compute_undulation_spectrum()

        # Restore to the final frame so downstream single-frame plots reflect
        # the last (most-equilibrated) snapshot rather than an interior
        # frame left over from the ~20-frame sampling loop above.
        self.u.trajectory[-1]
        self.extract_surface_data()
        self.create_surface_grid()
        self.calculate_curvature()
        self.calculate_lipid_deflection()
        self.calculate_curvature_frustration()
        self.calculate_area_per_lipid()
        self.analyze_pore_radius()
        self.analyze_water_permeation()
        if len(self.protein) > 0:
            self.analyze_protein_induced_pore()

    # ==================== TOP & SIDE VIEW PLOTS ====================

    def plot_top_side_view(self):
        """Create top and side view like reference image"""
        fig = plt.figure(figsize=(18, 8))

        # TOP VIEW
        ax_top = fig.add_subplot(121)

        # Plot lipid atoms as red scatter
        lipid_xy = self.lipid_atoms.positions[:, :2]
        ax_top.scatter(lipid_xy[:, 0], lipid_xy[:, 1], c='red', s=1, alpha=0.4, label='Lipid atoms')

        # Draw upper leaflet contours (blue)
        contours_u = ax_top.contour(self.xi, self.yi, self.zu, levels=8, colors='blue', linewidths=2.5)

        # Draw lower leaflet contours (red)
        contours_l = ax_top.contour(self.xi, self.yi, self.zl, levels=8, colors='red', linewidths=1.5, linestyles='--', alpha=0.7)

        ax_top.set_xlabel('X (Å)', fontsize=13, fontweight='bold')
        ax_top.set_ylabel('Y (Å)', fontsize=13, fontweight='bold')
        ax_top.set_title('Top View', fontsize=15, fontweight='bold', pad=15)
        ax_top.legend(['Lipid atoms', 'Upper leaflet', 'Lower leaflet'], fontsize=11, loc='upper right')
        ax_top.grid(True, alpha=0.2)
        ax_top.set_aspect('equal')

        # SIDE VIEW
        ax_side = fig.add_subplot(122)

        # Get the middle cross-section (y = mean)
        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :]
        zu_profile = self.zu[y_idx, :]
        zl_profile = self.zl[y_idx, :]

        # Plot upper leaflet
        ax_side.fill_between(x_profile, zu_profile, zu_profile + 10, color='blue', alpha=0.4, label='Upper leaflet')
        ax_side.plot(x_profile, zu_profile, 'b-', linewidth=3)

        # Plot lower leaflet
        ax_side.fill_between(x_profile, zl_profile - 10, zl_profile, color='red', alpha=0.4, label='Lower leaflet')
        ax_side.plot(x_profile, zl_profile, 'r-', linewidth=3)

        # Bilayer midplane reference (z = 0 by construction; see create_surface_grid)
        ax_side.axhline(0, color='black', linewidth=1.8, linestyle='-', alpha=0.8,
                        label='Bilayer midplane (z=0)')
        ax_side.text(x_profile[-1], 0, '  z=0', va='center', ha='left', fontsize=9, fontweight='bold')

        # Add radius of curvature text
        rad_curv = np.nanmean(self.radius_of_curvature[~np.isinf(self.radius_of_curvature)])
        ax_side.text(0.5, 0.95, f'Radius of curvature = {rad_curv:.1f} Å',
                    transform=ax_side.transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    verticalalignment='top', horizontalalignment='center')

        ax_side.set_xlabel('X (Å)', fontsize=13, fontweight='bold')
        ax_side.set_ylabel('Z (Å), midplane-centered', fontsize=13, fontweight='bold')
        ax_side.set_title('Side View', fontsize=15, fontweight='bold', pad=15)
        ax_side.legend(fontsize=10, loc='upper right')
        ax_side.grid(True, alpha=0.2)

        plt.tight_layout()
        self._save_fig('top_side_view.png')

    # ==================== CROSS-SECTION PROFILES ====================

    def plot_cross_section_profiles(self):
        """Plot cross-section profiles (Z vs X)"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        # Get middle cross-section
        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :]           # Å
        zu_profile = self.zu[y_idx, :]           # Å
        zl_profile = self.zl[y_idx, :]           # Å
        thickness_profile = self.thickness[y_idx, :]   # Å
        curvature_profile = self.curvature[y_idx, :]    # Å^-1

        # Plot 1: Z profile
        ax1 = fig.add_subplot(gs[0, :])
        ax1.fill_between(x_profile, zu_profile, zu_profile + 10, color='blue', alpha=0.3, label='Upper leaflet')
        ax1.plot(x_profile, zu_profile, 'b-', linewidth=2.5, label='Upper surface')
        ax1.fill_between(x_profile, zl_profile - 10, zl_profile, color='red', alpha=0.3, label='Lower leaflet')
        ax1.plot(x_profile, zl_profile, 'r-', linewidth=2.5, label='Lower surface')
        ax1.axhline(0, color='black', linewidth=1.8, alpha=0.8, label='Bilayer midplane (z=0)')
        ax1.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Z (Å), midplane-centered', fontsize=12, fontweight='bold')
        ax1.set_title('Membrane Z-Profile (Cross-Section)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper right', framealpha=0.95)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Thickness profile
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(x_profile, thickness_profile, 'g-', linewidth=2.5)
        ax2.fill_between(x_profile, thickness_profile, alpha=0.3, color='green')
        ax2.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Thickness (Å)', fontsize=12, fontweight='bold')
        ax2.set_title('Membrane Thickness Profile', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Curvature profile
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(x_profile, curvature_profile, 'purple', linewidth=2.5)
        ax3.fill_between(x_profile, curvature_profile, alpha=0.3, color='purple')
        ax3.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Curvature (Å⁻¹)', fontsize=12, fontweight='bold')
        ax3.set_title('Membrane Curvature Profile', fontsize=13, fontweight='bold')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)

        plt.suptitle('Cross-Section Analysis', fontsize=16, fontweight='bold', y=0.995)
        self._save_fig('cross_section_profiles.png')

    # ==================== SPONTANEOUS CURVATURE FRUSTRATION PLOTS (Marsh 2007) ====================

    def plot_curvature_frustration(self):
        """Plot the actual leaflet curvature c1, the composition-weighted
        spontaneous curvature c0, and the resulting elastic bending /
        frustration energy density maps, following Marsh (2007)."""
        fig = plt.figure(figsize=(18, 11))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

        extent = [self.xi.min(), self.xi.max(), self.yi.min(), self.yi.max()]

        # c1 maps (actual curvature, Å^-1)
        c1_min = min(np.nanmin(self.curvature_u), np.nanmin(self.curvature_l))
        c1_max = max(np.nanmax(self.curvature_u), np.nanmax(self.curvature_l))
        c1_lim = max(abs(c1_min), abs(c1_max))

        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(self.curvature_u, extent=extent, origin='lower', cmap='RdBu_r',
                          vmin=-c1_lim, vmax=c1_lim, aspect='equal')
        ax1.set_title(f'Upper leaflet c1  (mean {np.nanmean(self.curvature_u):+.4f} Å⁻¹)',
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (Å)'); ax1.set_ylabel('Y (Å)')
        plt.colorbar(im1, ax=ax1, label='c1 (Å⁻¹)', shrink=0.85)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(self.curvature_l, extent=extent, origin='lower', cmap='RdBu_r',
                          vmin=-c1_lim, vmax=c1_lim, aspect='equal')
        ax2.set_title(f'Lower leaflet c1  (mean {np.nanmean(self.curvature_l):+.4f} Å⁻¹)',
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (Å)'); ax2.set_ylabel('Y (Å)')
        plt.colorbar(im2, ax=ax2, label='c1 (Å⁻¹)', shrink=0.85)

        # Summary text panel
        ax3 = fig.add_subplot(gs[0, 2])
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
            f"  upper = {self.gc_flat_upper * self.xi.ptp() * self.yi.ptp():.1f} kBT\n"
            f"  lower = {self.gc_flat_lower * self.xi.ptp() * self.yi.ptp():.1f} kBT"
        )
        ax3.text(0.02, 0.98, summary, transform=ax3.transAxes, fontsize=10.5,
                 verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                           edgecolor='black', linewidth=1.5))

        # gc energy density maps
        gc_max = max(np.nanmax(self.gc_upper), np.nanmax(self.gc_lower))

        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(self.gc_upper, extent=extent, origin='lower', cmap='inferno',
                          vmin=0, vmax=gc_max, aspect='equal')
        ax4.set_title('Upper leaflet gc (elastic energy density)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('X (Å)'); ax4.set_ylabel('Y (Å)')
        plt.colorbar(im4, ax=ax4, label='gc (kBT/Å²)', shrink=0.85)

        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(self.gc_lower, extent=extent, origin='lower', cmap='inferno',
                          vmin=0, vmax=gc_max, aspect='equal')
        ax5.set_title('Lower leaflet gc (elastic energy density)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('X (Å)'); ax5.set_ylabel('Y (Å)')
        plt.colorbar(im5, ax=ax5, label='gc (kBT/Å²)', shrink=0.85)

        # Cross-section of c1 vs c0 (upper leaflet, middle row)
        ax6 = fig.add_subplot(gs[1, 2])
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

        plt.suptitle('Spontaneous Curvature Frustration Analysis (Marsh, Biophys J 2007;93:3884)',
                      fontsize=15, fontweight='bold', y=0.995)
        self._save_fig('curvature_frustration_analysis.png')

    def plot_frustration_energy_timeseries(self):
        """Plot the total elastic bending/frustration energy over time"""
        if len(self.time_points) == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        time_ns = self._time_ns()
        ax.plot(time_ns, self.frustration_energy_over_time, 'o-', color='darkorange', linewidth=2.5)
        ax.fill_between(time_ns, self.frustration_energy_over_time, alpha=0.25, color='darkorange')
        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total elastic bending energy, gc (kBT)', fontsize=12, fontweight='bold')
        ax.set_title('Membrane Curvature-Frustration Energy Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_fig('frustration_energy_timeseries.png')

    # ==================== TIME-SERIES ANALYSIS ====================

    def plot_radius_of_curvature_timeseries(self):
        """Plot radius of curvature over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # All time axes in this script are in nanoseconds (ns)
        time_ns = self._time_ns()

        # Plot 1: Radius of curvature over time
        ax = axes[0, 0]
        rad_curv_list = []
        for curvature in self.curvature_over_time:
            if curvature > 1e-6:
                rad_curv_list.append(2.0 / curvature)   # R = 2/|c1|, c1 = 2H convention
            else:
                rad_curv_list.append(np.nan)

        ax.plot(time_ns, rad_curv_list, 'b-', linewidth=2, label='Instantaneous (differential-geometry)')
        ax.fill_between(time_ns, rad_curv_list, alpha=0.3)

        # Add moving average
        if len(rad_curv_list) > 5:
            rad_curv_clean = np.array(rad_curv_list)
            rad_curv_clean = np.nan_to_num(rad_curv_clean, nan=np.nanmean(rad_curv_clean))
            moving_avg = np.convolve(rad_curv_clean, np.ones(5)/5, mode='same')
            ax.plot(time_ns, moving_avg, 'r--', linewidth=2.5, label='Moving average (5-frame)')

        # Overlay the circle-fit radius of curvature (Mahmood, Noguchi &
        # Okazaki 2019, Fig. 5b style) for direct comparison
        if len(self.circlefit_radius_over_time) == len(time_ns):
            ax.plot(time_ns, self.circlefit_radius_over_time, 'g:', linewidth=2.2,
                    label='Circle-fit R(t) (Fig. 5b style)')

        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Radius of Curvature (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Radius of Curvature Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: Thickness over time
        ax = axes[0, 1]
        ax.plot(time_ns, self.thickness_over_time, 'g-', linewidth=2)
        ax.fill_between(time_ns, self.thickness_over_time, alpha=0.3, color='green')
        ax.axhline(y=np.mean(self.thickness_over_time), color='r', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(self.thickness_over_time):.2f}')
        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Thickness (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Membrane Thickness Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Plot 3: Deflection over time
        ax = axes[1, 0]
        ax.plot(time_ns, self.deflection_over_time, 'orange', linewidth=2)
        ax.fill_between(time_ns, self.deflection_over_time, alpha=0.3, color='orange')
        ax.axhline(y=np.mean(self.deflection_over_time), color='r', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(self.deflection_over_time):.2f}')
        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Deflection (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Lipid Deflection Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Plot 4: Statistics
        ax = axes[1, 1]
        ax.axis('off')

        rad_curv_clean = np.array(rad_curv_list)
        rad_curv_clean = np.nan_to_num(rad_curv_clean, nan=np.nanmean(rad_curv_clean))

        stats_text = f"""TIME-SERIES STATISTICS

Radius of Curvature (differential geometry):
  Mean: {np.nanmean(rad_curv_clean):.1f} Å
  Min: {np.nanmin(rad_curv_clean):.1f} Å
  Max: {np.nanmax(rad_curv_clean):.1f} Å
  Std: {np.nanstd(rad_curv_clean):.1f} Å

Radius of Curvature (circle fit, Fig. 5b style):
  Mean: {np.nanmean(self.circlefit_radius_over_time):.1f} Å
  Std: {np.nanstd(self.circlefit_radius_over_time):.1f} Å

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

        plt.suptitle('Time-Series Analysis', fontsize=16, fontweight='bold', y=0.995)
        self._save_fig('radius_curvature_timeseries.png')

    def plot_thickness_over_time(self):
        """Standalone membrane thickness vs. time plot (ns), with the
        average thickness computed over the ENTIRE sampled trajectory
        explicitly calculated, annotated on the plot, and printed to the
        console. Saved as its own individual file."""
        time_ns = self._time_ns()
        thickness = np.asarray(self.thickness_over_time, dtype=float)

        # ---- Average calculation over the entire time series ----
        mean_t = float(np.mean(thickness))
        std_t = float(np.std(thickness))
        min_t = float(np.min(thickness))
        max_t = float(np.max(thickness))
        sem_t = std_t / np.sqrt(len(thickness)) if len(thickness) > 1 else 0.0
        self.avg_thickness = mean_t
        self.std_thickness = std_t

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_ns, thickness, 'g-', linewidth=2.2, marker='o', markersize=4,
                label='Instantaneous thickness')
        ax.fill_between(time_ns, thickness, alpha=0.25, color='green')
        ax.axhline(mean_t, color='red', linestyle='--', linewidth=2.2,
                   label=f'Average over full trajectory: {mean_t:.2f} ± {std_t:.2f} Å')
        ax.fill_between(time_ns, mean_t - std_t, mean_t + std_t, color='red', alpha=0.08)

        stats_txt = (f"AVERAGE MEMBRANE THICKNESS\n"
                     f"(entire {time_ns[-1]-time_ns[0]:.1f} ns trajectory, N={len(thickness)} frames)\n\n"
                     f"Mean   = {mean_t:.3f} Å\n"
                     f"Std    = {std_t:.3f} Å\n"
                     f"SEM    = {sem_t:.3f} Å\n"
                     f"Min    = {min_t:.3f} Å\n"
                     f"Max    = {max_t:.3f} Å")
        ax.text(0.02, 0.98, stats_txt, transform=ax.transAxes, fontsize=10, va='top', ha='left',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                                               edgecolor='black', linewidth=1.3))

        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Membrane Thickness (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Membrane Thickness Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_fig('membrane_thickness_over_time.png')

        print(f"✓ Average membrane thickness over full trajectory ({time_ns[-1]-time_ns[0]:.1f} ns, "
              f"N={len(thickness)} frames): {mean_t:.3f} ± {std_t:.3f} Å "
              f"(min {min_t:.3f} / max {max_t:.3f} Å)")

    def plot_radius_of_curvature_over_time(self):
        """Standalone radius-of-curvature vs. time plot (ns), comparing the
        differential-geometry estimate (R = 2/|c1|) with the Mahmood,
        Noguchi & Okazaki (2019)-style circle-fit estimate (Fig. 5b). The
        average radius of curvature over the ENTIRE sampled trajectory is
        explicitly calculated for both methods, annotated on the plot, and
        printed to the console. Saved as its own individual file."""
        time_ns = self._time_ns()

        rad_curv_list = []
        for curvature in self.curvature_over_time:
            if curvature > 1e-6:
                rad_curv_list.append(2.0 / curvature)
            else:
                rad_curv_list.append(np.nan)
        rad_curv = np.array(rad_curv_list, dtype=float)

        # ---- Average calculation over the entire time series ----
        mean_dg = float(np.nanmean(rad_curv))
        std_dg = float(np.nanstd(rad_curv))
        min_dg = float(np.nanmin(rad_curv))
        max_dg = float(np.nanmax(rad_curv))
        self.avg_radius_of_curvature = mean_dg
        self.std_radius_of_curvature = std_dg

        have_cf = len(self.circlefit_radius_over_time) == len(time_ns)
        if have_cf:
            cf = np.asarray(self.circlefit_radius_over_time, dtype=float)
            mean_cf = float(np.nanmean(cf))
            std_cf = float(np.nanstd(cf))
            min_cf = float(np.nanmin(cf))
            max_cf = float(np.nanmax(cf))
            self.avg_radius_of_curvature_circlefit = mean_cf
            self.std_radius_of_curvature_circlefit = std_cf

        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.plot(time_ns, rad_curv, 'b-o', linewidth=2, markersize=4,
                label='Differential-geometry R(t) = 2/|c1|')
        ax.axhline(mean_dg, color='blue', linestyle='--', linewidth=2,
                   label=f'Diff.-geom. average: {mean_dg:.1f} ± {std_dg:.1f} Å')

        if have_cf:
            ax.plot(time_ns, cf, 'g-s', linewidth=2, markersize=4,
                    label='Circle-fit R(t) (Fig. 5b style)')
            ax.axhline(mean_cf, color='green', linestyle=':', linewidth=2.2,
                       label=f'Circle-fit average: {mean_cf:.1f} ± {std_cf:.1f} Å')

        if len(rad_curv) > 5:
            rad_curv_clean = np.nan_to_num(rad_curv, nan=np.nanmean(rad_curv))
            moving_avg = np.convolve(rad_curv_clean, np.ones(5)/5, mode='same')
            ax.plot(time_ns, moving_avg, 'r-.', linewidth=1.6, alpha=0.7,
                    label='Moving average (5-frame, diff. geom.)')

        stats_txt = (f"AVERAGE RADIUS OF CURVATURE\n"
                     f"(entire {time_ns[-1]-time_ns[0]:.1f} ns trajectory, N={len(rad_curv)} frames)\n\n"
                     f"Diff.-geom.: {mean_dg:.1f} ± {std_dg:.1f} Å  (min {min_dg:.1f} / max {max_dg:.1f})")
        if have_cf:
            stats_txt += f"\nCircle fit : {mean_cf:.1f} ± {std_cf:.1f} Å  (min {min_cf:.1f} / max {max_cf:.1f})"
        ax.text(0.02, 0.98, stats_txt, transform=ax.transAxes, fontsize=9.5, va='top', ha='left',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                                               edgecolor='black', linewidth=1.3))

        ax.set_xlabel('Time (ns)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Radius of Curvature (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Radius of Curvature Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9.5, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_fig('radius_of_curvature_over_time.png')

        print(f"✓ Average radius of curvature (differential geometry) over full trajectory "
              f"({time_ns[-1]-time_ns[0]:.1f} ns, N={len(rad_curv)} frames): "
              f"{mean_dg:.1f} ± {std_dg:.1f} Å (min {min_dg:.1f} / max {max_dg:.1f} Å)")
        if have_cf:
            print(f"✓ Average radius of curvature (circle fit, Fig. 5b style) over full trajectory: "
                  f"{mean_cf:.1f} ± {std_cf:.1f} Å (min {min_cf:.1f} / max {max_cf:.1f} Å)")

    def plot_circlefit_curvature_analysis(self):
        """Circle-fit membrane radius of curvature, following Mahmood,
        Noguchi & Okazaki (2019), Fig. 5b: the x-z projection of the
        membrane midline is fit by a least-squares circle after centering
        on the peak position; R(t) tracks how strongly the membrane is
        bent over the course of the trajectory."""
        x_c, z_c, cx, cz, R_now = self.analyze_membrane_circlefit_curvature()

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        ax = axes[0]
        ax.plot(x_c, z_c, 'o', color='steelblue', markersize=3, alpha=0.6, label='Membrane midline (fit)')
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(cx + R_now * np.cos(theta), cz + R_now * np.sin(theta), 'g-', linewidth=2.5,
                label=f'Circle fit, R = {R_now:.1f} Å')
        ax.plot([cx], [cz], 'k+', markersize=12, markeredgewidth=2, label='Fitted centre')
        ax.set_xlabel('X (centered) (Å)', fontweight='bold')
        ax.set_ylabel('Z (centered) (Å)', fontweight='bold')
        ax.set_title('Least-Squares Circle Fit (current frame)', fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if len(self.circlefit_radius_over_time) > 0:
            t_ns = self._time_ns()
            R_t = self.circlefit_radius_over_time
            ax.plot(t_ns, R_t, 'o-', color='seagreen', linewidth=2)
            ax.axhline(np.nanmean(R_t), color='red', linestyle='--',
                       label=f'Mean: {np.nanmean(R_t):.1f} Å')
            ax.set_xlabel('Time (ns)', fontweight='bold')
            ax.set_ylabel('Circle-fit radius of curvature (Å)', fontweight='bold')
            ax.set_title('Radius of Curvature Over Time (circle fit)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        plt.suptitle('Membrane Circle-Fit Curvature\n(cf. Mahmood, Noguchi & Okazaki, Sci Rep 2019, Fig. 5b)',
                      fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._save_fig('membrane_circlefit_curvature.png')

    # ==================== EXTENDED DYNAMICS: APL, UNDULATIONS, PORE/WATER ====================

    def plot_extended_dynamics_analysis(self):
        """Four-panel figure covering the new analyses added to this
        script: (1) the Helfrich undulation spectrum <|h(q)|^2>/A vs q on
        log-log axes with the fitted q^-4 bending-modulus curve, (2) area
        per lipid (APL) for each leaflet over time, (3) maximum pore radius
        over time, and (4) water molecules within the membrane region over
        time (permeation/hydration dynamics)."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))

        # ---- Panel 1: undulation spectrum ----
        ax = axes[0, 0]
        if self.q_bins is not None and np.any(np.isfinite(self.Sq_bins)):
            valid = np.isfinite(self.Sq_bins) & (self.Sq_bins > 0) & (self.q_bins > 0)
            ax.loglog(self.q_bins[valid], self.Sq_bins[valid], 'o', color='darkorange',
                      markersize=6, label='Binned $\\langle|h(q)|^2\\rangle / A$')
            if np.isfinite(self.kc_undulation_kbt):
                q_line = np.geomspace(self.q_bins[valid].min(), self.q_bins[valid].max(), 100)
                ax.loglog(q_line, 1.0 / (self.kc_undulation_kbt * q_line ** 4), '-',
                          color='navy', linewidth=2.2,
                          label=(f'Fit: $k_c$={self.kc_undulation_kbt:.1f}'
                                 f'±{self.kc_undulation_err_kbt:.1f} $k_BT$'))
                ax.loglog(q_line, 1.0 / (self.kc_kbt * q_line ** 4), '--',
                          color='gray', linewidth=1.6, alpha=0.8,
                          label=f'Literature $k_c$={self.kc_kbt:.1f} $k_BT$ (Marsh model input)')
            ax.set_xlabel(r'$q$ (Å$^{-1}$)', fontweight='bold')
            ax.set_ylabel(r'$\langle|h(q)|^2\rangle / A$ (Å$^2$)', fontweight='bold')
            ax.set_title('Undulation Spectrum → Bending Modulus $k_c$', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, which='both', alpha=0.25)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Undulation spectrum unavailable\n(need ≥3 sampled frames)',
                    ha='center', va='center', transform=ax.transAxes)

        # ---- Panel 2: area per lipid over time ----
        ax = axes[0, 1]
        t_ns = self._time_ns()
        if len(self.apl_upper_over_time) == len(t_ns) and len(t_ns) > 0:
            ax.plot(t_ns, self.apl_upper_over_time, 'o-', color='steelblue',
                    label=f'Upper (mean {np.nanmean(self.apl_upper_over_time):.1f} Å²)')
            ax.plot(t_ns, self.apl_lower_over_time, 's-', color='indianred',
                    label=f'Lower (mean {np.nanmean(self.apl_lower_over_time):.1f} Å²)')
            ax.set_xlabel('Time (ns)', fontweight='bold')
            ax.set_ylabel('Area per lipid (Å²)', fontweight='bold')
            ax.set_title('Area Per Lipid Over Time', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        # ---- Panel 3: pore radius over time ----
        ax = axes[1, 0]
        if len(self.pore_radius_over_time) == len(t_ns) and len(t_ns) > 0:
            ax.plot(t_ns, self.pore_radius_over_time, 'o-', color='purple')
            ax.axhline(np.nanmean(self.pore_radius_over_time), color='red', linestyle='--',
                       label=f'Mean: {np.nanmean(self.pore_radius_over_time):.2f} Å')
            ax.set_xlabel('Time (ns)', fontweight='bold')
            ax.set_ylabel('Max pore radius (Å)', fontweight='bold')
            ax.set_title('Pore Radius Over Time', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        # ---- Panel 4: water permeation over time ----
        ax = axes[1, 1]
        if len(self.water_influx_over_time) == len(t_ns) and len(t_ns) > 0:
            ax.plot(t_ns, self.water_influx_over_time, 'o-', color='teal')
            ax.axhline(np.nanmean(self.water_influx_over_time), color='red', linestyle='--',
                       label=f'Mean: {np.nanmean(self.water_influx_over_time):.0f}')
            ax.set_xlabel('Time (ns)', fontweight='bold')
            ax.set_ylabel('Water molecules in membrane region', fontweight='bold')
            ax.set_title('Water Permeation Over Time', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        plt.suptitle('Extended Dynamics: Undulations, Area/Lipid, Pore & Water Permeation',
                      fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        self._save_fig('extended_dynamics_analysis.png')

    # ==================== WATER PERMEATION DEPTH PROFILE (midplane = 0) ====================

    def plot_water_permeation_depth_profile(self):
        """Water permeation depth profile in the midplane-centered frame
        (z = 0 at the bilayer center). Explicitly marks the mean upper-
        and lower-leaflet positions (blue/red, matching the color
        convention used everywhere else in this script) so it is
        immediately clear how far any given water molecule has penetrated
        past the headgroups and into the hydrophobic core, relative to
        each leaflet -- not just an unlabeled z value."""
        if not hasattr(self, 'water_depths') or len(self.water_depths) == 0:
            print("⚠ No water depth data available -- skipping water permeation depth profile plot.")
            return

        mean_zu = float(np.nanmean(self.zu))
        mean_zl = float(np.nanmean(self.zl))

        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

        # ---- Panel 1: depth histogram, midplane = 0, leaflets marked ----
        ax = axes[0]
        ax.hist(self.water_depths, bins=50, color='steelblue', alpha=0.75,
                edgecolor='k', linewidth=0.3, orientation='horizontal')
        ax.axhspan(0, mean_zu + 5, color='royalblue', alpha=0.07, label='_nolegend_')
        ax.axhspan(mean_zl - 5, 0, color='firebrick', alpha=0.07, label='_nolegend_')
        ax.axhline(0, color='black', linewidth=2.2, label='Bilayer midplane (z = 0)')
        ax.axhline(mean_zu, color='royalblue', linewidth=2, linestyle='--',
                   label=f'Upper leaflet (mean {mean_zu:.1f} Å)')
        ax.axhline(mean_zl, color='firebrick', linewidth=2, linestyle='--',
                   label=f'Lower leaflet (mean {mean_zl:.1f} Å)')
        ax.set_ylabel('Depth relative to bilayer midplane, z (Å)', fontweight='bold')
        ax.set_xlabel('Water molecule count', fontweight='bold')
        ax.set_title('Water Depth Distribution (midplane = 0)', fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

        # ---- Panel 2: broad "in bilayer" vs narrow "midplane/core" water over time ----
        ax2 = axes[1]
        t_ns = self._time_ns()
        if len(self.water_influx_over_time) == len(t_ns) and len(t_ns) > 0:
            ax2.plot(t_ns, self.water_influx_over_time, 'o-', color='steelblue',
                     label=f'In bilayer envelope (mean {np.nanmean(self.water_influx_over_time):.0f})')
        if len(self.water_core_count_over_time) == len(t_ns) and len(t_ns) > 0:
            ax2.plot(t_ns, self.water_core_count_over_time, 's-', color='darkorange',
                     label=f'In midplane/core (mean {np.nanmean(self.water_core_count_over_time):.0f})')
        ax2.set_xlabel('Time (ns)', fontweight='bold')
        ax2.set_ylabel('Water molecule count', fontweight='bold')
        ax2.set_title('Water Permeation Over Time', fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Water Permeation Relative to the Bilayer Midplane (z = 0)',
                      fontsize=15, fontweight='bold')
        plt.tight_layout()
        self._save_fig('water_permeation_depth_profile.png')

    # ==================== PROTEIN/PEPTIDE-INDUCED PORE PLOT ====================

    def plot_protein_induced_pore_analysis(self):
        """Visualize the protein/peptide-induced pore: the thickness map
        with the protein's lateral footprint and induced-pore region
        overlaid, the midplane/core water density (hydration-defect
        signature), a thinning comparison bar chart, and (if sampled) the
        induced pore radius over time. Entirely skipped if no protein is
        present -- there is nothing to visualize."""
        if len(self.protein) == 0 or getattr(self, 'protein_footprint_mask', None) is None:
            print("⚠ No protein-induced pore data available -- skipping plot "
                  "(no protein/peptide in this system).")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        extent = [self.xi.min(), self.xi.max(), self.yi.min(), self.yi.max()]

        # (a) thickness map + footprint + induced pore + protein atoms
        ax = axes[0, 0]
        im = ax.imshow(self.thickness, extent=extent, origin='lower', cmap='viridis', aspect='equal')
        plt.colorbar(im, ax=ax, label='Thickness (Å) = upper − lower leaflet z', shrink=0.85)
        ax.contour(self.xi, self.yi, self.protein_footprint_mask.astype(float), levels=[0.5],
                   colors='white', linewidths=2, linestyles='--')
        if self.protein_pore_mask is not None and np.any(self.protein_pore_mask):
            ax.contourf(self.xi, self.yi, self.protein_pore_mask.astype(float),
                        levels=[0.5, 1.5], colors=['red'], alpha=0.45)
        ax.scatter(self.protein.positions[:, 0], self.protein.positions[:, 1],
                  s=4, color='black', alpha=0.5, label='Protein atoms')
        ax.set_title('Thickness: Protein Footprint (white dashed) &\nInduced Pore (red)',
                     fontweight='bold', fontsize=11)
        ax.set_xlabel('X (Å)'); ax.set_ylabel('Y (Å)')
        ax.legend(fontsize=9, loc='upper right')

        # (b) core-water density map (hydration-defect signature)
        ax = axes[0, 1]
        im2 = ax.imshow(self.water_core_density, extent=extent, origin='lower',
                        cmap='Blues', aspect='equal')
        plt.colorbar(im2, ax=ax, label='Midplane/core water density (counts)', shrink=0.85)
        ax.contour(self.xi, self.yi, self.protein_footprint_mask.astype(float), levels=[0.5],
                   colors='orange', linewidths=2, linestyles='--')
        ax.set_title('Hydrophobic-Core Water Density\n(Hydration-Defect Signature)',
                     fontweight='bold', fontsize=11)
        ax.set_xlabel('X (Å)'); ax.set_ylabel('Y (Å)')

        # (c) thinning bar chart + numeric summary
        ax = axes[1, 0]
        ax.bar(['Under protein', 'Bulk membrane'],
               [self.thickness_in_footprint, self.thickness_outside_footprint],
               color=['firebrick', 'seagreen'])
        ax.set_ylabel('Mean thickness (Å)', fontweight='bold')
        thinning_pct = self.thinning_fraction * 100 if np.isfinite(self.thinning_fraction) else float('nan')
        ax.set_title(f'Membrane Thinning Under Protein ({thinning_pct:.1f}%)',
                    fontweight='bold', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        ratio_txt = (f"{self.water_core_density_ratio_protein:.2f}×"
                     if np.isfinite(self.water_core_density_ratio_protein) else "N/A")
        txt = (f"Protein-induced pore radius = {self.protein_pore_radius:.2f} Å\n"
               f"Protein-induced pore area   = {self.protein_pore_area:.1f} Å²\n"
               f"Core-water density ratio (protein/bulk) = {ratio_txt}")
        ax.text(0.5, -0.30, txt, transform=ax.transAxes, ha='center', va='top', fontsize=9.5,
               family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        # (d) induced pore radius over time
        ax = axes[1, 1]
        t_ns = self._time_ns()
        if len(self.protein_pore_radius_over_time) == len(t_ns) and len(t_ns) > 0:
            ax.plot(t_ns, self.protein_pore_radius_over_time, 'o-', color='crimson')
            ax.axhline(np.nanmean(self.protein_pore_radius_over_time), color='gray', linestyle='--',
                      label=f'Mean: {np.nanmean(self.protein_pore_radius_over_time):.2f} Å')
            ax.set_xlabel('Time (ns)', fontweight='bold')
            ax.set_ylabel('Protein-induced pore radius (Å)', fontweight='bold')
            ax.set_title('Protein-Induced Pore Radius Over Time', fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Insufficient time-series data\nfor a pore-radius time series',
                   ha='center', va='center', transform=ax.transAxes)

        plt.suptitle('Protein/Peptide-Induced Pore Analysis', fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        self._save_fig('protein_induced_pore_analysis.png')

    def plot_protein_intrinsic_curvature(self):
        """Radius of curvature of the protein's own banana-shaped backbone
        over time, cf. Mahmood, Noguchi & Okazaki (2019), Figs 2b, 3b & S1,
        where the F-BAR dimer's intrinsic curvature was tracked and shown
        to vary/relax during the simulation (structural flexibility)."""
        if len(self.protein) == 0:
            print("⚠ No protein atoms found — skipping protein intrinsic curvature analysis")
            return
        R = np.array(self.protein_radius_over_time, dtype=float)
        valid = np.isfinite(R)
        if valid.sum() < 2:
            print("⚠ Insufficient protein data — skipping protein intrinsic curvature analysis")
            return
        t_ns = self._time_ns(self.time_points[valid])
        R = R[valid]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        ax = axes[0]
        ax.plot(t_ns, R, 'o-', color='purple', linewidth=2)
        ax.axhline(np.mean(R), color='red', linestyle='--', label=f'Mean: {np.mean(R):.1f} Å')
        ax.set_xlabel('Time (ns)', fontweight='bold')
        ax.set_ylabel('Protein radius of curvature (Å)', fontweight='bold')
        ax.set_title("Protein's Intrinsic (Banana-Shape) Curvature", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.hist(R, bins=min(15, max(3, len(R)//2)), color='mediumpurple', edgecolor='black', alpha=0.8)
        ax.axvline(np.mean(R), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Radius of curvature (Å)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution', fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')

        plt.suptitle('Protein Intrinsic Curvature Over Time\n'
                      '(cf. Mahmood, Noguchi & Okazaki, Sci Rep 2019, Fig. 2b/3b)',
                      fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._save_fig('protein_intrinsic_curvature.png')

    def plot_protein_curvature_sensing(self):
        """Curvature-sensing analysis of the bound protein, following
        Mahmood, Noguchi & Okazaki (2019), Fig. 6b,c and Eq. 4:
        C(s,theta) = C(s)*cos^2(theta). Because a general MD trajectory
        does not have an idealized analytic buckled shape, s_norm, theta
        and the local curvature C(s) are obtained numerically here (see
        _protein_position_orientation_on_arc) rather than from the
        closed-form elliptic-buckling expression used for the idealized
        system in the original paper.
        """
        if len(self.protein) == 0:
            print("⚠ No protein atoms found — skipping curvature-sensing analysis")
            return
        s = np.array(self.protein_snorm_over_time, dtype=float)
        th = np.array(self.protein_theta_over_time, dtype=float)
        C = np.array(self.protein_local_curvature_over_time, dtype=float)
        valid = np.isfinite(s) & np.isfinite(th) & np.isfinite(C)
        if valid.sum() < 3:
            print("⚠ Insufficient protein position data — skipping curvature-sensing analysis")
            return
        s, th, C = s[valid], th[valid], C[valid]
        C_eff = C * np.cos(np.radians(th)) ** 2  # Eq. 4
        with np.errstate(divide='ignore'):
            R_eff = np.where(np.abs(C_eff) > 1e-6, 1.0 / np.abs(C_eff), np.nan)

        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.32)

        # (a) time series of s_norm, theta
        ax1 = fig.add_subplot(gs[0, 0])
        t_ns = self._time_ns(self.time_points[valid])
        ax1.plot(t_ns, s, 'o-', color='navy', label=r'$s_{norm}$')
        ax1b = ax1.twinx()
        ax1b.plot(t_ns, th, 's-', color='firebrick', label=r'$\theta$ (deg)')
        ax1.set_xlabel('Time (ns)', fontweight='bold')
        ax1.set_ylabel(r'$s_{norm}$ (protein position)', color='navy', fontweight='bold')
        ax1b.set_ylabel(r'$\theta$ (deg, orientation)', color='firebrick', fontweight='bold')
        ax1.set_title('Protein Position & Orientation on Membrane Arc', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # (b) frequency map of (s_norm, theta)
        ax2 = fig.add_subplot(gs[0, 1])
        if len(s) >= 4:
            hb = ax2.hexbin(s, th, gridsize=10, cmap='viridis', mincnt=1)
            plt.colorbar(hb, ax=ax2, label='Frequency')
        else:
            ax2.scatter(s, th, color='navy')
        ax2.set_xlabel(r'$s_{norm}$', fontweight='bold')
        ax2.set_ylabel(r'$\theta$ (deg)', fontweight='bold')
        ax2.set_title('Position–Orientation Frequency Map', fontweight='bold')

        # (c) local curvature/radius sensed vs (s_norm, theta)
        ax3 = fig.add_subplot(gs[1, 0])
        sc = ax3.scatter(s, th, c=R_eff, cmap='inferno_r', s=60, edgecolor='k', linewidth=0.4)
        plt.colorbar(sc, ax=ax3, label='Effective radius of curvature, R(s,θ) (Å)')
        ax3.set_xlabel(r'$s_{norm}$', fontweight='bold')
        ax3.set_ylabel(r'$\theta$ (deg)', fontweight='bold')
        ax3.set_title('Sensed Curvature Map (Eq. 4: C(s,θ)=C(s)cos²θ)', fontweight='bold')

        # (d) free energy profile along R
        ax4 = fig.add_subplot(gs[1, 1])
        R_clean = R_eff[np.isfinite(R_eff)]
        if len(R_clean) >= 3:
            n_bins = min(12, max(3, len(R_clean) // 2))
            counts, edges = np.histogram(R_clean, bins=n_bins)
            centers = 0.5 * (edges[1:] + edges[:-1])
            p = counts / counts.sum()
            with np.errstate(divide='ignore', invalid='ignore'):
                dG = -np.log(p / p.max())
            dG[~np.isfinite(dG)] = np.nan
            ax4.plot(centers, dG, 'o-', color='darkgreen', linewidth=2.5)
            ax4.set_xlabel('Radius of curvature, R (Å)', fontweight='bold')
            ax4.set_ylabel(r'$\Delta G$ (k$_B$T)', fontweight='bold')
            ax4.set_title('Curvature-Sensing Free Energy Profile', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            if np.any(np.isfinite(dG)):
                r_min = centers[np.nanargmin(dG)]
                ax4.axvline(r_min, color='red', linestyle='--', alpha=0.7,
                            label=f'Preferred R ≈ {r_min:.0f} Å')
                ax4.legend(fontsize=10)
        else:
            ax4.axis('off')
            ax4.text(0.5, 0.5, 'Insufficient sampling\nfor a free-energy profile\n'
                                '(need more trajectory frames)',
                      ha='center', va='center', transform=ax4.transAxes)

        plt.suptitle('Protein Curvature Sensing\n'
                      '(cf. Mahmood, Noguchi & Okazaki, Sci Rep 2019, Fig. 6)',
                      fontsize=15, fontweight='bold', y=0.995)
        self._save_fig('protein_curvature_sensing_analysis.png')

    # ==================== CURVATURE-SENSING SCHEMATIC (reference-image style) ====================

    def plot_curvature_sensing_schematic(self, filename='curvature_sensing_schematic.png',
                                          n_points=9, n_highlight=3, frame_label=None):
        """Reproduce the curvature-sensing schematic style of the supplied
        reference figure and of Mahmood, Noguchi & Okazaki (2019) Fig. 6a:
        the membrane midline is sampled at points (gold spheres); at three
        representative locations a local quadratic surface S(x) is fit and
        drawn in blue/red/green; straight "radius" vectors run from the
        circle-fit centre (focal point) to each sample point, and the
        angular spacing Δφ between neighbouring vectors is the discrete
        analogue of the local curvature C(s) = dψ/ds = Δφ/Δs (paper Eq. 3).
        """
        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :]
        z_profile = self.zi[y_idx, :]

        x_c, z_c, cx, cz, R = self.analyze_membrane_circlefit_curvature()
        peak_idx = np.argmax(np.abs(z_profile - z_profile.mean()))
        focal_x = cx + x_profile[peak_idx]
        focal_z = cz + z_profile.mean()

        fig, ax = plt.subplots(figsize=(14, 9))

        thickness_half = np.nanmean(self.thickness) / 2 if hasattr(self, 'thickness') else 15.0
        ax.fill_between(x_profile, z_profile - thickness_half, z_profile + thickness_half,
                         color='#bcdada', alpha=0.6, zorder=1)
        ax.plot(x_profile, z_profile + thickness_half, color='#6f9c9c', lw=1, zorder=1)
        ax.plot(x_profile, z_profile - thickness_half, color='#6f9c9c', lw=1, zorder=1)
        ax.plot(x_profile, z_profile, 'k--', lw=1.6, zorder=3, label='Membrane midline (fit)')

        sample_idx = np.unique(np.linspace(0, len(x_profile) - 1, n_points).astype(int))
        ax.scatter(x_profile[sample_idx], z_profile[sample_idx], s=150, color='gold',
                   edgecolor='#8a6d00', linewidth=1.3, zorder=5, label='Sample points')

        for i in sample_idx:
            ax.plot([focal_x, x_profile[i]], [focal_z, z_profile[i]],
                    color='black', lw=0.6, alpha=0.5, zorder=2)

        colors = ['#1f5fd6', '#d61f1f', '#1f9c4a']  # blue, red, green
        hi_pos = np.linspace(0, len(sample_idx) - 1, n_highlight).astype(int)
        highlight_idx = sample_idx[hi_pos]
        window = max(4, len(x_profile) // 12)
        for k, i in enumerate(highlight_idx):
            lo, hi = max(0, i - window), min(len(x_profile), i + window + 1)
            xs, zs = x_profile[lo:hi], z_profile[lo:hi]
            if len(xs) < 4:
                continue
            coeffs = np.polyfit(xs - x_profile[i], zs, 2)
            xx = np.linspace(xs.min(), xs.max(), 60)
            zz = np.polyval(coeffs, xx - x_profile[i])
            c = colors[k % len(colors)]
            ax.plot(xx, zz, color=c, lw=3.2, zorder=6)
            ax.plot([focal_x, x_profile[i]], [focal_z, z_profile[i]], color=c, lw=2.2, zorder=6)
            ax.text(xx[-1], zz[-1] + thickness_half * 1.4, 'S(x)', color=c, fontsize=13,
                    fontweight='bold', style='italic', ha='center', zorder=7)

        ax.scatter([focal_x], [focal_z], s=70, color='black', zorder=7)

        if n_highlight >= 2:
            mid = n_highlight // 2
            i1 = highlight_idx[max(0, mid - 1)]
            i2 = highlight_idx[mid]
            ang1 = np.degrees(np.arctan2(z_profile[i1] - focal_z, x_profile[i1] - focal_x))
            ang2 = np.degrees(np.arctan2(z_profile[i2] - focal_z, x_profile[i2] - focal_x))
            r_arc = 0.18 * abs(z_profile[peak_idx] - focal_z + 1e-9)
            arc = Arc((focal_x, focal_z), 2 * r_arc, 2 * r_arc, angle=0,
                      theta1=min(ang1, ang2), theta2=max(ang1, ang2), color='black', lw=1.6, zorder=8)
            ax.add_patch(arc)
            mid_ang = np.radians((ang1 + ang2) / 2)
            ax.text(focal_x + r_arc * 1.5 * np.cos(mid_ang), focal_z + r_arc * 1.5 * np.sin(mid_ang),
                    r'$\Delta\varphi$', fontsize=15, fontweight='bold', zorder=8, ha='center')

        # Schematic coordinate triad (bottom-left), axes-fraction coordinates
        ox, oy = 0.05, 0.08
        ax.annotate('', xy=(ox + 0.08, oy), xytext=(ox, oy), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', lw=2, color='black'))
        ax.text(ox + 0.10, oy - 0.015, 'X', transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.annotate('', xy=(ox, oy + 0.11), xytext=(ox, oy), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', lw=2, color='black'))
        ax.text(ox - 0.02, oy + 0.13, 'Z', transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.annotate('', xy=(ox - 0.06, oy - 0.06), xytext=(ox, oy), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', lw=2, color='black'))
        ax.text(ox - 0.10, oy - 0.09, 'Y', transform=ax.transAxes, fontsize=12, fontweight='bold')

        title = 'Curvature-Sensing Schematic — local S(x) fits & Δφ'
        if frame_label:
            title += f'  ({frame_label})'
        title += f'\nGlobal circle-fit R = {R:.1f} Å'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('X (Å)', fontweight='bold')
        ax.set_ylabel('Z (Å)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        plt.tight_layout()
        self._save_fig(filename)

    def plot_curvature_sensing_schematic_series(self, n_snapshots=4):
        """Generate the curvature-sensing schematic (see
        plot_curvature_sensing_schematic) at several time points spanning
        the trajectory, to visualize how local membrane curvature evolves
        over time. Each snapshot is saved as its own individual file."""
        n_frames = len(self.u.trajectory)
        idxs = np.unique(np.linspace(0, n_frames - 1, n_snapshots).astype(int))
        for k, fi in enumerate(idxs):
            ts = self.u.trajectory[fi]
            self.extract_surface_data()
            self.create_surface_grid()
            self.calculate_curvature()
            label = f't = {ts.time/1000:.2f} ns'
            fname = f'curvature_sensing_schematic_t{k+1:02d}.png'
            self.plot_curvature_sensing_schematic(filename=fname, frame_label=label)

        # Restore state to the final frame for any subsequent single-frame analyses
        self.u.trajectory[-1]
        self.extract_surface_data()
        self.create_surface_grid()
        self.calculate_curvature()
        self.calculate_lipid_deflection()

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
            for collection in contour.collections:
                for path in collection.get_paths():
                    vertices = path.vertices
                    if len(vertices) > 0:
                        fig.add_trace(go.Scatter(
                            x=vertices[:, 0], y=vertices[:, 1],
                            mode='lines', line=dict(color='blue', width=2),
                            showlegend=False, name='Upper'
                        ), row=1, col=1)

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

        self._save_html(fig, 'interactive_top_side_view.html')

    # ==================== 2D/3D STANDARD PLOTS ====================

    def plot_2d_standard_all(self):
        """Create standard 2D analysis plots"""
        analyses = [
            (self.thickness, 'RdYlGn_r', 'Thickness', 'thickness_2d_analysis.png'),
            (self.curvature_norm, 'RdBu_r', 'Curvature', 'curvature_2d_analysis.png'),
            (self.water_density, 'Blues', 'Water Density', 'water_2d_analysis.png'),
        ]

        for data, cmap, title, filename in analyses:
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

            # Heatmap
            ax0 = fig.add_subplot(gs[0, :])
            im = ax0.imshow(data, cmap=cmap, origin='lower',
                           extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()])
            contours = ax0.contour(self.xi, self.yi, data, levels=12, colors='white', alpha=0.2)
            ax0.clabel(contours, inline=True, fontsize=8)
            cbar = plt.colorbar(im, ax=ax0, label=title, pad=0.02)
            ax0.set_xlabel('X (Å)', fontweight='bold')
            ax0.set_ylabel('Y (Å)', fontweight='bold')
            ax0.set_title(f'{title} Heatmap', fontweight='bold', fontsize=13)
            ax0.grid(True, alpha=0.1)

            # Histogram
            ax1 = fig.add_subplot(gs[1, 0])
            ax1.hist(data.flatten(), bins=60, color='steelblue', edgecolor='black', alpha=0.7)
            ax1.axvline(np.mean(data), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(data):.2f}')
            ax1.set_xlabel(title, fontweight='bold')
            ax1.set_ylabel('Frequency', fontweight='bold')
            ax1.set_title('Distribution', fontweight='bold')
            ax1.legend(fontsize=10, framealpha=0.9)
            ax1.grid(True, alpha=0.2, axis='y')

            # Statistics
            ax2 = fig.add_subplot(gs[1, 1])
            ax2.axis('off')
            stats_text = f"""{title.upper()} STATISTICS

Mean: {np.mean(data):.3f}
Median: {np.median(data):.3f}
Std Dev: {np.std(data):.3f}
Min: {np.min(data):.3f}
Max: {np.max(data):.3f}"""
            ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=11,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))

            plt.suptitle(f'{title} Analysis', fontsize=16, fontweight='bold', y=0.995)
            self._save_fig(filename)

    def plot_3d_standard_all(self):
        """Create standard 3D plots"""
        # Thickness 3D
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('<b>Upper Leaflet</b>', '<b>Lower Leaflet</b>'),
            horizontal_spacing=0.1
        )

        fig.add_trace(go.Surface(x=self.xi, y=self.yi, z=self.zu, colorscale='Blues', showscale=False, opacity=0.95), row=1, col=1)
        fig.add_trace(go.Surface(x=self.xi, y=self.yi, z=self.zl, colorscale='Reds', showscale=False, opacity=0.95), row=1, col=2)

        fig.update_layout(
            title=dict(text='<b>3D Membrane Leaflet Surfaces</b>', x=0.5, xanchor='center', font=dict(size=18)),
            width=1500, height=700,
            font=dict(family='Arial', size=12)
        )

        self._save_html(fig, 'thickness_3d_analysis.html')

        # Curvature 3D
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('<b>Upper Curvature</b>', '<b>Lower Curvature</b>'),
            horizontal_spacing=0.1
        )

        curv_min = min(self.curvature_u.min(), self.curvature_l.min())
        curv_max = max(self.curvature_u.max(), self.curvature_l.max())

        fig.add_trace(
            go.Surface(x=self.xi, y=self.yi, z=self.zu, surfacecolor=self.curvature_u,
                      colorscale='RdBu', showscale=True, colorbar=dict(title='Curvature', len=0.6, x=0.46),
                      cmin=curv_min, cmax=curv_max, opacity=0.95),
            row=1, col=1
        )

        fig.add_trace(
            go.Surface(x=self.xi, y=self.yi, z=self.zl, surfacecolor=self.curvature_l,
                      colorscale='RdBu', showscale=False, cmin=curv_min, cmax=curv_max, opacity=0.95),
            row=1, col=2
        )

        fig.update_layout(
            title=dict(text='<b>3D Membrane Curvature</b>', x=0.5, xanchor='center', font=dict(size=18)),
            width=1500, height=700,
            font=dict(family='Arial', size=12)
        )

        self._save_html(fig, 'curvature_3d_analysis.html')

        # Deflection 3D
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('<b>Upper Deflection</b>', '<b>Lower Deflection</b>'),
            horizontal_spacing=0.1
        )

        defl_min = min(self.deflection_u.min(), self.deflection_l.min())
        defl_max = max(self.deflection_u.max(), self.deflection_l.max())

        fig.add_trace(
            go.Surface(x=self.xi, y=self.yi, z=self.zu, surfacecolor=self.deflection_u,
                      colorscale='Viridis', showscale=True, colorbar=dict(title='Deflection (Å)', len=0.6, x=0.46),
                      cmin=defl_min, cmax=defl_max, opacity=0.95),
            row=1, col=1
        )

        fig.add_trace(
            go.Surface(x=self.xi, y=self.yi, z=self.zl, surfacecolor=self.deflection_l,
                      colorscale='Viridis', showscale=False, cmin=defl_min, cmax=defl_max, opacity=0.95),
            row=1, col=2
        )

        fig.update_layout(
            title=dict(text='<b>3D Phospholipid Deflection</b>', x=0.5, xanchor='center', font=dict(size=18)),
            width=1500, height=700,
            font=dict(family='Arial', size=12)
        )

        self._save_html(fig, 'deflection_3d_analysis.html')

        # Pore 3D
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

        self._save_html(fig, 'pore_3d_analysis.html')

    # ==================== MAIN EXECUTION ====================

    def run_analysis(self, protein_R_bind_angstrom=None, protein_dG_bind_kbt=None,
                      bilayer_kc_kbt=DEFAULT_BILAYER_KC_KBT, n_schematic_snapshots=4):
        """Execute complete analysis. All figures are written into
        self.output_dir (created in __init__).

        protein_R_bind_angstrom, protein_dG_bind_kbt: optional literature/
        computed values (Å, kBT) for the Pacsin1-paper tube areal-density
        estimate (Eqs 5-6). If either is None, that optional step is
        skipped, since these numbers require a PMF/umbrella-sampling
        calculation not reproduced by this geometry-only script.
        """
        # Single frame analysis
        self.extract_surface_data()
        self.create_surface_grid()
        self.calculate_curvature()
        self.calculate_lipid_deflection()

        print("\n" + "="*80)
        print("SPONTANEOUS CURVATURE FRUSTRATION (Marsh, Biophys J 2007;93:3884-3899)")
        print("="*80)
        self.calculate_curvature_frustration()

        self.analyze_pore_radius()
        self.analyze_water_permeation()

        print("\n" + "="*80)
        print("GENERATING AREA-PER-LIPID MAP & ACYL-CHAIN ORDER PARAMETERS")
        print("="*80)
        self.plot_area_per_lipid_map(leaflet='upper')
        self.plot_thickness_order_parameter_analysis()

        print("\n" + "="*80)
        print("PROTEIN/PEPTIDE-INDUCED PORE ANALYSIS")
        print("="*80)
        if len(self.protein) > 0:
            self.analyze_protein_induced_pore()
        else:
            print("⚠ No protein/peptide present -- skipping (nothing to attribute an "
                  "induced pore to).")

        # Time-series analysis
        self.collect_time_series_data()

        print("\n" + "="*80)
        print("GENERATING TOP & SIDE VIEW ANALYSIS")
        print("="*80)
        self.plot_top_side_view()
        self.plot_3d_interactive_topside()

        print("\n" + "="*80)
        print("GENERATING CROSS-SECTION PROFILES")
        print("="*80)
        self.plot_cross_section_profiles()

        print("\n" + "="*80)
        print("GENERATING CURVATURE FRUSTRATION ANALYSIS")
        print("="*80)
        self.plot_curvature_frustration()
        self.plot_frustration_energy_timeseries()

        print("\n" + "="*80)
        print("GENERATING TIME-SERIES ANALYSIS")
        print("="*80)
        self.plot_radius_of_curvature_timeseries()
        self.plot_thickness_over_time()
        self.plot_radius_of_curvature_over_time()
        self.plot_circlefit_curvature_analysis()

        print("\n" + "="*80)
        print("GENERATING EXTENDED DYNAMICS: UNDULATIONS, AREA/LIPID, PORE & WATER")
        print("="*80)
        self.plot_extended_dynamics_analysis()
        self.plot_water_permeation_depth_profile()
        if len(self.protein) > 0:
            self.plot_protein_induced_pore_analysis()

        print("\n" + "="*80)
        print("GENERATING PACSIN1-PAPER-STYLE CURVATURE SENSING ANALYSIS")
        print("(cf. Mahmood, Noguchi & Okazaki, Sci Rep 9, 14557, 2019)")
        print("="*80)
        self.plot_curvature_sensing_schematic()
        self.plot_curvature_sensing_schematic_series(n_snapshots=n_schematic_snapshots)
        if len(self.protein) > 0:
            self.plot_protein_intrinsic_curvature()
            self.plot_protein_curvature_sensing()
            if protein_R_bind_angstrom is not None and protein_dG_bind_kbt is not None:
                self.estimate_tube_areal_density(protein_R_bind_angstrom, protein_dG_bind_kbt,
                                                  kc_bilayer_kbt=bilayer_kc_kbt)
        else:
            print("⚠ No protein selected — skipping protein-specific curvature-sensing "
                  "analyses (intrinsic curvature, s_norm/θ sensing map, tube density).")

        print("\n" + "="*80)
        print("GENERATING STANDARD 2D ANALYSIS PLOTS")
        print("="*80)
        self.plot_2d_standard_all()

        print("\n" + "="*80)
        print("GENERATING STANDARD 3D ANALYSIS PLOTS")
        print("="*80)
        self.plot_3d_standard_all()

        print("\n" + "="*80)
        print("✓ COMPLETE ANALYSIS FINISHED!")
        print("="*80)

        print("\n📐 AVERAGE VALUES OVER THE ENTIRE SAMPLED TRAJECTORY:")
        if hasattr(self, 'avg_thickness'):
            print(f"  • Membrane thickness       : {self.avg_thickness:.3f} ± {self.std_thickness:.3f} Å")
        if hasattr(self, 'avg_radius_of_curvature'):
            print(f"  • Radius of curvature (diff. geometry): "
                  f"{self.avg_radius_of_curvature:.1f} ± {self.std_radius_of_curvature:.1f} Å")
        if hasattr(self, 'avg_radius_of_curvature_circlefit'):
            print(f"  • Radius of curvature (circle fit)    : "
                  f"{self.avg_radius_of_curvature_circlefit:.1f} ± "
                  f"{self.std_radius_of_curvature_circlefit:.1f} Å")
        if len(self.apl_upper_over_time) > 0:
            print(f"  • Area per lipid (upper leaflet)      : "
                  f"{np.nanmean(self.apl_upper_over_time):.2f} Å²")
            print(f"  • Area per lipid (lower leaflet)      : "
                  f"{np.nanmean(self.apl_lower_over_time):.2f} Å²")
        if hasattr(self, 'kc_undulation_kbt') and np.isfinite(self.kc_undulation_kbt):
            print(f"  • Bending modulus kc (undulation spectrum): "
                  f"{self.kc_undulation_kbt:.2f} ± {self.kc_undulation_err_kbt:.2f} kBT "
                  f"(vs. literature input {self.kc_kbt:.2f} kBT used in Marsh model)")
        if len(self.pore_radius_over_time) > 0:
            print(f"  • Max pore radius (mean over trajectory): "
                  f"{np.nanmean(self.pore_radius_over_time):.2f} Å")
        if len(self.water_influx_over_time) > 0:
            print(f"  • Water molecules in bilayer envelope (mean): "
                  f"{np.nanmean(self.water_influx_over_time):.0f}")
        if len(self.water_core_count_over_time) > 0:
            print(f"  • Water molecules in midplane/hydrophobic-core region (mean): "
                  f"{np.nanmean(self.water_core_count_over_time):.0f}")
        if len(self.protein) > 0 and len(self.protein_pore_radius_over_time) > 0:
            print(f"  • Protein-induced pore radius (mean over trajectory): "
                  f"{np.nanmean(self.protein_pore_radius_over_time):.2f} Å")

        print(f"\n📁 OUTPUT FOLDER: {os.path.abspath(self.output_dir)}")
        print("\n📊 GENERATED FILES SUMMARY:")
        print("\nTOP/SIDE & CROSS-SECTION:")
        print("  ✓ top_side_view.png")
        print("  ✓ interactive_top_side_view.html")
        print("  ✓ cross_section_profiles.png")
        print("\nCURVATURE-FRUSTRATION (Marsh 2007):")
        print("  ✓ curvature_frustration_analysis.png")
        print("  ✓ frustration_energy_timeseries.png")
        print("\nTIME-DEPENDENT CURVATURE:")
        print("  ✓ radius_curvature_timeseries.png")
        print("  ✓ membrane_thickness_over_time.png")
        print("  ✓ radius_of_curvature_over_time.png")
        print("  ✓ membrane_circlefit_curvature.png")
        print("  ✓ extended_dynamics_analysis.png (undulation spectrum kc, area/lipid, "
              "pore radius, water permeation)")
        print("  ✓ water_permeation_depth_profile.png (midplane = 0, upper/lower leaflets marked)")
        if len(self.protein) > 0:
            print("  ✓ protein_induced_pore_analysis.png")
        print("\nAREA-PER-LIPID MAP & ORDER PARAMETERS:")
        print("  ✓ area_per_lipid_map_upper.png (Voronoi-based local APL, nm², peptide overlaid)")
        print("  ✓ thickness_order_parameter_analysis.png (A: thickness, B: order-parameter map, "
              "C: Rim/Core/Bilayer vs. carbon)")
        print("\nPACSIN1-PAPER-STYLE ANALYSES (Mahmood, Noguchi & Okazaki, Sci Rep 2019):")
        print("  ✓ curvature_sensing_schematic.png (+ _t01..t%02d.png time series)" % n_schematic_snapshots)
        if len(self.protein) > 0:
            print("  ✓ protein_intrinsic_curvature.png")
            print("  ✓ protein_curvature_sensing_analysis.png")
            if protein_R_bind_angstrom is not None and protein_dG_bind_kbt is not None:
                print("  ✓ tube_areal_density_estimate.png")
        print("\nSTANDARD 2D ANALYSIS (PNG - 300 DPI):")
        print("  ✓ thickness_2d_analysis.png")
        print("  ✓ curvature_2d_analysis.png")
        print("  ✓ water_2d_analysis.png")
        print("\nSTANDARD 3D ANALYSIS (Interactive HTML):")
        print("  ✓ thickness_3d_analysis.html")
        print("  ✓ curvature_3d_analysis.html")
        print("  ✓ deflection_3d_analysis.html")
        print("  ✓ pore_3d_analysis.html")
        print("\n" + "="*80 + "\n")

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

    # Optional: Pacsin1 tube-formation areal-density estimate (Mahmood,
    # Noguchi & Okazaki 2019, Eqs 5-6). These example numbers are the
    # PAPER's own reported values (R_bind ~ 25 nm, dG_bind ~ -2 kBT at
    # kc(bilayer) ~ 20 kBT) and are only illustrative -- replace with
    # values appropriate to your system (e.g. from your own PMF/umbrella
    # sampling along the curvature-sensing coordinate), or leave as None
    # to skip this optional step.
    PROTEIN_R_BIND_ANGSTROM = None    # e.g. 250.0  (25 nm, in Å)
    PROTEIN_DG_BIND_KBT = None        # e.g. -2.0

    analyzer = ProfessionalMembraneSurfaceAnalyzer(
        TPR, XTC, LIPIDS, grid_points=250, smoothing_sigma=2,
        kc_kbt=9.9,                 # monolayer bending modulus (kBT), Marsh 2007
        c0_overrides=C0_OVERRIDES,  # per-lipid spontaneous curvature (Å^-1)
        temperature=310.0,          # K, for kBT <-> kJ/mol conversion
        output_dir=None,            # auto-timestamped output folder; or set e.g. "my_run_output"
    )
    analyzer.run_analysis(
        protein_R_bind_angstrom=PROTEIN_R_BIND_ANGSTROM,
        protein_dG_bind_kbt=PROTEIN_DG_BIND_KBT,
        bilayer_kc_kbt=20.0,
        n_schematic_snapshots=4,
    )
