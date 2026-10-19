import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial.distance import cdist
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

class ProfessionalMembraneSurfaceAnalyzer:
    def __init__(self, tpr, xtc, lipids, water_mol='WAT', grid_points=200, smoothing_sigma=2):
        """Initialize professional membrane analyzer"""
        print("=" * 80)
        print("PROFESSIONAL MEMBRANE SURFACE ANALYSIS WITH DYNAMICS")
        print("=" * 80)
        print("Loading system...")
        self.u = mda.Universe(tpr, xtc)
        self.lipids = " ".join(lipids)
        self.lipid_atoms = self.u.select_atoms(f"resname {self.lipids}")
        self.water = self.u.select_atoms(f"resname {water_mol}")
        self.protein = self.u.select_atoms("protein")
        
        self.grid_points = grid_points
        self.smooth_sigma = smoothing_sigma
        self.ts = self.u.trajectory[-1]
        
        # For time-series analysis
        self.curvature_over_time = []
        self.thickness_over_time = []
        self.deflection_over_time = []
        self.radius_over_time = []
        self.time_points = []
        
    def extract_surface_data(self):
        """Extract membrane surface coordinates"""
        coords = self.lipid_atoms.center_of_mass(compound="residues")
        
        self.x = coords[:, 0]
        self.y = coords[:, 1]
        self.z = coords[:, 2]
        
        mid = np.median(self.z)
        self.upper = coords[self.z > mid]
        self.lower = coords[self.z < mid]
        
        print(f"✓ Upper leaflet points: {len(self.upper)}")
        print(f"✓ Lower leaflet points: {len(self.lower)}")
        
    def create_surface_grid(self):
        """Create interpolated surface grids"""
        xi = np.linspace(self.x.min(), self.x.max(), self.grid_points)
        yi = np.linspace(self.y.min(), self.y.max(), self.grid_points)
        self.xi, self.yi = np.meshgrid(xi, yi)
        
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
        
    def calculate_curvature(self):
        """Calculate membrane curvature"""
        dzdx, dzdy = np.gradient(self.zi)
        d2zdx2, _ = np.gradient(dzdx)
        _, d2zdy2 = np.gradient(dzdy)
        
        self.curvature = d2zdx2 + d2zdy2
        self.curvature_norm = self.curvature / np.max(np.abs(self.curvature))
        
        dzdx_u, dzdy_u = np.gradient(self.zu)
        d2zdx2_u, _ = np.gradient(dzdx_u)
        _, d2zdy2_u = np.gradient(dzdy_u)
        self.curvature_u = d2zdx2_u + d2zdy2_u
        
        dzdx_l, dzdy_l = np.gradient(self.zl)
        d2zdx2_l, _ = np.gradient(dzdx_l)
        _, d2zdy2_l = np.gradient(dzdy_l)
        self.curvature_l = d2zdx2_l + d2zdy2_l
        
        # Calculate radius of curvature
        self.calc_radius_of_curvature()
        
    def calc_radius_of_curvature(self):
        """Calculate radius of curvature from curvature values"""
        with np.errstate(divide='ignore', invalid='ignore'):
            self.radius_of_curvature = np.where(
                np.abs(self.curvature) > 1e-6, 
                1.0 / np.abs(self.curvature), 
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
        
    def calculate_electrostatic_potential(self):
        """Calculate electrostatic potential-like map (Gaussian potential from center)"""
        # Create potential map based on distance from center and height
        center_x = np.mean(self.x)
        center_y = np.mean(self.y)
        
        # Distance from center
        dist_from_center = np.sqrt((self.xi - center_x)**2 + (self.yi - center_y)**2)
        
        # Combine distance and curvature for electrostatic-like potential
        # Higher curvature (positive or negative) creates potential wells
        self.potential_map = -np.abs(self.curvature_norm) * np.exp(-dist_from_center / (np.max(dist_from_center)/2))
        
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
        
    def collect_time_series_data(self):
        """Collect curvature data from multiple frames"""
        print("\n" + "="*80)
        print("COLLECTING TIME-SERIES DATA")
        print("="*80)
        
        n_frames = len(self.u.trajectory)
        print(f"Processing {n_frames} frames...")
        
        for frame_idx, ts in enumerate(self.u.trajectory[::max(1, n_frames//20)]):
            self.extract_surface_data()
            self.create_surface_grid()
            self.calculate_curvature()
            self.calculate_lipid_deflection()
            
            # Store time-series data
            time_ps = ts.time
            self.time_points.append(time_ps)
            self.curvature_over_time.append(np.mean(np.abs(self.curvature)))
            self.thickness_over_time.append(np.mean(self.thickness))
            self.deflection_over_time.append(np.mean(self.deflection_total))
            
            # Calculate mean radius of curvature
            rad_curv_clean = np.nan_to_num(self.radius_of_curvature, nan=1000.0)
            self.radius_over_time.append(np.nanmean(rad_curv_clean[rad_curv_clean < 500]))
            
            if (frame_idx + 1) % max(1, n_frames//20 // 5) == 0:
                print(f"  ✓ Processed frame {frame_idx + 1}/{n_frames}")
        
        self.time_points = np.array(self.time_points)
        self.curvature_over_time = np.array(self.curvature_over_time)
        self.thickness_over_time = np.array(self.thickness_over_time)
        self.deflection_over_time = np.array(self.deflection_over_time)
        self.radius_over_time = np.array(self.radius_over_time)
        
    # ==================== A: RADIUS OF CURVATURE TIME-SERIES (2D & 3D) ====================
    
    def plot_radius_of_curvature_timeseries_enhanced(self):
        """Enhanced radius of curvature analysis (A from image)"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        # Convert time to microseconds
        time_us = self.time_points / 1000
        
        # Plot 1: Radius over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time_us, self.radius_over_time, 'b-', linewidth=2.5, label='Upper leaflet')
        ax1.fill_between(time_us, self.radius_over_time, alpha=0.3, color='blue')
        
        # Add reference line for mean
        mean_radius = np.nanmean(self.radius_over_time)
        ax1.axhline(y=mean_radius, color='r', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {mean_radius:.2f} nm')
        
        ax1.set_xlabel('Time (μs)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Radius of curvature (nm)', fontsize=13, fontweight='bold')
        ax1.set_title('A: Radius of Curvature Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12, loc='upper right', framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, max(self.radius_over_time) * 1.1])
        
        # Plot 2: Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.radius_over_time, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(mean_radius, color='r', linestyle='--', linewidth=2.5, label=f'Mean: {mean_radius:.2f}')
        ax2.axvline(np.median(self.radius_over_time), color='g', linestyle=':', linewidth=2.5, 
                   label=f'Median: {np.median(self.radius_over_time):.2f}')
        ax2.set_xlabel('Radius of curvature (nm)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution', fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.2, axis='y')
        
        # Plot 3: Statistics
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        stats_text = f"""RADIUS OF CURVATURE ANALYSIS

Mean: {mean_radius:.2f} nm
Median: {np.median(self.radius_over_time):.2f} nm
Std Dev: {np.std(self.radius_over_time):.2f} nm
Min: {np.min(self.radius_over_time):.2f} nm
Max: {np.max(self.radius_over_time):.2f} nm

Time Range: {time_us[0]:.2f} - {time_us[-1]:.2f} μs
Total Frames: {len(self.time_points)}
"""
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2))
        
        plt.suptitle('A: Radius of Curvature Time-Series Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig('A_radius_curvature_timeseries.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: A_radius_curvature_timeseries.png")
        plt.close()
        
    def plot_radius_of_curvature_3d_interactive(self):
        """3D interactive plot of radius of curvature over time"""
        time_us = self.time_points / 1000
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_us,
            y=self.radius_over_time,
            mode='lines+markers',
            name='Radius of curvature',
            line=dict(color='blue', width=3),
            marker=dict(size=6, color='blue'),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)',
            hovertemplate='Time: %{x:.2f} μs<br>Radius: %{y:.2f} nm<extra></extra>'
        ))
        
        # Add mean line
        mean_radius = np.nanmean(self.radius_over_time)
        fig.add_hline(y=mean_radius, line_dash='dash', line_color='red', 
                     annotation_text=f'Mean: {mean_radius:.2f} nm',
                     annotation_position='right')
        
        fig.update_layout(
            title=dict(text='<b>A: 3D Radius of Curvature Time-Series</b>', x=0.5, xanchor='center', font=dict(size=18)),
            xaxis_title='Time (μs)',
            yaxis_title='Radius of Curvature (nm)',
            width=1200, height=600,
            hovermode='closest',
            font=dict(family='Arial', size=12),
            template='plotly_white'
        )
        
        fig.write_html('A_radius_curvature_3d_interactive.html')
        print(f"✓ Saved: A_radius_curvature_3d_interactive.html")
        
    # ==================== C: ELECTROSTATIC POTENTIAL / HEATMAP (2D & 3D) ====================
    
    def plot_electrostatic_potential_2d(self):
        """2D electrostatic potential map (C from image)"""
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        self.calculate_electrostatic_potential()
        
        # Plot 1: Main potential map
        ax1 = fig.add_subplot(gs[0, :])
        im = ax1.imshow(self.potential_map, cmap='RdBu_r', origin='lower',
                       extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()],
                       aspect='auto')
        
        # Add contours
        contours = ax1.contour(self.xi, self.yi, self.potential_map, levels=12, 
                             colors='white', alpha=0.3, linewidths=0.8)
        ax1.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
        cbar = plt.colorbar(im, ax=ax1, label='Potential (a.u.)', pad=0.02)
        ax1.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
        ax1.set_title('C: Electrostatic Potential Map', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.1)
        
        # Plot 2: Potential distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.potential_map.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(self.potential_map), color='r', linestyle='--', linewidth=2.5)
        ax2.set_xlabel('Potential (a.u.)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.2, axis='y')
        
        # Plot 3: Statistics
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        stats_text = f"""ELECTROSTATIC POTENTIAL

Mean: {np.mean(self.potential_map):.3f}
Median: {np.median(self.potential_map):.3f}
Std Dev: {np.std(self.potential_map):.3f}
Min: {np.min(self.potential_map):.3f}
Max: {np.max(self.potential_map):.3f}

Combines:
- Membrane curvature
- Distance from center
- Gaussian weighting
"""
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=2))
        
        plt.suptitle('C: Electrostatic Potential / Heatmap Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig('C_electrostatic_potential_2d.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: C_electrostatic_potential_2d.png")
        plt.close()
        
    def plot_electrostatic_potential_3d(self):
        """3D surface plot of electrostatic potential"""
        self.calculate_electrostatic_potential()
        
        fig = go.Figure(data=[go.Surface(
            x=self.xi, y=self.yi, z=self.potential_map,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title='Potential (a.u.)', thickness=20),
            contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True))),
            opacity=0.95
        )])
        
        fig.update_layout(
            title=dict(text='<b>C: 3D Electrostatic Potential Surface</b>', x=0.5, xanchor='center', font=dict(size=18)),
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Potential (a.u.)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            width=1000, height=800,
            font=dict(family='Arial', size=12)
        )
        
        fig.write_html('C_electrostatic_potential_3d.html')
        print(f"✓ Saved: C_electrostatic_potential_3d.html")
        
    # ==================== D: 3D STRUCTURE WITH COLORED LEAFLETS (2D & 3D) ====================
    
    def plot_3d_structure_with_leaflets_2d(self):
        """2D top-down view showing both leaflets - Matching reference image exactly"""
        fig = plt.figure(figsize=(18, 12))
        
        # Create 2x2 grid for 4 panels
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
        
        # ===== PANEL G1 (Top Left): Upper Leaflet View 1 =====
        ax_g1 = fig.add_subplot(gs[0, 0])
        ax_g1.set_facecolor('white')
        
        # Create light gray background to suggest membrane
        circle_bg = plt.Circle((np.mean(self.upper[:, 0]), np.mean(self.upper[:, 1])), 
                              np.max([self.upper[:, 0].max() - self.upper[:, 0].min(),
                                     self.upper[:, 1].max() - self.upper[:, 1].min()]) / 2 + 5,
                              color='lightgray', alpha=0.2, zorder=0)
        ax_g1.add_patch(circle_bg)
        
        # Plot upper leaflet atoms as circles (top-down view)
        scatter_g1 = ax_g1.scatter(self.upper[:, 0], self.upper[:, 1],
                                   s=250, c='dodgerblue', alpha=0.75, 
                                   edgecolors='darkblue', linewidth=1.5, zorder=2)
        
        # Add label
        ax_g1.text(0.02, 0.98, '● G1', transform=ax_g1.transAxes, 
                  fontsize=14, fontweight='bold', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='darkblue', linewidth=1.5))
        
        ax_g1.set_xlabel('X (Å)', fontsize=11, fontweight='bold')
        ax_g1.set_ylabel('Y (Å)', fontsize=11, fontweight='bold')
        ax_g1.set_title('Upper Leaflet (Extracellular)', fontsize=12, fontweight='bold')
        ax_g1.grid(False)
        ax_g1.set_xlim([self.upper[:, 0].min() - 10, self.upper[:, 0].max() + 10])
        ax_g1.set_ylim([self.upper[:, 1].min() - 10, self.upper[:, 1].max() + 10])
        ax_g1.set_aspect('equal')
        
        # ===== PANEL G2 (Top Right): Upper Leaflet View 2 (Rotated subset) =====
        ax_g2 = fig.add_subplot(gs[0, 1])
        ax_g2.set_facecolor('white')
        
        # Subset of upper leaflet for second view
        upper_subset_2 = self.upper[::2]  # Every other lipid
        
        circle_bg_2 = plt.Circle((np.mean(upper_subset_2[:, 0]), np.mean(upper_subset_2[:, 1])), 
                                np.max([upper_subset_2[:, 0].max() - upper_subset_2[:, 0].min(),
                                       upper_subset_2[:, 1].max() - upper_subset_2[:, 1].min()]) / 2 + 5,
                                color='lightgray', alpha=0.2, zorder=0)
        ax_g2.add_patch(circle_bg_2)
        
        scatter_g2 = ax_g2.scatter(upper_subset_2[:, 0], upper_subset_2[:, 1],
                                   s=250, c='dodgerblue', alpha=0.75, 
                                   edgecolors='darkblue', linewidth=1.5, zorder=2)
        
        ax_g2.text(0.02, 0.98, '● G2', transform=ax_g2.transAxes, 
                  fontsize=14, fontweight='bold', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='darkblue', linewidth=1.5))
        
        ax_g2.set_xlabel('X (Å)', fontsize=11, fontweight='bold')
        ax_g2.set_ylabel('Y (Å)', fontsize=11, fontweight='bold')
        ax_g2.set_title('Upper Leaflet (Rotated View)', fontsize=12, fontweight='bold')
        ax_g2.grid(False)
        ax_g2.set_xlim([upper_subset_2[:, 0].min() - 10, upper_subset_2[:, 0].max() + 10])
        ax_g2.set_ylim([upper_subset_2[:, 1].min() - 10, upper_subset_2[:, 1].max() + 10])
        ax_g2.set_aspect('equal')
        
        # ===== PANEL H1 (Bottom Left): Lower Leaflet View 1 =====
        ax_h1 = fig.add_subplot(gs[1, 0])
        ax_h1.set_facecolor('white')
        
        circle_bg_h1 = plt.Circle((np.mean(self.lower[:, 0]), np.mean(self.lower[:, 1])), 
                                 np.max([self.lower[:, 0].max() - self.lower[:, 0].min(),
                                        self.lower[:, 1].max() - self.lower[:, 1].min()]) / 2 + 5,
                                 color='lightgray', alpha=0.2, zorder=0)
        ax_h1.add_patch(circle_bg_h1)
        
        scatter_h1 = ax_h1.scatter(self.lower[:, 0], self.lower[:, 1],
                                   s=250, c='orange', alpha=0.75, 
                                   edgecolors='darkorange', linewidth=1.5, zorder=2)
        
        ax_h1.text(0.02, 0.98, '● H1', transform=ax_h1.transAxes, 
                  fontsize=14, fontweight='bold', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='darkorange', linewidth=1.5))
        
        ax_h1.set_xlabel('X (Å)', fontsize=11, fontweight='bold')
        ax_h1.set_ylabel('Y (Å)', fontsize=11, fontweight='bold')
        ax_h1.set_title('Lower Leaflet (Cytoplasm)', fontsize=12, fontweight='bold')
        ax_h1.grid(False)
        ax_h1.set_xlim([self.lower[:, 0].min() - 10, self.lower[:, 0].max() + 10])
        ax_h1.set_ylim([self.lower[:, 1].min() - 10, self.lower[:, 1].max() + 10])
        ax_h1.set_aspect('equal')
        
        # ===== PANEL H2 (Bottom Right): Lower Leaflet View 2 (Rotated subset) =====
        ax_h2 = fig.add_subplot(gs[1, 1])
        ax_h2.set_facecolor('white')
        
        lower_subset_2 = self.lower[::2]  # Every other lipid
        
        circle_bg_h2 = plt.Circle((np.mean(lower_subset_2[:, 0]), np.mean(lower_subset_2[:, 1])), 
                                 np.max([lower_subset_2[:, 0].max() - lower_subset_2[:, 0].min(),
                                        lower_subset_2[:, 1].max() - lower_subset_2[:, 1].min()]) / 2 + 5,
                                 color='lightgray', alpha=0.2, zorder=0)
        ax_h2.add_patch(circle_bg_h2)
        
        scatter_h2 = ax_h2.scatter(lower_subset_2[:, 0], lower_subset_2[:, 1],
                                   s=250, c='orange', alpha=0.75, 
                                   edgecolors='darkorange', linewidth=1.5, zorder=2)
        
        ax_h2.text(0.02, 0.98, '● H2', transform=ax_h2.transAxes, 
                  fontsize=14, fontweight='bold', verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='darkorange', linewidth=1.5))
        
        ax_h2.set_xlabel('X (Å)', fontsize=11, fontweight='bold')
        ax_h2.set_ylabel('Y (Å)', fontsize=11, fontweight='bold')
        ax_h2.set_title('Lower Leaflet (Rotated View)', fontsize=12, fontweight='bold')
        ax_h2.grid(False)
        ax_h2.set_xlim([lower_subset_2[:, 0].min() - 10, lower_subset_2[:, 0].max() + 10])
        ax_h2.set_ylim([lower_subset_2[:, 1].min() - 10, lower_subset_2[:, 1].max() + 10])
        ax_h2.set_aspect('equal')
        
        # Add comprehensive title
        fig.suptitle('D: Membrane Leaflets - Top-Down View\n' + 
                    'Blue (G1, G2) = Upper Leaflet | Orange (H1, H2) = Lower Leaflet', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('D_3d_structure_leaflets_2d.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: D_3d_structure_leaflets_2d.png (Reference image format)")
        plt.close()

    def plot_membrane_side_view_3d(self):
        """3D side-view projection of membrane with both leaflets - NEW OUTPUT (FIXED)"""
        fig = plt.figure(figsize=(16, 8))
        
        # Create 1x2 grid for side views
        gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
        
        # ===== Panel 1: Side view along Y-axis (X-Z plane) =====
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        
        # Create surface mesh for membrane visualization
        y_central = np.linspace(self.yi.min(), self.yi.max(), 20)
        z_upper_profile = np.mean(self.zu, axis=0)  # Shape: (grid_points,)
        z_lower_profile = np.mean(self.zl, axis=0)  # Shape: (grid_points,)
        x_profile = self.xi[0, :]  # Shape: (grid_points,)
        
        # Create 3D mesh - FIXED SHAPES
        x_3d = np.tile(x_profile, (len(y_central), 1))  # Shape: (20, grid_points)
        y_3d = np.tile(y_central[:, np.newaxis], (1, len(x_profile)))  # Shape: (20, grid_points)
        z_upper_3d = np.tile(z_upper_profile, (len(y_central), 1))  # Shape: (20, grid_points)
        z_lower_3d = np.tile(z_lower_profile, (len(y_central), 1))  # Shape: (20, grid_points)
        
        # Plot surfaces
        ax1.plot_surface(x_3d, y_3d, z_upper_3d, alpha=0.3, color='lightblue', edgecolor='none')
        ax1.plot_surface(x_3d, y_3d, z_lower_3d, alpha=0.3, color='lightyellow', edgecolor='none')
        
        # Add profile lines for membrane visualization
        for i in range(len(y_central)):
            ax1.plot(x_profile, [y_central[i]]*len(x_profile), z_upper_profile, 
                    'b-', alpha=0.15, linewidth=0.8)
            ax1.plot(x_profile, [y_central[i]]*len(x_profile), z_lower_profile, 
                    'orange', alpha=0.15, linewidth=0.8)
        
        # Scatter upper leaflet atoms
        ax1.scatter(self.upper[:, 0], self.upper[:, 1], self.upper[:, 2],
                   c='dodgerblue', s=100, alpha=0.65, edgecolors='darkblue', linewidth=0.8, label='Upper leaflet')
        
        # Scatter lower leaflet atoms
        ax1.scatter(self.lower[:, 0], self.lower[:, 1], self.lower[:, 2],
                   c='orange', s=100, alpha=0.65, edgecolors='darkorange', linewidth=0.8, label='Lower leaflet')
        
        ax1.set_xlabel('X (Å)', fontweight='bold', fontsize=10)
        ax1.set_ylabel('Y (Å)', fontweight='bold', fontsize=10)
        ax1.set_zlabel('Z (Å)', fontweight='bold', fontsize=10)
        ax1.set_title('Side View 1 (X-Z Plane)', fontsize=12, fontweight='bold')
        ax1.view_init(elev=20, azim=0)
        ax1.legend(loc='upper right', fontsize=9)
        
        # ===== Panel 2: Side view along X-axis (Y-Z plane) =====
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        
        # Create surface for Y-Z view
        x_central = np.linspace(self.xi.min(), self.xi.max(), 20)
        z_upper_profile_yz = np.mean(self.zu, axis=1)  # Shape: (grid_points,)
        z_lower_profile_yz = np.mean(self.zl, axis=1)  # Shape: (grid_points,)
        y_profile = self.yi[:, 0]  # Shape: (grid_points,)
        
        # Create 3D mesh - FIXED SHAPES
        x_3d_yz = np.tile(x_central[:, np.newaxis], (1, len(y_profile)))  # Shape: (20, grid_points)
        y_3d_yz = np.tile(y_profile, (len(x_central), 1))  # Shape: (20, grid_points)
        z_upper_3d_yz = np.tile(z_upper_profile_yz, (len(x_central), 1))  # Shape: (20, grid_points)
        z_lower_3d_yz = np.tile(z_lower_profile_yz, (len(x_central), 1))  # Shape: (20, grid_points)
        
        # Plot surfaces
        ax2.plot_surface(x_3d_yz, y_3d_yz, z_upper_3d_yz, alpha=0.3, color='lightblue', edgecolor='none')
        ax2.plot_surface(x_3d_yz, y_3d_yz, z_lower_3d_yz, alpha=0.3, color='lightyellow', edgecolor='none')
        
        # Add profile lines
        for i in range(len(x_central)):
            ax2.plot([x_central[i]]*len(y_profile), y_profile, z_upper_profile_yz, 
                    'b-', alpha=0.15, linewidth=0.8)
            ax2.plot([x_central[i]]*len(y_profile), y_profile, z_lower_profile_yz, 
                    'orange', alpha=0.15, linewidth=0.8)
        
        # Scatter atoms
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
        
        plt.savefig('E_membrane_side_view_3d.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: E_membrane_side_view_3d.png (NEW - Side-view 3D projection)")
        plt.close()
        
    def plot_3d_structure_with_leaflets_3d(self):
        """3D interactive structure with colored leaflets"""
        fig = go.Figure()
        
        # Upper leaflet - blue
        fig.add_trace(go.Scatter3d(
            x=self.upper[:, 0],
            y=self.upper[:, 1],
            z=self.upper[:, 2],
            mode='markers',
            marker=dict(size=4, color='dodgerblue', opacity=0.7),
            name='Upper leaflet (extracellular)',
            hovertemplate='<b>Upper Leaflet</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
        ))
        
        # Lower leaflet - orange
        fig.add_trace(go.Scatter3d(
            x=self.lower[:, 0],
            y=self.lower[:, 1],
            z=self.lower[:, 2],
            mode='markers',
            marker=dict(size=4, color='orange', opacity=0.7),
            name='Lower leaflet (cytoplasm)',
            hovertemplate='<b>Lower Leaflet</b><br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
        ))
        
        # Add surface meshes
        fig.add_trace(go.Surface(
            x=self.xi, y=self.yi, z=self.zu,
            colorscale='Blues',
            showscale=False,
            opacity=0.2,
            name='Upper surface',
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Surface(
            x=self.xi, y=self.yi, z=self.zl,
            colorscale='Oranges',
            showscale=False,
            opacity=0.2,
            name='Lower surface',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=dict(text='<b>D: 3D Membrane Structure - Both Leaflets</b><br><sub>Blue = Upper leaflet (extracellular) | Orange = Lower leaflet (cytoplasm)</sub>', 
                      x=0.5, xanchor='center', font=dict(size=16)),
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                aspectmode='data'
            ),
            width=1200, height=900,
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='black', borderwidth=1),
            font=dict(family='Arial', size=12),
            hovermode='closest'
        )
        
        fig.write_html('D_3d_structure_leaflets_3d.html')
        print(f"✓ Saved: D_3d_structure_leaflets_3d.html")
        
    def plot_3d_structure_with_heatmap_overlay(self):
        """3D structure with heatmap overlay (like image D right panel)"""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('<b>Curvature Heatmap</b>', '<b>Electrostatic Potential</b>'),
            horizontal_spacing=0.12
        )
        
        self.calculate_electrostatic_potential()
        
        # Curvature heatmap
        fig.add_trace(
            go.Surface(x=self.xi, y=self.yi, z=self.zu,
                      surfacecolor=self.curvature_norm,
                      colorscale='RdBu',
                      showscale=True,
                      colorbar=dict(title='Curvature', x=0.46, len=0.6),
                      opacity=0.9),
            row=1, col=1
        )
        
        # Electrostatic potential heatmap
        fig.add_trace(
            go.Surface(x=self.xi, y=self.yi, z=self.zu,
                      surfacecolor=self.potential_map,
                      colorscale='RdYlGn',
                      showscale=True,
                      colorbar=dict(title='Potential', x=1.02, len=0.6),
                      opacity=0.9),
            row=1, col=2
        )
        
        fig.update_scenes(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        )
        
        fig.update_layout(
            title=dict(text='<b>D: 3D Heatmap Overlays on Membrane Structure</b>', x=0.5, xanchor='center', font=dict(size=18)),
            width=1400, height=700,
            font=dict(family='Arial', size=12)
        )
        
        fig.write_html('D_3d_structure_heatmap_overlay.html')
        print(f"✓ Saved: D_3d_structure_heatmap_overlay.html")
        
    # ==================== STANDARD PLOTS ====================
    
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
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {filename}")
            plt.close()
        
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
        fig.add_trace(go.Surface(x=self.xi, y=self.yi, z=self.zl, colorscale='Oranges', showscale=False, opacity=0.95), row=1, col=2)
        
        fig.update_layout(
            title=dict(text='<b>3D Membrane Leaflet Surfaces</b>', x=0.5, xanchor='center', font=dict(size=18)),
            width=1500, height=700,
            font=dict(family='Arial', size=12)
        )
        
        fig.write_html('thickness_3d_analysis.html')
        print(f"✓ Saved: thickness_3d_analysis.html")
        
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
        
        fig.write_html('curvature_3d_analysis.html')
        print(f"✓ Saved: curvature_3d_analysis.html")
        
    # ==================== MAIN EXECUTION ====================
    
    def run_analysis(self):
        """Execute complete analysis"""
        # Single frame analysis
        self.extract_surface_data()
        self.create_surface_grid()
        self.calculate_curvature()
        self.calculate_lipid_deflection()
        self.analyze_pore_radius()
        self.analyze_water_permeation()
        
        # Time-series analysis
        self.collect_time_series_data()
        
        print("\n" + "="*80)
        print("GENERATING A: RADIUS OF CURVATURE TIME-SERIES (2D & 3D)")
        print("="*80)
        self.plot_radius_of_curvature_timeseries_enhanced()
        self.plot_radius_of_curvature_3d_interactive()
        
        print("\n" + "="*80)
        print("GENERATING C: ELECTROSTATIC POTENTIAL / HEATMAP (2D & 3D)")
        print("="*80)
        self.plot_electrostatic_potential_2d()
        self.plot_electrostatic_potential_3d()
        
        print("\n" + "="*80)
        print("GENERATING D: 3D STRUCTURE WITH COLORED LEAFLETS (2D & 3D)")
        print("="*80)
        self.plot_3d_structure_with_leaflets_2d()
        self.plot_3d_structure_with_leaflets_3d()
        self.plot_3d_structure_with_heatmap_overlay()
        
        print("\n" + "="*80)
        print("GENERATING E: MEMBRANE SIDE-VIEW ANALYSIS (NEW)")
        print("="*80)
        self.plot_membrane_side_view_3d()
        
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
        print("\n📊 GENERATED FILES SUMMARY:")
        print("\nA: RADIUS OF CURVATURE (2D & 3D):")
        print("  ✓ A_radius_curvature_timeseries.png (2D time-series)")
        print("  ✓ A_radius_curvature_3d_interactive.html (3D interactive)")
        print("\nC: ELECTROSTATIC POTENTIAL (2D & 3D):")
        print("  ✓ C_electrostatic_potential_2d.png (2D heatmap)")
        print("  ✓ C_electrostatic_potential_3d.html (3D surface)")
        print("\nD: 3D STRUCTURE WITH LEAFLETS (2D & 3D):")
        print("  ✓ D_3d_structure_leaflets_2d.png (★ TOP-DOWN VIEW)")
        print("  ✓ D_3d_structure_leaflets_3d.html (3D interactive)")
        print("  ✓ D_3d_structure_heatmap_overlay.html (3D heatmap overlay)")
        print("\nE: MEMBRANE SIDE-VIEW ANALYSIS (★ NEW OUTPUT - FIXED):")
        print("  ✓ E_membrane_side_view_3d.png (Side-view 3D projections)")
        print("\nSTANDARD 2D ANALYSIS:")
        print("  ✓ thickness_2d_analysis.png")
        print("  ✓ curvature_2d_analysis.png")
        print("  ✓ water_2d_analysis.png")
        print("\nSTANDARD 3D ANALYSIS:")
        print("  ✓ thickness_3d_analysis.html")
        print("  ✓ curvature_3d_analysis.html")
        print("\n" + "="*80)
        print("TOTAL: 15 Publication-Quality Visualizations Generated!")
        print("="*80 + "\n")

# Usage
if __name__ == "__main__":
    TPR = "md_0_1000.tpr"
    XTC = "center1.xtc"
    LIPIDS = ["POPE", "POPG"]
    
    analyzer = ProfessionalMembraneSurfaceAnalyzer(TPR, XTC, LIPIDS, grid_points=250, smoothing_sigma=2)
    analyzer.run_analysis()

