import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.optimize import curve_fit
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
        # Radius of curvature = 1 / |curvature|
        # Avoid division by zero
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
        
        for frame_idx, ts in enumerate(self.u.trajectory[::max(1, n_frames//20)]):  # Sample ~20 frames
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
            
            if (frame_idx + 1) % max(1, n_frames//20 // 5) == 0:
                print(f"  ✓ Processed frame {frame_idx + 1}/{n_frames}")
        
        self.time_points = np.array(self.time_points)
        self.curvature_over_time = np.array(self.curvature_over_time)
        self.thickness_over_time = np.array(self.thickness_over_time)
        self.deflection_over_time = np.array(self.deflection_over_time)
        
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
        
        ax_top.set_xlabel('X (nm)', fontsize=13, fontweight='bold')
        ax_top.set_ylabel('Y (nm)', fontsize=13, fontweight='bold')
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
        
        # Add radius of curvature text
        rad_curv = np.nanmean(self.radius_of_curvature[~np.isinf(self.radius_of_curvature)])
        ax_side.text(0.5, 0.95, f'Radius of curvature = {rad_curv:.1f} Å', 
                    transform=ax_side.transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    verticalalignment='top', horizontalalignment='center')
        
        ax_side.set_xlabel('X (nm)', fontsize=13, fontweight='bold')
        ax_side.set_ylabel('Z (nm)', fontsize=13, fontweight='bold')
        ax_side.set_title('Side View', fontsize=15, fontweight='bold', pad=15)
        ax_side.legend(fontsize=11, loc='upper right')
        ax_side.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig('top_side_view.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: top_side_view.png")
        plt.close()
        
    # ==================== CROSS-SECTION PROFILES ====================
    
    def plot_cross_section_profiles(self):
        """Plot cross-section profiles (Z vs X)"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        # Get middle cross-section
        y_idx = self.grid_points // 2
        x_profile = self.xi[y_idx, :] / 10  # Convert to nm
        zu_profile = self.zu[y_idx, :] / 10
        zl_profile = self.zl[y_idx, :] / 10
        thickness_profile = self.thickness[y_idx, :] / 10
        curvature_profile = self.curvature[y_idx, :]
        
        # Plot 1: Z profile
        ax1 = fig.add_subplot(gs[0, :])
        ax1.fill_between(x_profile, zu_profile, zu_profile + 1, color='blue', alpha=0.3, label='Upper leaflet')
        ax1.plot(x_profile, zu_profile, 'b-', linewidth=2.5, label='Upper surface')
        ax1.fill_between(x_profile, zl_profile - 1, zl_profile, color='red', alpha=0.3, label='Lower leaflet')
        ax1.plot(x_profile, zl_profile, 'r-', linewidth=2.5, label='Lower surface')
        ax1.set_xlabel('X (nm)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Z (nm)', fontsize=12, fontweight='bold')
        ax1.set_title('Membrane Z-Profile (Cross-Section)', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper right', framealpha=0.95)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Thickness profile
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(x_profile, thickness_profile, 'g-', linewidth=2.5)
        ax2.fill_between(x_profile, thickness_profile, alpha=0.3, color='green')
        ax2.set_xlabel('X (nm)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Thickness (nm)', fontsize=12, fontweight='bold')
        ax2.set_title('Membrane Thickness Profile', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Curvature profile
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(x_profile, curvature_profile, 'purple', linewidth=2.5)
        ax3.fill_between(x_profile, curvature_profile, alpha=0.3, color='purple')
        ax3.set_xlabel('X (nm)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Curvature', fontsize=12, fontweight='bold')
        ax3.set_title('Membrane Curvature Profile', fontsize=13, fontweight='bold')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Cross-Section Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig('cross_section_profiles.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: cross_section_profiles.png")
        plt.close()
        
    # ==================== TIME-SERIES ANALYSIS ====================
    
    def plot_radius_of_curvature_timeseries(self):
        """Plot radius of curvature over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Convert time to microseconds if needed
        time_us = self.time_points / 1000  # ps to us
        
        # Plot 1: Radius of curvature over time
        ax = axes[0, 0]
        rad_curv_list = []
        for curvature in self.curvature_over_time:
            if curvature > 1e-6:
                rad_curv_list.append(1.0 / curvature)
            else:
                rad_curv_list.append(np.nan)
        
        ax.plot(time_us, rad_curv_list, 'b-', linewidth=2, label='Instantaneous')
        ax.fill_between(time_us, rad_curv_list, alpha=0.3)
        
        # Add moving average
        if len(rad_curv_list) > 5:
            rad_curv_clean = np.array(rad_curv_list)
            rad_curv_clean = np.nan_to_num(rad_curv_clean, nan=np.nanmean(rad_curv_clean))
            moving_avg = np.convolve(rad_curv_clean, np.ones(5)/5, mode='same')
            ax.plot(time_us, moving_avg, 'r--', linewidth=2.5, label='Moving average (5-frame)')
        
        ax.set_xlabel('Time (μs)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Radius of Curvature (nm)', fontsize=12, fontweight='bold')
        ax.set_title('Radius of Curvature Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Thickness over time
        ax = axes[0, 1]
        ax.plot(time_us, self.thickness_over_time, 'g-', linewidth=2)
        ax.fill_between(time_us, self.thickness_over_time, alpha=0.3, color='green')
        ax.axhline(y=np.mean(self.thickness_over_time), color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(self.thickness_over_time):.2f}')
        ax.set_xlabel('Time (μs)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Thickness (Å)', fontsize=12, fontweight='bold')
        ax.set_title('Membrane Thickness Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Deflection over time
        ax = axes[1, 0]
        ax.plot(time_us, self.deflection_over_time, 'orange', linewidth=2)
        ax.fill_between(time_us, self.deflection_over_time, alpha=0.3, color='orange')
        ax.axhline(y=np.mean(self.deflection_over_time), color='r', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(self.deflection_over_time):.2f}')
        ax.set_xlabel('Time (μs)', fontsize=12, fontweight='bold')
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

Radius of Curvature:
  Mean: {np.nanmean(rad_curv_clean):.1f} nm
  Min: {np.nanmin(rad_curv_clean):.1f} nm
  Max: {np.nanmax(rad_curv_clean):.1f} nm
  Std: {np.nanstd(rad_curv_clean):.1f} nm

Thickness:
  Mean: {np.mean(self.thickness_over_time):.2f} Å
  Std: {np.std(self.thickness_over_time):.2f} Å

Deflection:
  Mean: {np.mean(self.deflection_over_time):.2f} Å
  Std: {np.std(self.deflection_over_time):.2f} Å

Time Range: {time_us[0]:.2f} - {time_us[-1]:.2f} μs
Total Frames: {len(self.time_points)}
"""
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2))
        
        plt.suptitle('Time-Series Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig('radius_curvature_timeseries.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: radius_curvature_timeseries.png")
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
        
        fig.write_html('interactive_top_side_view.html')
        print(f"✓ Saved: interactive_top_side_view.html")
        
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
        fig.add_trace(go.Surface(x=self.xi, y=self.yi, z=self.zl, colorscale='Reds', showscale=False, opacity=0.95), row=1, col=2)
        
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
        
        fig.write_html('deflection_3d_analysis.html')
        print(f"✓ Saved: deflection_3d_analysis.html")
        
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
        
        fig.write_html('pore_3d_analysis.html')
        print(f"✓ Saved: pore_3d_analysis.html")
        

    def plot_leaflet_radial_structure(self):
        """Create phospholipid leaflet radial structure plot similar to publication figure"""
        fig, ax = plt.subplots(figsize=(4.2, 4.2))

        # Center coordinates
        x_center = np.mean(self.x)
        y_center = np.mean(self.y)

        # Radial coordinates (convert Å to nm)
        upper_r = np.sqrt((self.upper[:, 0] - x_center)**2 +
                          (self.upper[:, 1] - y_center)**2) / 10.0
        lower_r = np.sqrt((self.lower[:, 0] - x_center)**2 +
                          (self.lower[:, 1] - y_center)**2) / 10.0

        upper_z = self.upper[:, 2] / 10.0
        lower_z = self.lower[:, 2] / 10.0

        # Sort for smooth guide curves
        upper_sort = np.argsort(upper_r)
        lower_sort = np.argsort(lower_r)

        upper_r_sorted = upper_r[upper_sort]
        upper_z_sorted = upper_z[upper_sort]
        lower_r_sorted = lower_r[lower_sort]
        lower_z_sorted = lower_z[lower_sort]

        # Smooth guide lines
        window = max(5, len(upper_r_sorted) // 30)
        if window % 2 == 0:
            window += 1

        def moving_average(data, w):
            kernel = np.ones(w) / w
            return np.convolve(data, kernel, mode='same')

        upper_smooth = moving_average(upper_z_sorted, window)
        lower_smooth = moving_average(lower_z_sorted, window)

        # Background membrane shading
        ax.axhspan(0, 3.2, color='#dceaf7', alpha=0.7)
        ax.axhspan(-3.2, 0, color='#f6eadc', alpha=0.7)

        # Guide curves
        ax.plot(upper_r_sorted, upper_smooth, color='#1f77b4', linewidth=2.0, alpha=0.9)
        ax.plot(lower_r_sorted, lower_smooth, color='#ff7f0e', linewidth=2.0, alpha=0.9)

        # Scatter phospholipids
        ax.scatter(
            upper_r, upper_z,
            s=42, facecolor='#1f77b4', edgecolor='black',
            linewidth=1.1, alpha=0.95, zorder=5
        )

        ax.scatter(
            lower_r, lower_z,
            s=42, facecolor='#ff7f0e', edgecolor='black',
            linewidth=1.1, alpha=0.95, zorder=5
        )

        # Midplane
        ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)

        # Styling
        ax.set_xlim(0, max(np.max(upper_r), np.max(lower_r)) * 1.02)
        ax.set_ylim(min(lower_z) - 0.3, max(upper_z) + 0.3)
        ax.set_xlabel('R [nm]', fontsize=12, fontweight='bold')
        ax.set_ylabel('z [nm]', fontsize=12, fontweight='bold')
        ax.set_title('Phospholipid Leaflet Structure', fontsize=13, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.tick_params(direction='in', width=1.2, length=5)
        ax.grid(alpha=0.12)

        # Panel label
        ax.text(
            0.03, 0.97, 'D',
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top'
        )

        plt.tight_layout()
        plt.savefig(
            'D_3d_structure_leaflets_2d.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
        print("✓ Saved: D_3d_structure_leaflets_2d.png")
        plt.close()

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
        print("GENERATING TOP & SIDE VIEW ANALYSIS")
        print("="*80)
        self.plot_top_side_view()
        self.plot_3d_interactive_topside()
        
        print("\n" + "="*80)
        print("GENERATING CROSS-SECTION PROFILES")
        print("="*80)
        self.plot_cross_section_profiles()
        self.plot_leaflet_radial_structure()
        
        print("\n" + "="*80)
        print("GENERATING TIME-SERIES ANALYSIS")
        print("="*80)
        self.plot_radius_of_curvature_timeseries()
        
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
        print("\nNEW ANALYSIS (Image-based):")
        print("  ✓ top_side_view.png - Top and side views")
        print("  ✓ interactive_top_side_view.html - Interactive 3D views")
        print("  ✓ cross_section_profiles.png - Z, thickness, curvature profiles")
        print("  ✓ D_3d_structure_leaflets_2d.png - Phospholipid leaflet radial structure")
        print("  ✓ radius_curvature_timeseries.png - Radius of curvature over time")
        print("\nSTANDARD 2D ANALYSIS (PNG - 300 DPI):")
        print("  ✓ thickness_2d_analysis.png")
        print("  ✓ curvature_2d_analysis.png")
        print("  ✓ water_2d_analysis.png")
        print("\nSTANDARD 3D ANALYSIS (Interactive HTML):")
        print("  ✓ thickness_3d_analysis.html")
        print("  ✓ curvature_3d_analysis.html")
        print("  ✓ deflection_3d_analysis.html")
        print("  ✓ pore_3d_analysis.html")
        print("\n" + "="*80)
        print("TOTAL: 12 Publication-Quality Visualizations Generated!")
        print("="*80 + "\n")

# Usage
if __name__ == "__main__":
    TPR = "md_0_1000.tpr"
    XTC = "center1.xtc"
    LIPIDS = ["POPE", "POPG"]
    
    analyzer = ProfessionalMembraneSurfaceAnalyzer(TPR, XTC, LIPIDS, grid_points=250, smoothing_sigma=2)
    analyzer.run_analysis()
