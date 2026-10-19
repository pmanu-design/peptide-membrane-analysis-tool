import matplotlib
matplotlib.use('Agg')

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

def calculate_tilt_and_rotation(peptide, ref_resid, ref_atom_name):
    tilt_angles = []
    rotation_angles = []

    ref_atoms = peptide.select_atoms(f"resid {ref_resid} and name {ref_atom_name}")
    if len(ref_atoms) == 0:
        raise ValueError(f"Reference atom not found: resid {ref_resid} and name {ref_atom_name}")
    ref_atom = ref_atoms[0]

    for ts in peptide.universe.trajectory:
        try:
            helix_vector = peptide.positions[-1] - peptide.positions[0]
            helix_vector /= np.linalg.norm(helix_vector)

            tilt = np.degrees(np.arccos(np.dot(helix_vector, [0, 0, 1])))
            tilt_angles.append(tilt)

            ref_vec = ref_atom.position - peptide.center_of_geometry()
            ref_proj = ref_vec - np.dot(ref_vec, helix_vector) * helix_vector
            x_proj = np.dot(ref_proj, [1, 0, 0])
            y_proj = np.dot(ref_proj, [0, 1, 0])
            rho = np.degrees(np.arctan2(y_proj, x_proj)) % 360
            rotation_angles.append(rho)
        except:
            tilt_angles.append(np.nan)
            rotation_angles.append(np.nan)

    return np.array(tilt_angles), np.array(rotation_angles)

def plot_tilt_rotation(tilt, rotation, title, outfile):
    print(f"[INFO] Plotting tilt/rotation for: {title}")
    plt.figure(figsize=(6, 5))
    bins = [60, 60]
    hist, xedges, yedges = np.histogram2d(rotation, tilt, bins=bins, density=True)
    hist_smooth = gaussian_filter(hist, sigma=1)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    im = plt.imshow(hist.T, extent=extent, origin='lower', aspect='auto', cmap="YlGnBu")
    cs = plt.contour((xedges[:-1] + xedges[1:]) / 2,
                     (yedges[:-1] + yedges[1:]) / 2,
                     hist_smooth.T, levels=5, colors='black', linewidths=0.8)
    plt.clabel(cs, inline=True, fontsize=8)

    cbar = plt.colorbar(im)
    cbar.set_label("Density")

    mean_tau = np.nanmean(tilt)
    mean_rho = np.nanmean(rotation)
    plt.axvline(mean_rho, color='red', linestyle='--')
    plt.axhline(mean_tau, color='red', linestyle='--')
    plt.scatter(mean_rho, mean_tau, color='red')

    plt.xlabel("Rotation Angle ρ [°]")
    plt.ylabel("Tilt Angle τ [°]")
    plt.xlim(0, 360)
    plt.ylim(0, 180)
    plt.title(f"{title}\nτ = {mean_tau:.0f}°, ρ = {mean_rho:.0f}°")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

def plot_residue_angles(peptide, outfile_tilt, outfile_rho):
    resids = sorted(set(atom.resid for atom in peptide))
    tilt_vals = []
    rho_vals = []

    for resid in resids:
        res_atoms = peptide.select_atoms(f"resid {resid}")
        if res_atoms.n_atoms < 2:
            tilt_vals.append(np.nan)
            rho_vals.append(np.nan)
            continue
        try:
            vec = res_atoms.positions[-1] - res_atoms.positions[0]
            vec /= np.linalg.norm(vec)
            tilt = np.degrees(np.arccos(np.dot(vec, [0, 0, 1])))
            tilt_vals.append(tilt)

            ref_vec = res_atoms.positions[0] - peptide.center_of_geometry()
            ref_proj = ref_vec - np.dot(ref_vec, vec) * vec
            x_proj = np.dot(ref_proj, [1, 0, 0])
            y_proj = np.dot(ref_proj, [0, 1, 0])
            rho = np.degrees(np.arctan2(y_proj, x_proj)) % 360
            rho_vals.append(rho)
        except:
            tilt_vals.append(np.nan)
            rho_vals.append(np.nan)

    plt.figure(figsize=(6, 3.5))
    plt.plot(resids, tilt_vals, marker='o')
    plt.xlabel("Residue ID")
    plt.ylabel("Tilt Angle τ [°]")
    plt.title("Residue-Level Tilt")
    plt.ylim(0, 180)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile_tilt, dpi=300)
    plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.plot(resids, rho_vals, marker='o', color='green')
    plt.xlabel("Residue ID")
    plt.ylabel("Rotation Angle ρ [°]")
    plt.title("Residue-Level Rotation")
    plt.ylim(0, 360)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile_rho, dpi=300)
    plt.close()

def run_analysis(topology, trajectory, ref_resid=5, ref_atom_name="CA", peptide_sel="protein"):
    print("[INFO] Starting analysis...")
    u = mda.Universe(topology, trajectory)
    protein = u.select_atoms(peptide_sel)

    chain_ids = set(atom.segid.strip() for atom in protein.atoms)
    if not chain_ids or list(chain_ids) == ['']:
        print("[WARNING] No chain IDs found. Using full protein as single segment.")
        chain_ids = ['ALL']

    for chain_id in chain_ids:
        if chain_id == 'ALL':
            peptide = protein
        else:
            peptide = u.select_atoms(f"{peptide_sel} and segid {chain_id}")

        if peptide.n_atoms < 2:
            continue

        title = f"Chain {chain_id}" if chain_id != 'ALL' else "Full Protein"
        base = f"chain_{chain_id}" if chain_id != 'ALL' else "full_protein"

        tilt, rotation = calculate_tilt_and_rotation(peptide, ref_resid, ref_atom_name)
        plot_tilt_rotation(tilt, rotation, title, f"{base}_tilt_rotation.png")
        plot_residue_angles(peptide, f"{base}_residue_tilt.png", f"{base}_residue_rotation.png")

if __name__ == "__main__":
    topology = "md_0_300.tpr"
    trajectory = "cen20.xtc"
    run_analysis(topology, trajectory)

