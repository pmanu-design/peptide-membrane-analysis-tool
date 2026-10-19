# MD Membrane Analysis Suite — Launcher GUI

A single desktop GUI that wraps all 17 of your MDAnalysis-based membrane
scripts. Pick an analysis family, pick the specific variant, point it at
your topology/trajectory files, hit Run, and watch live log output.

## Folder layout

```
launcher.py
scripts/                  <- all 17 original analysis scripts (unmodified)
```

Keep `launcher.py` and the `scripts/` folder together — don't separate them.

## Requirements

- Python 3.8+
- Tkinter (ships with most Python installs; on Linux you may need
  `sudo apt install python3-tk`)
- Whatever each underlying analysis script needs: MDAnalysis, numpy, scipy,
  matplotlib, tqdm, etc. Install these once, e.g.:
  ```
  pip install MDAnalysis numpy scipy matplotlib tqdm
  ```

## Running it

```
python launcher.py
```

## How it works

The left panel groups your 17 scripts into 4 analysis families based on what
they actually compute, with each original script appearing as a sub-option
button under its family:

1. **Membrane Surface & Topology Analysis** (7 variants) — Version 34/35 and
   their "leaflet structure" / "updated" revisions. `Version35_UPDATED3` is
   flagged as recommended since it fixes a performance bug and has a real
   `--tpr/--xtc/--lipids/--water/--output-dir` CLI.
2. **Curvature & Pore Analysis** (7 variants) — the `curvature-5-*` and
   `curvature-combined-*` family, ranging from lightweight pore/curvature
   mapping up to the "Combined Ultra v10" script which does essentially
   everything (SCD order parameters, Voronoi packing defects, tilt angle,
   asymmetry, etc).
3. **Peptide Depth Insertion** (1 script) — ligand/peptide insertion depth
   vs. time.
4. **MD Density Analysis** (2 variants) — 1D mass-density profiles along the
   membrane normal.

Clicking a variant loads a parameter form on the right: topology file,
trajectory file, lipid residue names, peptide/protein residue name (when the
script supports one), water residue name (only for the v35_UPDATED3 CLI),
and an output folder. Click **Run Analysis** and the log streams live below.

### How parameter injection works

Most of these scripts hard-code their input file paths and selection
strings near the top (`TPR_FILE = "md_0_1000.tpr"`, etc.) rather than taking
command-line arguments. Rather than editing your original files, the
launcher:

- Makes a **patched copy** of the chosen script in a `_patched_scripts/`
  folder (your originals in `scripts/` are never modified), substituting in
  the file paths and selection strings you entered in the form.
- Runs that copy with its working directory set to your chosen output
  folder, since the scripts mostly save figures/data to relative paths
  (`output/...` or the current directory).
- The one script that already has a proper CLI
  (`advanced_membrane_analysis_Version35_UPDATED3.py`) is launched directly
  via its `--nogui` flag instead of being patched.

### Stopping a run

The **Stop** button sends a terminate signal to the running script. Use
**Open Output Folder** to jump straight to your results once a run
finishes.

## Notes / things you may want to tailor

- Default lipid names are pre-filled as `POPE POPG` and peptide as
  `PROTEIN` (matching what's hard-coded in most of your scripts) — change
  them in the form as needed per system.
- If you rename or move any script inside `scripts/`, update the
  corresponding `"file"` entry in the `CATALOG` dict near the top of
  `launcher.py`.
- The variable names used for patching (`TPR_FILE`, `XTC_FILE`, `TPR`,
  `XTC`, `topology_file`, `trajectory_file`, `LIPIDS`, `LIPID_RESNAMES`,
  `PEPTIDE_RESNAME`) were detected directly from your scripts. If you add a
  new script to the suite with different variable names, add a new entry to
  `CATALOG` with a matching `patch` spec.
