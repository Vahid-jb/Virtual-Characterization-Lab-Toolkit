# Structure Analyzer and Crystallographic Identification

## Overview
The **Structure Analyzer** module provides a comprehensive workflow for identifying atomic structures and extracting precise lattice parameters from simulation data. The process is divided into two sequential stages:

*   **Crystallographic Identification**: Classifies atoms into structural phases (FCC, BCC, HCP, etc.) and separates them into distinct files.
*   **Lattice Parameter Estimation**: Analyzes a specific phase to determine its lattice constants ($a, b, c$) and angles ($\alpha, \beta, \gamma$) using up to three statistical methods.

---

## Part I: Crystallographic Identification

### Physics and Methodology
This module utilizes [Polyhedral Template Matching (PTM)](https://iopscience.iop.org/article/10.1088/0965-0393/24/5/055007) to classify the local structural environment of each atom. PTM is a robust method capable of identifying common crystal structures even at finite temperatures where thermal vibrations distort positions.

The method compares the local atomic neighborhood against ideal structural templates (polyhedra). The quality of the match is quantified by the **Root-Mean-Square Deviation (RMSD)**:

$$ RMSD = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} | \mathbf{r}_i - \mathbf{r}_{i}^{ideal} |^2 } $$

Where $\mathbf{r}_i$ are the coordinates of the neighbors and $\mathbf{r}_{i}^{ideal}$ are the template coordinates. If the RMSD is below a specified cutoff, the structure is identified.

**Supported Structures:**

*   Face-Centered Cubic (FCC)
*   Body-Centered Cubic (BCC)
*   Hexagonal Close-Packed (HCP)
*   Icosahedral (ICO)
*   Simple Cubic (SC)
*   Diamond (Cubic & Hexagonal)
*   Graphene

**Grain Segmentation**:
Following PTM, the module applies a [graph clustering algorithm](https://arxiv.org/abs/1806.01664) to group connected atoms of the same phase into grains. This allows for the filtration of small, insignificant ordered clusters (noise) based on a minimum grain size.

### Input Parameters


| Parameter | Type | Description | Effect |
| :--- | :--- | :--- | :--- |
| `input_file` | File Path | Path to the atomic structure file (e.g., `.xyz`, `.lammpstrj`). | Source data for analysis. |
| `input_file_type` | Enum | `single` or `traj`. | Specifies if the file is a single snapshot or a trajectory. |
| `trj_timestep` | Integer | (Trajectory only) The specific timestep to analyze. | Selects the frame from a trajectory file. |
| `RMSD_cutoff` | Float | PTM sensitivity threshold (Default: 0.1). | Higher values allow more distorted atoms to be classified but increase false positives. |
| `min_grain_size` | Integer | Minimum number of atoms per grain. | Filters out small clusters. The default of 5 is very permissive — a five-atom "grain" is usually a classification artifact near a dislocation, stacking fault or surface rather than a grain. Raise it (hundreds of atoms) for genuine grain statistics; keep it low only when the goal is to find small ordered clusters. |

### Outputs
1.  **Phase Distribution Report**: A text summary (`structure_analysis.txt`) showing the percentage and elemental composition of each detected phase.
2.  **Phase-Separated Files**: Separate coordinate files for each identified phase (e.g., `parent_phase_fcc.xyz`, `parent_phase_bcc.xyz`, `parent_phase_other.xyz`). These serve as inputs for Part II.
3.  **Grain Details**: A list of identified grains, their sizes (atom count), and mean orientation quaternions.
4.  **Phase Manifest** (`ptm_phases.json`): A machine-readable summary of the detected phases. The GUI reads it to build the 3D phase view.

| Key | Description |
| :--- | :--- |
| `schema_version` | Format version of the manifest (currently `1`). |
| `input_file` | The analysed structure file. |
| `rmsd_cutoff` | RMSD cutoff used for the run. |
| `total_atoms` | Number of atoms in the analysed frame. |
| `phases` | One entry per detected phase, with the keys below. |

| Phase key | Description |
| :--- | :--- |
| `name` | Phase name as in the report: `Other`, `FCC`, `HCP`, `BCC`, `ICO`, `SC`, `Cubic diamond`, `Hexagonal diamond` or `Graphene`. |
| `structure_type` | PTM structure type ID: 0 Other, 1 FCC, 2 HCP, 3 BCC, 4 ICO, 5 SC, 6 cubic diamond, 7 hexagonal diamond, 8 graphene. |
| `count` | Number of atoms in the phase. |
| `percentage` | Share of all atoms, in percent. |
| `composition` | Atom count per element, e.g. `{"Zn": 16300, "Cu": 15659}`. Empty if the input file has no element names. |
| `color` | OVITO's RGB color for the structure type, with components from 0 to 1. |
| `file` | Name of the phase-separated `.xyz` file in the output directory. |

### 3D Phase View (GUI)
After a successful run, the GUI loads every phase listed in `ptm_phases.json` into an interactive OVITO viewport. Atoms are colored with OVITO's standard PTM colors, e.g. FCC green, HCP red, BCC blue and Other white. Drag to rotate the view and use the mouse wheel to zoom.

*   **Phases** tab: lists each phase with its color, atom count, share and composition. Untick a phase to hide it in the 3D view.
*   **Report** tab: the full text report.
*   **Color by**: switches between phase colors and element colors.
*   **Zoom to fit**: moves the camera so that all visible atoms fit the view.

The GUI deletes `ptm_phases.json` before each run, so the view always shows the current run. Phase files left over from earlier runs in the same output directory are ignored.

---

## Part II: Lattice Parameter Estimation

### Physics and Methodology
Once a specific phase has been isolated, this module determines its unit cell parameters. To ensure accuracy across different sample types (bulk, datasets with defects, finite clusters), the module employs **three distinct statistical methods** and reports results from all applicable ones.

#### Method 1: Nearest-Neighbour Distance from the RDF
A robust baseline, effective for finite clusters and non-periodic inputs.

1.  **Radial distribution function**: builds a volume-normalised $g(r)$ from KD-tree
    neighbour queries around randomly chosen interior atoms.
2.  **First shell**: takes the centroid of the first $g(r)$ peak as the nearest-neighbour
    distance $d_{nn}$.
3.  **Derivation**: converts $d_{nn}$ to the conventional lattice constant using the
    assumed geometry ($a_{fcc} = d_{nn}\sqrt{2}$, $a_{bcc} = 2 d_{nn}/\sqrt{3}$,
    $a_{hcp} = d_{nn}$).


#### Method 2: Vector Statistics
A rotation-invariant approach for bulk and polycrystalline systems. It reconstructs the
lattice geometry from the distribution of interatomic vectors:

1.  **Directional clustering**: projects first-shell interatomic vectors onto the unit
    sphere and clusters them (DBSCAN) to find the primary lattice directions,
    independently of sample orientation. A direction and its antipode are merged into a
    single lattice axis (matching on $|\cos\theta|$).
2.  **Lattice reconstruction**: selects the best linearly independent triplet of
    cluster centres, scoring by cell volume and cluster population.
3.  **Reduced primitive cell**: the measured basis is reduced over short integer
    combinations and put into the Niggli sign convention, so the reported primitive
    cell is reproducible and matches tabulated values — FCC as $(60,60,60)$ and BCC as
    $(109.47,109.47,109.47)$, stable across random seeds. Cluster-centroid signs are
    arbitrary and the clustering may return any three of the six $\langle110\rangle$
    axes, so without this the same lattice came out as $(60,60,90)$ or
    $(70.53,70.53,109.47)$ depending on the run.
4.  **Conventional cell, measured not assumed**: For a phase identified as FCC or
    BCC, the conventional cell axes are obtained from short integer combinations
    of the measured primitive vectors. The code selects the most nearly
    orthogonal triplet whose volume equals \(N V_{\text{prim}}\), where \(N=4\)
    for FCC and \(N=2\) for BCC. This basis-independent procedure preserves
    measured lattice distortions rather than automatically assigning
    \(\alpha=\beta=\gamma=90^\circ\).

    | imposed \(c/a\) (BCT test) | reported \(a\) | \(b\) | \(c\) | measured \(c/a\) | `cell_type` |
    | ---: | ---: | ---: | ---: | ---: | :--- |
    | 1.00 | 2.8713 | 2.8713 | 2.8713 | 1.0000 | `conventional_cubic` |
    | 1.05 | 2.8662 | 2.8662 | 3.0095 | 1.0500 | `conventional_distorted` |
    | 1.15 | 2.9661 | 2.9661 | 3.4110 | 1.1500 | `conventional_distorted` |
    | 1.41 | 2.8648 | 2.8691 | 4.0575 | 1.4163 | `conventional_distorted` |

    The output includes `cubic_consistent`, `axial_ratio_c_over_a`,
    `max_length_deviation`, and `max_angle_deviation_deg`. A non-cubic result
    includes an explanatory `note`.

    The measured primitive-cell data are also reported as `a_primitive`,
    `alpha_primitive`, `primitive_vectors`, and `volume_primitive`.
    `cell_type` indicates whether the reported \(a\), \(b\), \(c\),
    \(\alpha\), \(\beta\), and \(\gamma\) describe a primitive cell, a
    conventional cubic cell, or a conventional distorted cell.

#### Method 3: Coordination-Shell Ratios
Identifies the structure and refines the lattice constant from the positions of the
first several coordination shells.

1.  **Shell extraction**: locates the first four peaks of the volume-normalised $g(r)$,
    each to sub-bin precision via a local centroid.
2.  **Ratio matching**: compares the measured ratios $d_i/d_1$ against the ideal ratios
    for each candidate structure. Ratios are scale-free, so this genuinely discriminates:

    | Structure | $d_2/d_1$ | $d_3/d_1$ | $d_4/d_1$ |
    | :--- | :--- | :--- | :--- |
    | FCC | 1.4142 | 1.7321 | 2.0000 |
    | BCC | 1.1547 | 1.6330 | 1.9149 |
    | HCP | 1.4142 | 1.6330 | 1.7321 |

3.  **Scoring**: the RMS relative residual is reported per candidate as `scores`, and
    `reliability_score` is derived from it. On generated crystals the correct structure
    scores $<0.01$ (stable to $0.10$ Å of thermal displacement) while the wrong ones
    score $>0.11$; a uniform random point cloud scores $\approx 0.33$.
4.  **Gaussian refinement**: a Gaussian is also fitted to the first peak of the
    nearest-neighbour histogram and reported as `d_nn_fit`:

    $$ f(r) = A \cdot \exp\left( - \frac{1}{2} \left( \frac{r - \mu}{\sigma} \right)^2 \right) + BG $$

### Honesty of the reported result
The module refuses to report a structure it cannot support:

*   If the best shell-ratio residual exceeds $0.06$, the structure is reported as
    `UNKNOWN`, no $a,b,c$ are emitted, and only $d_{nn}$ is given. Amorphous and liquid
    configurations land here.
*   If fewer than two shells are resolved, the structure is **not** guessed: $d_{nn}$
    fixes the scale but carries no information about which lattice it is.
*   `ambiguous` is set whenever the leading candidate is FCC or HCP. Those two share
    shells 1 and 2 exactly ($1, \sqrt{2}$) and first differ at shell 3, where the HCP
    shell ($r = c$) has multiplicity 2 against FCC's 24 — too weak to resolve reliably
    in an RDF. **PTM (Part I) is the authority for that distinction**, which is why the
    two-stage workflow exists.
*   For an HCP structure, the output field `c_source`
    indicates whether the axial lattice parameter \(c\) was measured from a
    resolved coordination shell (`c_axis_shell`) or assigned using the ideal HCP
    ratio (`assumed_ideal`):

    $$
    \frac{c}{a}=\sqrt{\frac{8}{3}}.
    $$

    The \(c\)-axis shell can be difficult to resolve because it contains only two
    neighbors and may overlap with nearby coordination shells. Therefore, treat
    the reported HCP \(c/a\) value as assumed unless `c_source = c_axis_shell`.
    For strongly non-ideal HCP structures, in-plane and out-of-plane neighbor
    distances can overlap. In such cases, the reported \(a\) and \(c/a\) values
    should be interpreted cautiously.

*   When PTM supplies a phase, the independent RDF classification is still run and
    reported as `rdf_detected_structure` / `rdf_agrees_with_structure`, so a
    disagreement is visible rather than hidden.

> **Part I and Part II answer different questions.** PTM classifies each atom's *local
> polyhedron* against ideal templates, so strained material — dislocation cores, free
> surfaces, grain boundaries — is binned as `Other` even when it sits on a perfectly
> good lattice. Part II measures the *average* lattice of whatever atoms it is given.
> Running Part II on a `parent_phase_other.xyz` file can therefore legitimately return
> the parent lattice.

### Neighbour cutoff
`vector_stats_max_cutoff` is an absolute distance in Ångström and is therefore not
transferable between materials, phases, strains or temperatures. Set it to `auto` to
scale it from the measured nearest-neighbour distance ($1.45\,d_{nn}$), which is the
recommended setting for exploratory work. A fixed value below $1.15\,d_{nn}$ truncates
the first coordination shell and now raises an explicit warning.

### Periodic input
When a cell is available it is used for neighbour searching: a $3\times3\times3$ block
of periodic images is built, so atoms near a box face keep their full neighbourhood.
Shell *positions* are insensitive to this — truncating a neighbour sphere removes whole
neighbours but does not move the distances of those that remain, and the shell centroid
is symmetric — so the lattice constant was already unbiased. Coordination *counts* are
not: for a $4\times4\times4$ FCC box the modal coordination number came out 8 instead
of 12 without images, which matters for the no-PTM fallback classifier.

If the input carries a cell (extended-XYZ `Lattice="..."`, or a format OVITO reads),
the simulation box is reported separately as `supercell_A/B/C` and is **not** used as
the unit cell — a $10\times10\times10$ FCC box has $A = 36.15$ Å, not $a = 3.615$ Å.
The lattice constant is always measured from the atoms. The ratio box$/a$ is reported as
`supercell_repeats` and flagged via `supercell_commensurate`, which is a useful
consistency check: it should be near-integer for a commensurate box.

### Input Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input` | File Path | `parent_phase_*.xyz` | The phase file generated in Part I. |
| `verbose` | Boolean | `on` | Enables detailed logging of the vector search process. |
| `use_advanced_methods`| Boolean | `on` | Enables Method 2 (Vector Statistics) and Method 3 (Coordination-Shell Ratios). |
| `rmsd_cutoff` | Float | 0.1 | RMSD cutoff for the PTM classification run by the lattice detector. |
| `plot` | Boolean | `off` | Generates the comparison-histogram plot. |
| `vector_stats_k_nn` | Integer | 20 | Number of neighbors for Method 2 (recommended: 14 for BCC, 18 for FCC). |
| `vector_stats_max_cutoff`| Float or `auto` | 5.0 Å | Physical range limit for the neighbour search. Absolute distances are material-specific; `auto` sets $1.45\,d_{nn}$ from the measured nearest-neighbour distance. `auto` is accepted in parameter files and on the command line; the GUI field is numeric only. |
| `vector_stats_dbscan_eps`| Float | 0.07 | Direction-cluster tolerance, as a chord distance between unit vectors (0.07 corresponds to about 4°). |
| `vector_stats_merge_threshold`| Float | 0.85 | \|cos θ\| above which two direction clusters are treated as the same lattice axis. |
| `vector_stats_min_neighbors`| Integer | 6 | DBSCAN `min_samples` for a valid direction cluster. |
| `vector_stats_random_seed`| Integer | 42 | Seeds all atom sampling; identical inputs give identical output. |
| `vector_stats_tolerance_factor`| Float | 0.12 | Relative width of the first-shell vector filter. |
| `vector_stats_sample_size`| Integer | 3000 | Atoms sampled for vector statistics. |
| `vector_stats_hist_bins`| Integer | 300 | Histogram bins for the diagnostics. |
| `neighbor_k` | Integer | 20 | Neighbours per atom for the diagnostic nearest-neighbour statistics. |
| `ls_sample_size` | Integer | 2000 | Atoms sampled for the NN/RDF statistics. |
| `ls_k_neighbors` | Integer | 6 | Neighbours per atom for the KD-tree query. |
| `ls_bin_width` | Float | 0.01 | Bin width in Å for the nearest-neighbour histogram fit. |
| `ls_use_periodic` | Boolean/Auto | Auto | Forces Periodic Boundary Conditions (PBC) for Method 3. Requires a cell in the input file; otherwise the run is non-periodic regardless. |

British spellings (`neighbour_k`, `ls_k_neighbours`, `vector_stats_min_neighbours`, `vector_stats_k_nn_neighbours`) are
accepted as aliases. Any key that is neither recognised nor an alias now produces an
explicit warning rather than being silently ignored. Parameter files are read as UTF-8.

`vector_stats_n_trials` is accepted for backward compatibility but is **not used**: the
basis search is an exhaustive scan over at most six candidate directions, so there is no
trial loop for it to control.

### Outputs
1.  **Lattice Parameters** (per method), written to `<input>_lattice_analysis.txt`:
    *   Lengths $a, b, c$ (Å) and angles $\alpha, \beta, \gamma$ (°), with `cell_type`
        stating whether these are the conventional or the primitive cell
    *   `d_nn` and `d_nn_source`; `d_nn_median` as a diagnostic
    *   `shell_distances`, `shell_ratios`, `shell_ratio_error` and per-structure `scores`
    *   `volume`; plus `primitive_vectors` and `volume_primitive` for Method 2
    *   `supercell_*` and `supercell_repeats` for periodic input
2.  **Reliability**: `reliability_score` (0–1, from the shell-ratio residual) mapped to
    High/Medium/Low per method, together with the `ambiguous` flag and, for HCP,
    `c_source`.
3.  **Visualisation**: `comparison_histograms.png` and `lattice_hist_comp_data.json`
    (the raw histogram data, which the GUI reads to draw the plot itself). Both are
    written for periodic and non-periodic input when `plot = on` and both the
    vector-statistics and the nearest-neighbour data are available. The GUI deletes an
    existing `lattice_hist_comp_data.json` in the output folder before each run, so a
    plot from an earlier run is never shown as the current result.


---

## Notes
The combination of Crystallographic Identification and Lattice Parameter Estimation as a workflow is specifically beneficial for analyzing local crystallography and the structure of extended defects and interphases, as well as for following microstructural evolution or investigating phase transition sequences.
By iteratively adjusting RMSD parameters and visualizing the resulting isolated phase 