# XRD Analysis Module

## Overview
This module provides three distinct methodologies for computing X-ray Diffraction (XRD) patterns, catering to different system types (crystals, non-periodic clusters, large systems) and accuracy requirements.

1. **XRD-ReciprocalSum**: Direct reciprocal-space summation, intended primarily for periodic crystals.
2. **XRD-Kinematical**: Crystallographic periodic-unit-cell approach with advanced corrections such as Debye–Waller damping, absorption, instrument broadening, and indexed output.
3. **Debye Scattering**: Pairwise scattering calculation for finite non-periodic nanoparticles, clusters, amorphous structures, liquids, or extracted defective regions. It supports anomalous scattering.

---

## Part I: XRD-ReciprocalSum

### Physics and Methodology
This method calculates diffraction intensity by summing atomic scattering factors over reciprocal-space vectors defined by the simulation box. It follows the reciprocal-space convention and intensity normalization used by the LAMMPS `compute xrd` command.


The reciprocal vector is:

$$
\mathbf{K}
=
(i\Delta K_x,\;j\Delta K_y,\;k\Delta K_z)
$$

where, in automatic mode,

$$
\Delta K_i=\frac{c_i}{L_i}.
$$

Here \(L_i\) is the box length in direction \(i\), and \(c_i\) is a user-defined reciprocal-space resolution parameter.

The reciprocal-space convention is:

$$
|\mathbf{K}|
=
\frac{2\sin\theta}{\lambda}
=
\frac{1}{d},
$$

therefore:

$$
s
=
\frac{\sin\theta}{\lambda}
=
\frac{|\mathbf{K}|}{2}.
$$

This differs from the conventional XRD scattering-vector convention:

$$
q
=
\frac{4\pi\sin\theta}{\lambda}.
$$

The two are related by:

$$
q=2\pi|\mathbf{K}|.
$$

The structure factor is:

$$
F(\mathbf{K})
=
\sum_{j=1}^{N}
f_j(s)
\exp\left[
2\pi i\mathbf{K}\cdot\mathbf{r}_j
\right].
$$

The intensity at a reciprocal-space mesh point is:

$$
I(\mathbf{K})
=
\frac{|F(\mathbf{K})|^2}
{N_{\mathrm{atoms}}}.
$$

When Lorentz–polarization correction is enabled, the code applies the square root of the LP factor to the real and imaginary structure-factor amplitudes. The final intensity therefore contains the full LP factor:

$$
I_{\mathrm{LP}}(\mathbf{K})
=
L_p(\theta)
\frac{|F(\mathbf{K})|^2}
{N_{\mathrm{atoms}}}.
$$

The atomic scattering factor is represented by a Cromer–Mann-type expression:

$$
f(s)
=
\sum_{m=1}^{4}
A_m\exp(-B_m s^2)
+
C.
$$

The Lorentz–polarization factor is:

$$
L_p(\theta)
=
\frac{1+\cos^2(2\theta)}
{\cos\theta\sin^2\theta}.
$$

### Key Features


- **Periodic systems**: It is most appropriate for periodic orthogonal simulation cells with known box dimensions.



- **Manual reciprocal mesh**: In `manual = 1` mode, the mesh uses:

    $$
    \Delta K_i=c_i.
    $$

    Therefore, `prd` is recorded for provenance but does not alter the reciprocal-space mesh in manual mode.

- **Automatic reciprocal mesh**: In automatic mode, reciprocal mesh spacing is set by the physical box dimensions:

    $$
    \Delta K_i=\frac{c_i}{L_i}.
    $$

    Missing or incorrect box dimensions change the reciprocal mesh and therefore alter sampled diffraction positions.

- **Histogram output**: The final \(2\theta\) pattern is a weighted sum over accepted reciprocal mesh nodes. It is not automatically a normalized orientational powder average.

- **Orthogonal cells**: The strict LAMMPS-compatible mode is intended for orthogonal cells. Triclinic cells are not equivalent to the LAMMPS `compute xrd` formulation used here.

### Input Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `structure_file` | Path | Required | Input structure (XYZ, LAMMPS Data, etc.). |
| `output_file` | Path | Required | Output diffraction data file. |
| `plot_file` | Path | `xrd_plot.png` | Output plot file. |
| `wavelength` | Float | `1.5406` | X-ray wavelength in Å. This module uses a numeric wavelength|
| `2Theta` | Float List | `10, 179` | Minimum and maximum \(2\theta\) angles in degrees. An invalid range falls back to the default. |
| `pbc` | Bool List | From file | Periodic boundaries in x, y, and z. |
| `atom_types` | List | Required | Ordered scattering-factor labels. |
| `atom_type_mode` | Enum | `auto` | Atom-type interpretation: `auto`, `lammps_numeric`, or `chemical_symbols`. |
| `manual` | Bool | `0` | Enable manual reciprocal-space mesh mode. |
| `prd` | Float List | None | Box dimensions recorded for provenance. |
| `c` | Float List | `1,1,1` | Reciprocal-space resolution modifiers. Lower values create a finer reciprocal grid and increase computational cost. |
| `num_bins` | Int | `250` | Number of \(2\theta\) histogram bins. |
| `LP` | Bool | `1` | Apply the Lorentz–polarization factor. |
| `compatibility_mode` | Enum | `strict_lammps` | `strict_lammps` rejects unsupported cells; `relaxed` permits non-benchmark approximations. |
| `max_reciprocal_candidates` | Int | `20000000` | Maximum permitted candidate reciprocal nodes before a safety stop. |
| `allow_large_reciprocal_grid` | Bool | `0` | Allow a reciprocal grid larger than the safety cap. |
| `plot_smoothing_sigma_deg` | Float | `0.0` | Display-only Gaussian smoothing of the binned diffraction curve. |
| `echo` | Bool | `0` | Print progress and diagnostic information. |
| `plot` | Bool | `0` | Generate a diffraction plot. |

---

## Notes

This module follows the methodology of the [LAMMPS compute xrd](https://docs.lammps.org/compute_xrd.html) command. Users can consult the corresponding LAMMPS documentation for detailed background information.



Do not use inferred box dimensions when benchmarking a periodic LAMMPS calculation. Since:

$$
\Delta K_i=\frac{c_i}{L_i},
$$

an invented box length changes the reciprocal-space sampling grid.

---

## Part II: XRD-Kinematical

### Physics and Methodology
This module implements a rigorous **kinematical diffraction** approach, enhancing crystallographic structure-factor calculations with optional experimental corrections.

It is designed for a periodic crystal, represented either by a unit cell or by a periodic simulation box. Peak positions arise from the lattice that is used; atomic coordinates determine the structure factor and reflection intensity.

The crystal does **not** have to be ideal. Vacancies, substitutional disorder, thermal displacement, strain and extended defects are all computed correctly, if the lattice has the true periodicity of the atoms. When the structure file carries a simulation box, or one can be recovered from the unit cell, every atom is kept and nothing is folded, so the defects reach the structure factors.

What is genuinely out of scope is an **aperiodic** structure: a finite nanoparticle, an isolated cluster, or any configuration with free surfaces. For those there is no lattice to supply, and the module stops with a diagnostic pointing at `XRD-Debye_Scattering`. See *Periodicity Assessment* below.

The crystallographic structure factor is:

$$
F_{hkl}
=
\sum_j
f_j(s)
\exp\left[
2\pi i(hx_j+ky_j+lz_j)
\right].
$$

The base reflection intensity is calculated by `pymatgen` through `XRDCalculator`. The reported integrated intensities include crystallographic multiplicity and the Lorentz–polarization factor implemented by pymatgen.

The intended correction order is:

1. Pymatgen kinematical integrated reflection intensity.
2. Optional species-resolved Debye–Waller damping.
3. Optional sample absorption correction.
4. Optional user scale factor.
5. Convolution with a Gaussian line profile.
6. Optional display normalization.

The applied correction order is written into the output file headers.

### 1. Absorption Correction

X-ray absorption is based on the Beer–Lambert relation:

$$
I=I_0\exp(-\mu x),
$$

where \(\mu\) is the linear absorption coefficient and \(x\) is the X-ray path length.

Absorption correction is disabled unless:

```text
apply_absorption = yes
```

The code supports several explicit sample geometries.

- **Debye–Scherrer cylindrical capillary**:

    $$
    A(2\theta)
    =
    \left\langle
    \exp[-\mu(L_{\mathrm{in}}+L_{\mathrm{out}})]
    \right\rangle.
    $$

    The average is evaluated numerically over the capillary cross section. Here \(L_{\mathrm{in}}\) and \(L_{\mathrm{out}}\) are the incoming and outgoing path lengths for each scattering point inside the cylindrical sample.

    For small \(\mu R\), the cylindrical result approaches:

    $$
    A
    \rightarrow
    1-\frac{16}{3\pi}\mu R.
    $$

- **Flat plate in Bragg–Brentano reflection geometry**:

    $$
    A(\theta)
    =
    \frac{
    1-\exp[-2\mu t/\sin\theta]
    }{
    2\mu t/\sin\theta
    }.
    $$

    Here \(t\) is the sample thickness. This expression is normalized so that \(A\rightarrow1\) as \(\mu\rightarrow0\).

- **Flat plate in symmetric transmission geometry**:

    $$
    A(\theta)
    =
    \exp\left[
    -\mu t
    \left(
    \frac{1}{\cos\theta}-1
    \right)
    \right].
    $$

- **Approximate slab attenuation**:

    $$
    A(\theta)
    =
    \exp\left[
    -\mu t
    \left(
    \frac{1}{\sin\theta}-1
    \right)
    \right].
    $$

    This is explicitly an approximate single-path attenuation expression and should not be used as a substitute for a geometry-specific diffraction absorption correction.

### 2. Instrumental Broadening

To simulate finite instrument resolution, discrete Bragg reflections are convolved with a Gaussian profile:

$$
I_{\mathrm{curve}}(2\theta)
=
\sum_i I_i
G(2\theta-2\theta_i;\sigma),
$$

where the normalized Gaussian line profile is:

$$
G(x;\sigma)
=
\frac{1}{\sigma\sqrt{2\pi}}
\exp\left[
-\frac{x^2}{2\sigma^2}
\right].
$$

The Gaussian is area normalized:

$$
\int_{-\infty}^{\infty}
G(x;\sigma)\,dx
=
1.
$$

Therefore:

$$
\int
I_{\mathrm{curve}}(2\theta)\,
d(2\theta)
\approx
\sum_i I_i.
$$

The discrete reflection list contains integrated intensities, whereas the continuous curve contains intensity density per degree \(2\theta\). Their numerical values should therefore not be expected to be equal.

The total Gaussian width is calculated by quadrature:

$$
\sigma_{\mathrm{total}}
=
\sqrt{
\sigma_{\mathrm{base}}^2+
\sigma_{\mathrm{inst}}^2+
\sigma_{\mathrm{plot}}^2
}.
$$

where:

- \(\sigma_{\mathrm{base}}\) is `gaussian_width`.
- \(\sigma_{\mathrm{inst}}\) is `instrument_sigma_deg`.
- \(\sigma_{\mathrm{plot}}\) is `plot_smoothing_sigma_deg`.

All widths are Gaussian standard deviations in degrees \(2\theta\), not FWHM values. The conversion is:

$$
\mathrm{FWHM}
=
2\sqrt{2\ln2}\,\sigma
\approx2.35482\sigma.
$$

### 3. Peak Filtering

The algorithm separates the physical diffraction curve from the display-peak list.

The complete calculated reflection list is always used to create the continuous broadened curve. Reflection filtering affects only:

- Peak labels in the plot.
- `xrd_peaks_display.txt`.
- The concise tabulated display peak list.

The display selection can apply:

1. **Intensity cutoff**: retain reflections above a defined fraction or percent of the maximum reflection intensity.
2. **Minimum separation**: when two reflections are closer than `min_peak_separation`, retain the strongest reflection for display.
3. **Maximum count**: limit the number of displayed peaks with `max_peaks`.

Filtering does not remove overlapping reflections from the physical continuous curve. Their Gaussian profiles add.

### Key Features

- **Periodic crystal model**: Appropriate for crystallographic cells and periodic simulation boxes, **including defective ones** — vacancies, disorder, strain and extended defects are computed, not excluded. Only aperiodic structures (finite clusters, free surfaces) are outside its scope.

- **Cell sourcing**: The periodic cell is taken from the structure file, from a cell ASE can read, from a separate `cell_file`, from a whole-number multiple of the supplied unit cell, or inferred from the coordinates alone at any orientation. If every route fails the structure is not a periodic crystal and the run stops with a pointer to `XRD-Debye_Scattering.py`.

- **Cost guard**: The work scales as cell volume × atoms, so an oversized box is refused before the calculation starts, with the alternatives named. See `max_reflection_cost`.

- **Coordinate handling**: Coordinates can be read as Cartesian or fractional coordinates. Cartesian coordinates are converted to fractional coordinates relative to the supplied lattice.

- **Coordinate wrapping**: `wrap_coords = yes` folds fractional coordinates into \([0,1)\). Folding is used only in `unit_cell` mode; in `simulation_box` mode the box is the lattice and nothing is folded, which is what preserves the defects. Folding is not appropriate for a finite particle or a non-commensurate supercell, and the periodicity assessment refuses those cases rather than folding them.

- **Image merging**: In `unit_cell` mode, `check_duplicate_sites = yes` **merges** the images that folding stacked onto one crystallographic site; it does not merely warn about them. Without this step the cell would retain every atom of the supercell inside the volume of a single cell.

- **Structure diagnostics**: The code reports how many atoms lay outside the supplied cell, how many images each site received, and whether any site received more than one element. An uneven spread of images is flagged, because folding a defective supercell keeps only the average cell and discards the defects.

- **Experimental corrections**:
    - Lorentz–polarization is included in the base pymatgen reflection calculation.
    - Debye–Waller damping is optional.
    - Absorption is optional and geometry-specific.
    - Instrumental broadening is optional.
    - Display normalization is optional.

- **Three-output design**:
    - `xrd_reflections_full.txt`: complete unfiltered integrated reflection list.
    - `xrd_peaks_display.txt`: filtered display-only peak list.
    - `xrd_curve.txt`: continuous broadened curve with intensity density per degree \(2\theta\).

- **Outputs**:  
  Structure information includes number of atoms, composition, lattice parameters, and volume.  
  Peak analysis reports the number of computed and displayed reflections. It lists the strongest reflections by integrated intensity, including \(2\theta\), intensity, d-spacing, multiplicity, and Miller indices.  
  Plotting creates an indexed XRD pattern.

### Periodicity Assessment

The lattice given to pymatgen sets both which reflections exist and what their structure factors are, so it has to be the true periodicity of the atoms. The module establishes it automatically; `lattice_mode = auto` applies the following rule in order.

1. **Box in the file.** If line 2 of the structure file carries an extended-XYZ `Lattice="ax ay az bx by bz cx cy cz"` entry (as written by ASE, OVITO and LAMMPS dump-style output), that box is the lattice. Every atom is kept and nothing is folded, so defects are computed. Mode: `simulation_box`.

2. **Cell read by ASE.** Otherwise, with `use_ase = yes`, ASE is asked to read a cell from the structure file itself. This covers the formats that carry their box outside an extended-XYZ comment line — LAMMPS dump and data files, `POSCAR`/`CONTCAR`, CIF and the rest of the ASE reader set. The first frame is taken, to match the frame the coordinates are read from. Mode: `simulation_box`.

3. **Cell from `cell_file`.** Otherwise, if `cell_file` names a second file, its box is applied to these coordinates. This is the route for a dump that lost its header, or for coordinates that belong with the box of the relaxed data file they came from. Mode: `simulation_box`.

4. **Box recovered from the unit cell.** Otherwise, if `a`, `b`, `c` and the angles were supplied, the module looks for a whole-number multiple of that cell. The span of the coordinates is measured in **fractional** units and rounded to an integer — measuring in fractional rather than Cartesian units is what makes this valid for monoclinic and triclinic cells, where the extent along \(x,y,z\) is not the extent along \(a,b,c\). Each candidate box is then **confirmed rather than trusted**: it is accepted only if it repairs the coordination, that is, if fewer than `periodicity_surface_tol` of the atoms remain under-coordinated once the box is applied. A genuine periodic cell becomes fully coordinated under its own box; a cluster does not, whatever box is tried. Mode: `simulation_box`.

5. **Cell inferred from the coordinates.** Otherwise, with `infer_cell = yes`, the box is recovered from the coordinates alone, at any orientation. Two searches run. A per-axis scan looks for the repeat length along each of \(x,y,z\), scoring a trial period by how well the slab it implies restores coordination; an axis clears when it reaches `axis_period_min_score`. In parallel, candidate translations within `cell_inference_reach` × \(r_1\) are tested for mapping the structure onto itself **species by species**, so that B2 yields its simple-cubic cell rather than the bcc cell of the underlying point set. A translation is accepted as a lattice vector when it maps at least `cell_inference_min_score` of the atoms onto an atom of the same species, within `cell_inference_tol` × \(r_1\). This is the route that handles a cell built on directions such as \([111]\), the usual dislocation setup, which no multiple of \(a,b,c\) along \(x,y,z\) can reproduce. Mode: `simulation_box`.

6. **Fold into the unit cell.** If no box can be established, the coordinates are folded into the supplied cell and the images are merged back onto their crystallographic sites. The result is accepted only if the fold is meaningful, judged by element-agnostic gates: the number of images per site must be a whole number, no site may receive two different elements, and the surviving sites must clear a hard-sphere packing floor. Mode: `unit_cell`.

    No absolute volume-per-site window is used, because a physical cell ranges from about 5.7 Å³ per atom for diamond to about 110 Å³ per atom for caesium; any fixed window would reject real crystals.

7. **Refusal.** If none of the above holds, the structure is not a periodic crystal. The module stops and reports the evidence — atoms, bounding box, the under-coordinated fraction, and a per-axis verdict with the period found and the score reached — then points at `XRD-Debye_Scattering.py`, which computes the pattern directly from the interatomic distances, needs no lattice at all, and keeps the crystallite-size broadening a periodic calculation throws away. There is no override: `allow_aperiodic` is retired and now only produces a warning saying so, because a pattern computed from a structure with no lattice is not a result worth producing.

The coordination shell used throughout is taken from the **first minimum of the radial distribution function** rather than a fixed multiple of the nearest-neighbour distance. A fixed multiple such as \(1.25\,r_1\) suits fcc, hcp, diamond and rocksalt, where \(r_2/r_1 = 1.41\) to \(1.63\), but straddles the first two shells of bcc, where \(r_2/r_1 = 1.155\). The first-minimum rule has no such dependence on crystal system.

#### Limitations

- **Folding** into a supplied cell (`lattice_mode = unit_cell`) assumes the cell axes are aligned with \(x,y,z\); it cannot be used for a cell built on directions such as \([111]\). Inference is what covers those, so a rotated or non-axis-aligned periodic cell no longer needs its box given explicitly — though supplying one is still faster and exact.

- An **axis-aligned crystalline cluster** has coordinates indistinguishable from those of a periodic cell, so it is accepted and folded. The calculation is then that of the infinite crystal: the Bragg peaks carry no crystallite-size broadening and the free surfaces contribute nothing. A warning is issued whenever a fold succeeds on a structure that still shows a substantial under-coordinated fraction.

### Input Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **I. Input & Lattice** |  |  |  |
| `xyz_file` | Path | Required | Input structure file. |
| `a`, `b`, `c` | Float | Optional | **Unit-cell** constants in Å — the crystallographic cell, never the supercell. A supercell box is read from the structure file or recovered internally from this cell; see `lattice_mode`. |
| `alpha`, `beta`, `gamma` | Float | Optional | **Unit-cell** angles in degrees. Supply them with `a`, `b`, `c` or omit all six; they are required for `lattice_mode = unit_cell` and for `coordinate_mode = fractional`. |
| `species_mode` | Enum | `chemical_symbols` | Species interpretation: `chemical_symbols`, `atomic_numbers`, or `lammps_types`. |
| `type_map` | Map | None | Mapping for `lammps_types`, for example `1:Cu,2:Zn`. |
| `coordinate_mode` | Enum | `cartesian` | Coordinate type stored in the input structure: `cartesian` or `fractional`. |
| `lattice_mode` | Enum | `auto` | Which lattice is given to pymatgen: `auto`, `simulation_box`, or `unit_cell`. See *Periodicity Assessment*. |
| `wrap_coords` | Bool | `yes` | Wrap fractional coordinates into \([0,1)\). |
| `check_duplicate_sites` | Bool | `yes` | In `unit_cell` mode, **merge** the folded images back onto their crystallographic sites. |
| `collapse_tol` | Float | `0.25` | Radius in Å within which folded images count as the same site. It must exceed the thermal displacement of a site. |
| `validate_proximity` | Bool | `yes` | Apply pymatgen's `Structure.DISTANCE_TOLERANCE` check and refuse a cell whose sites are closer than 0.5 Å. Above 5000 sites the same tolerance is checked with a neighbour tree, because pymatgen's own check builds \(N \times N\) arrays; above `max_proximity_check_atoms` it is skipped altogether. |
| `periodicity_surface_tol` | Float | `0.05` | A candidate box is accepted when it leaves fewer than this fraction of atoms under-coordinated. |
| `use_ase` | Bool | `yes` | Let ASE read a cell from the structure file when it carries no extended-XYZ `Lattice=` entry (LAMMPS dump or data, `POSCAR`, CIF, …). The first frame is used. |
| `cell_file` | Path | None | A second file whose box is applied to these coordinates, for example the relaxed data file a dump came from. |
| `infer_cell` | Bool | `yes` | Recover the box from the coordinates alone, at any orientation. This is what handles a cell built on directions such as \([111]\). |
| `max_proximity_check_atoms` | Int | `20000` | Above this many sites the proximity check is skipped entirely. |
| `cell_inference_tol` | Float | `0.15` | How close two atoms must sit, as a fraction of the nearest-neighbour distance \(r_1\), for a translation to count as mapping one onto the other. |
| `cell_inference_min_score` | Float | `0.90` | Fraction of atoms a candidate translation must map onto an atom of the **same species** before it is accepted as a lattice vector. |
| `cell_inference_reach` | Float | `1.8` | Radius, in units of \(r_1\), within which candidate translation vectors are generated. |
| `cell_inference_max_vectors` | Int | `24` | Stop after this many accepted translation vectors. |
| `cell_inference_max_residual` | Float | `0.10` | How far, in units of \(r_1\), the inferred cell may be from closing on the coordinates before it is rejected. |
| `axis_period_margin` | Float | `1.0` | Slab thickness, in units of the first-shell cutoff, used when scanning an axis for its repeat length. |
| `axis_period_min_score` | Float | `0.92` | Score an axis must reach before it counts as periodic. |
| `max_box_scan_atoms` | Int | `200000` | Upper size for the per-axis box scan. The scan builds a neighbour tree per trial period, so it is slow well before this limit. |
| `max_collapse_atoms` | Int | `20000` | Upper size for the image-merging step, which builds an \(N \times N\) distance matrix. Above this the images are left unmerged. |
| `repeats_integer_tol` | Float | `0.01` | Tolerance on the requirement that the number of images per site be a whole number. |
| **II. Calculation** |  |  |  |
| `wavelength` | Enum/Float | `CuKa` | Radiation source. Named values are resolved by pymatgen. `CuKa = 1.54184` Å is a K\(\alpha_1\)/K\(\alpha_2\) weighted average. |
| `two_theta_min` | Float | `10.0` | Minimum \(2\theta\) angle in degrees. |
| `two_theta_max` | Float | `90.0` | Maximum \(2\theta\) angle in degrees. |
| `reflection_cost_warn` | Float | `5e7` | Warn above this many structure-factor terms (reflections × atoms). |
| `max_reflection_cost` | Float | `1e10` | Refuse the run above this many terms. The number of reflections grows with the **cell volume** and each one costs a sum over every atom, so the work scales as volume × atoms — a large MD box is the worst case for this method. Use `XRD-ReciprocalSum` for a large periodic box, or `XRD-Debye_Scattering` if it has free surfaces anyway. |
| `scale_factor` | Float | `1.0` | Global intensity scaling factor. |
| `float_precision` | Int | `4` | Decimal places written to output files. |
| **III. Corrections** |  |  |  |
| `experimental_correction` | Bool | `no` | Convenience switch enabling Debye–Waller and instrumental broadening only when their corresponding values are supplied and the matching `apply_*` key is not set explicitly. In the GUI an unticked correction box is left unset, so this switch can still enable it. It never enables absorption automatically. |
| `apply_debye_waller` | Bool | `no` | Enable Debye–Waller thermal damping. |
| `debye_waller_factors` | String | None | Element-specific \(B_{\mathrm{iso}}\) values in Å\(^2\), for example `Zn=0.5,Cu=0.6`. They are checked against the structure only when `apply_debye_waller = yes`. |
| `apply_absorption` | Bool | `no` | Enable absorption correction. |
| `absorption_geometry` | Enum | `cylinder` | `cylinder`, `debye_scherrer`, `bragg_brentano_reflection`, `symmetric_transmission`, or `slab_attenuation_approx`. |
| `mu_linear_cm_inverse` | Float | None | Linear absorption coefficient \(\mu\) in cm\(^{-1}\). |
| `capillary_radius_cm` | Float | `0.05` | Capillary radius in cm for cylindrical Debye–Scherrer geometry. |
| `sample_thickness_cm` | Float | `0.1` | Sample thickness in cm for flat-plate geometries. |
| **IV. Broadening** |  |  |  |
| `apply_instrumental_broadening` | Bool | `no` | Enable instrumental Gaussian broadening. |
| `instrument_sigma_deg` | Float | `0.0` | Instrument Gaussian \(\sigma\) in degrees \(2\theta\). |
| `apply_plot_smoothing` | Bool | `no` | Enable cosmetic display smoothing. |
| `plot_smoothing_sigma_deg` | Float | `0.0` | Cosmetic Gaussian \(\sigma\) in degrees \(2\theta\). |
| `gaussian_width` | Float | `0.0` | Base Gaussian \(\sigma\), added in quadrature with other widths. |
| `gaussian_points` | Int | `1000` | Number of points in the continuous curve. |
| `curve_padding_deg` | Float | `0.0` | Additional plot/curve range outside the requested \(2\theta\) interval. |
| **V. Display Filtering** |  |  |  |
| `min_intensity_fraction` | Float | `0.005` | Minimum display intensity as fraction of maximum reflection intensity. |
| `min_intensity_percent` | Float | None | Minimum display intensity as percent of maximum reflection intensity. |
| `min_intensity` | Float | None | Legacy parameter interpreted as a percent. |
| `max_peaks` | Int | `0` | Maximum number of labelled/displayed reflections; `0` means all selected reflections. |
| `min_peak_separation` | Float | `0.0` | Minimum separation in degrees for display labels. The strongest reflection in a cluster is retained. |
| **VI. Normalisation & Output** |  |  |  |
| `normalize_mode` | Enum | `none` | `none`, `curve_max_100`, or `reflection_max_100`. |
| `normalize_max` | Float | `100.0` | Target maximum used by normalization modes. |
| `reflections_file` | Path | `xrd_reflections_full.txt` | Full calculated reflection list. |
| `peaks_file` | Path | `xrd_peaks_display.txt` | Filtered display-peak list. |
| `curve_file` | Path | `xrd_curve.txt` | Continuous Gaussian-broadened curve. |
| `save_cif` | Bool | `no` | Save the periodic structure that was used. In `simulation_box` mode this is the full supercell; in `unit_cell` mode it is the cell with the merged sites. By default it is reduced to the primitive cell, but **only when the reduction is lossless** — a box with vacancies, substitutions or thermal displacement keeps every atom. |
| `cif_content` | Enum | `primitive` | What the CIF holds: `primitive` (the smallest cell the structure reduces to, losslessly), `conventional` (the standard crystallographic setting), or `as_used` (every atom of the structure the pattern was computed from). |
| `cif_symprec` | Float | `0.01` | Symmetry tolerance in Å for the reduction. Larger values reduce more aggressively and can erase small displacements. |
| `cif_file` | Path | `structure.cif` | Filename for the saved CIF. |
| `make_plot` | Bool | `yes` | Generate the indexed XRD plot. |
| `show_plot` | Bool | `no` | Display the plot interactively. |
| `show_structure_info` | Bool | `yes` | Print the structure and peak-analysis summary described under *Outputs*. |
| `plot_filename` | Path | `xrd_plot.png` | Plot output file. |
| `plot_dpi` | Int | `150` | Plot resolution. |
| `plot_figsize` | List | `[12, 6]` | Plot size in inches. |
| `show_grid` | Bool | `yes` | Display plot grid. |
| `grid_alpha` | Float | `0.3` | Grid transparency. |
| `marker_at` | Enum | `smoothed` | Marker location: `smoothed` or `sticks`. |

---

## Notes

The core diffraction calculation of this module uses [Pymatgen’s XRDCalculator](https://pymatgen.org/pymatgen.analysis.diffraction.html#pymatgen.analysis.diffraction.xrd.XRDCalculator). Users can consult the corresponding documentation for detailed crystallographic background information.

For a finite nanoparticle, isolated cluster, liquid snapshot, or any structure with free surfaces, use the Debye scattering module instead; the Kinematical module assumes a periodic structure and will refuse an aperiodic one. A **periodic** cell containing defects — vacancies, substitutional disorder, strain, a grain boundary or a dislocation — is handled by this module: supply the simulation box, and every atom contributes to the structure factors.

---

## Part III: XRD-Debye Scattering

### Workflow Overview
The script transforms a finite set of atomic coordinates into a powder-averaged diffraction pattern using the Debye Scattering Equation (DSE).

The workflow is divided into the following stages:

1. **Atomic Form-Factor Calculation**: The script calculates X-ray scattering factors for each chemical species at every \(q\) value. It supports standard Cromer–Mann form factors and optional anomalous corrections \(f'\) and \(f''\).

2. **Computational Core**: The DSE is evaluated by either an exact pairwise sum or a distance-binned pair sum.

3. **Measurement Corrections and Normalization**: Optional Lorentz–polarization, instrumental broadening, scale factor, and display normalization are applied after intrinsic Debye scattering has been calculated.

4. **Optional Structural Diagnostics**: Partial RDFs and pair-distance distributions may be generated for structural analysis. These diagnostics do not enter the Debye scattering calculation.

### Mathematical Framework

The script is grounded in kinematic X-ray scattering from a finite atom set.

#### A. Scattering Vector (\(q\))

The relationship between scattering angle \(2\theta\) and scattering-vector magnitude \(q\) is:

$$
q
=
\frac{4\pi\sin\theta}{\lambda}.
$$

where \(\lambda\) is the radiation wavelength.

The Cromer–Mann variable is:

$$
s
=
\frac{\sin\theta}{\lambda}
=
\frac{q}{4\pi}.
$$

The current implementation passes \(q\) in Å\(^{-1}\) to `periodictable`, which internally uses the appropriate \(s=q/(4\pi)\) convention.

#### B. Debye Scattering Equation (DSE)

The coherent powder-averaged scattering intensity is:

$$
I(q)
=
\sum_i
|f_i(q)|^2
+
2
\sum_{i<j}
\operatorname{Re}
\left[
f_i(q)f_j^*(q)
\right]
\frac{\sin(qr_{ij})}{qr_{ij}}.
$$

where:

- \(f_i(q)\) is the scattering factor of atom \(i\).
- \(r_{ij}\) is the distance between atoms \(i\) and \(j\).
- \(f_j^*(q)\) is the complex conjugate of the scattering factor.
- The sinc kernel is:

    $$
    \operatorname{sinc}(qr)
    =
    \frac{\sin(qr)}{qr}.
    $$

For non-anomalous real form factors, the expression reduces to:

$$
I(q)
=
\sum_i f_i^2(q)
+
2\sum_{i<j}
f_i(q)f_j(q)
\frac{\sin(qr_{ij})}{qr_{ij}}.
$$

The code treats the \(q\rightarrow0\) limit safely:

$$
\lim_{q\rightarrow0}
\frac{\sin(qr)}{qr}
=
1.
$$

#### C. Complex Anomalous Scattering

When:

```text
apply_anomalous = yes
```

the atomic scattering factor is:

$$
f(q,\lambda)
=
f^0(q)
+
f'(\lambda)
+
if''(\lambda).
$$

The unordered-pair interference contribution is:

$$
2
\operatorname{Re}
\left[
f_i(q)f_j^*(q)
\right]
\operatorname{sinc}(qr_{ij}).
$$

User-supplied anomalous terms have priority:

```text
fprime_Cu = ...
fdouble_Cu = ...
```

If user values are absent, the code attempts to obtain anomalous data from the `periodictable` energy-dependent tables. In that convention, the table value \(f_1\) includes the forward-scattering \(Z\) term, so the anomalous real correction is:

$$
f'=f_1-Z,
$$

and:

$$
f''=f_2.
$$

If no anomalous data are available, the code uses:

$$
f'=0,
\qquad
f''=0,
$$

and prints a warning.

#### D. Binned Pair-Distance Debye Sum

The binned calculation is not an RDF reconstruction. It groups physical unordered atom pairs into narrow radial-distance bins.

The binned expression is:

$$
I(q)
=
\sum_a
N_a|f_a(q)|^2
+
2
\sum_{a\le b}
\operatorname{Re}
\left[
f_a(q)f_b^*(q)
\right]
\sum_k
N_{ab}(r_k)
\frac{\sin(q\bar r_k)}{q\bar r_k}.
$$

where:

- \(N_a\) is the number of atoms of species \(a\).
- \(N_{ab}(r_k)\) is the number of unordered \(a\)-\(b\) pairs in distance bin \(k\).
- \(\bar r_k\) is the mean pair distance within the bin.

The pairwise mode is exact for the included pair list. The binned mode is an approximation controlled by the radial bin width. It uses the mean distance in each bin, which reduces binning error compared with using only bin centers.

The binned method does not use:

- Number density.
- Cluster volume.
- Ideal-gas baseline.
- RDF normalization.
- Accessible-fraction correction.
- Lorch window.

Those quantities are not terms in the finite-configuration Debye scattering equation.

#### E. Debye–Waller Factor

For an isotropic crystallographic displacement parameter \(B\) in Å\(^2\), the atomic amplitude damping factor is:

$$
T(q)
=
\exp\left[
-\frac{Bq^2}{16\pi^2}
\right]
=
\exp(-Bs^2).
$$

The code applies this factor only when:

```text
apply_debye_waller = yes
```

For a pair contribution, the amplitude factors multiply:

$$
f_i(q)f_j^*(q)
\rightarrow
f_i(q)T_i(q)
\left[
f_j(q)T_j(q)
\right]^*.
$$

For a finite-temperature MD snapshot, applying a static Debye–Waller factor can double count thermal disorder already present in the coordinates. This should be treated as a modeling decision.

#### F. Measurement Corrections

Intrinsic Debye scattering and experimental measurement corrections are separate:

$$
I_{\mathrm{obs}}(2\theta)
=
\mathrm{Scale}
\times
L_p(2\theta)
\times
\left[
I_{\mathrm{DSE}}(2\theta)
\ast
R(2\theta)
\right].
$$

Here:

- \(L_p\) is an optional Lorentz–polarization factor.
- \(R\) is an instrumental-resolution function.
- `Scale` is `scale_factor`.

The Lorentz–polarization options are:

- With polarization:

    $$
    L_p(\theta)
    =
    \frac{
    1+\cos^2(2\theta)
    }{
    \sin^2\theta\cos\theta
    }.
    $$

- Lorentz only:

    $$
    L(\theta)
    =
    \frac{
    1
    }{
    \sin^2\theta\cos\theta
    }.
    $$

These factors are measurement-geometry models, not intrinsic parts of the DSE. Their use should match the experimental geometry.

The low-angle behavior may be numerically clipped through:

```text
LP_theta_min_clip_deg
LP_max_clip
```

#### G. Instrumental Broadening

A constant Gaussian instrumental width may be used through:

```text
instrument_sigma_deg
```

For angle-dependent Caglioti broadening, the code uses:

$$
\mathrm{FWHM}^2
=
U\tan^2\theta
+
V\tan\theta
+
W.
$$

The corresponding Gaussian standard deviation is:

$$
\sigma(\theta)
=
\frac{
\mathrm{FWHM}(\theta)
}{
2\sqrt{2\ln2}
}.
$$

The variable-width implementation distributes intensity from each source point using the width at that source point and an area-normalized Gaussian kernel.

#### H. March–Dollase Texture

March–Dollase texture correction is not applied to the isotropic Debye scattering pattern.

A March–Dollase correction depends on the angle between a reflection vector and a preferred-orientation axis. It is defined per reflection family \(hkl\). Since the DSE result is already an isotropic orientational average and has no individual \(hkl\) reflection vectors, a \(2\theta\)-only March–Dollase multiplier is not physically valid.

If:

```text
apply_march_dollase = yes
```

is requested, the code raises an explanatory error. Use the Kinematical module for reflection-resolved texture modeling.

### Key Features

- **Finite nonperiodic structures**: Suitable for nanoparticles, isolated clusters, amorphous structures, liquid snapshots, and extracted defective regions.

- **Pairwise mode**: Exact Debye scattering sum over the included unordered pair list.

- **Binned mode**: Fast pair-distance-binned approximation to the same DSE.

- **Automatic method selection**:
    - `pairwise` for small systems.
    - `binned` for larger systems.
    - `auto` selects according to `direct_threshold`.

- **Anomalous scattering**: Supports complex form factors.

- **Partial intensities**: Can report species-pair contributions. Like-species partial contributions include the corresponding self-scattering term.

- **Pair cutoff warning**: A user-supplied cutoff smaller than the maximum pair distance truncates the DSE and is reported as such.

- **Structural diagnostics**: RDF and pair-distance outputs are optional and never modify the calculated DSE intensity.

### Input Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **I. Input** |  |  |  |
| `xyz_file` | Path | Required | Input structure file. |
| `species_mode` | Enum | `chemical_symbols` | Species interpretation: `chemical_symbols`, `atomic_numbers`, or `lammps_types`. |
| `type_map` | Map | None | Mapping for `lammps_types`, for example `1:Cu,2:Zn`. |
| `wavelength` | Enum/Float | `CuKa` | Radiation source. `CuKa = 1.54184` Å in this module. |
| **II. Angular Grid** |  |  |  |
| `two_theta_min` | Float | `10.0` | Minimum \(2\theta\) in degrees. |
| `two_theta_max` | Float | `90.0` | Maximum \(2\theta\) in degrees. |
| `q_max` | Float | None | Alternative upper bound in Å\(^{-1}\). If both `q_max` and `two_theta_max` are supplied, `two_theta_max` takes precedence. |
| `n_points` | Int | `500` | Number of angular-grid points. |
| **III. Calculation Method** |  |  |  |
| `debye_method` | Enum | `auto` | `pairwise`, `binned`, or `auto`. `histogram` is accepted as a deprecated spelling of `binned`. |
| `direct_threshold` | Int | `2000` | Atom count below which `auto` selects pairwise calculation. |
| `bins_per_angstrom` | Float | `20` | Radial bin resolution used in `binned` mode. |
| `max_distance` | Float/String | `adaptive` | Maximum pair distance for binned calculation. Use `adaptive` for the exact largest pair distance. |
| `max_interaction_distance` | Float | None | Explicit pair cutoff for pairwise calculation. A smaller value truncates the DSE. |
| `max_pairs` | Int | `2000000000` | Pair-count safety cap. |
| `allow_large_pair_count` | Bool | `no` | Override the pair-count safety cap. |
| **IV. Memory** |  |  |  |
| `chunk_size` | Int | `2048` | Atom centres per neighbour-search chunk. Only used when the pair sum is truncated (a cutoff below the largest pair distance); it affects memory use and speed, not the result. |
| `pair_subchunk_size` | Int | `200000` | Number of pairs evaluated per reciprocal-space subchunk. |
| `max_pair_matrix_elements` | Int | `100000000` | Maximum \(n_q\times n_{\mathrm{pairs}}\) matrix size. |
| `max_distance_block_elements` | Int | `4000000` | Memory limit for direct pair-distance blocks. |
| **V. Physics** |  |  |  |
| `apply_anomalous` | Bool | `no` | Enable anomalous corrections \(f'\) and \(f''\). |
| `fprime_<El>` | Float | Table/default | User override for \(f'\) of an element. |
| `fdouble_<El>` | Float | Table/default | User override for \(f''\) of an element. |
| `apply_debye_waller` | Bool | `no` | Apply isotropic Debye–Waller damping. |
| `debye_waller_factors` | String | None | Element-specific \(B\) values in Å\(^2\), for example `Zn=0.5,Cu=0.6`. |
| `B_<El>` | Float | `0.0` | Per-element Debye–Waller parameter in Å\(^2\). |
| **VI. Measurement Corrections** |  |  |  |
| `experimental_correction` | Bool | `no` | Enables LP, instrumental broadening, and Debye–Waller damping, but only where `apply_LP`, `apply_instrumental_broadening` or `apply_debye_waller` is not set explicitly; an explicit value always wins. In the GUI an unticked correction box is left unset, so this switch still enables it. It does not automatically enable anomalous scattering. |
| `apply_LP` | Bool | `no` | Apply Lorentz–polarization correction. |
| `LP_variant` | Enum | `with_polarization` | `with_polarization`, `lorentz_only`, or `none`. |
| `LP_theta_min_clip_deg` | Float | `0.5` | Low-angle clipping threshold for LP evaluation. |
| `LP_max_clip` | Float | `10000` | Maximum LP value. |
| `apply_instrumental_broadening` | Bool | `no` | Apply instrumental broadening. |
| `instrument_sigma_deg` | Float | `0.05` | Constant Gaussian \(\sigma\) in degrees \(2\theta\). |
| `use_caglioti` | Bool | `no` | Use Caglioti angle-dependent FWHM. |
| `caglioti_U` | Float | `0.01` | Caglioti \(U\) coefficient. |
| `caglioti_V` | Float | `0.01` | Caglioti \(V\) coefficient. |
| `caglioti_W` | Float | `0.01` | Caglioti \(W\) coefficient. |
| `scale_factor` | Float | `1.0` | Global intensity scale factor. |
| **VII. Normalisation** |  |  |  |
| `normalize_intensity` | Bool | `no` | Normalize calculated intensity. |
| `normalize_max` | Float | `100.0` | Target maximum for normalized intensity. |
| **VIII. Diagnostics** |  |  |  |
| `compute_rdf` | Bool | `no` | Calculate partial RDFs for structural diagnostics. |
| `plot_rdf` | Bool | `no` | Plot partial RDFs. |
| `rdf_max_distance` | Float | Automatic | Maximum RDF distance. |
| `rdf_sample_size` | Int | `0` | Number of centers sampled for RDF; `0` uses all atoms. |
| `apply_accessible_fraction` | Bool | `no` | Apply finite-cluster boundary correction to reported RDFs only. |
| `rdf_access_dirs` | Int | `200` | Directions sampled for RDF accessible-fraction estimation. |
| `rdf_access_centers` | Int | `500` | Centers sampled for RDF accessible-fraction estimation. |
| `compute_pair_distribution` | Bool | `no` | Write pair-distance probability density and cumulative coordination diagnostic. |
| `pdf_subset_size` | Int | `5000` | Maximum atom count sampled for pair-distance diagnostics. |
| `pdf_bins` | Int | `100` | Number of pair-distance bins. |
| `plot_pdf` | Bool | `no` | Plot the pair-distance diagnostic. |
| `pdf_output` | Path | `pair_distance_distribution.txt` | Output file for the pair-distance diagnostic. |
| `random_seed` | Int | `0` | Random seed for reproducible subsampling. |
| **IX. Output & Plotting** |  |  |  |
| `output_pattern` | Path | `debye_pattern.txt` | Main diffraction pattern output. |
| `partial_output` | Path | `debye_partials.txt` | Species-pair partial-intensity output. |
| `output_dir` | Path | `.` | Directory for the partial-RDF output files (created if missing). |
| `compute_partial_intensities` | Bool | `yes` | Write species-resolved intensity contributions. |
| `make_plot` | Bool | `yes` | Generate main Debye pattern plot. |
| `plot_filename` | Path | `debye_plot.png` | Main plot output. |
| `plot_dpi` | Int | `300` | Plot resolution. |
| `show_plot` | Bool | `no` | Display figure interactively. |
| `plot_partials` | Bool | `no` | Plot partial intensity contributions. |
| `verbose` | Bool | `yes` | Print detailed diagnostics. |

---

## Summary of Use Cases

| Module | Best For |
| :--- | :--- |
| **XRD-ReciprocalSum** | Reciprocal-space calculations, periodic orthogonal simulation cells. |
| **XRD-Kinematical** | Indexed powder diffraction from periodic crystallographic structures — unit cells and periodic simulation boxes, including defective ones. |
| **XRD-Debye Scattering** | Finite nanoparticles, clusters, amorphous structures, liquid snapshots, and any aperiodic configuration with free surfaces. |

---

## Notes

To obtain highly accurate XRD data for defects, multiphase materials, and polycrystals, first analyze the structure using the **Structure Analyzer and Crystallographic Identification** module. By using phase-specific or structurally isolated files rather than the entire system at once, you can determine which reflections correspond to particular planes of each identified phase.