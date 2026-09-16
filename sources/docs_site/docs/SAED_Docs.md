# SAED Analysis Module

## Overview
This module simulates **Selected Area Electron Diffraction (SAED)** patterns from atomic structures. It employs a kinematic diffraction approximation to calculate the diffraction intensity in reciprocal space and generates visual representations of the patterns. This workflow is essential for linking atomic-scale simulations with experimental TEM (Transmission Electron Microscopy) observations.

The process consists of two stages:

1.  **Diffraction Calculation**: Computes the 3D reciprocal space intensity grid.
2.  **Pattern Visualization**: Slices the 3D data at the Ewald-sphere intersection to generate 2D diffraction images.

---

## Physics and Methodology

The diffraction intensity $I(\mathbf{k})$ at a reciprocal lattice vector $\mathbf{k}$ is calculated using the kinematic scattering theory:

$$
I(\mathbf{k}) =
\frac{1}{N}
\left|
\sum_{j=1}^{N}
f_j(s)
\exp\left(2\pi i\,\mathbf{k}\cdot\mathbf{r}_j\right)
\right|^2
$$

Where:

*   $N$ is the number of atoms; the calculated intensity is normalized by $N$.
*   $\mathbf{r}_j$ is the Cartesian position of atom $j$.
*   $f_j(s)$ is the atomic scattering factor for electrons.
*   $s=\sin\theta/\lambda=|\mathbf{k}|/2$ is the scattering-factor variable.

The electron scattering factor is evaluated as:

$$
f_j(s)
=
\sum_{C=1}^{5}
a_C
\exp(-b_Cs^2)
$$

**References:**

*   (Coleman) Coleman, Spearot, Capolungo, *MSMSE*, 21, 055020 (2013).
*   (Brown) Brown et al. *International Tables for Crystallography Volume C: Mathematical and Chemical Tables*, 554-95 (2004).
*   (Fox) Fox, O’Keefe, Tabbernor, *Acta Crystallogr. A*, 45, 786-93 (1989).

### Automated Cell & Boundary Detection
The module includes features to handle various input formats (e.g., standard XYZ files which often lack cell information):

*   **Automatic Cell Inference**: If cell information is unavailable, the code may estimate an orthogonal coordinate-span box from the global minimum and maximum atomic positions ($\mathbf{r}_{max} - \mathbf{r}_{min}$). For periodic calculations, provide the actual simulation cell; the inferred box is not a physical substitute for the periodic cell.
*   **Periodic Boundary Conditions (PBC)**: Users can explicitly set PBCs. In automatic mode, reciprocal mesh spacing is determined from the simulation-cell dimensions:

    $$
    \Delta k_i
    =
    \frac{c_i}{L_i}.
    $$

### Mesh Size Limit
The mesh holds every node with $|\mathbf{k}| < k_{max}$ (and, for a nonzero `zone`, inside the Ewald-sphere shell), with $N_i=\lceil k_{max}/\Delta k_i\rceil$ nodes on each side of the origin. Because $\Delta k_i=c_i/L_i$, its size grows as $(k_{max}L/c)^3$. Before the mesh is built, the number of nodes is estimated as

$$
N_{\mathrm{est}}
=
\frac{\pi}{6}
\prod_{i=1}^{3}\left(2N_i+1\right)
\min\left(1,\ \frac{6\delta}{k_{max}}\right),
$$

where the last factor is dropped for `zone = 0,0,0`. If $N_{\mathrm{est}}$ exceeds `max_mesh_nodes`, the run stops with an error that reports the estimated node count and memory (about 20 bytes per node) and how to reduce them: set a zone axis, increase `c`, or reduce `kmax`. A full reciprocal volume on a fine mesh is the usual cause: for `example_files/tem/screw.xyz` (box ≈ 30 Å), `zone = 0,0,0` with `c = 0.05` gives $N_{\mathrm{est}}\approx 4.7\times10^{9}$ nodes (≈ 94 GB) and is refused, while `zone = 0,0,1` with `c = 0.2` gives $N_{\mathrm{est}}\approx 2.6\times10^{6}$. The GUI default `c = 0.025` on the same box gives $N_{\mathrm{est}}\approx 1.2\times10^{9}$ (≈ 23 GB), which is why the GUI ships `max_mesh_nodes = 1500000000`: the fine mesh is deliberate, so the guard is set above it rather than stopping it. Lower the limit if that is not memory you have.


---

## Part I: Diffraction Calculation

This step generates a standard VTK file containing the 3D diffraction intensity.

### Input Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `structure_file` | File Path | Required | Path to the atomic structure (XYZ, LAMMPS data, etc.). |
| `vtk_base` | String | None | Base filename for VTK output. The output is written as `<vtk_base>_<vtk_index>.vtk`. The value is taken literally (no number or boolean conversion). |
| `vtk_index` | Integer | `0` | Integer appended to the VTK output filename. |
| `wavelength` | Float | 0.0251 | Electron wavelength in Ångstroms (e.g., 0.0251 Å for 200 keV). |
| `atom_types` | List | Required | List of elements in the file (e.g., `Al, O`). Must match the structure. |
| `kmax` | Float | 1.70 | Maximum reciprocal distance ($Å^{-1}$) to compute. Determines the Field of View. |
| `drewald` | Float | 0.005 | Half-thickness $\delta$ of the Ewald-sphere shell in $Å^{-1}$. For a nonzero `zone`, reciprocal nodes satisfying $R_{\mathrm{Ewald}}-\delta < r < R_{\mathrm{Ewald}}+\delta$ are retained. |
| `zone` | Vector | 1,0,0 | Zone axis orientation. Use `0,0,0` to compute the full 3D reciprocal volume. |
| `c` | Vector | 1,1,1 | Reciprocal-mesh scaling factors. In automatic mode, $\Delta k_i=c_i/L_i$. In manual mode, $\Delta k_i=c_i$ in $Å^{-1}$. Lower values create a finer mesh and increase computational cost. The module default is `1,1,1`, the LAMMPS default; the GUI ships `0.025,0.025,0.025`, which is much finer and needs a correspondingly high `max_mesh_nodes`. |
| `manual` | Boolean | False | If `True`, sets $\Delta k_i=c_i$ independently of the simulation cell. For non-periodic clusters. |
| `prd` | Vector | None | Box dimensions in Å. In manual mode, the reciprocal mesh is determined by `c`; `prd` does not change $\Delta k_i$. |
| `pbc` | Boolean List | 1,1,1 | Periodic flags ($x$, $y$, $z$). When supplied in the input file, these values override periodicity information read from the structure file. Set to `0` for non-periodic directions. |
| `engine` | Enum | `auto` | Structure-factor kernel: `auto` (numba if installed, else numpy), `numba` (multi-threaded), `numpy` (vectorized, no extra dependency) or `reference` (literal per-point transcription; slowest, for auditing). All engines give identical results for every physically significant intensity. The command-line option `--engine` overrides this key. |
| `chunk_size` | Integer | `65536` | Reciprocal mesh points per kernel chunk. Lower it to reduce memory use. |
| `mesh_engine` | Enum | `vectorized` | Reciprocal-mesh construction: `vectorized` or `reference`. |
| `mesh_check` | Boolean | `False` | Cross-checks the vectorized mesh against the reference construction (also `--mesh-check`). |
| `max_mesh_nodes` | Integer | `200000000` | Upper limit on the estimated number of reciprocal-mesh nodes (see [Mesh Size Limit](#mesh-size-limit)). Above it the run stops before the mesh is built. `0` disables the check. The GUI ships `1500000000` to match its finer default `c`. |
| `output_file` | File Path | None | Optional plain-text intensity listing (`i j k intensity`, one line per mesh point). Written only when set; it can become very large on fine meshes. |
| `echo` | Boolean | `False` | Print progress and diagnostic information. |

---

## Part II: Visualization

This step converts the 3D VTK output into a high-quality 2D image (PNG, JPEG, TIFF, BMP, PPM, RGB or PostScript). It performs a spherical slice operation to simulate the intersection of the Ewald sphere with the reciprocal lattice.

### Input Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `vtk_file` | File Path | Required | The VTK output file from the diffraction calculation step. |
| `output_file` | Filename | `saed.png` | Output image file. The extension selects the format: `.png` (also used when there is no extension), `.jpg`/`.jpeg`, `.tif`/`.tiff`, `.bmp`, `.ppm`, `.rgb` or `.ps`; other extensions are rejected. The image is saved under exactly this name. |
| `visit_path` | File Path | Optional | Path to the VisIt executable (quotes allowed). If omitted or not found, VisIt is auto-detected, e.g. `visit` on the PATH. |
| `iso_lower` | Float | `0` | Lower Isovolume threshold. Use `0` to remove VTK ghost values written as `-1`. |
| `iso_upper` | Float | `1e+37` | Upper Isovolume threshold. |
| `pseudocolor_min` | Float | `1` | Lower intensity value for the logarithmic pseudocolor scale. |
| `pseudocolor_max` | Float | Automatic | Upper intensity value for the logarithmic pseudocolor scale. Leave unset to let VisIt choose the maximum. |
| `view_normal` | Vector | `-1,0,0` | Normal vector for the camera view. |
| `view_up` | Vector | `0,1,0` | "Up" vector for image orientation. |
| `sphere_radius` | Float | `39.84063` | Radius of the Ewald sphere, $1/\lambda$ in $Å^{-1}$. The default corresponds to $\lambda=0.0251$ Å; change it together with `wavelength` and `sphere_origin`. |
| `sphere_origin` | Vector | Depends on wavelength and zone axis | Center of the Ewald sphere slice. <br> Example for $\lambda=0.0251$ along x-axis: `39.84063, 0, 0` <br> `sphere_origin = 39.84063, 0, 0` |
| `resolution` | Integer | `1200` | Width and height of the output image in pixels. |
| `show_3d_axes` | Boolean | `False` | Toggles 3D axes in the final image. |
| `show_2d_axes` | Boolean | `False` | Toggles 2D axes in the final image. |
| `show_user_info` | Boolean | `False` | Toggles user-information annotation. |
| `show_database_info` | Boolean | `False` | Toggles database-information annotation. |
| `show_legend` | Boolean | `True` | Toggles the pseudocolor intensity legend. |

---

## Notes
This module generally follows the methodology of the [LAMMPS compute saed](https://docs.lammps.org/compute_saed.html) command. Users can consult the corresponding LAMMPS documentation for more detailed background information.