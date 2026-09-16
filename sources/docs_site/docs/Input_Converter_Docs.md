# Input Converter and Format Manager

## 1. Introduction
The **Input Converter** is a versatile tool designed to bridge the gap between various computational materials science software packages. Its primary function is to convert atomic structure files into formats compatible with LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) and visualization tools like Ovito. 

It automatically handles tasks such as:

*   **Orthogonalization** of triclinic simulation cells.
*   Assignment of **Charge** and **Spin** degrees of freedom for **reactive MD or spin-lattice dynamics**.

## 2. Particle Definitions and LAMMPS Formats
The tool supports three distinct "modes" corresponding to different physical models in Molecular Dynamics (MD).

| Mode | Description | LAMMPS Format | Columns in `.lmp` |
| :--- | :--- | :--- | :--- |
| **Classic** | Classical MD with neutral point particles. | `atom_style atomic` | `ID type x y z` |
| **Charge** | MD with explicit Coulombic interactions (ions). | `atom_style charge` | `ID type charge x y z` |
| **Spin** | Magnetic MD with atomic spin moments. | `atom_style spin` | `ID type x y z sx sy sz scalar_spin` |

*   **Atomic Spin**: In `spin` mode, the script adds 4 extra columns. `sx, sy, sz` represent the direction of the spin vector, and `scalar_spin` represents the magnitude (in Bohr magnetons).
*   **Initialization**: All `charge` and `spin` values are initially set to **zero** for convenience. Users can assign specific charge and spin values in their LAMMPS input script using the `set` command.

## 3. Output Files
The script generates different outputs based on the requested filename extension: `.lmp`, `.data`, `.lammps` (LAMMPS data); `.dump`, `.lammpstrj` (LAMMPS dump); `.xyz`, `.extxyz` (extended XYZ); `.cif`; `.cfg`; `.xsf`; `.pdb`, `.ent`; and `.vasp`, `.poscar`, `.contcar` (VASP). Any other extension, including extension-less names such as `POSCAR`, is written as extended XYZ. Input structures are read with the **Atomic Simulation Environment (ASE)** backend (LAMMPS files with a built-in reader); see the [ASE IO documentation](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) for readable formats.

Specific details for key formats:

1.  **LAMMPS Data (`.lmp`, `.data`)**:
    *   Header containing `N atoms`, `N atom types`.
    *   Box bounds: `xlo xhi`, `ylo yhi`, `zlo zhi` (and `xy xz yz` for triclinic).
    *   `Masses` section: Inferred from periodic table based on element symbols.
    *   `Atoms` section: The coordinate data in the selected format (Classic/Charge/Spin), labelled `Atoms # atomic`, `Atoms # charge` or `Atoms # spin` so that OVITO and ASE can detect the atom style (ASE has no reader for the spin style).

2.  **LAMMPS Dump (`.dump`, `.lammpstrj`)**:
    *   Written when the input contains multiple frames (a requested `.lmp`/`.data` name is then changed to `.dump`), or whenever the output name ends in `.dump`/`.lammpstrj`, even for a single frame.
    *   Standard LAMMPS dump format with `ITEM: TIMESTEP`, `ITEM: NUMBER OF ATOMS`, `ITEM: BOX BOUNDS`, `ITEM: ATOMS ...`. Atom types are stored as numbers, without element names; triclinic cells use the `ITEM: BOX BOUNDS xy xz yz pp pp pp` form.

3.  **Extended XYZ (`.xyz`, `.extxyz`)**:
    *   Compatible with **Ovito**.
    *   Includes Lattice string and Properties line (e.g., `Properties=species:S:1:pos:R:3`).
