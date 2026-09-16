#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
Selected Area Electron Diffraction computes electron diffraction intensity on a mesh of reciprocal lattice nodes.
"""
import sys
import math
import numpy as np
import argparse
from datetime import datetime
import time
import os

from common_utils import parse_input_file, validate_params

try:
    from ase.io import read
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

from saed_consts import SAED_MAX_TYPE, SAED_TYPE_LIST, ASFSAED

from vtk_writer import write_saed_vtk_from_compute

ASFSAED_NP = np.asarray(ASFSAED, dtype=np.float64)       

_DEFAULT_CHUNK = 65536

ENGINES = ("auto", "numba", "numpy", "reference")

WARN = "[WARN]"


def sanitize_params(params):
    if params is None:
        return {}
    p = dict(params)
    for prefix in ('saed.', 'compute_saed.', 'basic.'):
        for key in list(p.keys()):
            if isinstance(key, str) and key.startswith(prefix):
                new_key = key[len(prefix):]
                p.setdefault(new_key, p[key])
                del p[key]
    return p


def as_bool(params, key, default=False):
    v = params.get(key, default)
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    return str(v).strip().lower() in ('yes', 'true', 'y', '1', 'on')


def as_float(params, key, default):
    v = params.get(key, default)
    if isinstance(v, (list, tuple)):
        v = v[0] if v else default
    try:
        return float(v)
    except Exception:
        return float(default)


def as_int(params, key, default):
    try:
        return int(as_float(params, key, default))
    except Exception:
        return int(default)


def as_str(params, key, default=''):
    v = params.get(key, default)
    if isinstance(v, (list, tuple)):
        v = v[0] if v else default
    return str(v).strip()


def as_list(params, key, default, cast=None, length=None, pad=None):
    v = params.get(key, None)
    if v is None:
        items = list(default)
    elif isinstance(v, (list, tuple)):
        items = list(v)
    else:
        items = [x.strip() for x in str(v).replace(',', ' ').split() if x.strip()]
    if cast is not None:
        out = []
        for item in items:
            try:
                out.append(cast(item))
            except Exception:
                out.append(item)
        items = out
    if length is not None:
        if len(items) < length:
            items = items + [pad] * (length - len(items))
        items = items[:length]
    return items


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True, fastmath=False)
    def _kernel_numba(K_chunk, SinTheta_lambda, xlocal, typelocal, a_c, b_c):
        TWO_PI = 2.0 * np.pi
        nk = K_chunk.shape[0]
        na = xlocal.shape[0]
        nt = a_c.shape[0]
        Freal = np.zeros(nk, dtype=np.float64)
        Fimag = np.zeros(nk, dtype=np.float64)
        for n in prange(nk):
            K0 = K_chunk[n, 0]
            K1 = K_chunk[n, 1]
            K2 = K_chunk[n, 2]
            s = SinTheta_lambda[n]
            f = np.zeros(nt)
            for ii in range(nt):
                acc = 0.0
                for C in range(5):
                    acc += a_c[ii, C] * math.exp(-1 * b_c[ii, C] * s * s)
                f[ii] = acc
            Fatom1 = 0.0
            Fatom2 = 0.0
            for ii in range(na):
                typei = typelocal[ii]
                inners = TWO_PI * (K0 * xlocal[ii, 0]
                                   + K1 * xlocal[ii, 1]
                                   + K2 * xlocal[ii, 2])
                Fatom1 += f[typei] * math.cos(inners)
                Fatom2 += f[typei] * math.sin(inners)
            Freal[n] = Fatom1
            Fimag[n] = Fatom2
        return Freal, Fimag
else:
    _kernel_numba = None


def _kernel_numpy(K_chunk, SinTheta_lambda, xlocal, typelocal, a_c, b_c):
    TWO_PI = 2.0 * np.pi
    s = SinTheta_lambda[None, None, :]
    exponent = (-1.0 * b_c[:, :, None] * s) * s              # (nt, 5, nk)
    f_table = np.sum(a_c[:, :, None] * np.exp(exponent), axis=1)   # (nt, nk)
    phases = TWO_PI * (xlocal @ K_chunk.T)                   # (na, nk)
    fj = f_table[typelocal, :]                               # (na, nk)
    Freal = np.sum(fj * np.cos(phases), axis=0)
    Fimag = np.sum(fj * np.sin(phases), axis=0)
    return Freal, Fimag


def _kernel_reference(K_chunk, SinTheta_lambda, xlocal, typelocal, a_c, b_c):
    TWO_PI = 2.0 * math.pi
    nk = K_chunk.shape[0]
    nt = a_c.shape[0]
    Freal = np.empty(nk, dtype=np.float64)
    Fimag = np.empty(nk, dtype=np.float64)
    f = np.zeros(nt, dtype=np.float64)
    for n in range(nk):
        s = SinTheta_lambda[n]
        for ii in range(nt):
            acc = 0.0
            for C in range(5):
                acc += a_c[ii, C] * math.exp(-1 * b_c[ii, C] * s * s)
            f[ii] = acc
        inners = TWO_PI * np.dot(xlocal, K_chunk[n])
        fj = f[typelocal]
        Freal[n] = np.sum(fj * np.cos(inners))
        Fimag[n] = np.sum(fj * np.sin(inners))
    return Freal, Fimag


def resolve_engine(requested):
    eng = str(requested).strip().lower()
    if eng not in ENGINES:
        raise ValueError("Unknown engine %r; choose one of %s"
                         % (requested, ", ".join(ENGINES)))
    if eng == "auto":
        return "numba" if NUMBA_AVAILABLE else "numpy"
    if eng == "numba" and not NUMBA_AVAILABLE:
        return "numpy"
    return eng


_KERNELS = {"numpy": _kernel_numpy, "reference": _kernel_reference}

class ComputeSAED:

    def __init__(self, input_file):
        """
        Initialize the SAED computation from input file.
        """
        self.me = 0
        self.nprocs = 1
        self.read_input(input_file)
        self.validate_inputs()
        self.read_structure_file()
        if not self.manual:
            self.validate_cell_orthogonality()
            if hasattr(self, 'cell') and self.cell is not None:
                dimension = sum(1 for i in range(3) if np.linalg.norm(self.cell[i]) > 1e-10)
                if dimension == 2:
                    raise ValueError("Compute SAED does not work with 2d structures")
        self.setup_reciprocal_mesh()
        self.initialize_arrays()
        
    @classmethod
    def from_input(cls, input_file, engine=None, mesh_check=None,
                   chunk_size=None):
      
        obj = cls.__new__(cls)
        obj.me = 0
        obj.nprocs = 1
        obj.read_input(input_file)
        if engine is not None:
            obj.engine_requested = engine
            obj.engine = resolve_engine(engine)
        if mesh_check is not None:
            obj.mesh_check = bool(mesh_check)
        if chunk_size is not None:
            obj._compute_chunk_size = int(chunk_size)
        obj.validate_inputs()
        obj.read_structure_file()
        if not obj.manual:
            obj.validate_cell_orthogonality()
            if getattr(obj, "cell", None) is not None:
                dimension = sum(1 for i in range(3)
                                if np.linalg.norm(obj.cell[i]) > 1e-10)
                if dimension == 2:
                    raise ValueError("Compute SAED does not work with 2d structures")
        obj.setup_reciprocal_mesh()
        obj.initialize_arrays()
        return obj

    def read_input(self, input_file):

        if self.me == 0:
            print(f"Reading input from {input_file}")
            
        try:
            params = sanitize_params(
                {str(k).strip().lower(): v
                 for k, v in parse_input_file(input_file).items()})
        except Exception as e:
            print(f"Error reading input file: {e}")
            sys.exit(1)

        if not validate_params(params, ['atom_types'], 'compute_saed'):
            raise ValueError("Missing 'atom_types' parameter in input file")

        self.structure_file = as_str(params, 'structure_file', 'structure.xyz')
        self.output_file = as_str(params, 'output_file', 'saed_results.txt')
        self.lambda_val = as_float(params, 'wavelength', 0.0251)
        self.atom_types = as_list(params, 'atom_types', ['Al', 'O'], cast=str)

        self.Kmax = as_float(params, 'kmax', 1.70)
        self.Zone = as_list(params, 'zone', [1.0, 0.0, 0.0], cast=float,
                            length=3, pad=0.0)
        self.dR_Ewald = as_float(params, 'drewald', 0.01 / 2)
        self.c = as_list(params, 'c', [1.0, 1.0, 1.0], cast=float,
                         length=3, pad=1.0)
        self.manual = as_bool(params, 'manual', False)
        self.echo = as_bool(params, 'echo', False)
        self.prd = params.get('prd', None)
        self.vtk_base = as_str(params, 'vtk_base', '') or None
        self.vtk_index = as_int(params, 'vtk_index', 0)

        self.use_ase = as_bool(params, 'use_ase', True)
        raw_type_map = params.get('type_map', None)
        if raw_type_map is not None:
            self.type_map = self._parse_type_map(raw_type_map)
        else:
            self.type_map = {}

        self.engine_requested = as_str(params, 'engine', 'auto')
        self.engine = resolve_engine(self.engine_requested)
        self._compute_chunk_size = as_int(params, 'chunk_size', _DEFAULT_CHUNK)
        if self._compute_chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        self.mesh_engine = as_str(params, 'mesh_engine', 'vectorized').lower()
        if self.mesh_engine not in ('vectorized', 'reference'):
            raise ValueError("mesh_engine must be 'vectorized' or 'reference'")
        self.mesh_check = as_bool(params, 'mesh_check', False)
        self.max_mesh_nodes = as_int(params, 'max_mesh_nodes', 200_000_000)

        self.pbc = [bool(int(x))
                    for x in as_list(params, 'pbc', [1, 1, 1], cast=int,
                                     length=3, pad=1)]

        if self.me == 0 and self.echo:
            print(f"Periodic boundaries: {self.pbc}")

        if self.manual:
            self.prd = as_list(params, 'prd', [], cast=float,
                               length=3, pad=0.0) or None
            if params.get('prd', None) is None:
                self.prd = None
        else:
            self.prd = None
        
        if self.me == 0 and self.echo:
            print("\n=== Input Parameters ===")
            print(f"Structure file: {self.structure_file}")
            print(f"Output file: {self.output_file}")
            print(f"Wavelength: {self.lambda_val} Å")
            print(f"Atom types: {', '.join(self.atom_types)}")
            print(f"Kmax: {self.Kmax}")
            print(f"Zone axis: {self.Zone}")
            print(f"dR_Ewald: {self.dR_Ewald}")
            print(f"c parameters: {self.c}")
            print(f"Manual mode: {self.manual}")
            print(f"Engine: {self.engine}"
                  + ("" if self.engine == self.engine_requested
                     else f" (requested '{self.engine_requested}')")
                  + ("  [bit-identical to LAMMPS]" if self.engine == "numba" else ""))
            if self.manual:
                print(f"Box dimensions (prd): {self.prd if self.prd is not None else 'Will be inferred from atomic positions'}")
            else:
                print("Box dimensions: Will be inferred from structure file (ignoring any prd parameter)")
    
    def setup_reciprocal_mesh(self):

        if self.me == 0 and self.echo:
            print("\n--- Setting up reciprocal space mesh ---")

        if hasattr(self, 'cell') and self.cell is not None:
            off_diagonal_sum = abs(self.cell[0,1]) + abs(self.cell[0,2]) + abs(self.cell[1,0]) + \
                              abs(self.cell[1,2]) + abs(self.cell[2,0]) + abs(self.cell[2,1])
            if off_diagonal_sum > 1e-10:
                raise ValueError("Compute SAED does not work with triclinic cells. "
                               "Use orthogonal simulation box or enable manual mode.")
        
        self.prd_inv = [0.0, 0.0, 0.0]
        
        if not self.manual:
            if not hasattr(self, 'cell') or self.cell is None or np.allclose(self.cell, 0.0):
                if self.echo and self.me == 0:
                    print("Warning: No valid cell dimensions found in structure file.")
                if hasattr(self, 'positions') and self.positions is not None:
                    if self.echo and self.me == 0:
                        print("Inferring box dimensions from atomic positions...")
                    self.prd = self.infer_box_dimensions(self.positions)
                    self.cell = np.diag(self.prd)
                    if self.echo and self.me == 0:
                        print(f"✓ Using inferred box dimensions: {self.prd}")
                    box_dims = self.prd.copy()
                else:
                    raise ValueError("No cell dimensions available and no atomic positions for inference. "
                                   "Either provide a structure file with cell information or enable manual mode.")
            else:
                box_dims = [np.linalg.norm(self.cell[i]) for i in range(3)]
                invalid_dims = [i for i, dim in enumerate(box_dims) if dim <= 1e-10]
                if invalid_dims:
                    if self.echo and self.me == 0:
                        print(f"Warning: Zero/invalid cell dimensions detected in directions {invalid_dims}.")
                    if hasattr(self, 'positions') and self.positions is not None:
                        inferred_dims = self.infer_box_dimensions(self.positions)
                        for i in invalid_dims:
                            box_dims[i] = inferred_dims[i]
                        
                        self.cell = np.diag(box_dims)
                        self.prd = box_dims
                        if self.echo and self.me == 0:
                            print(f"✓ Corrected box dimensions using inference: {box_dims}")
                    else:
                        raise ValueError(f"Invalid cell dimensions in directions {invalid_dims} and no atomic positions available for inference.")
                
                self.prd = box_dims
            
            periodic_count = sum(1 for i in range(3) if self.pbc[i])
            if periodic_count == 0:
                raise ValueError("Compute SAED must have at least one periodic boundary unless manual spacing specified")
            
            ave_inv = 0.0
            for i in range(3):
                if self.pbc[i] and box_dims[i] > 0:
                    self.prd_inv[i] = 1.0 / box_dims[i]
                    ave_inv += self.prd_inv[i]
            
            ave_inv = ave_inv / periodic_count
            
            for i in range(3):
                if not self.pbc[i]:
                    self.prd_inv[i] = ave_inv
            
            if self.me == 0 and self.echo:
                print(f"Before final prd assignment: {self.prd if hasattr(self, 'prd') else 'Not set'}")

            if self.me == 0 and self.echo:
                print(f"After final prd assignment (unchanged): {self.prd}")
        
        if self.manual:
            for i in range(3):
                self.prd_inv[i] = 1.0

            if self.prd is None and hasattr(self, 'positions') and self.positions is not None:
                self.prd = self.infer_box_dimensions(self.positions)

            if self.echo and self.me == 0:
                print("Manual mode enabled: dK = c exactly (LAMMPS semantics).")
                print(f"  c (= dK) : {self.c} Å⁻¹")
                print(f"  box dims : {self.prd} (reported only, not used for dK)")
        
        self.dK = [0.0, 0.0, 0.0]
        self.Knmax = [0, 0, 0]
        
        for i in range(3):
            self.dK[i] = self.prd_inv[i] * self.c[i]
            if self.dK[i] <= 0:
                raise ValueError(f"Invalid reciprocal spacing in direction {i}: {self.dK[i]}")
            self.Knmax[i] = int(math.ceil(self.Kmax / self.dK[i]))
        
        self.R_Ewald = 1.0 / self.lambda_val

        if (abs(self.Zone[0]) < 1e-10 and abs(self.Zone[1]) < 1e-10 and abs(self.Zone[2]) < 1e-10):
            pass
        else:
            zone_norm_sq = self.Zone[0]*self.Zone[0] + self.Zone[1]*self.Zone[1] + self.Zone[2]*self.Zone[2]
            
            if zone_norm_sq < 1e-20:
                if self.echo and self.me == 0:
                    print("Warning: Zone axis has extremely small magnitude. Treating as zero vector.")
                self.Zone = [0.0, 0.0, 0.0]
            else:
                zone_norm = math.sqrt(zone_norm_sq)
                Rnorm = self.R_Ewald / zone_norm
                self.Zone = [
                    self.Zone[0] * Rnorm,
                    self.Zone[1] * Rnorm, 
                    self.Zone[2] * Rnorm
                ]

        self._check_mesh_size()
        self.count_reciprocal_points()
        
        if self.me == 0 and self.echo:
            print(f"Reciprocal space setup complete:")
            print(f"  Box dimensions: {[float(x) for x in self.prd] if getattr(self, 'prd', None) is not None else 'Not determined'}")
            print(f"  Periodic boundaries: {self.pbc}")
            print(f"  dK spacing: {[float(x) for x in self.dK]}")
            print(f"  Knmax: {self.Knmax}")
    
    def _check_mesh_size(self):

        if self.max_mesh_nodes <= 0:
            return

        grid = 1
        for i in range(3):
            grid *= (2 * self.Knmax[i] + 1)

        full_space = (abs(self.Zone[0]) < 1e-10 and abs(self.Zone[1]) < 1e-10
                      and abs(self.Zone[2]) < 1e-10)
        if full_space:
            est = int(grid * math.pi / 6.0)
        else:
            shell = min(1.0, 6.0 * self.dR_Ewald / max(self.Kmax, 1e-12))
            est = int(grid * (math.pi / 6.0) * shell)

        if est <= self.max_mesh_nodes:
            return

        gb = est * (3 * 4 + 8) / 1e9
        msg = [
            "",
            "  " + "!" * 68,
            "  ! ERROR: the requested reciprocal-space mesh is far too large.",
            f"  !   dK      : {self.dK[0]:.6g}, {self.dK[1]:.6g}, {self.dK[2]:.6g} 1/Angstrom",
            f"  !   Knmax   : {self.Knmax}",
            f"  !   nodes   : ~{est:,} (limit max_mesh_nodes = {self.max_mesh_nodes:,})",
            f"  !   memory  : ~{gb:.1f} GB just for the mesh + intensity arrays",
            "  ! dK = c / L, so the mesh grows as (Kmax * L / c)^3.",
            f"  !   your box L : {self.prd[0]:.3f}, {self.prd[1]:.3f}, {self.prd[2]:.3f} Angstrom",
            f"  !   your c     : {self.c[0]:g}, {self.c[1]:g}, {self.c[2]:g}",
            f"  !   your Kmax  : {self.Kmax:g}",
            "  ! Fix by doing one or more of:",
        ]
        if full_space:
            msg.append("  !   (a) set a zone axis (e.g. zone = 0, 0, 1). zone = 0,0,0 explores")
            msg.append("  !       the FULL 3d reciprocal space and is ~100x more nodes.")
        msg += [
            "  !   (b) increase c (coarser spacing). c = 1,1,1 is the LAMMPS default;",
            "  !       c = 0.05 is only sensible for large boxes with a zone axis.",
            "  !   (c) reduce Kmax.",
            "  !   (d) raise the limit deliberately with 'max_mesh_nodes = <n>' in the",
            "  !       input file if you really do have the memory for it.",
            "  " + "!" * 68,
            "",
        ]
        raise ValueError(chr(10).join(msg))

    def infer_box_dimensions(self, positions):
        global_min = np.min(positions, axis=0)
        global_max = np.max(positions, axis=0)
        lengths = global_max - global_min
        min_box_size = 10.0 
        lengths = np.maximum(lengths, min_box_size)
        
        if self.me == 0:
            clamped = [i for i in range(3)
                       if (global_max[i] - global_min[i]) < min_box_size]
            print("")
            print("  " + "!" * 68)
            print("  ! WARNING: no simulation cell in the structure file.")
            print("  ! Box lengths were GUESSED from the atom extent (max - min).")
            print("  ! Reciprocal spacing is dK = c / L, so an incorrect L rescales")
            print("  ! every K coordinate in the VTK output without any visible error.")
            print(f"  !   guessed L : Lx={lengths[0]:.4f}  Ly={lengths[1]:.4f}  Lz={lengths[2]:.4f} Angstrom")
            print(f"  !   extent    : X[{global_min[0]:.3f}, {global_max[0]:.3f}]  "
                  f"Y[{global_min[1]:.3f}, {global_max[1]:.3f}]  "
                  f"Z[{global_min[2]:.3f}, {global_max[2]:.3f}]")
            if clamped:
                axes = ", ".join("xyz"[i] for i in clamped)
                print(f"  !   NOTE: axes [{axes}] were CLAMPED UP to the {min_box_size:g} Angstrom")
                print("  !         floor, so their L is a hardcoded constant, not your structure.")
            print("  ! For LAMMPS-equivalent results, do one of:")
            print("  !   (a) supply the real cell (extended XYZ 'Lattice=\"...\"', or a")
            print("  !       LAMMPS data / dump file with box bounds), or")
            print("  !   (b) set manual = True and give c = the reciprocal spacing")
            print("  !       directly in 1/Angstrom (LAMMPS 'manual' semantics).")
            print("  " + "!" * 68)
            print("")

        return lengths
    
    def read_structure_file(self):

        if self.me == 0 and self.echo:
            print("\n--- Reading Structure File ---")
            print(f"Structure file: {self.structure_file}")
            print(f"Available atom types: {', '.join(self.atom_types)}")

        positions = None
        atom_names = None
        cell = None
        pbc = None
        num_atoms = 0

        if self.use_ase:
            if self.echo and self.me == 0:
                print("Attempting to read structure file with ASE...")
            result = self.read_with_ase(self.structure_file)
            if result is not None:
                positions, atom_names, cell, pbc, num_atoms = self._apply_type_map(
                    result, self.type_map)
                if self.me == 0 and self.echo:
                    print(f"  ASE successfully read structure file")
                    print(f"  Atoms: {num_atoms}")
                    if cell is not None:
                        print(f"  Cell: a={np.linalg.norm(cell[0]):.4f}, "
                              f"b={np.linalg.norm(cell[1]):.4f}, "
                              f"c={np.linalg.norm(cell[2]):.4f} A")
                    else:
                        print(f"  Cell: not provided by the file")
                    if pbc is not None:
                        print(f"  Periodic boundaries: {list(pbc)}")
                    print(f"  Species found: {sorted(set(atom_names))}")

        if positions is None:
            low = str(self.structure_file).lower()
            if low.endswith('.data') or low.endswith('.lmp') or low.endswith('.lammpstrj'):
                if self.echo and self.me == 0:
                    print("ASE failed or not available. Trying LAMMPS data format...")
                result = self.read_lammps_data(self.structure_file)
                if result is not None:
                    positions, atom_names, cell, pbc, num_atoms = self._apply_type_map(
                        result, self.type_map)
                    if self.me == 0 and self.echo:
                        print(f"  LAMMPS data file read successfully")
                        print(f"  Atoms: {num_atoms}")
                        print(f"  Species found: {sorted(set(atom_names))}")
                        if cell is not None:
                            print(f"  Box lengths: {cell[0,0]:.4f}, {cell[1,1]:.4f}, {cell[2,2]:.4f} A")

            if positions is None:
                if self.echo and self.me == 0:
                    print("Trying XYZ format as fallback...")
                result = self.read_xyz_fallback(self.structure_file)
                if result is not None:
                    positions, atom_names, cell, pbc, num_atoms = self._apply_type_map(
                        result, self.type_map)
                    if self.me == 0 and self.echo:
                        print(f"  XYZ format read successfully")
                        print(f"  Atoms: {num_atoms}")
                        print(f"  Species found: {sorted(set(atom_names))}")

        if positions is None:
            raise ValueError(
                f"Could not read structure file {self.structure_file!r} with any available method.\n"
                f"  ASE, the LAMMPS data reader and the XYZ reader all failed.")

        self.positions = positions
        self.atom_names = atom_names
        self.cell = cell
        self.num_atoms = num_atoms

        self.typelocal_full = []
        for atom_name in atom_names:
            found = False
            atom_name_clean = str(atom_name).strip()
            for i, target_type in enumerate(self.atom_types):
                if atom_name_clean.lower() == target_type.lower():
                    self.typelocal_full.append(i)
                    found = True
                    break
            if not found:
                raise ValueError(f"Compute SAED: Atom type '{atom_name_clean}' in structure file "
                               f"does not match any input atom type ({', '.join(self.atom_types)}).")

        if self.echo and self.me == 0:
            print(f"  Structure loaded successfully:")
            print(f"  Total atoms: {num_atoms}")
            print(f"  Atom types used: {set(self.atom_types)}")
            print(f"  Atom type mapping: {dict(zip(set(atom_names), set(self.typelocal_full)))}")
    
    def read_with_ase(self, filename):
        try:
            from ase.io import read as ase_read
        except ImportError:
            return None

        try:
            atoms = ase_read(filename)
        except Exception as exc:
            if self.me == 0 and self.echo:
                if str(exc).find("Unknown') format") != -1:
                    print(f"  ASE could not determine the file format. Supported formats "
                          f"include XYZ, LAMMPS data, POSCAR/CONTCAR, CIF, VASP OUTCAR, etc.")
                else:
                    print(f"  ASE error reading file: {exc}")
            return None

        try:
            positions = np.asarray(atoms.get_positions(), dtype=float)
            atom_names = [str(s) for s in atoms.get_chemical_symbols()]
        except Exception as exc:
            if self.me == 0 and self.echo:
                print(f"  ASE read the file but no positions/species could be extracted: {exc}")
            return None

        if len(positions) == 0:
            return None

        cell = None
        try:
            matrix = np.asarray(atoms.get_cell(), dtype=float)
            if matrix.shape == (3, 3) and abs(np.linalg.det(matrix)) > 1e-8:
                cell = matrix
        except Exception:
            cell = None

        try:
            pbc = np.asarray(atoms.get_pbc(), dtype=bool)
        except Exception:
            pbc = None

        num_atoms = len(positions)
        return np.array(positions), atom_names, cell, pbc, num_atoms
    
    def read_lammps_data(self, filename):
        if not os.path.exists(filename):
            return None

        try:
            with open(filename, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            num_atoms = 0
            atom_types = {}
            positions = []
            atom_names = []

            cell = np.zeros((3, 3))
            have_box = [False, False, False]
            pbc = [True, True, True]

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                if not line or line.startswith('#'):
                    i += 1
                    continue

                if "atoms" in line and "atom types" not in line:
                    parts = line.split()
                    for j, part in enumerate(parts):
                        if part == "atoms":
                            num_atoms = int(parts[j - 1])
                            break

                elif "xlo xhi" in line:
                    parts = line.split()
                    xlo, xhi = float(parts[0]), float(parts[1])
                    cell[0, 0] = xhi - xlo
                    have_box[0] = True
                elif "ylo yhi" in line:
                    parts = line.split()
                    ylo, yhi = float(parts[0]), float(parts[1])
                    cell[1, 1] = yhi - ylo
                    have_box[1] = True
                elif "zlo zhi" in line:
                    parts = line.split()
                    zlo, zhi = float(parts[0]), float(parts[1])
                    cell[2, 2] = zhi - zlo
                    have_box[2] = True

                elif line.startswith("Masses"):
                    i += 2
                    while i < len(lines) and lines[i].strip():
                        mass_line = lines[i].strip()
                        if not mass_line.startswith('#'):
                            parts = mass_line.split()
                            if len(parts) >= 2:
                                atype = int(parts[0])
                                mass = float(parts[1])
                                if 190 < mass < 200:
                                    atom_types[atype] = "Au"
                                elif 180 < mass < 190:
                                    atom_types[atype] = "W"
                                elif 160 < mass < 180:
                                    atom_types[atype] = "Ta"
                                elif 140 < mass < 160:
                                    atom_types[atype] = "Gd"
                                elif 120 < mass < 140:
                                    atom_types[atype] = "Sn"
                                elif 90 < mass < 100:
                                    atom_types[atype] = "Mo"
                                elif 50 < mass < 70:
                                    atom_types[atype] = "Fe"
                                elif 40 < mass < 50:
                                    atom_types[atype] = "Ca"
                                elif 26 < mass < 28:
                                    atom_types[atype] = "Al"
                                elif 23 < mass < 25:
                                    atom_types[atype] = "Mg"
                                elif 10 < mass < 20:
                                    atom_types[atype] = "C"
                                elif 14 < mass < 16:
                                    atom_types[atype] = "N"
                                elif 15 < mass < 17:
                                    atom_types[atype] = "O"
                                elif 1 < mass < 2:
                                    atom_types[atype] = "H"
                                else:
                                    atom_types[atype] = str(atype)
                        i += 1
                    continue

                elif "Atoms" in line:
                    i += 2
                    atom_count = 0
                    while atom_count < num_atoms and i < len(lines):
                        atom_line = lines[i].strip()
                        if atom_line and not atom_line.startswith('#'):
                            parts = atom_line.split()
                            if len(parts) >= 5:
                                atype = int(parts[1])
                                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                                positions.append([x, y, z])
                                atom_names.append(atom_types.get(atype, str(atype)))
                                atom_count += 1
                        i += 1
                    break

                i += 1

            if len(positions) == 0:
                return None

            if not all(have_box):
                missing = [t for t, h in zip('xyz', have_box) if not h]
                print(f"{WARN} LAMMPS data file carries no {missing} box bounds; "
                      f"the cell is reported as unknown rather than guessed.")
                cell = None

            if self.me == 0 and self.echo:
                print(f"  LAMMPS data file read successfully:")
                print(f"    Atoms: {len(positions)}")
                print(f"    Species found: {sorted(set(atom_names))}")
                if cell is not None:
                    print(f"    Box lengths: {cell[0,0]:.4f}, {cell[1,1]:.4f}, "
                          f"{cell[2,2]:.4f} A")

            return np.asarray(positions, dtype=float), atom_names, cell, np.asarray(pbc, dtype=bool), len(positions)

        except Exception as exc:
            if self.me == 0 and self.echo:
                print(f"  Error reading LAMMPS data file: {exc}")
            return None

    def _apply_type_map(self, result, type_map):
        positions, atom_names, cell, pbc, num_atoms = result
        if not type_map:
            return positions, atom_names, cell, pbc, num_atoms
        remapped = []
        hits = 0
        for name in atom_names:
            key = str(name).strip()
            if key.isdigit() and int(key) in type_map:
                remapped.append(type_map[int(key)])
                hits += 1
            elif key in ('', 'X') and 0 in type_map:
                remapped.append(type_map[0])
                hits += 1
            else:
                remapped.append(key)
        if hits and self.me == 0 and self.echo:
            print(f"    type_map applied to {hits} of {len(atom_names)} sites")
        return positions, remapped, cell, pbc, num_atoms

    def _parse_type_map(self, raw):
        result = {}
        if isinstance(raw, str):
            for token in raw.replace(',', ' ').split():
                if ':' in token:
                    k, v = token.split(':', 1)
                    try:
                        result[int(k.strip())] = v.strip()
                    except ValueError:
                        pass
        return result
    
    def read_xyz_fallback(self, filename):
        if not os.path.exists(filename):
            if self.me == 0 and self.echo:
                print(f"File not found: {filename}")
            return None
            
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                return None
            
            try:
                num_atoms = int(lines[0].strip())
                if num_atoms <= 0:
                    raise ValueError("Invalid number of atoms")
            except:
                num_atoms = 0
                for line in lines[2:]:
                    if line.strip() and len(line.split()) >= 4:
                        num_atoms += 1
                if num_atoms == 0:
                    return None
            
            positions = []
            atom_names = []
            start_line = 2
            
            comment_line = lines[1].strip()
            
            cell = None

            if hasattr(self, 'pbc_input') and self.pbc_input is not None:
                pbc = self.pbc_input.copy()
                if self.echo and self.me == 0:
                    print(f"Using PBC values from input file: {pbc}")
            else:
                pbc = [False, False, False]
                if self.echo and self.me == 0:
                    print("Warning: XYZ file format does not contain PBC information.")
                    print("Using non-periodic boundaries by default. Specify 'pbc' parameter in input file for correct behavior.")
            
            if "BOX BOUNDS" in comment_line.upper():
                box_lines = []
                for line in lines[2:]:
                    if "ITEM:" not in line and line.strip():
                        box_lines.append(line.strip())
                    if len(box_lines) >= 3:
                        break
                if len(box_lines) >= 3:
                    cell = np.zeros((3, 3))
                    for i in range(3):
                        parts = box_lines[i].split()
                        if len(parts) >= 2:
                            lo = float(parts[0])
                            hi = float(parts[1])
                            cell[i, i] = hi - lo
            
            elif "CELL" in comment_line.upper() or "LATTICE" in comment_line.upper():
                import re
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", comment_line)
                if len(nums) >= 6:
                    a = float(nums[0])
                    b = float(nums[1])
                    c = float(nums[2])
                    alpha = math.radians(float(nums[3]))
                    beta = math.radians(float(nums[4]))
                    gamma = math.radians(float(nums[5]))
                    
                    # Convert to cell matrix
                    cell = np.zeros((3, 3))
                    cell[0, 0] = a
                    cell[0, 1] = b * math.cos(gamma)
                    cell[0, 2] = c * math.cos(beta)
                    cell[1, 1] = b * math.sin(gamma)
                    cell[1, 2] = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
                    cell[2, 2] = c * math.sqrt(1 - math.cos(alpha)**2 - math.cos(beta)**2 - math.cos(gamma)**2 +
                                             2*math.cos(alpha)*math.cos(beta)*math.cos(gamma)) / math.sin(gamma)
            
            if cell is None:
                temp_positions = []
                for i in range(min(num_atoms, len(lines)-start_line)):
                    if start_line + i >= len(lines):
                        break
                    line = lines[start_line + i].strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        temp_positions.append([x, y, z])
                        atom_names.append(parts[0].strip())
                    except:
                        continue
                
                if len(temp_positions) == 0:
                    return None
                
                positions = np.array(temp_positions)
                
                min_coords = np.min(positions, axis=0)
                max_coords = np.max(positions, axis=0)
                box_lengths = max_coords - min_coords
                
                buffer = 0.1 * box_lengths
                box_lengths += 2 * buffer
                
                cell = np.diag(box_lengths)
                
                if self.echo and self.me == 0:
                    print(f"✓ Estimated box dimensions from atomic positions:")
                    print(f"  X: {box_lengths[0]:.2f} Å (min: {min_coords[0]:.2f}, max: {max_coords[0]:.2f})")
                    print(f"  Y: {box_lengths[1]:.2f} Å (min: {min_coords[1]:.2f}, max: {max_coords[1]:.2f})")
                    print(f"  Z: {box_lengths[2]:.2f} Å (min: {min_coords[2]:.2f}, max: {max_coords[2]:.2f})")
                    print(f"  Added 10% buffer to each dimension")
            
            if len(positions) == 0:
                for i in range(min(num_atoms, len(lines)-start_line)):
                    if start_line + i >= len(lines):
                        break
                    line = lines[start_line + i].strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    atom_name = parts[0].strip()
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    except:
                        continue
                    atom_names.append(atom_name)
                    positions.append([x, y, z])
                positions = np.array(positions)
            
            if len(positions) == 0:
                return None
            
            return positions, atom_names, cell, pbc, len(positions)
        except Exception as e:
            if self.me == 0 and self.echo:
                print(f"Error reading XYZ file: {e}")
            return None
            
            
    def validate_cell_orthogonality(self):
        if not hasattr(self, 'cell') or self.cell is None:
            return True  
        
        off_diagonal_sum = sum(abs(self.cell[i,j]) for i in range(3) for j in range(3) if i != j)
        if off_diagonal_sum > 1e-10:
            raise ValueError("Compute SAED does not work with triclinic cells. "
                           "Use orthogonal simulation box or enable manual mode.")
        return True         
    
    def validate_inputs(self):

        if not self.manual and not any(self.pbc):
            raise ValueError("Non-periodic system requires manual mode (set manual=1 in input file)")
        
        if self.lambda_val <= 0:
            raise ValueError("Wavelength must be greater than zero")
            
        if self.Kmax / 2 < 0 or self.Kmax / 2 > 6:
            raise ValueError("|K|max/2 must be between 0 and 6")
            
        if any(c < 0 for c in self.c):
            raise ValueError("dKs must be greater than 0")
            
        if self.dR_Ewald < 0:
            raise ValueError("dR_Ewald slice must be greater than 0")
            
        if hasattr(self, 'dimension') and self.dimension == 2:
            raise ValueError("Compute SAED does not work with 2d structures")
            
        self.ztype = []
        for atom_type in self.atom_types:
            found = False
            atom_type_clean = str(atom_type).strip()
            for i, valid_type in enumerate(SAED_TYPE_LIST):
                if atom_type_clean.lower() == valid_type.lower():
                    self.ztype.append(i)
                    self.atom_types[self.atom_types.index(atom_type)] = valid_type 
                    found = True
                    break
            if not found:
                raise ValueError(f"Compute SAED: Invalid ASF atom type '{atom_type_clean}'. "
                               f"Available types are: {', '.join(SAED_TYPE_LIST[:20])}, ...")
                
        if len(self.ztype) == 0:
            raise ValueError("No valid atom types provided")

    def _mesh_reference(self):
        Kmax2 = self.Kmax * self.Kmax
        full_space = (abs(self.Zone[0]) < 1e-10 and abs(self.Zone[1]) < 1e-10
                      and abs(self.Zone[2]) < 1e-10)
        EmdR2 = (self.R_Ewald - self.dR_Ewald) ** 2
        EpdR2 = (self.R_Ewald + self.dR_Ewald) ** 2
        K = [0.0, 0.0, 0.0]
        out = []
        for k in range(-self.Knmax[2], self.Knmax[2] + 1):
            for j in range(-self.Knmax[1], self.Knmax[1] + 1):
                for i in range(-self.Knmax[0], self.Knmax[0] + 1):
                    K[0] = i * self.dK[0]
                    K[1] = j * self.dK[1]
                    K[2] = k * self.dK[2]
                    dinv2 = K[0]*K[0] + K[1]*K[1] + K[2]*K[2]
                    if dinv2 < Kmax2:
                        if full_space:
                            out.append((i, j, k))
                        else:
                            r2 = sum((K[m] - self.Zone[m])**2 for m in range(3))
                            if EmdR2 < r2 < EpdR2:
                                out.append((i, j, k))
        return np.array(out, dtype=np.int32).reshape(-1, 3)

    def _mesh_vectorized(self):
        dK = np.asarray(self.dK, dtype=np.float64)
        Kmax2 = self.Kmax * self.Kmax
        ix = np.arange(-self.Knmax[0], self.Knmax[0] + 1, dtype=np.int64)
        iy = np.arange(-self.Knmax[1], self.Knmax[1] + 1, dtype=np.int64)
        iz = np.arange(-self.Knmax[2], self.Knmax[2] + 1, dtype=np.int64)

        full_space = (abs(self.Zone[0]) < 1e-10 and abs(self.Zone[1]) < 1e-10
                      and abs(self.Zone[2]) < 1e-10)

        gj, gk = np.meshgrid(iy, iz, indexing="ij")
        gj = gj.ravel()
        gk = gk.ravel()
        Ky = gj * dK[1]
        Kz = gk * dK[2]
        Ky2 = Ky * Ky           
        Kz2 = Kz * Kz
        if not full_space:
            Zone = np.asarray(self.Zone, dtype=np.float64)
            EmdR2 = (self.R_Ewald - self.dR_Ewald) ** 2
            EpdR2 = (self.R_Ewald + self.dR_Ewald) ** 2
            dy2 = (Ky - Zone[1]) ** 2
            dz2 = (Kz - Zone[2]) ** 2

        parts = []
        for i in ix:
            Kx = i * dK[0]
            mask = (Kx * Kx + Ky2) + Kz2 < Kmax2
            if not full_space:
                r2 = ((Kx - Zone[0]) ** 2 + dy2) + dz2
                mask &= (r2 > EmdR2) & (r2 < EpdR2)
            nsel = int(np.count_nonzero(mask))
            if nsel:
                parts.append(np.column_stack([
                    np.full(nsel, i, dtype=np.int64), gj[mask], gk[mask]]))

        if not parts:
            return np.empty((0, 3), dtype=np.int32)
        st = np.vstack(parts)
        order = np.lexsort((st[:, 0], st[:, 1], st[:, 2]))  
        return np.ascontiguousarray(st[order].astype(np.int32))

    def count_reciprocal_points(self):
        if self.mesh_engine == "reference":
            self.store_tmp = self._mesh_reference()
        else:
            self.store_tmp = self._mesh_vectorized()
            if self.mesh_check:
                ref = self._mesh_reference()
                if not (ref.shape == self.store_tmp.shape
                        and np.array_equal(ref, self.store_tmp)):
                    raise ValueError(
                        "mesh_check failed: vectorized mesh (%d nodes) differs "
                        "from the LAMMPS reference mesh (%d nodes)"
                        % (len(self.store_tmp), len(ref)))
                if self.me == 0 and self.echo:
                    print(f"  mesh_check: vectorized mesh identical to reference "
                          f"({len(ref)} nodes)")

        self.nRows = len(self.store_tmp)
        if self.me == 0 and self.echo:
            print(f"Number of reciprocal lattice points: {self.nRows}")

    def initialize_arrays(self):
        self.vector = np.zeros(self.nRows, dtype=np.float64)

    def init(self):
        if not hasattr(self, "store_tmp") or len(self.store_tmp) != self.nRows:
            raise ValueError(
                f"Nrows inconsistent: store_tmp has "
                f"{0 if not hasattr(self, 'store_tmp') else len(self.store_tmp)} "
                f"rows, expected {self.nRows}")
        if self.me == 0 and self.echo:
            print(f"Reciprocal lattice points ready: {self.nRows} "
                  f"(order k -> j -> i, LAMMPS canonical)")

    def compute_vector(self):
        if self.me == 0 and self.echo:
            print()
            print("--- Computing SAED intensities ---")
            print(f"  Engine : {self.engine}", end="")
            if self.engine == "numba":
                import numba
                print(f" ({numba.config.NUMBA_NUM_THREADS} threads)")
            else:
                print("")
        t0 = time.time()

        natoms = self.num_atoms
        nlocal = natoms // self.nprocs
        remainder = natoms % self.nprocs
        if self.me < remainder:
            nlocal += 1

        start_idx = 0
        end_idx = natoms
        if self.nprocs > 1:
            start_idx = self.me * nlocal
            if self.me > remainder:
                start_idx += remainder
            end_idx = min(start_idx + nlocal, natoms)
            nlocal = end_idx - start_idx

        xlocal = np.ascontiguousarray(self.positions[start_idx:end_idx],
                                      dtype=np.float64)
        typelocal = np.ascontiguousarray(
            np.asarray(self.typelocal_full[start_idx:end_idx], dtype=np.int64))
        self.xlocal = xlocal
        self.typelocal = typelocal
        self.nlocalgroup = nlocal
        self.ntypes = len(self.atom_types)

        if self.me == 0 and self.echo:
            print(f"  Total atoms       : {natoms:,}")
            print(f"  Atoms / process   : {nlocal:,}")
            print(f"  Reciprocal points : {self.nRows:,}")

        Smax = self.Kmax / 2.0
        offset = 0 if Smax <= 2.0 else 10
        ztype = np.asarray(self.ztype, dtype=np.int64)
        a_c = np.ascontiguousarray(ASFSAED_NP[ztype, offset:offset + 5])
        b_c = np.ascontiguousarray(ASFSAED_NP[ztype, offset + 5:offset + 10])

        if self.me == 0 and self.echo:
            print(f"  Smax = {Smax:.2f}, ASF coefficient offset = {offset}")

        dK = np.asarray(self.dK, dtype=np.float64)

        if PSUTIL_AVAILABLE:
            try:
                available_memory = psutil.virtual_memory().available
            except Exception:
                available_memory = 4 * 1024 ** 3
        else:
            available_memory = 4 * 1024 ** 3

        bytes_per_kpt = 64 + 88 * self.ntypes
        if self.engine == "numpy":
            bytes_per_kpt += 32 * max(1, self.nlocalgroup)

        target = (available_memory * 0.5) / max(1, self.nprocs)
        memory_cap = max(1, int(target / bytes_per_kpt))
        chunk_size = min(self._compute_chunk_size, memory_cap, max(1, self.nRows))

        if self.me == 0 and self.echo:
            print(f"  Chunk size        : {chunk_size:,} "
                  f"(requested {self._compute_chunk_size:,}, "
                  f"memory cap {memory_cap:,})")

        kernel = _kernel_numba if self.engine == "numba" else _KERNELS[self.engine]

        n_chunks = int(math.ceil(self.nRows / chunk_size)) if self.nRows else 0
        frac = 0.1
        for ci in range(n_chunks):
            lo = ci * chunk_size
            hi = min(lo + chunk_size, self.nRows)

            K_chunk = np.ascontiguousarray(
                self.store_tmp[lo:hi].astype(np.float64) * dK[None, :])

            dinv2 = ((K_chunk[:, 0] * K_chunk[:, 0]
                      + K_chunk[:, 1] * K_chunk[:, 1])
                     + K_chunk[:, 2] * K_chunk[:, 2])
            SinTheta_lambda = np.ascontiguousarray(0.5 * np.sqrt(dinv2))

            Freal, Fimag = kernel(K_chunk, SinTheta_lambda,
                                  xlocal, typelocal, a_c, b_c)

            self.vector[lo:hi] = (Freal * Freal + Fimag * Fimag) / natoms

            if self.echo and self.me == 0 and self.nRows:
                while hi >= round(frac * self.nRows) and frac <= 1.0:
                    print(f"{frac*100:2.0f}% -", end="", flush=True)
                    frac += 0.1
        if self.echo and self.me == 0:
            print("100%")

        t2 = time.time()
        bytes_used = self.memory_usage()
        if self.me == 0 and self.echo:
            print()
            print("Computation completed:")
            print(f"  Time elapsed: {t2-t0:.2f} seconds")
            print(f"  Memory usage: {bytes_used/1024/1024:.2f} MB/processor")
            if self.nRows:
                print(f"  Average intensity: {np.mean(self.vector):.6f}")
                print(f"  Max intensity: {np.max(self.vector):.6f}")
                print(f"  Min intensity: {np.min(self.vector):.6f}")

    def memory_usage(self):

        chunk = min(getattr(self, '_compute_chunk_size', _DEFAULT_CHUNK),
                    max(1, self.nRows))
        bytes = self.nRows * 8              
        bytes += self.nRows * 3 * 4         
        bytes += chunk * 3 * 8              
        bytes += chunk * 8 * 2              
        bytes += chunk * 8 * 2             
        bytes += self.ntypes * 10 * 8       
        bytes += self.nlocalgroup * 3 * 8   
        bytes += self.nlocalgroup * 8       
        if getattr(self, 'engine', 'numpy') == 'numpy':
            bytes += chunk * self.ntypes * 88         
            bytes += chunk * self.nlocalgroup * 32    

        return bytes
        
        
    def write_vtk(self, output_base, output_index=0):
        if self.me == 0:
            return write_saed_vtk_from_compute(self, output_base, output_index)
        return None
        
    def write_results(self):
        if self.me != 0:
            return None
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("# SAED results from compute_saed.py" + chr(10))
            f.write(f"# Computation time: {datetime.now()}" + chr(10))
            f.write(f"# Structure file: {self.structure_file}" + chr(10))
            f.write(f"# Wavelength: {self.lambda_val} Angstrom" + chr(10))
            f.write(f"# Kmax: {self.Kmax}" + chr(10))
            f.write(f"# Zone axis (scaled to R_Ewald): {self.Zone}" + chr(10))
            f.write(f"# dR_Ewald: {self.dR_Ewald}" + chr(10))
            f.write(f"# Box dimensions: {[float(x) for x in self.prd] if self.prd is not None else None}" + chr(10))
            f.write(f"# Periodic boundaries: {self.pbc}" + chr(10))
            f.write(f"# Manual mode: {self.manual}" + chr(10))
            f.write(f"# dK spacing: {[float(x) for x in self.dK]}" + chr(10))
            f.write(f"# Engine: {self.engine}" + chr(10))
            f.write(f"# Total atoms: {self.num_atoms}" + chr(10))
            f.write(f"# Atom types: {chr(44).join(self.atom_types)}" + chr(10))
            f.write(f"# Number of reciprocal points: {self.nRows}" + chr(10))
            f.write("# i j k intensity" + chr(10))
            st = self.store_tmp
            for n in range(self.nRows):
                f.write("%d %d %d %.10g%s" % (st[n, 0], st[n, 1], st[n, 2],
                                              self.vector[n], chr(10)))
        print()
        print(f"Results written to {self.output_file}")
        print(f"  Total reciprocal points: {self.nRows}")
        print(f"  Total atoms processed: {self.num_atoms}")
        return self.output_file


def _configure_stdio():
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main():
    _configure_stdio()

    if not ASE_AVAILABLE:
        print("Warning: ASE not available. Limited file format support.")
        print("Install ASE with: pip install ase")

    parser = argparse.ArgumentParser(
        description='Compute SAED pattern from atomic positions '
                    '(port of LAMMPS compute saed)')
    parser.add_argument('--input', type=str, default='input_saed_lammps.txt',
                        help='Input file with SAED parameters')
    parser.add_argument('--engine', type=str, default=None, choices=ENGINES,
                        help="Override the compute engine from the input file. "
                             "'numba' is bit-identical to LAMMPS; 'numpy' is "
                             "the vectorized fallback; 'reference' is the "
                             "literal per-point transcription.")
    parser.add_argument('--mesh-check', action='store_true',
                        help='Verify the vectorized reciprocal mesh against '
                             'the literal LAMMPS triple loop before computing.')

    args = parser.parse_args()

    me = 0
    if me == 0:
        print(f"Starting SAED computation at {datetime.now()}")
        print(f"Input file: {args.input}")

    saed = ComputeSAED.from_input(args.input, engine=args.engine,
                                  mesh_check=args.mesh_check or None)

    saed.init()
    saed.compute_vector()

    if saed.output_file:
        saed.write_results()

    if saed.vtk_base is not None:
        vtk_file = saed.write_vtk(saed.vtk_base, saed.vtk_index)
        if me == 0 and vtk_file:
            print()
            print(f"VTK file written to: {vtk_file}")

    if me == 0:
        print()
        print(f"SAED computation completed successfully at {datetime.now()}")


if __name__ == "__main__":
    main()
