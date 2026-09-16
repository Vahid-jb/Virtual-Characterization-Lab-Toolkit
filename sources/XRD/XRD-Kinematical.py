#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
XRD-Kinematical.py - Periodic kinematical XRD, with optional
user-selected sample and instrument corrections.

"""

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS

from common_utils import (parse_input_file, read_lammps_xyz, validate_params,
                          species_read_options)

OK = "[OK]"
WARN = "[WARN]"
ERR = "[ERROR]"


def sanitize_params(params):
    if params is None:
        return {}
    p = dict(params)
    for prefix in ('xrd.', 'kinematical.', 'basic.'):
        for key in list(p.keys()):
            if isinstance(key, str) and key.startswith(prefix):
                new_key = key[len(prefix):]
                p.setdefault(new_key, p[key])
                del p[key]
    if 'absorption_thickness' in p and 'sample_thickness' not in p:
        p['sample_thickness'] = p['absorption_thickness']
    if 'duplicate_site_tol' in p and 'collapse_tol' not in p:
        print(f"{WARN} duplicate_site_tol is no longer used; the folded images are "
              f"merged with collapse_tol (default 0.25 A), which has to exceed the "
              f"thermal displacement. Set collapse_tol explicitly to override.")
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


def format_hkls(hkls_entry, max_show=2):
    if not hkls_entry:
        return None
    labels = []
    for item in hkls_entry:
        hkl = item.get('hkl') if isinstance(item, dict) else item
        if hkl is None:
            continue
        try:
            labels.append("(" + "".join(
                (f"{int(v)}" if int(v) >= 0 else f"-{abs(int(v))}") for v in hkl) + ")")
        except Exception:
            labels.append(str(hkl))
    if not labels:
        return None
    if len(labels) > max_show:
        return ", ".join(labels[:max_show]) + f" +{len(labels) - max_show}"
    return ", ".join(labels)


def hkl_multiplicity(hkls_entry):
    if not hkls_entry:
        return 0
    total = 0
    for item in hkls_entry:
        if isinstance(item, dict):
            total += int(item.get('multiplicity', 1) or 1)
        else:
            total += 1
    return total


ABSORPTION_GEOMETRIES = ('cylinder', 'debye_scherrer', 'bragg_brentano_reflection',
                         'symmetric_transmission', 'slab_attenuation_approx')


def absorption_cylinder(mu, radius, two_theta_deg, n_grid=241):

    two_theta = np.radians(np.asarray(two_theta_deg, dtype=float))
    R = float(radius)
    if R <= 0 or mu <= 0:
        return np.ones_like(two_theta)

    lin = np.linspace(-R, R, int(n_grid))
    X, Y = np.meshgrid(lin, lin, indexing='ij')
    r2 = X * X + Y * Y
    inside = r2 <= R * R
    Px, Py, P2 = X[inside], Y[inside], r2[inside]
    if Px.size == 0:
        return np.ones_like(two_theta)

    pu_in = -Px
    L_in = -pu_in + np.sqrt(np.maximum(R * R - P2 + pu_in * pu_in, 0.0))

    A = np.empty_like(two_theta)
    for i, tt in enumerate(np.atleast_1d(two_theta)):
        ux, uy = np.cos(tt), np.sin(tt)
        pu_out = Px * ux + Py * uy
        L_out = -pu_out + np.sqrt(np.maximum(R * R - P2 + pu_out * pu_out, 0.0))
        A[i] = np.mean(np.exp(-mu * (L_in + L_out)))
    return A


def absorption_bragg_brentano(mu, thickness, two_theta_deg):

    theta = np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0)
    sin_theta = np.maximum(np.sin(theta), 1e-8)
    x = 2.0 * mu * thickness / sin_theta
    out = np.ones_like(x)
    big = x > 1e-12
    out[big] = (1.0 - np.exp(-x[big])) / x[big]
    return out


def absorption_symmetric_transmission(mu, thickness, two_theta_deg):

    theta = np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0)
    cos_theta = np.maximum(np.cos(theta), 1e-8)
    return np.exp(-mu * thickness * (1.0 / cos_theta - 1.0))


def absorption_slab_attenuation(mu, thickness, two_theta_deg):

    theta = np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0)
    sin_theta = np.maximum(np.sin(theta), 1e-8)
    return np.exp(-mu * thickness * (1.0 / sin_theta - 1.0))


def apply_absorption_correction(intensities, two_theta, params, verbose=True):
    if not as_bool(params, 'apply_absorption', False):
        return intensities, None

    mu = as_float(params, 'mu_linear_cm_inverse',
                  as_float(params, 'linear_absorption_coefficient', 0.0))
    if mu <= 0:
        print(f"{WARN} apply_absorption is on but the linear absorption coefficient "
              f"is {mu}; no correction applied.")
        return intensities, None

    geometry = as_str(params, 'absorption_geometry', 'cylinder').lower()
    if geometry not in ABSORPTION_GEOMETRIES:
        raise ValueError(
            f"Unknown absorption_geometry {geometry!r}. Choose one of: "
            + ", ".join(ABSORPTION_GEOMETRIES))

    radius = as_float(params, 'capillary_radius_cm', as_float(params, 'sample_radius', 0.05))
    thickness = as_float(params, 'sample_thickness_cm',
                         as_float(params, 'sample_thickness', 0.1))

    if geometry in ('cylinder', 'debye_scherrer'):
        muR = mu * radius
        A = absorption_cylinder(mu, radius, two_theta)
        desc = (f"cylinder / Debye-Scherrer, numerical cross-section integral, "
                f"mu = {mu:g} cm^-1, R = {radius:g} cm, muR = {muR:.3f}")
        if muR > 10:
            print(f"{WARN} mu*R = {muR:.2f} is very large: transmission is "
                  f"{float(np.min(A)):.2e}-{float(np.max(A)):.2e} and the correction "
                  f"dominates the pattern. Check the units (mu in cm^-1, R in cm) and "
                  f"consider a thinner capillary or a validated experimental correction.")
    elif geometry == 'bragg_brentano_reflection':
        A = absorption_bragg_brentano(mu, thickness, two_theta)
        desc = (f"flat-plate reflection (Bragg-Brentano), mu = {mu:g} cm^-1, "
                f"t = {thickness:g} cm, mu*t = {mu * thickness:.3f}")
    elif geometry == 'symmetric_transmission':
        A = absorption_symmetric_transmission(mu, thickness, two_theta)
        desc = (f"flat-plate symmetric transmission, mu = {mu:g} cm^-1, "
                f"t = {thickness:g} cm, mu*t = {mu * thickness:.3f}")
    else:
        A = absorption_slab_attenuation(mu, thickness, two_theta)
        desc = (f"single-path slab attenuation (APPROXIMATE), mu = {mu:g} cm^-1, "
                f"t = {thickness:g} cm, mu*t = {mu * thickness:.3f}")

    if geometry in ('bragg_brentano_reflection', 'symmetric_transmission',
                    'slab_attenuation_approx') and mu * thickness > 10:
        print(f"{WARN} mu*t = {mu * thickness:.2f} is large; absorption dominates and a "
              f"validated, geometry-specific correction is required for quantitative work.")

    if verbose:
        print(f"  Absorption: {desc}")
        print(f"              A(2theta) ranges {float(np.min(A)):.5f} to {float(np.max(A)):.5f}")

    return intensities * A, desc


def parse_debye_waller(params, structure=None, verbose=True):

    dw = params.get('debye_waller_factors', None)
    if dw in (None, '', False):
        return None

    items = None
    factors = {}
    if isinstance(dw, dict):
        for el, val in dw.items():
            factors[str(el).strip()] = float(val)
    elif isinstance(dw, (list, tuple)):
        items = [str(x).strip() for x in dw if str(x).strip()]
    else:
        items = [x.strip() for x in str(dw).split(',') if x.strip()]

    if items is not None:
        for item in items:
            if '=' not in item:
                raise ValueError(
                    f"Invalid debye_waller_factors entry {item!r}. "
                    f"Expected 'Element=value', e.g. 'Zn=0.5,Cu=0.6'.")
            el, val = item.split('=', 1)
            try:
                factors[el.strip()] = float(val.strip())
            except ValueError:
                raise ValueError(f"Non-numeric Debye-Waller value in {item!r}")

    for el, val in factors.items():
        if not np.isfinite(val) or val < 0:
            raise ValueError(
                f"Debye-Waller factor for {el} is {val}; B_iso must be finite and "
                f"non-negative (typical range 0.2-2 A^2).")

    if structure is not None:
        present = {str(site.specie.symbol) for site in structure}
        unknown = set(factors) - present
        if unknown:
            raise ValueError(
                f"Debye-Waller factors supplied for elements not in the structure: "
                f"{sorted(unknown)}. Structure contains {sorted(present)}.")
        missing = present - set(factors)
        if missing and verbose:
            print(f"{WARN} no Debye-Waller factor for {sorted(missing)}; those species "
                  f"get B = 0 (no thermal damping).")

    if verbose:
        print(f"  Debye-Waller factors (B_iso, A^2): "
              + ", ".join(f"{k}={v:g}" for k, v in sorted(factors.items())))
    return factors or None


LATTICE_MODES = ('auto', 'simulation_box', 'unit_cell')

PYMATGEN_DISTANCE_TOLERANCE = 0.5

# Above this many sites the proximity check uses a neighbour tree: pymatgen's own
# check builds N x N x 3 arrays (83 GiB for 60 000 sites).
PROXIMITY_MATRIX_LIMIT = 5000


def parse_extxyz_lattice(xyz_file, verbose=True):
    try:
        with open(xyz_file, 'r', encoding='utf-8-sig', errors='replace') as fh:
            fh.readline()
            comment = fh.readline()
    except OSError:
        return None
    if not comment:
        return None

    start = comment.lower().find('lattice')
    if start < 0:
        return None
    eq = comment.find('=', start)
    if eq < 0:
        return None
    rest = comment[eq + 1:].lstrip()
    if not rest or rest[0] not in ('"', "'"):
        return None
    quote = rest[0]
    end = rest.find(quote, 1)
    if end < 0:
        return None

    try:
        values = [float(tok) for tok in rest[1:end].replace(',', ' ').split()]
    except ValueError:
        if verbose:
            print(f"{WARN} could not parse the Lattice= entry in the comment line; "
                  f"ignoring it.")
        return None
    if len(values) != 9:
        if verbose:
            print(f"{WARN} Lattice= carries {len(values)} numbers, expected 9; "
                  f"ignoring it.")
        return None

    matrix = np.asarray(values, dtype=float).reshape(3, 3)
    if abs(float(np.linalg.det(matrix))) < 1e-8:
        if verbose:
            print(f"{WARN} the Lattice= entry is singular (zero cell volume); "
                  f"ignoring it.")
        return None
    return matrix


def _periodic_images(cart, matrix):
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
                       for k in (-1, 0, 1)], dtype=float)
    offsets = shifts @ np.asarray(matrix, dtype=float)
    return (cart[None, :, :] + offsets[:, None, :]).reshape(-1, 3)


def _ball_counts(tree, points, cutoff):
    try:
        return np.asarray(tree.query_ball_point(points, cutoff, return_length=True),
                          dtype=float)
    except TypeError:
        return np.array([len(tree.query_ball_point(p, cutoff)) for p in points],
                        dtype=float)


def first_shell_cutoff(cart, matrix=None, n_bins=160):
    from scipy.spatial import cKDTree

    cart = np.asarray(cart, dtype=float)
    if len(cart) < 2:
        return 1.0, 0.0
    points = cart if matrix is None else _periodic_images(cart, matrix)
    tree = cKDTree(points)

    r1 = float(np.median(tree.query(cart, k=2)[0][:, 1]))
    if not np.isfinite(r1) or r1 <= 0:
        return 1.0, 0.0

    r_probe = 2.5 * r1
    k = int(min(60, len(points)))
    dists = tree.query(cart, k=k)[0]
    pooled = dists[:, 1:].ravel() if dists.ndim == 2 else dists
    pooled = pooled[(pooled > 1e-9) & (pooled <= r_probe)]
    if pooled.size < 8:
        return 1.15 * r1, r1

    hist, edges = np.histogram(pooled, bins=n_bins, range=(0.0, r_probe))
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(hist.astype(float), np.ones(5) / 5.0, mode='same')

    window = max(int(np.searchsorted(centers, 1.35 * r1)), 2)
    peak = int(np.argmax(smooth[:window]))
    floor_level = 0.15 * smooth[peak]
    for i in range(peak + 1, n_bins - 1):
        if smooth[i] <= floor_level or (smooth[i] <= smooth[i - 1]
                                        and smooth[i] <= smooth[i + 1]):
            cutoff = float(centers[i])
            if cutoff > 1.02 * r1:
                return cutoff, r1
    return 1.15 * r1, r1


def surface_fraction(cart, matrix=None, cutoff=None):
    from scipy.spatial import cKDTree

    cart = np.asarray(cart, dtype=float)
    if len(cart) < 2:
        return 0.0, np.zeros(len(cart)), (cutoff or 1.0)
    if cutoff is None:
        cutoff, _ = first_shell_cutoff(cart, matrix)

    tree = cKDTree(cart if matrix is None else _periodic_images(cart, matrix))
    cn = _ball_counts(tree, cart, cutoff) - 1.0

    bulk = float(np.percentile(cn, 90))
    if bulk <= 0:
        return 1.0, cn, cutoff
    return float(np.mean(cn < 0.75 * bulk)), cn, cutoff


def read_cell_with_ase(path, verbose=True):
    try:
        from ase.io import read as ase_read
    except ImportError:
        return None
    try:
        # index=0: read_lammps_xyz and parse_extxyz_lattice both take the first
        # frame, and ASE defaults to the last. On a trajectory that mismatch
        # would pair one frame's cell with another frame's coordinates.
        atoms = ase_read(path, index=0)
    except Exception as exc:
        if verbose:
            print(f"  ASE could not read {os.path.basename(path)}: {exc}")
        return None
    matrix = np.array(atoms.get_cell(), dtype=float)
    if matrix.shape != (3, 3) or abs(np.linalg.det(matrix)) < 1e-8:
        return None
    return matrix


def read_with_ase(filename, verbose=True):
    try:
        from ase.io import read as ase_read
    except ImportError:
        return None

    try:
        # index=0: the first frame, as in read_cell_with_ase and read_lammps_xyz
        atoms = ase_read(filename, index=0)
    except Exception as exc:
        if verbose:
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
        if verbose:
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

    if verbose:
        print(f"  ASE successfully read the structure file:")
        print(f"    Atoms: {len(positions)}")
        if cell is None:
            print(f"    Cell: not provided by the file")
        else:
            print(f"    Cell: a={np.linalg.norm(cell[0]):.4f}, "
                  f"b={np.linalg.norm(cell[1]):.4f}, "
                  f"c={np.linalg.norm(cell[2]):.4f} A")
        if pbc is not None:
            print(f"    Periodic boundaries: {list(pbc)}")
        print(f"    Species found: {sorted(set(atom_names))}")

    return positions, atom_names, cell, pbc


def read_lammps_data(filename, verbose=True):
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
                            atom_type = int(parts[0])
                            mass = float(parts[1])
                            if 90 < mass < 100:
                                atom_types[atom_type] = "Mo"
                            elif 50 < mass < 70:
                                atom_types[atom_type] = "Fe"
                            elif 20 < mass < 50:
                                atom_types[atom_type] = "Ca"
                            elif 10 < mass < 20:
                                atom_types[atom_type] = "C"
                            else:
                                atom_types[atom_type] = "H"
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
                            atom_type = int(parts[1])
                            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])

                            positions.append([x, y, z])
                            atom_names.append(atom_types.get(atom_type, "X"))
                            atom_count += 1
                    i += 1
                break

            i += 1

        if len(positions) == 0:
            return None

        if not all(have_box):
            missing = [t for t, h in zip('xyz', have_box) if not h]
            print(f"{WARN} LAMMPS data file carries no {missing} box bounds; the cell "
                  f"is reported as unknown rather than guessed.")
            cell = None

        if verbose:
            print(f"  LAMMPS data file read successfully:")
            print(f"    Atoms: {len(positions)}")
            print(f"    Species found: {sorted(set(atom_names))}")
            if cell is not None:
                print(f"    Box lengths: {cell[0, 0]:.4f}, {cell[1, 1]:.4f}, "
                      f"{cell[2, 2]:.4f} A")

        return np.asarray(positions, dtype=float), atom_names, cell, np.asarray(pbc, dtype=bool)

    except Exception as exc:
        if verbose:
            print(f"  Error reading LAMMPS data file: {exc}")
        return None


def read_structure_file(filename, params, numeric_mode, type_map, verbose=True):
    if as_bool(params, 'use_ase', True):
        if verbose:
            print(f"Attempting to read structure file with ASE...")
        result = read_with_ase(filename, verbose=verbose)
        if result is not None:
            return _apply_type_map(result, type_map, verbose=verbose)

    low = str(filename).lower()
    if low.endswith('.data') or low.endswith('.lmp'):
        if verbose:
            print(f"ASE failed or not available. Trying LAMMPS data format...")
        result = read_lammps_data(filename, verbose=verbose)
        if result is not None:
            return _apply_type_map(result, type_map, verbose=verbose)

    if verbose:
        print(f"Trying XYZ format as fallback...")
    try:
        species, coords = read_lammps_xyz(filename, verbose=verbose,
                                          numeric_mode=numeric_mode, type_map=type_map)
    except Exception as exc:
        raise ValueError(
            f"Could not read structure file {filename!r} with any available method.\n"
            f"  ASE, the LAMMPS data reader and the XYZ reader all failed.\n"
            f"  Last error from the XYZ reader: {exc}")

    return np.asarray(coords, dtype=float), [str(s) for s in species], None, None


def _apply_type_map(result, type_map, verbose=True):
    positions, atom_names, cell, pbc = result
    if not type_map:
        return positions, atom_names, cell, pbc

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
    if hits and verbose:
        print(f"    type_map applied to {hits} of {len(atom_names)} sites")
    return positions, remapped, cell, pbc


def _species_codes(species):
    order = {s: i for i, s in enumerate(sorted(set(map(str, species))))}
    return np.array([order[str(s)] for s in species])


def candidate_translations(cart, species, cutoff, r1, params, verbose=True):
    from scipy.spatial import cKDTree

    cart = np.asarray(cart, dtype=float)
    codes = _species_codes(species)
    tol = as_float(params, 'cell_inference_tol', 0.15) * r1
    min_score = as_float(params, 'cell_inference_min_score', 0.90)
    reach = as_float(params, 'cell_inference_reach', 1.8) * r1
    tree = cKDTree(cart)

    lo, hi = cart.min(axis=0), cart.max(axis=0)
    interior = np.all((cart > lo + cutoff) & (cart < hi - cutoff), axis=1)
    if interior.sum() < 4:
        interior = np.ones(len(cart), dtype=bool)

    pairs = tree.query_pairs(reach, output_type='ndarray')
    if len(pairs) == 0:
        return []
    vecs = cart[pairs[:, 1]] - cart[pairs[:, 0]]
    vecs = np.vstack([vecs, -vecs])
    vecs = vecs[np.argsort(np.linalg.norm(vecs, axis=1))]

    accepted = []
    for v in vecs:
        if any(np.linalg.norm(v - u) < tol for u, _ in accepted):
            continue
        d, j = tree.query(cart[interior] + v, k=1)
        score = float(np.mean((d < tol) & (codes[j] == codes[interior])))
        if score < min_score:
            continue
        d, j = tree.query(cart + v, k=1)
        good = (d < tol) & (codes[j] == codes)
        accepted.append(((cart[j[good]] - cart[good]).mean(axis=0), score))
        if len(accepted) >= as_int(params, 'cell_inference_max_vectors', 24):
            break
    return accepted


def infer_crystal_lattice(cart, species, cutoff, r1, params, verbose=True):
    info = {}
    trans = candidate_translations(cart, species, cutoff, r1, params, verbose=verbose)
    info['n_candidates'] = len(trans)
    if len(trans) < 3:
        info['reason'] = (f"only {len(trans)} translation vector(s) map the "
                          f"structure onto itself")
        return None, info

    info['mean_score'] = float(np.mean([s for _, s in trans]))
    vecs = [v for v, _ in sorted(trans, key=lambda t: np.linalg.norm(t[0]))]

    basis = []
    for v in vecs:
        trial = np.array(basis + [v])
        if np.linalg.matrix_rank(trial, tol=1e-6 * r1) == len(trial):
            basis.append(v)
        if len(basis) == 3:
            break
    if len(basis) < 3:
        info['reason'] = "the translation vectors found are coplanar"
        return None, info

    matrix = np.array(basis)
    for _ in range(3):
        idx = np.array([np.round(v @ np.linalg.inv(matrix)) for v in vecs])
        keep = np.array([np.linalg.norm(m @ matrix - v) < 0.25 * r1
                         for m, v in zip(idx, vecs)])
        if keep.sum() < 3 or np.linalg.matrix_rank(idx[keep]) < 3:
            break
        matrix = np.linalg.lstsq(idx[keep], np.array(vecs)[keep], rcond=None)[0]

    lengths = np.linalg.norm(matrix, axis=1)
    volume = abs(np.linalg.det(matrix))
    info['volume'] = float(volume)
    if volume < 0.05 * float(np.prod(lengths)):
        info['reason'] = ("the fitted cell is degenerate - its vectors came out "
                          "nearly coplanar")
        return None, info

    residual = float(np.mean([np.linalg.norm(np.round(v @ np.linalg.inv(matrix))
                                             @ matrix - v) for v in vecs]))
    info['snap_residual'] = residual
    if residual > as_float(params, 'cell_inference_max_residual', 0.10) * r1:
        info['reason'] = (f"the translation vectors do not close onto a single "
                          f"lattice (residual {residual:.3f} A)")
        return None, info

    if verbose:
        lat = Lattice(matrix)
        print(f"  lattice inferred from the coordinates: a={lat.a:.4f}, "
              f"b={lat.b:.4f}, c={lat.c:.4f} A, alpha={lat.alpha:.2f}, "
              f"beta={lat.beta:.2f}, gamma={lat.gamma:.2f} deg")
        print(f"    from {len(trans)} self-mapping translations (mean match "
              f"{100 * info['mean_score']:.1f}%, closure residual "
              f"{residual:.3f} A)")
    return matrix, info


def axis_period(cart, axis, cutoff, r1, bulk_cn, params, n_coarse=80, n_fine=40):
    from scipy.spatial import cKDTree

    cart = np.asarray(cart, dtype=float)
    lo, hi = cart[:, axis].min(), cart[:, axis].max()
    span = hi - lo
    if span <= 0 or bulk_cn <= 0:
        return None, 0.0

    margin = as_float(params, 'axis_period_margin', 1.0) * cutoff
    inner = np.ones(len(cart), dtype=bool)
    for k in (i for i in range(3) if i != axis):
        klo, khi = cart[:, k].min(), cart[:, k].max()
        inner &= (cart[:, k] > klo + margin) & (cart[:, k] < khi - margin)
    face = inner & ((cart[:, axis] > hi - cutoff) | (cart[:, axis] < lo + cutoff))
    if face.sum() < 4:
        return None, 0.0
    probes = cart[face]

    near = cart[(cart[:, axis] < lo + 2.5 * cutoff) | (cart[:, axis] > hi - 2.5 * cutoff)]
    low = cart[cart[:, axis] < lo + 1.5 * cutoff]
    high = cart[cart[:, axis] > hi - 1.5 * cutoff]
    e = np.zeros(3)
    e[axis] = 1.0
    floor = 0.8 * r1

    def score_of(length):
        pts = np.vstack([near, low + length * e, high - length * e])
        tree = cKDTree(pts)
        if tree.query(near, k=2)[0][:, 1].min() < floor:
            return -1.0
        cn = _ball_counts(tree, probes, cutoff) - 1.0
        return float(1.0 - np.mean(np.abs(cn - bulk_cn)) / bulk_cn)

    grid = np.linspace(span, span + 1.3 * cutoff, n_coarse)
    scores = np.array([score_of(L) for L in grid])
    best = int(np.argmax(scores))
    if scores[best] <= 0:
        return None, 0.0
    step = grid[1] - grid[0]
    fine = np.linspace(max(span, grid[best] - step), grid[best] + step, n_fine)
    fine_scores = np.array([score_of(L) for L in fine])
    j = int(np.argmax(fine_scores))
    if fine_scores[j] >= scores[best]:
        return float(fine[j]), float(fine_scores[j])
    return float(grid[best]), float(scores[best])


def recover_box_from_coordinates(cart, species, params, cutoff, r1, bulk_cn,
                                 verbose=True):
    info = {'axis_scores': [], 'axis_periods': []}
    limit = as_int(params, 'max_box_scan_atoms', 200000)
    if len(cart) > limit:
        info['reason'] = f"{len(cart)} atoms exceeds max_box_scan_atoms = {limit}"
        return None, info

    min_score = as_float(params, 'axis_period_min_score', 0.92)
    periods = []
    for axis in range(3):
        length, score = axis_period(cart, axis, cutoff, r1, bulk_cn, params)
        periods.append(length)
        info['axis_periods'].append(length)
        info['axis_scores'].append(score)
    info['periodic_axes'] = [k for k in range(3) if periods[k] is not None
                             and info['axis_scores'][k] >= min_score]
    if verbose:
        print(f"  searching for a periodic box in the coordinates:")
        for k, name in enumerate('xyz'):
            length, score = periods[k], info['axis_scores'][k]
            verdict = 'periodic' if k in info['periodic_axes'] else 'NOT periodic'
            head = f"period {length:.4f} A" if length else "no period found"
            print(f"    {name}: {head}, coordination repaired "
                  f"{100 * score:.1f}% -> {verdict}")
    if len(info['periodic_axes']) < 3:
        info['reason'] = (f"the structure is periodic along "
                          f"{len(info['periodic_axes'])} of its 3 cartesian axes")
        return None, info

    matrix = np.diag(periods)
    prim, prim_info = infer_crystal_lattice(cart, species, cutoff, r1, params,
                                            verbose=verbose)
    info['inferred_lattice'] = prim
    info['inferred_info'] = prim_info
    if prim is not None:
        inverse = np.linalg.inv(prim)
        snapped = np.array([np.round(matrix[k] @ inverse) @ prim for k in range(3)])
        if abs(np.linalg.det(snapped)) > 1e-8:
            drift = float(np.max(np.abs(np.linalg.norm(snapped, axis=1)
                                        - np.linalg.norm(matrix, axis=1))))
            if drift < cutoff:
                matrix = snapped
                info['snapped'] = True
                if verbose:
                    print(f"    box snapped onto the inferred lattice "
                          f"(moved by at most {drift:.3f} A)")

    f_boxed, _, _ = surface_fraction(cart, matrix, cutoff)
    info['surface_fraction_boxed'] = f_boxed
    tol = as_float(params, 'periodicity_surface_tol', 0.05)
    if f_boxed >= tol:
        info['reason'] = (f"the recovered box still leaves {100 * f_boxed:.1f}% of "
                          f"the atoms under-coordinated")
        return None, info
    if verbose:
        lat = Lattice(matrix)
        print(f"  recovered box: a={lat.a:.4f}, b={lat.b:.4f}, c={lat.c:.4f} A, "
              f"alpha={lat.alpha:.2f}, beta={lat.beta:.2f}, "
              f"gamma={lat.gamma:.2f} deg "
              f"({100 * f_boxed:.1f}% under-coordinated after applying it)")
    return matrix, info


def recover_supercell_box(frac, lattice, params, cutoff=None, verbose=True):
    frac = np.asarray(frac, dtype=float)
    tol = as_float(params, 'periodicity_surface_tol', 0.05)
    span = frac.max(axis=0) - frac.min(axis=0)

    candidates = []
    for i in range(3):
        base = max(1, int(np.floor(span[i] + 1e-6)) + 1)
        candidates.append(sorted({base, base + 1}))

    shifted = frac - frac.min(axis=0)
    if cutoff is None:
        cutoff, _ = first_shell_cutoff(lattice.get_cartesian_coords(frac), None)

    best = (None, None, 1.0)
    for n1 in candidates[0]:
        for n2 in candidates[1]:
            for n3 in candidates[2]:
                n = np.array([n1, n2, n3], dtype=float)
                matrix = lattice.matrix * n[:, None]
                in_box = (shifted / n) % 1.0
                cart = (in_box * n) @ lattice.matrix
                f_surf, _, _ = surface_fraction(cart, matrix, cutoff)
                if f_surf < best[2]:
                    best = (matrix, n.astype(int), f_surf)
                if f_surf < tol:
                    if verbose:
                        print(f"  recovered box: {int(n1)} x {int(n2)} x {int(n3)} "
                              f"unit cells (surface fraction {100 * f_surf:.1f}% "
                              f"after applying it)")
                    return matrix, n.astype(int), f_surf
    return None, None, best[2]


def collapse_folded_sites(structure, tol, params, verbose=True):
    notes = []
    n_atoms = len(structure)
    if n_atoms < 2:
        return structure, 1.0, notes, []

    limit = as_int(params, 'max_collapse_atoms', 20000)
    if n_atoms > limit:
        notes.append(f"{n_atoms} atoms exceeds max_collapse_atoms = {limit}; the "
                     f"folded images were NOT merged")
        if verbose:
            print(f"{WARN} {notes[-1]}. Raise max_collapse_atoms, or supply the "
                  f"simulation box so that no folding is needed.")
        return structure, 1.0, notes, []

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    dist = np.array(structure.distance_matrix, dtype=float)
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    clusters = fcluster(linkage(squareform(dist, checks=False), method='average'),
                        tol, 'distance')

    species, coords, conflicts = [], [], []
    for label in np.unique(clusters):
        members = np.where(clusters == label)[0]
        names = {structure[i].species_string for i in members}
        if len(names) > 1:
            conflicts.append(sorted(names))
        base = structure[members[0]].frac_coords
        offsets = np.array([structure[i].frac_coords - base for i in members])
        offsets -= np.round(offsets)
        species.append(structure[members[0]].species)
        coords.append(base + offsets.mean(axis=0))

    collapsed = Structure(structure.lattice, species, coords,
                          coords_are_cartesian=False)
    repeats = n_atoms / max(len(collapsed), 1)
    if len(collapsed) < n_atoms:
        note = (f"{n_atoms} atoms folded onto {len(collapsed)} crystallographic "
                f"sites ({repeats:.4g} images per site)")
        notes.append(note)
        if verbose:
            print(f"  collapsed: {note}")
    if conflicts and verbose:
        print(f"{WARN} {len(conflicts)} of {len(species)} sites received more than "
              f"one element. A correct fold maps every image of a site onto the "
              f"same element.")

    spread = max((float(dist[np.ix_(m, m)].max())
                  for m in (np.where(clusters == lab)[0] for lab in np.unique(clusters))
                  if len(m) > 1), default=0.0)
    if spread > 2.0 * tol:
        note = (f"the widest merged site spans {spread:.3f} A, well beyond the "
                f"{tol:.3f} A collapse_tol, so these are not images of one site")
        notes.append(note)
        if verbose:
            print(f"{WARN} {note}. The cell constants are probably wrong for this "
                  f"structure; check a, b, c against the nearest-neighbour "
                  f"distance, or let the module recover the box itself "
                  f"(lattice_mode = auto).")

    sizes = np.bincount(clusters)[1:]
    if sizes.size and sizes.min() != sizes.max():
        note = (f"the fold was uneven: sites received between {sizes.min()} and "
                f"{sizes.max()} images, so the structure is not a perfect supercell "
                f"and the defects were averaged away")
        notes.append(note)
        if verbose:
            print(f"{WARN} {note}. Folding keeps only the average cell: vacancies, "
                  f"local disorder and strain leave no trace in the structure "
                  f"factors. To compute them, give the simulation box instead "
                  f"(lattice_mode = simulation_box).")
    return collapsed, repeats, notes, conflicts


def _aperiodic_error(cart, params, info, extra=""):
    span = cart.max(axis=0) - cart.min(axis=0) if len(cart) else np.zeros(3)
    f_open = info.get('surface_fraction_open')
    box_info = info.get('box_scan', {})
    scores = box_info.get('axis_scores', [])
    periods = box_info.get('axis_periods', [])
    periodic = box_info.get('periodic_axes', [])

    lines = [
        "This structure has no periodic lattice, so a kinematical calculation is "
        "not defined for it.",
        "",
        f"    atoms                      : {len(cart)}",
        f"    bounding box               : {span[0]:.3f} x {span[1]:.3f} x "
        f"{span[2]:.3f} A",
    ]
    if f_open is not None:
        lines.append(f"    under-coordinated atoms    : {100 * f_open:.1f}% "
                     f"(a filled periodic cell gives ~0%)")
    if scores:
        lines.append("    periodicity per axis       : "
                     + ", ".join(
                         f"{name} {'yes' if k in periodic else 'no'}"
                         + (f" ({periods[k]:.3f} A)" if k in periodic
                            and periods[k] else "")
                         + f" [{100 * scores[k]:.0f}%]"
                         for k, name in enumerate('xyz')))
    inferred = info.get('inferred_lattice_params')
    if inferred:
        lines.append(f"    lattice in the coordinates : {inferred}")
    if extra:
        lines.append(f"    {extra}")

    lines += [
        "",
        "  Every route to a lattice was tried and none succeeded:",
        "    1. no cell in the structure file (no extended-XYZ Lattice= entry, and "
        "no cell that ASE could read);",
        "    2. no cell_file was supplied;",
        "    3. " + ("no integer multiple of the supplied a, b, c repairs the "
                     "coordination;" if info.get('had_cell_params')
                     else "no a, b, c were supplied to try multiples of;"),
        "    4. no box recovered from the coordinates - see the per-axis verdicts "
        "above.",
        "",
        "  ==> Use XRD-Debye_Scattering. It computes the pattern directly from "
        "the interatomic",
        "      distances, needs no lattice at all, and is the physically correct "
        "model for a finite",
        "      cluster, a nanoparticle, an amorphous cell or a defect region with "
        "free surfaces. It",
        "      also keeps the crystallite-size broadening that a periodic "
        "calculation throws away.",
        "",
        "  If you believe the structure IS periodic, give it a cell in one of these "
        "ways:",
        "    - add Lattice=\"ax ay az bx by bz cx cy cz\" to line 2 of the "
        "structure file (extended XYZ);",
        "    - point cell_file at a file that carries the box (LAMMPS dump or data, "
        "POSCAR, CIF, ...);",
        "    - set lattice_mode = unit_cell and give the correct a, b, c, alpha, "
        "beta, gamma, but note",
        "      that folding assumes the cell axes are aligned with x, y, z - it "
        "cannot be used for a cell",
        "      built on directions such as [111], which is the usual dislocation "
        "setup.",
    ]
    raise ValueError("\n".join(lines))


def classify_periodicity(coords, lattice, params, xyz_file, verbose=True):
    info = {'had_cell_params': params.get('_have_cell_params', True)}
    requested = as_str(params, 'lattice_mode', 'auto').lower()
    if requested not in LATTICE_MODES:
        print(f"{WARN} unknown lattice_mode {requested!r}; using 'auto'")
        requested = 'auto'
    info['requested_mode'] = requested
    coords = np.asarray(coords, dtype=float)
    species = params.get('_species', ['X'] * len(coords))

    box = parse_extxyz_lattice(xyz_file, verbose=verbose)
    source = 'Lattice= entry in the structure file'
    if box is None and as_bool(params, 'use_ase', True):
        box = read_cell_with_ase(xyz_file, verbose=verbose)
        source = 'cell read from the structure file by ASE'

    if box is None:
        cell_file = as_str(params, 'cell_file', '')
        if cell_file:
            if not os.path.exists(cell_file):
                raise ValueError(f"cell_file not found: {cell_file}")
            box = (parse_extxyz_lattice(cell_file, verbose=verbose)
                   if cell_file.lower().endswith('.xyz') else None)
            if box is None:
                box = read_cell_with_ase(cell_file, verbose=verbose)
            if box is None:
                raise ValueError(
                    f"cell_file = {cell_file} carries no readable cell.\n"
                    f"  Use a LAMMPS dump or data file, a POSCAR, a CIF, or an "
                    f"extended XYZ with a Lattice= entry.")
            source = f'cell read from {os.path.basename(cell_file)}'

    if box is None:
        reader_cell = params.get('_reader_cell', None)
        if reader_cell is not None:
            candidate = np.asarray(reader_cell, dtype=float)
            if candidate.shape == (3, 3) and abs(np.linalg.det(candidate)) > 1e-8:
                box = candidate
                source = 'cell read from the structure file by the structure reader'

    if box is not None:
        info['box_source'] = source
        if requested == 'unit_cell':
            if verbose:
                print(f"  a cell is available ({source}), but lattice_mode = "
                      f"unit_cell was requested; folding into the supplied cell "
                      f"instead.")
        else:
            if verbose:
                print(f"  {source}: a={np.linalg.norm(box[0]):.4f}, "
                      f"b={np.linalg.norm(box[1]):.4f}, "
                      f"c={np.linalg.norm(box[2]):.4f} A")
            return 'simulation_box', box, info

    cutoff, r1 = first_shell_cutoff(coords, None)
    info['first_shell_cutoff'] = cutoff
    info['nearest_neighbour'] = r1
    f_open, cn, _ = surface_fraction(coords, None, cutoff)
    info['surface_fraction_open'] = f_open
    bulk_cn = float(np.percentile(cn, 90)) if len(cn) else 0.0
    info['bulk_coordination'] = bulk_cn
    if verbose:
        print(f"  first-shell cutoff {cutoff:.3f} A (nearest neighbour "
              f"{r1:.3f} A), bulk coordination {bulk_cn:.0f}")

    if requested == 'unit_cell':
        info['box_source'] = 'lattice_mode = unit_cell (requested)'
        return 'unit_cell', None, info

    if params.get('_have_cell_params', True):
        frac = lattice.get_fractional_coords(coords)
        box, repeats, f_boxed = recover_supercell_box(frac, lattice, params,
                                                      cutoff=cutoff, verbose=verbose)
        info['surface_fraction_boxed'] = f_boxed
        if box is not None:
            info['box_source'] = (f"recovered from the supplied unit cell: "
                                  f"{repeats[0]} x {repeats[1]} x {repeats[2]} cells")
            info['repeats'] = repeats
            return 'simulation_box', box, info

    if as_bool(params, 'infer_cell', True):
        box, scan = recover_box_from_coordinates(coords, species, params, cutoff,
                                                 r1, bulk_cn, verbose=verbose)
        info['box_scan'] = scan
        prim = scan.get('inferred_lattice')
        if prim is not None:
            lat = Lattice(prim)
            info['inferred_lattice_params'] = (
                f"a={lat.a:.4f}, b={lat.b:.4f}, c={lat.c:.4f} A, "
                f"alpha={lat.alpha:.2f}, beta={lat.beta:.2f}, "
                f"gamma={lat.gamma:.2f} deg")
        if box is not None:
            info['box_source'] = 'box recovered from the coordinates'
            return 'simulation_box', box, info

    if requested == 'simulation_box':
        _aperiodic_error(coords, params, info,
                         extra="lattice_mode = simulation_box was requested")

    if f_open >= as_float(params, 'periodicity_surface_tol', 0.05):
        _aperiodic_error(coords, params, info)

    info['box_source'] = 'none (folded into the supplied unit cell)'
    return 'unit_cell', None, info

def _warn_finite_size(structure, info, params, verbose=True):
    f_open = info.get('surface_fraction_open')
    if f_open is None or f_open < as_float(params, 'periodicity_surface_tol', 0.05):
        return None
    note = (f"{100 * f_open:.1f}% of the atoms are under-coordinated and no periodic "
            f"box could be established, so this is most likely a finite cluster that "
            f"happens to fold cleanly onto the supplied cell")
    if verbose:
        print(f"{WARN} {note}. The pattern is that of the INFINITE crystal: the "
              f"Bragg peaks carry no crystallite-size broadening and the free "
              f"surfaces contribute nothing. For the finite particle use "
              f"XRD-Debye_Scattering.")
    return note


def build_structure(params, verbose=True):

    missing = [k for k in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')
               if k not in params]
    params['_have_cell_params'] = not missing

    if missing:
        listed = ", ".join(missing)
        is_are = 'is' if len(missing) == 1 else 'are'
        if as_str(params, 'lattice_mode', 'auto').lower() == 'unit_cell':
            raise ValueError(
                f"lattice_mode = unit_cell folds the structure into the cell you "
                f"supply, but {listed} {is_are} missing.\n"
                f"  Either give the cell parameters, or leave lattice_mode = auto "
                f"and let the module read or infer the cell.")
        if as_str(params, 'coordinate_mode', 'cartesian').lower() == 'fractional':
            raise ValueError(
                f"coordinate_mode = fractional needs a cell to convert the "
                f"coordinates, but {listed} {is_are} missing.")
        if verbose:
            print(f"  no unit cell supplied ({listed}); it will be read from the "
                  f"structure file or inferred from the coordinates")

    lattice = Lattice.from_parameters(
        as_float(params, 'a', 1.0), as_float(params, 'b', 1.0), as_float(params, 'c', 1.0),
        as_float(params, 'alpha', 90.0), as_float(params, 'beta', 90.0),
        as_float(params, 'gamma', 90.0))

    species_mode, numeric_mode, type_map = species_read_options(params)

    coords, species, reader_cell, reader_pbc = read_structure_file(
        params['xyz_file'], params, numeric_mode, type_map, verbose=verbose)
    coords = np.asarray(coords, dtype=float)
    params['_reader_cell'] = reader_cell
    params['_reader_pbc'] = reader_pbc

    coordinate_mode = as_str(params, 'coordinate_mode', 'cartesian').lower()
    if coordinate_mode not in ('cartesian', 'fractional'):
        print(f"{WARN} unknown coordinate_mode {coordinate_mode!r}; using 'cartesian'")
        coordinate_mode = 'cartesian'
    cart = (coords if coordinate_mode == 'cartesian'
            else lattice.get_cartesian_coords(coords))

    diagnostics = []
    if verbose:
        print(f"\nPeriodicity assessment:")
    params['_species'] = species
    mode, box, info = classify_periodicity(cart, lattice, params,
                                           params['xyz_file'], verbose=verbose)
    params['_lattice_mode'] = mode
    params['_box_source'] = info.get('box_source', '?')

    if mode == 'simulation_box':
        used_lattice = Lattice(box)
        frac = used_lattice.get_fractional_coords(cart)
        if as_bool(params, 'wrap_coords', True):
            frac = frac - np.floor(frac)
        structure = _make_structure(used_lattice, species, frac, params, verbose)
        diagnostics.append(f"lattice taken from the simulation box "
                           f"[{info.get('box_source', '?')}]; all "
                           f"{len(structure)} atoms kept, nothing folded")
        _report_structure(structure, mode, info, coordinate_mode, species_mode,
                          verbose)
        return structure, diagnostics

    frac = lattice.get_fractional_coords(cart)
    outside = int(np.sum(np.any((frac < -1e-9) | (frac >= 1.0 + 1e-9), axis=1)))
    wrap = as_bool(params, 'wrap_coords', True)
    if wrap:
        frac = frac - np.floor(frac)
        if verbose:
            print(f"  folded fractional coordinates into [0, 1)")
    if outside:
        diagnostics.append(
            f"{outside} of {len(frac)} atoms "
            f"({100.0 * outside / max(len(frac), 1):.1f}%) lay outside the [0,1) "
            f"range of the supplied cell"
            + (" and were folded in" if wrap else " and were NOT folded in"))

    folded = Structure(lattice, species, frac, coords_are_cartesian=False)

    if as_bool(params, 'check_duplicate_sites', True):
        collapsed, repeats, notes, conflicts = collapse_folded_sites(
            folded, as_float(params, 'collapse_tol', 0.25), params, verbose=verbose)
        diagnostics.extend(notes)
    else:
        collapsed, repeats, conflicts = folded, 1.0, []
        if verbose and outside:
            print(f"{WARN} check_duplicate_sites is off, so the folded images were "
                  f"not merged; the cell keeps all {len(folded)} atoms.")

    _check_fold_is_meaningful(collapsed, repeats, len(folded), cart, params, info,
                              conflicts, verbose=verbose)
    finite_note = _warn_finite_size(collapsed, info, params, verbose=verbose)
    if finite_note:
        diagnostics.append(finite_note)
    structure = _make_structure(
        collapsed.lattice, [site.species for site in collapsed],
        np.array([site.frac_coords for site in collapsed]), params, verbose)
    _report_structure(structure, mode, info, coordinate_mode, species_mode, verbose)
    return structure, diagnostics


def _check_fold_is_meaningful(structure, repeats, n_folded, cart, params, info,
                              conflicts=(), verbose=True):
    n_sites = len(structure)
    tol = as_float(params, 'repeats_integer_tol', 0.01)
    off = abs(repeats - round(repeats)) / max(round(repeats), 1)

    r1 = info.get('nearest_neighbour') or 0.0
    cell_per_site = structure.volume / max(n_sites, 1)
    floor = 0.35 * r1 ** 3
    too_dense = r1 > 0 and cell_per_site < floor

    if off <= tol and repeats >= 1.0 and not conflicts and not too_dense:
        return

    if as_bool(params, 'allow_aperiodic', False):
        if verbose:
            print(f"{WARN} allow_aperiodic is set, but it no longer suppresses "
                  f"this check. It only ever produced a pattern that looked "
                  f"plausible and meant nothing, which is worse than no answer at "
                  f"all. Use XRD-Debye_Scattering for an aperiodic structure.")

    span = cart.max(axis=0) - cart.min(axis=0) if len(cart) else np.zeros(3)
    file_density = float(np.prod(span)) / max(len(cart), 1)
    cell_density = structure.volume / max(n_sites, 1)
    f_open = info.get('surface_fraction_open')
    f_boxed = info.get('surface_fraction_boxed')

    clash = ""
    if conflicts:
        shown = ", ".join("/".join(c) for c in conflicts[:3])
        clash = (f"    sites with mixed elements  : {len(conflicts)} of {n_sites} "
                 f"({shown}{', ...' if len(conflicts) > 3 else ''})\n")

    coordination = ""
    if f_open is not None:
        coordination = (f"    under-coordinated atoms    : "
                        f"{100 * f_open:.1f}% with no box")
        coordination += (f", {100 * f_boxed:.1f}% with the best recovered box\n"
                         if f_boxed is not None else "\n")

    packing = ""
    if too_dense:
        packing = (f"    packing                    : {cell_per_site:.3f} A^3 per "
                   f"site, below the {floor:.3f} A^3 hard-sphere floor set by this "
                   f"structure's\n"
                   f"                                 own nearest-neighbour "
                   f"distance of {r1:.3f} A. No crystal packs that tightly, so the "
                   f"cell is\n"
                   f"                                 too small for the structure - "
                   f"check a, b, c against {2 * r1 / np.sqrt(3):.4f} A (bcc) or "
                   f"{r1 * np.sqrt(2):.4f} A (fcc).\n")

    raise ValueError(
        f"This structure is not a periodic crystal, so a kinematical calculation "
        f"is not defined for it.\n\n"
        f"  The structure file has no Lattice= entry, no integer multiple of the "
        f"supplied unit cell repairs\n"
        f"  the coordination, and folding into that cell does not land the atoms "
        f"on crystallographic sites:\n\n"
        f"    atoms                      : {n_folded}\n"
        f"    distinct sites after fold  : {n_sites} ({repeats:.4g} images per "
        f"site; a supercell gives a whole number)\n"
        f"    volume per site in the cell: {cell_density:.3f} A^3\n"
        f"    volume per atom in the file: {file_density:.3f} A^3 (a factor of "
        f"{file_density / max(cell_density, 1e-12):.0f} apart)\n"
        f"{clash}{coordination}{packing}\n"
        f"  Free surfaces and an unusable fold point the same way: this is a "
        f"finite cluster, or an\n"
        f"  otherwise aperiodic configuration. Use XRD-Debye_Scattering, which "
        f"computes the pattern\n"
        f"  directly from the interatomic distances and needs no lattice at all.\n\n"
        f"  If the structure IS periodic: add "
        f"Lattice=\"ax ay az bx by bz cx cy cz\" to the comment line (line 2)\n"
        f"  of the structure file, point cell_file at a file that carries the box, "
        f"or correct\n"
        f"  a, b, c, alpha, beta, gamma. There is no override: a fold that lands "
        f"atoms off their\n"
        f"  crystallographic sites cannot produce a meaningful pattern, whatever "
        f"is set.")


def _closest_site_pair(structure):
    """Closest pair of distinct sites under periodic boundaries, found with a
    neighbour tree instead of pymatgen's N x N distance matrix.

    Returns (i, j, distance, n_close); n_close counts the sites whose nearest
    other site lies within PYMATGEN_DISTANCE_TOLERANCE.
    """
    from scipy.spatial import cKDTree

    lattice = structure.lattice
    lll = lattice.get_lll_reduced_lattice()
    frac = lattice.get_lll_frac_coords(structure.frac_coords)
    cart = lll.get_cartesian_coords(frac - np.floor(frac))
    n = len(cart)
    dist, idx = cKDTree(_periodic_images(cart, lll.matrix)).query(cart, k=3)
    rows = np.arange(n)
    site = idx % n
    # a point's own entry and the images of its own site are not pairs
    masked = np.where(site != rows[:, None], dist, np.inf)
    col = np.argmin(masked, axis=1)
    d_other = masked[rows, col]
    i = int(np.argmin(d_other))
    return (i, int(site[i, col[i]]), float(d_other[i]),
            int(np.sum(d_other < PYMATGEN_DISTANCE_TOLERANCE)))


def _overlap_error(loose, i, j, closest, count_line):
    return ValueError(
        f"The constructed cell has overlapping sites, so it is not a physical "
        f"structure.\n\n"
        f"    closest pair               : {closest:.3f} A "
        f"({loose[i].species_string}-{loose[j].species_string}), tolerance "
        f"{PYMATGEN_DISTANCE_TOLERANCE} A\n"
        f"{count_line}"
        f"    sites in the cell          : {len(loose)}\n"
        f"    cell volume                : {loose.volume:.3f} A^3\n\n"
        f"  This is pymatgen's Structure.DISTANCE_TOLERANCE check. It "
        f"usually means the lattice\n"
        f"  parameters do not match the structure file, or that a non-periodic "
        f"configuration was folded\n"
        f"  into a unit cell. Check a, b, c, alpha, beta, gamma, or use "
        f"XRD-Debye_Scattering for a\n"
        f"  finite or aperiodic structure. Set validate_proximity = no to skip "
        f"this check.")


def _make_structure(lattice, species, frac, params, verbose=True):
    check = as_bool(params, 'validate_proximity', True)
    limit = as_int(params, 'max_proximity_check_atoms', 20000)
    if check and len(frac) > limit:
        check = False
        if verbose:
            print(f"  proximity check skipped: {len(frac):,} sites exceeds "
                  f"max_proximity_check_atoms = {limit:,}, and the check needs an "
                  f"N x N distance matrix ({len(frac) ** 2 * 24 / 2 ** 40:.3g} TiB "
                  f"at this size).")
    # Above PROXIMITY_MATRIX_LIMIT the same tolerance is enforced with a
    # neighbour tree, because pymatgen's own check builds N x N x 3 arrays.
    if check and len(frac) > PROXIMITY_MATRIX_LIMIT:
        loose = Structure(lattice, species, frac, coords_are_cartesian=False)
        i, j, closest, n_close = _closest_site_pair(loose)
        if closest < PYMATGEN_DISTANCE_TOLERANCE:
            raise _overlap_error(loose, i, j, closest,
                                 f"    sites below the tolerance  : {n_close}\n")
        return loose
    try:
        return Structure(lattice, species, frac, coords_are_cartesian=False,
                         validate_proximity=check)
    except Exception as exc:
        text = str(exc).lower()
        if 'angstrom apart' not in text and 'proximity' not in text:
            raise
        loose = Structure(lattice, species, frac, coords_are_cartesian=False)
        i, j, closest, n_close = _closest_site_pair(loose)
        raise _overlap_error(loose, i, j, closest,
                             f"    sites below the tolerance  : {n_close}\n") from None


def _report_structure(structure, mode, info, coordinate_mode, species_mode, verbose):
    if not verbose:
        return
    lat = structure.lattice
    print(f"\nStructure information:")
    print(f"  Lattice mode: {mode} [{info.get('box_source', '?')}]")
    print(f"  Atoms: {len(structure):,}")
    print(f"  Composition: {structure.composition.formula}")
    print(f"  Lattice: a={lat.a:.4f}, b={lat.b:.4f}, c={lat.c:.4f} A")
    print(f"           alpha={lat.alpha:.3f}, beta={lat.beta:.3f}, "
          f"gamma={lat.gamma:.3f} deg")
    print(f"  Volume: {structure.volume:.4f} A^3 "
          f"({structure.volume / max(len(structure), 1):.3f} A^3 per atom)")
    print(f"  Coordinate mode: {coordinate_mode}; species mode: {species_mode}")


class Reflections:

    def __init__(self, two_theta, intensity, hkls, d_hkls):
        self.two_theta = np.asarray(two_theta, dtype=float)
        self.intensity = np.asarray(intensity, dtype=float)
        self.hkls = list(hkls) if hkls is not None else []
        self.d_hkls = list(d_hkls) if d_hkls is not None else []

    def __len__(self):
        return self.two_theta.size

    def subset(self, idx):
        idx = list(idx)
        return Reflections(self.two_theta[idx], self.intensity[idx],
                           [self.hkls[i] for i in idx] if self.hkls else [],
                           [self.d_hkls[i] for i in idx] if self.d_hkls else [])


def _check_reflection_cost(structure, wavelength, two_theta_max, params, verbose=True):
    volume = float(structure.volume)
    n_atoms = len(structure)
    d_min = wavelength / (2.0 * np.sin(np.radians(min(two_theta_max, 179.9) / 2.0)))
    if d_min <= 0:
        return
    n_hkl = (4.0 * np.pi / 3.0) * (1.0 / d_min) ** 3 * volume / 2.0
    terms = n_hkl * n_atoms

    warn_at = as_float(params, 'reflection_cost_warn', 5e7)
    limit = as_float(params, 'max_reflection_cost', 1e10)

    if terms > limit:
        raise ValueError(
            f"This cell is too large for a kinematical calculation to finish.\n\n"
            f"    atoms                      : {n_atoms:,}\n"
            f"    cell volume                : {volume:,.1f} A^3\n"
            f"    hkl inside the limiting sphere at 2theta = {two_theta_max:g} deg"
            f" : ~{n_hkl:,.0f}\n"
            f"    structure-factor terms     : ~{terms:.3g} "
            f"(limit {limit:.3g}, set by max_reflection_cost)\n\n"
            f"  The number of reflections grows with the CELL VOLUME and each one "
            f"costs a sum over\n"
            f"  every atom, so the work scales as volume x atoms. A big MD box is "
            f"the worst case for\n"
            f"  this method, and no amount of waiting fixes it.\n\n"
            f"  What to use instead:\n"
            f"    - XRD-Debye_Scattering  : sums over interatomic distances. "
            f"Independent of cell size,\n"
            f"                                 and the right model if the box has "
            f"free surfaces anyway.\n"
            f"    - XRD-ReciprocalSum     : samples reciprocal space on a grid "
            f"(the LAMMPS compute_xrd\n"
            f"                                 approach), which is what large "
            f"periodic boxes are meant for.\n\n"
            f"  To stay with this module, either cut two_theta_max (the cost falls "
            f"as sin^3), or run a\n"
            f"  smaller representative cell. Raise max_reflection_cost only if you "
            f"know what you are\n"
            f"  asking for.")

    if verbose and terms > warn_at:
        print(f"{WARN} large cell: ~{n_hkl:,.0f} reflections over {n_atoms:,} atoms "
              f"(~{terms:.3g} structure-factor terms). This may take a long time. "
              f"Reducing two_theta_max cuts the cost as sin^3; "
              f"XRD-ReciprocalSum handles large periodic boxes better.")


def calculate_reflections(structure, params, verbose=True):
    wavelength_input = params.get('wavelength', 'CuKa')
    if isinstance(wavelength_input, str) and wavelength_input.strip() in WAVELENGTHS:
        wavelength_input = wavelength_input.strip()
    if isinstance(wavelength_input, str) and wavelength_input not in WAVELENGTHS:
        try:
            wavelength_input = float(wavelength_input)
        except ValueError:
            raise ValueError(
                f"Unknown wavelength {wavelength_input!r}. Use a number in Angstrom or "
                f"one of: " + ", ".join(sorted(WAVELENGTHS)))
    wavelength_value = (WAVELENGTHS[wavelength_input]
                        if isinstance(wavelength_input, str) else float(wavelength_input))

    debye_waller_factors = parse_debye_waller(
        params, structure=structure,
        verbose=verbose and as_bool(params, 'apply_debye_waller', False))
    if not as_bool(params, 'apply_debye_waller', False):
        if debye_waller_factors and verbose:
            print(f"Note: debye_waller_factors were supplied but apply_debye_waller is "
                  f"off, so no thermal damping is applied.")
        debye_waller_factors = None
    elif not debye_waller_factors:
        print(f"{WARN} apply_debye_waller is on but no debye_waller_factors were "
              f"supplied, so every species keeps B = 0 and NO thermal damping is "
              f"applied. Add e.g. debye_waller_factors = Fe=0.35 (B_iso in A^2), or "
              f"set apply_debye_waller = no.")

    two_theta_min = as_float(params, 'two_theta_min', 10.0)
    two_theta_max = as_float(params, 'two_theta_max', 90.0)
    if two_theta_max <= two_theta_min:
        raise ValueError(f"two_theta_max ({two_theta_max}) must exceed "
                         f"two_theta_min ({two_theta_min})")

    _check_reflection_cost(structure, wavelength_value, two_theta_max, params, verbose)

    calc = XRDCalculator(wavelength=wavelength_input,
                         debye_waller_factors=debye_waller_factors)
    pattern = calc.get_pattern(structure, two_theta_range=(two_theta_min, two_theta_max),
                               scaled=False)

    refl = Reflections(pattern.x, pattern.y,
                       getattr(pattern, 'hkls', None), getattr(pattern, 'd_hkls', None))

    applied = ["unscaled kinematical reflection intensities "
               "(includes multiplicity and the Lorentz-polarization factor)"]
    if debye_waller_factors:
        applied.append("species-resolved Debye-Waller factor exp(-B s^2)")

    if len(refl) == 0:
        print(f"{WARN} no allowed Bragg reflections in "
              f"{two_theta_min:g}-{two_theta_max:g} deg for this lattice.")
        return refl, wavelength_value, applied

    inten, abs_desc = apply_absorption_correction(refl.intensity, refl.two_theta,
                                                  params, verbose=verbose)
    refl.intensity = np.asarray(inten, dtype=float)
    if abs_desc:
        applied.append(f"absorption correction [{abs_desc}]")

    scale_factor = as_float(params, 'scale_factor', 1.0)
    if scale_factor != 1.0:
        refl.intensity = refl.intensity * scale_factor
        applied.append(f"user scale_factor = {scale_factor:g}")

    if verbose:
        print(f"\nCalculated {len(refl)} reflections in "
              f"{two_theta_min:g}-{two_theta_max:g} deg "
              f"(lambda = {wavelength_value:.6f} A)")

    return refl, wavelength_value, applied


def build_profile(refl, params, verbose=True):
    x_min = as_float(params, 'two_theta_min', 10.0)
    x_max = as_float(params, 'two_theta_max', 90.0)
    padding = as_float(params, 'curve_padding_deg', 0.0)
    n_points = max(2, as_int(params, 'gaussian_points', 1000))
    x_dense = np.linspace(x_min - padding, x_max + padding, n_points)

    inst_sigma = (as_float(params, 'instrument_sigma_deg', 0.0)
                  if as_bool(params, 'apply_instrumental_broadening', False) else 0.0)
    plot_sigma = (as_float(params, 'plot_smoothing_sigma_deg', 0.0)
                  if as_bool(params, 'apply_plot_smoothing', False) else 0.0)
    base_sigma = as_float(params, 'gaussian_width', 0.0)

    total_sigma = float(np.sqrt(base_sigma ** 2 + inst_sigma ** 2 + plot_sigma ** 2))
    profile_model = 'gaussian_constant_sigma'

    y_dense = np.zeros_like(x_dense)
    if len(refl) == 0:
        if verbose:
            print(f"{WARN} no reflections: the curve is zero over the requested range.")
        return x_dense, y_dense, total_sigma, profile_model

    if total_sigma <= 0:
        step = x_dense[1] - x_dense[0]
        idx = np.clip(np.round((refl.two_theta - x_dense[0]) / step).astype(int),
                      0, n_points - 1)
        np.add.at(y_dense, idx, refl.intensity / step)
        if verbose:
            print("  Profile: no broadening (delta-like bins). Set gaussian_width or "
                  "enable apply_instrumental_broadening for a smooth curve.")
        return x_dense, y_dense, total_sigma, 'delta_bins'

    n_peaks = len(refl)
    max_elements = int(as_float(params, 'max_profile_matrix_elements', 2e7))
    norm = 1.0 / (total_sigma * np.sqrt(2.0 * np.pi))
    if n_peaks * n_points <= max_elements:
        X = x_dense[None, :] - refl.two_theta[:, None]
        G = np.exp(-0.5 * (X / total_sigma) ** 2) * norm
        y_dense = (refl.intensity[:, None] * G).sum(axis=0)
    else:
        for xp, yp in zip(refl.two_theta, refl.intensity):
            y_dense += yp * np.exp(-0.5 * ((x_dense - xp) / total_sigma) ** 2) * norm

    if verbose:
        parts = []
        if base_sigma:
            parts.append(f"base {base_sigma:g}")
        if inst_sigma:
            parts.append(f"instrument {inst_sigma:g}")
        if plot_sigma:
            parts.append(f"plot smoothing {plot_sigma:g}")
        print(f"  Profile: {profile_model}, sigma = {total_sigma:.4f} deg "
              f"(quadrature sum of {', '.join(parts)})")
        print(f"           note: a constant-sigma Gaussian does not model "
              f"angle-dependent instrument resolution, peak asymmetry, "
              f"crystallite-size or microstrain broadening.")

    return x_dense, y_dense, total_sigma, profile_model


def select_display_peaks(refl, params, verbose=True):
    if len(refl) == 0:
        return refl

    if 'min_intensity_fraction' in params:
        frac = as_float(params, 'min_intensity_fraction', 0.005)
    elif 'min_intensity_percent' in params:
        frac = as_float(params, 'min_intensity_percent', 0.5) / 100.0
    elif 'min_intensity' in params:
        frac = as_float(params, 'min_intensity', 0.5) / 100.0
        if verbose:
            print(f"  note: 'min_intensity' is interpreted as a PERCENT of the maximum "
                  f"({frac * 100:g}%). Use min_intensity_fraction or "
                  f"min_intensity_percent to be explicit.")
    else:
        frac = 0.005

    y = refl.intensity
    threshold = float(np.max(y)) * frac if y.size else 0.0
    keep = [i for i in range(len(refl)) if y[i] >= threshold]

    min_sep = as_float(params, 'min_peak_separation', 0.0)
    if min_sep > 0 and keep:
        order = sorted(keep, key=lambda i: -y[i])
        chosen = []
        for i in order:
            if all(abs(refl.two_theta[i] - refl.two_theta[j]) >= min_sep for j in chosen):
                chosen.append(i)
        keep = sorted(chosen)

    max_peaks = as_int(params, 'max_peaks', 0)
    if max_peaks > 0 and len(keep) > max_peaks:
        keep = sorted(sorted(keep, key=lambda i: -y[i])[:max_peaks])

    if verbose:
        print(f"  Display peak list: {len(keep)} of {len(refl)} reflections "
              f"(threshold {frac * 100:g}% of max"
              + (f", min separation {min_sep:g} deg" if min_sep > 0 else "")
              + (f", capped at {max_peaks}" if max_peaks > 0 else "") + ")")
        dropped = len(refl) - len(keep)
        if dropped:
            weakest = 100.0 * float(np.min(y[keep])) / float(np.max(y)) if keep else 0.0
            print(f"    {dropped} weaker reflection(s) are not labelled, the "
                  f"faintest kept being {weakest:.3g}% of max. They remain in the "
                  f"reflection list and in the curve - only this labelling list "
                  f"is filtered.")
            print(f"    For defect, superlattice or diffuse work set "
                  f"min_intensity_percent = 0 and max_peaks = 0 to label every "
                  f"reflection.")
    return refl.subset(keep)


def _header(params, wavelength_value, applied, extra=(), structure=None):
    if structure is not None:
        lat = structure.lattice
        lattice_line = (f"lattice used      : a={lat.a:.6f} b={lat.b:.6f} c={lat.c:.6f} "
                        f"alpha={lat.alpha:.4f} beta={lat.beta:.4f} gamma={lat.gamma:.4f} "
                        f"({len(structure)} sites, V = {structure.volume:.4f} A^3)")
    else:
        lattice_line = (f"lattice used      : a={as_float(params, 'a', 0):.6f} "
                        f"b={as_float(params, 'b', 0):.6f} c={as_float(params, 'c', 0):.6f} "
                        f"alpha={as_float(params, 'alpha', 90):.4f} "
                        f"beta={as_float(params, 'beta', 90):.4f} "
                        f"gamma={as_float(params, 'gamma', 90):.4f}")
    lines = [
        "Periodic kinematical XRD",
        f"structure file    : {params.get('xyz_file', '?')}",
        f"unit cell input   : a={as_float(params, 'a', 0):.6f} b={as_float(params, 'b', 0):.6f} "
        f"c={as_float(params, 'c', 0):.6f} alpha={as_float(params, 'alpha', 90):.4f} "
        f"beta={as_float(params, 'beta', 90):.4f} gamma={as_float(params, 'gamma', 90):.4f}",
        lattice_line,
        f"lattice mode      : {params.get('_lattice_mode', 'unit_cell')}",
        f"radiation         : {params.get('wavelength', 'CuKa')} "
        f"(lambda = {wavelength_value:.6f} A)",
        f"2theta range      : {as_float(params, 'two_theta_min', 10.0):g} to "
        f"{as_float(params, 'two_theta_max', 90.0):g} deg",
        "correction order  :",
    ]
    for i, step in enumerate(applied, 1):
        lines.append(f"   {i}. {step}")
    lines.extend(extra)
    return lines


def save_reflection_list(refl, filename, params, wavelength_value, applied,
                         normalisation_note, label, structure=None):
    prec = as_int(params, 'float_precision', 4)
    with open(filename, 'w', encoding='utf-8') as f:
        for line in _header(params, wavelength_value, applied, structure=structure,
                            extra=[f"content           : {label}",
                                   f"normalisation     : {normalisation_note}",
                                   "intensities       : INTEGRATED reflection intensities"]):
            f.write(f"# {line}\n")
        f.write("#\n")
        f.write("# ID   2Theta(deg)      Intensity        d(A)      Multiplicity  hkl\n")
        f.write("# " + "-" * 78 + "\n")
        for i in range(len(refl)):
            hkls_entry = refl.hkls[i] if i < len(refl.hkls) else None
            mult = hkl_multiplicity(hkls_entry)
            hkl_str = format_hkls(hkls_entry, max_show=4) or "N/A"
            d = refl.d_hkls[i] if i < len(refl.d_hkls) else 0.0
            f.write(f"{i + 1:4d} {refl.two_theta[i]:14.{prec}f} "
                    f"{refl.intensity[i]:16.{prec}f} {d:12.{prec}f} "
                    f"{mult:10d}   {hkl_str}\n")
    print(f"{OK} saved {label} to {filename}")


def save_curve_data(x_dense, y_dense, filename, params, wavelength_value, applied,
                    sigma, profile_model, normalisation_note, structure=None):
    prec = as_int(params, 'float_precision', 4)
    with open(filename, 'w', encoding='utf-8') as f:
        for line in _header(params, wavelength_value, applied, structure=structure, extra=[
                f"profile model     : {profile_model}, sigma = {sigma:.6f} deg 2theta",
                f"                    area-normalised Gaussian; profile area is conserved",
                f"points            : {len(x_dense)}",
                f"normalisation     : {normalisation_note}",
                "intensities       : intensity DENSITY per degree 2theta "
                "(integrate over 2theta to recover an integrated intensity)"]):
            f.write(f"# {line}\n")
        f.write("#\n")
        f.write("# 2Theta(deg)     Intensity\n")
        for x, y in zip(x_dense, y_dense):
            f.write(f"{x:{prec + 8}.{prec}f} {y:{prec + 12}.{prec}f}\n")
    print(f"{OK} saved continuous curve to {filename}")


def plot_pattern(refl_full, refl_display, x_dense, y_dense, params, wavelength_value,
                 normalised):
    try:
        figsize = params.get('plot_figsize', (12, 6))
        if isinstance(figsize, (list, tuple)) and len(figsize) >= 2:
            figsize = (float(figsize[0]), float(figsize[1]))
        else:
            figsize = (12.0, 6.0)

        plt.figure(figsize=figsize)
        plt.plot(x_dense, y_dense, 'b-', linewidth=1.8, label='XRD pattern')

        x_left = as_float(params, 'two_theta_min', 10.0)
        x_right = as_float(params, 'two_theta_max', 90.0)

        marker_mode = as_str(params, 'marker_at', 'smoothed').lower()
        if marker_mode not in ('sticks', 'smoothed'):
            marker_mode = 'smoothed'

        marker_x, marker_y, marker_labels = [], [], []
        if len(refl_display):
            if marker_mode == 'smoothed' and y_dense.size > 2 and np.max(y_dense) > 0:
                prom = as_float(params, 'smoothed_peak_prominence', 0.01) * np.max(y_dense)
                dx = x_dense[1] - x_dense[0]
                dist = max(1, int(round(as_float(params, 'smoothed_peak_distance_deg', 0.5) / dx)))
                idx, _ = find_peaks(y_dense, prominence=prom, distance=dist)
                tol = as_float(params, 'smoothed_peak_match_tol_deg', 0.5)
                for j in idx:
                    mx = x_dense[j]
                    k = int(np.argmin(np.abs(refl_display.two_theta - mx)))
                    if abs(refl_display.two_theta[k] - mx) <= tol:
                        marker_x.append(mx)
                        marker_y.append(y_dense[j])
                        marker_labels.append(
                            format_hkls(refl_display.hkls[k]
                                        if k < len(refl_display.hkls) else None))
            if not marker_x:
                step = x_dense[1] - x_dense[0] if x_dense.size > 1 else 1.0
                for k in range(len(refl_display)):
                    mx = refl_display.two_theta[k]
                    j = int(np.clip(round((mx - x_dense[0]) / step), 0, len(x_dense) - 1))
                    marker_x.append(mx)
                    marker_y.append(y_dense[j])
                    marker_labels.append(
                        format_hkls(refl_display.hkls[k]
                                    if k < len(refl_display.hkls) else None))

        if marker_x:
            plt.scatter(marker_x, marker_y, color='red', s=70, marker='^',
                        alpha=0.9, zorder=6, label='indexed peaks')

        plt.xlim(x_left, x_right)
        y_top = float(np.max(y_dense)) if y_dense.size and np.max(y_dense) > 0 else 1.0
        plt.ylim(0.0, y_top * (1.0 + as_float(params, 'plot_y_headroom_frac', 0.10)))

        offset = as_int(params, 'annotation_offset_points', 8)
        for mx, my, lab in zip(marker_x, marker_y, marker_labels):
            if lab:
                plt.annotate(lab, xy=(mx, my), xytext=(0, offset),
                             textcoords='offset points', ha='center', va='bottom',
                             fontsize=9, fontweight='bold', color='darkgreen')

        plt.xlabel("2theta (degrees)", fontsize=13)
        plt.ylabel("Intensity (normalised)" if normalised
                   else "Intensity density (arb. units per degree)", fontsize=13)
        plt.title(f"Kinematical XRD - {params.get('wavelength', 'CuKa')} "
                  f"(lambda = {wavelength_value:.5f} A)", fontsize=13)

        if as_bool(params, 'show_grid', True):
            plt.grid(True, alpha=as_float(params, 'grid_alpha', 0.3), linestyle='--')
        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()

        plot_file = as_str(params, 'plot_filename', 'xrd_plot.png') or 'xrd_plot.png'
        plt.savefig(plot_file, dpi=as_int(params, 'plot_dpi', 150), bbox_inches='tight')
        print(f"{OK} saved plot to {plot_file}")
        if as_bool(params, 'show_plot', False):
            plt.show()
        plt.close()
        return True
    except Exception as e:
        print(f"{ERR} plotting failed: {e}")
        plt.close('all')
        return False


def print_summary(refl_full, refl_display, structure, params, wavelength_value):
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  Lattice mode         : {params.get('_lattice_mode', 'unit_cell')} "
          f"[{params.get('_box_source', '?')}]")
    print(f"  Atoms in cell        : {len(structure):,}")
    print(f"  Composition          : {structure.composition.formula}")
    print(f"  Cell parameters      : a={structure.lattice.a:.4f}, "
          f"b={structure.lattice.b:.4f}, c={structure.lattice.c:.4f} A")
    print(f"  Cell volume          : {structure.volume:.4f} A^3 "
          f"({structure.volume / max(len(structure), 1):.3f} A^3 per atom)")
    print(f"  Radiation            : {params.get('wavelength', 'CuKa')} "
          f"(lambda = {wavelength_value:.6f} A)")
    print(f"  Reflections computed : {len(refl_full)}")
    print(f"  Reflections labelled : {len(refl_display)}")
    if len(refl_full) == 0:
        return
    n_show = min(10, len(refl_full))
    order = np.argsort(refl_full.intensity)[::-1][:n_show]
    print(f"\n  Top {n_show} reflections by integrated intensity:")
    print(f"  {'rank':>4}  {'2theta':>10}  {'intensity':>14}  {'d(A)':>9}  "
          f"{'mult':>5}  hkl")
    imax = float(np.max(refl_full.intensity))
    for rank, i in enumerate(order, 1):
        hkls_entry = refl_full.hkls[i] if i < len(refl_full.hkls) else None
        d = refl_full.d_hkls[i] if i < len(refl_full.d_hkls) else 0.0
        print(f"  {rank:>4}  {refl_full.two_theta[i]:10.4f}  "
              f"{refl_full.intensity[i]:14.4f}  {d:9.4f}  "
              f"{hkl_multiplicity(hkls_entry):5d}  "
              f"{format_hkls(hkls_entry, max_show=3) or 'N/A'}"
              f"   ({100 * refl_full.intensity[i] / imax:5.1f}% of max)")

def cif_structure(structure, params, verbose=True):
    mode = as_str(params, 'cif_content', 'primitive').lower()
    if mode not in ('primitive', 'conventional', 'as_used'):
        print(f"{WARN} unknown cif_content {mode!r}; using 'primitive'")
        mode = 'primitive'
    if mode == 'as_used' or len(structure) < 2:
        return structure, "the structure exactly as calculated"

    symprec = as_float(params, 'cif_symprec', 0.01)
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        reduced = (sga.get_primitive_standard_structure() if mode == 'primitive'
                   else sga.get_conventional_standard_structure())
        group = sga.get_space_group_symbol()
    except Exception as exc:
        return structure, (f"the structure as calculated; symmetry analysis failed "
                           f"({type(exc).__name__})")

    if not len(reduced) or len(reduced) >= len(structure):
        return structure, (f"the structure as calculated; it does not reduce at "
                           f"symprec = {symprec:g} ({group})")

    factor = len(structure) / len(reduced)
    per_atom_before = structure.volume / len(structure)
    per_atom_after = reduced.volume / len(reduced)
    exact = abs(factor - round(factor)) < 1e-6
    same_density = abs(per_atom_after - per_atom_before) <= 1e-3 * per_atom_before
    if not (exact and same_density):
        return structure, (f"the structure as calculated; the {mode} cell would "
                           f"not reproduce it exactly "
                           f"({factor:.4g} atoms per reduced atom, "
                           f"{per_atom_after:.4f} vs {per_atom_before:.4f} A^3 each)")

    return reduced, (f"the {mode} cell: {len(structure)} atoms -> {len(reduced)}, "
                     f"exactly {round(factor)}x, space group {group} at "
                     f"symprec = {symprec:g}")


def run_kinematical_xrd(params):
    params = sanitize_params(params)

    print("\n" + "=" * 70)
    print("KINEMATICAL XRD CALCULATION (periodic crystal)")
    print("=" * 70)

    required = ['xyz_file', 'wavelength']
    if not validate_params(params, required, "Kinematical XRD"):
        return False
    if as_bool(params, 'experimental_correction', False):
        enabled = []
        if params.get('debye_waller_factors') not in (None, '', False):
            if 'apply_debye_waller' not in params:
                params['apply_debye_waller'] = True
                enabled.append('Debye-Waller (factors were supplied)')
        if 'instrument_sigma_deg' in params:
            if 'apply_instrumental_broadening' not in params:
                params['apply_instrumental_broadening'] = True
                enabled.append(f"instrumental broadening "
                               f"(sigma = {as_float(params, 'instrument_sigma_deg', 0):g} deg)")
        print("experimental_correction: " +
              ("; ".join(enabled) if enabled else "nothing to enable"))
        print("  Absorption is NEVER enabled automatically: it needs a material-specific "
              "mu and an explicit geometry. Set apply_absorption = yes to use it.")

    structure, diagnostics = build_structure(params, verbose=True)

    refl_full, wavelength_value, applied = calculate_reflections(structure, params,
                                                                 verbose=True)

    if len(refl_full) == 0:
        x_dense, y_dense, sigma, model = build_profile(refl_full, params, verbose=True)
        save_curve_data(x_dense, y_dense, as_str(params, 'curve_file', 'xrd_curve.txt'),
                        params, wavelength_value, applied, sigma, model, "none",
                        structure=structure)
        print(f"{WARN} no reflections were found in the requested interval; a zero "
              f"curve was written. Check the lattice parameters and the 2theta range.")
        return True

    x_dense, y_dense, sigma, profile_model = build_profile(refl_full, params, verbose=True)
    refl_display = select_display_peaks(refl_full, params, verbose=True)

    mode = as_str(params, 'normalize_mode', '').lower()
    if not mode:
        mode = 'curve_max_100' if as_bool(params, 'normalize_intensity', False) else 'none'
    if mode not in ('none', 'curve_max_100', 'reflection_max_100'):
        print(f"{WARN} unknown normalize_mode {mode!r}; using 'none'")
        mode = 'none'

    target = as_float(params, 'normalize_max', 100.0)
    normalisation_note = "none (raw calculated intensities)"
    normalised = False
    if mode == 'curve_max_100' and y_dense.size and np.max(y_dense) > 0:
        s = target / float(np.max(y_dense))
        y_dense = y_dense * s
        normalisation_note = (f"continuous curve scaled so its maximum is {target:g} "
                              f"(factor {s:.6e}); the reflection list is NOT scaled")
        normalised = True
    elif mode == 'reflection_max_100' and len(refl_full) and np.max(refl_full.intensity) > 0:
        s = target / float(np.max(refl_full.intensity))
        refl_full.intensity = refl_full.intensity * s
        refl_display.intensity = refl_display.intensity * s
        y_dense = y_dense * s
        normalisation_note = (f"reflection list scaled so its maximum is {target:g} "
                              f"(factor {s:.6e}); the curve uses the same factor")
        normalised = True
    if normalised:
        print(f"  Normalisation: {normalisation_note}")

    if diagnostics:
        applied = applied + [f"structure diagnostics: {d}" for d in diagnostics]

    save_reflection_list(refl_full,
                         as_str(params, 'reflections_file', 'xrd_reflections_full.txt'),
                         params, wavelength_value, applied, normalisation_note,
                         "COMPLETE calculated reflection list (unfiltered)",
                         structure=structure)
    save_reflection_list(refl_display,
                         as_str(params, 'peaks_file', 'xrd_peaks_display.txt'),
                         params, wavelength_value, applied, normalisation_note,
                         "filtered DISPLAY peak list (for labelling/tabulation only)",
                         structure=structure)
    save_curve_data(x_dense, y_dense, as_str(params, 'curve_file', 'xrd_curve.txt'),
                    params, wavelength_value, applied, sigma, profile_model,
                    normalisation_note, structure=structure)

    if as_bool(params, 'save_cif', False):
        cif_file = as_str(params, 'cif_file', 'structure.cif') or 'structure.cif'
        try:
            to_write, how = cif_structure(structure, params, verbose=True)
            to_write.to(filename=cif_file)
            print(f"{OK} saved structure to {cif_file} "
                  f"({len(to_write)} sites, V = {to_write.volume:.4f} A^3, "
                  f"lattice mode {params.get('_lattice_mode', 'unit_cell')})")
            print(f"     contains {how}.")
            if len(to_write) < len(structure):
                print(f"     The pattern was computed from all "
                      f"{len(structure):,} atoms; set cif_content = as_used to "
                      f"write those instead.")
        except Exception as e:
            print(f"{ERR} could not save CIF: {e}")

    if as_bool(params, 'make_plot', True):
        plot_pattern(refl_full, refl_display, x_dense, y_dense, params,
                     wavelength_value, normalised)

    if as_bool(params, 'show_structure_info', True):
        print_summary(refl_full, refl_display, structure, params, wavelength_value)

    return True

def self_test():
    print("=" * 70)
    print("SELF-TEST: XRD-Kinematical")
    print("=" * 70)
    ok = fail = 0

    def check(name, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
        else:
            fail += 1
            print(f"  FAIL  {name}  {detail}")

    lat = Lattice.cubic(3.147)
    st = Structure(lat, ["Mo", "Mo"], [[0, 0, 0], [.5, .5, .5]])
    ref = XRDCalculator(wavelength="CuKa").get_pattern(st, two_theta_range=(10, 90),
                                                       scaled=False)
    p = {'xyz_file': 'x', 'a': 3.147, 'b': 3.147, 'c': 3.147,
         'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0, 'wavelength': 'CuKa',
         'two_theta_min': 10.0, 'two_theta_max': 90.0,
         'apply_debye_waller': False, 'apply_absorption': False,
         'apply_instrumental_broadening': False, 'normalize_intensity': False}
    refl, wl, applied = calculate_reflections(st, p, verbose=False)
    same = (np.allclose(refl.two_theta, ref.x) and np.allclose(refl.intensity, ref.y))
    check("uncorrected reflections", same,
          f"{len(refl)} reflections, lambda = {wl}")
    check("CuKa resolves to pymatgen's 1.54184", abs(wl - 1.54184) < 1e-9, f"{wl}")

    parsed = parse_input_file.__doc__ is not None
    st2 = Structure(Lattice.cubic(3.6), ["Cu", "Zn"], [[0, 0, 0], [.5, .5, .5]])
    for raw, label in ((['Zn=0.5', 'Cu=0.6'], "list (what the parser produces)"),
                       ("Zn=0.5,Cu=0.6", "string"),
                       ({'Zn': 0.5, 'Cu': 0.6}, "dict")):
        got = parse_debye_waller({'debye_waller_factors': raw}, structure=st2,
                                 verbose=False)
        check(f"debye_waller_factors as {label}", got == {'Zn': 0.5, 'Cu': 0.6}, f"{got}")
    check("no {'default': 0.005} fallback",
          parse_debye_waller({'debye_waller_factors': None}, verbose=False) is None)
    try:
        parse_debye_waller({'debye_waller_factors': 'Ag=0.5'}, structure=st2, verbose=False)
        check("DW factor for an absent element is rejected", False)
    except ValueError:
        check("DW factor for an absent element is rejected", True)

    p_dw = dict(p, debye_waller_factors={'Mo': 1.0}, apply_debye_waller=True)
    refl_dw, _, _ = calculate_reflections(st, p_dw, verbose=False)
    s = np.sin(np.radians(refl.two_theta / 2.0)) / wl
    ratio = refl_dw.intensity / refl.intensity
    expected = np.exp(-2.0 * 1.0 * s ** 2)
    check("Debye-Waller attenuates as exp(-2 B s^2)",
          np.allclose(ratio, expected, rtol=0.02),
          f"ratio {np.round(ratio, 4)} vs {np.round(expected, 4)}")

    for sig in (0.05, 0.2, 0.8):
        pp = dict(p, gaussian_width=sig, gaussian_points=8000,
                  two_theta_min=10.0, two_theta_max=90.0)
        x, y, s_used, _ = build_profile(refl, pp, verbose=False)
        area = np.trapezoid(y, x) if hasattr(np, 'trapezoid') else np.trapz(y, x)
        total = float(np.sum(refl.intensity))
        check(f"profile area conserved at sigma = {sig}",
              abs(area - total) / total < 0.02,
              f"area {area:.4g} vs sum(I) {total:.4g}")

    pp = dict(p, gaussian_width=0.1, two_theta_min=10.0, two_theta_max=90.0)
    x, y, _, _ = build_profile(refl, pp, verbose=False)
    check("curve spans the requested range even though the first reflection is at 40.5 deg",
          abs(x[0] - 10.0) < 1e-9 and abs(x[-1] - 90.0) < 1e-9,
          f"{x[0]:.3f} to {x[-1]:.3f}")

    empty = Reflections([], [], [], [])
    pe = dict(p, two_theta_min=5.0, two_theta_max=9.0, gaussian_width=0.1)
    xe, ye, _, _ = build_profile(empty, pe, verbose=False)
    check("empty reflection list gives a zero curve over the requested range",
          xe.size > 0 and np.all(ye == 0) and abs(xe[0] - 5.0) < 1e-9)

    tri = Lattice.from_parameters(5.0, 5.2, 7.1, 88.0, 91.0, 97.0)
    st3 = Structure(tri, ["Cu", "Zn"], [[0, 0, 0], [.31, .27, .44]])
    r3, _, _ = calculate_reflections(st3, dict(p, two_theta_min=10.0, two_theta_max=80.0),
                                     verbose=False)
    close = [(i, j) for i in range(len(r3)) for j in range(i + 1, len(r3))
             if abs(r3.two_theta[i] - r3.two_theta[j]) < 0.1]
    if close:
        pp3 = dict(p, gaussian_width=0.1, gaussian_points=8000,
                   two_theta_min=10.0, two_theta_max=80.0,
                   min_peak_separation=0.1, min_intensity_percent=0.5)
        x3, y3, _, _ = build_profile(r3, pp3, verbose=False)
        disp = select_display_peaks(r3, pp3, verbose=False)
        area3 = np.trapezoid(y3, x3) if hasattr(np, 'trapezoid') else np.trapz(y3, x3)
        check("curve keeps ALL reflections while the display list is filtered",
              abs(area3 - float(np.sum(r3.intensity))) / float(np.sum(r3.intensity)) < 0.02
              and len(disp) < len(r3),
              f"{len(r3)} reflections, {len(disp)} displayed, "
              f"{len(close)} overlapping pairs")
        i, j = close[0]
        strong = i if r3.intensity[i] >= r3.intensity[j] else j
        weak = j if strong == i else i
        tt_kept = set(np.round(disp.two_theta, 6))
        if r3.intensity[strong] >= float(np.max(r3.intensity)) * 0.005:
            check("the STRONGEST member of an overlapping cluster is the one kept",
                  round(float(r3.two_theta[strong]), 6) in tt_kept
                  or round(float(r3.two_theta[weak]), 6) not in tt_kept,
                  f"strong at {r3.two_theta[strong]:.4f}, weak at {r3.two_theta[weak]:.4f}")
    else:
        print("  SKIP  overlapping-reflection test (no pair closer than 0.1 deg)")

    d1 = select_display_peaks(refl, dict(p, min_intensity_percent=0.5), verbose=False)
    d2 = select_display_peaks(refl, dict(p, min_intensity_fraction=0.005), verbose=False)
    check("min_intensity_percent and min_intensity_fraction agree",
          len(d1) == len(d2), f"{len(d1)} vs {len(d2)}")

    R = 0.05
    for muR in (0.01, 0.02, 0.05):
        A = float(absorption_cylinder(muR / R, R, np.array([0.0]))[0])
        analytic = 1.0 - 16.0 * muR / (3.0 * np.pi)
        check(f"cylinder absorption matches 1 - 16 muR/(3 pi) at muR = {muR}",
              abs(A - analytic) < 5e-3, f"numeric {A:.6f} vs analytic {analytic:.6f}")
    A0 = absorption_cylinder(1e-9, R, np.array([0.0, 45.0, 90.0]))
    check("cylinder absorption -> 1 as mu -> 0", np.allclose(A0, 1.0, atol=1e-6),
          f"{np.round(A0, 8)}")
    A1 = absorption_cylinder(0.2 / R, R, np.array([0.0, 90.0, 180.0]))
    check("cylinder transmission increases with 2theta",
          A1[0] < A1[1] < A1[2], f"{np.round(A1, 5)}")
    Abb = absorption_bragg_brentano(1e-9, 0.1, np.array([20.0, 60.0]))
    check("Bragg-Brentano absorption -> 1 as mu -> 0", np.allclose(Abb, 1.0, atol=1e-6))
    check("unknown absorption_geometry is rejected",
          _raises(lambda: apply_absorption_correction(
              np.array([1.0]), np.array([30.0]),
              {'apply_absorption': True, 'linear_absorption_coefficient': 100,
               'absorption_geometry': 'flat_plate'}, verbose=False), ValueError))

    import tempfile

    def _write_xyz(path, symbols, xyz, lattice_line=None):
        eol = chr(10)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(str(len(symbols)) + eol)
            fh.write((lattice_line or "generated by the self-test") + eol)
            for sym, (x, y, z) in zip(symbols, xyz):
                fh.write(f"{sym} {x:.8f} {y:.8f} {z:.8f}" + eol)

    def _b2(n, a=2.9206, noise=0.0, nvac=0, rot=0.0, seed=0):
        rng = np.random.default_rng(seed)
        pts, sym = [], []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for b, e in ((np.zeros(3), "Cu"), (np.full(3, 0.5), "Zn")):
                        pts.append((np.array([i, j, k], dtype=float) + b) * a)
                        sym.append(e)
        pts = np.array(pts)
        if noise:
            pts = pts + rng.normal(0, noise, pts.shape)
        if nvac:
            keep = np.ones(len(pts), bool)
            keep[rng.choice(len(pts), nvac, replace=False)] = False
            pts, sym = pts[keep], [e for e, k in zip(sym, keep) if k]
        if rot:
            t = np.radians(rot)
            R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
            pts = pts @ R.T
        return sym, pts

    A = 2.9206
    base = {'species_mode': 'chemical_symbols', 'a': A, 'b': A, 'c': A,
            'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0, 'wavelength': 'CuKa',
            'coordinate_mode': 'cartesian', 'two_theta_min': 10.0,
            'two_theta_max': 90.0}
    _tmp_dir = tempfile.TemporaryDirectory(prefix='xrd_kin_selftest_')
    tmp = _tmp_dir.name

    lat_line = f'Lattice="{4*A} 0.0 0.0 0.0 {4*A} 0.0 0.0 0.0 {4*A}" Properties=species:S:1:pos:R:3'
    f_box = os.path.join(tmp, 'boxed.xyz')
    sym, pts = _b2(4, A, noise=0.05, nvac=3, seed=1)
    _write_xyz(f_box, sym, pts, lat_line)
    m = parse_extxyz_lattice(f_box, verbose=False)
    check("extended-XYZ Lattice= is parsed",
          m is not None and np.allclose(m, np.eye(3) * 4 * A), f"{None if m is None else np.round(m.diagonal(), 4)}")
    f_plain = os.path.join(tmp, 'plain.xyz')
    _write_xyz(f_plain, sym, pts)
    check("a file with no Lattice= returns None",
          parse_extxyz_lattice(f_plain, verbose=False) is None)

    st_box, diag = build_structure(dict(base, xyz_file=f_box), verbose=False)
    check("a file box is used as the lattice and keeps every atom, defects included",
          len(st_box) == len(sym) and abs(st_box.lattice.a - 4 * A) < 1e-6,
          f"{len(st_box)} sites of {len(sym)} atoms, a = {st_box.lattice.a:.4f}")

    st_rec, _ = build_structure(dict(base, xyz_file=f_plain), verbose=False)
    check("a box is recovered for a defective supercell with no Lattice= entry",
          len(st_rec) == len(sym) and abs(st_rec.lattice.a - 4 * A) < 1e-6,
          f"{len(st_rec)} sites, a = {st_rec.lattice.a:.4f}")

    f_perfect = os.path.join(tmp, 'perfect.xyz')
    sym_p, pts_p = _b2(3, A)
    _write_xyz(f_perfect, sym_p, pts_p)
    st_fold, _ = build_structure(dict(base, xyz_file=f_perfect,
                                      lattice_mode='unit_cell'), verbose=False)
    check("a perfect aligned supercell folds and collapses onto the unit cell",
          len(st_fold) == 2 and abs(st_fold.volume - A ** 3) < 1e-6,
          f"{len(pts_p)} atoms -> {len(st_fold)} sites, V = {st_fold.volume:.4f} A^3")
    check("the collapsed cell has one Cu and one Zn",
          st_fold.composition.reduced_formula in ('CuZn', 'ZnCu'),
          f"{st_fold.composition.formula}")

    f_clu = os.path.join(tmp, 'cluster.xyz')
    sym_c, pts_c = _b2(8, A, noise=0.05, rot=23.0, seed=2)
    centre = pts_c.mean(axis=0)
    keep = np.linalg.norm(pts_c - centre, axis=1) < 8.0
    _write_xyz(f_clu, [e for e, k in zip(sym_c, keep) if k], pts_c[keep])
    try:
        build_structure(dict(base, xyz_file=f_clu), verbose=False)
        check("a finite cluster is rejected with a pointer to the Debye module", False)
    except ValueError as exc:
        check("a finite cluster is rejected with a pointer to the Debye module",
              'Debye' in str(exc), f"{str(exc).splitlines()[0][:58]}")
    check("allow_aperiodic no longer forces a pattern out of a cluster",
          _raises(lambda: build_structure(
              dict(base, xyz_file=f_clu, allow_aperiodic=True,
                   validate_proximity=False), verbose=False), ValueError))

    f_aligned = os.path.join(tmp, 'aligned_cluster.xyz')
    sym_a, pts_a = _b2(8, A, noise=0.05, seed=3)
    centre = pts_a.mean(axis=0)
    keep = np.linalg.norm(pts_a - centre, axis=1) < 8.0
    _write_xyz(f_aligned, [e for e, k in zip(sym_a, keep) if k], pts_a[keep])
    try:
        build_structure(dict(base, xyz_file=f_aligned), verbose=False)
        check("an ALIGNED cluster is refused under lattice_mode = auto", False)
    except ValueError as exc:
        check("an ALIGNED cluster is refused under lattice_mode = auto",
              'Debye' in str(exc), f"{str(exc).splitlines()[0][:58]}")
    st_al, diag_al = build_structure(dict(base, xyz_file=f_aligned,
                                          lattice_mode='unit_cell'), verbose=False)
    check("lattice_mode = unit_cell still folds it, with the finite size flagged",
          len(st_al) == 2 and any('under-coordinated' in d for d in diag_al),
          f"{len(st_al)} sites; {sum('under-coordinated' in d for d in diag_al)} note(s)")

    v1 = A * np.array([.5, .5, .5])
    v2 = A * np.array([1.0, -1.0, 0.0])
    v3 = A * np.array([1.0, 1.0, -2.0])
    reps = (6, 3, 2)
    cell_c = np.array([reps[0] * v1, reps[1] * v2, reps[2] * v3])
    rot = np.array([v / np.linalg.norm(v) for v in (v1, v2, v3)])
    inv_c = np.linalg.inv(cell_c)
    grid = range(-14, 15)
    rp = [(np.array([i, j, k], float) + b) * A
          for i in grid for j in grid for k in grid
          for b in ([0, 0, 0], [.5, .5, .5])]
    rp = np.array([p for p in rp
                   if np.all(p @ inv_c >= -1e-9) and np.all(p @ inv_c < 1 - 1e-9)])
    true_box = np.array([reps[i] * np.linalg.norm(v) for i, v in enumerate((v1, v2, v3))])
    rp = rp @ rot.T + np.random.default_rng(5).normal(0, 0.04, (len(rp), 3))
    f_rot = os.path.join(tmp, 'rotated_periodic.xyz')
    _write_xyz(f_rot, ['Mo'] * len(rp), rp)

    no_cell = {k: v for k, v in base.items()
               if k not in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')}
    st_rot, diag_rot = build_structure(dict(no_cell, xyz_file=f_rot), verbose=False)
    got_box = np.sort(np.array([st_rot.lattice.a, st_rot.lattice.b, st_rot.lattice.c]))
    err = float(np.max(np.abs(got_box - np.sort(true_box)) / np.sort(true_box)))
    check("a rotated periodic cell is recovered with NO cell information at all",
          len(st_rot) == len(rp) and err < 0.01,
          f"{len(st_rot)} atoms, box {np.round(got_box, 4)} vs "
          f"{np.round(np.sort(true_box), 4)}, max error {100 * err:.2f}%")

    ref_bcc = XRDCalculator(wavelength="CuKa").get_pattern(
        Structure(Lattice.cubic(A), ["Mo", "Mo"], [[0, 0, 0], [.5, .5, .5]]),
        two_theta_range=(10, 90), scaled=False)
    r_rot, _, _ = calculate_reflections(st_rot, dict(p, two_theta_min=10.0,
                                                     two_theta_max=90.0), verbose=False)
    strong = r_rot.two_theta[r_rot.intensity > 0.01 * r_rot.intensity.max()]
    drift = max(float(np.min(np.abs(ref_bcc.x - t))) for t in strong)
    check("its reflections land on the ideal bcc lines, with no forbidden peaks",
          drift < 0.15, f"{len(strong)} reflections, worst offset {drift:.3f} deg "
          f"from {np.round(ref_bcc.x, 2)}")

    lat_rot = 'Lattice="%.6f 0 0 0 %.6f 0 0 0 %.6f"' % tuple(true_box)
    f_rot_box = os.path.join(tmp, 'rotated_periodic_boxed.xyz')
    _write_xyz(f_rot_box, ['Mo'] * len(rp), rp, lat_rot)
    st_rb, _ = build_structure(dict(no_cell, xyz_file=f_rot_box), verbose=False)
    check("the file's own cell is preferred over inference and agrees with it",
          abs(st_rb.volume - float(np.prod(true_box))) < 1e-2
          and abs(st_rb.volume - st_rot.volume) / st_rb.volume < 0.02,
          f"V = {st_rb.volume:.3f} (file) vs {st_rot.volume:.3f} (inferred) A^3")

    check("ASE reads the cell of an extended-XYZ file",
          (lambda m: m is not None and abs(np.linalg.det(m)
                                           - float(np.prod(true_box))) < 1e-3)(
              read_cell_with_ase(f_rot_box, verbose=False)))
    check("a cell_file supplies the box when the structure file has none",
          abs(build_structure(dict(no_cell, xyz_file=f_rot, cell_file=f_rot_box),
                              verbose=False)[0].volume
              - float(np.prod(true_box))) < 1e-2)

    cut_rp, r1_rp = first_shell_cutoff(rp, None)
    prim, prim_info = infer_crystal_lattice(rp, ['Mo'] * len(rp), cut_rp, r1_rp,
                                            {}, verbose=False)
    check("the inferred lattice has the right volume per atom",
          prim is not None
          and abs(abs(np.linalg.det(prim)) - A ** 3 / 2) / (A ** 3 / 2) < 0.02,
          f"V = {abs(np.linalg.det(prim)):.4f} vs {A ** 3 / 2:.4f} A^3 per lattice point")

    sym_b2, pts_b2 = _b2(4, A, noise=0.03, seed=11)
    cut_b2, r1_b2 = first_shell_cutoff(pts_b2, None)
    prim_b2, _ = infer_crystal_lattice(pts_b2, sym_b2, cut_b2, r1_b2, {}, verbose=False)
    check("species-aware inference finds the SIMPLE CUBIC cell of B2, not bcc",
          prim_b2 is not None
          and abs(abs(np.linalg.det(prim_b2)) - A ** 3) / A ** 3 < 0.02,
          f"V = {abs(np.linalg.det(prim_b2)):.4f} vs {A ** 3:.4f} A^3")

    sym_sl, pts_sl = _b2(6, A, noise=0.04, seed=12)
    pts_sl = pts_sl[pts_sl[:, 2] < 3.2 * A]
    cut_sl, r1_sl = first_shell_cutoff(pts_sl, None)
    _, cn_sl, _ = surface_fraction(pts_sl, None, cut_sl)
    _, slab_info = recover_box_from_coordinates(
        pts_sl, ['Cu'] * len(pts_sl), {}, cut_sl, r1_sl,
        float(np.percentile(cn_sl, 90)), verbose=False)
    check("a slab is reported as periodic in x and y but not z",
          slab_info.get('periodic_axes') == [0, 1],
          f"periodic axes {slab_info.get('periodic_axes')}, "
          f"scores {np.round(slab_info.get('axis_scores', []), 3)}")

    f_wrong = os.path.join(tmp, 'wrong_constant.xyz')
    a_true = 2.8400
    wrong = np.array([(np.array([i, j, k], float) + b) * a_true
                      for i in range(10) for j in range(10) for k in range(10)
                      for b in ([0, 0, 0], [.5, .5, .5])])
    _write_xyz(f_wrong, ['Fe'] * len(wrong), wrong)
    check("a fold into a WRONG lattice constant is refused, not chained together",
          _raises(lambda: build_structure(
              dict(base, xyz_file=f_wrong, a=A, b=A, c=A, lattice_mode='unit_cell',
                   validate_proximity=False), verbose=False), ValueError),
          f"true a = {a_true}, supplied a = {A}")
    st_true, _ = build_structure(dict(base, xyz_file=f_wrong, a=a_true, b=a_true,
                                      c=a_true), verbose=False)
    check("the same file with the RIGHT constant recovers its 10x10x10 box",
          len(st_true) == len(wrong)
          and abs(st_true.lattice.a - 10 * a_true) / (10 * a_true) < 0.01,
          f"{len(st_true)} atoms, a = {st_true.lattice.a:.4f} vs {10 * a_true:.4f}")

    a_fe = 2.84005
    NREP = 4
    perfect = np.array([(np.array([i, j, k], float) + b) / NREP
                        for i in range(NREP) for j in range(NREP) for k in range(NREP)
                        for b in ([0, 0, 0], [.5, .5, .5])])
    box_fe = Lattice.cubic(NREP * a_fe)
    st_perfect = Structure(box_fe, ['Fe'] * len(perfect), perfect)

    red, how = cif_structure(st_perfect, {}, verbose=False)
    check("a perfect supercell is reduced for the CIF, losslessly",
          len(red) == 1 and abs(red.volume - a_fe ** 3 / 2) < 1e-6,
          f"{len(st_perfect)} atoms -> {len(red)}; {how[:60]}")

    r_full, _, _ = calculate_reflections(st_perfect, dict(p, two_theta_min=10.0,
                                                          two_theta_max=90.0), verbose=False)
    r_red, _, _ = calculate_reflections(red, dict(p, two_theta_min=10.0,
                                                  two_theta_max=90.0), verbose=False)
    strong_full = np.sort(r_full.two_theta[r_full.intensity > 0.01 * r_full.intensity.max()])
    strong_red = np.sort(r_red.two_theta[r_red.intensity > 0.01 * r_red.intensity.max()])
    check("the reduced CIF cell reproduces the pattern of the full box",
          len(strong_full) == len(strong_red)
          and np.allclose(strong_full, strong_red, atol=1e-6),
          f"{np.round(strong_red, 4)}")

    rng_cif = np.random.default_rng(21)
    for label, build in (
            ("vacancies", lambda: (perfect[np.array([i not in set(
                rng_cif.choice(len(perfect), 4, replace=False)) for i in range(len(perfect))])],
                None)),
            ("thermal displacement", lambda: (
                perfect + rng_cif.normal(0, 0.05 / (NREP * a_fe), perfect.shape), None)),
            ("substitutions", lambda: (perfect, ['Cr' if i % 31 == 0 else 'Fe'
                                                 for i in range(len(perfect))]))):
        coords, syms = build()
        st_def = Structure(box_fe, syms or ['Fe'] * len(coords), coords)
        kept, why = cif_structure(st_def, {}, verbose=False)
        check(f"a supercell with {label} is NOT reduced for the CIF",
              len(kept) == len(st_def), f"{len(kept)} of {len(st_def)} kept; {why[:52]}")

    kept, _ = cif_structure(st_perfect, {'cif_content': 'as_used'}, verbose=False)
    check("cif_content = as_used writes every atom",
          len(kept) == len(st_perfect), f"{len(kept)} sites")
    conv, how_c = cif_structure(st_perfect, {'cif_content': 'conventional'}, verbose=False)
    check("cif_content = conventional gives the 2-atom cubic cell",
          len(conv) == 2 and abs(conv.lattice.a - a_fe) < 1e-4,
          f"{len(conv)} sites, a = {conv.lattice.a:.4f}")

    f_tri = os.path.join(tmp, 'triclinic.xyz')
    tri_lat = Lattice.from_parameters(5.0, 5.2, 7.1, 88.0, 91.0, 97.0)
    basis = np.array([[0.0, 0.0, 0.0], [0.31, 0.27, 0.44]])
    tri_frac = np.array([basis + [i, j, k] for i in range(2) for j in range(2)
                         for k in range(2)]).reshape(-1, 3)
    _write_xyz(f_tri, ["Cu", "Zn"] * 8, tri_lat.get_cartesian_coords(tri_frac))
    tri_params = dict(base, xyz_file=f_tri, a=5.0, b=5.2, c=7.1,
                      alpha=88.0, beta=91.0, gamma=97.0, lattice_mode='unit_cell')
    st_tri, _ = build_structure(tri_params, verbose=False)
    check("folding and collapse work for a TRICLINIC cell (not just cubic)",
          len(st_tri) == 2 and abs(st_tri.volume - tri_lat.volume) < 1e-6,
          f"16 atoms -> {len(st_tri)} sites, V = {st_tri.volume:.4f} A^3")

    big_bcc = _b2(6, A)[1]
    cut_bcc, r1_bcc = first_shell_cutoff(big_bcc, np.eye(3) * 6 * A)
    cn_bcc = surface_fraction(big_bcc, np.eye(3) * 6 * A, cut_bcc)[1]
    check("the first-shell cutoff closes a bcc coordination shell (8 or 8+6)",
          int(np.median(cn_bcc)) in (8, 14),
          f"r1 = {r1_bcc:.3f}, cutoff = {cut_bcc:.3f}, CN = {int(np.median(cn_bcc))}")

    fcc_a, n_fcc = 3.615, 5
    fcc = np.array([(np.array([i, j, k], dtype=float) + b) * fcc_a
                    for i in range(n_fcc) for j in range(n_fcc) for k in range(n_fcc)
                    for b in ([0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0])])
    fcc_box = np.eye(3) * n_fcc * fcc_a
    cut_fcc, r1_fcc = first_shell_cutoff(fcc, fcc_box)
    cn_fcc = surface_fraction(fcc, fcc_box, cut_fcc)[1]
    check("the first-shell cutoff closes an fcc coordination shell (12 or 12+6)",
          int(np.median(cn_fcc)) in (12, 18),
          f"r1 = {r1_fcc:.3f}, cutoff = {cut_fcc:.3f}, CN = {int(np.median(cn_fcc))}")

    f_open, _, _ = surface_fraction(pts_p, None)
    f_pbc, _, _ = surface_fraction(pts_p, np.eye(3) * 3 * A)
    check("the PBC-repair test separates a periodic cell from an open one",
          f_pbc < 0.05 <= f_open, f"open {100*f_open:.1f}%, boxed {100*f_pbc:.1f}%")

    try:
        _make_structure(Lattice.cubic(3.0), ["Cu", "Zn"],
                        np.array([[0, 0, 0], [0.05, 0, 0]]), {}, verbose=False)
        check("validate_proximity rejects overlapping sites", False)
    except ValueError as exc:
        check("validate_proximity rejects overlapping sites",
              'overlapping sites' in str(exc), f"{0.05*3:.2f} A apart")

    st_shift = Structure(lat, ["Mo", "Mo"], [[1.0, 2.0, -3.0], [1.5, 2.5, -2.5]])
    r_shift, _, _ = calculate_reflections(st_shift, p, verbose=False)
    check("a lattice-vector translation leaves intensities unchanged",
          np.allclose(np.sort(r_shift.intensity), np.sort(refl.intensity), rtol=1e-8))

    print("=" * 70)
    print(f"SELF-TEST: {ok} passed, {fail} failed")
    print("=" * 70)
    _tmp_dir.cleanup()
    return fail == 0


def _raises(fn, exc):
    try:
        fn()
        return False
    except exc:
        return True
    except Exception:
        return False

def main():
    if len(sys.argv) >= 2 and sys.argv[1] in ('--self-test', '-t'):
        sys.exit(0 if self_test() else 1)
    if len(sys.argv) < 2:
        print("Usage: python XRD-Kinematical input.txt")
        print("       python XRD-Kinematical --self-test")
        sys.exit(2)

    params = parse_input_file(sys.argv[1])
    try:
        ok = run_kinematical_xrd(params)
    except (ValueError, FileNotFoundError) as exc:
        print(f"\n{ERR} {exc}")
        sys.exit(3)
    sys.exit(0 if ok else 4)


if __name__ == "__main__":
    main()
