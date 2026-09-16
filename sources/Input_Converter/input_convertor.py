#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

from __future__ import annotations
import os
import re
import sys
import math
import numpy as np
from typing import List, Tuple, Optional
try:
    from ase import io as aseio
    from ase import Atoms
    from ase.io import write
except Exception:
    io = None
    Atoms = None
    write = None

import io

if sys.platform == "win32":
    if sys.stdout is not None and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if sys.stderr is not None and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)


PERIODIC_MASSES = {
 'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811,
 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797,
 'Na': 22.98976928, 'Mg': 24.3050, 'Al': 26.9815386, 'Si': 28.0855, 'P': 30.973762,
 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.0983, 'Ca': 40.078,
 'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938045,
 'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.38,
 'Ga': 69.723, 'Ge': 72.64, 'As': 74.92160, 'Se': 78.96, 'Br': 79.904,
 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585, 'Zr': 91.224,
 'Nb': 92.90638, 'Mo': 95.96, 'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.90550,
 'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.710,
 'Sb': 121.760, 'Te': 127.60, 'I': 126.90447, 'Xe': 131.293, 'Cs': 132.9054519,
 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116, 'Pr': 140.90765, 'Nd': 144.242,
 'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.92535,
 'Dy': 162.500, 'Ho': 164.93032, 'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.04,
 'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.94788, 'W': 183.84, 'Re': 186.207,
 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.59,
 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.98040, 'Po': 209.0, 'At': 210.0,
 'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.03806,
 'Pa': 231.03588, 'U': 238.02891, 'Np': 237.0, 'Pu': 244.0, 'Am': 243.0,
 'Cm': 247.0, 'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0,
 'Md': 258.0, 'No': 259.0, 'Lr': 262.0, 'Rf': 261.0, 'Db': 262.0,
 'Sg': 266.0, 'Bh': 264.0, 'Hs': 277.0, 'Mt': 268.0, 'Ds': 271.0,
 'Rg': 272.0, 'Cn': 285.0, 'Nh': 284.0, 'Fl': 289.0, 'Mc': 288.0,
 'Lv': 293.0, 'Ts': 294.0, 'Og': 294.0
}

def prepend_vcl_header(filename: str):
    if get_output_format_from_extension(filename) == 'vasp':
        return  # POSCAR has no comment syntax: line 1 is the title, line 2 the scale factor
    header = "# This file is generated using Virtual Characterization Lab (VCL) Toolkit\n"
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(header + content)
    except Exception as e:
        print(f"Warning: Could not prepend header to {filename}: {e}")

def normalize_path_for_os(path: str) -> str:
    if not path:
        return path
    if os.name == "nt":
        if path.startswith("/mnt/") and len(path) > 7:
            drive = path[5]
            rest = path[7:].replace("/", "\\")
            return drive.upper() + ":\\" + rest
        return path
    if len(path) >= 3 and path[1] == ":":
        drive = path[0].lower()
        rest = path[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return path

def ask(prompt: str, default: Optional[str] = None) -> str:
    if default is not None:
        s = input(f"{prompt} [{default}]: ").strip()
        return s if s else default
    while True:
        s = input(f"{prompt}: ").strip()
        if s:
            return s
        print("This field is required.")

def build_type_mapping(symbols: List[str]) -> dict:
    mapping = {}
    for s in symbols:
        if s not in mapping:
            mapping[s] = len(mapping) + 1
    return mapping

def triangular_cell_components(cell: np.ndarray) -> Tuple[float,float,float,float,float,float]:
    a = np.array(cell[0], dtype=float); b = np.array(cell[1], dtype=float); c = np.array(cell[2], dtype=float)
    ax = np.linalg.norm(a)
    if ax == 0:
        raise RuntimeError("Cell vector a has zero length.")
    e1 = a / ax
    bx = np.dot(b, e1)
    b_perp = b - bx*e1
    by = np.linalg.norm(b_perp)
    if by > 1e-12:
        e2 = b_perp / by
    else:
        e2 = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(e2, e1)) > 0.9:
            e2 = np.array([0.0, 0.0, 1.0])
        e2 = e2 - np.dot(e2, e1)*e1
        e2 = e2 / np.linalg.norm(e2)
    cx = np.dot(c, e1)
    cy = np.dot(c, e2)
    c_perp = c - cx*e1 - cy*e2
    cz = np.linalg.norm(c_perp)
    return ax, bx, cx, by, cy, cz

def guess_masses_for_symbols(symbols: List[str]) -> dict:
    return {s: PERIODIC_MASSES.get(s, 1.0) for s in symbols}

def nearest_element_by_mass(mass: float) -> str:
    best = None; best_diff = float('inf')
    for sym,m in PERIODIC_MASSES.items():
        d = abs(m - mass)
        if d < best_diff:
            best_diff = d; best = sym
    return best or 'X'

def wrap_positions_in_place(atoms: Atoms):
    if Atoms is None:
        raise RuntimeError("ASE required.")
    try:
        scaled = atoms.get_scaled_positions(wrap=False)
    except TypeError:
        pos = atoms.get_positions()
        cell = np.array(atoms.get_cell(), dtype=float)
        invC = np.linalg.inv(cell)
        scaled = pos.dot(invC)
    scaled_wrapped = np.mod(scaled, 1.0)
    cell = np.array(atoms.get_cell(), dtype=float)
    atoms.set_positions(scaled_wrapped.dot(cell))

def orthogonalize_and_wrap_atoms(atoms: Atoms) -> Atoms:
    if Atoms is None:
        raise RuntimeError("ASE required.")
    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell(), dtype=float)
    vol = np.linalg.det(cell)
    if abs(vol) < 1e-12:
        raise RuntimeError("Input cell has near-zero volume or invalid cell.")
    invC = np.linalg.inv(cell)
    frac = pos.dot(invC)
    frac_wrapped = np.mod(frac, 1.0)
    lengths = np.linalg.norm(cell, axis=1)
    new_cell = np.diag(lengths)
    new_pos = frac_wrapped.dot(new_cell)
    new_atoms = Atoms(symbols=atoms.get_chemical_symbols(), positions=new_pos, cell=new_cell, pbc=True)
    for name, arr in getattr(atoms, "arrays", {}).items():
        if name in ("positions",):
            continue
        try:
            new_atoms.set_array(name, np.array(arr))
        except Exception:
            pass
    return new_atoms

def is_triclinic(cell: np.ndarray, tol=1e-8) -> bool:
    cm = np.array(cell, dtype=float)
    off = cm.copy(); np.fill_diagonal(off, 0.0)
    return np.any(np.abs(off) > tol)

def detect_charge_spin(atoms: Atoms) -> Tuple[bool,bool,bool]:
    arrs = getattr(atoms, "arrays", {})
    keys = set(arrs.keys())
    has_charge = any(k.lower() == 'charge' for k in keys)
    has_spin_any = any(k.lower().startswith('spin') for k in keys)
    has_spin_components = all(k in keys for k in ('Spin0','Spin1','Spin2','Spin3'))
    return has_charge, has_spin_any, has_spin_components

def write_lammps_data_from_ase(filename: str, atoms: Atoms, mode: int = 1):
    natoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = np.array(atoms.get_cell(), dtype=float)
    typemap = build_type_mapping(symbols)
    types = [typemap[s] for s in symbols]
    mass_map_symbol = guess_masses_for_symbols(list(typemap.keys()))
    masses_map = {typemap[sym]: mass_map_symbol[sym] for sym in typemap.keys()}
    ax, bx, cx, by, cy, cz = triangular_cell_components(cell)

    xlo, xhi = 0.0, ax
    ylo, yhi = 0.0, by
    zlo, zhi = 0.0, cz
    xy, xz, yz = bx, cx, cy

    with open(filename, 'w') as fh:
        fh.write("# This file is generated using Virtual Characterization Lab (VCL) Toolkit\n\n")
        fh.write(f"{natoms} atoms\n")
        fh.write(f"{len(typemap)} atom types\n\n")
        fh.write(f"{xlo:.12g} {xhi:.12g} xlo xhi\n")
        fh.write(f"{ylo:.12g} {yhi:.12g} ylo yhi\n")
        fh.write(f"{zlo:.12g} {zhi:.12g} zlo zhi\n")
        fh.write(f"{xy:.12g} {xz:.12g} {yz:.12g} xy xz yz\n\n")
        fh.write("Masses\n\n")
        for tid in sorted(masses_map.keys()):
            fh.write(f"{tid} {masses_map[tid]:.6f}\n")
        atom_style = {1: 'atomic', 2: 'charge'}.get(mode, 'spin')
        fh.write(f"\nAtoms # {atom_style}\n\n")
        for i in range(natoms):
            aid = i + 1
            typ = types[i]
            x, y, z = positions[i]
            arrs = getattr(atoms, "arrays", {})
            if mode == 1:
                fh.write(f"{aid} {typ} {x:.12g} {y:.12g} {z:.12g}\n")
            elif mode == 2:
                ch = 0.0
                if 'charge' in arrs:
                    ch = float(arrs['charge'][i])
                fh.write(f"{aid} {typ} {ch:.6f} {x:.12g} {y:.12g} {z:.12g}\n")
            else:
                s0 = s1 = s2 = s3 = 0.0
                if 'Spin0' in arrs:
                    s0 = float(arrs['Spin0'][i])
                if 'Spin1' in arrs:
                    s1 = float(arrs['Spin1'][i])
                if 'Spin2' in arrs:
                    s2 = float(arrs['Spin2'][i])
                if 'Spin3' in arrs:
                    s3 = float(arrs['Spin3'][i])
                fh.write(f"{aid} {typ} {x:.12g} {y:.12g} {z:.12g} {s0:.6f} {s1:.6f} {s2:.6f} {s3:.6f}\n")
    print(f"Wrote LAMMPS data: {filename}")

def write_lammps_dump(filename: str, frames: List[Atoms], mode: int = 1):
    if len(frames) == 0:
        raise RuntimeError("No frames to write.")
    with open(filename, 'w') as fh:
        for tstep, atoms in enumerate(frames):
            natoms = len(atoms)
            positions = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            typemap = build_type_mapping(symbols)
            types = [typemap[s] for s in symbols]
            cell = np.array(atoms.get_cell(), dtype=float)
            ax, bx, cx, by, cy, cz = triangular_cell_components(cell)
            xlo,xhi = 0.0, ax; ylo,yhi = 0.0, by; zlo,zhi = 0.0, cz
            xy, xz, yz = bx, cx, cy

            fh.write("ITEM: TIMESTEP\n"); fh.write(f"{tstep}\n")
            fh.write("ITEM: NUMBER OF ATOMS\n"); fh.write(f"{natoms}\n")
            if xy != 0.0 or xz != 0.0 or yz != 0.0:
                # Triclinic dump: bounding box of the tilted cell plus the tilt factors
                fh.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
                fh.write(f"{xlo + min(0.0, xy, xz, xy + xz):.12g} {xhi + max(0.0, xy, xz, xy + xz):.12g} {xy:.12g}\n")
                fh.write(f"{ylo + min(0.0, yz):.12g} {yhi + max(0.0, yz):.12g} {xz:.12g}\n")
                fh.write(f"{zlo:.12g} {zhi:.12g} {yz:.12g}\n")
            else:
                fh.write("ITEM: BOX BOUNDS pp pp pp\n")
                fh.write(f"{xlo:.12g} {xhi:.12g}\n"); fh.write(f"{ylo:.12g} {yhi:.12g}\n"); fh.write(f"{zlo:.12g} {zhi:.12g}\n")
            if mode == 1:
                fh.write("ITEM: ATOMS id type x y z\n")
                for i in range(natoms):
                    aid = i+1; typ = types[i]; x,y,z = positions[i]
                    fh.write(f"{aid} {typ} {x:.12g} {y:.12g} {z:.12g}\n")
            elif mode == 2:
                fh.write("ITEM: ATOMS id type charge x y z\n")
                arrs = getattr(atoms, "arrays", {})
                charges = arrs.get('charge', np.zeros(natoms))
                for i in range(natoms):
                    aid = i+1; typ = types[i]; ch = float(charges[i]); x,y,z = positions[i]
                    fh.write(f"{aid} {typ} {ch:.6f} {x:.12g} {y:.12g} {z:.12g}\n")
            else:
                fh.write("ITEM: ATOMS id type x y z Spin0 Spin1 Spin2 Spin3\n")
                arrs = getattr(atoms, "arrays", {})
                s0 = arrs.get('Spin0', np.zeros(natoms)); s1 = arrs.get('Spin1', np.zeros(natoms))
                s2 = arrs.get('Spin2', np.zeros(natoms)); s3 = arrs.get('Spin3', np.zeros(natoms))
                for i in range(natoms):
                    aid = i+1; typ = types[i]; x,y,z = positions[i]
                    fh.write(f"{aid} {typ} {x:.12g} {y:.12g} {z:.12g} {float(s0[i]):.6f} {float(s1[i]):.6f} {float(s2[i]):.6f} {float(s3[i]):.6f}\n")
    print(f"Wrote LAMMPS dump: {filename}")

def write_extxyz_for_ovito(filename: str, frames: List[Atoms]):

    if len(frames) == 0:
        raise RuntimeError("No frames to write.")
    
    with open(filename, 'w') as fh:
        for frame_idx, atoms in enumerate(frames):
            natoms = len(atoms)
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            cell = atoms.get_cell()
            
            fh.write(f"{natoms}\n")
            
            if cell is not None and np.any(cell):
                cell_matrix = cell.reshape(9)
                fh.write(f'Lattice="{cell_matrix[0]:.6f} {cell_matrix[1]:.6f} {cell_matrix[2]:.6f} ')
                fh.write(f'{cell_matrix[3]:.6f} {cell_matrix[4]:.6f} {cell_matrix[5]:.6f} ')
                fh.write(f'{cell_matrix[6]:.6f} {cell_matrix[7]:.6f} {cell_matrix[8]:.6f}" ')
            
            fh.write('Properties=species:S:1:pos:R:3\n')
            
            for i in range(natoms):
                symbol = symbols[i]
                x, y, z = positions[i]
                fh.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
    
    print(f"Wrote extended XYZ: {filename}")

_float_re = r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?'
_bounds_any_re = re.compile(r'^\s*(' + _float_re + r')\s+(' + _float_re + r')\s*$')
_tilt_re = re.compile(r'^\s*(' + _float_re + r')\s+(' + _float_re + r')\s+(' + _float_re + r')\s*$')

def _is_number_token(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False


def parse_lammps_data(path: str) -> Atoms:

    with open(path, 'r') as fh:
        raw_lines = fh.readlines()

    lines = [ln.rstrip('\n') for ln in raw_lines]
    stripped = [ln.strip() for ln in lines]

    natoms = None
    masses_section = {}
    bounds_temp = []
    bounds = {}
    tilt = None
    in_masses = False
    in_atoms = False
    atom_lines = []
    found_bounds_lines = 0

    for i, s in enumerate(stripped):
        if not s:
            continue
        m = re.match(r'^\s*(\d+)\s+atoms\b', s, re.IGNORECASE)
        if m and natoms is None:
            natoms = int(m.group(1))
            continue
        if s.lower().startswith('masses'):
            in_masses = True; in_atoms = False; continue
        if in_masses:
            m2 = re.match(r'^\s*(\d+)\s+(' + _float_re + r')', s)
            if m2:
                tid = int(m2.group(1)); mass = float(m2.group(2)); masses_section[tid] = mass; continue
            else:
                in_masses = False
        m_bounds = re.match(r'^\s*(' + _float_re + r')\s+(' + _float_re + r')\s+(xlo|ylo|zlo)\s+(xhi|yhi|zhi)\s*$', s, re.IGNORECASE)
        m_tilt = re.match(r'^\s*(' + _float_re + r')\s+(' + _float_re + r')\s+(' + _float_re + r')\s+xy\s+xz\s+yz\s*$', s, re.IGNORECASE)
        if m_bounds:
            bounds_temp.append((float(m_bounds.group(1)), float(m_bounds.group(2))))
            found_bounds_lines += 1
            if found_bounds_lines == 3:
                bounds['x'], bounds['y'], bounds['z'] = bounds_temp[:3]
        
        if m_tilt:
            tilt = (float(m_tilt.group(1)), float(m_tilt.group(2)), float(m_tilt.group(3)))

        if s.lower().startswith('atoms'):
            in_atoms = True; in_masses = False; continue
        if in_atoms:
            if re.match(r'^\s*\d+', s):
                atom_lines.append(s); continue
            else:
                if re.match(r'^[A-Za-z]', s) and not re.match(r'^\d', s):
                    in_atoms = False
                continue

    if len(atom_lines) == 0:
        raise RuntimeError(f"No atom lines parsed from '{path}'. (checked {len(stripped)} lines)")
    
    if natoms is None:
        natoms = len(atom_lines)
        print(f"Warning: Could not find 'atoms' count in header, using {natoms} from atom lines.")

    ids = []; types = []; xs=[]; ys=[]; zs=[]; charges_list = []; spin0=[]; spin1=[]; spin2=[]; spin3=[]
    format_detected = None 
    
    for ln in atom_lines:
        core = ln.split('#')[0].strip()
        toks = core.split()
        nt = len(toks)
        last3_numeric = nt >= 3 and all(_is_number_token(toks[-3 + j]) for j in range(3))
        last4_numeric = nt >= 4 and all(_is_number_token(toks[-4 + j]) for j in range(4))
        parsed = False

        if nt == 9 and last4_numeric:

            if all(_is_number_token(toks[j]) for j in range(2, 5)):
                aid = int(toks[0])
                typ = int(toks[1])
                x = float(toks[2]); y = float(toks[3]); z = float(toks[4])
                s0 = float(toks[5]); s1 = float(toks[6]); s2 = float(toks[7]); s3 = float(toks[8])
                ids.append(aid); types.append(typ); xs.append(x); ys.append(y); zs.append(z)
                spin0.append(s0); spin1.append(s1); spin2.append(s2); spin3.append(s3)
                charges_list.append(None)
                parsed = True
                if format_detected is None:
                    format_detected = 'spin'
        
        if not parsed and nt == 6:
            if _is_number_token(toks[2]) and last3_numeric:
                aid = int(toks[0])
                typ = int(toks[1])
                ch = float(toks[2])
                x = float(toks[3]); y = float(toks[4]); z = float(toks[5])
                ids.append(aid); types.append(typ); xs.append(x); ys.append(y); zs.append(z)
                charges_list.append(ch)
                spin0.append(0.0); spin1.append(0.0); spin2.append(0.0); spin3.append(0.0)
                parsed = True
                if format_detected is None:
                    format_detected = 'charge'
        
        if not parsed and nt == 5 and last3_numeric:
            aid = int(toks[0])
            typ = int(toks[1])
            x = float(toks[2]); y = float(toks[3]); z = float(toks[4])
            ids.append(aid); types.append(typ); xs.append(x); ys.append(y); zs.append(z)
            charges_list.append(None)
            spin0.append(0.0); spin1.append(0.0); spin2.append(0.0); spin3.append(0.0)
            parsed = True
            if format_detected is None:
                format_detected = 'classic'

        if not parsed:
            coord_found = False
            toklen = len(toks)
            for start in range(toklen - 3, -1, -1):
                if all(_is_number_token(toks[start + j]) for j in range(3)):
                    try:
                        x = float(toks[start]); y = float(toks[start+1]); z = float(toks[start+2])
                    except Exception:
                        continue
                    coord_idx = start
                    coord_found = True
                    break
            if not coord_found:
                raise RuntimeError(f"Unable to parse coordinates from atom line: '{ln}'")
            aid = None; typ = None
            for tok in toks[:min(4, coord_idx)]:
                if re.match(r'^\d+$', tok):
                    aid = int(tok); break
            if aid is None:
                aid = len(ids) + 1
            for tok in toks[1:coord_idx]:
                if re.match(r'^\d+$', tok):
                    typ = int(tok); break
            if typ is None:
                typ = 1
            ids.append(aid); types.append(typ); xs.append(x); ys.append(y); zs.append(z)
            charges_list.append(None)
            spin0.append(0.0); spin1.append(0.0); spin2.append(0.0); spin3.append(0.0)
            if format_detected is None:
                format_detected = 'unknown'

    print(f"Detected format: {format_detected}")
    print(f"Parsed {len(ids)} atoms from {len(atom_lines)} atom lines")

    type_to_symbol = {}
    if masses_section:
        for tid,mass in masses_section.items():
            type_to_symbol[tid] = nearest_element_by_mass(mass)
    else:
        for tid in sorted(set(types)):
            type_to_symbol[tid] = 'X'

    symbols = [ type_to_symbol.get(t, 'X') for t in types ]
    positions = np.column_stack((xs, ys, zs))
    atoms = Atoms(symbols=symbols, positions=positions, pbc=True)

    if 'x' in bounds and 'y' in bounds and 'z' in bounds:
        xlo,xhi = bounds['x']; ylo,yhi = bounds['y']; zlo,zhi = bounds['z']
        ax = xhi - xlo; by = yhi - ylo; cz = zhi - zlo
        bx = cx = cy = 0.0
        if tilt is not None:
            bx, cx, cy = tilt
        cell = np.array([[ax,0.0,0.0],[bx,by,0.0],[cx,cy,cz]])
        atoms.set_cell(cell)
        atoms.set_pbc(True)
        atoms.set_positions(atoms.get_positions() - np.array([xlo,ylo,zlo]))

    if format_detected == 'spin':
        atoms.set_array('Spin0', np.array(spin0, dtype=float))
        atoms.set_array('Spin1', np.array(spin1, dtype=float))
        atoms.set_array('Spin2', np.array(spin2, dtype=float))
        atoms.set_array('Spin3', np.array(spin3, dtype=float))
    
    if format_detected == 'charge' or any(ch is not None for ch in charges_list):
        ch_arr = np.array([0.0 if ch is None else ch for ch in charges_list], dtype=float)
        atoms.set_array('charge', ch_arr)

    return atoms


def read_frames_with_ase(path: str) -> List[Atoms]:
    if io is None:
        raise RuntimeError("ASE not installed.")
    
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ('.lmp', '.data'):
        try:
            atom = parse_lammps_data(path)
            return [atom]
        except Exception as e_parse:
            print(f"Custom parser failed: {e_parse}, trying ASE...")
    
    try:
        frames = aseio.read(path, index=':')
        if isinstance(frames, Atoms):
            frames = [frames]
        return frames
    except Exception as e_auto:
        # Try specific formats
        tried = []
        if ext in ('.dump', '.lammpstrj', '.lammpstrj.gz', '.traj'):
            tried.append('lammps-dump')
        
        for fmt in tried:
            try:
                frames = aseio.read(path, format=fmt, index=':')
                if isinstance(frames, Atoms):
                    frames = [frames]
                return frames
            except Exception:
                pass
        
        try:
            single = aseio.read(path, index=0)
            return [single]
        except Exception as e_final:
            raise RuntimeError(f"ASE failed to read '{path}'. Autodetect error: {e_auto}. Attempted formats: {tried}. Last error: {e_final}")

def infer_box_from_frames(frames: List[Atoms]) -> Tuple[np.ndarray, List[Atoms]]:
    all_mins = np.array([np.min(f.get_positions(axis=0), axis=0) for f in frames])
    all_maxs = np.array([np.max(f.get_positions(axis=0), axis=0) for f in frames])
    global_min = np.min(all_mins, axis=0)
    global_max = np.max(all_maxs, axis=0)
    lengths = global_max - global_min
    tiny = 1.0
    lengths = np.where(lengths < 1e-8, tiny, lengths)
    new_cell = np.diag(lengths)
    new_frames = []
    for f in frames:
        pos = f.get_positions()
        pos_shifted = pos - global_min
        new_at = Atoms(symbols=f.get_chemical_symbols(), positions=pos_shifted, cell=new_cell, pbc=True)
        for name, arr in getattr(f, "arrays", {}).items():
            if name in ('positions',):
                continue
            try:
                new_at.set_array(name, np.array(arr))
            except Exception:
                pass
        new_frames.append(new_at)
    return new_cell, new_frames

def maybe_orthogonalize_and_apply(frames: List[Atoms], orthogonalize: bool = True) -> List[Atoms]:
    if not frames:
        return frames
    
    first = frames[0]
    cell = np.array(first.get_cell(), dtype=float)
    try:
        vol = np.linalg.det(cell)
    except Exception:
        vol = 0.0
    
    if abs(vol) < 1e-12:
        return frames
    
    if orthogonalize:
        if is_triclinic(cell):
            ax, bx, cx, by, cy, cz = triangular_cell_components(cell)
            print(f"Your cell appears triclinic with tilt-values: xy={bx:.6g}, xz={cx:.6g}, yz={cy:.6g}")
            print("Making it orthogonal as requested.")
            new_frames = []
            for f in frames:
                try:
                    new_frames.append(orthogonalize_and_wrap_atoms(f))
                except Exception as e:
                    print(f"Orthogonalize failed: {e}. Wrapping only.")
                    try:
                        wrap_positions_in_place(f)
                    except Exception:
                        pass
                    new_frames.append(f)
            return new_frames
        else:
            print("Your cell is orthogonal. Unskewing and aligning as requested.")
            new_frames = []
            for f in frames:
                try:
                    new_frames.append(orthogonalize_and_wrap_atoms(f))
                except Exception as e:
                    print(f"Orthogonalize failed: {e}. Wrapping only.")
                    try:
                        wrap_positions_in_place(f)
                    except Exception:
                        pass
                    new_frames.append(f)
            return new_frames
    else:
        print("Keeping original cell geometry (no orthogonalization).")
        for f in frames:
            try:
                wrap_positions_in_place(f)
            except Exception:
                pos = f.get_positions()
                cell = np.array(f.get_cell(), dtype=float)
                try:
                    invC = np.linalg.inv(cell)
                    frac = pos.dot(invC)
                    f.set_positions(np.mod(frac, 1.0).dot(cell))
                except Exception:
                    pass
        return frames

def is_lammps_format(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ('.lmp', '.data', '.lammps', '.dump', '.lammpstrj')

def get_output_format_from_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    
    if ext in ('lmp', 'data', 'lammps'):
        return 'lammps_data'
    elif ext in ('dump', 'lammpstrj'):
        return 'lammps_dump'
    
    elif ext == 'cif':
        return 'cif'
    elif ext == 'xyz':
        return 'extxyz'
    elif ext == 'cfg':
        return 'cfg'
    elif ext == 'xsf':
        return 'xsf'
    elif ext in ('vasp', 'poscar', 'contcar'):
        return 'vasp'
    elif ext in ('pdb', 'ent'):
        return 'pdb'
    elif ext == 'extxyz':
        return 'extxyz'
    else:
        return 'extxyz'

def convert_file(input_file: str, output_file: str, target_mode: str = "classic", orthogonalize: bool = True):
    if io is None:
        print("ERROR: ASE not installed. pip install ase")
        return
    
    input_file = normalize_path_for_os(input_file)
    output_file = normalize_path_for_os(output_file)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    print(f"Converting: {input_file} -> {output_file}")
    print(f"Target mode: {target_mode}")
    print(f"Orthogonalize: {'yes' if orthogonalize else 'no'}")
    
    frames = read_frames_with_ase(input_file)
    print(f"Read {len(frames)} frame(s).")
    
    output_format = get_output_format_from_extension(output_file)
    print(f"Output format: {output_format}")
    
    first_cell = np.array(frames[0].get_cell(), dtype=float)
    try: 
        vol = np.linalg.det(first_cell)
    except Exception: 
        vol = 0.0
    
    if abs(vol) < 1e-12:
        _, frames = infer_box_from_frames(frames)
        print("Inferred orthogonal cell (no cell present in input).")
    else:
        frames = maybe_orthogonalize_and_apply(frames, orthogonalize)
    
    if output_format.startswith('lammps'):
        if target_mode == "classic":
            for f in frames:
                for k in list(getattr(f, "arrays", {}).keys()):
                    if k.lower() == 'charge' or k.lower().startswith('spin'):
                        try: 
                            del f.arrays[k]
                        except Exception: 
                            f.arrays.pop(k, None)
            mode = 1
            print("Using LAMMPS classic format (mode 1)")
            
        elif target_mode == "charge":
            for f in frames:
                for k in list(getattr(f, "arrays", {}).keys()):
                    if k.lower().startswith('spin'):
                        try: 
                            del f.arrays[k]
                        except Exception: 
                            f.arrays.pop(k, None)
                if 'charge' not in getattr(f, "arrays", {}):
                    f.set_array('charge', np.zeros(len(f)), dtype=float)
            mode = 2
            print("Using LAMMPS charge format (mode 2)")
            
        elif target_mode == "spin":
            for f in frames:
                for k in list(getattr(f, "arrays", {}).keys()):
                    if k.lower() == 'charge':
                        try: 
                            del f.arrays[k]
                        except Exception: 
                            f.arrays.pop(k, None)
                if not all(k in getattr(f, "arrays", {}) for k in ('Spin0', 'Spin1', 'Spin2', 'Spin3')):
                    for spin_key in ['Spin0', 'Spin1', 'Spin2', 'Spin3']:
                        f.set_array(spin_key, np.zeros(len(f)), dtype=float)
            mode = 3
            print("Using LAMMPS spin format (mode 3)")
        else:
            raise ValueError(f"Unknown target_mode: {target_mode}")
        
        if len(frames) == 1 and output_format == 'lammps_data':
            write_lammps_data_from_ase(output_file, frames[0], mode=mode)
        else:
            base, ext = os.path.splitext(output_file)
            if ext.lower() not in ['.dump', '.lammpstrj']:
                output_file = base + ".dump"
            write_lammps_dump(output_file, frames, mode=mode)
    
    else:
        if write is None:
            print("ERROR: ASE write module not available")
            return
        
        if output_format == 'extxyz':
            write_extxyz_for_ovito(output_file, frames)
            print(f"Converted to extended XYZ: {output_file}")
            return
        
        if len(frames) == 1:
            try:
                write(output_file, frames[0], format=output_format)
                prepend_vcl_header(output_file)
                print(f"Converted to {output_format.upper()}: {output_file}")
            except Exception as e:
                print(f"Error writing to {output_format}: {e}")
                try:
                    write(output_file, frames[0])
                    prepend_vcl_header(output_file)
                    print(f"Converted (using ASE default): {output_file}")
                except Exception as e2:
                    print(f"Failed to write file: {e2}")
        else:
            base, ext = os.path.splitext(output_file)
            
            if output_format in ('xyz', 'xsf', 'extxyz'):
                try:
                    if output_format == 'extxyz':
                        write_extxyz_for_ovito(output_file, frames)
                    else:
                        write(output_file, frames, format=output_format)
                        prepend_vcl_header(output_file)
                    print(f"Converted {len(frames)} frames to {output_format.upper()}: {output_file}")
                except Exception as e:
                    print(f"Warning: Could not write multiple frames to single {output_format} file: {e}")
                    for i, frame in enumerate(frames):
                        frame_file = f"{base}_frame{i}{ext}"
                        if output_format == 'extxyz':
                            write_extxyz_for_ovito(frame_file, [frame])
                        else:
                            write(frame_file, frame, format=output_format)
                            prepend_vcl_header(frame_file)
                        print(f"  Wrote frame {i}: {frame_file}")
            else:
                for i, frame in enumerate(frames):
                    frame_file = f"{base}_frame{i}{ext}"
                    write(frame_file, frame, format=output_format)
                    prepend_vcl_header(frame_file)
                    print(f"  Wrote frame {i}: {frame_file}")
    
    print(f"\n✓✓✓ SUCCESS! ✓✓✓")
    print(f"Converted: {input_file} -> {output_file}")

def parse_input_file(input_file_path: str) -> dict:
    params = {}
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            value = parts[1].strip()
            
            if key == "input_file":
                params["input_file"] = value
            elif key == "output_file":
                params["output_file"] = value
            elif key == "charge/spin":
                if value.lower() in ["classic", "charge", "spin"]:
                    params["target_mode"] = value.lower()
                else:
                    raise ValueError(f"Invalid charge/spin value: {value}. Must be 'classic', 'charge', or 'spin'")
            elif key == "unskew_and_align":
                value_lower = value.lower()
                if value_lower in ["yes", "y", "true", "1"]:
                    params["orthogonalize"] = True
                elif value_lower in ["no", "n", "false", "0"]:
                    params["orthogonalize"] = False
                else:
                    raise ValueError(f"Invalid unskew_and_align value: {value}. Must be 'yes' or 'no'")
    
    if "input_file" not in params:
        raise ValueError("Missing 'input_file' in input file")
    if "output_file" not in params:
        raise ValueError("Missing 'output_file' in input file")
    
    if "target_mode" not in params:
        params["target_mode"] = "classic"
    if "orthogonalize" not in params:
        params["orthogonalize"] = True
    
    return params

def convert_to_classic_md_interactive():
    if io is None:
        print("ERROR: ASE not installed. pip install ase"); return
    infile = ask("Please enter structure filename (in current directory)")
    infile = normalize_path_for_os(infile)
    if not os.path.exists(infile):
        print("File not found:", infile); return
    out_default = os.path.splitext(os.path.basename(infile))[0] + "_classic.data"
    outname = ask("Enter desired output filename for modified LAMMPS file", out_default)
    
    orthogonalize = ask("Would you like to unskew and align it? (Y/n)", "Y").strip().lower() not in ('n', 'no')
    
    convert_file(infile, outname, target_mode="classic", orthogonalize=orthogonalize)

def add_charge_interactive():
    if io is None:
        print("ERROR: ASE not installed. pip install ase"); return
    infile = ask("Enter structure filename (current directory)"); infile = normalize_path_for_os(infile)
    if not os.path.exists(infile): print("File not found:", infile); return
    
    orthogonalize = ask("Would you like to unskew and align it? (Y/n)", "Y").strip().lower() not in ('n', 'no')
    
    out_default = os.path.splitext(os.path.basename(infile))[0] + "_charge.data"
    outname = ask("Output filename for charged LAMMPS file", out_default)
    
    convert_file(infile, outname, target_mode="charge", orthogonalize=orthogonalize)

def add_spin_interactive():
    if io is None:
        print("ERROR: ASE not installed. pip install ase"); return
    infile = ask("Enter structure filename (current directory)"); infile = normalize_path_for_os(infile)
    if not os.path.exists(infile): print("File not found:", infile); return
    
    orthogonalize = ask("Would you like to unskew and align it? (Y/n)", "Y").strip().lower() not in ('n', 'no')
    
    out_default = os.path.splitext(os.path.basename(infile))[0] + "_spin.data"
    outname = ask("Output filename for spin LAMMPS file", out_default)
    
    convert_file(infile, outname, target_mode="spin", orthogonalize=orthogonalize)

def convert_from_lammps_interactive():
    if io is None:
        print("ERROR: ASE not installed. pip install ase"); return
    
    infile = ask("Enter input filename")
    infile = normalize_path_for_os(infile)
    if not os.path.exists(infile):
        print("File not found:", infile); return
    
    base_name = os.path.splitext(os.path.basename(infile))[0]
    out_default = base_name + ".cif"
    outname = ask("Enter output filename (extension determines format, e.g., .cif, .xyz, .cfg)", out_default)
    
    orthogonalize = ask("Would you like to unskew and align it? (Y/n)", "Y").strip().lower() not in ('n', 'no')
    
    convert_file(infile, outname, target_mode="classic", orthogonalize=orthogonalize)

def replicate_only_interactive():
    if io is None:
        print("ERROR: ASE not installed. pip install ase"); return
    infile = ask("Enter filename to replicate (current directory)"); infile = normalize_path_for_os(infile)
    if not os.path.exists(infile): print("File not found:", infile); return
    rep_str = ask("Enter replication factors nx ny nz (e.g. '2 2 1')", "1 1 1")
    try: nx,ny,nz = [int(x) for x in rep_str.split()]
    except Exception:
        print("Invalid replication factors."); return
    frames = read_frames_with_ase(infile)
    rep_frames = [f.repeat((nx,ny,nz)) for f in frames]
    
    orthogonalize = ask("Would you like to unskew and align it? (Y/n)", "Y").strip().lower() not in ('n', 'no')
    
    base, ext = os.path.splitext(os.path.basename(infile))
    suggested = f"{base}_rep{nx}x{ny}x{nz}{ext}"
    outname = ask("Output filename for replicated file", suggested)
    
    if orthogonalize:
        rep_frames = maybe_orthogonalize_and_apply(rep_frames, orthogonalize)
    
    output_format = get_output_format_from_extension(outname)
    
    if output_format.startswith('lammps'):

        has_charge, has_spin_any, has_spin_components = detect_charge_spin(frames[0])
        if has_charge:
            mode = 2
        elif has_spin_any:
            mode = 3
        else:
            mode = 1
        
        try:
            if len(rep_frames) == 1:
                write_lammps_data_from_ase(outname, rep_frames[0], mode=mode)
            else:
                outdump = os.path.splitext(outname)[0] + ".dump"; write_lammps_dump(outdump, rep_frames, mode=mode); outname = outdump
            print(f"\n✓✓✓ SUCCESS! ✓✓✓")
            print(f"Your replicated structure is ready: {outname}")
            return
        except Exception as e:
            print("Internal LAMMPS writer failed:", e)
    
    try:
        if len(rep_frames) == 1:
            if output_format == 'extxyz':
                write_extxyz_for_ovito(outname, rep_frames)
            else:
                aseio.write(outname, rep_frames[0])
                prepend_vcl_header(outname)
        else:
            try: 
                if output_format == 'extxyz':
                    write_extxyz_for_ovito(outname, rep_frames)
                else:
                    aseio.write(outname, rep_frames)
                    prepend_vcl_header(outname)
            except Exception:
                outxyz = os.path.splitext(outname)[0] + ".xyz"; 
                write_extxyz_for_ovito(outxyz, rep_frames)
                outname = outxyz
        print(f"\n✓✓✓ SUCCESS! ✓✓✓")
        print(f"Your replicated structure is ready: {outname}")
    except Exception as e:
        print("Failed to write replicated file:", e)

def run_input_convertor_app():
    if io is None:
        print("ERROR: ASE required. pip install ase"); return
    print("************************************************")
    print("Welcome to the LAMMPS Input File Converter! (:")
    print("************************************************")
    while True:
        print("\n--- LAMMPS Format Converter ---")
        print("0. Exit")
        print("1. Convert file (auto-detect format from extension)")
        print("2. Add charge to a file (for LAMMPS)")
        print("3. Add spin to a file (for LAMMPS)")
        print("4. Replicate a structure")
        choice = ask("Select an option (0-4)", "0")
        if choice == '0':
            print("Exiting converter."); return
        elif choice == '1':
            convert_to_classic_md_interactive()
        elif choice == '2':
            add_charge_interactive()
        elif choice == '3':
            add_spin_interactive()
        elif choice == '4':
            replicate_only_interactive()
        else:
            print("Invalid choice.")

def main():
    if io is None:
        print("ERROR: ASE required. Please install ASE: pip install ase")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        try:
            params = parse_input_file(input_file)
            convert_file(
                params["input_file"], 
                params["output_file"], 
                params["target_mode"], 
                params["orthogonalize"]
            )
        except Exception as e:
            print(f"Error processing input file: {e}")
            print("\nInput file format should be:")
            print("input_file ./FeCr.xyz")
            print("output_file ./FeCr.lmp")
            print("charge/spin classic/charge/spin  # optional, default: classic")
            print("unskew_and_align yes/no          # optional, default: yes")
            print("\nOr:")
            print("input_file ./FeCr.lmp")
            print("output_file ./FeCr.cif")
            print("unskew_and_align yes/no          # optional, default: yes")
            print("\nThe script automatically detects conversion direction from file extensions.")
            print("\nNOTE: .xyz files will be written in extended XYZ format")
            sys.exit(1)
    else:
        run_input_convertor_app()

if __name__ == "__main__":
    main()