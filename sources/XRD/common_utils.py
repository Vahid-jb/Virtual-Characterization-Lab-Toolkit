#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
common_utils.py - Shared input parsing and structure reading for the XRD modules.
"""

import os

import numpy as np

# Input file parsing

_TRUE_WORDS = ('yes', 'true', 'y', 'on')
_FALSE_WORDS = ('no', 'false', 'n', 'off')


def _convert_scalar(value):
    text = value.strip()
    if text == '':
        return ''
    low = text.lower()
    if low in _TRUE_WORDS:
        return True
    if low in _FALSE_WORDS:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _split_list(value):
    text = value.strip()
    if len(text) >= 2 and text[0] in '([' and text[-1] in ')]':
        text = text[1:-1]
    parts = [p.strip() for p in text.split(',')]
    parts = [p for p in parts if p != '']
    return [_convert_scalar(p) for p in parts]


def parse_input_file(input_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    params = {}
    current_section = 'general'

    with open(input_file, 'r', encoding='utf-8-sig', errors='replace') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#') or line.startswith('!'):
                continue

            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].strip().lower()
                continue

            if '=' not in line:
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # strip inline comments
            for marker in ('#', '!'):
                if marker in value:
                    value = value.split(marker, 1)[0].strip()

            # strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]

            bracketed = (len(value) >= 2 and value[0] in '([' and value[-1] in ')]')
            if (',' in value) or bracketed:
                converted = _split_list(value)
                if len(converted) == 1 and not bracketed:
                    converted = converted[0]
            else:
                converted = _convert_scalar(value)

            full_key = f"{current_section}.{key}" if current_section != 'general' else key
            params[full_key] = converted

    return params


def replace_dots(s):
    if 'e' in s.lower():
        parts = s.lower().split('e')
        return parts[0].replace('.', '') + 'e' + parts[1]
    return s.replace('.', '')



# Structure reading
def read_lammps_xyz(xyz_filename, verbose=True, numeric_mode='atomic_number',
                    type_map=None):
    if verbose:
        print(f"Reading XYZ file: {xyz_filename}")

    if not os.path.exists(xyz_filename):
        raise FileNotFoundError(f"Structure file not found: {xyz_filename}")

    with open(xyz_filename, 'r', encoding='utf-8-sig', errors='replace') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError(f"File {xyz_filename} is too short")

    try:
        natoms = int(lines[0].strip())
    except ValueError:
        natoms = sum(1 for line in lines[2:] if len(line.strip().split()) >= 4)
        if verbose:
            print(f"Warning: could not parse the atom count on line 1; "
                  f"assuming {natoms} atoms.")

    species = []
    coords = []
    atoms_read = 0
    numeric_seen = set()

    for i in range(2, len(lines)):
        if atoms_read >= natoms and natoms > 0:
            break

        line = lines[i].strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 4:
            continue

        try:
            spec = parts[0]
            x, y, z = map(float, parts[1:4])
        except (ValueError, IndexError):
            if verbose:
                print(f"Warning: could not parse line {i + 1}: {line[:50]}...")
            continue

        if spec.isdigit():
            numeric_seen.add(int(spec))
            if numeric_mode == 'atomic_number':
                try:
                    from pymatgen.core.periodic_table import Element
                    spec = Element.from_Z(int(spec)).symbol
                except Exception:
                    try:
                        import periodictable
                        spec = periodictable.elements[int(spec)].symbol
                    except Exception:
                        pass
            elif numeric_mode == 'lammps_type':
                if not type_map:
                    raise ValueError(
                        f"{xyz_filename} has numeric atom types "
                        f"{sorted(numeric_seen)} and numeric_mode = 'lammps_type', "
                        f"but no type_map was supplied. Add e.g. type_map = 1:Mo,2:S")
                key = int(spec)
                if key not in type_map:
                    raise ValueError(
                        f"LAMMPS atom type {key} is not present in type_map "
                        f"{sorted(type_map)}")
                spec = type_map[key]
            # 'raw': leave as the original token

        species.append(spec)
        coords.append([x, y, z])
        atoms_read += 1

    if atoms_read == 0:
        raise ValueError(f"No atoms could be parsed from {xyz_filename}")

    if verbose:
        print(f"Successfully read {atoms_read} atoms")
        if numeric_seen and numeric_mode == 'atomic_number':
            print(f"Note: numeric species column {sorted(numeric_seen)} was read as "
                  f"ATOMIC NUMBERS. If these are LAMMPS atom types, set "
                  f"species_mode = lammps_types and supply type_map (e.g. 1:Cu,2:Zn).")

    return species, np.array(coords)


def parse_type_map(value):
    if value in (None, ''):
        return None
    if isinstance(value, dict):
        return {int(k): str(v).strip() for k, v in value.items()}
    items = value if isinstance(value, (list, tuple)) else str(value).split(',')
    out = {}
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if ':' not in text:
            raise ValueError(f"Invalid type_map entry {text!r}; expected '1:Mo'")
        k, v = text.split(':', 1)
        out[int(k.strip())] = v.strip()
    return out or None


SPECIES_MODES = {'chemical_symbols': 'atomic_number',
                 'atomic_numbers': 'atomic_number',
                 'lammps_types': 'lammps_type'}


def species_read_options(params):
    """Return (species_mode, numeric_mode, type_map) for read_lammps_xyz."""
    mode = params.get('species_mode', 'chemical_symbols')
    mode = str(mode).strip().lower() if mode not in (None, '') else 'chemical_symbols'
    if mode not in SPECIES_MODES:
        print(f"[WARN] unknown species_mode {mode!r}; using 'chemical_symbols'")
        mode = 'chemical_symbols'
    return mode, SPECIES_MODES[mode], parse_type_map(params.get('type_map', None))


def check_dependencies():
    missing = []
    for name, hint in (("numpy", "pip install numpy"),
                       ("scipy", "pip install scipy"),
                       ("matplotlib", "pip install matplotlib")):
        try:
            __import__(name)
        except ImportError:
            missing.append(f"{name} ({hint})")
    return missing


def save_structure_cif(structure, filename, comment=""):
    try:
        structure.to(filename=filename)
        print(f"Saved structure to {filename}")
        return True
    except Exception as e:
        print(f"Warning: could not save CIF file: {e}")
        return False


def validate_params(params, required_keys, module_name):
    missing = [key for key in required_keys if key not in params]
    if missing:
        print(f"Error: missing required parameters for {module_name}:")
        for key in missing:
            print(f"  - {key}")
        return False
    return True
