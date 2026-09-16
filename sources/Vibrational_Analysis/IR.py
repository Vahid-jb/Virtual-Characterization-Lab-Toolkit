#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
IR Spectrum Calculator – static & dynamic charges, aligned with VDOS

"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import signal
from scipy.signal import savgol_filter
from datetime import datetime
import argparse
import io

if sys.platform == "win32":
    if sys.stdout is not None and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if sys.stderr is not None and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Physical constants
C_CM_S = 2.9979245899e10         # cm/s (match VDOS)
H_BAR = 1.054571817e-34          # J·s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

VALID_WINDOWS = ['gaussian', 'blackman-harris', 'hamming', 'hann']
VALID_ESTIMATORS = ['acf', 'welch']

def lag_window(nlags, window_kind, delta_t_fs, window_width_ps):
 
    wk = window_kind.lower()
    if wk not in VALID_WINDOWS:
        sys.exit(f"Error: window_kind must be one of {', '.join(VALID_WINDOWS)}. Got '{window_kind}'.")
    k = np.arange(nlags, dtype=float)

    if wk == "gaussian":
        sigma_frames = window_width_ps * 1000.0 / (2.354820045 * delta_t_fs)
        if sigma_frames <= 0.0:
            sys.exit("Error: window_width_ps must be positive.")
        w = np.exp(-0.5 * (k / sigma_frames) ** 2)
    else:
        builder = {"blackman-harris": signal.windows.blackmanharris,
                   "hamming": signal.windows.hamming,
                   "hann": signal.windows.hann}[wk]
        full = builder(2 * nlags - 1, sym=True)
        w = full[nlags - 1:] / full[nlags - 1]

    w = np.array(w, dtype=float)
    w[0] = 1.0
    return w


def signal_window(nperseg, window_kind):
    
    wk = window_kind.lower()
    if wk not in VALID_WINDOWS:
        sys.exit(f"Error: window_kind must be one of {', '.join(VALID_WINDOWS)}. Got '{window_kind}'.")
    if wk == "gaussian":
        
        return signal.windows.gaussian(nperseg, std=nperseg / 6.0, sym=False)
    return {"blackman-harris": signal.windows.blackmanharris,
            "hamming": signal.windows.hamming,
            "hann": signal.windows.hann}[wk](nperseg, sym=False)


def vector_acf_biased(series):

    x = np.asarray(series, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    x = x - np.mean(x, axis=0, keepdims=True)
    n = x.shape[0]
    acf = np.zeros(n, dtype=float)
    for a in range(x.shape[1]):
        col = x[:, a]
        acf += signal.fftconvolve(col, col[::-1], mode='full')[n - 1:]
    return acf / n


def spectrum_from_acf(acf, w, delta_t_s):
    
    cw = np.asarray(acf, dtype=float) * np.asarray(w, dtype=float)
    nfft = int(2 ** math.ceil(math.log2(max(2 * len(cw) - 1, 2))))   # nfft >= 2N-1
    two_sided = 2.0 * np.fft.rfft(cw, n=nfft).real - cw[0]
    psd = 2.0 * delta_t_s * two_sided
    psd[0] *= 0.5                       # DC bin is not doubled
    if nfft % 2 == 0:
        psd[-1] *= 0.5                  # Nyquist bin likewise
    freq = np.fft.rfftfreq(nfft, d=delta_t_s)
    return freq, psd, nfft


def spectrum_welch(series, delta_t_s, window_kind, segment_ps, overlap_fraction):
    
    x = np.asarray(series, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    delta_t_fs = delta_t_s * 1e15
    nperseg = int(round(segment_ps * 1000.0 / delta_t_fs))
    if nperseg < 16:
        sys.exit(f"Error: welch_segment_ps={segment_ps} ps is only {nperseg} frames. Increase it.")
    if nperseg > n:
        print(f"WARNING: welch_segment_ps ({segment_ps} ps) exceeds the trajectory "
              f"({n * delta_t_fs / 1000.0:.2f} ps). Clamping to a single segment.")
        nperseg = n
    noverlap = int(round(nperseg * overlap_fraction))
    noverlap = min(max(noverlap, 0), nperseg - 1)
    win = signal_window(nperseg, window_kind)
    freq, psd = signal.welch(x, fs=1.0 / delta_t_s, window=win, nperseg=nperseg,
                             noverlap=noverlap, detrend='constant',
                             return_onesided=True, scaling='density', axis=0)
    nseg = 1 + (n - nperseg) // (nperseg - noverlap) if nperseg < n else 1
    return freq, np.sum(psd, axis=1), nperseg, noverlap, nseg


def reduced_frequency(wavenumber_cm, temperature):
    
    omega = 2.0 * np.pi * np.asarray(wavenumber_cm, dtype=float) * C_CM_S
    return H_BAR * omega / (BOLTZMANN_CONSTANT * temperature)


def qcf_harmonic(wavenumber_cm, temperature):
   
    x = reduced_frequency(wavenumber_cm, temperature)
    q = np.ones_like(x)
    m = x > 1e-8
    q[m] = x[m] / (-np.expm1(-x[m]))
    return q


def spectral_resolution_cm(w, delta_t_s, role):
    
    n = len(w)
    nfft = int(2 ** math.ceil(math.log2(max(64 * n, 1024))))
    if role == 'lag':
        sw = 2.0 * np.fft.rfft(np.asarray(w, dtype=float), n=nfft).real - w[0]
    else:
        sw = np.abs(np.fft.rfft(np.asarray(w, dtype=float), n=nfft)) ** 2
    freq = np.fft.rfftfreq(nfft, d=delta_t_s)
    peak = sw[0]
    if peak <= 0:
        return float('nan')
    below = np.where(sw < 0.5 * peak)[0]
    if len(below) == 0:
        return float('nan')
    i = below[0]
  
    f0, f1 = freq[i - 1], freq[i]
    s0, s1 = sw[i - 1], sw[i]
    f_half = f0 + (0.5 * peak - s0) * (f1 - f0) / (s1 - s0) if s1 != s0 else f1
    return 2.0 * f_half / C_CM_S


def nyquist_wavenumber(delta_t_s):
   
    return 1.0 / (2.0 * delta_t_s * C_CM_S)


def check_nyquist(plot_max_wavenumber, delta_t_s):
    nyq = nyquist_wavenumber(delta_t_s)
    if plot_max_wavenumber > nyq:
        print(f"WARNING: plot_max_wavenumber ({plot_max_wavenumber:.1f} cm-1) exceeds the "
              f"Nyquist limit 1/(2*dt*c) = {nyq:.1f} cm-1.")
        print(f"  Content above {nyq:.1f} cm-1 is aliased, not physical. "
              f"Reduce plot_max_wavenumber or use a smaller delta_t.")
    return nyq


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in ["true", "1", "yes", "y", "t"]


def build_masses_array(symbols, atomic_masses, context=""):
   
    missing = sorted({s for s in symbols if s not in atomic_masses})
    if missing:
        sys.exit(f"Error: no mass defined for element(s) {', '.join(missing)}{context}.\n"
                 f"  Add them to the 'masses' parameter, e.g. masses = C 12.011; H 1.008\n"
                 f"  Currently defined: {', '.join(sorted(atomic_masses))}")
    return np.array([atomic_masses[s] for s in symbols], dtype=float)


def validate_total_charge(charges_frame, symbols, charge_tolerance, fatal):
    
    tot_q = float(np.sum(charges_frame))
    natoms = len(charges_frame)
    print(f"\n--- Charge balance ---")
    counts = {}
    for s, q in zip(symbols, charges_frame):
        counts.setdefault(s, [0, 0.0])
        counts[s][0] += 1
        counts[s][1] += float(q)
    for s, (n, q_sum) in sorted(counts.items()):
        print(f"  {s:>3} x {n:<5} mean q = {q_sum / n:+.6f} e   subtotal {q_sum:+.6f} e")
    print(f"  total charge = {tot_q:+.8f} e   (tolerance {charge_tolerance:.2e} e)")
    if abs(tot_q) > charge_tolerance:
        msg = (f"Total charge {tot_q:+.6f} e exceeds charge_tolerance {charge_tolerance:.2e} e "
               f"({tot_q / natoms:+.6e} e per atom).\n"
               f"  A charged system contaminates J = sum_i q_i v_i with overall translation:\n"
               f"  COM correction enforces sum_i m_i v_i = 0, not sum_i v_i = 0, so with\n"
               f"  unequal masses the net-charge term does not cancel.\n"
               f"  Fix the static_charges so they sum to zero for this composition, or take\n"
               f"  them directly from the force field used to run the MD.")
        if fatal:
            sys.exit(f"Error: {msg}")
        print(f"  WARNING: {msg}")
        print(f"  Continuing because charge_tolerance_fatal = False.")
    else:
        print("  OK: system is neutral within tolerance.")
    return tot_q


def normalize_path_for_os(path):
    return os.path.normpath(path)


def print_ir_welcome_message():
    print("""
************************************************************
* IR Spectrum Calculator (VDOS-consistent axis)            *
* - Static or dynamic charges                             *
* - COM-corrected velocities                              *
* - Optional r·dq/dt term in dynamic mode                 *
* - ACF + FFT axis identical to VDOS implementation       *
************************************************************
""")


def infer_box_dimensions(positions_all):

    print("Inferring box dimensions from trajectory positions...")
    all_positions = positions_all.reshape(-1, 3) 
    global_min = np.min(all_positions, axis=0)
    global_max = np.max(all_positions, axis=0)
    
    lengths = global_max - global_min
    # A wrapped coordinate jumps by one box length, so where wraps occur the median
    # jump is the box length (the position extent is only a fallback). No minimum
    # size: a floor such as 10 A corrupts the unwrapping of every smaller box.
    for d in range(3):
        if lengths[d] > 1e-6:
            disp = np.abs(np.diff(positions_all[:, :, d], axis=0))
            jumps = disp[disp > lengths[d] / 2]
            if jumps.size:
                lengths[d] = np.median(jumps)
    
    print(f"Inferred box dimensions: Lx={lengths[0]:.2f}Å, Ly={lengths[1]:.2f}Å, Lz={lengths[2]:.2f}Å")
    print(f"Global position range: X[{global_min[0]:.2f}, {global_max[0]:.2f}], "
          f"Y[{global_min[1]:.2f}, {global_max[1]:.2f}], Z[{global_min[2]:.2f}, {global_max[2]:.2f}]")
    
    return lengths



def unwrap_positions(positions, box_lengths):
    nframes, natoms, ndim = positions.shape
    unwrapped = np.copy(positions)
    if ndim != 3 or len(box_lengths) != 3:
        return positions
    
    for d in range(3):
        L = box_lengths[d]
        if L <= 1e-6:
            continue

        disp = np.diff(positions[:, :, d], axis=0)
        
        jump_mask = np.abs(disp) > L/2
        jump_indices = np.where(jump_mask)
        
        corr = np.zeros((nframes, natoms))
        for frame_idx, atom_idx in zip(*jump_indices):
            if disp[frame_idx, atom_idx] > L/2:
                corr[frame_idx+1:, atom_idx] -= L
            elif disp[frame_idx, atom_idx] < -L/2:
                corr[frame_idx+1:, atom_idx] += L
                
        unwrapped[:, :, d] += corr
    
    return unwrapped


def parse_static_charges(charge_str):
    charges = {}
    if charge_str.strip():
        entries = charge_str.strip().split(';')
        for entry in entries:
            if entry.strip():
                parts = entry.split()
                if len(parts) >= 2:
                    at = parts[0].strip()
                    try:
                        qv = float(parts[1])
                        charges[at] = qv
                        print(f"Static charge for {at}: {qv}")
                    except ValueError:
                        pass
    return charges if charges else None
    
    
def parse_atomic_masses(mass_str):
    masses = {}
    if mass_str.strip():
        entries = mass_str.strip().split(';')
        for entry in entries:
            if entry.strip():
                parts = entry.split()
                if len(parts) >= 2:
                    at = parts[0].strip()
                    try:
                        mv = float(parts[1])
                        masses[at] = mv
                        print(f"Atomic mass for {at}: {mv}")
                    except ValueError:
                        pass
    return masses if masses else {'H': 1.008, 'C': 12.011, 'O': 16.00, 'N': 14.007}  # defaults
    

def save_ir_spectrum_data(wavenumbers, intensity, output_file, meta=None,
                          plot_min=None, plot_max=None):
    print(f"\nSaving IR spectrum data to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for line in (meta or ["IR spectrum calculated from MD trajectory"]):
            f.write(f"# {line}\n" if line else "#\n")
        f.write("#\n")
        f.write("# Wavenumber(cm-1)\tIntensity(a.u.)\n")
        f.write("#\n")
        for x, y in zip(wavenumbers, intensity):
            f.write(f"{x:.6f}\t{y:.8e}\n")
    print(f"IR spectrum data saved to: {output_file}")
    print("Note: the saved data is the RAW spectrum over the full frequency range;")
    if plot_min is not None and plot_max is not None:
        print(f"      plot_min_wavenumber={plot_min:g} / plot_max_wavenumber={plot_max:g} cm-1 "
              f"affect the plot only.")


def read_lammps_dump_with_q(input_file, require_charges=True):

    print("\nReading LAMMPS dump file with charges...")
    positions_all = []
    charges_all = []
    symbols_all = []
    box_lengths_all = []
    boundary_types_all = []
    with open(input_file, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("ITEM: TIMESTEP"):
                _ = f.readline()
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                natoms = int(f.readline().strip())
                positions = np.zeros((natoms, 3), float)
                charges = np.zeros(natoms, float)
                symbols = [""] * natoms
            elif line.startswith("ITEM: BOX BOUNDS"):
                parts = line.split()
                btypes = parts[3:]
                boundary_types_all.append(btypes)
                box = []
                for _ in range(3):
                    lo, hi = f.readline().split()[:2]
                    box.append(float(hi) - float(lo))
                box_lengths_all.append(box)
            elif line.startswith("ITEM: ATOMS"):
                headers = line.split()[2:]
                element_idx = headers.index("element") if "element" in headers else 0
                x_idx = headers.index("x")
                y_idx = headers.index("y")
                z_idx = headers.index("z")
                q_idx = headers.index("q") if "q" in headers else None
    
                if q_idx is None and require_charges:
                    sys.exit("ERROR: no q column in dump, which charge_mode = dynamic requires.\n"
                             "  Add q to your dump, e.g. 'dump 3 all custom 1 traj.xyz element x y z q',\n"
                             "  or use charge_mode = static.")
                for i in range(natoms):
                    parts = f.readline().split()
                    symbols[i] = parts[element_idx]
                    positions[i, 0] = float(parts[x_idx])
                    positions[i, 1] = float(parts[y_idx])
                    positions[i, 2] = float(parts[z_idx])
                    charges[i] = float(parts[q_idx]) if q_idx is not None else 0.0

                positions_all.append(positions.copy())
                charges_all.append(charges.copy())
                symbols_all.append(symbols.copy())
    if not positions_all:
        sys.exit(f"Error: no LAMMPS dump frames ('ITEM: ATOMS') found in '{input_file}'.\n"
                 "  charge_mode = dynamic needs a LAMMPS custom dump with a q column, e.g.\n"
                 "  'dump 3 all custom 1 traj.xyz element x y z q'. For plain XYZ files use\n"
                 "  charge_mode = static.")
    positions_all = np.array(positions_all)
    charges_all = np.array(charges_all)
    symbols_all = np.array(symbols_all, dtype=object)
    if box_lengths_all:
        box_lengths_mean = np.mean(box_lengths_all, axis=0)
        boundary_types = boundary_types_all[0]
    else:
        box_lengths_mean = np.array([0.0, 0.0, 0.0])
        boundary_types = ["non-periodic"] * 3
    print(f"Total frames read: {positions_all.shape[0]}")
    return positions_all, charges_all, symbols_all, box_lengths_mean, boundary_types
    
    
    
def read_trajectory_for_static_mode(input_file, static_charges_dict=None):

    print(f"\nAttempting flexible read of: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        first_lines = [f.readline().strip() for _ in range(10)]
    
    is_lammps_format = any(line.startswith("ITEM:") for line in first_lines)
    
    if is_lammps_format:
        print("Detected LAMMPS dump format. Using LAMMPS reader...")
        positions_all, charges_all, symbols_all, box_lengths, boundary_types = read_lammps_dump_with_q(
            input_file, require_charges=False)
        is_periodic = any("pp" in bt.lower() for bt in boundary_types)
        print(f"Successfully read LAMMPS file. Frames: {positions_all.shape[0]}, Atoms: {positions_all.shape[1]}")
        
        if static_charges_dict:
            print("Applying static charges override to LAMMPS file data...")
            for i_atom, sym in enumerate(symbols_all[0]):
                if sym in static_charges_dict:
                    charges_all[:, i_atom] = static_charges_dict[sym]
        return positions_all, charges_all, symbols_all, box_lengths, boundary_types, is_periodic
    
    else:
        print("Attempting to read as standard XYZ format...")
        try:
            from ase.io import read as ase_read
        except ImportError:
            sys.exit("Error: ASE required for XYZ files. Install with 'pip install ase'.")
        
        try:
            traj = ase_read(input_file, index=":")
            nframes = len(traj)
            natoms = len(traj[0])
            positions_all = np.array([fr.get_positions() for fr in traj])
            symbols_all = np.array([fr.get_chemical_symbols() for fr in traj], dtype=object)
            
            charges_all = np.zeros((nframes, natoms), float)
            
            if static_charges_dict:
                for i_atom, sym in enumerate(symbols_all[0]):
                    qv = static_charges_dict.get(sym, 0.0)
                    charges_all[:, i_atom] = qv

            box_lengths = np.array([0.0, 0.0, 0.0])
            boundary_types = ["non-periodic"] * 3
            
            if hasattr(traj[0], 'get_cell') and hasattr(traj[0], 'get_pbc'):
                cell = traj[0].get_cell()
                pbc_flags = traj[0].get_pbc()
                if np.all(cell.lengths() > 1e-6) and np.any(pbc_flags):
                    box_lengths = np.diag(cell) if np.allclose(cell, np.diag(np.diag(cell))) else cell.lengths()
                    boundary_types = ["periodic" if pbc_flags[i] else "non-periodic" for i in range(3)]
                    print(f"Box dimensions from ASE frame: {box_lengths}")
            
            is_periodic = any("periodic" in bt.lower() for bt in boundary_types)
            print(f"Successfully read XYZ file. Frames: {nframes}, Atoms: {natoms}")
            return positions_all, charges_all, symbols_all, box_lengths, boundary_types, is_periodic
            
        except Exception as e:
            print(f"ASE failed to read file: {e}")
            print("Attempting manual XYZ/extended XYZ read...")
    
    return read_manual_xyz_format(input_file, static_charges_dict)


def read_manual_xyz_format(input_file, static_charges_dict=None):
    print("Using manual XYZ format reader...")
    
    positions_all = []
    symbols_all = []
    charges_all = []
    box_lengths = np.array([0.0, 0.0, 0.0])
    boundary_types = ["non-periodic"] * 3
    is_periodic = False
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        try:
            natoms = int(line)
        except ValueError:
            i += 1
            continue
        
        i += 1  
        
        positions = []
        symbols = []
        charges = []
        
        for j in range(natoms):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            if len(parts) < 4:
                i += 1
                continue
            
            sym = parts[0]
            x, y, z = map(float, parts[1:4])
            
            q = 0.0
            if len(parts) >= 5:
                try:
                    q = float(parts[4])
                except ValueError:
                    pass
            
            symbols.append(sym)
            positions.append([x, y, z])
            charges.append(q)
            i += 1
        
        if len(positions) == natoms:
            positions_all.append(np.array(positions))
            symbols_all.append(np.array(symbols))
            charges_all.append(np.array(charges))
    
    if not positions_all:
        sys.exit(f"Error: Could not parse any frames from {input_file}")
    
    positions_all = np.array(positions_all)
    symbols_all = np.array(symbols_all, dtype=object)
    charges_all = np.array(charges_all)
    
    nframes = positions_all.shape[0]
    natoms = positions_all.shape[1]
    
    if static_charges_dict:
        for i_atom, sym in enumerate(symbols_all[0]):
            if sym in static_charges_dict:
                charges_all[:, i_atom] = static_charges_dict[sym]
    
    print(f"Manually read {nframes} frames, {natoms} atoms")
    return positions_all, charges_all, symbols_all, box_lengths, boundary_types, is_periodic
    
    


def run_ir_calculation(input_file_path):
    print_ir_welcome_message()

    params = {}
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                params[k.strip()] = v.strip()

    input_file_raw = params.get("input_file", "traj-custom.xyz")
    ir_output = params.get("ir_output", "ir_spectrum.txt")
    ir_plot = params.get("ir_plot", "ir_spectrum.png")
    dpi = float(params.get("dpi", "300"))
    temperature = float(params.get("temperature", "300.0"))

    pbc_str = params.get("PBC", "False").strip().lower()
    pbc = pbc_str in ["true", "1", "yes", "y", "t"]
    com_correction_str = params.get("Center_of_mass_correction", "True").strip().lower()
    com_correction = com_correction_str in ["true", "1", "yes", "y", "t"]
    delta_t = float(params.get("delta_t", "0.25"))      
    nskip = int(params.get("nskip", "0"))
    nmeasure = int(params.get("nmeasure", "1"))
    charge_mode_raw = params.get("charge_mode", "static").lower()
    charge_mode = "dynamic" if charge_mode_raw == "dynamic" else "static"
    static_charges_str = params.get("static_charges", "")
    window_kind = params.get("window_kind", "Gaussian")
    masses_str = params.get("masses", "C 12.011; H 1.008")  
    atomic_masses = parse_atomic_masses(masses_str)
    window_width_ps = float(params.get("window_width_ps", "1.0"))  

    quantum_correction = parse_bool(params.get("quantum_correction"), True)


    spectral_estimator = params.get("spectral_estimator", "acf").strip().lower()
    if spectral_estimator not in VALID_ESTIMATORS:
        sys.exit(f"Error: spectral_estimator must be one of {', '.join(VALID_ESTIMATORS)}. "
                 f"Got '{spectral_estimator}'.")
    welch_segment_ps = float(params.get("welch_segment_ps", str(window_width_ps)))
    welch_overlap = float(params.get("welch_overlap", "0.5"))
    if not 0.0 <= welch_overlap < 1.0:
        sys.exit("Error: welch_overlap must be in [0.0, 1.0).")

    plot_min_wavenumber = float(params.get("plot_min_wavenumber", "50.0"))
    plot_max_wavenumber = float(params.get("plot_max_wavenumber", "4500.0"))

    charge_tolerance = float(params.get("charge_tolerance", "1e-6"))
    charge_tolerance_fatal = parse_bool(params.get("charge_tolerance_fatal"), False)

    dqdt_savgol_window = int(params.get("dqdt_savgol_window", "5"))
    if dqdt_savgol_window % 2 == 0 or dqdt_savgol_window < 5:
        sys.exit("Error: dqdt_savgol_window must be an odd integer >= 5.")

    if nmeasure < 1:
        sys.exit("Error: nmeasure must be >= 1.")
    if window_kind.lower() not in VALID_WINDOWS:
        sys.exit(f"Error: window_kind must be one of {', '.join(VALID_WINDOWS)}. Got '{window_kind}'.")

    input_file = normalize_path_for_os(input_file_raw)
    if not os.path.exists(input_file):
        sys.exit(f"Error: Input file '{input_file}' not found.")
    with open(input_file, 'rb') as f:
        magic = f.read(4)
    if magic[:3] == b'CDF' or magic == b'\x89HDF':
        sys.exit(f"Error: '{input_file}' is a NetCDF file, which IR.py cannot read.\n"
                 "  IR.py reads XYZ and LAMMPS dump text trajectories; NetCDF is supported by vdos.py only.")

    print(f"\n{'='*60}")
    print(f"READING TRAJECTORY FILE: {input_file}")
    print(f"CHARGE MODE: {charge_mode.upper()}")
    print(f"{'='*60}")

    static_charges = parse_static_charges(static_charges_str) if static_charges_str.strip() else None

    if charge_mode == "dynamic":
        positions_all, charges_all, symbols_all, box_lengths, boundary_types = read_lammps_dump_with_q(input_file)
        is_periodic = pbc
        if is_periodic:
            if np.all(box_lengths > 1e-6):  
                print(f"PBC enabled by user parameter (overriding auto-detection)")
                positions_all = unwrap_positions(positions_all, box_lengths)
            else:
                print(f"WARNING: PBC enabled but no valid box dimensions available. Cannot unwrap positions.")
                print(f"Please ensure your trajectory contains cell information or use a format that supports PBC.")
        else:
            print(f"PBC disabled by user parameter (overriding auto-detection)")
    else:
        print("\nSTATIC MODE: Using flexible file reader...")
        
        try:
            from ase.io import read as ase_read
        except ImportError:
            sys.exit("Error: ASE required for XYZ files. Install with 'pip install ase'.")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            first_lines = [f.readline().strip() for _ in range(10)]
        
        is_lammps_format = any(line.startswith("ITEM:") for line in first_lines)
        
        if is_lammps_format:
            print("Detected LAMMPS dump format. Using LAMMPS reader...")
            positions_all, charges_all, symbols_all, box_lengths, boundary_types = read_lammps_dump_with_q(
                input_file, require_charges=False)
            is_periodic = any("pp" in bt.lower() for bt in boundary_types)
            print(f"Successfully read LAMMPS file. Frames: {positions_all.shape[0]}, Atoms: {positions_all.shape[1]}")

            if static_charges:
                print("Applying static charges override to LAMMPS file data...")
                unknown = sorted({s for s in symbols_all[0] if s not in static_charges})
                if unknown:
                    sys.exit(f"Error: no static charge defined for element(s) {', '.join(unknown)}.\n"
                             f"  Every element must appear in static_charges, otherwise it is\n"
                             f"  silently treated as neutral and its motion is invisible to the\n"
                             f"  dipole current. Defined: {', '.join(sorted(static_charges))}")
                for i_atom, sym in enumerate(symbols_all[0]):
                    charges_all[:, i_atom] = static_charges[sym]
        else:
            print("Reading as standard XYZ format...")
            traj = ase_read(input_file, index=":")
            nframes = len(traj)
            natoms = len(traj[0])
            positions_all = np.array([fr.get_positions() for fr in traj])
            symbols_all = np.array([fr.get_chemical_symbols() for fr in traj], dtype=object)
            charges_all = np.zeros((nframes, natoms), float)
            
            if static_charges:
                unknown = sorted({s for s in symbols_all[0] if s not in static_charges})
                if unknown:
                    sys.exit(f"Error: no static charge defined for element(s) {', '.join(unknown)}.\n"
                             f"  Every element must appear in static_charges, otherwise it is\n"
                             f"  silently treated as neutral and its motion is invisible to the\n"
                             f"  dipole current. Defined: {', '.join(sorted(static_charges))}")
                for i_atom, sym in enumerate(symbols_all[0]):
                    charges_all[:, i_atom] = static_charges[sym]

            box_lengths = np.array([0.0, 0.0, 0.0])
            boundary_types = ["non-periodic"] * 3
            
            if hasattr(traj[0], 'get_cell') and hasattr(traj[0], 'get_pbc'):
                cell = traj[0].get_cell()
                pbc_flags = traj[0].get_pbc()
                if np.all(cell.lengths() > 1e-6) and np.any(pbc_flags):
                    box_lengths = np.diag(cell) if np.allclose(cell, np.diag(np.diag(cell))) else cell.lengths()
                    boundary_types = ["periodic" if pbc_flags[i] else "non-periodic" for i in range(3)]
                    print(f"Box dimensions from ASE frame: {box_lengths}")
            
            is_periodic = any("periodic" in bt.lower() for bt in boundary_types)
            print(f"Successfully read XYZ file. Frames: {nframes}, Atoms: {natoms}")
        
        user_pbc_str = params.get("PBC", "False").strip().lower()
        user_pbc = user_pbc_str in ["true", "1", "yes", "y", "t"]
        
        if user_pbc and not is_periodic:
            print(f"User requested PBC=True but file appears non-periodic. Enabling PBC with inferred box...")
            if np.all(box_lengths < 1e-6):  # No valid box info
                box_lengths = infer_box_dimensions(positions_all)
            is_periodic = True
        elif not user_pbc and is_periodic:
            print(f"User requested PBC=False but file appears periodic. Disabling PBC...")
            is_periodic = False
        
        print(f"Final PBC setting: {is_periodic}, Box dimensions: {box_lengths}")

        is_periodic = pbc

        if is_periodic:
            if np.all(box_lengths < 1e-6): 
                print(f"PBC enabled by user parameter, but no cell info available in XYZ file")
                box_lengths = infer_box_dimensions(positions_all)
                boundary_types = ["periodic", "periodic", "periodic"]
                print(f"Using inferred box dimensions: {box_lengths}")
            else:
                print(f"PBC enabled by user parameter with box dimensions: {box_lengths}")
        else:
            print(f"PBC disabled by user parameter")

        if is_periodic and np.any(box_lengths > 1e-6):
            print(f"Unwrapping positions for periodic boundary conditions...")
            positions_all = unwrap_positions(positions_all, box_lengths)
        elif is_periodic:
            print(f"WARNING: Cannot unwrap positions - invalid box dimensions: {box_lengths}")

    nframes, natoms, _ = positions_all.shape
    print(f"Frames: {nframes}, Atoms: {natoms}")
    print(f"System periodic: {is_periodic}")

    indices = np.arange(nskip, nframes, nmeasure)
    selected_positions = positions_all[indices]
    selected_charges = charges_all[indices]
    n_selected = len(indices)
    print(f"Using {n_selected} frames (nskip={nskip}, nmeasure={nmeasure})")

    delta_t_s = delta_t * nmeasure * 1e-15

    nyq_cm = check_nyquist(plot_max_wavenumber, delta_t_s)

    velocities_raw = np.gradient(selected_positions, delta_t_s, axis=0)
    atten = np.sinc(2.0 * plot_max_wavenumber * C_CM_S * delta_t_s) ** 2   # central differences
    if atten < 0.95:
        print(f"WARNING: numerical velocities attenuate the spectrum by "
              f"{100.0 * (1.0 - atten):.0f}% at {plot_max_wavenumber:.0f} cm-1 "
              f"(effective dt = {delta_t * nmeasure:.3f} fs). Reduce delta_t or nmeasure.")

    masses_array = build_masses_array(symbols_all[0], atomic_masses,
                                      context=" (from the trajectory)")

    tot_q = validate_total_charge(selected_charges[0], list(symbols_all[0]),
                                  charge_tolerance, charge_tolerance_fatal)

    if com_correction:
        print(f"Applying Center of Mass correction (enabled by user)")
        total_mass = np.sum(masses_array)
        com_vel = np.sum(velocities_raw * masses_array[np.newaxis, :, np.newaxis], axis=1) / total_mass
        velocities = velocities_raw - com_vel[:, np.newaxis, :]
        com_pos = np.sum(selected_positions * masses_array[np.newaxis, :, np.newaxis], axis=1) / total_mass
        positions_com = selected_positions - com_pos[:, np.newaxis, :]
    else:
        print(f"Center of Mass correction disabled by user parameter")
        velocities = velocities_raw.copy()
        positions_com = selected_positions.copy()

    if charge_mode == "dynamic":
        print(f"Dynamic charges: using dq/dt term with Savitzky-Golay "
              f"(window_length={dqdt_savgol_window}, polyorder=2).")
        dqdt = savgol_filter(
            selected_charges,
            window_length=dqdt_savgol_window,
            polyorder=2,
            deriv=1,
            delta=delta_t_s,
            axis=0
        )
    else:
        print("Static charges: dq/dt = 0; J(t) = sum_i q_i v_i(t).")
        dqdt = np.zeros_like(selected_charges)


    print("\nCalculating dipole current J(t) = sum_i q_i v_i ...")
    dipole_current = np.einsum('ti,tia->ta', selected_charges, velocities)
    if charge_mode == "dynamic":

        dipole_current += np.einsum('ti,tia->ta', dqdt, positions_com)

    print("Removing DC component...")
    dipole_current -= np.mean(dipole_current, axis=0)

    print(f"\nSpectral estimator: {spectral_estimator}")
    est_meta = {}

    if spectral_estimator == "welch":
        freq, intensity_full, nperseg, noverlap, nseg = spectrum_welch(
            dipole_current, delta_t_s, window_kind, welch_segment_ps, welch_overlap)
        est_meta = {"segment_frames": nperseg, "overlap_frames": noverlap, "segments": nseg,
                    "bin_spacing_cm": 1.0 / (nperseg * delta_t_s * C_CM_S),
                    "resolution_cm": spectral_resolution_cm(
                        signal_window(nperseg, window_kind), delta_t_s, 'signal')}
        print(f"Welch: {nseg} segments of {nperseg} frames "
              f"({nperseg * delta_t * nmeasure / 1000.0:.3f} ps), overlap {noverlap} frames")
        print(f"  bin spacing {est_meta['bin_spacing_cm']:.2f} cm-1, "
              f"effective resolution {est_meta['resolution_cm']:.1f} cm-1")
    else:
        w_lag = lag_window(n_selected, window_kind, delta_t * nmeasure, window_width_ps)
        acf_total = vector_acf_biased(dipole_current)
        freq, intensity_full, nfft = spectrum_from_acf(acf_total, w_lag, delta_t_s)
        est_meta = {"lags": n_selected, "nfft": nfft,
                    "bin_spacing_cm": 1.0 / (nfft * delta_t_s * C_CM_S),
                    "resolution_cm": spectral_resolution_cm(w_lag, delta_t_s, 'lag')}
        print(f"Blackman-Tukey: {n_selected} lags, nfft={nfft}, "
              f"lag window w(0)=1 ({window_kind}, "
              f"{f'{window_width_ps} ps' if window_kind.lower() == 'gaussian' else 'window_width_ps unused'})")
        print(f"  bin spacing {est_meta['bin_spacing_cm']:.2f} cm-1, "
              f"effective resolution {est_meta['resolution_cm']:.1f} cm-1")

    wavenumber = freq / C_CM_S
    intensity = intensity_full

    if quantum_correction:
        print("\nApplying the harmonic quantum correction factor x/(1-exp(-x)) once...")
        intensity = intensity * qcf_harmonic(wavenumber, temperature)
        spectrum_kind = ("quantum corrected IR (dipole-current PSD x harmonic QCF "
                         "x/(1-exp(-x)))")
    else:
        print("\nNo quantum correction applied (classical dipole-current spectrum).")
        spectrum_kind = "raw classical IR (dipole-current PSD, no quantum correction)"

    if np.max(intensity) <= 1e-12:
        print("ERROR: All intensity values are zero or near-zero. Check charges and J(t).")
        sys.exit(1)

    print(f"IR spectrum calculated. Range: {np.min(wavenumber):.1f}-{np.max(wavenumber):.1f} cm⁻¹")
    print(f"Number of points: {len(wavenumber)}")
    trajectory_length_ps = n_selected * delta_t * nmeasure / 1000.0

    if spectral_estimator == "welch":
        nu_dof = 2.0 * est_meta["segments"] * 3.0
    else:

        eff_lags = max(float(np.sum(w_lag)), 1.0)
        nu_dof = 2.0 * (n_selected / eff_lags) * 3.0
    rel_err = math.sqrt(2.0 / nu_dof) if nu_dof > 0 else float('nan')
    print(f"\n--- Statistical uncertainty ---")
    print(f"  trajectory length            : {trajectory_length_ps:.3f} ps")
    print(f"  equivalent degrees of freedom: {nu_dof:.0f}")
    print(f"  approx. relative error on band intensities: {100.0 * rel_err:.0f}%")
    if rel_err > 0.20:
        print("  NOTE: this is set by the trajectory length and the requested resolution,")
        print("        not by the estimator. Longer or multiple trajectories are the only fix.")

    #  metadata 
    meta = [
        "IR spectrum calculated from MD trajectory (IR.py, VCL-toolkit)",
        f"date                    : {datetime.now().isoformat(timespec='seconds')}",
        f"input trajectory        : {input_file}",
        f"spectrum content        : {spectrum_kind}",
        f"spectral estimator      : {spectral_estimator}",
        f"charge mode             : {charge_mode}",
        f"static charges          : "
        + ('; '.join(f'{k} {v}' for k, v in sorted(static_charges.items())) if static_charges
           else 'n/a (charges taken from trajectory)'),
        f"total charge            : {tot_q:+.8f} e (tolerance {charge_tolerance:.2e} e, "
        f"fatal={charge_tolerance_fatal})",
        f"dipole current          : J(t) = sum_i q_i v_i(t)"
        + ("  +  sum_i (dq_i/dt) r_i(t)" if charge_mode == "dynamic" else ""),
        f"velocity source         : numerical (np.gradient of positions)",
        f"effective timestep      : {delta_t * nmeasure:.6f} fs "
        f"(delta_t={delta_t} fs x nmeasure={nmeasure})",
        f"frames used             : {n_selected} ({trajectory_length_ps:.4f} ps), nskip={nskip}",
        f"window kind             : {window_kind}",
    ]
    if spectral_estimator == "welch":
        meta += [
            f"window role             : centred signal window, applied per segment by scipy.signal.welch",
            f"segment duration        : {welch_segment_ps} ps ({est_meta['segment_frames']} frames)",
            f"segment overlap         : {welch_overlap:.2f} ({est_meta['overlap_frames']} frames)",
            f"segments averaged       : {est_meta['segments']}",
        ]
    else:
        meta += [
            f"window role             : decaying one-sided lag window, w(0)=1",
            (f"lag window duration     : {window_width_ps} ps (FWHM of equivalent symmetric window)"
             if window_kind.lower() == "gaussian" else
             f"lag window duration     : all {est_meta['lags']} lags ({window_kind}; window_width_ps unused)"),
            f"lags / nfft             : {est_meta['lags']} / {est_meta['nfft']} (nfft >= 2N-1)",
        ]
    meta += [
        f"bin spacing             : {est_meta['bin_spacing_cm']:.4f} cm-1 "
        f"(zero padding interpolates; it does not add information)",
        f"effective resolution    : {est_meta['resolution_cm']:.2f} cm-1 "
        f"(FWHM of the spectral window)",
        f"Nyquist limit           : {nyq_cm:.1f} cm-1",
        f"quantum correction      : {quantum_correction}"
        + (f" (harmonic QCF x/(1-exp(-x)), applied once, T={temperature} K)"
           if quantum_correction else ""),
        f"temperature             : {temperature} K",
        f"COM correction          : {com_correction}",
        f"PBC unwrapping          : {pbc}",
        f"masses                  : {'; '.join(f'{k} {v}' for k, v in sorted(atomic_masses.items()))}",
        f"est. relative error     : {100.0 * rel_err:.0f}% on band intensities "
        f"({nu_dof:.0f} equivalent dof)",
        "",
        "The spectrum below is the RAW, UNFILTERED estimate over the full frequency",
        f"range. Display limits (plot_min_wavenumber={plot_min_wavenumber:g}, "
        f"plot_max_wavenumber={plot_max_wavenumber:g} cm-1) affect the plot only.",
        "Intensity is in arbitrary units and is NOT normalised.",
    ]
    if charge_mode == "static":
        meta += [
            "",
            "LIMITATION: fixed charges omit charge flux (dq/dt), which for C-H stretches",
            "is comparable to the q*v term. Peak POSITIONS are reliable; relative IR",
            "INTENSITIES from a fixed-charge model are qualitative only.",
        ]

    save_ir_spectrum_data(wavenumber, intensity, ir_output, meta,
                          plot_min_wavenumber, plot_max_wavenumber)

    print("\nGenerating IR spectrum plot...")
    plot_mask = (wavenumber >= plot_min_wavenumber) & (wavenumber <= plot_max_wavenumber)
    if not np.any(plot_mask):
        sys.exit(f"Error: no spectral points between plot_min_wavenumber="
                 f"{plot_min_wavenumber} and plot_max_wavenumber={plot_max_wavenumber} cm-1.")
    wn_plot = wavenumber[plot_mask]
    int_plot = intensity[plot_mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(wn_plot, int_plot, 'b-', linewidth=2.0)

    from scipy.signal import find_peaks
    min_sep_cm = 120.0
    distance_bins = max(1, int(round(min_sep_cm / est_meta['bin_spacing_cm'])))
    peaks, _ = find_peaks(int_plot, height=np.max(int_plot) * 0.1, distance=distance_bins)
    for idx in peaks:
        ax.axvline(x=wn_plot[idx], color='r', alpha=0.3, linestyle='--')
        ax.text(wn_plot[idx], np.max(int_plot)*0.1,
                f"{wn_plot[idx]:.0f}", rotation=90,
                va='bottom', fontsize=8, color='red')

    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14)
    ax.set_ylabel("Intensity (a.u.)", fontsize=14)
    ax.set_title("IR Absorption Spectrum", fontsize=16, fontweight='bold')
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.offsetText.set_fontsize(14)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=plot_max_wavenumber)
    ax.grid(True, linestyle="--", alpha=0.7)

    annotation_text = (
        f"PBC: {pbc}\n"
        f"COM Correction: {com_correction}\n"
        f"T = {temperature} K\n"
        f"Δt = {delta_t} fs, nmeasure = {nmeasure}\n"
        f"estimator: {spectral_estimator}\n"
        f"QCF: {'harmonic' if quantum_correction else 'none'}\n"
        f"charges: {charge_mode}, Σq = {tot_q:+.3f} e"
    )
    ax.text(0.02, 0.95, annotation_text,
            transform=ax.transAxes, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig(ir_plot, dpi=dpi)
    plt.close()
    print(f"IR spectrum plot saved to {ir_plot}")
    print(f"Peak wavenumbers (cm⁻¹): {[f'{wn_plot[i]:.0f}' for i in peaks]}")


if __name__ == "__main__":
    os.environ["PYTHONUTF8"] = "1"
    parser = argparse.ArgumentParser(description="IR spectrum from trajectory (static/dynamic charges, VDOS-aligned).")
    parser.add_argument("input_file", type=str, help="Input parameter file (input_IR.txt)")
    args = parser.parse_args()
    if not os.path.exists(args.input_file):
        sys.exit(f"Error: Input parameter file '{args.input_file}' not found.")
    run_ir_calculation(args.input_file)
