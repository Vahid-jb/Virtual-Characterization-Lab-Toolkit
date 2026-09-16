# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
from scipy import signal
import io

if sys.platform == "win32":
    if sys.stdout is not None and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if sys.stderr is not None and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)


C_CM_S = 2.9979245899e10         # cm/s (match IR code)
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


def qcf_kinetic_energy(wavenumber_cm, temperature):

    x = reduced_frequency(wavenumber_cm, temperature)
    q = np.ones_like(x)
    m = x > 1e-8
    q[m] = (x[m] / 2.0) / np.tanh(x[m] / 2.0)
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


def normalize_path_for_os(path):
    return os.path.normpath(path)
    
def unwrap_positions(positions, box_lengths):
    nframes, natoms, ndim = positions.shape
    unwrapped = np.copy(positions)

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
                    except ValueError:
                        pass

    return masses if masses else {'H': 1.008, 'C': 12.011, 'O': 16.00, 'N': 14.007}


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


def unwrap_netcdf_coordinates(trajectory, coordinates):
    """Unwrap NetCDF coordinates with the AMBER 'cell_lengths' (orthorhombic), else an inferred box."""
    if 'cell_lengths' in trajectory.variables:
        box_lengths = np.array(trajectory.variables['cell_lengths'].data, dtype=float).reshape(-1, 3).mean(axis=0)
        print(f"Box dimensions from NetCDF cell_lengths: {box_lengths}")
    else:
        print("PBC enabled by user parameter, but the NetCDF file has no cell_lengths")
        box_lengths = infer_box_dimensions(coordinates)
    if not np.any(box_lengths > 1e-6):
        print(f"WARNING: Cannot unwrap positions - invalid box dimensions: {box_lengths}")
        return coordinates, "False (no valid box dimensions)"
    print("Unwrapping positions for periodic boundary conditions...")
    return unwrap_positions(coordinates, box_lengths), "True"


def run_vdos_script(input_file_path):
    """Main function for the VDOS calculation and visualization."""
    
    def print_vdos_welcome_message():
        print("""
************************************************************
* This tool calculates the Vibrational Density of States   *
* (VDOS) from molecular dynamics trajectory files.         *
* Assumes input trajectory is SORTED (dump_modify sort id) *
************************************************************
""")
    
    def apply_frame_selection(coordinates, velocities=None, distances=None, mode="full", contains_velocities=False, force_numerical=False, nskip=0, nmeasure=1):
        """Apply frame selection to trajectory data."""
        nsteps = len(coordinates)
        if nskip > 0 or nmeasure > 1:
            print(f"Applying frame selection: nskip={nskip}, nmeasure={nmeasure}")
            nskip = min(nskip, nsteps-1)
            indices = np.arange(nskip, nsteps, nmeasure)
            if len(indices) == 0:
                sys.exit(f"Error: No frames selected after nskip={nskip} and nmeasure={nmeasure}. Reduce nskip or nmeasure.")
            
            coordinates = coordinates[indices]
            nsteps = len(coordinates)
            
            if mode == "full":
                if velocities is not None and contains_velocities and not force_numerical:
                    velocities = velocities[indices]
            else:
                if distances is not None:
                    distances = distances[indices]
            
            print(f"Using {nsteps} frames after selection (nskip={nskip}, nmeasure={nmeasure})")
        
        return coordinates, velocities, distances, nsteps
    
    def check_libraries_and_file(input_name):
        if (input_name.endswith('.xyz') or input_name.endswith('.XYZ')):
            try:
                from ase.io import read
            except ImportError:
                sys.exit("Error: ASE library is required for XYZ files but is not installed. Please install it with 'pip install ase'.")
        if not os.path.exists(input_name):
            sys.exit(f"Error: Input file '{input_name}' not found.")
    
    def calc_derivative(array_1D, delta_t):
        dy = np.gradient(array_1D)
        return np.divide(dy, delta_t)
    
    def zero_padding(sample_data):
        return int(2 ** math.ceil(math.log(len(sample_data), 2)))

    print_vdos_welcome_message()
    
    params = {}
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        params[key.strip()] = value.strip()
    except Exception as e:
        sys.exit(f"Error reading input file '{input_file_path}': {e}")
    
    input_file_raw = params.get('input_file', 'traj.xyz')
    output_data_raw = params.get('output_data', 'VDOS.txt')
    output_info = params.get('output_plot', 'VDOS.png')
    save_dpi = params.get('dpi', '150')
    mode = params.get('mode', 'full').lower()
    delta_t = float(params.get('delta_t', '0.25'))
    window_kind = params.get('window_kind', 'Gaussian')
    force_numerical = parse_bool(params.get('force_numerical'), False)
    window_width_ps = float(params.get('window_width_ps', '1.0'))

    spectral_estimator = params.get('spectral_estimator', 'acf').strip().lower()
    if spectral_estimator not in VALID_ESTIMATORS:
        sys.exit(f"Error: spectral_estimator must be one of {', '.join(VALID_ESTIMATORS)}. "
                 f"Got '{spectral_estimator}'.")
    welch_segment_ps = float(params.get('welch_segment_ps', str(window_width_ps)))
    welch_overlap = float(params.get('welch_overlap', '0.5'))
    if not 0.0 <= welch_overlap < 1.0:
        sys.exit("Error: welch_overlap must be in [0.0, 1.0).")

    VELOCITY_UNITS = {'angstrom/ps': 1.0e12, 'angstrom/fs': 1.0e15, 'angstrom/s': 1.0}
    velocity_unit = params.get('velocity_unit', 'angstrom/ps').strip().lower()
    if velocity_unit not in VELOCITY_UNITS:
        sys.exit(f"Error: velocity_unit must be one of {', '.join(VELOCITY_UNITS)}. "
                 f"Got '{velocity_unit}'.")
    velocity_scale = VELOCITY_UNITS[velocity_unit]

    plot_min_wavenumber = float(params.get('plot_min_wavenumber', '50.0'))
    plot_max_wavenumber = float(params.get('plot_max_wavenumber', '4500.0'))

    if parse_bool(params.get('use_normalized_vectors'), False):
        sys.exit("Error: use_normalized_vectors is no longer a supported VDOS mode.\n"
                 "  |r| is origin-dependent and |v| is a non-negative scalar whose spectrum\n"
                 "  contains doubled frequencies and a large DC component; neither is a\n"
                 "  vibrational density of states. Set use_normalized_vectors = False.")

    masses_str = params.get("masses", "C 12.011; H 1.008")
    atomic_masses = parse_atomic_masses(masses_str)

    PBC = parse_bool(params.get("PBC"), False)
    COM_CORRECTION = parse_bool(params.get("Center_of_mass_correction"), True)

    nskip = int(params.get('nskip', '0'))
    nmeasure = int(params.get('nmeasure', '1'))
    if nmeasure < 1:
        sys.exit("Error: nmeasure must be >= 1.")

    temperature = float(params.get('temperature', '300.0')) 
    quantum_correction = parse_bool(params.get('quantum_correction'), False)

    output_name, output_format = os.path.splitext(output_info)
    output_format = output_format.strip('.').lower()
    if not output_format:
        output_format = "png"
    
    if output_format == "pdf":
        save_kwargs = {'dpi': float(save_dpi), 'metadata': {
            'CreationDate': datetime.now(),
            'Creator': 'LAMMPS Postprocessor',
            'Producer': 'Matplotlib'
        }}
    elif output_format in ['eps', 'ps']:
        save_kwargs = {'dpi': float(save_dpi)}
    elif output_format in ['svg', 'svgz']:
        save_kwargs = {'dpi': float(save_dpi)}
    else:
        save_kwargs = {'dpi': float(save_dpi)}
    
    output_png_raw = output_name + "." + output_format
    input_file = normalize_path_for_os(input_file_raw)
    output_data = normalize_path_for_os(output_data_raw)
    output_png = normalize_path_for_os(output_png_raw)
    
    check_libraries_and_file(input_file)
    
    if mode not in ['full', 'bond']:
        sys.exit("Error: Mode must be 'full' or 'bond'.")
    
    delta_t_s = delta_t * nmeasure * 1e-15
    
    bond_indices = None
    
    if mode == 'bond':
        bond_str = params.get('bond_indices', '').split()
        if len(bond_str) != 2:
            sys.exit("Error: For 'bond' mode, two integer indices are required in bond_indices parameter.")
        bond_indices = [int(i) - 1 for i in bond_str]
    
    if window_kind.lower() not in VALID_WINDOWS:
        sys.exit(f"Error: Window kind for FFT must be one of: {', '.join(VALID_WINDOWS)}. Got: '{window_kind}'")

    nyq_cm = check_nyquist(plot_max_wavenumber, delta_t_s)

    contains_velocities = False
    velocities = None
    pbc_status = "False" if not PBC else "False (no valid box dimensions)"
    velocity_source = "numerical (np.gradient of positions)"

    if input_file.lower().endswith('.xyz'):
        try:
            from ase.io import read
        except ImportError:
            sys.exit("Error: ASE library is required for XYZ files. Install with 'pip install ase'.")
        
        print("\nReading XYZ file using ASE library...")
        trajectory = read(input_file, index=':')
        nsteps = len(trajectory)
        natoms = len(trajectory[0])
        coordinates = np.empty((nsteps, natoms, 3))
        for i, frame in enumerate(trajectory):
            coordinates[i] = frame.get_positions()

        box_lengths = np.array([0.0, 0.0, 0.0])
        boundary_types = ["non-periodic"] * 3
        
        if hasattr(trajectory[0], 'get_cell') and hasattr(trajectory[0], 'get_pbc'):
            cell = trajectory[0].get_cell()
            pbc_flags = trajectory[0].get_pbc()
            if np.all(cell.lengths() > 1e-6) and np.any(pbc_flags):
                box_lengths = np.diag(cell) if np.allclose(cell, np.diag(np.diag(cell))) else cell.lengths()
                boundary_types = ["periodic" if pbc_flags[i] else "non-periodic" for i in range(3)]
                print(f"Box dimensions from ASE frame: {box_lengths}")
        
       
        symbols = trajectory[0].get_chemical_symbols()
        print(f"XYZ file: Got symbols for {len(symbols)} atoms")

        is_periodic_auto = any("periodic" in bt.lower() for bt in boundary_types)
        
        user_pbc_str = params.get("PBC", "False").strip().lower()
        user_pbc = user_pbc_str in ["true", "1", "yes", "y", "t"]
        
        is_periodic = user_pbc 
        
        if is_periodic:
            if np.all(box_lengths < 1e-6): 
                print(f"PBC enabled by user parameter, but no cell info available")
                box_lengths = infer_box_dimensions(coordinates)
                boundary_types = ["periodic", "periodic", "periodic"]
                print(f"Using inferred box dimensions: {box_lengths}")
            else:
                print(f"PBC enabled by user parameter with box dimensions: {box_lengths}")
        else:
            print(f"PBC disabled by user parameter")

        if is_periodic:
            if np.any(box_lengths > 1e-6): 
                print(f"Unwrapping positions for periodic boundary conditions...")
                coordinates = unwrap_positions(coordinates, box_lengths)
                pbc_status = "True"
            else:
                print(f"WARNING: Cannot unwrap positions - invalid box dimensions: {box_lengths}")
        else:
            print(f"PBC disabled")

        
        if mode == "full":
            print("\nCalculating VDOS for all atoms.")
            print("Velocities will be calculated numerically from positions.")
        else:
            print("\nCalculating VDOS for the specified bond.")
            print("Derivatives will be calculated numerically.")
            distances = np.linalg.norm((coordinates[:, bond_indices[0], :] - coordinates[:, bond_indices[1], :]), axis=1)

        coordinates, _, distances, nsteps = apply_frame_selection(
            coordinates, None, distances if mode == "bond" else None,
            mode, False, False, nskip, nmeasure
        )

    else:
        try:
            from scipy.io import netcdf_file
        except ImportError:
            sys.exit("Error: SciPy library is required for NetCDF files but is not installed. Please install it with 'pip install scipy'.")
        
        print("\nReading NETCDF file using SciPy library...")
        try:
            trajectory = netcdf_file(input_file, 'r')
        except Exception as e:
            sys.exit(f"Error opening NetCDF file: {e}")
        
        if mode == "full":
            print("\nCalculating VDOS for all atoms.")
            contains_velocities = "velocities" in trajectory.variables

            coordinates = np.array(trajectory.variables['coordinates'].data)
            nsteps = len(coordinates)
            natoms = len(coordinates[0])
            if contains_velocities and not force_numerical:
                print("Velocities will be read from the trajectory file.")
                velocities = np.array(trajectory.variables['velocities'].data)
                if len(velocities) != nsteps or len(velocities[0]) != natoms:
                    sys.exit(f"Error: NetCDF 'velocities' shape {np.shape(velocities)} does not "
                             f"match 'coordinates' shape {np.shape(coordinates)}.")
                if PBC:
                    pbc_status = "not needed (stored velocities used, positions not differentiated)"
            else:
                if contains_velocities and force_numerical:
                    print("Found velocities but numerical calculation is forced.")
                print("Velocities will be calculated numerically from positions.")
                if PBC:
                    coordinates, pbc_status = unwrap_netcdf_coordinates(trajectory, coordinates)
        else:
            print("\nCalculating VDOS for the specified bond.")
            print("Derivatives will be calculated numerically.")
            coordinates = np.array(trajectory.variables['coordinates'].data)
            nsteps = len(coordinates)
            natoms = len(coordinates[0])
            if PBC:
                coordinates, pbc_status = unwrap_netcdf_coordinates(trajectory, coordinates)
            distances = np.linalg.norm((coordinates[:, bond_indices[0], :] - coordinates[:, bond_indices[1], :]), axis=1)

        symbols = params.get('symbols', '').split()
        if len(symbols) != natoms:
            sys.exit(f"Error: NetCDF files carry no chemical symbols, so the {natoms} atomic\n"
                     f"  masses cannot be assigned. Provide them via a 'symbols' parameter\n"
                     f"  with one entry per atom, in trajectory order, e.g.\n"
                     f"    symbols = C C H H H ...\n"
                     f"  (got {len(symbols)} entries). Previously every atom was silently\n"
                     f"  assumed to be carbon, which corrupts the mass weighting.")

        coordinates, velocities, distances, nsteps = apply_frame_selection(
            coordinates,
            velocities if (mode == "full" and contains_velocities and not force_numerical) else None,
            distances if mode == "bond" else None,
            mode, contains_velocities, force_numerical, nskip, nmeasure
        )

    masses_array = build_masses_array(symbols, atomic_masses, context=" (from the trajectory)")

    if contains_velocities and not force_numerical and velocities is not None and mode == "full":
        vel_all_raw = np.asarray(velocities, dtype=float) * velocity_scale
        velocity_source = f"stored in trajectory file (interpreted as {velocity_unit})"
        print(f"Using velocities stored in the trajectory file, interpreted as {velocity_unit}.")
        print("  If the sum-rule T_eff below is absurd, velocity_unit is wrong.")
    else:
        vel_all_raw = np.gradient(coordinates, delta_t_s, axis=0)
        velocity_source = "numerical (np.gradient of positions)"
        atten = np.sinc(2.0 * plot_max_wavenumber * C_CM_S * delta_t_s) ** 2   # central differences
        if atten < 0.95:
            print(f"WARNING: numerical velocities attenuate the spectrum by "
                  f"{100.0 * (1.0 - atten):.0f}% at {plot_max_wavenumber:.0f} cm-1 "
                  f"(effective dt = {delta_t * nmeasure:.3f} fs).")
            print("  Reduce delta_t/nmeasure, or supply stored velocities.")

    if COM_CORRECTION:
        print("Applying center-of-mass velocity correction (with atomic masses)")
        total_mass = np.sum(masses_array)
        com_vel = np.sum(vel_all_raw * masses_array[np.newaxis, :, np.newaxis], axis=1) / total_mass
        vel_all = vel_all_raw - com_vel[:, np.newaxis, :]
        print(f"Total system mass: {total_mass:.2f} amu")
    else:
        print("Center-of-mass correction disabled")
        vel_all = vel_all_raw.copy()

    print(f"\nSpectral estimator: {spectral_estimator}")
    est_meta = {}

    if mode == "full":
        if spectral_estimator == "welch":
            freq, psd_atom0, nperseg, noverlap, nseg = spectrum_welch(
                vel_all[:, 0, :], delta_t_s, window_kind, welch_segment_ps, welch_overlap)
            intensity_full = masses_array[0] * psd_atom0
            for i in range(1, natoms):
                f_i, psd_i, _, _, _ = spectrum_welch(
                    vel_all[:, i, :], delta_t_s, window_kind, welch_segment_ps, welch_overlap)
                intensity_full += masses_array[i] * psd_i
            est_meta = {"segment_frames": nperseg, "overlap_frames": noverlap, "segments": nseg,
                        "bin_spacing_cm": 1.0 / (nperseg * delta_t_s * C_CM_S),
                        "resolution_cm": spectral_resolution_cm(
                            signal_window(nperseg, window_kind), delta_t_s, 'signal')}
            print(f"Welch: {nseg} segments of {nperseg} frames "
                  f"({nperseg * delta_t * nmeasure / 1000.0:.3f} ps), overlap {noverlap} frames")
            print(f"  bin spacing {est_meta['bin_spacing_cm']:.2f} cm-1, "
                  f"effective resolution {est_meta['resolution_cm']:.1f} cm-1")
        else:
            w_lag = lag_window(nsteps, window_kind, delta_t * nmeasure, window_width_ps)
            acf_i = vector_acf_biased(vel_all[:, 0, :])
            freq, psd_i, nfft = spectrum_from_acf(acf_i, w_lag, delta_t_s)
            intensity_full = masses_array[0] * psd_i
            for i in range(1, natoms):
                acf_i = vector_acf_biased(vel_all[:, i, :])
                _, psd_i, _ = spectrum_from_acf(acf_i, w_lag, delta_t_s)
                intensity_full += masses_array[i] * psd_i
            est_meta = {"lags": nsteps, "nfft": nfft,
                        "bin_spacing_cm": 1.0 / (nfft * delta_t_s * C_CM_S),
                        "resolution_cm": spectral_resolution_cm(w_lag, delta_t_s, 'lag')}
            print(f"Blackman-Tukey: {nsteps} lags, nfft={nfft}, "
                  f"lag window w(0)=1 ({window_kind}, "
                  f"{f'{window_width_ps} ps' if window_kind.lower() == 'gaussian' else 'window_width_ps unused'})")
            print(f"  bin spacing {est_meta['bin_spacing_cm']:.2f} cm-1, "
                  f"effective resolution {est_meta['resolution_cm']:.1f} cm-1")
    else:
        distances_velocities = calc_derivative(distances, delta_t_s)
        if spectral_estimator == "welch":
            freq, intensity_full, nperseg, noverlap, nseg = spectrum_welch(
                distances_velocities, delta_t_s, window_kind, welch_segment_ps, welch_overlap)
            est_meta = {"segment_frames": nperseg, "overlap_frames": noverlap, "segments": nseg,
                        "bin_spacing_cm": 1.0 / (nperseg * delta_t_s * C_CM_S),
                        "resolution_cm": spectral_resolution_cm(
                            signal_window(nperseg, window_kind), delta_t_s, 'signal')}
        else:
            w_lag = lag_window(nsteps, window_kind, delta_t * nmeasure, window_width_ps)
            acf_b = vector_acf_biased(distances_velocities)
            freq, intensity_full, nfft = spectrum_from_acf(acf_b, w_lag, delta_t_s)
            est_meta = {"lags": nsteps, "nfft": nfft,
                        "bin_spacing_cm": 1.0 / (nfft * delta_t_s * C_CM_S),
                        "resolution_cm": spectral_resolution_cm(w_lag, delta_t_s, 'lag')}

    wavenumber = freq / C_CM_S
    intensity = intensity_full

    t_eff = None
    if mode == "full":
        amu_to_kg = 1.66053906660e-27
        ang_to_m = 1.0e-10
        integral = np.trapezoid(intensity, freq) if hasattr(np, "trapezoid") \
            else np.trapz(intensity, freq)
        sum_mv2 = integral * amu_to_kg * ang_to_m ** 2
        n_dof = 3 * natoms - (3 if COM_CORRECTION else 0)
        t_eff = sum_mv2 / (n_dof * BOLTZMANN_CONSTANT)
        print(f"\n--- Velocity sum rule ---")
        print(f"  integral of mass-weighted PSD  -> sum_i m_i <v_i^2> = {sum_mv2:.4e} J")
        print(f"  N_dof = {n_dof} (3N minus 3 for COM removal)" if COM_CORRECTION
              else f"  N_dof = {n_dof} (3N)")
        print(f"  effective temperature T_eff    = {t_eff:.1f} K   (MD input: {temperature:.1f} K)")
        if temperature > 0 and abs(t_eff - temperature) / temperature > 0.30:
            print(f"  WARNING: T_eff deviates from the stated 'temperature' parameter by "
                  f"{100.0 * abs(t_eff - temperature) / temperature:.0f}%.")
            print(f"  T_eff is measured from the trajectory itself, so the most likely")
            print(f"  explanation is that the 'temperature' parameter ({temperature:.1f} K) does not")
            print(f"  match this simulation. Set temperature = {t_eff:.0f} if that is the")
            print(f"  thermostat value -- it is used by the quantum correction factor.")
            print("  Otherwise check delta_t, nmeasure, trajectory units, constraints")
            print("  (rigid bonds reduce N_dof), or equilibration.")

    if quantum_correction:
        print("\nApplying quantum kinetic-energy weighting (x/2)*coth(x/2)...")
        print("  NOTE: the output is then a quantum kinetic-energy-weighted vibrational")
        print("        spectrum, NOT a density of states.")
        intensity = intensity * qcf_kinetic_energy(wavenumber, temperature)
        spectrum_kind = "quantum kinetic-energy weighted (mass-weighted PSD x (x/2)coth(x/2))"
    else:
        print("\nNo quantum correction applied (classical mass-weighted VDOS).")
        spectrum_kind = "raw classical mass-weighted VDOS (no quantum correction)"

    #  metadata 
    meta = [
        "generated by VCL-toolkit (vdos.py)",
        f"date                    : {datetime.now().isoformat(timespec='seconds')}",
        f"input trajectory        : {input_file}",
        f"mode                    : {mode}",
        f"spectrum content        : {spectrum_kind}",
        f"spectral estimator      : {spectral_estimator}",
        f"velocity source         : {velocity_source}",
        f"effective timestep      : {delta_t * nmeasure:.6f} fs "
        f"(delta_t={delta_t} fs x nmeasure={nmeasure})",
        f"frames used             : {nsteps} ({nsteps * delta_t * nmeasure / 1000.0:.4f} ps), nskip={nskip}",
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
        + (f" ((x/2)coth(x/2), T={temperature} K)" if quantum_correction else ""),
        f"temperature             : {temperature} K",
        f"mass weighting          : {'yes (sum_i m_i S_i)' if mode == 'full' else 'n/a (bond mode)'}",
        f"COM correction          : {COM_CORRECTION}",
        f"PBC unwrapping          : {pbc_status}",
        f"masses                  : {'; '.join(f'{k} {v}' for k, v in sorted(atomic_masses.items()))}",
    ]
    if t_eff is not None:
        meta.append(f"sum-rule T_eff          : {t_eff:.1f} K (target {temperature:.1f} K)")
    meta += [
        "",
        "The spectrum below is the RAW, UNFILTERED estimate over the full frequency",
        f"range. Display limits (plot_min_wavenumber={plot_min_wavenumber:g}, "
        f"plot_max_wavenumber={plot_max_wavenumber:g} cm-1) affect the plot only.",
        "Intensity is in arbitrary units.",
    ]

    #  Write raw (unfiltered) data to a new txt file
    with open(output_data, "w", encoding="utf-8") as f_out:
        for line in meta:
            f_out.write(f"# {line}\n" if line else "#\n")
        f_out.write("#\n")
        f_out.write("# Wavenumber(cm-1)\tIntensity(a.u.)\n")
        f_out.write("#\n")
        for x, y in zip(wavenumber, intensity):
            f_out.write(f"{x:.6f}\t{y:.8e}\n")

    print(f"\n(*^_^*)/ VDOS data saved to: {output_data}\n")
    print("Note: the saved data is the RAW spectrum over the full frequency range;")
    print(f"      plot_min_wavenumber/plot_max_wavenumber affect the plot only.")
    print("Units are cm-1 for wavenumber and arbitrary units for intensity\n")
    print("Generating plot... Please be patient.")

    plot_mask = (wavenumber >= plot_min_wavenumber) & (wavenumber <= plot_max_wavenumber)
    if not np.any(plot_mask):
        sys.exit(f"Error: no spectral points between plot_min_wavenumber="
                 f"{plot_min_wavenumber} and plot_max_wavenumber={plot_max_wavenumber} cm-1.")
    wavenumber_plot = wavenumber[plot_mask]
    intensity_plot = intensity[plot_mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(wavenumber_plot, intensity_plot, linewidth=2.0)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=16)
    ax.set_ylabel("Intensity (a.u.)", fontsize=16)
    ax.set_title("Quantum KE-Weighted Vibrational Spectrum" if quantum_correction
                 else "Vibrational Density of States (VDOS)", fontsize=16, fontweight="bold")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.offsetText.set_fontsize(16)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=plot_max_wavenumber)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_png, **save_kwargs)
    plt.close(fig)
    
    print(rf" *\(^o^)/* VDOS figure saved as: {output_png}\n")

if __name__ == "__main__":
    import argparse
    os.environ["PYTHONUTF8"] = "1"
    parser = argparse.ArgumentParser(description='Calculate Vibrational Density of States (VDOS) from trajectory files.')
    parser.add_argument('input_file', type=str, help='Input file containing VDOS parameters (input_vdos.txt)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        sys.exit(f"Error: Input parameter file '{args.input_file}' not found.")
    
    run_vdos_script(args.input_file)