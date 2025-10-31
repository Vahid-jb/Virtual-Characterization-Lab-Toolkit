#!/usr/bin/env python3
"""
lammps_postprocessor.py
An interactive script to post-process LAMMPS output files for XRD, VDOS, and SAED visualization.
"""

import subprocess
import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import signal
from scipy.io import netcdf_file
from datetime import datetime

# --- Helper Functions (Consolidated) ---
def normalize_path_for_os(path: str) -> str:
    """Normalizes a file path for Windows or Linux, supporting WSL paths."""
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

def ask(prompt: str, default: str = None) -> str:
    """Gets user input with an optional default value, preventing duplicate prompts."""
    if default is not None:
        s = input(f"{prompt} [{default}]: ").strip()
        return s if s else default
    while True:
        s = input(f"{prompt}: ").strip()
        if s:
            return s
        print("This field is required.")

def run_command(cmd):
    """Run a command and stream stdout/stderr in real time."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}")

def find_executable(name):
    """Try to locate an executable as 'name' or 'name.exe'."""
    for exe in [name, f"{name}.exe"]:
        if shutil.which(exe) is not None:
            return exe
    raise EnvironmentError(f"Executable '{name}' not found (tried '{name}' and '{name}.exe').")

# --- XRD Visualization Script ---
def run_xrd_script():
    """Main function for the XRD visualization."""
    data_path_raw = ask("Please enter name of xrd file in the current directory")
    if not data_path_raw:
        sys.exit("[ERROR] No file path provided.")
    data_path = normalize_path_for_os(data_path_raw)
    if not os.path.exists(data_path):
        sys.exit(f"[ERROR] File not found: {data_path}")

    base = os.path.splitext(os.path.basename(data_path))[0]
    default_png = f"Deg2Theta.png"
    output_data_filename = ask("Please enter output data filename", "XRD.txt")
    output_data_path = os.path.join(os.path.dirname(data_path), output_data_filename)
    
    output_info = ask(
        "Please enter output file name and its format (e.g. Deg2Theta.png). Options: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, webp",
        default_png
    )
    # Split the filename and format
    output_name, output_format = os.path.splitext(output_info)
    output_format = output_format.strip('.').lower()
    if not output_format:
        output_format = "png"

    # Ask for quality/resolution for all formats
    if output_format == "pdf":
        # For PDF, we use metadata to control quality
        save_dpi = ask("Please set resolution (DPI) for embedded raster elements", "300")
        save_kwargs = {'dpi': float(save_dpi), 'metadata': {
            'CreationDate': datetime.now(),
            'Creator': 'LAMMPS Postprocessor',
            'Producer': 'Matplotlib'
        }}
    elif output_format in ['eps', 'ps']:
        # For EPS/PS, DPI affects the resolution of embedded raster elements
        save_dpi = ask("Please set resolution (DPI) for embedded raster elements", "300")
        save_kwargs = {'dpi': float(save_dpi)}
    elif output_format in ['svg', 'svgz']:
        # For SVG, DPI affects how coordinates are calculated
        save_dpi = ask("Please set resolution (DPI) for coordinate calculation", "96")
        save_kwargs = {'dpi': float(save_dpi)}
    else:
        # Raster formats
        save_dpi = ask("Please set resolution (DPI)", "150")
        save_kwargs = {'dpi': float(save_dpi)}

    out_path = os.path.join(os.path.dirname(data_path), output_name + "." + output_format)

    choice = ask("Plot intensity as 'Count' or 'Count/Total'", "Count").lower()
    mode = "count" if "count/total" not in choice else "norm"

    print("Please be patient, the plot may take a while to generate.")

    # --- Parse file ---
    x_vals, y_vals = [], []
    with open(data_path, "r") as f:
        lines = f.readlines()
    
        # Skip the first 4 header lines
        for line in lines[4:]:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue
            
            parts = stripped_line.split()
            if len(parts) < 4:
                continue
            try:
                # The data columns are: Bin, 2theta, Count, Count/Total
                two_theta = float(parts[1])
                count = float(parts[2])
                count_total = float(parts[3])
            except (ValueError, IndexError):
                # Skip lines that can't be parsed correctly
                continue
            
            x_vals.append(two_theta)
            y_vals.append(count if mode == "count" else count_total)

    if not x_vals:
        sys.exit("[ERROR] No data parsed from file.")
        
    # --- Write data to a new txt file ---
    with open(output_data_path, "w") as f_out:
        f_out.write("# 2theta (degree) vs. Intensity (a.u.)\n")
        f_out.write("#\n")
        f_out.write("# generated by VCL-toolkit\n")
        f_out.write("#\n")
        for x, y in zip(x_vals, y_vals):
            f_out.write(f"{x}\t{y}\n")
    print(f"(*^_^*)/ XRD data saved to: {output_data_path}")

    # --- Plotting all data points ---
    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(x_vals, y_vals, linewidth=1.5)

    ax.set_xlabel(r"2$\theta$ (degree)", fontsize=14)
    ax.set_ylabel("Intensity (a.u.)", fontsize=14)

    title_suffix = "Count" if mode == "count" else "Count/Total"
    ax.set_title(f"LAMMPS XRD Pattern ({title_suffix})", fontsize=14, fontweight="bold")

    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.offsetText.set_fontsize(14)
    ax.set_ylim(bottom=0)
      
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    print(rf" *\(^o^)/* xrd figure saved: {out_path}")

# --- VDOS Script ---
def run_vdos_script():
    """Main function for the VDOS calculation and visualization."""
    def print_vdos_welcome_message():
        print("""
    ************************************************************
    * This tool calculates the Vibrational Density of States   *
    * (VDOS) from molecular dynamics trajectory files.         *
    ************************************************************
    """)

    def check_libraries_and_file(input_name):
        """Checks for necessary libraries and file existence."""
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

    def choose_window(nsteps, window_kind):
        if window_kind == "Gaussian":
            sigma = 2 * math.sqrt(2 * math.log(2))
            std = 4000.0
            window_function = signal.windows.gaussian(nsteps, std / sigma, sym=False)
        elif window_kind == "Blackman-Harris":
            window_function = signal.windows.blackmanharris(nsteps, sym=False)
        elif window_kind == "Hamming":
            window_function = signal.windows.hamming(nsteps, sym=False)
        elif window_kind == "Hann":
            window_function = signal.windows.hann(nsteps, sym=False)
        return window_function

    def calc_FFT(array_1D, window):
        WE = sum(window) / len(array_1D)
        wf = window / WE
        sig = array_1D * wf
        N = zero_padding(sig)
        yfft = np.fft.fft(sig, N, axis=0) / len(sig)
        return np.square(np.absolute(yfft))

    def zero_padding(sample_data):
        return int(2 ** math.ceil(math.log(len(sample_data), 2)))

    def calc_ACF(array_1D):
        yunbiased = array_1D - np.mean(array_1D, axis=0)
        ynorm = np.sum(np.power(yunbiased, 2), axis=0)
        autocor = signal.fftconvolve(array_1D,
                                     array_1D[::-1],
                                     mode='full')[len(array_1D) - 1:] / ynorm
        return autocor

    import math
    print_vdos_welcome_message()
    
    input_file_raw = ask("Please enter the name of input file in the current directory")
    
    output_data_raw = ask("Please enter output data filename", "VDOS.txt")
    
    output_info = ask(
        "Please enter output file name and its format (e.g. VDOS.png). Options: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, webp",
        "VDOS.png"
    )
    output_name, output_format = os.path.splitext(output_info)
    output_format = output_format.strip('.').lower()
    if not output_format:
        output_format = "png"
    
    # Ask for quality/resolution for all formats
    if output_format == "pdf":
        # For PDF, we use metadata to control quality
        save_dpi = ask("Please set resolution (DPI) for embedded raster elements", "300")
        save_kwargs = {'dpi': float(save_dpi), 'metadata': {
            'CreationDate': datetime.now(),
            'Creator': 'LAMMPS Postprocessor',
            'Producer': 'Matplotlib'
        }}
    elif output_format in ['eps', 'ps']:
        # For EPS/PS, DPI affects the resolution of embedded raster elements
        save_dpi = ask("Please set resolution (DPI) for embedded raster elements", "300")
        save_kwargs = {'dpi': float(save_dpi)}
    elif output_format in ['svg', 'svgz']:
        # For SVG, DPI affects how coordinates are calculated
        save_dpi = ask("Please set resolution (DPI) for coordinate calculation", "96")
        save_kwargs = {'dpi': float(save_dpi)}
    else:
        # Raster formats
        save_dpi = ask("Please set resolution (DPI)", "150")
        save_kwargs = {'dpi': float(save_dpi)}

    output_png_raw = output_name + "." + output_format
    
    input_file = normalize_path_for_os(input_file_raw)
    output_data = normalize_path_for_os(output_data_raw)
    output_png = normalize_path_for_os(output_png_raw)

    check_libraries_and_file(input_file)
    
    mode = ask("Please enter mode of operation ('full' or 'bond')", "full").lower()
    if mode not in ['full', 'bond']:
        sys.exit("Error: Mode must be 'full' or 'bond'.")
        
    delta_t = float(ask("Please enter delta time in femtoseconds (-dt/--delta_t)"))
    delta_t_s = delta_t * 1e-15
    
    bond_indices = None
    if mode == 'bond':
        bond_str = ask("Please enter bond indices (e.e., '1 2')", "").split()
        if len(bond_str) != 2:
            sys.exit("Error: For 'bond' mode, two integer indices are required.")
        bond_indices = [int(i) - 1 for i in bond_str]

    window_kind = ask("Please enter window kind for FFT ('Gaussian', 'Blackman-Harris', 'Hamming', 'Hann')", "Gaussian").capitalize()
    if window_kind not in ['Gaussian', 'Blackman-Harris', 'Hamming', 'Hann']:
        sys.exit("Error: Window kind for FFT must be 'Gaussian', 'Blackman-Harris', 'Hamming', or 'Hann'.")
        
    force_numerical = ask("Would you like to force numerical calculation of velocities? ('True' or 'False')", "False").lower() == "true"
    use_normalized_vectors = ask("Would you like to use the norm of coordinates/velocities? ('True' or 'False')", "False").lower() == "true"

    contains_velocities = False
    if input_file.lower().endswith('.xyz'):
        from ase.io import read
        print("\nReading XYZ file using ASE library...")
        trajectory = read(input_file, index=':')
        nsteps = len(trajectory)
        natoms = len(trajectory[0])
        coordinates = np.empty((nsteps, natoms, 3))
        for i, frame in enumerate(trajectory):
            coordinates[i] = frame.get_positions()
        
        if mode == "full":
            print("\nCalculating VDOS for all atoms.")
            print("Velocities will be calculated numerically.")
            if use_normalized_vectors:
                normal_vectors = np.linalg.norm(coordinates, axis=-1)
            else:
                yfft = np.zeros(0)
        else:
            print("\nCalculating VDOS for the specified bond.")
            print("Derivatives will be calculated numerically.")
            distances = np.linalg.norm((coordinates[:, bond_indices[0], :] - coordinates[:, bond_indices[1], :]), axis=1)
    else:
        print("\nReading NETCDF file using SciPy library...")
        try:
            trajectory = netcdf_file(input_file, 'r')
        except Exception as e:
            sys.exit(f"Error opening NetCDF file: {e}")
        
        if mode == "full":
            print("\nCalculating VDOS for all atoms.")
            contains_velocities = "velocities" in trajectory.variables
            if contains_velocities and not force_numerical:
                print("Velocities will be read from the trajectory file.")
                velocities = np.array(trajectory.variables['velocities'].data)
                nsteps = len(velocities)
                natoms = len(velocities[0])
                if use_normalized_vectors:
                    normal_vectors = np.linalg.norm(velocities, axis=-1)
                else:
                    yfft = np.zeros(0)
            else:
                if contains_velocities and force_numerical:
                    print("Found velocities but numerical calculation is forced.")
                print("Velocities will be calculated numerically.")
                coordinates = np.array(trajectory.variables['coordinates'].data)
                nsteps = len(coordinates)
                natoms = len(coordinates[0])
                if use_normalized_vectors:
                    normal_vectors = np.linalg.norm(coordinates, axis=-1)
                else:
                    yfft = np.zeros(0)
        else:
            print("\nCalculating VDOS for the specified bond.")
            print("Derivatives will be calculated numerically.")
            coordinates = np.array(trajectory.variables['coordinates'].data)
            nsteps = len(coordinates)
            distances = np.linalg.norm((coordinates[:, bond_indices[0], :] - coordinates[:, bond_indices[1], :]), axis=1)

    window = choose_window(nsteps, window_kind)
    
    if mode == "full":
        yfft = np.zeros(1)
        if use_normalized_vectors:
            for i in range(natoms):
                if contains_velocities and not force_numerical:
                    atom_velocities = normal_vectors[:, i]
                else:
                    atom_velocities = calc_derivative(normal_vectors[:, i], delta_t_s)
                ACF = calc_ACF(atom_velocities)
                yfft_i = calc_FFT(ACF, window)
                if i == 0:
                    yfft = yfft_i
                else:
                    yfft += yfft_i
        else:
            for i in range(natoms):
                for j in range(0, 3):
                    if contains_velocities and not force_numerical:
                        atom_velocities = velocities[:, i, j]
                    else:
                        atom_velocities = calc_derivative(coordinates[:, i, j], delta_t_s)
                    ACF = calc_ACF(atom_velocities)
                    yfft_i = calc_FFT(ACF, window)
                    if i == 0 and j == 0:
                        yfft = yfft_i
                    else:
                        yfft += yfft_i
    else:
        distances_velocities = calc_derivative(distances, delta_t_s)
        ACF = calc_ACF(distances_velocities)
        yfft = calc_FFT(ACF, window)

    wavenumber = np.fft.fftfreq(len(yfft), delta_t_s * 2.9979245899e10)[0:int(len(yfft) / 2)]
    intensity = yfft[0:int(len(yfft) / 2)]
       
    
    # --- Write data to a new txt file ---
    with open(output_data, "w") as f_out:
        f_out.write("# generated by VCL-toolkit\n")
        f_out.write("#\n")
        f_out.write("# Wavenumber(cm-1)   # Intensity (a. u.)\n")
        f_out.write("#\n")
        for x, y in zip(wavenumber, intensity):
            f_out.write(f"{x}\t{y}\n")
    print(f"\n(*^_^*)/ VDOS data saved to: {output_data}\n")
    print("Units are cm-1 for wavenumber and arbitrary units for intensity\n")
    print("Generating plot... Please be patient.")
    print("Note: The data point at Wavenumber = 0 is neglected in the plot.")


    wavenumber_plot = wavenumber[1:]
    intensity_plot = intensity[1:]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(wavenumber_plot, intensity_plot, linewidth=2.0)
    
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=16)
    ax.set_ylabel("Intensity (a.u.)", fontsize=16)
    ax.set_title("Vibrational Density of States (VDOS)", fontsize=16, fontweight="bold")
    
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.offsetText.set_fontsize(16)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=4500)
    
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    
    fig.savefig(output_png, **save_kwargs)
    plt.close(fig)
    print(rf" *\(^o^)/* VDOS figure saved as: {output_png}\n")




# --- SAED Script ---
def run_saed_script():
    """Interactively configures and runs the SAED visualization script via VisIt."""
    temp_script_name = "modified_saed_temp.py"
    
    try:
        visit_exe = find_executable("visit")
        print(f"[INFO] Using VisIt executable: {visit_exe}")
    except EnvironmentError as e:
        print(f"[ERROR] {e}")
        return

    print("\n--- SAED Visualization Setup ---")
    print("The SAED data visualization is achieved by Visit software 'https://visit-dav.github.io/visit-website/index.html'.")

    # Interactive Inputs
    vtk_file_raw = ask("Please enter name of vtk file in the current directory")
    
    output_info = ask(
        "Please enter output file name and its format (e.g. saed.png). Options: BMP, JPEG, PNG, PPM, Postscript, RGB, TIFF",
        "saed.png"
    )
    # Split the filename and format
    output_name, output_format = os.path.splitext(output_info)
    output_format = output_format.strip('.').upper()
    if not output_format:
        output_format = "PNG"

    lbound = ask("Please set lower bound for Isovolume operator", "0")
    ubound = ask("Please set upper bound for Isovolume operator", "1e+37")
    
    pc_min = ask("Please set minimum value for Pseudocolor plot", "1")

    ss_origin_str = ask("Please enter spherical slice origin, sphere centered at R_Ewald=(1/lambda) along the chosen zone axis (e.g. along x axis, '39.84063, 0, 0' for lambda = 0.0251)", "39.84063, 0, 0")
    ss_radius = ask("Please enter spherical slice radius (e.g., '39.84063' for lambda = 0.0251)", "39.84063")

    v_viewNormal_str = ask("Please enter view normal vector (e.g., '-1, 0, 0')", "-1, 0, 0")
    v_viewUp_str = ask("Please enter view up vector (e.g., '0, 1, 0')", "0, 1, 0")
    
    save_width = ask("Please set resolution (witdth=Height), width in pixels", "1200")
    
    # Interactive annotation settings
    print("\n--- Annotation Settings ---")
    show_axes3d = ask("show 3D axes? ('True' or 'False')", "False").lower() == "true"
    show_axes2d = ask("show 2D axes? ('True' or 'False')", "False").lower() == "true"
    show_user_info = ask("show user info? ('True' or 'False')", "False").lower() == "true"
    show_database_info = ask("show database info? ('True' or 'False')", "False").lower() == "true"
    show_legend = ask("show legend/colorbar? ('True' or 'False')", "True").lower() == "true"

    # Map boolean to VisIt's 0/1 values
    axes3d_visible = 1 if show_axes3d else 0
    axes2d_visible = 1 if show_axes2d else 0
    user_info_flag = 1 if show_user_info else 0
    database_info_flag = 1 if show_database_info else 0
    legend_info_flag = 1 if show_legend else 0
    
    # Generate the script string
    script_content = f"""# Dynamically generated VisIt script

import os
import sys

# --- Helper to fix WSL paths ---
def normalize_path(path):
    if path.startswith("/mnt/"):
        drive = path[5]
        rest = path[7:]
        rest = rest.replace("/", "\\\\")
        return drive.upper() + ":\\\\" + rest
    return path

# --- Check file existence ---
vtk_file = normalize_path("{vtk_file_raw}")
if not os.path.exists(vtk_file):
    print(f"[ERROR] VTK file not found: {{vtk_file}}")
    sys.exit(1)

output_name = "{output_name}"

# === Load data ===
OpenDatabase(vtk_file)

# Pseudocolor plot of intensity
AddPlot("Pseudocolor", "intensity")

# Isovolume operator (remove ghost data)
AddOperator("Isovolume")
iso = IsovolumeAttributes()
iso.lbound = {lbound}
iso.ubound = {ubound}
iso.variable = "intensity"
SetOperatorOptions(iso)

# Pseudocolor settings (log scale, min=1, keep legend ON)
pc = PseudocolorAttributes()
pc.scaling = pc.Log
pc.minFlag = 1
pc.min = {pc_min}
pc.maxFlag = 0
pc.legendFlag = 1
SetPlotOptions(pc)

# Spherical slice operator
AddOperator("SphereSlice")
ss = SphereSliceAttributes()
ss.origin = ({ss_origin_str})
ss.radius = {ss_radius}
SetOperatorOptions(ss)

DrawPlots()

# --- Auto-fit the data, then adjust view ---
ResetView()
v = GetView3D()
v.viewNormal = ({v_viewNormal_str})
v.viewUp = ({v_viewUp_str})
v.perspective = 0
SetView3D(v)

# --- Annotations ---
a = AnnotationAttributes()
a.axes3D.visible = {axes3d_visible}
a.axes2D.visible = {axes2d_visible}
a.userInfoFlag = {user_info_flag}
a.databaseInfoFlag = {database_info_flag}
a.legendInfoFlag = {legend_info_flag}
SetAnnotationAttributes(a)

# --- Save final image ---
SaveWindowAtts = SaveWindowAttributes()
SaveWindowAtts.outputToCurrentDirectory = 1
SaveWindowAtts.fileName = output_name
SaveWindowAtts.family = 0
SaveWindowAtts.format = SaveWindowAtts.{output_format}
SaveWindowAtts.width = {save_width}
SaveWindowAtts.height = {save_width}
SetSaveWindowAttributes(SaveWindowAtts)
SaveWindow()

print(f"[INFO] Saved diffraction pattern as {{output_name}}")
sys.exit(0)
"""
    # Write the script to a temporary file
    try:
        with open(temp_script_name, "w") as f:
            f.write(script_content)

        print(f"\n[INFO] Generated temporary script '{temp_script_name}'.")
        print("[INFO] Running VisIt... Please wait, this may take a moment.")
    
        # Run the VisIt command
        cmd = [visit_exe, "-nowin", "-cli", "-s", temp_script_name]
        run_command(cmd)
        print(r"\n *\(^o^)/* VisIt execution completed and saed figure saved.")
    except Exception as e:
        print(f"\n❌ An error occurred while running VisIt: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_script_name):
            os.remove(temp_script_name)
            print(f"[INFO] Cleaned up temporary script '{temp_script_name}'.")

# --- Main Program Logic ---
def run_post_processing_app(): # Renamed from main_menu
    """Main menu of the script."""
    print("************************************************")
    print("Welcome to lammps outputs postprocessing (:")
    print("************************************************")
    
    first_run = True
    while True:
        if first_run:
            print("\nPlease choose a visualization task.")
            first_run = False
        else:
            print("\n--- Next Step ---")
            print("Now, you can choose another visualization task.")
        
        print("0. Back to main menu") # Exit option to return to main workflow
        print("1. XRD visualization")
        print("2. Vibrational density of state (VDOS) visualization")
        print("3. SAED visualization")

        next_step = ask("Please select an option (0, 1, 2, or 3)", "1")

        if next_step == '0':
            print("Exiting Post-processor. Returning to main workflow. (:")
            # Don't call sys.exit() here when part of a larger workflow
            return # Exit this function to return to main_workflow.py
        elif next_step == '1':
            run_xrd_script()
        elif next_step == '2':
            print("\nThe VDOS extracted from trajectory output file produced by lammps using script available in: https://github.com/JonathanSemelak/VDOS")
            run_vdos_script()
        elif next_step == '3':
            run_saed_script()
        else:
            print("[ERROR] Invalid option. Please enter a valid number.")
            # For standalone, we might exit, but in workflow, we just loop again or return
            # For now, let's just print error and let the loop continue
            # sys.exit(0) # Removed to keep workflow running
            
# Entry point of the script
if __name__ == "__main__":
    run_post_processing_app() # Call the new function