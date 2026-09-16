#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
CLI version of SAED visualization script that reads parameters from input file.
Converts VTK SAED output to regular figure format using VisIt software.
"""

import os
import sys
import subprocess
import argparse
from shutil import which
import io

if sys.platform == "win32":
    if sys.stdout is not None and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    if sys.stderr is not None and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import platform

import shutil

def normalize_path_for_os(path: str) -> str:
    if not path:
        return path
    if len(path) > 7 and path.startswith("/mnt/"):
        drive = path[5]
        rest = path[7:].replace("/", "\\")
        return drive.upper() + ":\\" + rest
    return path

def find_executable(name):
    for exe in [name, f"{name}.exe"]:
        if shutil.which(exe) is not None:
            return exe
    raise EnvironmentError(f"Executable '{name}' not found (tried '{name}' and '{name}.exe').")


def run_command(cmd):
    try:
        if os.name == 'nt' or sys.platform == 'win32':
            cflags = 0x08000000
            startupinfo = None
            if hasattr(subprocess, 'STARTUPINFO'):
                startupinfo = getattr(subprocess, 'STARTUPINFO')()
                startupinfo.dwFlags |= getattr(subprocess, 'STARTF_USESHOWWINDOW')
                startupinfo.wShowWindow = 0 
            
            result = subprocess.run(cmd, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr, 
                                    creationflags=cflags, startupinfo=startupinfo)
        else:
            result = subprocess.run(cmd, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with return code {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout.strip()}")
        if e.stderr:
            print(f"stderr: {e.stderr.strip()}")
        raise
def parse_input_file(input_file):
    params = {
        'visit_path': '', 
        'vtk_file': '',
        'output_info': 'saed.png',
        'lbound': '0',
        'ubound': '1e+37',
        'pc_min': '1',
        'pc_max': '',   
        'ss_origin_str': '39.84063, 0, 0',
        'ss_radius': '39.84063',
        'v_viewNormal_str': '-1, 0, 0',
        'v_viewUp_str': '0, 1, 0',
        'save_width': '1200',
        'show_axes3d': 'False',
        'show_axes2d': 'False',
        'show_user_info': 'False',
        'show_database_info': 'False',
        'show_legend': 'True'
    }
    
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file '{input_file}' not found!")
        sys.exit(1)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    param_map = {
                        'visit_path': 'visit_path',
                        'vtk_file': 'vtk_file',
                        'output_file': 'output_info',
                        'iso_lower': 'lbound',
                        'iso_upper': 'ubound',
                        'pseudocolor_min': 'pc_min',
                        'pseudocolor_max': 'pc_max',
                        'sphere_origin': 'ss_origin_str',
                        'sphere_radius': 'ss_radius',
                        'view_normal': 'v_viewNormal_str',
                        'view_up': 'v_viewUp_str',
                        'resolution': 'save_width',
                        'show_3d_axes': 'show_axes3d',
                        'show_2d_axes': 'show_axes2d',
                        'show_user_info': 'show_user_info',
                        'show_database_info': 'show_database_info',
                        'show_legend': 'show_legend'
                    }
                    
                    for input_key, internal_key in param_map.items():
                        if key == input_key:
                            params[internal_key] = value
                            break
    
        if not params['vtk_file']:
            print("[ERROR] vtk_file parameter is required in input file!")
            sys.exit(1)
            
        output_name, output_ext = os.path.splitext(params['output_info'])
        output_format = output_ext.strip('.').upper()
        if not output_format:
            output_format = "PNG"
            output_ext = ".png"
        # Map the file extension to VisIt's SaveWindow format name
        visit_formats = {'PNG': 'PNG', 'JPG': 'JPEG', 'JPEG': 'JPEG', 'TIF': 'TIFF', 'TIFF': 'TIFF',
                         'BMP': 'BMP', 'PPM': 'PPM', 'RGB': 'RGB', 'PS': 'POSTSCRIPT', 'POSTSCRIPT': 'POSTSCRIPT'}
        if output_format not in visit_formats:
            print(f"[ERROR] Unsupported output format '{output_ext}'. "
                  f"Supported: {', '.join('.' + k.lower() for k in visit_formats)}")
            sys.exit(1)
        params['output_name'] = output_name
        params['output_format'] = visit_formats[output_format]
        params['output_path'] = output_name + output_ext
        
        bool_params = ['show_axes3d', 'show_axes2d', 'show_user_info', 
                       'show_database_info', 'show_legend']
        for param in bool_params:
            params[param] = params[param].lower()
            
        if 'visit_path' in params:
            params['visit_path'] = params['visit_path'].strip().strip('"\'')
        
        return params
        
    except Exception as e:
        print(f"[ERROR] Failed to parse input file '{input_file}': {e}")
        sys.exit(1)

def run_saed_script(input_file="input_vtk_to_fig.txt"):
    temp_script_name = "modified_saed_temp.py"
    print("\n--- SAED Visualization (CLI Mode) ---")
    print("The SAED data visualization is achieved by Visit software 'https://visit-dav.github.io/visit-website/index.html'.")

    params = parse_input_file(input_file)
    
    visit_path_param = params.get('visit_path', '').strip()
    
    visit_exe = None
    
    if visit_path_param:
        normalized_path = normalize_path_for_os(visit_path_param)
        if os.path.exists(normalized_path) or shutil.which(normalized_path):
             visit_exe = normalized_path
        else:
             print(f"[WARNING] Provided visit_path '{normalized_path}' not found. Attempting auto-detection...")
    
    if not visit_exe:
        try:
            visit_exe = find_executable("visit")
        except EnvironmentError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    print(f"[INFO] Using VisIt executable: {visit_exe}")
    
    print(f"[INFO] Reading parameters from: {input_file}")
    print(f"[INFO] Processing VTK file: {params['vtk_file']}")
    print(f"[INFO] Output file: {params['output_info']}")
    
    # Map boolean to VisIt's 0/1 values
    axes3d_visible = 1 if params['show_axes3d'] == 'true' else 0
    axes2d_visible = 1 if params['show_axes2d'] == 'true' else 0
    user_info_flag = 1 if params['show_user_info'] == 'true' else 0
    database_info_flag = 1 if params['show_database_info'] == 'true' else 0
    legend_info_flag = 1 if params['show_legend'] == 'true' else 0
    _NL = chr(10)
    _pc_max = params.get('pc_max', '').strip()
    if _pc_max:
        _pc_max_lines = _NL.join(['pc.maxFlag = 1', 'pc.max = ' + _pc_max])
    else:
        _pc_max_lines = 'pc.maxFlag = 0'

    script_content = f"""# Dynamically generated VisIt script (CLI Mode)

import os
import sys

def normalize_path(path):
    if path.startswith("/mnt/"):
        drive = path[5]
        rest = path[7:]
        rest = rest.replace("/", "\\\\")
        return drive.upper() + ":\\\\" + rest
    return path

vtk_file = normalize_path("{params['vtk_file']}")
if not os.path.exists(vtk_file):
    print(f"[ERROR] VTK file not found: {{vtk_file}}")
    sys.exit(1)

output_name = "{params['output_name']}"

# === Load data ===
# On Windows, VisIt mistakenly thinks 'C:/...' means host 'C' and fails to connect.
# We prepend 'localhost:' to force it to use the local machine.
OpenDatabase("localhost:" + vtk_file)

# Pseudocolor plot of intensity
AddPlot("Pseudocolor", "intensity")

# Isovolume operator (remove ghost data)
AddOperator("Isovolume")
iso = IsovolumeAttributes()
iso.lbound = {params['lbound']}
iso.ubound = {params['ubound']}
iso.variable = "intensity"
SetOperatorOptions(iso)

# Pseudocolor settings (log scale, min=1, keep legend ON)
pc = PseudocolorAttributes()
pc.scaling = pc.Log
pc.minFlag = 1
pc.min = {params['pc_min']}
{_pc_max_lines}
pc.legendFlag = 1
SetPlotOptions(pc)

# Spherical slice operator
AddOperator("SphereSlice")
ss = SphereSliceAttributes()
ss.origin = ({params['ss_origin_str']})
ss.radius = {params['ss_radius']}
SetOperatorOptions(ss)

DrawPlots()

# --- Auto-fit the data, then adjust view ---
ResetView()
v = GetView3D()
v.viewNormal = ({params['v_viewNormal_str']})
v.viewUp = ({params['v_viewUp_str']})
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
SaveWindowAtts.format = SaveWindowAtts.{params['output_format']}
SaveWindowAtts.width = {params['save_width']}
SaveWindowAtts.height = {params['save_width']}
SetSaveWindowAttributes(SaveWindowAtts)
saved = SaveWindow()

# VisIt appends its own extension (.jpeg, .tif): rename to the requested file name
output_path = {params['output_path']!r}
if saved and os.path.exists(saved) and os.path.abspath(saved) != os.path.abspath(output_path):
    os.replace(saved, output_path)

print(f"[INFO] Saved diffraction pattern as {{output_path}}")
sys.exit(0)
"""
    try:
        with open(temp_script_name, "w", encoding="utf-8") as f:
            f.write(script_content)

        print(f"\n[INFO] Generated temporary script '{temp_script_name}'.")
        print("[INFO] Running VisIt... Please wait, this may take a moment.")
    
        # Run the VisIt command
        if os.name == 'nt' or sys.platform == 'win32':
            visit_dir = os.path.dirname(visit_exe)
            cli_exe = os.path.join(visit_dir, "cli.exe")
            if os.path.exists(cli_exe):
                cmd = [cli_exe, "-nowin", "-s", temp_script_name]
                
                env = os.environ.copy()
                env["PYTHONHOME"] = os.path.join(visit_dir, "lib", "python")
                env["PYTHONPATH"] = os.path.join(visit_dir, "lib")
                env["VISITHOME"] = visit_dir
                env["VISITLOC"] = visit_dir
                env["VISITPLUGINDIR"] = f"{os.path.join(os.environ.get('USERPROFILE', ''), 'Documents', 'VisIt')};{visit_dir}"
                env["VISITSSH"] = os.path.join(visit_dir, "qtssh.exe")
                env["VISITSSHARGS"] = "-no-antispoof"
                env["VISITULTRAHOME"] = os.path.join(visit_dir, "ultrawrapper")
                env["VISITUSERHOME"] = os.path.join(os.environ.get('USERPROFILE', ''), 'Documents', 'VisIt')
                env["PATH"] = f"{visit_dir};{env.get('PATH', '')}"

                old_env = os.environ.copy()
                os.environ.update(env)
                try:
                    run_command(cmd)
                finally:
                    os.environ.clear()
                    os.environ.update(old_env)
            else:
                cmd = [visit_exe, "-nowin", "-cli", "-s", temp_script_name]
                run_command(cmd)
        else:
            cmd = [visit_exe, "-nowin", "-cli", "-s", temp_script_name]
            run_command(cmd)
            
        print(r"\n *\(^o^)/* VisIt execution completed and saed figure saved.")
    except Exception as e:
        print(f"\n❌ An error occurred while running VisIt: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(temp_script_name):
            os.remove(temp_script_name)
            print(f"[INFO] Cleaned up temporary script '{temp_script_name}'.")

def main():
    parser = argparse.ArgumentParser(description='Convert VTK SAED output to figure format using VisIt')
    parser.add_argument('--input', type=str, default='input_vtk_to_fig.txt',
                        help='Input parameter file (default: input_vtk_to_fig.txt)')
    
    args = parser.parse_args()
    
    print("SAED VTK to Figure Converter")
    print("============================")
    print(f"Input parameter file: {args.input}")
    print()
    
    run_saed_script(args.input)

if __name__ == "__main__":
    main()