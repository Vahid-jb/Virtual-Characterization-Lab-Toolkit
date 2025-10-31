#!/usr/bin/env python3
"""
run_lammps.py
Interactive script to run a LAMMPS simulation.
"""

import subprocess
import os
import sys
import shutil

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

def open_file_in_editor(filepath: str):
    """
    Opens a file in a text editor based on the operating system.
    For Windows, it attempts to open with notepad.exe in a new window.
    For macOS, it uses 'open -t' to open with the default text editor.
    For Linux/WSL, it prioritizes 'nano' (blocking current terminal)
    then 'xdg-open' for graphical editors, then other CLI editors.
    Blocks execution until the editor is closed.
    """
    editor_cmd = []

    if sys.platform == "win32":
        # On Windows, use 'start' to open notepad.exe in a new window/process
        # shell=True is needed for 'start' command
        full_cmd = ["start", "notepad.exe", filepath]
        try:
            subprocess.run(full_cmd, shell=True, check=True)
            return # Exit function if Windows command succeeded
        except subprocess.CalledProcessError as e:
            print(f"❌ Error opening editor (Windows 'start' failed). Error code {e.returncode}.")
        except FileNotFoundError:
            print(f"❌ Error: 'notepad.exe' not found on Windows. Please ensure Notepad is available.")
        except Exception as e:
            print(f"❌ An unexpected error occurred on Windows: {e}")
        return # Always return after attempting Windows specific command
        
    elif sys.platform == "darwin":  # macOS
        editor_cmd = ["open", "-t", filepath]
    else:  # Linux, WSL, etc.
        # Prioritize nano as explicitly requested
        if shutil.which("nano"):
            editor_cmd = ["nano", filepath]
        elif shutil.which("xdg-open"): # For graphical editors
            editor_cmd = ["xdg-open", filepath]
        else:
            # Fallback to other common command-line editors
            for cli_editor in ["vi", "vim", "code"]:
                if shutil.which(cli_editor):
                    editor_cmd = [cli_editor, filepath]
                    break
            if not editor_cmd:
                raise EnvironmentError("No suitable text editor found (tried nano, xdg-open, vi, vim, code).")
    
    # For non-Windows (or if Windows-specific command failed/wasn't applicable), run directly
    try:
        # This blocks until the editor process closes
        subprocess.run(editor_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error opening editor. Editor exited with code {e.returncode}.")
        print(f"Error details: {e.stderr or e.stdout or 'No additional output.'}")
    except FileNotFoundError:
        print(f"❌ Error: Editor command '{editor_cmd[0]}' not found.")
        print("Please ensure a text editor is installed and its executable is in your system's PATH.")
    except Exception as e:
        print(f"❌ An unexpected error occurred while trying to open the editor: {e}")

def modify_and_run_lammps_file():
    """
    Handles the workflow for modifying and running a LAMMPS input file.
    """
    # Ask for LAMMPS input file
    in_file = ask("Please enter the name of LAMMPS input file in the current directory or full path to it (e.g. in.relax)")
    if not os.path.exists(in_file):
        print(f"❌ [ERROR] File not found: {in_file}")
        return # Return to main menu if file not found
        
    # Offer to edit the input file
    edit_choice = ask("Would you like to edit the input file? If yes then I will open it using 'GNU nano' for your convinience (y/n)", "n").lower()
    if edit_choice in ['y', 'yes']:
        print(" Modifying ...")
        open_file_in_editor(in_file)
        print("[OK] Your modified input is ready to run")

    # Ask for the full command line
    cmd_str = ask("Please provide your desired LAMMPS running command (e.g. /path/to/lmp_serial -in in.spin for serial mode)")
    cmd = cmd_str.split()

    # Run the command
    print("\n[INFO] Starting LAMMPS simulation...")
    try:
        run_command(cmd)
        print(r"\n *\(^o^)/* LAMMPS simulation completed.")
    except RuntimeError as e:
        print(f"\n❌ [ERROR] {e}")
        print("Please check the command and try again.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

def run_lammps_simulation_app(): # Renamed from main
    """
    Main entry point for the interactive LAMMPS runner script.
    Displays a menu and loops until the user chooses to exit.
    """
    print("************************************************")
    print("Welcome to LAMMPS Interactive Runner! (:")
    print("************************************************")

    while True:
        print("\n--- Main Menu ---")
        print("0. Back to main menu") # Exit option to return to main workflow
        print("1. Modifying and running a LAMMPS input file")
        
        choice = ask("Select an option", "1")

        if choice == '0':
            print("Exiting LAMMPS Runner. Returning to main workflow. (:")
            # Don't call sys.exit() here when part of a larger workflow
            return # Exit this function to return to main_workflow.py
        elif choice == '1':
            modify_and_run_lammps_file()
        else:
            print("[ERROR] Invalid option. Please enter '0' or '1'.")
            
# Entry point of the script
if __name__ == "__main__":
    run_lammps_simulation_app() # Call the new function