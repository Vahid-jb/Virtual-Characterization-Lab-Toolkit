
import subprocess
import sys
import os
import time 

# Helper function to run other Python scripts
def run_script_interactively(script_name: str, interactive: bool = True): 
    """
    Runs another Python script.
    If 'interactive' is True, it allows direct user interaction.
    """
    print(f"\n--- Starting {script_name.replace('_', ' ').replace('.py', '')} ---")
    process = subprocess.run(
        [sys.executable, script_name],
        stdin=sys.stdin,    
        stdout=sys.stdout,  
        stderr=sys.stderr,  
        text=True,
        check=False 
    )
    print(f"--- Finished {script_name.replace('_', ' ').replace('.py', '')} ---\n")
    if process.returncode != 0 and process.returncode != 1: 
        print(f"/!\\ {script_name} exited with code {process.returncode}. Please check its output for details.")
    return


def ask(prompt: str, default: str = None) -> str:
    if default is not None:
        s = input(f"{prompt} [{default}]: ").strip()
        return s if s else default
    while True:
        s = input(f"{prompt}: ").strip()
        if s:
            return s
        print("This field is required.")

def main():
    print("************************************************")
    print("Welcome to the Integrated LAMMPS Workflow! (:")
    print("************************************************")

    while True:
        print("\n--- Main Menu ---")
        print("0. Exit")
        print("1. Making input file (Input_Convertor)")
        print("2. Running calculation (LAMMPS_Run)")
        print("3. Post processing (Post_Processing)")

        choice = ask("Select an option", "1")

        if choice == '0':
            print("Exiting the Integrated LAMMPS Workflow. Goodbye! (;")
            sys.exit(0)
        elif choice == '1':
            run_script_interactively("Input_Convertor.py", interactive=True)
        elif choice == '2':
            run_script_interactively("LAMMPS_Run.py", interactive=True)
        elif choice == '3':
            run_script_interactively("Post_Processing.py", interactive=True)
        else:
            print("❌ Invalid option. Please enter a number between 0 and 3.")

if __name__ == "__main__":
    main()
