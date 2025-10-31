⚛️ Virtual Characterization Lab (VCL) Toolkit

The Virtual Characterization Lab (VCL) Toolkit is an open-source, unified, CLI-based Python framework designed to introduce a new paradigm in how materials researchers process and analyze molecular dynamics (MD) simulation data. Its core function is to generate outputs that are directly comparable side-by-side with experimental characterization results.

The VCL streamlines the entire workflow—from initial structure file to final, publication-ready visualization—all within a single environment.

✨ Key Features & Capabilities

📊 Direct Experimental Comparison: Virtually generate crucial characterization data that mirrors physical lab results: X-ray Diffraction (XRD), Selected Area Electron Diffraction (SAED), and Vibrational Density of States (VDOS). 
🔬 Unified Workflow: Integrates Input File Generation, Calculation Execution, and Post-Processing from a single, intuitive command line environment. 
💻 Python-Native & Open-Source: Ensures cross-platform compatibility and easy integration with other data science tools.

🚀 Getting Started

To use the VCL Toolkit, create a dedicated folder, place all the Python scripts (VCL-toolkit.py, Input_Convertor.py, LAMMPS_Run.py, Post_Processing.py) inside it, and follow these steps:

⚙️ Requirements

    Python: Requires Python 3.6 or later (3.9 – 3.12 recommended).

    Python Libraries: Install the primary dependencies: pip install numpy matplotlib scipy ase

    External Programs: The workflow relies on these external programs being installed and accessible in your system PATH:

        LAMMPS: For running MD simulations. (Download: https://www.lammps.org/download.html)

        VisIt: For visualizing SAED patterns. (Download: https://visit-dav.github.io/visit-website/releases-as-tables/)

    Fonts (Linux/WSL): For Matplotlib to render plots with Arial (pre-installed on Windows): sudo apt install ttf-mscorefonts-installer

💡 Implementation (Typical Usage)

Navigate to your toolkit directory and launch the interactive environment: python VCL-toolkit.py

This command launches the toolkit's interactive CLI, which will guide you through the Input File Generation, LAMMPS Calculation, and Virtual Characterization steps.

📖 Documentation

Further detailed information is available in the Doc.pdf user manual.
