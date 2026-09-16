<img width="1735" height="300" alt="grafik" src="https://github.com/user-attachments/assets/0b1af335-aadf-4e56-8340-f58724af26cf" />


<img width="4582" height="3123" alt="Intro_Fig-3" src="https://github.com/user-attachments/assets/752927c3-3dd5-438f-b9b8-14bcd12c3392" />

# Virtual Characterization Lab (VCL) Toolkit

The Virtual Characterization Lab (VCL) is a unified, open-source toolkit designed to bridge the gap between computational models and experimental characterization. The VCL streamlines pre- and post-processing workflows, primarily optimized for—but not limited to—molecular dynamics (MD) simulations. It enables researchers to perform crystallographic analysis and generate essential characterization data, including X-ray Diffraction (XRD), Selected Area Electron Diffraction (SAED), Vibrational Density of States (VDOS), and Infrared (IR) spectra, all within a single, intuitive graphical user interface (GUI).

By integrating every step from initial structure preparation to final data visualization, the VCL accelerates materials discovery by making virtual characterization more efficient, accessible, and directly comparable to experimental results. The pre- and post-processing modules support classical MD, reactive MD (RMD), and Spin-Lattice Dynamics (SLD). The architecture allows each computational module to be executed independently via a command-line interface (CLI) or through the centralized GUI. Additionally, a local HTTP server is integrated to serve static HTML documentation directly to the user's browser for offline reference.

## Getting Started

Download the archive for your platform — Windows or Ubuntu/Linux — from the [Releases](https://github.com/Vahid-jb/Virtual-Characterization-Lab-Toolkit/releases) page, , extract it, and launch the `vcl_gui` executable directly. No build step and no Python environment required.

### GUI and Command-Line Use

VCL supports both graphical user interface (GUI) and command-line interface (CLI) workflows.

For most users, we recommend using the provided binaries through the GUI. After selecting the desired module, setting the parameters, and clicking Run, VCL automatically saves the corresponding input file in `.txt` format in the selected working directory.

Each module can also be executed independently from the command line using its saved input file. For example, in the Linux version:

```bash
XRD-ReciprocalSum input.txt
```
Users may also create or edit input files manually when required. The CLI workflow is particularly useful for high-throughput calculations, constructing automated workflows, and running VCL modules on computing clusters.

## Documentation

Open the in-app help (the **Docs (port 8000)** button in any module) for the bundled module
guides, or browse [docs/modules](sources/docs_site/docs/index.md).
