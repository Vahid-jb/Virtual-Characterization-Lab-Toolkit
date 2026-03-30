<img width="1589" height="167" alt="grafik" src="https://github.com/user-attachments/assets/b141b306-44c6-4514-ad2a-d89b56d27b46" />

<img width="4582" height="3123" alt="Intro_Fig-3" src="https://github.com/user-attachments/assets/752927c3-3dd5-438f-b9b8-14bcd12c3392" />

⚛️ Virtual Characterization Lab (VCL) Toolkit

The Virtual Characterization Lab (VCL) is a unified, open-source toolkit designed to bridge the gap between computational models and experimental characterization. The VCL streamlines pre- and post-processing workflows, primarily optimized for—but not limited to—molecular dynamics (MD) simulations. It enables researchers to perform crystallographic analysis and generate essential characterization data, including X-ray Diffraction (XRD), Selected Area Electron Diffraction (SAED), Vibrational Density of States (VDOS), and Infrared (IR) spectra, all within a single, intuitive graphical user interface (GUI).

By integrating every step from initial structure preparation to final data visualization, the VCL accelerates materials discovery by making virtual characterization more efficient, accessible, and directly comparable to experimental results. The pre- and post-processing modules support classical MD, reactive MD (RMD), and Spin-Lattice Dynamics (SLD). The architecture allows each computational module to be executed independently via a command-line interface (CLI) or through the centralized GUI. Additionally, a local HTTP server is integrated to serve static HTML documentation directly to the user's browser for offline reference.

## 🚀 Getting Started

Download the current release for your platform from the [GitHub Releases](https://github.com/Vahid-jb/Virtual-Characterization-Lab-Toolkit/releases) page and extract the archive to a folder of your choice.

The GUI can be launched by running the `vcl_gui` executable. The compute scripts can also be used standalone by invoking them directly from a terminal with an input file. Refer to the [documentation](Doc.pdf) for details.

## ⚙️ Requirements

Visualizing SAED patterns requires [VisIt](https://visit-dav.github.io/visit-website/releases-as-tables/), which must be installed and accessible on your system `PATH`.

📖 Note

Further detailed information is available in the Doc.pdf user manual.
