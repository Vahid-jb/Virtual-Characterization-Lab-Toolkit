# Third-Party Notices

The VCL Toolkit is licensed under the **MIT License** (see [`LICENSE`](LICENSE)).

The distributed application **binaries** (the PyInstaller bundles published as
release assets) embed third-party components that remain under **their own
licenses**. Those licenses and their attribution/notice requirements are listed
below. This file is shipped inside the binary bundle to satisfy those
obligations. Full license texts for each component are available from the
component's own project (linked below) and, where present, in the corresponding
package directory inside the bundle (`_internal/`).

> This is an engineering-maintained notice, not formal legal advice.

## Bundled components

| Component | License | Notes |
|-----------|---------|-------|
| PyInstaller bootloader | GPL-2.0-or-later **with bootloader exception** | The exception permits distributing the frozen application under any license. |
| PySide6 / Shiboken6 (Qt for Python) and the Qt 6 libraries | LGPL-3.0-only (also offered as GPL-2.0/GPL-3.0) | Used under the LGPL. Dynamically linked; the bundle is a folder, so the Qt libraries in `_internal/` can be replaced with modified versions. Source: <https://code.qt.io/> and <https://download.qt.io/official_releases/QtForPython/>. |
| OVITO (PyPI `ovito`) | MIT | The MIT PyPI module — **not** the proprietary `conda.ovito.org` OVITO Pro build. Bundles the NVIDIA CUDA runtime — `libcudart.so.12` on Linux, `cudart64_12.dll` on Windows (see the CUDA / NVIDIA note). |
| ASE (Atomic Simulation Environment) | LGPL-2.1-or-later | Dynamically imported Python dependency. <https://gitlab.com/ase/ase> |
| pymatgen | MIT | |
| NumPy | BSD-3-Clause (with 0BSD, MIT, Zlib and CC0-1.0 parts) | Wheels bundle OpenBLAS (BSD-3-Clause) and the GCC runtime libraries (GPL-3.0 with the GCC Runtime Library Exception). |
| SciPy | BSD-3-Clause | Wheels bundle OpenBLAS (BSD-3-Clause) and the GCC runtime libraries (GPL-3.0 with the GCC Runtime Library Exception). |
| Matplotlib | Matplotlib License (PSF-based) | Bundles the DejaVu (Bitstream Vera / public domain), STIX (SIL OFL-1.1) and other fonts under the licenses listed in its `LICENSE` directory. |
| FreeType (in Matplotlib's `ft2font`) | FreeType License (FTL) | Dual-licensed FTL / GPL-2.0-or-later; used under the FTL. Portions of this software are copyright © The FreeType Project (<https://freetype.org>). All rights reserved. |
| periodictable | Public domain | |
| QtAwesome (`qtawesome`) | MIT (code) | Supplies the action-button icons. Bundles icon **fonts** under their own terms, each requiring attribution: Font Awesome Free 5 & 6 (icons CC-BY-4.0, fonts SIL OFL-1.1, code MIT — <https://fontawesome.com/license/free>), Material Design Icons 5 & 6 (Apache-2.0 — <https://pictogrammers.com/docs/general/license/>), Phosphor Icons (MIT), Remix Icon (Apache-2.0), Elusive Icons (SIL OFL-1.1), Microsoft Codicons (icons CC-BY-4.0, code MIT). Only the Font Awesome 5 solid set is used, but QtAwesome registers every bundled family at initialisation, so all of them ship. |
| psutil, scikit-learn, mpi4py | BSD-3-Clause | |
| pandas, spglib, networkx, sympy, monty, and other Python dependencies | BSD-3-Clause / MIT / PSF / Apache-2.0 (permissive) | Reproduced under their permissive terms. |
| Microsoft Visual C++ runtime (Windows bundle only) | Microsoft Visual C++ Redistributable license | `vcruntime140*.dll` / `msvcp140*.dll`, shipped with Python and Qt for Windows. |

## Bundled documentation site

The offline documentation (`docs_site/site`) served by the GUI includes:

| Component | License |
|-----------|---------|
| Bootstrap | MIT |
| Font Awesome Free | Icons CC-BY-4.0, fonts SIL OFL-1.1, code MIT — <https://fontawesome.com/license/free>. Also ships as a runtime UI font via QtAwesome; see the row above. |
| lunr.js (search) | MIT |

highlight.js (BSD-3-Clause) and MathJax (Apache-2.0) are loaded from their
CDNs when a page is viewed; they are not part of the bundle.

## CUDA / NVIDIA note

NVIDIA CUDA libraries (`libcudnn*`, `libcublas*`, `cublas64_*.dll`, …) carry
the proprietary NVIDIA Software License and are **not** included, with one
disclosed exception:

- **The NVIDIA CUDA Runtime** is redistributed in both bundles: at
  `_internal/ovito/plugins/libcudart.so.12` on Linux (and a symlink to it at
  `_internal/`), and at `_internal/ovito/plugins/cudart64_12.dll` on Windows.
  It ships inside the MIT-licensed OVITO PyPI wheel and is a hard load-time
  dependency of OVITO's bindings — a `NEEDED` entry of `ovito_bindings.so` and
  an import-table entry of `ovito_bindings.pyd`. It is neither `dlopen`'d nor
  delay-loaded, so it cannot be removed without removing OVITO. NVIDIA's CUDA
  EULA permits redistributing the CUDA runtime as part of an application. The
  license gate (`scripts/check_licenses.py`) exempts these two paths and
  continues to fail on every other CUDA library anywhere in the bundle.

## Components intentionally excluded from the redistributable binary

The following are **not** redistributable and must not be bundled in published
release binaries:

- **OVITO Pro** (the `conda.ovito.org` build of `ovito`) — proprietary. The
  license gate requires `ovito` to come from PyPI.
- **Intel MKL** (`libmkl_*`, `mkl_*.dll`, `libiomp5*`) — Intel Simplified
  Software License (non-free). NumPy and SciPy use OpenBLAS instead.
- **NVIDIA CUDA libraries** other than the runtime exception above.

## VisIt (external, user-installed)

Visualizing SAED patterns uses **VisIt** (BSD-3-Clause,
<https://visit-dav.github.io/visit-website/>), which the user installs
separately and makes available on `PATH`. It is invoked as a separate program
and is not part of, nor redistributed with, the VCL Toolkit.
