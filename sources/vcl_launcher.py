# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
Entry-point module for the VCL Toolkit.

This thin wrapper exists so that ``pip install .`` can register a console
script (``vcl-toolkit``) that launches the converged UI.  It also ensures
that ``multiprocessing.freeze_support()`` is called before anything else
when the app is run as a frozen binary.
"""
from __future__ import annotations

import multiprocessing
import sys
import os
from pathlib import Path

# --- LINUX DISPLAY BACKEND FIX FOR STANDALONE BUNDLE ---
# Must be set before ANY Qt / VTK import.
# PyInstaller bundles old Mesa/EGL libraries from the build container that
# conflict with the host GPU drivers, causing "EGL not available" and
# "QOpenGLWidget is not supported" errors under Wayland.
# Forcing the xcb (X11 / XWayland) platform avoids the broken Wayland-EGL
# path entirely and works on virtually every Linux desktop.
if sys.platform == "linux" and getattr(sys, "frozen", False):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# --- DLL FIX FOR STANDALONE BUNDLE ---
if getattr(sys, 'frozen', False) or __file__:
    bundle_dir = Path(__file__).parent
    python_dir = bundle_dir / "python"
    
    if python_dir.exists():
        # Add the main DLL folders to the search path
        dll_folders = [
            python_dir,
            python_dir / "Library" / "bin",
            python_dir / "Lib" / "site-packages" / "PySide6",
            python_dir / "Lib" / "site-packages" / "shiboken6",
        ]
        for folder in dll_folders:
            if folder.exists():
                os.add_dll_directory(str(folder.resolve()))
# -------------------------------------

from PySide6.QtWidgets import QApplication  # Now this should work

def main() -> None:
    """Launch the PolyCycle VCL Toolkit GUI."""
    multiprocessing.freeze_support()

    # Ensure subpackages are importable regardless of how we were invoked
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from PySide6.QtWidgets import QApplication
    from VCL_utils.theme import apply_dark_theme

    # Import here to avoid circular / heavy imports at module level
    from converged_ui import PolyCycleMainWindow

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = PolyCycleMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
