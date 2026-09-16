# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
converged_ui.py
===============

Top-level launcher for the VCL Toolkit.

Instantiates every sub-module window eagerly (so PyInstaller can detect all
imports statically), embeds their control and visualization widgets into a
unified sidebar-driven shell, and wires per-module log signals to a shared
log panel at the bottom of the window.
"""

import sys
import os
import multiprocessing
import threading
import http.server
import socketserver
import webbrowser
import functools
from dataclasses import dataclass, field

# --- LINUX OPENGL / DISPLAY FIX FOR STANDALONE BUNDLE ---
# Must be set before ANY Qt / PySide6 / Ovito import.
#
# PyInstaller bundles Mesa libraries from the build container (Debian Bullseye).
# Those libraries cannot talk to the host GPU drivers (different kernel, different
# DRI version), so both EGL (Wayland) and GLX (X11) hardware paths fail:
#   - Wayland:  "qt.qpa.wayland: EGL not available"
#   - XCB/X11:  "Could not initialize GLX" -> crash
#
# The fix is twofold:
#   1. Force the xcb (X11 / XWayland) Qt platform plugin – it is the most
#      widely supported backend and avoids the Wayland-EGL code path entirely.
#   2. Force Mesa software rendering (llvmpipe) via LIBGL_ALWAYS_SOFTWARE.
#      This makes the *bundled* Mesa render without needing any GPU driver,
#      which is the only reliably portable option.  Performance is perfectly
#      acceptable for the Ovito 3-D viewport.
if sys.platform == "linux" and getattr(sys, "frozen", False):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QStackedWidget, QLabel, QFrame, QMessageBox, QTextEdit, QSplitter,
    QTabWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase

# Insert the project root so sub-packages are importable when running from source.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VCL_utils.theme import apply_dark_theme
from VCL_utils.general import get_base_dir

# Eager imports are required so PyInstaller's static analysis can detect every
# sub-module and bundle it into the frozen binary.
from Input_Converter.gui import InputConvertorWindow
from Structure_Analyzer.gui import StructureAnalyzerWindow
from Vibrational_Analysis.gui import VibrationalAnalysisWindow
from XRD.gui import XrayWindow
from TEM.lammps.gui import TEMWindow


@dataclass
class ModuleEntry:
    """Descriptor for a single VCL sub-module.

    Bundling the class reference, sidebar label, and documentation paths in one
    place means that adding a new module only requires a single entry in
    ``_MODULES`` — the navigation sidebar, stacked widgets, and documentation
    button all derive their data from it automatically.
    """
    label: str
    window_cls: type
    doc_path: str = "index.html"
    """Root documentation page, relative to the docs-site root."""
    doc_tab_paths: dict[str, str] = field(default_factory=dict)
    """Per-tab page overrides, keyed by the tab label shown in the GUI."""


# Module registry: order determines sidebar position and stack index.
_MODULES = [
    ModuleEntry(
        label="Input Converter",
        window_cls=InputConvertorWindow,
        doc_path="Input_Converter_Docs/index.html",
    ),
    ModuleEntry(
        label="Structure Analyzer",
        window_cls=StructureAnalyzerWindow,
        doc_path="Structure_Analyzer_Docs/index.html",
    ),
    ModuleEntry(
        label="Vibrational Analysis",
        window_cls=VibrationalAnalysisWindow,
        doc_path="Vibrational_Analysis_Docs/index.html",
        doc_tab_paths={
            "VDOS": "Vibrational_Analysis_Docs/index.html#part-i-vibrational-density-of-states-vdos",
            "IR":   "Vibrational_Analysis_Docs/index.html#part-ii-infrared-ir-spectra",
        },
    ),
    ModuleEntry(
        label="XRD Analysis",
        window_cls=XrayWindow,
        doc_path="XRD_Docs/index.html",
        doc_tab_paths={
            "XRD-Kinematical":   "XRD_Docs/index.html#part-ii-xrd-kinematical",
            "XRD-Debye":         "XRD_Docs/index.html#part-iii-xrd-debye-scattering",
            "XRD-ReciprocalSum": "XRD_Docs/index.html#part-i-xrd-reciprocalsum",
        },
    ),
    ModuleEntry(
        label="SAED",
        window_cls=TEMWindow,
        doc_path="SAED_Docs/index.html",
    ),
]


def _ui_font(size: int) -> QFont:
    """Return the platform's UI font at *size* points.

    Naming a family here would pin the sidebar to Segoe UI, which exists only on
    Windows; the other targets silently fell back to a default face that did not
    match the rest of the app.
    """
    return QFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family(), size)


class NavigationButton(QPushButton):
    """Checkable sidebar navigation button with a left-accent indicator when active."""

    def __init__(self, text: str, icon_name: str | None = None) -> None:
        super().__init__(text)
        self.setCheckable(True)
        self.setAutoExclusive(True)
        self.setMinimumHeight(50)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(_ui_font(11))

        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding-left: 20px;
                border: none;
                background-color: transparent;
                color: #aaaaaa;
                border-left: 4px solid transparent;
            }
            QPushButton:hover {
                background-color: #333333;
                color: #ffffff;
            }
            QPushButton:checked {
                background-color: #2d2d2d;
                color: #ffffff;
                border-left: 4px solid #2a82da;
                font-weight: bold;
            }
        """)


class HelpButton(QPushButton):
    """Sidebar button styled for secondary actions (e.g. Help / documentation)."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setMinimumHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(_ui_font(10))
        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding-left: 20px;
                border: none;
                background-color: transparent;
                color: #888;
            }
            QPushButton:hover {
                color: #2a82da;
                font-weight: bold;
            }
        """)


class DocumentationServer:
    """
    Simple HTTP server to serve static documentation files from a local directory.
    Runs in a separate daemon thread.
    """
    def __init__(self, root_dir: str, start_port: int = 8000) -> None:
        self.root_dir = root_dir
        self.port = start_port
        self.server = None
        self.thread = None
        self.is_running = False

    def start(self) -> int | None:
        if self.is_running:
            return self.port

        class SilentRequestHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass # Prevent writing to sys.stderr which is None in PyInstaller GUI apps
                
        while self.port < 65535:
            try:
                handler = functools.partial(
                    SilentRequestHandler,
                    directory=self.root_dir
                )
                self.server = http.server.ThreadingHTTPServer(("", self.port), handler)
                self.server.daemon_threads = True
                self.is_running = True

                self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
                self.thread.start()

                print(f"Documentation server started at http://localhost:{self.port}")
                return self.port
            except OSError:
                self.port += 1

        print("Could not find a free port for documentation server.")
        return None

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.is_running = False

    def get_url(self) -> str | None:
        if self.is_running:
            return f"http://localhost:{self.port}/index.html"
        return None


class PolyCycleMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Virtual Characterization Lab (VCL) Toolkit")
        self.resize(1280, 900)

        docs_path = os.path.join(get_base_dir(), "docs_site", "site")
        self.doc_server = DocumentationServer(docs_path)  # lazy-started on first use

        self._windows = [None] * len(_MODULES)

        # Central Widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Top Section: Sidebar + Content Splitter ---
        top_container = QWidget()
        top_layout = QHBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        # Sidebar
        self.sidebar = QFrame()
        # Scoped by object name: a selector-less stylesheet would cascade into every
        # child and take them out of the native style.
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setStyleSheet(
            "QFrame#sidebar { background-color: #1e1e1e; border-right: 1px solid #333; }")
        self.sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 20, 0, 20)
        sidebar_layout.setSpacing(10)

        app_title = QLabel("VCL")
        app_title.setStyleSheet(
            "color: white; font-size: 22px; font-weight: bold; padding-left: 20px; margin-bottom: 20px;"
        )
        sidebar_layout.addWidget(app_title)

        self.nav_group: list[QPushButton] = []

        self.btn_input       = self._add_nav_btn(sidebar_layout, "Input Converter",      0)
        self.btn_structure   = self._add_nav_btn(sidebar_layout, "Structure Analyzer",   1)
        self.btn_vibrational = self._add_nav_btn(sidebar_layout, "Vibrational Analysis", 2)
        self.btn_xrd         = self._add_nav_btn(sidebar_layout, "XRD Analysis",         3)
        self.btn_tem         = self._add_nav_btn(sidebar_layout, "SAED",                 4)

        sidebar_layout.addStretch()

        self.btn_help = HelpButton("Help / Documentation")
        self.btn_help.clicked.connect(self.open_documentation)
        sidebar_layout.addWidget(self.btn_help)

        footer = QLabel("v0.0.1")
        footer.setStyleSheet("color: #555; padding-left: 20px;")
        sidebar_layout.addWidget(footer)

        top_layout.addWidget(self.sidebar)

        # Horizontal Splitter for [Controls | Visualization]
        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.content_area = QStackedWidget()
        self.content_area.setStyleSheet("QStackedWidget { background-color: #2d2d2d; }")
        self.h_splitter.addWidget(self.content_area)

        self.viz_area = QStackedWidget()
        self.viz_area.setStyleSheet("QStackedWidget { background-color: #2b2b2b; }")
        self.h_splitter.addWidget(self.viz_area)

        self.h_splitter.setSizes([500, 700])
        self.h_splitter.setCollapsible(0, False)
        self.h_splitter.setCollapsible(1, False)

        top_layout.addWidget(self.h_splitter)

        self.log_stack = QStackedWidget()

        # Instantiate every sub-module window and embed its widgets.
        for i, entry in enumerate(_MODULES):
            try:
                window = entry.window_cls()
                self._windows[i] = window

                control = window.get_control_widget()
                viz = window.get_visualization_widget()

                self.content_area.addWidget(control)
                self.viz_area.addWidget(viz)

                log_view = QTextEdit()
                log_view.setReadOnly(True)
                self.log_stack.addWidget(log_view)
                window.log_message.connect(lambda msg, idx=i: self.append_log(idx, msg))
                self.append_log(i, f"INFO: {entry.label} loaded.")

            except Exception as e:
                # Placeholder widgets keep the stacked-widget indices aligned on failure.
                ph = QLabel(f"Failed to load {entry.label}:\n{e}")
                ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
                ph.setStyleSheet("color: #666; font-size: 14px;")
                self.content_area.addWidget(ph)
                self.viz_area.addWidget(QLabel(""))

                log_view = QTextEdit()
                log_view.setReadOnly(True)
                self.log_stack.addWidget(log_view)
                self.append_log(i, f"ERROR: {e}")

        # --- Bottom Section: Logs ---
        self.log_container = QFrame()
        self.log_container.setStyleSheet("""
            QFrame {
                background-color: #222222;
                border-top: 2px solid #3d3d3d;
            }
            QLabel {
                color: #ccc;
                font-weight: bold;
                padding: 5px;
            }
            QTextEdit {
                background-color: #1a1a1a;
                color: #ddd;
                border: 1px solid #444;
                font-family: Consolas, monospace;
            }
        """)
        log_layout = QVBoxLayout(self.log_container)
        log_layout.setContentsMargins(10, 5, 10, 10)

        log_header = QLabel("Interface Output / Parameters / Commands")
        log_layout.addWidget(log_header)
        log_layout.addWidget(self.log_stack)

        # Vertical splitter (top content + bottom logs)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top_container)
        splitter.addWidget(self.log_container)
        splitter.setSizes([650, 250])
        splitter.setHandleWidth(8)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        main_layout.addWidget(splitter)

        # Select default module
        self.btn_input.setChecked(True)
        self.switch_view(0)

    def switch_view(self, index: int) -> None:
        self.content_area.setCurrentIndex(index)
        self.viz_area.setCurrentIndex(index)
        self.log_stack.setCurrentIndex(index)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def append_log(self, index: int, message: str) -> None:
        log_view = self.log_stack.widget(index)
        if log_view is None:
            return
        if message.startswith("ERROR"):
            color = "#f44336"
        elif message.startswith("INFO"):
            color = "#4caf50"
        elif message.startswith("STDOUT"):
            color = "#ddd"
        else:
            color = "#aaa"
        log_view.append(f'<span style="color:{color}">{message}</span>')

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _add_nav_btn(self, sidebar_layout, text: str, index: int) -> QPushButton:
        btn = NavigationButton(text)
        btn.clicked.connect(lambda checked=False, i=index: self.switch_view(i))
        sidebar_layout.addWidget(btn)
        self.nav_group.append(btn)
        return btn

    # ------------------------------------------------------------------
    # Documentation
    # ------------------------------------------------------------------

    def open_documentation(self) -> None:
        if not self.doc_server.is_running:
            port = self.doc_server.start()
            if not port:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Could not start documentation server.\nNo free ports found."
                )
                return

        base_url = f"http://localhost:{self.doc_server.port}"
        current_idx = self.content_area.currentIndex()

        if current_idx < 0 or current_idx >= len(_MODULES):
            url = f"{base_url}/index.html"
        else:
            entry = _MODULES[current_idx]
            path = entry.doc_path

            if entry.doc_tab_paths:
                active_window = self._windows[current_idx]
                if active_window is not None:
                    control_widget = active_window.get_control_widget()
                    if isinstance(control_widget, QTabWidget):
                        tab_text = control_widget.tabText(control_widget.currentIndex())
                        path = entry.doc_tab_paths.get(tab_text, path)

            url = f"{base_url}/{path}"

        if url:
            self.btn_help.setText(f"Docs  (port {self.doc_server.port})")
            webbrowser.open(url)

    def closeEvent(self, event) -> None:
        if self.doc_server:
            self.doc_server.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    os.environ["PYTHONUTF8"] = "1"
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = PolyCycleMainWindow()

    try:
        import pyi_splash
        pyi_splash.close()
    except ImportError:
        pass

    window.show()

    sys.exit(app.exec())