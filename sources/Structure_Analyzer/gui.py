# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
Structure Analyzer GUI Application.

This application simplifies the process of analyzing material structures using VCL.
It is organized into tabs for Crystallographic ID and Lattice ID. Crystallographic ID
results are shown as an interactive OVITO 3D view of the detected phases.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QRadioButton,
    QGroupBox, QCheckBox, QTextEdit,
    QTabWidget, QFormLayout, QScrollArea, QFrame, QStackedWidget,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPixmap

from VCL_utils.theme import apply_dark_theme, NoWheelDoubleSpinBox, NoWheelSpinBox, NoWheelComboBox, make_action_button, make_primary_button, make_stop_button, add_run_row
from VCL_utils.general import browse_file, browse_directory, get_script_path
from VCL_utils.visualization import MatplotlibWidget, style_axis_light
from VCL_utils.base_window import BaseModuleWindow
# Must be imported before the ovito modules below: it switches OVITO into GUI mode.
from VCL_utils.ovito_viewer import OvitoViewer

from ovito.io import import_file
from ovito.modifiers import AssignColorModifier

# Phase summary written by Structure_Analyzer/PTM.py (which keeps its own copy of the name).
PHASE_MANIFEST = "ptm_phases.json"
_MANIFEST_SCHEMA_VERSION = 1

_COLOR_BY_PHASE = "Phase"
_COLOR_BY_ELEMENT = "Element"


@dataclass
class _PhaseView:
    """One detected phase: its manifest entry and, once loaded, its OVITO pipeline."""
    name: str
    structure_type: int
    count: int
    percentage: float
    composition: dict
    color: tuple
    path: str | None                    # None when the phase file is missing or failed to load
    pipeline: object = None
    color_modifier: object = None
    viewer_index: int | None = None


def _swatch_icon(rgb) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor.fromRgbF(*rgb))
    return QIcon(pixmap)


def _format_composition(counts: dict) -> str:
    """Format {element: atom count} as e.g. '51% Zn, 49% Cu'."""
    total = sum(counts.values())
    if not total:
        return "n/a"
    return ", ".join(f"{n / total * 100:.0f}% {element}" for element, n in counts.items())


class StructureAnalyzerWindow(BaseModuleWindow):
    """
    Main Window for the Structure Analyzer.

    Contains tabs for different analysis modes corresponding to the workflow
    diagrams in VCL.drawio.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Structure Analyzer")
        self.resize(900, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        header = QLabel("Structure analyzer")
        header.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(header)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self._create_cryst_id_tab()
        self._create_lattice_id_tab()

        # Visualization Stack
        self.viz_stack = QStackedWidget()

        # 0: Crystallographic ID — OVITO phase view, phase table and text report
        self._phase_views: list[_PhaseView] = []
        self.viz_stack.addWidget(self._create_cryst_viz_page())

        # 1: Lattice ID — matplotlib plot
        self.lattice_viz = MatplotlibWidget()
        self.viz_stack.addWidget(self.lattice_viz)

        self.tabs.currentChanged.connect(self.viz_stack.setCurrentIndex)

        # Register buttons
        self._calc_btns = [self.cryst_calc_btn, self.lattice_calc_btn]
        self._stop_btns = [self.cryst_stop_btn, self.lattice_stop_btn]

        # Public interface for PolyCycleMainWindow
        self.control_widget = self.tabs
        self.viz_widget = self.viz_stack

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _create_cryst_id_tab(self) -> None:
        """Creates the 'Crystallographic identification' tab."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll)

        group = QGroupBox("Crystallographic identification")
        form = QFormLayout(group)

        # Input file type
        type_layout = QHBoxLayout()
        self.cryst_single_radio = QRadioButton("single")
        self.cryst_single_radio.toggle()
        self.cryst_trj_radio = QRadioButton("trj")
        type_layout.addWidget(self.cryst_single_radio)
        type_layout.addWidget(self.cryst_trj_radio)
        form.addRow("Input file type:", type_layout)

        self.cryst_trj_timestep = NoWheelSpinBox()
        self.cryst_trj_timestep.setRange(0, 9_999_999)
        self.cryst_trj_timestep.setEnabled(False)
        form.addRow("Trajectory timestep:", self.cryst_trj_timestep)
        self.cryst_single_radio.toggled.connect(self._on_cryst_radio_toggled)

        self.cryst_input = QLineEdit()
        cryst_browse_btn = make_action_button("Browse")
        cryst_browse_btn.clicked.connect(lambda: browse_file(self, self.cryst_input,
            "XYZ / LAMMPS files (*.xyz *.lammps *.dump);;All files (*)"))
        input_row = QHBoxLayout()
        input_row.addWidget(self.cryst_input)
        input_row.addWidget(cryst_browse_btn)
        form.addRow("Input file:", input_row)

        self.cryst_out_dir = QLineEdit()
        cryst_out_btn = make_action_button("Browse")
        cryst_out_btn.clicked.connect(lambda: browse_directory(self, self.cryst_out_dir))
        cryst_out_row = QHBoxLayout()
        cryst_out_row.addWidget(self.cryst_out_dir)
        cryst_out_row.addWidget(cryst_out_btn)
        form.addRow("Output directory:", cryst_out_row)

        self.cryst_min_grain = NoWheelSpinBox()
        self.cryst_min_grain.setRange(0, 99_999)
        self.cryst_min_grain.setValue(5)
        form.addRow("Min grain size:", self.cryst_min_grain)

        self.cryst_rmsd = NoWheelDoubleSpinBox()
        self.cryst_rmsd.setSingleStep(0.1)
        self.cryst_rmsd.setValue(0.1)
        form.addRow("RMSD cutoff:", self.cryst_rmsd)

        self.cryst_calc_btn = make_primary_button("Calculate")
        self.cryst_stop_btn = make_stop_button()
        self.cryst_calc_btn.clicked.connect(self._run_cryst)
        self.cryst_stop_btn.clicked.connect(self._stop_calculation)

        layout.addWidget(group)
        add_run_row(layout, self.cryst_calc_btn, self.cryst_stop_btn)
        layout.addStretch()
        self.tabs.addTab(tab, "Crystallographic ID")

    def _create_lattice_id_tab(self) -> None:
        """Creates the 'Lattice identification' tab."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll)

        group = QGroupBox("Lattice identification")
        form = QFormLayout(group)

        self.lattice_input = QLineEdit()
        lat_btn = make_action_button("Browse")
        lat_btn.clicked.connect(lambda: browse_file(self, self.lattice_input,
            "XYZ / LAMMPS files (*.xyz *.lammps *.dump);;All files (*)"))
        input_row = QHBoxLayout()
        input_row.addWidget(self.lattice_input)
        input_row.addWidget(lat_btn)
        form.addRow("Input file:", input_row)

        self.lattice_out_dir = QLineEdit()
        lat_out_btn = make_action_button("Browse")
        lat_out_btn.clicked.connect(lambda: browse_directory(self, self.lattice_out_dir))
        lat_out_row = QHBoxLayout()
        lat_out_row.addWidget(self.lattice_out_dir)
        lat_out_row.addWidget(lat_out_btn)
        form.addRow("Output directory:", lat_out_row)

        self.lattice_rmsd = NoWheelDoubleSpinBox()
        self.lattice_rmsd.setValue(0.1)
        form.addRow("RMSD cutoff:", self.lattice_rmsd)

        self.lattice_verbose_check = QCheckBox("Verbose")
        self.lattice_verbose_check.setChecked(True)
        form.addRow(self.lattice_verbose_check)

        self.lattice_advanced_check = QCheckBox("Use advanced Methods")
        self.lattice_advanced_check.setChecked(True)
        form.addRow(self.lattice_advanced_check)

        self.lattice_neighbor_k = NoWheelSpinBox()
        self.lattice_neighbor_k.setValue(20)
        form.addRow("Neighbour k:", self.lattice_neighbor_k)

        self.lattice_ls_k_neigh = NoWheelSpinBox()
        self.lattice_ls_k_neigh.setValue(6)
        form.addRow("LS k neighbours:", self.lattice_ls_k_neigh)

        self.lattice_ls_sample = NoWheelSpinBox()
        self.lattice_ls_sample.setRange(0, 999_999)
        self.lattice_ls_sample.setValue(5000)
        form.addRow("LS sample size:", self.lattice_ls_sample)

        self.lattice_ls_bin = NoWheelDoubleSpinBox()
        self.lattice_ls_bin.setValue(0.01)
        form.addRow("LS bin width:", self.lattice_ls_bin)

        self.lattice_ls_periodic = NoWheelComboBox()
        self.lattice_ls_periodic.addItems(["None", "true", "false"])
        form.addRow("LS use periodic:", self.lattice_ls_periodic)

        self.lattice_vs_knn = NoWheelSpinBox()
        self.lattice_vs_knn.setValue(14)
        form.addRow("Vector stats k-NN:", self.lattice_vs_knn)

        self.lattice_vs_cutoff = NoWheelDoubleSpinBox()
        self.lattice_vs_cutoff.setValue(3.5)
        form.addRow("Vector stats max cutoff:", self.lattice_vs_cutoff)

        self.lattice_vs_eps = NoWheelDoubleSpinBox()
        self.lattice_vs_eps.setValue(0.07)
        form.addRow("Vector stats DBSCAN eps:", self.lattice_vs_eps)

        self.lattice_vs_min_neigh = NoWheelSpinBox()
        self.lattice_vs_min_neigh.setValue(6)
        form.addRow("Vector stats min neighbours:", self.lattice_vs_min_neigh)

        self.lattice_vs_merge = NoWheelDoubleSpinBox()
        self.lattice_vs_merge.setValue(0.85)
        form.addRow("Vector stats merge threshold:", self.lattice_vs_merge)

        self.lattice_vs_trials = NoWheelSpinBox()
        self.lattice_vs_trials.setRange(0, 10_000)
        self.lattice_vs_trials.setValue(200)
        form.addRow("Vector stats n trials:", self.lattice_vs_trials)

        self.lattice_vs_sample = NoWheelSpinBox()
        self.lattice_vs_sample.setRange(0, 10_000)
        self.lattice_vs_sample.setValue(3000)
        form.addRow("Vector stats sample size:", self.lattice_vs_sample)

        self.lattice_vs_bins = NoWheelSpinBox()
        self.lattice_vs_bins.setRange(0, 10_000)
        self.lattice_vs_bins.setValue(300)
        form.addRow("Vector stats hist bins:", self.lattice_vs_bins)

        self.lattice_vs_seed = NoWheelSpinBox()
        self.lattice_vs_seed.setValue(42)
        form.addRow("Vector stats random seed:", self.lattice_vs_seed)

        self.lattice_vs_tol = NoWheelDoubleSpinBox()
        self.lattice_vs_tol.setValue(0.12)
        form.addRow("Vector stats tolerance factor:", self.lattice_vs_tol)

        self.lattice_calc_btn = make_primary_button("Calculate")
        self.lattice_stop_btn = make_stop_button()
        self.lattice_calc_btn.clicked.connect(self._run_lattice)
        self.lattice_stop_btn.clicked.connect(self._stop_calculation)

        layout.addWidget(group)
        add_run_row(layout, self.lattice_calc_btn, self.lattice_stop_btn)
        layout.addStretch()
        self.tabs.addTab(tab, "Lattice ID")

    def _create_cryst_viz_page(self) -> QWidget:
        """Creates the Crystallographic ID result page: OVITO phase view above the phase table and report."""
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: toolbar + OVITO viewport
        view_panel = QWidget()
        view_layout = QVBoxLayout(view_panel)
        view_layout.setContentsMargins(0, 0, 0, 0)

        self.cryst_viewer = OvitoViewer("3D view of the detected phases\n(OVITO viewport will appear here)")
        self.cryst_viewer.error.connect(lambda msg: self.logger.log_message("ERROR", msg))

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Color by:"))
        self.cryst_color_combo = NoWheelComboBox()
        self.cryst_color_combo.addItems([_COLOR_BY_PHASE, _COLOR_BY_ELEMENT])
        self.cryst_color_combo.currentTextChanged.connect(self._on_cryst_color_mode_changed)
        toolbar.addWidget(self.cryst_color_combo)
        self.cryst_zoom_btn = make_action_button("Zoom to fit")
        self.cryst_zoom_btn.clicked.connect(lambda: self.cryst_viewer.zoom_all())
        toolbar.addWidget(self.cryst_zoom_btn)
        toolbar.addStretch()

        view_layout.addLayout(toolbar)
        view_layout.addWidget(self.cryst_viewer, 1)
        splitter.addWidget(view_panel)

        # Bottom: phase legend + text report
        self.cryst_result_tabs = QTabWidget()

        self.cryst_phase_table = QTableWidget(0, 4)
        self.cryst_phase_table.setHorizontalHeaderLabels(["Phase", "Atoms", "Share (%)", "Composition"])
        self.cryst_phase_table.verticalHeader().setVisible(False)
        self.cryst_phase_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cryst_phase_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self.cryst_phase_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.cryst_phase_table.itemChanged.connect(self._on_cryst_phase_item_changed)
        self.cryst_result_tabs.addTab(self.cryst_phase_table, "Phases")

        self.cryst_text_view = QTextEdit()
        self.cryst_text_view.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ddd;
                border: 1px solid #444;
                font-family: Consolas, monospace;
            }
        """)
        self.cryst_text_view.setReadOnly(True)
        self.cryst_result_tabs.addTab(self.cryst_text_view, "Report")

        splitter.addWidget(self.cryst_result_tabs)
        splitter.setSizes([450, 250])
        splitter.setCollapsible(0, False)
        return splitter

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_cryst_radio_toggled(self, checked: bool) -> None:
        self.cryst_trj_timestep.setEnabled(not checked)

    def _run_cryst(self) -> None:
        """Validate inputs, write parameter file, and launch PTM script."""
        self.cryst_text_view.clear()
        # Drop the previous result first so OVITO holds none of the files PTM is about to overwrite
        self._reset_phase_view()
        input_path = self.cryst_input.text().strip()
        out_dir_path = self.cryst_out_dir.text().strip()

        if not self._validate_input_file(input_path):
            return
        if not self._validate_output_dir(out_dir_path):
            return

        # Remove the phase manifest of an earlier run so its phases can't be shown as this run's result
        stale_manifest = os.path.join(out_dir_path, PHASE_MANIFEST)
        if os.path.isfile(stale_manifest):
            try:
                os.remove(stale_manifest)
            except OSError as e:
                self.logger.log_message("ERROR", f"Could not remove old phase manifest:\n{e}")
                return

        cryst_params = {
            "input_file_type": "traj" if self.cryst_trj_radio.isChecked() else "single",
            "trj_timestep": self.cryst_trj_timestep.value(),
            "input_file": input_path,
            "min_grain_size": self.cryst_min_grain.value(),
            "RMSD_cutoff": self.cryst_rmsd.value(),
        }

        # PTM uses space-separated params
        self.logger.log_message("INFO", f"Writing parameter file with: {cryst_params}")
        param_file_path = os.path.join(out_dir_path, "ptm_input.txt")
        with open(param_file_path, "w", encoding="utf-8") as f:
            for key, value in cryst_params.items():
                f.write(f"{key} {value}\n")

        self._set_running(True)
        self.runner.on_finished_cb = self._make_finish_callback(
            "Crystallographic Identification", self._plot_cryst
        )
        script_path = get_script_path("Structure_Analyzer", "PTM.py")
        self.runner.start(script_path, args=[param_file_path], cwd=out_dir_path)

    def _run_lattice(self) -> None:
        """Validate inputs, write parameter file, and launch lattice detector script."""
        self.lattice_viz.clear()
        input_path = self.lattice_input.text().strip()
        out_dir_path = self.lattice_out_dir.text().strip()

        if not self._validate_input_file(input_path):
            return
        if not self._validate_output_dir(out_dir_path):
            return

        lat_params = {
            "input": input_path,
            "verbose": "on" if self.lattice_verbose_check.isChecked() else "off",
            "use_advanced_methods": "on" if self.lattice_advanced_check.isChecked() else "off",
            "plot": "on",
            "rmsd_cutoff": self.lattice_rmsd.value(),
            "neighbor_k": self.lattice_neighbor_k.value(),
            "ls_k_neighbors": self.lattice_ls_k_neigh.value(),
            "ls_sample_size": self.lattice_ls_sample.value(),
            "ls_bin_width": self.lattice_ls_bin.value(),
            "ls_use_periodic": self.lattice_ls_periodic.currentText(),
            "vector_stats_k_nn": self.lattice_vs_knn.value(),
            "vector_stats_max_cutoff": self.lattice_vs_cutoff.value(),
            "vector_stats_dbscan_eps": self.lattice_vs_eps.value(),
            "vector_stats_min_neighbors": self.lattice_vs_min_neigh.value(),
            "vector_stats_merge_threshold": self.lattice_vs_merge.value(),
            "vector_stats_n_trials": self.lattice_vs_trials.value(),
            "vector_stats_sample_size": self.lattice_vs_sample.value(),
            "vector_stats_hist_bins": self.lattice_vs_bins.value(),
            "vector_stats_random_seed": self.lattice_vs_seed.value(),
            "vector_stats_tolerance_factor": self.lattice_vs_tol.value(),
        }

        # Remove histogram data from an earlier run so it can't be plotted as this run's result
        stale_json = os.path.join(out_dir_path, "lattice_hist_comp_data.json")
        if os.path.isfile(stale_json):
            try:
                os.remove(stale_json)
            except OSError as e:
                self.logger.log_message("ERROR", f"Could not remove old histogram data:\n{e}")
                return

        param_file_path = os.path.join(out_dir_path, "lat_input.txt")
        self._write_and_run(
            param_file_path, lat_params,
            ("Structure_Analyzer", "lattice_detector.py"),
            out_dir_path,
            self._make_finish_callback("Lattice Identification", self._plot_lattice),
        )

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def _plot_lattice(self) -> None:
        """Read lattice histogram JSON and render the comparison plot."""
        out_dir_path = self.lattice_out_dir.text().strip()
        data_path = os.path.join(out_dir_path, "lattice_hist_comp_data.json")
        if not os.path.isfile(data_path):
            self.logger.log_message("WARNING", "No histogram data was written by this run "
                                    "(see the log above); nothing to plot.")
            return
        try:
            with open(data_path, "r", encoding="utf-8") as jsonfile:
                data = json.load(jsonfile)

            vector_lengths = np.asarray(data["vector_lengths"], dtype=float)
            pairwise_distances = np.asarray(data["pairwise_distances"], dtype=float)

            def plot_func(fig):
                fig.clear()
                fig.patch.set_facecolor("white")

                ax1, ax2 = fig.subplots(1, 2, sharey=True)
                fig.subplots_adjust(wspace=0.25)

                ax1.hist(vector_lengths, bins=100, alpha=0.7, color="skyblue", edgecolor="black", density=True)
                ax1.set_xlabel("Vector Length (Å)", fontsize=12)
                ax1.set_ylabel("Normalized Frequency", fontsize=12)
                ax1.set_title("Interatomic Vector Lengths\n(Vector Statistics Method)", fontsize=14, fontweight="bold")
                ax1.grid(True, alpha=0.3)
                ax1.text(0.02, 0.98, f"Count: {len(vector_lengths):,}", transform=ax1.transAxes, va="top",
                         fontsize=10, color="black",
                         bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", linewidth=1))
                style_axis_light(ax1)

                ax2.hist(pairwise_distances, bins=100, alpha=0.7, color="lightcoral", edgecolor="black", density=True)
                ax2.set_xlabel("Nearest-Neighbour Distance (Å)", fontsize=12)
                ax2.set_ylabel("Normalized Frequency", fontsize=12)
                ax2.set_title("Nearest-Neighbour Distances\n(RDF Shell-Ratio Method)", fontsize=14, fontweight="bold")
                ax2.grid(True, alpha=0.3)
                ax2.text(0.02, 0.98, f"Count: {len(pairwise_distances):,}", transform=ax2.transAxes, va="top",
                         fontsize=10, color="black",
                         bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", linewidth=1))
                style_axis_light(ax2)

            self.lattice_viz.plot(plot_func)

        except Exception as e:
            self.logger.log_message("ERROR", f"Error plotting lattice histogram: {e}")

    def _plot_cryst(self) -> None:
        """Display the crystallographic text report and load the detected phases into the 3D view."""
        out_dir_path = self.cryst_out_dir.text().strip()
        try:
            result_file = os.path.join(out_dir_path, "structure_analysis.txt")
            with open(result_file) as f:
                content = f.read()
            self.cryst_text_view.setPlainText(content)
        except Exception as e:
            self.logger.log_message("ERROR", f"Error reading crystallographic results: {e}")

        # Separate from the report so a problem with the phase files never hides it
        try:
            self._load_phase_view(out_dir_path)
        except Exception as e:
            self._reset_phase_view()
            self.logger.log_message("ERROR", f"Error building the 3D phase view: {e}")

    # ------------------------------------------------------------------
    # Crystallographic ID phase view
    # ------------------------------------------------------------------

    def _reset_phase_view(self) -> None:
        """Empty the 3D view and the phase table, dropping the OVITO pipelines."""
        self.cryst_viewer.clear()
        self._phase_views.clear()
        self.cryst_phase_table.setRowCount(0)

    def _read_phase_manifest(self, out_dir_path: str) -> dict | None:
        """Return the PTM phase manifest in *out_dir_path*, or None (logged) if it is missing or unreadable."""
        manifest_path = os.path.join(out_dir_path, PHASE_MANIFEST)
        if not os.path.isfile(manifest_path):
            self.logger.log_message("WARNING", "No phase manifest was written by this run; "
                                    "3D phase view unavailable.")
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict) or manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION:
                raise ValueError("unsupported manifest format")
        except (OSError, ValueError) as e:
            self.logger.log_message("ERROR", f"Could not read the phase manifest {manifest_path}:\n{e}")
            return None
        return manifest

    def _load_phase_view(self, out_dir_path: str) -> None:
        """Load each phase listed in the PTM manifest into the 3D view and the phase table."""
        self._reset_phase_view()
        manifest = self._read_phase_manifest(out_dir_path)
        if manifest is None:
            self.cryst_result_tabs.setCurrentWidget(self.cryst_text_view)
            return

        for entry in manifest["phases"]:
            path = None
            if entry.get("file"):
                path = os.path.join(out_dir_path, os.path.basename(entry["file"]))
                if not os.path.isfile(path):
                    self.logger.log_message("WARNING", f"Phase file not found:\n{path}")
                    path = None
            self._phase_views.append(_PhaseView(
                name=str(entry["name"]),
                structure_type=int(entry["structure_type"]),
                count=int(entry["count"]),
                percentage=float(entry["percentage"]),
                composition=dict(entry.get("composition") or {}),
                color=tuple(min(max(float(c), 0.0), 1.0) for c in entry.get("color", (0.7, 0.7, 0.7))),
                path=path,
            ))

        # import_file() reads the phase files on the GUI thread
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for pv in self._phase_views:
                self._load_phase_pipeline(pv)
        finally:
            QApplication.restoreOverrideCursor()

        self._fill_phase_table()
        self.cryst_viewer.zoom_all()
        self.cryst_result_tabs.setCurrentWidget(self.cryst_phase_table)

    def _load_phase_pipeline(self, pv: _PhaseView) -> None:
        """Import a phase file into its own OVITO pipeline, colored by phase, and add it to the 3D view."""
        if pv.path is None:
            return
        try:
            pipeline = import_file(pv.path)
        except Exception as e:
            self.logger.log_message("ERROR", f"Could not load phase file {pv.path}:\n{e}")
            pv.path = None
            return
        pv.color_modifier = AssignColorModifier(color=pv.color)
        pv.color_modifier.enabled = self._color_by_phase()
        pipeline.modifiers.append(pv.color_modifier)
        pv.pipeline = pipeline
        pv.viewer_index = self.cryst_viewer.add_pipeline(pipeline)

    def _fill_phase_table(self) -> None:
        """Fill the phase table, which doubles as the legend of the 3D view."""
        table = self.cryst_phase_table
        # Setting check states emits itemChanged, which would toggle phases while filling
        table.blockSignals(True)
        try:
            table.setRowCount(len(self._phase_views))
            for row, pv in enumerate(self._phase_views):
                name_item = QTableWidgetItem(_swatch_icon(pv.color), pv.name)
                name_item.setData(Qt.ItemDataRole.UserRole, row)
                if pv.viewer_index is None:
                    name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    name_item.setForeground(QBrush(QColor("#777")))
                    name_item.setToolTip("Phase file missing or unreadable; not shown in the 3D view.")
                else:
                    name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                    name_item.setCheckState(Qt.CheckState.Checked)
                table.setItem(row, 0, name_item)

                for col, text in ((1, f"{pv.count:,}"), (2, f"{pv.percentage:.2f}"),
                                  (3, _format_composition(pv.composition))):
                    item = QTableWidgetItem(text)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    if col < 3:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    table.setItem(row, col, item)
        finally:
            table.blockSignals(False)

    def _on_cryst_phase_item_changed(self, item: QTableWidgetItem) -> None:
        """Show or hide a phase when its check box in the phase table is toggled."""
        if item.column() != 0:
            return
        row = item.data(Qt.ItemDataRole.UserRole)
        if row is None or not 0 <= row < len(self._phase_views):
            return
        pv = self._phase_views[row]
        if pv.viewer_index is not None:
            self.cryst_viewer.set_pipeline_visible(pv.viewer_index, item.checkState() == Qt.CheckState.Checked)

    def _color_by_phase(self) -> bool:
        return self.cryst_color_combo.currentText() == _COLOR_BY_PHASE

    def _on_cryst_color_mode_changed(self, _text: str) -> None:
        """Switch the 3D view between phase colors and element colors."""
        by_phase = self._color_by_phase()
        for pv in self._phase_views:
            if pv.color_modifier is not None:
                pv.color_modifier.enabled = by_phase


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = StructureAnalyzerWindow()
    window.show()
    sys.exit(app.exec())
