# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
Vibrational Analysis GUI Application.

Provides tools for analyzing vibrational properties, including
Vibrational Density of States (VDOS) and Infrared (IR) spectra.
"""

from __future__ import annotations

import os

import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy.signal import find_peaks

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit,
    QGroupBox, QCheckBox,
    QTabWidget, QFormLayout,
    QScrollArea, QFrame, QStackedWidget,
)

from VCL_utils.theme import apply_dark_theme, make_action_button, make_primary_button, make_stop_button, add_run_row, NoWheelComboBox, NoWheelSpinBox, NoWheelDoubleSpinBox
from VCL_utils.general import browse_file, create_file, browse_directory, get_script_path
from VCL_utils.validators import (
    apply_validator,
    SpaceIntListValidator,
    MassChargeListValidator,
)
from VCL_utils.visualization import MatplotlibWidget, style_axis_light
from VCL_utils.base_window import BaseModuleWindow

_XYZ_FILTER = ("Trajectory files (*.xyz *.nc *.netcdf);;XYZ / LAMMPS dump (*.xyz);;"
               "NetCDF files (*.nc *.netcdf);;All files (*)")
_IR_FILTER   = "XYZ / LAMMPS dump (*.xyz);;All files (*)"  # IR.py cannot read NetCDF
_TXT_FILTER  = "Text files (*.txt);;All files (*)"

_WINDOW_KINDS = ["Gaussian", "Hann", "Hamming", "Blackman-harris"]
_ESTIMATORS = ["acf", "welch"]
_PEAK_MIN_SEP_CM = 120.0  # minimum separation of labelled IR peaks (same as IR.py)


class VibrationalAnalysisWindow(BaseModuleWindow):
    """Main Window for Vibrational Analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vibrational Analysis")
        self.resize(800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        header = QLabel("Vibrational analysis")
        header.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(header)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self._create_vdos_tab()
        self._create_ir_tab()

        self.viz_stack = QStackedWidget()
        self.vdos_viz = MatplotlibWidget()
        self.viz_stack.addWidget(self.vdos_viz)
        self.ir_viz = MatplotlibWidget()
        self.viz_stack.addWidget(self.ir_viz)
        self.tabs.currentChanged.connect(self.viz_stack.setCurrentIndex)

        # Register buttons
        self._calc_btns = [self.vdos_calc_btn, self.ir_calc_btn]
        self._stop_btns = [self.vdos_stop_btn, self.ir_stop_btn]

        # Public interface for PolyCycleMainWindow
        self.control_widget = self.tabs
        self.viz_widget = self.viz_stack

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _create_vdos_tab(self) -> None:
        """Creates the 'VDOS' (Vibrational Density of States) analysis tab."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll)

        group = QGroupBox("VDOS Parameters")
        form = QFormLayout(group)

        self.vdos_input = QLineEdit()
        btn_in = make_action_button("Browse")
        btn_in.clicked.connect(lambda: browse_file(self, self.vdos_input, _XYZ_FILTER))
        row_in = QHBoxLayout()
        row_in.addWidget(self.vdos_input)
        row_in.addWidget(btn_in)
        form.addRow("Input (xyz/nc):", row_in)

        self.vdos_out_dir = QLineEdit()
        btn_out_dir = make_action_button("Browse")
        btn_out_dir.clicked.connect(lambda: browse_directory(self, self.vdos_out_dir))
        row_out_dir = QHBoxLayout()
        row_out_dir.addWidget(self.vdos_out_dir)
        row_out_dir.addWidget(btn_out_dir)
        form.addRow("Output directory:", row_out_dir)

        mode_layout = QHBoxLayout()
        self.vdos_mode = NoWheelComboBox()
        self.vdos_mode.addItems(["full", "bond"])
        self.vdos_mode.setToolTip("'full' uses all atoms; 'bond' restricts to atoms specified by bond_indices")
        self.vdos_mode.currentIndexChanged.connect(self._vdos_mode_index_changed)
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.vdos_mode)
        self.vdos_window = NoWheelComboBox()
        self.vdos_window.addItems(_WINDOW_KINDS)
        self.vdos_window.currentIndexChanged.connect(self._vdos_window_index_changed)
        self.vdos_window.setToolTip(
            "Window function applied before the transform.\n"
            "Gaussian is recommended: it is the only one whose spectral window is\n"
            "non-negative, which guarantees a non-negative spectrum."
        )
        mode_layout.addWidget(QLabel("Window:"))
        mode_layout.addWidget(self.vdos_window)
        form.addRow("Config:", mode_layout)

        self.vdos_bond_indices = QLineEdit()
        self.vdos_bond_indices.setPlaceholderText("e.g. 1 2 (ints)")
        self.vdos_bond_indices.setText("1 2")
        self.vdos_bond_indices.setToolTip("Space-separated atom indices (1-based) for 'bond' mode")
        form.addRow("Bond indices:", self.vdos_bond_indices)
        apply_validator(self.vdos_bond_indices, SpaceIntListValidator())
        self.vdos_bond_indices.setEnabled(False)

        self.vdos_estimator = NoWheelComboBox()
        self.vdos_estimator.addItems(_ESTIMATORS)
        self.vdos_estimator.setToolTip(
            "Spectral estimator:\n"
            "  'acf'   = Blackman-Tukey (biased vector ACF + decaying lag window)\n"
            "  'welch' = averaged modified periodogram\n"
            "Both return a one-sided PSD on the same scale; running both is a good cross-check."
        )
        self.vdos_estimator.currentIndexChanged.connect(self._vdos_estimator_index_changed)
        form.addRow("Spectral estimator:", self.vdos_estimator)

        self.vdos_window_width = NoWheelDoubleSpinBox()
        self.vdos_window_width.setRange(0.001, 1e6)
        self.vdos_window_width.setValue(1.0)
        self.vdos_window_width.setToolTip(
            "FWHM of the equivalent symmetric lag window, in picoseconds ('acf' + Gaussian).\n"
            "Sets the spectral resolution: ~29 cm-1 for 1.0 ps."
        )
        form.addRow("Window width (FWHM, ps):", self.vdos_window_width)

        self.vdos_welch_segment = NoWheelDoubleSpinBox()
        self.vdos_welch_segment.setRange(0.001, 1e6)
        self.vdos_welch_segment.setValue(1.0)
        self.vdos_welch_segment.setToolTip("Welch only: segment duration in picoseconds")
        self.vdos_welch_segment.setEnabled(False)
        form.addRow("Welch segment (ps):", self.vdos_welch_segment)

        self.vdos_welch_overlap = NoWheelDoubleSpinBox()
        self.vdos_welch_overlap.setRange(0.0, 0.99)
        self.vdos_welch_overlap.setSingleStep(0.05)
        self.vdos_welch_overlap.setValue(0.5)
        self.vdos_welch_overlap.setToolTip("Welch only: fractional segment overlap, in [0.0, 1.0)")
        self.vdos_welch_overlap.setEnabled(False)
        form.addRow("Welch overlap:", self.vdos_welch_overlap)

        self.vdos_delta_t = NoWheelDoubleSpinBox()
        self.vdos_delta_t.setDecimals(4)
        self.vdos_delta_t.setRange(1e-4, 1e6)
        self.vdos_delta_t.setValue(0.25)
        self.vdos_delta_t.setToolTip("MD timestep in femtoseconds")
        form.addRow("Delta t (fs):", self.vdos_delta_t)

        self.vdos_force_numerical = QCheckBox("Force numerical")
        self.vdos_force_numerical.setToolTip(
            "Recalculate velocities from positions even if the trajectory stores them"
        )
        form.addRow("", self.vdos_force_numerical)

        self.vdos_quantum_correction = QCheckBox("Quantum correction")
        self.vdos_quantum_correction.setChecked(False)
        self.vdos_quantum_correction.setToolTip(
            "OFF by default. A density of states is a property of the vibrational modes,\n"
            "not of their thermal population, so the classical VACF spectrum already IS the\n"
            "VDOS. Enabling this applies (x/2)coth(x/2) and turns the output into a quantum\n"
            "kinetic-energy-weighted vibrational spectrum, which is NOT a density of states."
        )
        form.addRow("", self.vdos_quantum_correction)

        self.vdos_nskip = NoWheelSpinBox()
        self.vdos_nskip.setRange(0, 10000000)
        self.vdos_nskip.setToolTip("Number of frames to skip at the start of the trajectory")
        form.addRow("N skip:", self.vdos_nskip)

        self.vdos_nmeasure = NoWheelSpinBox()
        self.vdos_nmeasure.setRange(1, 10000000)
        self.vdos_nmeasure.setValue(1)
        self.vdos_nmeasure.setToolTip("Downsampling factor: use every nmeasure-th frame")
        form.addRow("N measure:", self.vdos_nmeasure)

        self.vdos_masses = QLineEdit()
        self.vdos_masses.setPlaceholderText("e.g. C 12.011; H 1.008")
        self.vdos_masses.setText("C 12.011; H 1.008")
        self.vdos_masses.setToolTip(
            "Atom masses: 'Element mass_amu' pairs, separated by semicolons.\n"
            "An element present in the trajectory but missing here is a hard error."
        )
        form.addRow("Masses:", self.vdos_masses)
        apply_validator(self.vdos_masses, MassChargeListValidator())

        self.vdos_pbc = QCheckBox("PBC")
        self.vdos_pbc.setChecked(True)
        form.addRow(self.vdos_pbc)

        self.vdos_com_correction = QCheckBox("Center of mass correction")
        self.vdos_com_correction.setChecked(True)
        self.vdos_com_correction.setToolTip("Subtract center-of-mass motion before computing VDOS")
        form.addRow(self.vdos_com_correction)

        self.vdos_temp = NoWheelDoubleSpinBox()
        self.vdos_temp.setRange(0.001, 100000.0)
        self.vdos_temp.setValue(300.0)
        self.vdos_temp.setToolTip(
            "Simulation temperature in Kelvin. Used by the quantum correction and reported\n"
            "against the sum-rule effective temperature."
        )
        form.addRow("Temperature (K):", self.vdos_temp)

        self.vdos_plot_min = NoWheelDoubleSpinBox()
        self.vdos_plot_min.setRange(0.0, 1e6)
        self.vdos_plot_min.setValue(50.0)
        self.vdos_plot_min.setToolTip("Display only: lower plot limit (cm-1). The saved data is always raw.")
        form.addRow("Plot min (cm-1):", self.vdos_plot_min)

        self.vdos_plot_max = NoWheelDoubleSpinBox()
        self.vdos_plot_max.setRange(0.0, 1e6)
        self.vdos_plot_max.setValue(4500.0)
        self.vdos_plot_max.setToolTip("Display only: upper plot limit (cm-1). Warns if above Nyquist.")
        form.addRow("Plot max (cm-1):", self.vdos_plot_max)

        layout.addWidget(group)

        nc_group = QGroupBox("NetCDF input only")
        nc_form = QFormLayout(nc_group)

        self.vdos_velocity_unit = NoWheelComboBox()
        self.vdos_velocity_unit.addItems(["angstrom/ps", "angstrom/fs", "angstrom/s"])
        self.vdos_velocity_unit.setToolTip(
            "Units of the velocities stored in a NetCDF file (AMBER convention is angstrom/ps).\n"
            "A wrong choice shows up immediately as an absurd effective temperature."
        )
        nc_form.addRow("Velocity unit:", self.vdos_velocity_unit)

        self.vdos_symbols = QLineEdit()
        self.vdos_symbols.setPlaceholderText("e.g. C C H H H (leave empty for xyz input)")
        self.vdos_symbols.setToolTip(
            "Chemical symbols, one per atom in trajectory order.\n"
            "Required for NetCDF, which carries no element information."
        )
        nc_form.addRow("Symbols:", self.vdos_symbols)

        layout.addWidget(nc_group)

        self.vdos_calc_btn = make_primary_button("Calculate")
        self.vdos_stop_btn = make_stop_button()
        self.vdos_calc_btn.clicked.connect(self._run_vdos)
        self.vdos_stop_btn.clicked.connect(self._stop_calculation)
        add_run_row(layout, self.vdos_calc_btn, self.vdos_stop_btn)
        layout.addStretch()
        self.tabs.addTab(tab, "VDOS")

    def _create_ir_tab(self) -> None:
        """Creates the 'IR' (Infrared) analysis tab."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll)

        group = QGroupBox("IR Parameters")
        form = QFormLayout(group)

        self.ir_input = QLineEdit()
        btn_in = make_action_button("Browse")
        btn_in.clicked.connect(lambda: browse_file(self, self.ir_input, _IR_FILTER))
        row_in = QHBoxLayout()
        row_in.addWidget(self.ir_input)
        row_in.addWidget(btn_in)
        form.addRow("Input (xyz):", row_in)

        self.ir_output = QLineEdit()
        btn_out = make_action_button("Browse")
        btn_out.clicked.connect(lambda: create_file(self, self.ir_output, _TXT_FILTER))
        row_out = QHBoxLayout()
        row_out.addWidget(self.ir_output)
        row_out.addWidget(btn_out)
        form.addRow("Output (txt):", row_out)

        self.ir_out_dir = QLineEdit()
        btn_out_dir = make_action_button("Browse")
        btn_out_dir.clicked.connect(lambda: browse_directory(self, self.ir_out_dir))
        row_out_dir = QHBoxLayout()
        row_out_dir.addWidget(self.ir_out_dir)
        row_out_dir.addWidget(btn_out_dir)
        form.addRow("Output directory:", row_out_dir)

        self.ir_estimator = NoWheelComboBox()
        self.ir_estimator.addItems(_ESTIMATORS)
        self.ir_estimator.setToolTip(
            "Spectral estimator:\n"
            "  'acf'   = Blackman-Tukey (biased vector current ACF + decaying lag window)\n"
            "  'welch' = averaged modified periodogram\n"
            "Both return a one-sided PSD on the same scale; running both is a good cross-check."
        )
        self.ir_estimator.currentIndexChanged.connect(self._ir_estimator_index_changed)
        form.addRow("Spectral estimator:", self.ir_estimator)

        self.ir_window = NoWheelComboBox()
        self.ir_window.addItems(_WINDOW_KINDS)
        self.ir_window.currentIndexChanged.connect(self._ir_window_index_changed)
        self.ir_window.setToolTip(
            "Window function applied before the transform.\n"
            "Gaussian is recommended: it is the only one whose spectral window is\n"
            "non-negative, which guarantees a non-negative spectrum."
        )
        form.addRow("Window kind:", self.ir_window)

        self.ir_width = NoWheelDoubleSpinBox()
        self.ir_width.setRange(0.001, 1e6)
        self.ir_width.setValue(1.0)
        self.ir_width.setToolTip(
            "FWHM of the equivalent symmetric lag window, in picoseconds ('acf' + Gaussian).\n"
            "Sets the spectral resolution: ~29 cm-1 for 1.0 ps."
        )
        form.addRow("Window width (FWHM, ps):", self.ir_width)

        self.ir_welch_segment = NoWheelDoubleSpinBox()
        self.ir_welch_segment.setRange(0.001, 1e6)
        self.ir_welch_segment.setValue(1.0)
        self.ir_welch_segment.setToolTip("Welch only: segment duration in picoseconds")
        self.ir_welch_segment.setEnabled(False)
        form.addRow("Welch segment (ps):", self.ir_welch_segment)

        self.ir_welch_overlap = NoWheelDoubleSpinBox()
        self.ir_welch_overlap.setRange(0.0, 0.99)
        self.ir_welch_overlap.setSingleStep(0.05)
        self.ir_welch_overlap.setValue(0.5)
        self.ir_welch_overlap.setToolTip("Welch only: fractional segment overlap, in [0.0, 1.0)")
        self.ir_welch_overlap.setEnabled(False)
        form.addRow("Welch overlap:", self.ir_welch_overlap)

        self.ir_delta_t = NoWheelDoubleSpinBox()
        self.ir_delta_t.setDecimals(4)
        self.ir_delta_t.setRange(1e-4, 1e6)
        self.ir_delta_t.setValue(0.25)
        self.ir_delta_t.setToolTip("MD timestep in femtoseconds")
        form.addRow("Delta t (fs):", self.ir_delta_t)

        self.ir_nskip = NoWheelSpinBox()
        self.ir_nskip.setRange(0, 10000000)
        self.ir_nskip.setToolTip("Number of initial frames to skip")
        form.addRow("N skip:", self.ir_nskip)
        self.ir_nmeasure = NoWheelSpinBox()
        self.ir_nmeasure.setRange(1, 10000000)
        self.ir_nmeasure.setValue(1)
        self.ir_nmeasure.setToolTip("Downsampling factor: use every nmeasure-th frame")
        form.addRow("N measure:", self.ir_nmeasure)

        self.ir_charge_mode = NoWheelComboBox()
        self.ir_charge_mode.addItems(["static", "dynamic"])
        self.ir_charge_mode.setToolTip(
            "'static'  = constant charges, J = sum_i q_i v_i\n"
            "'dynamic' = charges read per frame from a LAMMPS dump 'q' column"
        )
        self.ir_charge_mode.currentIndexChanged.connect(self._ir_charge_mode_index_changed)
        form.addRow("Charge mode:", self.ir_charge_mode)

        self.ir_charges = QLineEdit()
        self.ir_charges.setPlaceholderText("C -0.117; H 0.06")
        self.ir_charges.setText("C -0.117; H 0.06")
        self.ir_charges.setToolTip(
            "Partial charges: 'Element charge_e' pairs, separated by semicolons.\n"
            "They must sum to zero for the composition in the trajectory and should come\n"
            "from the force field that produced the MD. Every element must appear here."
        )
        form.addRow("Static charges:", self.ir_charges)
        apply_validator(self.ir_charges, MassChargeListValidator())

        self.ir_charge_tol = NoWheelDoubleSpinBox()
        self.ir_charge_tol.setDecimals(8)
        self.ir_charge_tol.setRange(0.0, 1.0)
        self.ir_charge_tol.setSingleStep(1e-6)
        self.ir_charge_tol.setValue(1e-6)
        self.ir_charge_tol.setToolTip("Maximum allowed |sum_i q_i|, in units of e")
        form.addRow("Charge tolerance (e):", self.ir_charge_tol)

        self.ir_charge_tol_fatal = QCheckBox("Abort if charge tolerance exceeded")
        self.ir_charge_tol_fatal.setToolTip("Abort instead of warning when the tolerance is exceeded")
        form.addRow("", self.ir_charge_tol_fatal)

        self.ir_dqdt_window = NoWheelSpinBox()
        self.ir_dqdt_window.setRange(5, 999999)
        self.ir_dqdt_window.setSingleStep(2)
        self.ir_dqdt_window.setValue(5)
        self.ir_dqdt_window.setToolTip(
            "Dynamic mode only: Savitzky-Golay window (odd, >= 5) used for dq/dt"
        )
        self.ir_dqdt_window.setEnabled(False)
        form.addRow("dq/dt SavGol window:", self.ir_dqdt_window)

        self.ir_masses = QLineEdit()
        self.ir_masses.setPlaceholderText("C 12.011; H 1.008")
        self.ir_masses.setText("C 12.011; H 1.008")
        self.ir_masses.setToolTip(
            "Atom masses: 'Element mass_amu' pairs, separated by semicolons.\n"
            "An element present in the trajectory but missing here is a hard error."
        )
        form.addRow("Masses:", self.ir_masses)
        apply_validator(self.ir_masses, MassChargeListValidator())

        self.ir_temp = NoWheelDoubleSpinBox()
        self.ir_temp.setRange(0.001, 100000.0)
        self.ir_temp.setValue(300.0)
        self.ir_temp.setToolTip("Simulation temperature in Kelvin (used by the quantum correction)")
        form.addRow("Temperature (K):", self.ir_temp)

        self.ir_pbc = QCheckBox("PBC")
        self.ir_pbc.setChecked(True)
        form.addRow("", self.ir_pbc)
        self.ir_com = QCheckBox("Center of mass correction")
        self.ir_com.setChecked(True)
        form.addRow("", self.ir_com)
        self.ir_quantum_correction = QCheckBox("Quantum correction")
        self.ir_quantum_correction.setChecked(True)
        self.ir_quantum_correction.setToolTip(
            "Harmonic quantum correction factor x/(1-exp(-x)), applied exactly once"
        )
        form.addRow("", self.ir_quantum_correction)

        self.ir_plot_min = NoWheelDoubleSpinBox()
        self.ir_plot_min.setRange(0.0, 1e6)
        self.ir_plot_min.setValue(50.0)
        self.ir_plot_min.setToolTip("Display only: lower plot limit (cm-1). The saved data is always raw.")
        form.addRow("Plot min (cm-1):", self.ir_plot_min)

        self.ir_plot_max = NoWheelDoubleSpinBox()
        self.ir_plot_max.setRange(0.0, 1e6)
        self.ir_plot_max.setValue(4500.0)
        self.ir_plot_max.setToolTip("Display only: upper plot limit (cm-1). Warns if above Nyquist.")
        form.addRow("Plot max (cm-1):", self.ir_plot_max)

        layout.addWidget(group)

        self.ir_calc_btn = make_primary_button("Calculate")
        self.ir_stop_btn = make_stop_button()
        self.ir_calc_btn.clicked.connect(self._run_ir)
        self.ir_stop_btn.clicked.connect(self._stop_calculation)
        add_run_row(layout, self.ir_calc_btn, self.ir_stop_btn)
        layout.addStretch()
        self.tabs.addTab(tab, "IR")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _run_vdos(self) -> None:
        """Validate inputs, write parameter file, and launch the VDOS script."""
        self.vdos_viz.clear()
        input_path = self.vdos_input.text().strip()
        out_dir_path = self.vdos_out_dir.text().strip()

        if not self._validate_input_file(input_path):
            return
        if not self._validate_output_dir(out_dir_path):
            return
        if self.vdos_plot_min.value() >= self.vdos_plot_max.value():
            self.logger.log_message("ERROR", "Plot min wavenumber must be smaller than plot max."); return
        if not input_path.lower().endswith(".xyz") and not self.vdos_symbols.text().strip():
            self.logger.log_message(
                "ERROR", "NetCDF input carries no element information: please provide the "
                         "chemical symbols, one per atom in trajectory order."); return

        vdos_params = {
            "input_file": input_path, "output_data": "VDOS.txt",
            "output_plot": "VDOS.png", "dpi": 150,
            "mode": self.vdos_mode.currentText(),
            "bond_indices": self.vdos_bond_indices.text(),
            "delta_t": self.vdos_delta_t.value(),
            "spectral_estimator": self.vdos_estimator.currentText(),
            "window_kind": self.vdos_window.currentText(),
            "window_width_ps": self.vdos_window_width.value(),
            "welch_segment_ps": self.vdos_welch_segment.value(),
            "welch_overlap": self.vdos_welch_overlap.value(),
            "force_numerical": str(self.vdos_force_numerical.isChecked()),
            "temperature": self.vdos_temp.value(),
            "quantum_correction": str(self.vdos_quantum_correction.isChecked()),
            "PBC": str(self.vdos_pbc.isChecked()),
            "Center_of_mass_correction": str(self.vdos_com_correction.isChecked()),
            "masses": self.vdos_masses.text(),
            "nmeasure": self.vdos_nmeasure.value(),
            "nskip": self.vdos_nskip.value(),
            "plot_min_wavenumber": self.vdos_plot_min.value(),
            "plot_max_wavenumber": self.vdos_plot_max.value(),
            "velocity_unit": self.vdos_velocity_unit.currentText(),
        }
        if self.vdos_symbols.text().strip():
            vdos_params["symbols"] = self.vdos_symbols.text().strip()

        param_file_path = os.path.join(out_dir_path, "vdos_input.txt")
        self._write_and_run(
            param_file_path, vdos_params,
            ("Vibrational_Analysis", "vdos.py"),
            out_dir_path,
            self._make_finish_callback("VDOS calculation", self._plot_vdos),
        )

    def _run_ir(self) -> None:
        """Validate inputs, write parameter file, and launch the IR script."""
        self.ir_viz.clear()
        input_path = self.ir_input.text().strip()
        output_path = self.ir_output.text().strip()
        out_dir_path = self.ir_out_dir.text().strip()

        if not self._validate_input_file(input_path):
            return
        if not output_path:
            self.logger.log_message("ERROR", "Please choose an output file first."); return
        if not self._validate_output_dir(out_dir_path):
            return
        if self.ir_plot_min.value() >= self.ir_plot_max.value():
            self.logger.log_message("ERROR", "Plot min wavenumber must be smaller than plot max."); return

        charge_mode = self.ir_charge_mode.currentText()
        if charge_mode == "static" and not self.ir_charges.text().strip():
            self.logger.log_message(
                "ERROR", "Static charge mode requires a charge for every element in the "
                         "trajectory."); return

        output_path = self._resolve_to_outdir(self.ir_output, out_dir_path)

        ir_params = {
            "input_file": input_path, "ir_output": output_path,
            "ir_plot": "ir_spectrum.png",
            "spectral_estimator": self.ir_estimator.currentText(),
            "window_kind": self.ir_window.currentText(),
            "window_width_ps": self.ir_width.value(),
            "welch_segment_ps": self.ir_welch_segment.value(),
            "welch_overlap": self.ir_welch_overlap.value(), "dpi": 300,
            "PBC": str(self.ir_pbc.isChecked()),
            "Center_of_mass_correction": str(self.ir_com.isChecked()),
            "temperature": self.ir_temp.value(), "delta_t": self.ir_delta_t.value(),
            "nskip": self.ir_nskip.value(), "nmeasure": self.ir_nmeasure.value(),
            "charge_mode": charge_mode,
            "charge_tolerance": self.ir_charge_tol.value(),
            "charge_tolerance_fatal": str(self.ir_charge_tol_fatal.isChecked()),
            "dqdt_savgol_window": self.ir_dqdt_window.value(),
            "masses": self.ir_masses.text(),
            "quantum_correction": str(self.ir_quantum_correction.isChecked()),
            "plot_min_wavenumber": self.ir_plot_min.value(),
            "plot_max_wavenumber": self.ir_plot_max.value(),
        }
        if charge_mode == "static":
            ir_params["static_charges"] = self.ir_charges.text()

        param_file_path = os.path.join(out_dir_path, "ir_input.txt")
        self._write_and_run(
            param_file_path, ir_params,
            ("Vibrational_Analysis", "IR.py"),
            out_dir_path,
            self._make_finish_callback("IR calculation", self._plot_ir),
        )

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def _plot_vdos(self) -> None:
        try:
            out_dir_path = self.vdos_out_dir.text().strip()
            def plot_func(fig):
                fig.clear(); fig.patch.set_facecolor("white"); ax = fig.subplots()
                data = np.loadtxt(os.path.join(out_dir_path, "VDOS.txt"))
                wavenumber, intensity = data[:, 0], data[:, 1]
                # The file holds the raw spectrum up to Nyquist: cut it to the plot range
                # so the y-scaling follows the displayed data.
                in_range = ((wavenumber >= self.vdos_plot_min.value())
                            & (wavenumber <= self.vdos_plot_max.value()))
                wavenumber, intensity = wavenumber[in_range], intensity[in_range]
                ax.plot(wavenumber, intensity, linewidth=2.0, color="royalblue")
                ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
                ax.set_ylabel("Intensity (a.u.)", fontsize=12)
                ax.set_title("Quantum KE-Weighted Vibrational Spectrum"
                             if self.vdos_quantum_correction.isChecked()
                             else "Vibrational Density of States (VDOS)",
                             fontsize=14, fontweight="bold")
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.yaxis.offsetText.set_fontsize(12)
                ax.set_ylim(bottom=0)
                ax.set_xlim(left=self.vdos_plot_min.value(), right=self.vdos_plot_max.value())
                ax.grid(True, linestyle="--", alpha=0.3)
                ax.text(0.02, 0.98, f"Points: {len(wavenumber):,}", transform=ax.transAxes,
                        va="top", fontsize=10, color="black",
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", linewidth=1))
                style_axis_light(ax); fig.tight_layout()
            self.vdos_viz.plot(plot_func)
        except Exception as e:
            self.logger.log_message("ERROR", f"Error plotting VDOS: {e}")

    def _plot_ir(self) -> None:
        try:
            def plot_func(fig):
                fig.clear(); fig.patch.set_facecolor("white"); ax = fig.subplots()
                data = np.loadtxt(self.ir_output.text().strip())
                wavenumber, intensity = data[:, 0], data[:, 1]
                bin_cm = wavenumber[1] - wavenumber[0]
                # Cut the raw (up to Nyquist) spectrum to the plot range before the
                # y-scaling and before the 10 % peak threshold.
                in_range = ((wavenumber >= self.ir_plot_min.value())
                            & (wavenumber <= self.ir_plot_max.value()))
                wavenumber, intensity = wavenumber[in_range], intensity[in_range]
                ax.plot(wavenumber, intensity, linewidth=2.0, color="royalblue")
                peaks, _ = find_peaks(intensity, height=np.max(intensity) * 0.1,
                                      distance=max(1, int(round(_PEAK_MIN_SEP_CM / bin_cm))))
                for idx in peaks:
                    ax.axvline(x=wavenumber[idx], color="red", alpha=0.3, linestyle="--")
                    ax.text(wavenumber[idx], np.max(intensity) * 0.1,
                            f"{wavenumber[idx]:.0f}", rotation=90, va="bottom", fontsize=8, color="red")
                ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=12)
                ax.set_ylabel("Intensity (a.u.)", fontsize=12)
                ax.set_title("IR Absorption Spectrum", fontsize=14, fontweight="bold")
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.yaxis.offsetText.set_fontsize(12)
                ax.set_ylim(bottom=0)
                ax.set_xlim(left=self.ir_plot_min.value(), right=self.ir_plot_max.value())
                ax.grid(True, linestyle="--", alpha=0.3)
                annotation = (
                    f"Estimator: {self.ir_estimator.currentText()}, "
                    f"charges: {self.ir_charge_mode.currentText()}\n"
                    f"PBC: {'yes' if self.ir_pbc.isChecked() else 'no'}\n"
                    f"COM Correction: {'yes' if self.ir_com.isChecked() else 'no'}\n"
                    f"T = {self.ir_temp.value()} K\n"
                    f"Δt = {self.ir_delta_t.value()} fs, nmeasure = {self.ir_nmeasure.value()}"
                )
                ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", fontsize=10, color="black",
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", linewidth=1, alpha=0.8))
                style_axis_light(ax); fig.tight_layout()
            self.ir_viz.plot(plot_func)
        except Exception as e:
            self.logger.log_message("ERROR", f"Error plotting IR spectrum: {e}")

    def _vdos_mode_index_changed(self, index):
        self.vdos_bond_indices.setEnabled(index != 0)

    def _vdos_window_index_changed(self, index):
        self._vdos_sync_estimator_state()

    def _vdos_estimator_index_changed(self, index):
        self._vdos_sync_estimator_state()

    def _vdos_sync_estimator_state(self):
        """window_width_ps only applies to 'acf' + Gaussian; welch_* only to 'welch'."""
        is_welch = self.vdos_estimator.currentText() == "welch"
        is_gaussian = self.vdos_window.currentIndex() == 0
        self.vdos_window_width.setEnabled(is_gaussian and not is_welch)
        self.vdos_welch_segment.setEnabled(is_welch)
        self.vdos_welch_overlap.setEnabled(is_welch)

    def _ir_window_index_changed(self, index):
        self._ir_sync_estimator_state()

    def _ir_estimator_index_changed(self, index):
        self._ir_sync_estimator_state()

    def _ir_sync_estimator_state(self):
        """window_width_ps only applies to 'acf' + Gaussian; welch_* only to 'welch'."""
        is_welch = self.ir_estimator.currentText() == "welch"
        is_gaussian = self.ir_window.currentIndex() == 0
        self.ir_width.setEnabled(is_gaussian and not is_welch)
        self.ir_welch_segment.setEnabled(is_welch)
        self.ir_welch_overlap.setEnabled(is_welch)

    def _ir_charge_mode_index_changed(self, index):
        is_dynamic = self.ir_charge_mode.currentText() == "dynamic"
        self.ir_charges.setEnabled(not is_dynamic)
        self.ir_dqdt_window.setEnabled(is_dynamic)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = VibrationalAnalysisWindow()
    window.show()
    sys.exit(app.exec())
