# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
X-ray Analysis (XRD) GUI Application.

Provides tabs for Kinematical XRD, Debye Scattering, and Reciprocal Sum.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from matplotlib.ticker import ScalarFormatter

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit,
    QGroupBox, QCheckBox,
    QTabWidget, QFormLayout,
    QScrollArea, QStackedWidget,
)
from PySide6.QtCore import Signal

from VCL_utils.theme import (apply_dark_theme, make_action_button, make_primary_button,
                             make_stop_button, add_run_row, CollapsibleGroup,
                             NoWheelComboBox, NoWheelSpinBox, NoWheelDoubleSpinBox)
from VCL_utils.general import browse_file, create_file, browse_directory
from VCL_utils.validators import (
    apply_validator,
    ElementListValidator,
    IntTripletValidator,
    FloatTripletValidator,
    FloatPairValidator,
    DebyeWallerValidator,
    FilenameValidator,
    ScientificFloatValidator,
)
from VCL_utils.visualization import MatplotlibWidget, style_axis_light
from VCL_utils.base_window import BaseModuleWindow

_XYZ_FILTER = "XYZ files (*.xyz);;All files (*)"
_CELL_FILTER = ("Files carrying a cell (*.xyz *.cif *.data *.lmp *.dump POSCAR CONTCAR);;"
                "All files (*)")
_TXT_FILTER = "Text files (*.txt *.csv *.dat);;All files (*)"
_PNG_FILTER = "PNG images (*.png);;All images (*.png *.jpg *.svg);;All files (*)"

# Named radiations shared by XRD-Kinematical and XRD-Debye_Scattering.
# 'CuKa' is the Ka1/Ka2 weighted average (1.54184 A); 'CuKa1' is the single line.
_RADIATIONS = [
    "CuKa", "CuKa1", "CuKa2", "CuKb1",
    "MoKa", "MoKa1", "MoKa2", "MoKb1",
    "CrKa", "CrKa1", "CrKa2", "CrKb1",
    "FeKa", "FeKa1", "FeKa2", "FeKb1",
    "CoKa", "CoKa1", "CoKa2", "CoKb1",
    "AgKa", "AgKa1", "AgKa2", "AgKb1",
]

# Absorption geometries accepted by XRD-Kinematical.ABSORPTION_GEOMETRIES
_ABSORPTION_GEOMETRIES = [
    "cylinder", "debye_scherrer", "bragg_brentano_reflection",
    "symmetric_transmission", "slab_attenuation_approx",
]


def _yn(flag: bool) -> str:
    """Render a checkbox state the way the input parsers expect."""
    return "yes" if flag else "no"


def _set_apply_flags(params: dict, flags) -> None:
    """Write each apply_* key only when its checkbox is ticked.

    An omitted key keeps the script default (off) and lets experimental_correction
    switch it on; an explicit value in the input file always wins.
    """
    for key, box in flags:
        if box.isChecked():
            params[key] = "yes"


class AnomalousEntryRow(QWidget):
    """One anomalous entry: element symbol + f' + f'' + remove button."""
    remove_requested = Signal(QWidget)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.elem = QLineEdit()
        self.elem.setPlaceholderText("e.g. Cu")
        self.elem.setToolTip("Chemical element symbol (e.g. Cu, Fe, Zn)")
        apply_validator(self.elem, ElementListValidator())
        self.fprime = NoWheelDoubleSpinBox()
        self.fprime.setRange(-1e9, 1e9)
        self.fprime.setDecimals(4)
        self.fprime.setToolTip("Real part of the anomalous scattering correction f′")
        self.fdouble = NoWheelDoubleSpinBox()
        self.fdouble.setRange(-1e9, 1e9)
        self.fdouble.setDecimals(4)
        self.fdouble.setToolTip("Imaginary part of the anomalous scattering correction f″")
        remove_btn = make_action_button("Remove")
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Element:"))
        layout.addWidget(self.elem, 2)
        layout.addWidget(QLabel("f′:"))
        layout.addWidget(self.fprime, 1)
        layout.addWidget(QLabel("f″:"))
        layout.addWidget(self.fdouble, 1)
        layout.addWidget(remove_btn)

    def data(self) -> dict:
        return {"element": self.elem.text().strip(),
                "fprime": float(self.fprime.value()),
                "fdouble": float(self.fdouble.value())}


class XrayWindow(BaseModuleWindow):
    """Main Window for X-ray Analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("XRD Analysis")
        self.resize(1000, 800)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        header = QLabel("XRD Analysis")
        header.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(header)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Stored list of anomalous rows (avoids fragile index-scan pattern)
        self._anom_rows: list[AnomalousEntryRow] = []

        self._create_kin_tab()
        self._create_debye_tab()
        self._create_reciprocal_tab()
        self.viz_stack = QStackedWidget()
        self.kin_viz = MatplotlibWidget()
        self.viz_stack.addWidget(self.kin_viz)
        self.debye_viz = MatplotlibWidget()
        self.viz_stack.addWidget(self.debye_viz)
        self.reciprocal_viz = MatplotlibWidget()
        self.viz_stack.addWidget(self.reciprocal_viz)
        self.tabs.currentChanged.connect(self.viz_stack.setCurrentIndex)

        # Register buttons
        self._calc_btns = [self.kin_calc_btn, self.debye_calc_btn, self.reciprocal_calc_btn]
        self._stop_btns = [self.kin_stop_btn, self.debye_stop_btn, self.reciprocal_stop_btn]

        # Public interface for PolyCycleMainWindow
        self.control_widget = self.tabs
        self.viz_widget = self.viz_stack

    # ------------------------------------------------------------------ tab builders

    def _create_kin_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)

        input_group = QGroupBox("Input")
        input_form = QFormLayout(input_group)
        self.kin_input = QLineEdit()
        browse_btn = make_action_button("Browse")
        browse_btn.clicked.connect(lambda: browse_file(self, self.kin_input, _XYZ_FILTER))
        input_row = QHBoxLayout()
        input_row.addWidget(self.kin_input)
        input_row.addWidget(browse_btn)
        input_form.addRow("Input (.xyz):", input_row)

        self.kin_out_dir = QLineEdit()
        btn_out_dir = make_action_button("Browse")
        btn_out_dir.clicked.connect(lambda: browse_directory(self, self.kin_out_dir))
        out_dir_row = QHBoxLayout()
        out_dir_row.addWidget(self.kin_out_dir)
        out_dir_row.addWidget(btn_out_dir)
        input_form.addRow("Output directory:", out_dir_row)

        self.kin_species_mode = NoWheelComboBox()
        self.kin_species_mode.addItems(["chemical_symbols", "atomic_numbers", "lammps_types"])
        self.kin_species_mode.setToolTip(
            "How the species column is read. A bare number is an atomic number with "
            "'atomic_numbers' and a LAMMPS atom type with 'lammps_types' (needs a type map)")
        input_form.addRow("Species mode:", self.kin_species_mode)

        self.kin_type_map = QLineEdit()
        self.kin_type_map.setPlaceholderText("1:Cu,2:Zn")
        self.kin_type_map.setToolTip("Required for species_mode = lammps_types, e.g. 1:Cu,2:Zn")
        input_form.addRow("Type map:", self.kin_type_map)

        self.kin_coordinate_mode = NoWheelComboBox()
        self.kin_coordinate_mode.addItems(["cartesian", "fractional"])
        self.kin_coordinate_mode.setToolTip("What the coordinates in the structure file are")
        input_form.addRow("Coordinate mode:", self.kin_coordinate_mode)

        self.kin_lattice_mode = NoWheelComboBox()
        self.kin_lattice_mode.addItems(["auto", "simulation_box", "unit_cell"])
        self.kin_lattice_mode.setToolTip(
            "Which lattice pymatgen is given.\n"
            "auto: try each cell source in turn - the box in the file (extended-XYZ "
            "Lattice=), a cell ASE can read from it, the Cell file, an integer "
            "multiple of the unit cell below, then inference from the coordinates. "
            "If none succeeds the structure is not a periodic crystal and the run "
            "stops with a pointer to XRD-Debye_Scattering\n"
            "simulation_box: the periodic box is the lattice; every atom is kept and "
            "nothing is folded, so vacancies, strain and extended defects are computed\n"
            "unit_cell: the file holds one cell or a perfect aligned supercell; fold "
            "into the cell and merge the images back onto their sites")
        input_form.addRow("Lattice mode:", self.kin_lattice_mode)

        self.experimental_correction = QCheckBox("Experimental correction")
        self.experimental_correction.setChecked(True)
        self.experimental_correction.setToolTip(
            "Convenience switch: enables Debye-Waller and instrumental broadening only where "
            "the corresponding data is supplied. It never enables absorption")
        self.wrap_coords = QCheckBox("Wrap coords")
        self.wrap_coords.setChecked(True)
        self.wrap_coords.setToolTip("Fold fractional coordinates into [0,1) before the calculation")
        input_form.addRow(self.experimental_correction, self.wrap_coords)

        self.kin_check_duplicates = QCheckBox("Merge folded images")
        self.kin_check_duplicates.setChecked(True)
        self.kin_check_duplicates.setToolTip(
            "In unit_cell mode, merge the images that folding stacked onto one "
            "crystallographic site.\nWithout this the cell keeps every atom of the "
            "supercell in the volume of a single cell")
        self.kin_collapse_tol = NoWheelDoubleSpinBox()
        self.kin_collapse_tol.setRange(0.0, 100.0)
        self.kin_collapse_tol.setDecimals(4)
        self.kin_collapse_tol.setValue(0.25)
        self.kin_collapse_tol.setToolTip(
            "Radius within which folded images count as the same site (Å). It has to "
            "exceed the thermal displacement, so it is well above the old 0.05 Å "
            "duplicate-site tolerance")
        input_form.addRow(self.kin_check_duplicates, self.kin_collapse_tol)

        self.kin_validate_proximity = QCheckBox("Reject overlapping sites")
        self.kin_validate_proximity.setChecked(True)
        self.kin_validate_proximity.setToolTip(
            "pymatgen's Structure.DISTANCE_TOLERANCE check: refuse a cell whose "
            "sites are closer than 0.5 Å,\nwhich is not a physical structure")
        self.kin_surface_tol = NoWheelDoubleSpinBox()
        self.kin_surface_tol.setRange(0.0, 1.0)
        self.kin_surface_tol.setDecimals(4)
        self.kin_surface_tol.setSingleStep(0.01)
        self.kin_surface_tol.setValue(0.05)
        self.kin_surface_tol.setToolTip(
            "A candidate box is accepted when it leaves fewer than this fraction of "
            "atoms under-coordinated.\nReal periodic cells land near 0; finite "
            "clusters stay above 0.3")
        input_form.addRow(self.kin_validate_proximity, self.kin_surface_tol)

        self.kin_max_proximity_check_atoms = NoWheelSpinBox()
        self.kin_max_proximity_check_atoms.setRange(0, 100_000_000)
        self.kin_max_proximity_check_atoms.setGroupSeparatorShown(True)
        self.kin_max_proximity_check_atoms.setValue(20000)
        self.kin_max_proximity_check_atoms.setToolTip(
            "Above this many sites the proximity check is skipped altogether.\n"
            "Between 5000 sites and this limit it runs on a neighbour tree rather "
            "than an N x N distance matrix")
        input_form.addRow("Max atoms for the proximity check:",
                          self.kin_max_proximity_check_atoms)

        self.kin_max_collapse_atoms = NoWheelSpinBox()
        self.kin_max_collapse_atoms.setRange(0, 100_000_000)
        self.kin_max_collapse_atoms.setGroupSeparatorShown(True)
        self.kin_max_collapse_atoms.setValue(20000)
        self.kin_max_collapse_atoms.setToolTip(
            "Upper size for merging folded images, which builds an N x N distance "
            "matrix.\nAbove this the images are left unmerged")
        input_form.addRow("Max atoms for image merging:", self.kin_max_collapse_atoms)

        self.kin_repeats_integer_tol = NoWheelDoubleSpinBox()
        self.kin_repeats_integer_tol.setRange(0.0, 1.0)
        self.kin_repeats_integer_tol.setDecimals(4)
        self.kin_repeats_integer_tol.setSingleStep(0.01)
        self.kin_repeats_integer_tol.setValue(0.01)
        self.kin_repeats_integer_tol.setToolTip(
            "Tolerance on the requirement that the number of images per site be a "
            "whole number")
        input_form.addRow("Images-per-site tolerance:", self.kin_repeats_integer_tol)
        layout.addWidget(input_group)

        cell_group = QGroupBox("Cell Source")
        cell_form = QFormLayout(cell_group)
        cell_group.setToolTip(
            "Where the periodic cell comes from, in the order the routes are tried.\n"
            "If every route fails the structure is not a periodic crystal and the run "
            "stops with a pointer to XRD-Debye_Scattering")
        self.kin_use_ase = QCheckBox("Read the cell with ASE")
        self.kin_use_ase.setChecked(True)
        self.kin_use_ase.setToolTip(
            "Let ASE read a cell from the structure file when it carries no "
            "extended-XYZ Lattice= entry (LAMMPS dump or data, POSCAR, CIF, ...)")
        cell_form.addRow(self.kin_use_ase)

        self.kin_cell_file = QLineEdit()
        self.kin_cell_file.setPlaceholderText("optional - a separate file carrying the box")
        cell_browse_btn = make_action_button("Browse")
        cell_browse_btn.clicked.connect(
            lambda: browse_file(self, self.kin_cell_file, _CELL_FILTER))
        cell_row = QHBoxLayout()
        cell_row.addWidget(self.kin_cell_file)
        cell_row.addWidget(cell_browse_btn)
        self.kin_cell_file.setToolTip(
            "A second file whose box is used for these coordinates, for example the "
            "relaxed data file a dump came from")
        cell_form.addRow("Cell file:", cell_row)

        self.kin_infer_cell = QCheckBox("Infer the cell from the coordinates")
        self.kin_infer_cell.setChecked(True)
        self.kin_infer_cell.setToolTip(
            "Recover the box from the coordinates themselves, at any orientation.\n"
            "This is what handles a cell built on directions such as [111], which "
            "folding into a supplied cell cannot")
        cell_form.addRow(self.kin_infer_cell)

        self.kin_use_cell_params = QCheckBox("Supply unit-cell parameters below")
        self.kin_use_cell_params.setChecked(True)
        self.kin_use_cell_params.setToolTip(
            "Unticked, a, b, c and the angles are not written at all, and the cell "
            "is taken from the file or inferred.\n"
            "Required for lattice_mode = unit_cell and for fractional coordinates")
        cell_form.addRow(self.kin_use_cell_params)
        layout.addWidget(cell_group)

        self.kin_lattice_group = QGroupBox("Lattice Parameters")
        lattice_group = self.kin_lattice_group
        self.kin_use_cell_params.toggled.connect(lattice_group.setEnabled)
        lattice_form = QFormLayout(lattice_group)
        self.a = NoWheelDoubleSpinBox()
        self.a.setRange(0.001, 1e6)
        self.a.setDecimals(4)
        self.a.setValue(2.9206)
        self.a.setToolTip(
            "UNIT-cell parameter a (Å) - the crystallographic cell, never the supercell.\n"
            "A supercell box is read from the file or recovered from this cell; see Lattice mode")
        lattice_form.addRow("a (Å):", self.a)
        self.b = NoWheelDoubleSpinBox()
        self.b.setRange(0.001, 1e6)
        self.b.setDecimals(4)
        self.b.setValue(2.9206)
        self.b.setToolTip(
            "UNIT-cell parameter b (Å) - the crystallographic cell, never the supercell.\n"
            "A supercell box is read from the file or recovered from this cell; see Lattice mode")
        lattice_form.addRow("b (Å):", self.b)
        self.c_lat = NoWheelDoubleSpinBox()
        self.c_lat.setRange(0.001, 1e6)
        self.c_lat.setDecimals(4)
        self.c_lat.setValue(2.9206)
        self.c_lat.setToolTip(
            "UNIT-cell parameter c (Å) - the crystallographic cell, never the supercell.\n"
            "A supercell box is read from the file or recovered from this cell; see Lattice mode")
        lattice_form.addRow("c (Å):", self.c_lat)
        self.alpha = NoWheelDoubleSpinBox()
        self.alpha.setRange(1.0, 179.0)
        self.alpha.setValue(90.0)
        self.alpha.setToolTip("Unit-cell angle α (degrees)")
        lattice_form.addRow("α (°):", self.alpha)
        self.beta = NoWheelDoubleSpinBox()
        self.beta.setRange(1.0, 179.0)
        self.beta.setValue(90.0)
        self.beta.setToolTip("Unit-cell angle β (degrees)")
        lattice_form.addRow("β (°):", self.beta)
        self.gamma = NoWheelDoubleSpinBox()
        self.gamma.setRange(1.0, 179.0)
        self.gamma.setValue(90.0)
        self.gamma.setToolTip("Unit-cell angle γ (degrees)")
        lattice_form.addRow("γ (°):", self.gamma)
        layout.addWidget(lattice_group)

        # Tuning for the cell-inference and axis-periodicity search. Collapsed:
        # the defaults are right for ordinary structures, but nothing should be
        # reachable only by hand-editing an input file.
        adv_group = CollapsibleGroup("Advanced: Cell Inference")
        adv_form = adv_group.form

        def _adv_double(label, value, lo, hi, decimals, step, tip):
            box = NoWheelDoubleSpinBox()
            box.setRange(lo, hi)
            box.setDecimals(decimals)
            box.setSingleStep(step)
            box.setValue(value)
            box.setToolTip(tip)
            adv_form.addRow(label, box)
            return box

        self.kin_cell_inference_tol = _adv_double(
            "Match tolerance (× r₁):", 0.15, 0.0, 1.0, 4, 0.01,
            "How close two atoms must sit, as a fraction of the nearest-neighbour "
            "distance r₁, for a translation to count as mapping one onto the other")
        self.kin_cell_inference_min_score = _adv_double(
            "Min match score:", 0.90, 0.0, 1.0, 3, 0.01,
            "Fraction of atoms a candidate translation must map onto another atom "
            "of the same species before it is accepted as a lattice vector")
        self.kin_cell_inference_reach = _adv_double(
            "Search reach (× r₁):", 1.80, 0.5, 10.0, 3, 0.1,
            "Radius, in units of r₁, within which candidate translation vectors "
            "are generated. Larger values find longer lattice vectors, more slowly")
        self.kin_cell_inference_max_vectors = NoWheelSpinBox()
        self.kin_cell_inference_max_vectors.setRange(3, 1000)
        self.kin_cell_inference_max_vectors.setValue(24)
        self.kin_cell_inference_max_vectors.setToolTip(
            "Stop after this many accepted translation vectors")
        adv_form.addRow("Max translation vectors:", self.kin_cell_inference_max_vectors)
        self.kin_cell_inference_max_residual = _adv_double(
            "Max closure residual (× r₁):", 0.10, 0.0, 1.0, 4, 0.01,
            "How far the inferred cell may be from closing on the coordinates "
            "before it is rejected")
        self.kin_axis_period_margin = _adv_double(
            "Axis period margin (× cutoff):", 1.00, 0.0, 10.0, 3, 0.1,
            "Slab thickness, in units of the first-shell cutoff, used when scanning "
            "an axis for its repeat length")
        self.kin_axis_period_min_score = _adv_double(
            "Axis period min score:", 0.92, 0.0, 1.0, 3, 0.01,
            "Score an axis must reach before it counts as periodic")
        self.kin_max_box_scan_atoms = NoWheelSpinBox()
        self.kin_max_box_scan_atoms.setRange(0, 100_000_000)
        self.kin_max_box_scan_atoms.setGroupSeparatorShown(True)
        self.kin_max_box_scan_atoms.setValue(200000)
        self.kin_max_box_scan_atoms.setToolTip(
            "Upper size for the per-axis box scan. The scan builds a neighbour tree "
            "per trial period, so it gets slow well before this limit")
        adv_form.addRow("Max atoms for the box scan:", self.kin_max_box_scan_atoms)
        layout.addWidget(adv_group)

        xrd_group = QGroupBox("XRD Calculation Parameters")
        xrd_form = QFormLayout(xrd_group)
        self.kin_wavelength = NoWheelComboBox()
        self.kin_wavelength.addItems(_RADIATIONS)
        self.kin_wavelength.setToolTip(
            "X-ray source, resolved by pymatgen. 'CuKa' is the Ka1/Ka2 average (1.54184 Å); "
            "use 'CuKa1' for the single line")
        xrd_form.addRow("Wavelength:", self.kin_wavelength)
        self.kin_two_theta_min = NoWheelDoubleSpinBox()
        self.kin_two_theta_min.setRange(0.0, 180.0)
        self.kin_two_theta_min.setValue(10.0)
        xrd_form.addRow("2θ min (°):", self.kin_two_theta_min)
        self.kin_two_theta_max = NoWheelDoubleSpinBox()
        self.kin_two_theta_max.setRange(0.0, 180.0)
        self.kin_two_theta_max.setValue(90.0)
        xrd_form.addRow("2θ max (°):", self.kin_two_theta_max)

        # The cost is reflections x atoms, which scales as cell volume x atoms:
        # a large MD box is the worst case for a kinematical sum.
        self.kin_reflection_cost_warn = QLineEdit()
        self.kin_reflection_cost_warn.setText("5e7")
        self.kin_reflection_cost_warn.setToolTip(
            "Warn above this many structure-factor terms (reflections × atoms).\n"
            "Leave empty for the module default")
        apply_validator(self.kin_reflection_cost_warn, ScientificFloatValidator())
        xrd_form.addRow("Warn above cost:", self.kin_reflection_cost_warn)

        self.kin_max_reflection_cost = QLineEdit()
        self.kin_max_reflection_cost.setText("1e10")
        self.kin_max_reflection_cost.setToolTip(
            "Refuse the run above this many structure-factor terms.\n"
            "The cost scales as cell volume × atoms, so a large periodic box belongs "
            "in XRD-ReciprocalSum, and a box with free surfaces in XRD-Debye_Scattering.\n"
            "Leave empty for the module default")
        apply_validator(self.kin_max_reflection_cost, ScientificFloatValidator())
        xrd_form.addRow("Refuse above cost:", self.kin_max_reflection_cost)
        self.kin_scale_factor = NoWheelDoubleSpinBox()
        self.kin_scale_factor.setRange(0.0, 1e9)
        self.kin_scale_factor.setDecimals(4)
        self.kin_scale_factor.setValue(1.0)
        xrd_form.addRow("Scale factor:", self.kin_scale_factor)
        self.kin_normalize_mode = NoWheelComboBox()
        self.kin_normalize_mode.addItems(["none", "curve_max_100", "reflection_max_100"])
        self.kin_normalize_mode.setToolTip(
            "none: raw intensities. curve_max_100: scale the curve only. "
            "reflection_max_100: scale the reflection list and the curve by the same factor")
        xrd_form.addRow("Normalize mode:", self.kin_normalize_mode)
        self.kin_normalize_max = NoWheelDoubleSpinBox()
        self.kin_normalize_max.setRange(0.0, 1e9)
        self.kin_normalize_max.setValue(100.0)
        self.kin_normalize_max.setToolTip("Target maximum used by the normalization modes")
        xrd_form.addRow("Normalize max:", self.kin_normalize_max)
        self.kin_float_precision = NoWheelSpinBox()
        self.kin_float_precision.setRange(1, 15)
        self.kin_float_precision.setValue(4)
        xrd_form.addRow("Float precision:", self.kin_float_precision)
        layout.addWidget(xrd_group)

        abs_group = QGroupBox("Absorption Correction")
        abs_form = QFormLayout(abs_group)
        self.apply_abs = QCheckBox("Apply absorption")
        self.apply_abs.setToolTip(
            "Never enabled automatically: it needs a material-specific μ and an explicit geometry")
        abs_form.addRow(self.apply_abs)
        self.lin_abs = NoWheelDoubleSpinBox()
        self.lin_abs.setRange(0.0, 1e6)
        self.lin_abs.setDecimals(3)
        self.lin_abs.setValue(400.0)
        self.lin_abs.setToolTip("Linear absorption coefficient μ (cm⁻¹)")
        abs_form.addRow("Linear absorption coefficient (cm⁻¹):", self.lin_abs)
        self.abs_geo = NoWheelComboBox()
        self.abs_geo.addItems(_ABSORPTION_GEOMETRIES)
        self.abs_geo.setToolTip(
            "cylinder / debye_scherrer: capillary cross-section integral. "
            "bragg_brentano_reflection: flat plate in reflection. "
            "symmetric_transmission: flat plate, beam normal. "
            "slab_attenuation_approx: single-path exp(-μt/sin θ), APPROXIMATE")
        abs_form.addRow("Geometry:", self.abs_geo)
        self.sample_radius = NoWheelDoubleSpinBox()
        self.sample_radius.setRange(0.0, 1e6)
        self.sample_radius.setDecimals(4)
        self.sample_radius.setValue(0.05)
        self.sample_radius.setToolTip("Capillary radius (cm) — cylinder / Debye-Scherrer geometry")
        abs_form.addRow("Capillary radius (cm):", self.sample_radius)
        self.sample_thickness = NoWheelDoubleSpinBox()
        self.sample_thickness.setRange(0.0, 1e6)
        self.sample_thickness.setDecimals(4)
        self.sample_thickness.setValue(0.1)
        self.sample_thickness.setToolTip("Sample thickness (cm) — flat-plate geometries")
        abs_form.addRow("Sample thickness (cm):", self.sample_thickness)
        layout.addWidget(abs_group)

        filter_group = QGroupBox("Display Peak Selection")
        filter_form = QFormLayout(filter_group)
        self.kin_min_intensity = NoWheelDoubleSpinBox()
        self.kin_min_intensity.setRange(0.0, 100.0)
        self.kin_min_intensity.setDecimals(4)
        self.kin_min_intensity.setValue(0.5)
        self.kin_min_intensity.setToolTip(
            "Minimum display intensity as a PERCENT of the strongest reflection. "
            "Labelling and tables only — the continuous curve is never filtered")
        filter_form.addRow("Min intensity (%):", self.kin_min_intensity)
        self.kin_max_peaks = NoWheelSpinBox()
        self.kin_max_peaks.setRange(0, 10000)
        self.kin_max_peaks.setValue(50)
        self.kin_max_peaks.setToolTip("Maximum number of labelled reflections; 0 = all")
        filter_form.addRow("Max peaks:", self.kin_max_peaks)
        self.kin_min_peak_sep = NoWheelDoubleSpinBox()
        self.kin_min_peak_sep.setRange(0.0, 180.0)
        self.kin_min_peak_sep.setDecimals(4)
        self.kin_min_peak_sep.setValue(0.1)
        self.kin_min_peak_sep.setToolTip(
            "Within a cluster closer than this, only the strongest reflection is labelled")
        filter_form.addRow("Min peak separation (°):", self.kin_min_peak_sep)
        layout.addWidget(filter_group)

        dw_group = QGroupBox("Debye-Waller")
        dw_form = QFormLayout(dw_group)
        self.apply_debye_waller = QCheckBox("Apply Debye-Waller")
        self.apply_debye_waller.setChecked(True)
        self.apply_debye_waller.setToolTip("Apply exp(-B s²) to the scattering factors")
        dw_form.addRow(self.apply_debye_waller)
        self.debye_waller_factors = QLineEdit()
        self.debye_waller_factors.setPlaceholderText("Zn=0.5,Cu=0.6")
        self.debye_waller_factors.setToolTip(
            "B_iso per element in Å², e.g. Zn=0.5, Cu=0.6. Species with no entry get B = 0")
        apply_validator(self.debye_waller_factors, DebyeWallerValidator())
        dw_form.addRow("Debye Waller Factors:", self.debye_waller_factors)
        layout.addWidget(dw_group)

        inst_broad_group = QGroupBox("Line Profile")
        inst_broad_form = QFormLayout(inst_broad_group)
        self.apply_inst_broadening = QCheckBox()
        self.apply_inst_broadening.setChecked(True)
        inst_broad_form.addRow("Apply instrumental broadening:", self.apply_inst_broadening)
        self.instrument_sigma_deg = NoWheelDoubleSpinBox()
        self.instrument_sigma_deg.setRange(0.0, 10.0)
        self.instrument_sigma_deg.setDecimals(4)
        self.instrument_sigma_deg.setValue(0.10)
        self.instrument_sigma_deg.setToolTip("Physical resolution: Gaussian σ in degrees 2θ")
        inst_broad_form.addRow("Instrument sigma (deg):", self.instrument_sigma_deg)
        self.kin_apply_plot_smoothing = QCheckBox()
        self.kin_apply_plot_smoothing.setToolTip("Cosmetic display smoothing, added in quadrature")
        inst_broad_form.addRow("Apply plot smoothing:", self.kin_apply_plot_smoothing)
        self.kin_plot_smoothing_sigma = NoWheelDoubleSpinBox()
        self.kin_plot_smoothing_sigma.setRange(0.0, 10.0)
        self.kin_plot_smoothing_sigma.setDecimals(4)
        self.kin_plot_smoothing_sigma.setValue(0.0)
        inst_broad_form.addRow("Plot smoothing sigma (deg):", self.kin_plot_smoothing_sigma)
        self.gaussian_width = NoWheelDoubleSpinBox()
        self.gaussian_width.setRange(0.0, 100.0)
        self.gaussian_width.setDecimals(4)
        self.gaussian_width.setValue(0.0)
        self.gaussian_width.setToolTip(
            "Intrinsic base σ added in quadrature. 0 means switching every broadening option "
            "off really gives an unbroadened curve")
        inst_broad_form.addRow("Gaussian width (°):", self.gaussian_width)
        self.gaussian_points = NoWheelSpinBox()
        self.gaussian_points.setRange(2, 1000000)
        self.gaussian_points.setValue(2000)
        self.gaussian_points.setToolTip("Number of points in the continuous curve")
        inst_broad_form.addRow("Gaussian points:", self.gaussian_points)
        self.kin_curve_padding = NoWheelDoubleSpinBox()
        self.kin_curve_padding.setRange(0.0, 90.0)
        self.kin_curve_padding.setDecimals(4)
        self.kin_curve_padding.setValue(0.0)
        self.kin_curve_padding.setToolTip("Extra curve range outside the requested 2θ interval (°)")
        inst_broad_form.addRow("Curve padding (deg):", self.kin_curve_padding)
        layout.addWidget(inst_broad_group)

        out_group = QGroupBox("Output Files")
        out_form = QFormLayout(out_group)
        self.kin_reflections_file = QLineEdit()
        self.kin_reflections_file.setText("xrd_reflections_full.txt")
        self.kin_reflections_file.setToolTip("Complete calculated reflection list (unfiltered)")
        apply_validator(self.kin_reflections_file, FilenameValidator())
        out_form.addRow("Reflections file:", self.kin_reflections_file)
        self.kin_peaks_file = QLineEdit()
        self.kin_peaks_file.setText("xrd_peaks_display.txt")
        self.kin_peaks_file.setToolTip("Filtered display-peak list (labelling / tabulation only)")
        apply_validator(self.kin_peaks_file, FilenameValidator())
        out_form.addRow("Peaks file:", self.kin_peaks_file)
        self.kin_curve_file = QLineEdit()
        self.kin_curve_file.setText("xrd_curve.txt")
        self.kin_curve_file.setToolTip("Continuous Gaussian-broadened curve")
        apply_validator(self.kin_curve_file, FilenameValidator())
        out_form.addRow("Curve file:", self.kin_curve_file)
        self.kin_show_structure_info = QCheckBox("Show structure info")
        self.kin_show_structure_info.setChecked(True)
        out_form.addRow(self.kin_show_structure_info)
        self.kin_save_cif = QCheckBox("Save CIF")
        self.kin_save_cif.setToolTip(
            "Write the periodic structure that was used.\nBy default it is reduced to "
            "the primitive cell, but only when the reduction is lossless: a box with "
            "vacancies, substitutions or thermal displacement keeps every atom.\n"
            "Use cif_content = as_used to switch the reduction off")
        out_form.addRow(self.kin_save_cif)
        self.kin_cif_file = QLineEdit()
        self.kin_cif_file.setText("structure.cif")
        self.kin_cif_file.setToolTip("Filename for the saved CIF (no path separators)")
        apply_validator(self.kin_cif_file, FilenameValidator())
        out_form.addRow("CIF file:", self.kin_cif_file)
        self.kin_cif_content = NoWheelComboBox()
        self.kin_cif_content.addItems(["primitive", "conventional", "as_used"])
        self.kin_cif_content.setToolTip(
            "primitive: the smallest cell the structure reduces to, losslessly\n"
            "conventional: the standard crystallographic setting\n"
            "as_used: every atom of the structure the pattern was computed from")
        out_form.addRow("CIF content:", self.kin_cif_content)
        self.kin_cif_symprec = NoWheelDoubleSpinBox()
        self.kin_cif_symprec.setRange(0.0, 10.0)
        self.kin_cif_symprec.setDecimals(5)
        self.kin_cif_symprec.setSingleStep(0.01)
        self.kin_cif_symprec.setValue(0.01)
        self.kin_cif_symprec.setToolTip(
            "Symmetry tolerance (Å) for the reduction. Larger values reduce more "
            "aggressively and can erase small displacements")
        out_form.addRow("CIF symprec (Å):", self.kin_cif_symprec)
        layout.addWidget(out_group)

        viz_group = QGroupBox("Visualization")
        viz_form = QFormLayout(viz_group)
        self.kin_make_plot = QCheckBox("Make plot")
        self.kin_make_plot.setChecked(True)
        viz_form.addRow(self.kin_make_plot)
        self.kin_plot_filename = QLineEdit()
        self.kin_plot_filename.setText("xrd_plot.png")
        self.kin_plot_filename.setToolTip("Filename for saved plot (no path separators)")
        apply_validator(self.kin_plot_filename, FilenameValidator())
        viz_form.addRow("Plot filename:", self.kin_plot_filename)
        self.kin_plot_dpi = NoWheelSpinBox()
        self.kin_plot_dpi.setRange(30, 2000)
        self.kin_plot_dpi.setValue(150)
        viz_form.addRow("Plot DPI:", self.kin_plot_dpi)
        self.kin_plot_figsize = QLineEdit()
        self.kin_plot_figsize.setText("12, 6")
        self.kin_plot_figsize.setToolTip("Figure size in inches: width, height")
        apply_validator(self.kin_plot_figsize, FloatPairValidator())
        viz_form.addRow("Plot figsize (in):", self.kin_plot_figsize)
        self.kin_show_grid = QCheckBox("Show grid")
        self.kin_show_grid.setChecked(True)
        viz_form.addRow(self.kin_show_grid)
        self.kin_grid_alpha = NoWheelDoubleSpinBox()
        self.kin_grid_alpha.setRange(0.0, 1.0)
        self.kin_grid_alpha.setSingleStep(0.05)
        self.kin_grid_alpha.setValue(0.3)
        viz_form.addRow("Grid alpha:", self.kin_grid_alpha)
        self.kin_marker_at = NoWheelComboBox()
        self.kin_marker_at.addItems(["smoothed", "sticks"])
        self.kin_marker_at.setToolTip(
            "Place the hkl markers on the broadened curve maxima ('smoothed') or at the "
            "exact reflection positions ('sticks')")
        viz_form.addRow("Marker at:", self.kin_marker_at)
        layout.addWidget(viz_group)

        self.kin_calc_btn = make_primary_button("Calculate")
        self.kin_stop_btn = make_stop_button()
        self.kin_calc_btn.clicked.connect(self._run_kin)
        self.kin_stop_btn.clicked.connect(self._stop_calculation)
        add_run_row(layout, self.kin_calc_btn, self.kin_stop_btn)
        layout.addStretch()
        self.tabs.addTab(scroll, "XRD-Kinematical")

    def _create_debye_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)

        input_group = QGroupBox("Input")
        input_form = QFormLayout(input_group)
        self.debye_input = QLineEdit()
        browse_input_btn = make_action_button("Browse")
        browse_input_btn.clicked.connect(lambda: browse_file(self, self.debye_input, _XYZ_FILTER))
        input_row = QHBoxLayout()
        input_row.addWidget(self.debye_input)
        input_row.addWidget(browse_input_btn)
        input_form.addRow("Input (xyz):", input_row)
        self.debye_output = QLineEdit()
        self.debye_output.setText("debye_pattern.txt")
        self.debye_output.setToolTip("Main pattern file: q(1/Å), 2θ(deg), I(arb.)")
        browse_output_btn = make_action_button("Browse")
        browse_output_btn.clicked.connect(lambda: create_file(self, self.debye_output, _TXT_FILTER))
        output_row = QHBoxLayout()
        output_row.addWidget(self.debye_output)
        output_row.addWidget(browse_output_btn)
        input_form.addRow("Output (txt):", output_row)

        self.debye_partial_output = QLineEdit()
        self.debye_partial_output.setText("debye_partials.txt")
        self.debye_partial_output.setToolTip("Species-pair partial intensities")
        browse_partial_btn = make_action_button("Browse")
        browse_partial_btn.clicked.connect(
            lambda: create_file(self, self.debye_partial_output, _TXT_FILTER))
        partial_row = QHBoxLayout()
        partial_row.addWidget(self.debye_partial_output)
        partial_row.addWidget(browse_partial_btn)
        input_form.addRow("Partials (txt):", partial_row)

        self.debye_out_dir = QLineEdit()
        btn_out_dir = make_action_button("Browse")
        btn_out_dir.clicked.connect(lambda: browse_directory(self, self.debye_out_dir))
        out_dir_row = QHBoxLayout()
        out_dir_row.addWidget(self.debye_out_dir)
        out_dir_row.addWidget(btn_out_dir)
        input_form.addRow("Output directory:", out_dir_row)

        self.debye_species_mode = NoWheelComboBox()
        self.debye_species_mode.addItems(["chemical_symbols", "atomic_numbers", "lammps_types"])
        self.debye_species_mode.setToolTip(
            "How the species column is read. A bare number is an atomic number with "
            "'atomic_numbers' and a LAMMPS atom type with 'lammps_types' (needs a type map)")
        input_form.addRow("Species mode:", self.debye_species_mode)

        self.debye_type_map = QLineEdit()
        self.debye_type_map.setPlaceholderText("1:Cu,2:Zn")
        self.debye_type_map.setToolTip("Required for species_mode = lammps_types, e.g. 1:Cu,2:Zn")
        input_form.addRow("Type map:", self.debye_type_map)

        layout.addWidget(input_group)

        calc_group = QGroupBox("Calculation Parameters")
        calc_form = QFormLayout(calc_group)
        self.debye_wavelength = NoWheelComboBox()
        self.debye_wavelength.addItems(_RADIATIONS)
        self.debye_wavelength.setToolTip(
            "'CuKa' is the Ka1/Ka2 average (1.54184 Å); use 'CuKa1' for the single line")
        calc_form.addRow("Wavelength:", self.debye_wavelength)
        self.debye_two_theta_min = NoWheelDoubleSpinBox()
        self.debye_two_theta_min.setRange(0.0, 180.0)
        self.debye_two_theta_min.setValue(30.0)
        calc_form.addRow("2θ min (°):", self.debye_two_theta_min)
        self.debye_two_theta_max = NoWheelDoubleSpinBox()
        self.debye_two_theta_max.setRange(0.0, 180.0)
        self.debye_two_theta_max.setValue(90.0)
        calc_form.addRow("2θ max (°):", self.debye_two_theta_max)
        self.debye_use_qmax = QCheckBox("Use q max instead of 2θ max")
        self.debye_use_qmax.setToolTip(
            "q_max is an alternative upper bound. If both are supplied two_theta_max wins, "
            "so only one of them is written to the input file")
        calc_form.addRow(self.debye_use_qmax)
        self.debye_qmax = NoWheelDoubleSpinBox()
        self.debye_qmax.setRange(0.0, 1000.0)
        self.debye_qmax.setDecimals(4)
        self.debye_qmax.setValue(15.0)
        self.debye_qmax.setEnabled(False)
        self.debye_qmax.setToolTip("Maximum scattering vector magnitude q (Å⁻¹)")
        self.debye_use_qmax.toggled.connect(self.debye_qmax.setEnabled)
        self.debye_use_qmax.toggled.connect(
            lambda on: self.debye_two_theta_max.setEnabled(not on))
        calc_form.addRow("q max (Å⁻¹):", self.debye_qmax)
        self.debye_npoints = NoWheelSpinBox()
        self.debye_npoints.setRange(2, 1000000)
        self.debye_npoints.setValue(500)
        calc_form.addRow("Number of points:", self.debye_npoints)
        self.debye_method = NoWheelComboBox()
        self.debye_method.addItems(["pairwise", "binned", "auto"])
        self.debye_method.setToolTip(
            "pairwise: exact O(N²) pair sum. binned: same equation with bucketed pair "
            "distances, ~100x faster. auto: pairwise up to direct_threshold atoms, else binned")
        calc_form.addRow("Debye method:", self.debye_method)
        self.debye_direct_threshold = NoWheelSpinBox()
        self.debye_direct_threshold.setRange(0, 100000000)
        self.debye_direct_threshold.setValue(2000)
        self.debye_direct_threshold.setToolTip("Atom count below which 'auto' picks pairwise")
        calc_form.addRow("Direct threshold:", self.debye_direct_threshold)
        self.debye_bins_per_angst = NoWheelDoubleSpinBox()
        self.debye_bins_per_angst.setRange(0.1, 10000.0)
        self.debye_bins_per_angst.setDecimals(2)
        self.debye_bins_per_angst.setValue(20.0)
        self.debye_bins_per_angst.setToolTip("Radial bin resolution for debye_method = binned")
        calc_form.addRow("Bins per Ångström:", self.debye_bins_per_angst)
        self.debye_max_dist = NoWheelComboBox()
        self.debye_max_dist.addItems(["adaptive", "value"])
        self.debye_max_dist.setToolTip(
            "'adaptive' uses the exact maximum pair distance so the binned sum is complete. "
            "A smaller number TRUNCATES the Debye sum")
        calc_form.addRow("Max distance:", self.debye_max_dist)
        self.debye_max_dist_value = NoWheelDoubleSpinBox()
        self.debye_max_dist_value.setRange(0.0, 1e6)
        self.debye_max_dist_value.setDecimals(3)
        self.debye_max_dist_value.setValue(50.0)
        self.debye_max_dist_value.setEnabled(False)
        self.debye_max_dist.currentTextChanged.connect(
            lambda text: self.debye_max_dist_value.setEnabled(text == "value"))
        calc_form.addRow("Max distance value (Å):", self.debye_max_dist_value)
        self.debye_compute_partials = QCheckBox("Compute partial intensities")
        self.debye_compute_partials.setChecked(True)
        self.debye_compute_partials.setToolTip(
            "Species-resolved intensity contributions; turn off when only the total is needed")
        calc_form.addRow(self.debye_compute_partials)
        self.debye_norm_int = QCheckBox("Normalize intensity")
        self.debye_norm_int.setChecked(True)
        calc_form.addRow(self.debye_norm_int)
        self.debye_norm_max = NoWheelDoubleSpinBox()
        self.debye_norm_max.setRange(0, 1e9)
        self.debye_norm_max.setValue(100.0)
        calc_form.addRow("Normalize max:", self.debye_norm_max)
        self.debye_scale_factor = NoWheelDoubleSpinBox()
        self.debye_scale_factor.setRange(0.0, 1e9)
        self.debye_scale_factor.setDecimals(4)
        self.debye_scale_factor.setValue(1.0)
        calc_form.addRow("Scale factor:", self.debye_scale_factor)
        self.debye_random_seed = NoWheelSpinBox()
        self.debye_random_seed.setRange(0, 99999999)
        self.debye_random_seed.setToolTip("Seed for reproducible subsampling")
        calc_form.addRow("Random seed:", self.debye_random_seed)
        layout.addWidget(calc_group)

        chunk_group = QGroupBox("Chunking / Memory")
        chunk_form = QFormLayout(chunk_group)
        self.debye_chunk_size = NoWheelDoubleSpinBox()
        self.debye_chunk_size.setRange(1, 1e8)
        self.debye_chunk_size.setDecimals(0)
        self.debye_chunk_size.setValue(2048)
        self.debye_chunk_size.setToolTip("Centre atoms per neighbour-query chunk")
        chunk_form.addRow("Chunk size:", self.debye_chunk_size)
        self.debye_pair_sub = NoWheelDoubleSpinBox()
        self.debye_pair_sub.setRange(1, 1e9)
        self.debye_pair_sub.setDecimals(0)
        self.debye_pair_sub.setValue(200000)
        self.debye_pair_sub.setToolTip("Pairs held in memory at once")
        chunk_form.addRow("Pair subchunk size:", self.debye_pair_sub)
        self.debye_max_pair_matrix = NoWheelDoubleSpinBox()
        self.debye_max_pair_matrix.setRange(1, 1e12)
        self.debye_max_pair_matrix.setDecimals(0)
        self.debye_max_pair_matrix.setValue(1e8)
        self.debye_max_pair_matrix.setToolTip("Hard cap on n_q × n_pairs per subchunk")
        chunk_form.addRow("Max pair matrix elements:", self.debye_max_pair_matrix)
        self.debye_max_dist_block = NoWheelDoubleSpinBox()
        self.debye_max_dist_block.setRange(1, 1e12)
        self.debye_max_dist_block.setDecimals(0)
        self.debye_max_dist_block.setValue(4e6)
        self.debye_max_dist_block.setToolTip("Memory limit for direct pair-distance blocks")
        chunk_form.addRow("Max distance block elements:", self.debye_max_dist_block)
        self.debye_max_pairs = NoWheelDoubleSpinBox()
        self.debye_max_pairs.setRange(1, 1e15)
        self.debye_max_pairs.setDecimals(0)
        self.debye_max_pairs.setValue(2e9)
        self.debye_max_pairs.setToolTip("Pair-count safety cap")
        chunk_form.addRow("Max pairs:", self.debye_max_pairs)
        self.debye_allow_large_pairs = QCheckBox("Allow large pair count")
        self.debye_allow_large_pairs.setToolTip("Override the pair-count safety cap")
        chunk_form.addRow(self.debye_allow_large_pairs)
        self.debye_max_int_distance = NoWheelDoubleSpinBox()
        self.debye_max_int_distance.setRange(-1, 1e9)
        self.debye_max_int_distance.setDecimals(3)
        self.debye_max_int_distance.setValue(-1)
        self.debye_max_int_distance.setToolTip(
            "Explicit pair cutoff in Å (-1 = no limit). A finite value TRUNCATES the Debye sum")
        chunk_form.addRow("Max interaction distance (Å):", self.debye_max_int_distance)
        layout.addWidget(chunk_group)

        corr_group = QGroupBox("Measurement Corrections")
        corr_form = QFormLayout(corr_group)
        self.debye_exp_correct = QCheckBox("Experimental correction")
        self.debye_exp_correct.setToolTip(
            "Enables LP, instrumental broadening and Debye-Waller together. "
            "It does NOT enable anomalous dispersion")
        corr_form.addRow(self.debye_exp_correct)
        self.debye_apply_lp = QCheckBox("Apply LP")
        self.debye_apply_lp.setToolTip("Lorentz-polarization factor")
        corr_form.addRow(self.debye_apply_lp)
        self.debye_lp_variant = NoWheelComboBox()
        self.debye_lp_variant.addItems(["with_polarization", "lorentz_only", "none"])
        self.debye_lp_variant.setToolTip(
            "with_polarization: (1 + cos²2θ)/(sin²θ cos θ) for an unpolarised source. "
            "lorentz_only: 1/(sin²θ cos θ). none: 1")
        corr_form.addRow("LP variant:", self.debye_lp_variant)
        self.debye_lp_theta_min = NoWheelDoubleSpinBox()
        self.debye_lp_theta_min.setRange(0.0, 90.0)
        self.debye_lp_theta_min.setDecimals(4)
        self.debye_lp_theta_min.setValue(0.5)
        self.debye_lp_theta_min.setToolTip(
            "Lower clip angle for the LP factor, avoiding division by near-zero (degrees)")
        corr_form.addRow("LP θ min clip (deg):", self.debye_lp_theta_min)
        self.debye_lp_max_clip = NoWheelDoubleSpinBox()
        self.debye_lp_max_clip.setRange(0, 1e12)
        self.debye_lp_max_clip.setDecimals(1)
        self.debye_lp_max_clip.setValue(1e4)
        self.debye_lp_max_clip.setToolTip("Upper clip value for the LP correction factor")
        corr_form.addRow("LP max clip:", self.debye_lp_max_clip)
        self.debye_apply_broadening = QCheckBox("Apply instrumental broadening")
        corr_form.addRow(self.debye_apply_broadening)
        self.debye_instrument_sigma = NoWheelDoubleSpinBox()
        self.debye_instrument_sigma.setRange(0.0, 10.0)
        self.debye_instrument_sigma.setDecimals(4)
        self.debye_instrument_sigma.setValue(0.10)
        self.debye_instrument_sigma.setToolTip("Constant Gaussian σ in degrees 2θ")
        corr_form.addRow("Instrument sigma (deg):", self.debye_instrument_sigma)
        self.debye_use_caglioti = QCheckBox("Use Caglioti")
        self.debye_use_caglioti.setToolTip(
            "Angle-dependent resolution: FWHM² = U tan²θ + V tanθ + W")
        corr_form.addRow(self.debye_use_caglioti)
        self.debye_caglioti_u = NoWheelDoubleSpinBox()
        self.debye_caglioti_u.setRange(-1e6, 1e6)
        self.debye_caglioti_u.setDecimals(6)
        self.debye_caglioti_u.setValue(0.01)
        corr_form.addRow("Caglioti U:", self.debye_caglioti_u)
        self.debye_caglioti_v = NoWheelDoubleSpinBox()
        self.debye_caglioti_v.setRange(-1e6, 1e6)
        self.debye_caglioti_v.setDecimals(6)
        self.debye_caglioti_v.setValue(0.01)
        corr_form.addRow("Caglioti V:", self.debye_caglioti_v)
        self.debye_caglioti_w = NoWheelDoubleSpinBox()
        self.debye_caglioti_w.setRange(-1e6, 1e6)
        self.debye_caglioti_w.setDecimals(6)
        self.debye_caglioti_w.setValue(0.01)
        corr_form.addRow("Caglioti W:", self.debye_caglioti_w)
        layout.addWidget(corr_group)

        dw_group = QGroupBox("Thermal Damping")
        dw_form = QFormLayout(dw_group)
        self.debye_apply_dw = QCheckBox("Apply Debye-Waller")
        self.debye_apply_dw.setToolTip(
            "Amplitude factor exp(-B q²/(16π²)). Careful: applying this on top of a "
            "finite-temperature MD snapshot double counts the thermal displacement")
        dw_form.addRow(self.debye_apply_dw)
        self.debye_dw_factors = QLineEdit()
        self.debye_dw_factors.setPlaceholderText("Zn=0.5,Cu=0.6")
        self.debye_dw_factors.setToolTip("B per element in Å², e.g. Zn=0.5, Cu=0.6")
        apply_validator(self.debye_dw_factors, DebyeWallerValidator())
        dw_form.addRow("Debye Waller Factors:", self.debye_dw_factors)
        layout.addWidget(dw_group)

        anom_group = QGroupBox("Anomalous Scattering")
        anom_form = QFormLayout(anom_group)
        self.apply_anom = QCheckBox("Apply anomalous")
        self.apply_anom.setChecked(True)
        self.apply_anom.setToolTip(
            "f(q) = f0(q) + f′(λ) + i f″(λ). Values come from the entries below when given, "
            "otherwise from periodictable's Henke tables at the incident energy")
        anom_form.addRow(self.apply_anom)
        layout.addWidget(anom_group)

        self.anom_entries_group = QGroupBox("Anomalous elements (f′ / f″)")
        anom_entries_layout = QVBoxLayout(self.anom_entries_group)
        ctrl_row = QHBoxLayout()
        add_btn = make_action_button("Add element")
        add_btn.clicked.connect(self._add_anomalous_entry)
        ctrl_row.addWidget(add_btn)
        ctrl_row.addStretch(1)
        anom_entries_layout.addLayout(ctrl_row)
        self.anom_entries_container = QWidget()
        self.anom_entries_vbox = QVBoxLayout(self.anom_entries_container)
        self.anom_entries_vbox.setContentsMargins(0, 0, 0, 0)
        self.anom_entries_vbox.setSpacing(6)
        anom_entries_layout.addWidget(self.anom_entries_container)
        layout.addWidget(self.anom_entries_group)
        self._add_anomalous_entry()

        diag_group = QGroupBox("Structural Diagnostics (never feed I(q))")
        diag_form = QFormLayout(diag_group)
        self.debye_compute_rdf = QCheckBox("Compute RDF")
        self.debye_compute_rdf.setToolTip(
            "Partial RDFs g_ab(r) need a number density, a cluster volume and a boundary "
            "correction, none of which the DSE requires")
        diag_form.addRow(self.debye_compute_rdf)
        self.debye_plot_rdf = QCheckBox("Plot RDF")
        diag_form.addRow(self.debye_plot_rdf)
        self.debye_rdf_max_distance = NoWheelDoubleSpinBox()
        self.debye_rdf_max_distance.setRange(0.0, 1e6)
        self.debye_rdf_max_distance.setDecimals(3)
        self.debye_rdf_max_distance.setValue(20.0)
        diag_form.addRow("RDF max distance (Å):", self.debye_rdf_max_distance)
        self.debye_rdf_sample_s = NoWheelSpinBox()
        self.debye_rdf_sample_s.setRange(0, 10000000)
        self.debye_rdf_sample_s.setValue(0)
        self.debye_rdf_sample_s.setToolTip("0 = use every atom as a centre")
        diag_form.addRow("RDF sample size:", self.debye_rdf_sample_s)
        self.debye_apply_acc_frac = QCheckBox("Apply accessible fraction")
        self.debye_apply_acc_frac.setToolTip("Boundary correction for the REPORTED g(r) only")
        diag_form.addRow(self.debye_apply_acc_frac)
        self.debye_rdf_access_dirs = NoWheelSpinBox()
        self.debye_rdf_access_dirs.setRange(1, 100000)
        self.debye_rdf_access_dirs.setValue(200)
        diag_form.addRow("RDF access dirs:", self.debye_rdf_access_dirs)
        self.debye_rdf_access_centers = NoWheelSpinBox()
        self.debye_rdf_access_centers.setRange(1, 1000000)
        self.debye_rdf_access_centers.setValue(500)
        diag_form.addRow("RDF access centers:", self.debye_rdf_access_centers)
        self.debye_compute_pdf = QCheckBox("Compute pair distribution")
        diag_form.addRow(self.debye_compute_pdf)
        self.debye_pdf_subset = NoWheelSpinBox()
        self.debye_pdf_subset.setRange(1, 10000000)
        self.debye_pdf_subset.setValue(5000)
        diag_form.addRow("PDF subset size:", self.debye_pdf_subset)
        self.debye_pdf_bins = NoWheelSpinBox()
        self.debye_pdf_bins.setRange(2, 1000000)
        self.debye_pdf_bins.setValue(100)
        diag_form.addRow("PDF bins:", self.debye_pdf_bins)
        self.debye_plot_pdf = QCheckBox("Plot pair distribution")
        diag_form.addRow(self.debye_plot_pdf)
        layout.addWidget(diag_group)

        viz_group = QGroupBox("Visualization")
        viz_form = QFormLayout(viz_group)
        self.debye_make_plot = QCheckBox("Make plot")
        self.debye_make_plot.setChecked(True)
        viz_form.addRow(self.debye_make_plot)
        self.debye_plot_partials = QCheckBox("Plot partials")
        viz_form.addRow(self.debye_plot_partials)
        self.debye_plot_filename = QLineEdit()
        self.debye_plot_filename.setText("debye_plot.png")
        apply_validator(self.debye_plot_filename, FilenameValidator())
        viz_form.addRow("Plot filename:", self.debye_plot_filename)
        self.debye_plot_dpi = NoWheelSpinBox()
        self.debye_plot_dpi.setRange(30, 2000)
        self.debye_plot_dpi.setValue(300)
        viz_form.addRow("Plot DPI:", self.debye_plot_dpi)
        self.debye_verbose = QCheckBox("Verbose")
        self.debye_verbose.setChecked(True)
        viz_form.addRow(self.debye_verbose)
        layout.addWidget(viz_group)

        self.debye_calc_btn = make_primary_button("Calculate")
        self.debye_stop_btn = make_stop_button()
        self.debye_calc_btn.clicked.connect(self._run_debye)
        self.debye_stop_btn.clicked.connect(self._stop_calculation)
        add_run_row(layout, self.debye_calc_btn, self.debye_stop_btn)
        layout.addStretch()
        self.tabs.addTab(scroll, "XRD-Debye")

    def _add_anomalous_entry(self) -> None:
        row = AnomalousEntryRow()
        row.remove_requested.connect(self._remove_anomalous_entry)
        self._anom_rows.append(row)
        self.anom_entries_vbox.addWidget(row)

    def _remove_anomalous_entry(self, row: AnomalousEntryRow) -> None:
        self._anom_rows.remove(row)
        self.anom_entries_vbox.removeWidget(row)
        row.setParent(None); row.deleteLater()

    def _get_anomalous_entries(self) -> list[dict]:
        return [r.data() for r in self._anom_rows if r.data()["element"]]

    def _create_reciprocal_tab(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll.setWidget(tab)
        reciprocal_group = QGroupBox("XRD-ReciprocalSum")
        reciprocal_form = QFormLayout(reciprocal_group)

        self.reciprocal_input = QLineEdit()
        browse_input_btn = make_action_button("Browse")
        browse_input_btn.clicked.connect(lambda: browse_file(self, self.reciprocal_input, _XYZ_FILTER))
        input_row = QHBoxLayout()
        input_row.addWidget(self.reciprocal_input)
        input_row.addWidget(browse_input_btn)
        reciprocal_form.addRow("Input (xyz):", input_row)

        self.reciprocal_output = QLineEdit()
        self.reciprocal_output.setText("xrd_results.txt")
        self.reciprocal_output.setToolTip(
            "Histogram output: Bin, Coord (2θ), Count, Count/Total")
        browse_output_btn = make_action_button("Browse")
        browse_output_btn.clicked.connect(lambda: create_file(self, self.reciprocal_output, _TXT_FILTER))
        output_row = QHBoxLayout()
        output_row.addWidget(self.reciprocal_output)
        output_row.addWidget(browse_output_btn)
        reciprocal_form.addRow("Output File:", output_row)

        self.reciprocal_plot_location = QLineEdit()
        self.reciprocal_plot_location.setText("xrd_plot.png")
        browse_plot_btn = make_action_button("Browse")
        browse_plot_btn.clicked.connect(lambda: create_file(self, self.reciprocal_plot_location, _PNG_FILTER))
        plot_row = QHBoxLayout()
        plot_row.addWidget(self.reciprocal_plot_location)
        plot_row.addWidget(browse_plot_btn)
        reciprocal_form.addRow("Plot file:", plot_row)

        self.reciprocal_out_dir = QLineEdit()
        btn_out_dir = make_action_button("Browse")
        btn_out_dir.clicked.connect(lambda: browse_directory(self, self.reciprocal_out_dir))
        out_dir_row = QHBoxLayout()
        out_dir_row.addWidget(self.reciprocal_out_dir)
        out_dir_row.addWidget(btn_out_dir)
        reciprocal_form.addRow("Output directory:", out_dir_row)

        self.reciprocal_wavelength = NoWheelDoubleSpinBox()
        self.reciprocal_wavelength.setRange(0.001, 100.0)
        self.reciprocal_wavelength.setDecimals(4)
        self.reciprocal_wavelength.setValue(1.5406)
        self.reciprocal_wavelength.setToolTip(
            "X-ray wavelength in Å. This module takes a plain number so an existing LAMMPS "
            "run stays reproducible (1.5406 ≈ Cu Kα1)")
        reciprocal_form.addRow("Wavelength (Å):", self.reciprocal_wavelength)

        self.reciprocal_2theta = QLineEdit()
        self.reciprocal_2theta.setText("10, 90")
        self.reciprocal_2theta.setToolTip("2θ range as min, max (degrees); 0 < min < max < 180")
        apply_validator(self.reciprocal_2theta, FloatPairValidator())
        reciprocal_form.addRow("2θ min, max (°):", self.reciprocal_2theta)

        self.reciprocal_bins = NoWheelSpinBox()
        self.reciprocal_bins.setRange(1, 100000)
        self.reciprocal_bins.setValue(250)
        reciprocal_form.addRow("Number of bins:", self.reciprocal_bins)
        self.reciprocal_manual = NoWheelComboBox()
        self.reciprocal_manual.addItems(["0", "1"])
        self.reciprocal_manual.setCurrentIndex(1)
        self.reciprocal_manual.setToolTip(
            "1 uses a c-defined mesh independent of the box (the LAMMPS option for "
            "non-periodic systems); prd then has no effect")
        reciprocal_form.addRow("Manual:", self.reciprocal_manual)

        self.reciprocal_pbc = QLineEdit()
        self.reciprocal_pbc.setText("1, 0, 0")
        self.reciprocal_pbc.setToolTip("Periodic boundary conditions along x, y, z (1=periodic, 0=free)")
        apply_validator(self.reciprocal_pbc, IntTripletValidator())
        reciprocal_form.addRow("Periodic BC:", self.reciprocal_pbc)

        self.reciprocal_c = QLineEdit()
        self.reciprocal_c.setText("0.01, 0.01, 0.01")
        self.reciprocal_c.setToolTip(
            "Mesh spacing: dK[i] = c[i] in manual mode, c[i]/L[i] in auto mode (Å⁻¹). "
            "Node count scales as (Kmax/c)³, so halving c multiplies the cost by eight")
        apply_validator(self.reciprocal_c, FloatTripletValidator())
        reciprocal_form.addRow("C (Å⁻¹):", self.reciprocal_c)

        self.reciprocal_prd = QLineEdit()
        self.reciprocal_prd.setText("40.0, 40.0, 50.0")
        self.reciprocal_prd.setToolTip(
            "Box dimensions x, y, z (Å). Reported for provenance; in manual mode it does "
            "NOT affect the mesh")
        apply_validator(self.reciprocal_prd, FloatTripletValidator())
        reciprocal_form.addRow("Box dimensions (Å):", self.reciprocal_prd)

        self.reciprocal_atom_types = QLineEdit()
        self.reciprocal_atom_types.setText("Mo")
        self.reciprocal_atom_types.setToolTip(
            "ORDERED list: entry i is the scattering-factor label for LAMMPS atom type i+1. "
            "Labels must match the LAMMPS list exactly ('Mo', 'Fe2+', 'Cval', ...); "
            "there is no fuzzy matching")
        reciprocal_form.addRow("Atom types:", self.reciprocal_atom_types)

        self.reciprocal_atom_type_mode = NoWheelComboBox()
        self.reciprocal_atom_type_mode.addItems(["auto", "lammps_numeric", "chemical_symbols"])
        self.reciprocal_atom_type_mode.setToolTip(
            "auto: an integer column is read as lammps_numeric, otherwise as symbols")
        reciprocal_form.addRow("Atom type mode:", self.reciprocal_atom_type_mode)

        self.reciprocal_compat_mode = NoWheelComboBox()
        self.reciprocal_compat_mode.addItems(["strict_lammps", "relaxed"])
        self.reciprocal_compat_mode.setToolTip(
            "strict_lammps: LAMMPS formulas only, triclinic cells are rejected. "
            "relaxed: triclinic cells fall back to diagonal box lengths and the output is "
            "NOT LAMMPS-equivalent")
        reciprocal_form.addRow("Compatibility mode:", self.reciprocal_compat_mode)

        self.reciprocal_max_candidates = NoWheelDoubleSpinBox()
        self.reciprocal_max_candidates.setRange(1, 1e12)
        self.reciprocal_max_candidates.setDecimals(0)
        self.reciprocal_max_candidates.setValue(20000000)
        self.reciprocal_max_candidates.setToolTip(
            "Maximum permitted candidate reciprocal nodes before a safety stop")
        reciprocal_form.addRow("Max reciprocal candidates:", self.reciprocal_max_candidates)

        self.reciprocal_allow_large_grid = QCheckBox("Allow large reciprocal grid")
        self.reciprocal_allow_large_grid.setToolTip("Override the reciprocal-grid safety cap")
        reciprocal_form.addRow(self.reciprocal_allow_large_grid)

        self.reciprocal_smoothing = NoWheelDoubleSpinBox()
        self.reciprocal_smoothing.setRange(0.0, 10.0)
        self.reciprocal_smoothing.setDecimals(4)
        self.reciprocal_smoothing.setValue(0.0)
        self.reciprocal_smoothing.setToolTip(
            "Display-only Gaussian smoothing of the binned curve (degrees 2θ); 0 = off")
        reciprocal_form.addRow("Plot smoothing sigma (deg):", self.reciprocal_smoothing)

        self.reciprocal_lp = NoWheelSpinBox()
        self.reciprocal_lp.setRange(0, 1)
        self.reciprocal_lp.setValue(1)
        self.reciprocal_lp.setToolTip("Apply Lorentz-Polarization correction (1=yes, 0=no)")
        reciprocal_form.addRow("LP:", self.reciprocal_lp)
        self.reciprocal_echo = NoWheelSpinBox()
        self.reciprocal_echo.setRange(0, 1)
        self.reciprocal_echo.setValue(1)
        reciprocal_form.addRow("Echo:", self.reciprocal_echo)
        self.reciprocal_plot_spin = NoWheelSpinBox()
        self.reciprocal_plot_spin.setRange(0, 1)
        self.reciprocal_plot_spin.setValue(1)
        reciprocal_form.addRow("Plot:", self.reciprocal_plot_spin)

        layout.addWidget(reciprocal_group)

        self.reciprocal_calc_btn = make_primary_button("Calculate")
        self.reciprocal_stop_btn = make_stop_button()
        self.reciprocal_calc_btn.clicked.connect(self._run_reciprocal)
        self.reciprocal_stop_btn.clicked.connect(self._stop_calculation)
        add_run_row(layout, self.reciprocal_calc_btn, self.reciprocal_stop_btn)
        layout.addStretch()
        self.tabs.addTab(scroll, "XRD-ReciprocalSum")

    # ------------------------------------------------------------------ slots

    def _run_kin(self) -> None:
        self.kin_viz.clear()
        input_path = self.kin_input.text().strip()
        out_dir_path = self.kin_out_dir.text().strip()

        if not self._validate_input_file(input_path):
            return
        if not self._validate_output_dir(out_dir_path):
            return
        if self.kin_make_plot.isChecked() and not self.kin_plot_filename.text().strip():
            self.logger.log_message("ERROR", "Please enter a plot filename first."); return
        if self.kin_species_mode.currentText() == "lammps_types" and not self.kin_type_map.text().strip():
            self.logger.log_message(
                "ERROR", "species_mode = lammps_types requires a type map, e.g. 1:Cu,2:Zn."); return
        if not self.kin_use_cell_params.isChecked():
            if self.kin_coordinate_mode.currentText() == "fractional":
                self.logger.log_message(
                    "ERROR", "Fractional coordinates need a cell: tick 'Supply unit-cell "
                    "parameters below' and give a, b, c and the angles."); return
            if self.kin_lattice_mode.currentText() == "unit_cell":
                self.logger.log_message(
                    "ERROR", "lattice_mode = unit_cell folds into the supplied cell: tick "
                    "'Supply unit-cell parameters below' and give a, b, c and the angles."); return

        figsize = self.kin_plot_figsize.text().strip() or "12, 6"

        params = {
            "xyz_file": input_path,
            "species_mode": self.kin_species_mode.currentText(),
            "coordinate_mode": self.kin_coordinate_mode.currentText(),
            "lattice_mode": self.kin_lattice_mode.currentText(),
            "wrap_coords": _yn(self.wrap_coords.isChecked()),
            "check_duplicate_sites": _yn(self.kin_check_duplicates.isChecked()),
            "collapse_tol": self.kin_collapse_tol.value(),
            "validate_proximity": _yn(self.kin_validate_proximity.isChecked()),
            "periodicity_surface_tol": self.kin_surface_tol.value(),
            "max_proximity_check_atoms": self.kin_max_proximity_check_atoms.value(),
            "max_collapse_atoms": self.kin_max_collapse_atoms.value(),
            "repeats_integer_tol": self.kin_repeats_integer_tol.value(),
            "use_ase": _yn(self.kin_use_ase.isChecked()),
            "infer_cell": _yn(self.kin_infer_cell.isChecked()),
            "cell_inference_tol": self.kin_cell_inference_tol.value(),
            "cell_inference_min_score": self.kin_cell_inference_min_score.value(),
            "cell_inference_reach": self.kin_cell_inference_reach.value(),
            "cell_inference_max_vectors": self.kin_cell_inference_max_vectors.value(),
            "cell_inference_max_residual": self.kin_cell_inference_max_residual.value(),
            "axis_period_margin": self.kin_axis_period_margin.value(),
            "axis_period_min_score": self.kin_axis_period_min_score.value(),
            "max_box_scan_atoms": self.kin_max_box_scan_atoms.value(),
            "wavelength": self.kin_wavelength.currentText(),
            "two_theta_min": self.kin_two_theta_min.value(),
            "two_theta_max": self.kin_two_theta_max.value(),
            "scale_factor": self.kin_scale_factor.value(),
            "float_precision": self.kin_float_precision.value(),
            "experimental_correction": _yn(self.experimental_correction.isChecked()),
            "absorption_geometry": self.abs_geo.currentText(),
            "mu_linear_cm_inverse": self.lin_abs.value(),
            "capillary_radius_cm": self.sample_radius.value(),
            "sample_thickness_cm": self.sample_thickness.value(),
            "instrument_sigma_deg": self.instrument_sigma_deg.value(),
            "plot_smoothing_sigma_deg": self.kin_plot_smoothing_sigma.value(),
            "gaussian_width": self.gaussian_width.value(),
            "gaussian_points": self.gaussian_points.value(),
            "curve_padding_deg": self.kin_curve_padding.value(),
            "min_intensity_percent": self.kin_min_intensity.value(),
            "max_peaks": self.kin_max_peaks.value(),
            "min_peak_separation": self.kin_min_peak_sep.value(),
            "normalize_mode": self.kin_normalize_mode.currentText(),
            "normalize_max": self.kin_normalize_max.value(),
            "reflections_file": self.kin_reflections_file.text().strip() or "xrd_reflections_full.txt",
            "peaks_file": self.kin_peaks_file.text().strip() or "xrd_peaks_display.txt",
            "curve_file": self.kin_curve_file.text().strip() or "xrd_curve.txt",
            "save_cif": _yn(self.kin_save_cif.isChecked()),
            "cif_file": self.kin_cif_file.text().strip() or "structure.cif",
            "cif_content": self.kin_cif_content.currentText(),
            "cif_symprec": self.kin_cif_symprec.value(),
            "show_structure_info": _yn(self.kin_show_structure_info.isChecked()),
            "make_plot": _yn(self.kin_make_plot.isChecked()),
            "show_plot": "no",
            "plot_filename": self.kin_plot_filename.text().strip(),
            "plot_dpi": self.kin_plot_dpi.value(),
            "plot_figsize": f"[{figsize}]",
            "show_grid": _yn(self.kin_show_grid.isChecked()),
            "grid_alpha": self.kin_grid_alpha.value(),
            "marker_at": self.kin_marker_at.currentText(),
        }
        _set_apply_flags(params, (
            ("apply_debye_waller", self.apply_debye_waller),
            ("apply_absorption", self.apply_abs),
            ("apply_instrumental_broadening", self.apply_inst_broadening),
            ("apply_plot_smoothing", self.kin_apply_plot_smoothing),
        ))
        # Omitted rather than written empty: build_structure branches on whether the
        # six cell keys are present at all, and an empty value still counts as present.
        if self.kin_use_cell_params.isChecked():
            params.update({
                "a": self.a.value(), "b": self.b.value(), "c": self.c_lat.value(),
                "alpha": self.alpha.value(), "beta": self.beta.value(),
                "gamma": self.gamma.value(),
            })
        if self.kin_species_mode.currentText() == "lammps_types":
            params["type_map"] = self.kin_type_map.text().strip()
        if self.kin_cell_file.text().strip():
            params["cell_file"] = self.kin_cell_file.text().strip()
        if self.kin_reflection_cost_warn.text().strip():
            params["reflection_cost_warn"] = self.kin_reflection_cost_warn.text().strip()
        if self.kin_max_reflection_cost.text().strip():
            params["max_reflection_cost"] = self.kin_max_reflection_cost.text().strip()
        if self.debye_waller_factors.text().strip():
            params["debye_waller_factors"] = self.debye_waller_factors.text().strip()

        param_file_path = os.path.join(out_dir_path, "xrd_kin_input.txt")
        self._write_and_run(
            param_file_path, params,
            ("XRD", "XRD-Kinematical.py"),
            out_dir_path,
            self._make_finish_callback("XRD Kinematical", self._plot_kin),
        )

    def _run_debye(self) -> None:
        self.debye_viz.clear()
        input_path = self.debye_input.text().strip()
        out_dir_path = self.debye_out_dir.text().strip()

        if not self._validate_input_file(input_path):
            return
        if not self.debye_output.text().strip():
            self.logger.log_message("ERROR", "Please choose an output file first."); return
        if self.debye_species_mode.currentText() == "lammps_types" and not self.debye_type_map.text().strip():
            self.logger.log_message(
                "ERROR", "species_mode = lammps_types requires a type map, e.g. 1:Cu,2:Zn."); return
        if not self._validate_output_dir(out_dir_path):
            return

        debye_output_path = self._resolve_to_outdir(self.debye_output, out_dir_path)
        debye_partial_path = self._resolve_to_outdir(self.debye_partial_output, out_dir_path)

        params = {
            "xyz_file": input_path,
            "species_mode": self.debye_species_mode.currentText(),
            "wavelength": self.debye_wavelength.currentText(),
            "two_theta_min": self.debye_two_theta_min.value(),
            "n_points": self.debye_npoints.value(),
            "debye_method": self.debye_method.currentText(),
            "direct_threshold": self.debye_direct_threshold.value(),
            "bins_per_angstrom": self.debye_bins_per_angst.value(),
            "max_distance": ("adaptive" if self.debye_max_dist.currentText() == "adaptive"
                             else self.debye_max_dist_value.value()),
            "compute_partial_intensities": _yn(self.debye_compute_partials.isChecked()),
            "chunk_size": int(self.debye_chunk_size.value()),
            "pair_subchunk_size": int(self.debye_pair_sub.value()),
            "max_pair_matrix_elements": int(self.debye_max_pair_matrix.value()),
            "max_distance_block_elements": int(self.debye_max_dist_block.value()),
            "max_pairs": int(self.debye_max_pairs.value()),
            "allow_large_pair_count": _yn(self.debye_allow_large_pairs.isChecked()),
            "experimental_correction": _yn(self.debye_exp_correct.isChecked()),
            "LP_variant": self.debye_lp_variant.currentText(),
            "LP_theta_min_clip_deg": self.debye_lp_theta_min.value(),
            "LP_max_clip": self.debye_lp_max_clip.value(),
            "instrument_sigma_deg": self.debye_instrument_sigma.value(),
            "use_caglioti": _yn(self.debye_use_caglioti.isChecked()),
            "caglioti_U": self.debye_caglioti_u.value(),
            "caglioti_V": self.debye_caglioti_v.value(),
            "caglioti_W": self.debye_caglioti_w.value(),
            "scale_factor": self.debye_scale_factor.value(),
            "normalize_intensity": _yn(self.debye_norm_int.isChecked()),
            "normalize_max": self.debye_norm_max.value(),
            "compute_rdf": _yn(self.debye_compute_rdf.isChecked()),
            "plot_rdf": _yn(self.debye_plot_rdf.isChecked()),
            "rdf_max_distance": self.debye_rdf_max_distance.value(),
            "rdf_sample_size": self.debye_rdf_sample_s.value(),
            "rdf_access_dirs": self.debye_rdf_access_dirs.value(),
            "rdf_access_centers": self.debye_rdf_access_centers.value(),
            "compute_pair_distribution": _yn(self.debye_compute_pdf.isChecked()),
            "pdf_subset_size": self.debye_pdf_subset.value(),
            "pdf_bins": self.debye_pdf_bins.value(),
            "plot_pdf": _yn(self.debye_plot_pdf.isChecked()),
            "random_seed": self.debye_random_seed.value(),
            "output_dir": ".",
            "output_pattern": debye_output_path,
            "partial_output": debye_partial_path,
            "verbose": _yn(self.debye_verbose.isChecked()),
            "make_plot": _yn(self.debye_make_plot.isChecked()),
            "plot_partials": _yn(self.debye_plot_partials.isChecked()),
            "plot_filename": self.debye_plot_filename.text().strip() or "debye_plot.png",
            "plot_dpi": self.debye_plot_dpi.value(),
            "show_plot": "no",
        }
        _set_apply_flags(params, (
            ("apply_anomalous", self.apply_anom),
            ("apply_debye_waller", self.debye_apply_dw),
            ("apply_LP", self.debye_apply_lp),
            ("apply_instrumental_broadening", self.debye_apply_broadening),
            ("apply_accessible_fraction", self.debye_apply_acc_frac),
        ))
        if self.debye_species_mode.currentText() == "lammps_types":
            params["type_map"] = self.debye_type_map.text().strip()
        # two_theta_max and q_max are alternatives: two_theta_max always wins, so only
        # the selected one is written.
        if self.debye_use_qmax.isChecked():
            params["q_max"] = self.debye_qmax.value()
        else:
            params["two_theta_max"] = self.debye_two_theta_max.value()
        if self.debye_max_int_distance.value() > 0:
            params["max_interaction_distance"] = self.debye_max_int_distance.value()
        if self.debye_dw_factors.text().strip():
            params["debye_waller_factors"] = self.debye_dw_factors.text().strip()
        for e in self._get_anomalous_entries():
            params[f"fprime_{e['element']}"]  = e["fprime"]
            params[f"fdouble_{e['element']}"] = e["fdouble"]

        param_file_path = os.path.join(out_dir_path, "xrd_deb_input.txt")
        self._write_and_run(
            param_file_path, params,
            ("XRD", "XRD-Debye_Scattering.py"),
            out_dir_path,
            self._make_finish_callback("XRD Debye", self._plot_debye),
        )

    def _run_reciprocal(self) -> None:
        self.reciprocal_viz.clear()
        input_path = self.reciprocal_input.text().strip()
        out_dir_path = self.reciprocal_out_dir.text().strip()

        if not self._validate_input_file(input_path):
            return
        if not self.reciprocal_output.text().strip():
            self.logger.log_message("ERROR", "Please choose an output file first."); return
        if not self.reciprocal_plot_location.text().strip():
            self.logger.log_message("ERROR", "Please choose a plot output file first."); return
        if not self.reciprocal_atom_types.text().strip():
            self.logger.log_message(
                "ERROR", "Please list the atom types, in LAMMPS atom-type order."); return
        if not self._validate_output_dir(out_dir_path):
            return

        reciprocal_out  = self._resolve_to_outdir(self.reciprocal_output, out_dir_path)
        reciprocal_plot = self._resolve_to_outdir(self.reciprocal_plot_location, out_dir_path)

        params = {
            "structure_file": input_path, "output_file": reciprocal_out,
            "plot_file": reciprocal_plot,
            "wavelength": self.reciprocal_wavelength.value(), "2Theta": self.reciprocal_2theta.text(),
            "pbc": self.reciprocal_pbc.text(), "num_bins": self.reciprocal_bins.value(),
            "manual": self.reciprocal_manual.currentText(), "prd": self.reciprocal_prd.text(),
            "c": self.reciprocal_c.text(), "atom_types": self.reciprocal_atom_types.text(),
            "atom_type_mode": self.reciprocal_atom_type_mode.currentText(),
            "compatibility_mode": self.reciprocal_compat_mode.currentText(),
            "max_reciprocal_candidates": int(self.reciprocal_max_candidates.value()),
            "allow_large_reciprocal_grid": int(self.reciprocal_allow_large_grid.isChecked()),
            "plot_smoothing_sigma_deg": self.reciprocal_smoothing.value(),
            "LP": self.reciprocal_lp.value(), "echo": self.reciprocal_echo.value(),
            "plot": self.reciprocal_plot_spin.value(),
        }

        param_file_path = os.path.join(out_dir_path, "xrd_reciprocal_input.txt")
        self._write_and_run(
            param_file_path, params,
            ("XRD", "XRD-ReciprocalSum.py"),
            out_dir_path,
            self._make_finish_callback("XRD Reciprocal", self._plot_reciprocal),
        )

    # ------------------------------------------------------------------ plot methods

    def _plot_kin(self) -> None:
        try:
            def plot_func(fig):
                fig.clear(); fig.patch.set_facecolor("white")
                ax = fig.subplots()
                out_dir_path = self.kin_out_dir.text().strip()
                curve_name = self.kin_curve_file.text().strip() or "xrd_curve.txt"
                peaks_name = self.kin_peaks_file.text().strip() or "xrd_peaks_display.txt"
                curve = np.loadtxt(os.path.join(out_dir_path, curve_name))
                two_theta, intensity = curve[:, 0], curve[:, 1]
                ax.plot(two_theta, intensity, linewidth=2.0, color="royalblue", label="XRD Pattern")

                peak_tth, peak_hkl = [], []
                with open(os.path.join(out_dir_path, peaks_name), "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith("#") or s.startswith("-"): continue
                        parts = s.split()
                        if len(parts) < 6: continue
                        try: peak_tth.append(float(parts[1]))
                        except ValueError: continue
                        # ID 2Theta Intensity d Multiplicity hkl — the hkl field itself may
                        # hold several comma-separated indices, so keep every trailing token
                        peak_hkl.append(" ".join(parts[5:]))
                peak_tth = np.asarray(peak_tth, dtype=float)

                x_left, x_right = self.kin_two_theta_min.value(), self.kin_two_theta_max.value()
                ax.set_xlim(x_left, x_right)
                y_max = float(np.max(intensity)) if intensity.size else 1.0
                ax.set_ylim(0.0, y_max * 1.08 if y_max > 0 else 1.0)

                if peak_tth.size:
                    sidx = np.argsort(two_theta)
                    tts, ints = two_theta[sidx], intensity[sidx]
                    in_range = (peak_tth >= tts[0]) & (peak_tth <= tts[-1])
                    pu, hu = peak_tth[in_range], [h for h, ok in zip(peak_hkl, in_range) if ok]
                    py = np.interp(pu, tts, ints)
                    ax.scatter(pu, py, color="red", s=120, marker="^", alpha=0.9, zorder=6)
                    ymin, ymax = ax.get_ylim(); dy = (ymax - ymin) * 0.02
                    for x, y, lab in zip(pu, py, hu):
                        if x_left <= x <= x_right:
                            ax.text(x, y+dy, lab, ha="center", va="bottom", fontsize=12,
                                    fontweight="bold", color="darkgreen", zorder=7)

                ax.set_xlabel("2θ (degrees)", fontsize=12); ax.set_ylabel("Intensity (arbitrary units)", fontsize=12)
                ax.set_title(f"XRD Pattern - {self.kin_wavelength.currentText()}", fontsize=14, fontweight="bold")
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True)); ax.yaxis.offsetText.set_fontsize(12)
                ax.grid(True, linestyle="--", alpha=0.3); ax.legend(loc="upper right")
                annotation = (f"Wavelength: {self.kin_wavelength.currentText()}\n"
                               f"2θ range: {x_left:.1f}–{x_right:.1f}°\n"
                               f"Curve points: {len(two_theta):,}\nPeaks: {len(peak_tth):,}")
                ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", fontsize=10, color="black",
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", linewidth=1, alpha=0.8))
                style_axis_light(ax); fig.tight_layout()
            self.kin_viz.plot(plot_func)
        except Exception as e:
            self.logger.log_message("ERROR", f"Error plotting XRD pattern: {e}")

    @staticmethod
    def _read_partial_labels(path: str) -> list[str]:
        """Return the species-pair column labels from a debye_partials header."""
        labels: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                if "2theta(deg)" in line:
                    tokens = line.lstrip("#").split()
                    # drop the 'q(1/A)' and '2theta(deg)' column names
                    labels = tokens[2:]
        return labels

    def _debye_out_dir(self) -> str:
        return self.debye_out_dir.text().strip()

    def _debye_rdf_page(self, fig) -> None:
        """Partial RDFs g_ab(r), one curve per species pair.

        Reads the rdf_<a>-<b>.txt files the run writes, the same data behind the
        script's own combined_partial_rdfs.png.
        """
        fig.clear(); fig.patch.set_facecolor("white"); ax = fig.subplots()
        n_curves = 0
        for rdf_file in sorted(glob.glob(os.path.join(self._debye_out_dir(), "rdf_*.txt"))):
            try:
                data = np.loadtxt(rdf_file, comments="#")
            except Exception as e:
                self.logger.log_message("ERROR", f"Failed to read '{rdf_file}': {e}")
                continue
            if data.ndim == 1:
                data = data[None, :]
            if data.size == 0 or data.shape[1] < 2:
                continue
            label = os.path.basename(rdf_file)[len("rdf_"):-len(".txt")]
            ax.plot(data[:, 0], data[:, 1], linewidth=1.6, alpha=0.9, label=label)
            n_curves += 1
        if not n_curves:
            ax.text(0.5, 0.5, "No partial RDF data found", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
            style_axis_light(ax); fig.tight_layout(); return
        ax.axhline(1.0, color="gray", linewidth=1.0, linestyle=":")
        ax.set_xlabel("r (Å)", fontsize=12)
        ax.set_ylabel("g$_{ab}$(r)", fontsize=12)
        ax.set_title("Partial Radial Distribution Functions", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)
        style_axis_light(ax); fig.tight_layout()

    def _debye_pdf_page(self, path):
        """Return a page function plotting the pair-distance distribution at *path*."""
        def plot_func(fig):
            fig.clear(); fig.patch.set_facecolor("white"); ax = fig.subplots()
            data = np.loadtxt(path, comments="#")
            if data.ndim == 1:
                data = data[None, :]
            if data.size == 0 or data.shape[1] < 2:
                ax.text(0.5, 0.5, "No pair-distance data found", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12,
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
                style_axis_light(ax); fig.tight_layout(); return
            # columns: r(A)  P(r)  n(r) -- P(r) is a probability density, n(r) the
            # cumulative coordination number, so they need separate axes.
            r, p_r = data[:, 0], data[:, 1]
            ax.fill_between(r, p_r, color="royalblue", alpha=0.35)
            ax.plot(r, p_r, linewidth=1.8, color="royalblue")
            ax.set_ylabel("P(r)", fontsize=12)
            ax.set_title("Pair Distance Distribution", fontsize=14, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.text(0.02, 0.98, f"Bins: {len(r):,}", transform=ax.transAxes, va="top",
                    fontsize=10, color="black",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray",
                              linewidth=1, alpha=0.8))
            if data.shape[1] > 2:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
                ax_cn = fig.add_subplot(2, 1, 2, sharex=ax)
                ax.set_subplotspec(fig.add_gridspec(2, 1)[0])
                ax_cn.plot(r, data[:, 2], linewidth=1.8, color="firebrick")
                ax_cn.set_xlabel("r (Å)", fontsize=12)
                ax_cn.set_ylabel("cumulative n(r)", fontsize=12)
                ax_cn.grid(True, linestyle="--", alpha=0.3)
                style_axis_light(ax_cn)
            else:
                ax.set_xlabel("r (Å)", fontsize=12)
            style_axis_light(ax); fig.tight_layout()
        return plot_func

    def _plot_debye(self) -> None:
        try:
            def plot_func(fig):
                fig.clear(); fig.patch.set_facecolor("white"); ax = fig.subplots()
                out_dir_path = self.debye_out_dir.text().strip()
                pattern_path = self.debye_output.text().strip()
                if not os.path.isabs(pattern_path):
                    pattern_path = os.path.join(out_dir_path, pattern_path)
                # columns: q(1/A)  2theta(deg)  I(arb.)  -> plot column 2 vs column 3
                data = np.loadtxt(pattern_path, comments="#")
                if data.size == 0:
                    ax.text(0.5, 0.5, "No Debye data found", transform=ax.transAxes, ha="center", va="center",
                            fontsize=12, bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
                    style_axis_light(ax); fig.tight_layout(); return
                if data.ndim == 1: data = data[None, :]
                two_theta, intensity = data[:, 1], data[:, 2]
                ax.plot(two_theta, intensity, linewidth=2.0, color="royalblue", label="Debye")
                n_partials = 0
                if self.debye_plot_partials.isChecked():
                    partials_path = self.debye_partial_output.text().strip()
                    if partials_path and not os.path.isabs(partials_path):
                        partials_path = os.path.join(out_dir_path, partials_path)
                    if partials_path and os.path.isfile(partials_path):
                        try:
                            labels = self._read_partial_labels(partials_path)
                            pdat = np.loadtxt(partials_path, comments="#")
                            if pdat.ndim == 1: pdat = pdat[None, :]
                            x = pdat[:, 1]
                            for col in range(2, pdat.shape[1]):
                                lab = labels[col - 2] if col - 2 < len(labels) else f"partial {col - 1}"
                                ax.plot(x, pdat[:, col], linewidth=1.6, alpha=0.9, label=lab)
                                n_partials += 1
                        except Exception as e:
                            self.logger.log_message("ERROR", f"Failed to read partials '{partials_path}': {e}")
                ax.set_xlabel("2θ (deg)", fontsize=12); ax.set_ylabel("Intensity (arb. units)", fontsize=12)
                ax.set_title("Debye Scattering Pattern", fontsize=14, fontweight="bold")
                ax.grid(True, linestyle="--", alpha=0.3); ax.legend(loc="upper right", fontsize=10 if n_partials else 11)
                lines = [f"Points: {len(two_theta):,}"]
                if self.debye_plot_partials.isChecked(): lines.append(f"Partials plotted: {n_partials:,}")
                ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=10, color="black",
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", linewidth=1, alpha=0.8))
                style_axis_light(ax); fig.tight_layout()

            # A run can also write partial RDFs and a pair-distance histogram.
            # Those used to reach the disk and nowhere else; register them as
            # extra pages when their data files are present.
            pages = [("Debye scattering pattern", plot_func)]
            out_dir_path = self._debye_out_dir()
            if glob.glob(os.path.join(out_dir_path, "rdf_*.txt")):
                pages.append(("Partial RDFs", self._debye_rdf_page))
            pdf_path = os.path.join(out_dir_path, "pair_distance_distribution.txt")
            if os.path.isfile(pdf_path):
                pages.append(("Pair distance distribution", self._debye_pdf_page(pdf_path)))
            self.debye_viz.plot_pages(pages)
        except Exception as e:
            self.logger.log_message("ERROR", f"Error plotting Debye pattern: {e}")

    def _plot_reciprocal(self) -> None:
        try:
            def plot_func(fig):
                fig.clear(); fig.patch.set_facecolor("white"); ax = fig.subplots()
                data = np.loadtxt(self.reciprocal_output.text().strip(), comments="#")
                if data.size == 0:
                    ax.text(0.5, 0.5, "No XRD data found", transform=ax.transAxes, ha="center", va="center",
                            fontsize=12, bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
                    style_axis_light(ax); fig.tight_layout(); return
                if data.ndim == 1: data = data[None, :]
                # columns: Bin  Coord(2theta)  Count  Count/Total -> plot column 2 vs column 4
                two_theta, counts = data[:, 1].astype(float), data[:, 3].astype(float)
                valid = data[(two_theta > 0) & (counts > 0)]
                if valid.size == 0:
                    ax.text(0.5, 0.5, "No valid peaks to plot", transform=ax.transAxes, ha="center", va="center",
                            fontsize=12, bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
                    style_axis_light(ax); fig.tight_layout(); return
                valid = valid[np.argsort(valid[:, 1])]
                theta, inten = valid[:, 1].astype(float), valid[:, 3].astype(float)
                sigma = float(self.reciprocal_smoothing.value())
                if sigma > 0 and theta.size > 5:          # 0 = off, as the tooltip says
                    xs = np.linspace(theta[0], theta[-1], 1000)
                    ys = sum(I * np.exp(-((xs - t) ** 2) / (2.0 * sigma ** 2)) for t, I in zip(theta, inten))
                    ax.plot(xs, ys, linewidth=2.5, alpha=0.8, color="royalblue", label="Smoothed Pattern")
                ax.vlines(theta, 0.0, inten, colors="red", linewidth=2.0, alpha=0.8, label="Calculated Peaks")
                ax.plot(theta, inten, linestyle="None", marker="o", markersize=6, alpha=0.9, color="red")
                ax.set_xlabel("2θ (degrees)", fontsize=18, fontweight="bold")
                ax.set_ylabel("Intensity (a.u.)", fontsize=18, fontweight="bold")
                ax.set_title(f"XRD Pattern (λ = {self.reciprocal_wavelength.value():.4f} Å)",
                             fontsize=20, fontweight="bold")
                ax.tick_params(axis="both", which="major", labelsize=14, width=1.5, length=8, pad=8)
                for tick in ax.get_xticklabels() + ax.get_yticklabels(): tick.set_fontweight("bold")
                ax.grid(True, alpha=0.4, linestyle="--", linewidth=0.8); ax.legend(fontsize=14, framealpha=0.9)
                for spine in ax.spines.values(): spine.set_linewidth(1.5)
                style_axis_light(ax); fig.tight_layout(pad=2.0)
            self.reciprocal_viz.plot(plot_func)
        except Exception as e:
            self.logger.log_message("ERROR", f"Error plotting reciprocal XRD: {e}")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = XrayWindow()
    window.show()
    sys.exit(app.exec())
