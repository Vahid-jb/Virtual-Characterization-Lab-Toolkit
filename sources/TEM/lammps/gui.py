# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
SAED / TEM GUI Application.

Provides tools for computing Selected-Area Electron Diffraction (SAED) patterns
from LAMMPS structure files, and for rendering VTK diffraction volumes.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit,
    QGroupBox, QCheckBox,
    QTabWidget, QFormLayout,
    QScrollArea, QStackedWidget,
)
from PySide6.QtCore import Qt

from VCL_utils.theme import apply_dark_theme, make_action_button, make_primary_button, make_stop_button, add_run_row, NoWheelSpinBox, NoWheelDoubleSpinBox, NoWheelComboBox
from VCL_utils.general import browse_file, create_file, browse_directory, get_script_path
from VCL_utils.validators import (
    apply_validator,
    ElementListValidator,
    IntTripletValidator,
    FloatTripletValidator,
    ScientificFloatValidator,
)
from VCL_utils.widgets import ImageViewer
from VCL_utils.base_window import BaseModuleWindow

_XYZ_FILTER = "XYZ files (*.xyz);;All files (*)"
_VTK_FILTER = "VTK files (*.vtk);;All files (*)"
_IMG_FILTER = "Images (*.bmp *.jpg *.png *.tiff);;All files (*)"


class TEMWindow(BaseModuleWindow):
    """Main Window for SAED (Selected-Area Electron Diffraction) analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SAED")
        self.resize(1000, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        header = QLabel("SAED")
        header.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(header)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self._create_saed_tab()
        self._create_vtk_tab()

        # Visualization Stack
        self.viz_stack = QStackedWidget()

        self.saed_viz = QLabel("Visualization is done using the VTK visualizer")
        self.saed_viz.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.saed_viz.setStyleSheet("color: #777; font-size: 14px; border: 1px solid #444; background-color: #2b2b2b;")
        self.viz_stack.addWidget(self.saed_viz)

        self.vtk_viz = ImageViewer()
        self.viz_stack.addWidget(self.vtk_viz)

        self.tabs.currentChanged.connect(self.viz_stack.setCurrentIndex)

        # Register buttons
        self._calc_btns = [self.saed_calc_btn, self.vtk_calc_btn]
        self._stop_btns = [self.saed_stop_btn, self.vtk_stop_btn]

        # Public interface for PolyCycleMainWindow
        self.control_widget = self.tabs
        self.viz_widget = self.viz_stack

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _create_saed_tab(self) -> None:
        """Creates the LAMMPS SAED computation tab."""
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); layout = QVBoxLayout(content); scroll.setWidget(content)

        group = QGroupBox("LAMMPS"); form = QFormLayout(group)

        self.saed_input = QLineEdit()
        b1 = make_action_button("Browse")
        b1.clicked.connect(lambda: browse_file(self, self.saed_input, _XYZ_FILTER))
        r1 = QHBoxLayout(); r1.addWidget(self.saed_input); r1.addWidget(b1)
        form.addRow("Input (.xyz):", r1)

        self.saed_output = QLineEdit()
        b2 = make_action_button("Browse")
        b2.clicked.connect(lambda: create_file(self, self.saed_output, "VTK files (*.vtk);;All files (*)"))
        r2 = QHBoxLayout(); r2.addWidget(self.saed_output); r2.addWidget(b2)
        form.addRow("Output (vtk):", r2)

        self.saed_out_dir = QLineEdit()
        btn_out_dir = make_action_button("Browse")
        btn_out_dir.clicked.connect(lambda: browse_directory(self, self.saed_out_dir))
        r_out_dir = QHBoxLayout()
        r_out_dir.addWidget(self.saed_out_dir); r_out_dir.addWidget(btn_out_dir)
        form.addRow("Output directory:", r_out_dir)

        self.saed_wavelength = NoWheelDoubleSpinBox()
        self.saed_wavelength.setRange(1e-6, 100.0); self.saed_wavelength.setDecimals(6)
        self.saed_wavelength.setValue(0.0251)
        self.saed_wavelength.setToolTip("Electron wavelength in Å (e.g. 0.0251 Å for 200 kV electrons)")
        form.addRow("Wavelength (Å):", self.saed_wavelength)

        self.saed_atom_types = QLineEdit()
        self.saed_atom_types.setPlaceholderText("e.g. Mo")
        self.saed_atom_types.setText("Mo")
        self.saed_atom_types.setToolTip("Comma-separated element symbols present in the structure")
        form.addRow("Atom types:", self.saed_atom_types)
        apply_validator(self.saed_atom_types, ElementListValidator())

        self.saed_kmax = NoWheelDoubleSpinBox()
        self.saed_kmax.setRange(0.0, 1e6); self.saed_kmax.setValue(1.70)
        self.saed_kmax.setToolTip("Maximum k-vector magnitude (reciprocal-space cutoff, Å⁻¹)")
        form.addRow("k max (Å⁻¹):", self.saed_kmax)

        self.saed_zone = QLineEdit()
        self.saed_zone.setPlaceholderText("e.g. 0, 0, 1")
        self.saed_zone.setText("0, 0, 1")
        self.saed_zone.setToolTip(chr(10).join([
            "Zone axis as three integers h, k, l.",
            "0, 0, 0 computes the full 3D reciprocal volume instead of the",
            "Ewald-sphere shell, which needs ~100x more mesh nodes",
        ]))
        form.addRow("Zone [h,k,l]:", self.saed_zone)
        apply_validator(self.saed_zone, IntTripletValidator())

        self.saed_drewald = NoWheelDoubleSpinBox()
        self.saed_drewald.setRange(1e-6, 1e6); self.saed_drewald.setDecimals(4)
        self.saed_drewald.setValue(0.01)
        self.saed_drewald.setToolTip("Ewald sphere radius deviation tolerance (Å⁻¹)")
        form.addRow("d Ewald:", self.saed_drewald)

        self.saed_c = QLineEdit()
        self.saed_c.setPlaceholderText("e.g. 0.025, 0.025, 0.025")
        self.saed_c.setText("0.025, 0.025, 0.025")
        self.saed_c.setToolTip(chr(10).join([
            "Reciprocal-mesh scale factors along x, y, z.",
            "The mesh spacing is dK = c / L for box length L; in Manual mode dK = c (Å⁻¹).",
            "The node count grows as (k max · L / c)³, so small c is expensive",
        ]))
        form.addRow("Mesh scale c:", self.saed_c)
        apply_validator(self.saed_c, FloatTripletValidator())

        self.saed_manual = QCheckBox("Manual")
        self.saed_manual.setToolTip("Use manually specified periodic-box dimensions (prd) instead of auto-detect")
        form.addRow(self.saed_manual)

        self.saed_prd = QLineEdit()
        self.saed_prd.setPlaceholderText("e.g. 30.0, 10.0, 70.0")
        self.saed_prd.setText("30.0, 10.0, 70.0")
        self.saed_prd.setToolTip("Periodic box dimensions x, y, z (Å) — used when 'manual' is checked")
        form.addRow("Box dimensions (Å):", self.saed_prd)
        apply_validator(self.saed_prd, FloatTripletValidator())

        self.saed_pbc = QLineEdit()
        self.saed_pbc.setPlaceholderText("e.g. 1, 0, 0")
        self.saed_pbc.setText("1, 0, 0")
        self.saed_pbc.setToolTip("Periodic boundary conditions along x, y, z (1 = periodic, 0 = non-periodic)")
        form.addRow("Periodic BC:", self.saed_pbc)
        apply_validator(self.saed_pbc, IntTripletValidator())

        self.saed_engine = NoWheelComboBox()
        self.saed_engine.addItems(["auto", "numba", "numpy", "reference"])
        self.saed_engine.setCurrentText("auto")
        self.saed_engine.setToolTip(chr(10).join([
            "Structure-factor kernel. All engines evaluate the same LAMMPS",
            "expression and give identical results for every physically",
            "significant intensity; they differ only in speed and in how",
            "the atom sum is associated.",
            "",
            "auto      - numba if installed, else numpy (recommended)",
            "numba     - multi-threaded; bit-identical to LAMMPS",
            "numpy     - vectorized fallback, no extra dependency",
            "reference - literal per-point transcription; slowest, for auditing",
        ]))
        form.addRow("Engine:", self.saed_engine)

        self.saed_max_mesh_nodes = NoWheelSpinBox()
        self.saed_max_mesh_nodes.setRange(0, 2_000_000_000)
        self.saed_max_mesh_nodes.setSingleStep(10_000_000)
        self.saed_max_mesh_nodes.setGroupSeparatorShown(True)
        self.saed_max_mesh_nodes.setSpecialValueText("No limit")
        self.saed_max_mesh_nodes.setValue(1_500_000_000)
        self.saed_max_mesh_nodes.setToolTip(chr(10).join([
            "Upper limit on the estimated number of reciprocal-mesh nodes.",
            "The estimate is checked before the mesh is built; above the limit the",
            "run stops and reports the node count, the memory it would need and",
            "how to reduce it (zone axis, larger c, smaller k max).",
            "Each node costs ~20 bytes; 0 disables the check.",
            "The default is high enough to let the default mesh scale c run on a",
            "medium box, which can need tens of GB - lower it if that is not",
            "memory you have",
        ]))
        form.addRow("Max mesh nodes:", self.saed_max_mesh_nodes)

        self.saed_echo = QCheckBox("Echo"); self.saed_echo.setChecked(True)
        form.addRow(self.saed_echo)

        layout.addWidget(group)

        self.saed_calc_btn = make_primary_button("Calculate")
        self.saed_stop_btn = make_stop_button()
        self.saed_calc_btn.clicked.connect(self._run_saed)
        self.saed_stop_btn.clicked.connect(self._stop_calculation)
        add_run_row(layout, self.saed_calc_btn, self.saed_stop_btn)
        layout.addStretch()
        self.tabs.addTab(scroll, "LAMMPS")

    def _create_vtk_tab(self) -> None:
        """Creates the VTK visualizer tab."""
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); layout = QVBoxLayout(content); scroll.setWidget(content)

        group = QGroupBox("VTK visualizer"); form = QFormLayout(group)

        self.vtk_input = QLineEdit()
        b1 = make_action_button("Browse")
        b1.clicked.connect(lambda: browse_file(self, self.vtk_input, _VTK_FILTER))
        r1 = QHBoxLayout(); r1.addWidget(self.vtk_input); r1.addWidget(b1)
        form.addRow("VTK File (*.vtk):", r1)

        self.vtk_output = QLineEdit()
        b2 = make_action_button("Browse")
        b2.clicked.connect(lambda: create_file(self, self.vtk_output, _IMG_FILTER))
        r2 = QHBoxLayout(); r2.addWidget(self.vtk_output); r2.addWidget(b2)
        form.addRow("Output image:", r2)

        self.vtk_out_dir = QLineEdit()
        btn_out_dir = make_action_button("Browse")
        btn_out_dir.clicked.connect(lambda: browse_directory(self, self.vtk_out_dir))
        r_out_dir = QHBoxLayout()
        r_out_dir.addWidget(self.vtk_out_dir); r_out_dir.addWidget(btn_out_dir)
        form.addRow("Output directory:", r_out_dir)

        self.vtk_iso_lower = NoWheelDoubleSpinBox()
        self.vtk_iso_lower.setRange(-1e12, 1e12); self.vtk_iso_lower.setValue(0)
        self.vtk_iso_lower.setToolTip("Lower isosurface threshold for the VTK volume")
        form.addRow("Iso lower:", self.vtk_iso_lower)

        self.vtk_iso_upper = QLineEdit()
        self.vtk_iso_upper.setPlaceholderText("e.g. 1e37")
        self.vtk_iso_upper.setText("1e37")
        self.vtk_iso_upper.setToolTip("Upper isosurface threshold (use 1e37 for no upper limit)")
        form.addRow("Iso upper:", self.vtk_iso_upper)
        apply_validator(self.vtk_iso_upper, ScientificFloatValidator())

        self.vtk_pseudocolor_min = NoWheelDoubleSpinBox()
        self.vtk_pseudocolor_min.setRange(-1e12, 1e12); self.vtk_pseudocolor_min.setValue(1.0)
        form.addRow("Pseudocolor min:", self.vtk_pseudocolor_min)

        self.vtk_sphere_origin = QLineEdit()
        self.vtk_sphere_origin.setPlaceholderText("e.g. 0, 0, 39.84063")
        self.vtk_sphere_origin.setText("0, 0, 39.84063")
        self.vtk_sphere_origin.setToolTip(chr(10).join([
            "Centre of the Ewald sphere in reciprocal space (x, y, z): 1/λ along the zone axis.",
            "The default matches zone 0, 0, 1 at λ = 0.0251 Å",
        ]))
        form.addRow("Sphere origin:", self.vtk_sphere_origin)
        apply_validator(self.vtk_sphere_origin, FloatTripletValidator())

        self.vtk_sphere_radius = NoWheelDoubleSpinBox()
        self.vtk_sphere_radius.setRange(0.0, 1e6); self.vtk_sphere_radius.setDecimals(5)
        self.vtk_sphere_radius.setValue(39.84063)
        self.vtk_sphere_radius.setToolTip("Ewald sphere radius (= 1/λ)")
        form.addRow("Sphere radius:", self.vtk_sphere_radius)

        self.vtk_view_normal = QLineEdit()
        self.vtk_view_normal.setPlaceholderText("e.g. 0, 0, -1")
        self.vtk_view_normal.setText("0, 0, -1")
        self.vtk_view_normal.setToolTip("Camera view direction vector (x, y, z); look against the zone axis")
        form.addRow("View normal:", self.vtk_view_normal)
        apply_validator(self.vtk_view_normal, FloatTripletValidator())

        self.vtk_view_up = QLineEdit()
        self.vtk_view_up.setPlaceholderText("0, 1, 0")
        self.vtk_view_up.setText("0, 1, 0")
        self.vtk_view_up.setToolTip("Camera 'up' direction vector (x, y, z)")
        form.addRow("View up:", self.vtk_view_up)
        apply_validator(self.vtk_view_up, FloatTripletValidator())

        self.vtk_resolution = NoWheelSpinBox()
        self.vtk_resolution.setRange(10, 10000)
        self.vtk_resolution.setValue(1200)
        form.addRow("Resolution:", self.vtk_resolution)

        self.vtk_show_3d_axes = QCheckBox("Show 3D axes")
        form.addRow(self.vtk_show_3d_axes)

        self.vtk_show_2d_axes = QCheckBox("Show 2D axes")
        form.addRow(self.vtk_show_2d_axes)

        self.vtk_show_user_info = QCheckBox("Show user info")
        form.addRow(self.vtk_show_user_info)

        self.vtk_show_database_info = QCheckBox("Show database info")
        form.addRow(self.vtk_show_database_info)

        self.vtk_show_legend = QCheckBox("Show legend")
        form.addRow(self.vtk_show_legend)

        layout.addWidget(group)

        self.vtk_calc_btn = make_primary_button("Calculate")
        self.vtk_stop_btn = make_stop_button()
        self.vtk_calc_btn.clicked.connect(self._run_vtk)
        self.vtk_stop_btn.clicked.connect(self._stop_calculation)
        add_run_row(layout, self.vtk_calc_btn, self.vtk_stop_btn)
        layout.addStretch()
        self.tabs.addTab(scroll, "VTK visualizer")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _run_saed(self) -> None:
        """Validate inputs, write parameter file, and launch the SAED computation."""
        input_path = self.saed_input.text().strip()
        out_dir_path = self.saed_out_dir.text().strip()

        if input_path and not os.path.isfile(input_path):
            self.logger.log_message("ERROR", f"File not found:\n{input_path}")
            return
        if not self._validate_output_dir(out_dir_path):
            return

        # The output name the user chose was previously collected and then
        # discarded in favour of a hardcoded "saed_pattern". Honour it.
        vtk_name = self.saed_output.text().strip()
        vtk_base = os.path.splitext(os.path.basename(vtk_name))[0] if vtk_name else "saed_pattern"
        if vtk_base.endswith("_0"):
            vtk_base = vtk_base[:-2]          # a picked "name_0.vtk" means base "name"

        params = {
            "structure_file": input_path,
            "vtk_base": vtk_base,
            "vtk_index": 0,
            "engine": self.saed_engine.currentText(),
            "max_mesh_nodes": self.saed_max_mesh_nodes.value(),
            "wavelength": self.saed_wavelength.value(),
            "atom_types": self.saed_atom_types.text(),
            "kmax": self.saed_kmax.value(),
            "zone": self.saed_zone.text(),
            "drewald": self.saed_drewald.value(),
            "c": self.saed_c.text(),
            "manual": "True" if self.saed_manual.isChecked() else "False",
            "prd": self.saed_prd.text(),
            "pbc": self.saed_pbc.text(),
            "echo": "True" if self.saed_echo.isChecked() else "False",
        }

        param_file_path = os.path.join(out_dir_path, "tem_lammps_input.txt")
        self._write_and_run(
            param_file_path, params,
            ("TEM", "lammps", "compute_saed.py"),
            out_dir_path,
            self._make_finish_callback("SAED calculation"),
            extra_args=["--input"],
        )
        # Pre-fill the visualizer with the file this run will produce, so the
        # two tabs chain without the user hunting for it on disk.
        produced = os.path.join(out_dir_path, f"{vtk_base}_0.vtk")
        self.vtk_input.setText(produced)

    def _run_vtk(self) -> None:
        """Validate inputs, write parameter file, and launch the VTK renderer."""
        self.vtk_viz.clear()
        input_path = self.vtk_input.text().strip()
        out_dir_path = self.vtk_out_dir.text().strip()

        if not self._validate_input_file(input_path):
            return
        if not self.vtk_output.text().strip():
            self.logger.log_message("ERROR", "Please choose an output file first."); return
        if not self._validate_output_dir(out_dir_path):
            return

        vtk_out_file = self._resolve_to_outdir(self.vtk_output, out_dir_path)

        params = {
            "vtk_file": input_path,
            "output_file": vtk_out_file,
            "iso_lower": self.vtk_iso_lower.value(),
            "iso_upper": self.vtk_iso_upper.text(),
            "pseudocolor_min": self.vtk_pseudocolor_min.value(),
            "sphere_origin": self.vtk_sphere_origin.text(),
            "sphere_radius": self.vtk_sphere_radius.value(),
            "view_normal": self.vtk_view_normal.text(),
            "view_up": self.vtk_view_up.text(),
            "resolution": self.vtk_resolution.value(),
            "show_3d_axes": "True" if self.vtk_show_3d_axes.isChecked() else "False",
            "show_2d_axes": "True" if self.vtk_show_2d_axes.isChecked() else "False",
            "show_user_info": "True" if self.vtk_show_user_info.isChecked() else "False",
            "show_database_info": "True" if self.vtk_show_database_info.isChecked() else "False",
            "show_legend": "True" if self.vtk_show_legend.isChecked() else "False",
        }

        param_file_path = os.path.join(out_dir_path, "tem_vtk_input.txt")
        self._write_and_run(
            param_file_path, params,
            ("TEM", "lammps", "vtk_to_fig.py"),
            out_dir_path,
            self._on_vtk_finished,
            extra_args=["--input"],
        )

    def _on_vtk_finished(self, exit_code: int, exit_status: object) -> None:
        self._set_running(False)
        if exit_code == 0:
            self.logger.log_message("INFO", "VTK render finished successfully.")
            self.vtk_viz.set_image(self.vtk_output.text().strip())
        else:
            self.logger.log_message("ERROR", f"VTK render failed with code {exit_code}.")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = TEMWindow()
    window.show()
    sys.exit(app.exec())
