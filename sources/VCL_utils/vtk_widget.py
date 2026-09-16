# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
VCL_utils/vtk_widget.py
=======================

VTK rendering helpers for the SAED / TEM visualization panel.

Provides:
- ``VTKRenderParams``: dataclass collecting all parameters needed for a single
  diffraction-volume render.
- ``VTKQtViewer``: a ``QWidget`` that hosts a VTK render window and exposes
  ``render_vtk``, ``clear``, and ``save_png`` methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QTimer

import vtk
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
import vtkmodules.vtkInteractionStyle  # noqa: F401

try:
    from vtkmodules.qt.QVTKOpenGLNativeWidget import QVTKOpenGLNativeWidget as _VTKWidget
except Exception:  # pragma: no cover
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as _VTKWidget


@dataclass
class VTKRenderParams:
    vtk_file: str
    iso_lower: float = 0.0
    iso_upper: float = 1e37
    pc_min: float = 1.0

    sphere_origin: Tuple[float, float, float] = (39.84063, 0.0, 0.0)
    sphere_radius: float = 39.84063

    view_normal: Tuple[float, float, float] = (-1.0, 0.0, 0.0)
    view_up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    parallel_projection: bool = True

    show_axes: bool = False
    show_scalar_bar: bool = False


class VTKQtViewer(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = _VTKWidget(self)
        layout.addWidget(self.vtk_widget)

        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.SetMultiSamples(0)

        self.renderer = vtk.vtkRenderer()
        self.render_window.AddRenderer(self.renderer)
        self.renderer.SetBackground(0.15, 0.15, 0.15)

        self.interactor = self.render_window.GetInteractor()
        if self.interactor is not None:
            self.interactor.Initialize()

        self._refs = []
        self._last_poly = None  # keep last output around

    def clear(self) -> None:
        self.renderer.RemoveAllViewProps()
        self._refs.clear()
        self._last_poly = None
        if self.isVisible():
            self.render_window.Render()

    def render_vtk(self, p: VTKRenderParams) -> None:
        # Delay to ensure GL context exists
        QTimer.singleShot(0, lambda: self._render_impl(p))

    def _render_impl(self, p: VTKRenderParams) -> None:
        self.clear()

        # Read the STRUCTURED_POINTS dataset from the VTK file.
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(p.vtk_file)
        reader.Update()

        img = reader.GetOutput()
        if img is None or img.GetNumberOfPoints() == 0:
            raise RuntimeError("Failed to read STRUCTURED_POINTS dataset or dataset is empty.")

        arr = img.GetPointData().GetArray("intensity")
        if arr is None:
            raise RuntimeError("Scalar array 'intensity' not found in POINT_DATA.")
        img.GetPointData().SetActiveScalars("intensity")

        # Clamp values below pc_min (the dataset contains many -1 sentinel values).
        clamp = vtk.vtkImageThreshold()
        clamp.SetInputConnection(reader.GetOutputPort())
        clamp.ThresholdByLower(float(p.pc_min))
        clamp.ReplaceInOn()
        clamp.SetInValue(float(p.pc_min))
        clamp.ReplaceOutOff()
        clamp.SetOutputScalarTypeToFloat()
        clamp.Update()

        # Zero-out voxels outside the requested isosurface range.
        iso = vtk.vtkImageThreshold()
        iso.SetInputConnection(clamp.GetOutputPort())
        iso.ThresholdBetween(float(p.iso_lower), float(p.iso_upper))
        iso.ReplaceInOff()
        iso.ReplaceOutOn()
        iso.SetOutValue(0.0)
        iso.SetOutputScalarTypeToFloat()
        iso.Update()

        # Restrict the volume to the spherical region of interest.
        sphere = vtk.vtkSphere()
        sphere.SetCenter(*p.sphere_origin)
        sphere.SetRadius(float(p.sphere_radius))

        extract = vtk.vtkExtractGeometry()
        extract.SetInputConnection(iso.GetOutputPort())
        extract.SetImplicitFunction(sphere)
        extract.ExtractInsideOn()
        extract.ExtractBoundaryCellsOn()
        extract.Update()

        # Convert the extracted UnstructuredGrid to PolyData for the mapper.
        geom = vtk.vtkGeometryFilter()
        geom.SetInputConnection(extract.GetOutputPort())
        geom.Update()

        poly = geom.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("Nothing to render after clamp + iso + sphere selection.")

        # Ensure 'intensity' is available as POINT scalars on the PolyData.
        # Extraction may have moved it to CELL data, so convert it back if needed.
        point_int = poly.GetPointData().GetArray("intensity")
        cell_int = poly.GetCellData().GetArray("intensity")

        if point_int is None and cell_int is not None:
            c2p = vtk.vtkCellDataToPointData()
            c2p.SetInputData(poly)
            c2p.PassCellDataOff()
            c2p.Update()
            poly = c2p.GetOutput()

            point_int = poly.GetPointData().GetArray("intensity")
            if point_int is None and poly.GetPointData().GetScalars() is not None:
                poly.GetPointData().GetScalars().SetName("intensity")
                point_int = poly.GetPointData().GetArray("intensity")

            self._refs.append(c2p)

        if point_int is None:
            # Fallback: rename active scalars if present
            scal = poly.GetPointData().GetScalars()
            if scal is not None:
                scal.SetName("intensity")
                poly.GetPointData().SetActiveScalars("intensity")
                point_int = poly.GetPointData().GetArray("intensity")

        if point_int is None:
            raise RuntimeError(
                "After conversion to PolyData, no point scalar array 'intensity' is available "
                "(scalar coloring would be disabled)."
            )

        poly.GetPointData().SetActiveScalars("intensity")
        self._last_poly = poly

        # Set up the mapper now that point scalars are guaranteed to exist.
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetScalarVisibility(True)
        mapper.SetScalarModeToUsePointData()     # robust for PolyData
        mapper.SelectColorArray("intensity")
        mapper.SetColorModeToMapScalars()

        smin, smax = point_int.GetRange()
        smin = max(float(p.pc_min), float(smin))
        smax = max(float(smax), smin * 1.000001)

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.Build()
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(smin, smax)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self.renderer.AddActor(actor)

        self._apply_camera(p, poly)

        # Keep all pipeline objects alive to prevent premature garbage-collection.
        self._refs.extend([reader, clamp, iso, sphere, extract, geom, mapper, actor])

        self.render_window.Render()

    def save_png(self, out_path: str) -> None:
        # Defer the capture slightly so the GL frame has been fully composited.
        QTimer.singleShot(50, lambda: self._save_png_impl(out_path))

    def _save_png_impl(self, out_path: str) -> None:
        self.render_window.Render()

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.render_window)
        w2i.ReadFrontBufferOn()  # required when the widget is embedded in a layout
        w2i.SetInputBufferTypeToRGBA()
        w2i.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(out_path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()

        self._refs.extend([w2i, writer])

    def _apply_camera(self, p: VTKRenderParams, poly) -> None:
        cam = self.renderer.GetActiveCamera()

        bounds = poly.GetBounds()
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        cz = 0.5 * (bounds[4] + bounds[5])
        cam.SetFocalPoint(cx, cy, cz)

        diag = math.sqrt(
            (bounds[1] - bounds[0]) ** 2 +
            (bounds[3] - bounds[2]) ** 2 +
            (bounds[5] - bounds[4]) ** 2
        )
        dist = max(diag * 2.5, 1.0)

        nx, ny, nz = p.view_normal
        nlen = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        nx, ny, nz = nx / nlen, ny / nlen, nz / nlen
        cam.SetPosition(cx - nx * dist, cy - ny * dist, cz - nz * dist)

        cam.SetViewUp(*p.view_up)

        if p.parallel_projection:
            cam.ParallelProjectionOn()
        else:
            cam.ParallelProjectionOff()

        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()