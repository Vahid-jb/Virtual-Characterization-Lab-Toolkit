# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
VCL_utils/ovito_viewer.py
=========================

Interactive OVITO viewport that shows its own set of pipelines.

All module windows live in one process. ``Pipeline.add_to_scene()`` puts a
pipeline into the global ``ovito.scene``, which every viewport created without
an explicit scene renders, so pipelines shown that way leak into the other
modules' viewports. :class:`OvitoViewer` gives each viewer a private scene
instead.

OVITO's public API cannot create a scene (``ovito.Scene`` has a single global
instance), so this module uses ``ovito.nonpublic.Scene`` and
``ovito.nonpublic.SceneNode``, the same calls ``ovito.gui.create_qwidget()``
and ``create_ipywidget()`` make internally when given a pipeline (checked
against OVITO 3.14.1 and 3.15.x). ``tests/test_ovito_viewer.py`` fails if
they change.

Importing this module requests an OpenGL-capable OVITO session through
``OVITO_GUI_MODE``, so import it before any ``ovito`` module and only from
GUI code.
"""

from __future__ import annotations

import os

# Only effective before the first `import ovito` in the process.
os.environ["OVITO_GUI_MODE"] = "1"

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QStackedLayout, QWidget

import ovito.nonpublic
from ovito.gui import create_qwidget
from ovito.pipeline import Pipeline
from ovito.vis import Viewport

# Aspect ratio used by zoom_all() while the widget has no real size yet.
_FALLBACK_VIEW_SIZE = (800, 500)


class OvitoViewer(QWidget):
    """Placeholder label that turns into an interactive OVITO viewport once pipelines are added.

    The OVITO scene and the OpenGL widget are created lazily (the widget only
    once the viewer is on screen with something to show) and the widget is
    kept for the lifetime of the viewer.
    """

    error = Signal(str)

    def __init__(self, placeholder_text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._scene = None
        self._viewport: Viewport | None = None
        self._nodes: list = []          # one SceneNode per add_pipeline(); the index is the handle
        self._visible: list[bool] = []
        self._ovito_widget: QWidget | None = None
        self._widget_failed = False
        self._zoom_pending = False

        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QLabel(placeholder_text)
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: #aaa; font-size: 14px;")
        self._stack.addWidget(self._placeholder)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def viewport(self) -> Viewport | None:
        """The OVITO viewport, or None before the first pipeline is added."""
        return self._viewport

    @property
    def viewport_widget(self) -> QWidget | None:
        """The interactive OVITO widget, or None until it has been shown."""
        return self._ovito_widget

    def add_pipeline(self, pipeline: Pipeline, visible: bool = True) -> int:
        """Add *pipeline* to the viewer's scene and return a handle for :meth:`set_pipeline_visible`."""
        self._ensure_scene()
        self._nodes.append(ovito.nonpublic.SceneNode(pipeline=pipeline))
        self._visible.append(False)
        index = len(self._nodes) - 1
        self.set_pipeline_visible(index, visible)
        self._sync_page()
        return index

    def set_pipeline_visible(self, index: int, visible: bool) -> None:
        """Show or hide the pipeline added under *index*."""
        if self._visible[index] == visible:
            return
        if visible:
            self._scene.children.append(self._nodes[index])
        else:
            self._scene.children.remove(self._nodes[index])
        self._visible[index] = visible

    def visible_pipeline_count(self) -> int:
        return len(self._scene.children) if self._scene is not None else 0

    def zoom_all(self) -> None:
        """Fit the camera to the visible pipelines."""
        if not self.visible_pipeline_count():
            return
        if self._can_measure():
            self._viewport.zoom_all((self.width(), self.height()))
            self._zoom_pending = False
        else:
            # Give the first frame a sane camera, then redo it with the real aspect ratio once shown.
            self._viewport.zoom_all(_FALLBACK_VIEW_SIZE)
            self._zoom_pending = True

    def clear(self) -> None:
        """Remove all pipelines and show the placeholder again."""
        for node, shown in zip(self._nodes, self._visible):
            if shown:
                self._scene.children.remove(node)
        self._nodes.clear()
        self._visible.clear()
        self._zoom_pending = False
        self._stack.setCurrentWidget(self._placeholder)

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Deferred so the layout has given the viewer its final size.
        QTimer.singleShot(0, self._on_became_visible)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._zoom_pending and self._can_measure():
            self.zoom_all()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_became_visible(self) -> None:
        if not self.isVisible():
            return
        self._sync_page()
        if self._zoom_pending:
            self.zoom_all()

    def _can_measure(self) -> bool:
        return (self._ovito_widget is not None and self.isVisible()
                and self.width() > 1 and self.height() > 1)

    def _ensure_scene(self) -> None:
        if self._scene is None:
            self._scene = ovito.nonpublic.Scene()
            self._viewport = Viewport(type=Viewport.Type.Perspective, camera_dir=(-1, -1, -1),
                                      scene=self._scene)

    def _sync_page(self) -> None:
        """Show the OVITO widget when there is content and it can be created, else the placeholder."""
        if self._nodes and self.isVisible():
            self._ensure_widget()
        if self._nodes and self._ovito_widget is not None:
            self._stack.setCurrentWidget(self._ovito_widget)
        else:
            self._stack.setCurrentWidget(self._placeholder)

    def _ensure_widget(self) -> None:
        if self._ovito_widget is not None or self._widget_failed:
            return
        try:
            widget = create_qwidget(self._viewport, parent=self)
        except Exception as e:
            self._widget_failed = True
            self._placeholder.setText(f"3D view unavailable:\n{e}")
            self.error.emit(f"Could not create the OVITO viewport: {e}")
            return
        self._ovito_widget = widget
        self._stack.addWidget(widget)
