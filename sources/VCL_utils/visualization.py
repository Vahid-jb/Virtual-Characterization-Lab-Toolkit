# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
Shared visualization utilities for VCL GUI applications.

Provides a unified Matplotlib widget and axis styling helpers.
"""
from __future__ import annotations

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def style_axis_light(ax: Axes) -> None:
    """Apply a clean white/light style to *ax* — used by all plot functions."""
    ax.set_facecolor("white")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")


class MatplotlibWidget(QWidget):
    """
    A unified widget for displaying Matplotlib plots in PySide6 applications.
    Includes a page bar, a toolbar and a canvas.

    A run that produces several figures registers them with :meth:`plot_pages`;
    the page bar then steps between them. The Matplotlib toolbar's own back and
    forward buttons are *view history* for the current axes, not plot
    navigation, which is why a separate bar is needed. The bar hides itself
    whenever there is at most one page, so single-plot callers are unaffected.
    """

    def __init__(self, parent: QWidget | None = None, width: int = 5, height: int = 4, dpi: int = 100) -> None:
        super().__init__(parent)

        # Use _layout to avoid shadowing QWidget.layout()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self._pages: list[tuple[str | None, object]] = []
        self._page_index = 0

        self.page_bar = QWidget(self)
        page_layout = QHBoxLayout(self.page_bar)
        page_layout.setContentsMargins(4, 2, 4, 2)
        self.prev_button = QPushButton("\u25c0", self.page_bar)
        self.next_button = QPushButton("\u25b6", self.page_bar)
        for button in (self.prev_button, self.next_button):
            button.setFixedWidth(32)
            button.setAutoDefault(False)
        self.prev_button.setToolTip("Previous plot")
        self.next_button.setToolTip("Next plot")
        self.page_label = QLabel("", self.page_bar)
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        page_layout.addWidget(self.prev_button)
        page_layout.addWidget(self.page_label, 1)
        page_layout.addWidget(self.next_button)
        self.prev_button.clicked.connect(lambda: self._step_page(-1))
        self.next_button.clicked.connect(lambda: self._step_page(1))
        self.page_bar.setVisible(False)

        self._layout.addWidget(self.page_bar)
        self._layout.addWidget(self.toolbar)
        self._layout.addWidget(self.canvas)

        # Default styling to match dark theme
        self.figure.patch.set_facecolor("#2b2b2b")
        self.figure.patch.set_alpha(1.0)

    def get_figure(self) -> Figure:
        """The underlying Matplotlib Figure."""
        return self.figure

    def get_canvas(self) -> FigureCanvasQTAgg:
        """The FigureCanvasQTAgg embedded in this widget."""
        return self.canvas

    def plot(self, func, *args, **kwargs) -> None:
        """Clear the figure and call ``func(figure, *args, **kwargs)`` to render new content."""
        if args or kwargs:
            self.plot_pages([(None, lambda fig: func(fig, *args, **kwargs))])
        else:
            self.plot_pages([(None, func)])

    def plot_pages(self, pages) -> None:
        """Render *pages*, a sequence of ``(title, func)``, as navigable plots.

        Each ``func(figure)`` draws one page into the shared figure. Pages are
        rendered lazily, on the way in and on every page change, so a run with
        several large data files only pays for the page being looked at.
        """
        self._pages = [(title, func) for title, func in pages]
        self._page_index = 0
        if not self._pages:
            self.clear()
            return
        self.page_bar.setVisible(len(self._pages) > 1)
        self._show_page(0)

    def page_count(self) -> int:
        """The number of registered plot pages."""
        return len(self._pages)

    def current_page(self) -> int:
        """The zero-based index of the page on screen."""
        return self._page_index

    def _step_page(self, delta: int) -> None:
        if len(self._pages) > 1:
            self._show_page((self._page_index + delta) % len(self._pages))

    def _show_page(self, index: int) -> None:
        self._page_index = index
        title, func = self._pages[index]
        self.figure.clear()
        func(self.figure)
        self.canvas.draw()
        self.toolbar.update()  # drop the previous page's zoom/pan history
        label = f"{index + 1} / {len(self._pages)}"
        self.page_label.setText(f"{label} \u2014 {title}" if title else label)

    def clear(self) -> None:
        """Clear the figure, drop any pages and redraw an empty canvas."""
        self._pages = []
        self._page_index = 0
        self.page_bar.setVisible(False)
        self.page_label.setText("")
        self.figure.clear()
        self.canvas.draw()
