# SPDX-FileCopyrightText: 2026 The VCL Toolkit contributors
#
# SPDX-License-Identifier: MIT

"""
Shared theme utilities for all VCL GUI applications.

Applies a unified dark appearance via Qt Style Sheets (QSS) and QPalette.
Also exports the no-wheel spin-box/combo-box subclasses, the action-icon
registry and the reusable button factory functions used by every sub-module.
"""

import sys
from pathlib import Path
from string import Template

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

try:
    import qtawesome as qta
except Exception:  # pragma: no cover - qtawesome is optional at import time
    qta = None


class NoWheelSpinBox(QSpinBox):
    """QSpinBox that ignores scroll-wheel events to prevent accidental value changes."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that ignores scroll-wheel events to prevent accidental value changes."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelComboBox(QComboBox):
    """QComboBox that ignores scroll-wheel events to prevent accidental selection changes."""

    def wheelEvent(self, event) -> None:
        event.ignore()


# ---------------------------------------------------------------------------
# Action icons
# ---------------------------------------------------------------------------

_ACTION_ICONS = {
    "add": "fa5s.plus",
    "analyze": "fa5s.chart-bar",
    "apply": "fa5s.check",
    "browse": "fa5s.folder-open",
    "cancel": "fa5s.times",
    "convert": "fa5s.exchange-alt",
    "delete": "fa5s.trash-alt",
    "download": "fa5s.download",
    "export": "fa5s.file-export",
    "file": "fa5s.file-alt",
    "generate": "fa5s.play",
    "help": "fa5s.question-circle",
    "import": "fa5s.file-import",
    "remove": "fa5s.minus-circle",
    "reset": "fa5s.sync-alt",
    "save": "fa5s.save",
    "stop": "fa5s.stop",
    "test": "fa5s.vial",
    "zoom": "fa5s.expand",
}

_ICON_COLOR = "#dddddd"
_ICON_COLOR_DISABLED = "#888888"


def _icon(icon_name: str, *, color: str | None = None,
          disabled_color: str | None = None) -> QIcon:
    """Return a themed QtAwesome icon, or an empty icon when unavailable."""
    if qta is None:
        return QIcon()
    return qta.icon(icon_name,
                    color=color or _ICON_COLOR,
                    color_disabled=disabled_color or _ICON_COLOR_DISABLED)


def icon_for_action(action: str) -> QIcon:
    """Return the standard icon for an action key."""
    return _icon(_ACTION_ICONS[action])


def decorate_action_button(button: QPushButton, action: str, *,
                           icon_size: int = 14) -> QPushButton:
    """Attach a standard action icon and cursor to *button*."""
    button.setIcon(icon_for_action(action))
    button.setIconSize(QSize(icon_size, icon_size))
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def make_action_button(text: str, action: str | None = None, *,
                       icon_size: int = 14) -> QPushButton:
    """Create a standard secondary action button with icon and text.

    With *action* omitted the key is inferred from *text*, which is what the
    repeated Browse rows want.
    """
    return decorate_action_button(QPushButton(text),
                                  action or action_for_text(text),
                                  icon_size=icon_size)


def action_for_text(text: str, *, fallback: str = "generate") -> str:
    """Infer a standard action key from a button label."""
    normalized = text.lower()
    for keyword, action in (
        ("analy", "analyze"),
        ("apply", "apply"),
        ("browse", "browse"),
        ("cancel", "cancel"),
        ("convert", "convert"),
        ("delete", "delete"),
        ("download", "download"),
        ("export", "export"),
        ("import", "import"),
        ("remove", "remove"),
        ("reset", "reset"),
        ("rescale", "generate"),
        ("save", "save"),
        ("stop", "stop"),
        ("test", "test"),
        ("zoom", "zoom"),
        ("add", "add"),
    ):
        if keyword in normalized:
            return action
    return fallback


# ---------------------------------------------------------------------------
# Theme icon assets
# ---------------------------------------------------------------------------

def _theme_icon_dir() -> Path:
    """Return the theme icon directory in source and PyInstaller layouts.

    ``VCL_utils`` is a top-level package, so in the source tree the assets sit
    beside this module rather than a level further up.
    """
    if getattr(sys, "frozen", False):
        bundle_internal = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        bundle_root = Path(sys.executable).parent
        for base in (bundle_internal, bundle_root):
            candidate = base / "VCL_utils" / "assets" / "icons"
            if candidate.is_dir():
                return candidate
    return Path(__file__).resolve().parent / "assets" / "icons"


def _icon_declaration(icon_dir: Path, name: str) -> str:
    """Return a QSS ``image:`` declaration for *name*, or "" when it is absent.

    Qt draws nothing for a ``url()`` it cannot open and re-warns on every
    repaint, so a missing asset would leave a checked box as a flat fill plus an
    unbounded stream of ``qt.svg: Cannot open file``. Dropping the declaration
    gives the same appearance with a quiet log. ``as_posix`` matters on Windows:
    QSS reads a backslash as an escape character.
    """
    path = icon_dir / name
    if not path.is_file():
        return ""
    return f'image: url("{path.as_posix()}");'


def _optional_block(template: Template, icon_dir: Path, assets: dict[str, str]) -> str:
    """Return *template* filled with *assets*' paths, or "" if any is missing.

    Unlike a checkbox tick, a stepper button that is styled but whose arrow
    image fails to load renders blank - worse than letting Fusion draw the
    control. So the rules and the images they depend on drop out together.
    """
    paths = {}
    for key, name in assets.items():
        path = icon_dir / name
        if not path.is_file():
            return ""
        paths[key] = path.as_posix()
    return template.substitute(paths)


# Only reached when combo_arrow_down.svg is present. Styling ``::drop-down``
# without this rule is safe - Fusion still draws its own arrow - so this one
# rule can drop out on its own.
_COMBO_ARROW_TEMPLATE = Template("""        QComboBox::down-arrow {
            image: url("${combo_arrow_down}");
            width: 10px;
            height: 10px;
        }
""")

# Spin boxes were left to Fusion entirely before, so the field styling that
# matches QLineEdit comes in with the stepper buttons and drops out with them.
_SPIN_BUTTON_TEMPLATE = Template("""        QSpinBox, QDoubleSpinBox {
            background-color: #1e1e1e;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 4px;
            color: #eee;
        }
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #2a82da;
        }
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
            subcontrol-origin: padding;
            width: 18px;
            border-left: 1px solid #555;
            background-color: #3d3d3d;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-position: top right;
            height: 12px;
            border-top-right-radius: 3px;
            border-bottom: 1px solid #555;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-position: bottom right;
            height: 12px;
            border-bottom-right-radius: 3px;
        }
        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
            background-color: #4d4d4d;
        }
        QSpinBox::up-button:pressed, QSpinBox::down-button:pressed,
        QDoubleSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed {
            background-color: #2d2d2d;
        }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            image: url("${spin_arrow_up}");
            width: 8px;
            height: 8px;
        }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            image: url("${spin_arrow_down}");
            width: 8px;
            height: 8px;
        }
""")

# A string.Template: ${name} placeholders are filled by _build_stylesheet,
# so any literal "$" added to the sheet later must be written "$$".
_QSS_TEMPLATE = Template("""
        QToolTip { 
            color: #ffffff; 
            background-color: #2a82da; 
            border: 1px solid white; 
        }
        /* No font-family: Segoe UI exists only on Windows, so the other two
           targets fell through to a generic sans-serif anyway. Leaving it unset
           takes the platform's own UI font on every target. */
        QWidget {
            font-size: 10pt;
        }
        QGroupBox {
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
            color: #ccc;
        }
        /* A selector-less stylesheet on an ancestor pulls every descendant under
           QStyleSheetStyle, which stops Fusion drawing these indicators natively.
           Without explicit rules they render as a flat palette-derived box: dark
           on dark, and identical checked or not. */
        QCheckBox, QRadioButton {
            spacing: 8px;
        }
        /* Item-view checks are a separate sub-control that QCheckBox::indicator
           never reaches, so the phase table in Structure Analyzer is listed too. */
        QCheckBox::indicator,
        QGroupBox::indicator,
        QTableWidget::indicator,
        QAbstractItemView::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #8a8a8a;
            border-radius: 4px;
            background-color: #1e1e1e;
        }
        QRadioButton::indicator {
            width: 16px;
            height: 16px;
            border: 1px solid #8a8a8a;
            border-radius: 9px;
            background-color: #1e1e1e;
        }
        QCheckBox::indicator:hover,
        QRadioButton::indicator:hover,
        QGroupBox::indicator:hover,
        QTableWidget::indicator:hover,
        QAbstractItemView::indicator:hover {
            border-color: #2a82da;
        }
        /* The tick is an image, so the box-shaped indicators are styled apart
           from the radio one: a tick glyph inside a circle reads as a defect.
           ${checkbox_tick} is empty when the asset is missing, leaving the
           colour-only fill these rules had before. */
        QCheckBox::indicator:checked,
        QGroupBox::indicator:checked,
        QTableWidget::indicator:checked,
        QAbstractItemView::indicator:checked {
            background-color: #2a82da;
            border-color: #2a82da;
            ${checkbox_tick}
        }
        QCheckBox::indicator:checked:hover,
        QGroupBox::indicator:checked:hover,
        QTableWidget::indicator:checked:hover,
        QAbstractItemView::indicator:checked:hover {
            background-color: #3a92ea;
            border-color: #3a92ea;
            ${checkbox_tick}
        }
        /* A ring of the ground colour inside the accent fill, so the radio reads
           as a dot rather than the solid disc a plain fill gives. The width and
           height shrink by twice the border, because Qt lays a sub-control's
           border outside the size given here - left at 16px the checked radio
           would be 24px across, half again the size of a checkbox. 10px plus
           two 4px borders matches the checkbox's 16px plus two 1px borders. */
        QRadioButton::indicator:checked {
            width: 10px;
            height: 10px;
            background-color: #2a82da;
            border: 4px solid #1e1e1e;
        }
        QRadioButton::indicator:checked:hover {
            width: 10px;
            height: 10px;
            background-color: #3a92ea;
            border: 4px solid #1e1e1e;
        }
        /* Defensive: nothing in the tree calls setTristate yet. */
        QCheckBox::indicator:indeterminate {
            background-color: #6f6f6f;
            border-color: #8a8a8a;
        }
        QCheckBox::indicator:disabled,
        QRadioButton::indicator:disabled,
        QGroupBox::indicator:disabled,
        QTableWidget::indicator:disabled,
        QAbstractItemView::indicator:disabled {
            border-color: #555;
            background-color: #2b2b2b;
        }
        QCheckBox::indicator:checked:disabled,
        QGroupBox::indicator:checked:disabled,
        QTableWidget::indicator:checked:disabled,
        QAbstractItemView::indicator:checked:disabled {
            background-color: #3f5f80;
            border-color: #555;
            ${checkbox_tick}
        }
        QRadioButton::indicator:checked:disabled {
            width: 10px;
            height: 10px;
            background-color: #3f5f80;
            border: 4px solid #2b2b2b;
        }
        QPushButton {
            background-color: #3d3d3d;
            border-style: outset;
            border-width: 1px;
            border-radius: 4px;
            border-color: #555;
            padding: 6px;
            min-width: 60px;
        }
        QPushButton:hover {
            background-color: #4d4d4d;
            border-color: #2a82da;
        }
        QPushButton:pressed {
            background-color: #2d2d2d;
            border-style: inset;
        }
        QLineEdit {
            background-color: #1e1e1e;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 4px;
            color: #eee;
        }
        QLineEdit:focus {
            border: 1px solid #2a82da;
        }
        QComboBox {
            background-color: #3d3d3d;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 4px;
            min-width: 6em;
        }
        QComboBox:on {
            padding-top: 3px;
            padding-left: 4px;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 15px;
            border-left-width: 1px;
            border-left-color: darkgray;
            border-left-style: solid;
            border-top-right-radius: 3px; 
            border-bottom-right-radius: 3px;
        }
${combo_arrow}${spin_buttons}        QListWidget {
            background-color: #1e1e1e;
            border: 1px solid #555;
        }
        QTabWidget::pane {
            border-top: 1px solid #555;
        }
        QTabBar::tab {
            background: #3d3d3d;
            border: 1px solid #555;
            border-bottom-color: #555; /* same as pane color */
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 8ex;
            padding: 6px;
            color: #ccc;
        }
        QTabBar::tab:selected, QTabBar::tab:hover {
            background: #4d4d4d;
            color: #fff;
        }
        QTabBar::tab:selected {
            border-color: #555;
            border-bottom-color: #4d4d4d; /* same as pane color */
        }
        QHeaderView::section {
            background-color: #3d3d3d;
            padding: 4px;
            border: 1px solid #555;
            color: #eee;
        }
        QScrollBar:vertical {
            border: 1px solid #2d2d2d;
            background: #2d2d2d;
            width: 15px;
            margin: 22px 0 22px 0;
        }
        QScrollBar::handle:vertical {
            background: #555;
            min-height: 20px;
        }
        QScrollBar::add-line:vertical {
            border: 1px solid #2d2d2d;
            background: #2d2d2d;
            height: 20px;
            subcontrol-position: bottom;
            subcontrol-origin: margin;
        }
        QScrollBar::sub-line:vertical {
            border: 1px solid #2d2d2d;
            background: #2d2d2d;
            height: 20px;
            subcontrol-position: top;
            subcontrol-origin: margin;
        }
""")


def _build_stylesheet(icon_dir: Path) -> str:
    """Return the application stylesheet with the icon paths resolved.

    Kept separate from :func:`apply_dark_theme` so the asset handling can be
    tested without a QApplication and without replacing the stylesheet that
    every other widget test in the session shares.
    """
    return _QSS_TEMPLATE.substitute(
        checkbox_tick=_icon_declaration(icon_dir, "checkbox_tick.svg"),
        combo_arrow=_optional_block(
            _COMBO_ARROW_TEMPLATE, icon_dir,
            {"combo_arrow_down": "combo_arrow_down.svg"},
        ),
        spin_buttons=_optional_block(
            _SPIN_BUTTON_TEMPLATE, icon_dir,
            {"spin_arrow_up": "spin_arrow_up.svg",
             "spin_arrow_down": "spin_arrow_down.svg"},
        ),
    )


def apply_dark_theme(app: QApplication) -> None:
    """Apply a Fusion-based dark theme to *app*.

    Sets the application style to "Fusion" and installs a custom dark
    QPalette and QSS stylesheet that covers all common widget types.
    """
    app.setStyle("Fusion")
    
    dark_palette = QPalette()
    
    dark_color = QColor(45, 45, 45)
    disabled_color = QColor(127, 127, 127)
    text_color = QColor(220, 220, 220)
    highlight_color = QColor(42, 130, 218)
    highlighted_text_color = QColor(255, 255, 255)
    
    dark_palette.setColor(QPalette.ColorRole.Window, dark_color)
    dark_palette.setColor(QPalette.ColorRole.WindowText, text_color)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, dark_color)
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, text_color)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, text_color)
    dark_palette.setColor(QPalette.ColorRole.Text, text_color)
    dark_palette.setColor(QPalette.ColorRole.Button, dark_color)
    dark_palette.setColor(QPalette.ColorRole.ButtonText, text_color)
    dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.ColorRole.Link, highlight_color)
    dark_palette.setColor(QPalette.ColorRole.Highlight, highlight_color)
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, highlighted_text_color)

    # Without these the disabled group falls back to the default light palette,
    # which is unreadable on the dark ground.
    disabled = QPalette.ColorGroup.Disabled
    dark_palette.setColor(disabled, QPalette.ColorRole.WindowText, disabled_color)
    dark_palette.setColor(disabled, QPalette.ColorRole.Text, disabled_color)
    dark_palette.setColor(disabled, QPalette.ColorRole.ButtonText, disabled_color)
    dark_palette.setColor(disabled, QPalette.ColorRole.HighlightedText, disabled_color)

    app.setPalette(dark_palette)
    app.setStyleSheet(_build_stylesheet(_theme_icon_dir()))


# ---------------------------------------------------------------------------
# Reusable button factories
# ---------------------------------------------------------------------------

def make_primary_button(text: str = "Calculate", max_width: int = 220) -> QPushButton:
    """Return the standard primary action button used across all sub-interfaces.

    Use *text* to override the default label (e.g. 'Convert').
    """
    btn = QPushButton(text)
    btn.setMinimumHeight(50)
    btn.setMaximumWidth(max_width)
    decorate_action_button(btn, action_for_text(text), icon_size=15)
    btn.setStyleSheet(
        "QPushButton { font-size: 15px; font-weight: bold;"
        " background-color: #2a82da; color: white; border-radius: 5px; }"
        "QPushButton:hover { background-color: #3a92ea; }"
        "QPushButton:pressed { background-color: #1a72ca; }"
        "QPushButton:disabled { background-color: #555; color: #888; }"
    )
    return btn


def make_stop_button(max_width: int = 120) -> QPushButton:
    """Return the standard Stop button, initially disabled.

    The caller enables it when a computation is in progress and disables it
    again on completion or cancellation.
    """
    btn = QPushButton("Stop")
    btn.setMinimumHeight(50)
    btn.setMaximumWidth(max_width)
    btn.setEnabled(False)
    decorate_action_button(btn, "stop", icon_size=15)
    btn.setStyleSheet(
        "QPushButton { font-size: 15px; font-weight: bold;"
        " background-color: #c0392b; color: white; border-radius: 5px; }"
        "QPushButton:hover { background-color: #d44; }"
        "QPushButton:pressed { background-color: #a02020; }"
        "QPushButton:disabled { background-color: #444; color: #777; }"
    )
    return btn


def add_run_row(layout, calc_btn: QPushButton, stop_btn: QPushButton) -> None:
    """Append a horizontal row containing *calc_btn* and *stop_btn* to *layout*."""
    row = QHBoxLayout()
    row.addWidget(calc_btn)
    row.addWidget(stop_btn)
    row.addStretch()
    layout.addLayout(row)


class CollapsibleGroup(QGroupBox):
    """A group box that folds its contents away, collapsed by default.

    For parameters that need to be reachable without crowding the panel: the
    title checkbox is the toggle, and :attr:`form` is the layout to add rows to.
    """

    def __init__(self, title: str, parent: QWidget | None = None,
                 expanded: bool = False) -> None:
        super().__init__(title, parent)
        self.setCheckable(True)
        self._outer = QVBoxLayout(self)
        self.body = QWidget(self)
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(0, 0, 0, 0)
        self._outer.addWidget(self.body)
        self.toggled.connect(self._set_expanded)
        self.setChecked(expanded)
        self._set_expanded(expanded)

    def _set_expanded(self, expanded: bool) -> None:
        self.body.setVisible(expanded)
        # collapsed, the box is just its title bar: no empty band below it
        self._outer.setContentsMargins(6, 6, 6, 6) if expanded else \
            self._outer.setContentsMargins(6, 0, 6, 0)
