# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The Studio Session document: the current TritonConfig + SimScenario, with
validated field edits, a dirty flag, and undo/redo via a QUndoStack.

Every edit reconstructs the frozen dataclass via ``dataclasses.replace`` so the
sub-config's ``__post_init__`` validation runs; an invalid value raises and is
NOT committed. Views bind to ``set_field`` and listen to ``changed``.
"""
from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from ..config import TritonConfig
from ..rig.sim_rig import SimScenario

_CONFIG_GROUPS = ("camera", "turret", "jet", "detector", "servo", "mission")


class _SetFieldCommand(QUndoCommand):
    def __init__(self, session: Session, group: str, name: str, old, new) -> None:
        super().__init__(f"set {group}.{name}")
        self._s = session
        self._group = group
        self._name = name
        self._old = old
        self._new = new

    def redo(self) -> None:
        self._s._apply(self._group, self._name, self._new)

    def undo(self) -> None:
        self._s._apply(self._group, self._name, self._old)


class Session(QObject):
    """Holds the editable model. `changed` fires after every committed edit."""

    changed = Signal()

    def __init__(self, config: TritonConfig | None = None,
                 scenario: SimScenario | None = None, parent=None) -> None:
        super().__init__(parent)
        self._config = config or TritonConfig()
        self._scenario = scenario or SimScenario()
        self._dirty = False
        self.undo_stack = QUndoStack(self)

    @property
    def config(self) -> TritonConfig:
        return self._config

    @property
    def scenario(self) -> SimScenario:
        return self._scenario

    @property
    def dirty(self) -> bool:
        return self._dirty

    def field(self, group: str, name: str):
        obj = self._scenario if group == "scenario" else getattr(self._config, group)
        return getattr(obj, name)

    def set_field(self, group: str, name: str, value) -> None:
        """Validate then commit an edit (pushing an undo command). No-op if unchanged.
        Raises ValueError on an invalid value without mutating anything."""
        if group != "scenario" and group not in _CONFIG_GROUPS:
            raise ValueError(f"unknown group: {group!r}")
        old = self.field(group, name)
        if old == value:
            return
        self._validate(group, name, value)  # raises without mutating on invalid
        self.undo_stack.push(_SetFieldCommand(self, group, name, old, value))

    def undo(self) -> None:
        self.undo_stack.undo()

    def redo(self) -> None:
        self.undo_stack.redo()

    def mark_saved(self) -> None:
        self._dirty = False
        self.changed.emit()

    # -- internals -----------------------------------------------------------

    def _validate(self, group: str, name: str, value) -> None:
        if group == "scenario":
            replace(self._scenario, **{name: value})
        elif group in _CONFIG_GROUPS:
            replace(getattr(self._config, group), **{name: value})  # runs __post_init__
        else:
            raise ValueError(f"unknown group: {group!r}")

    def _apply(self, group: str, name: str, value) -> None:
        if group == "scenario":
            self._scenario = replace(self._scenario, **{name: value})
        else:
            new_sub = replace(getattr(self._config, group), **{name: value})
            self._config = replace(self._config, **{group: new_sub})
        self._dirty = True
        self.changed.emit()
