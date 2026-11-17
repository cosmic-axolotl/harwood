"""The middle of the window: the image and every graph, in tabs, and under the tabs
a strip of options that hold for all of them (the cursor).

QTabWidget puts its pages straight under its bar, and hides the bar -- with all it
holds -- while there is only the image. So the tabs are built here from their
parts, a QTabBar, the strip and a QStackedWidget, with the QTabWidget methods the
window uses: the strip stays when the bar hides.
"""

from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtWidgets


class Views(QtWidgets.QWidget):
    currentChanged = QtCore.Signal(int)
    tabCloseRequested = QtCore.Signal(int)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self._bar = QtWidgets.QTabBar()
        self._bar.setExpanding(False)
        self._stack = QtWidgets.QStackedWidget()
        self.strip = QtWidgets.QWidget()
        self.strip.setObjectName("viewstrip")
        self._options = QtWidgets.QHBoxLayout(self.strip)
        self._options.setContentsMargins(8, 2, 8, 2)
        self._options.setSpacing(12)
        self._options.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._bar)
        layout.addWidget(self.strip)
        layout.addWidget(self._stack, stretch=1)
        self._bar.currentChanged.connect(self._changed)
        self._bar.tabCloseRequested.connect(self.tabCloseRequested)

    def add_option(self, widget) -> None:
        """A control in the strip, from the left."""
        self._options.insertWidget(self._options.count() - 1, widget)

    def _changed(self, index: int) -> None:
        if 0 <= index < self._stack.count():
            self._stack.setCurrentIndex(index)
        self.currentChanged.emit(index)

    # ---- what the window used of QTabWidget
    def setDocumentMode(self, on: bool) -> None:
        self._bar.setDocumentMode(on)

    def setTabsClosable(self, on: bool) -> None:
        self._bar.setTabsClosable(on)

    def setTabBarAutoHide(self, on: bool) -> None:
        self._bar.setAutoHide(on)

    def tabBarAutoHide(self) -> bool:
        return self._bar.autoHide()

    def tabBar(self):
        return self._bar

    def addTab(self, widget, label: str) -> int:
        self._stack.addWidget(widget)
        return self._bar.addTab(label)

    def removeTab(self, index: int) -> None:
        widget = self._stack.widget(index)
        if widget is None:
            return
        self._stack.removeWidget(widget)
        self._bar.removeTab(index)

    def count(self) -> int:
        return self._stack.count()

    def widget(self, index: int):
        return self._stack.widget(index)

    def indexOf(self, widget) -> int:
        return self._stack.indexOf(widget)

    def currentIndex(self) -> int:
        return self._bar.currentIndex()

    def currentWidget(self):
        return self._stack.widget(self._bar.currentIndex())

    def setCurrentIndex(self, index: int) -> None:
        self._bar.setCurrentIndex(index)

    def setCurrentWidget(self, widget) -> None:
        index = self._stack.indexOf(widget)
        if index >= 0:
            self._bar.setCurrentIndex(index)

    def tabText(self, index: int) -> str:
        return self._bar.tabText(index)

    def setTabText(self, index: int, text: str) -> None:
        self._bar.setTabText(index, text)

    def setTabToolTip(self, index: int, text: str) -> None:
        self._bar.setTabToolTip(index, text)
