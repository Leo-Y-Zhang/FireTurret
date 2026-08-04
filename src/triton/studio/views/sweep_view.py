# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Sweep results visualization: a scatter of metric-vs-parameter for a 1-axis
sweep, or a heatmap over the two parameters for a 2-axis sweep. Values are
averaged over the seed set. Interactive (pyqtgraph), no matplotlib.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QLabel, QStackedLayout, QWidget

ACCENT = "#4cc3e8"

# metric name -> extractor
_METRICS = {
    "extinguished": lambda r: 1.0 if r.extinguished else 0.0,
    "water_l": lambda r: r.water_l,
    "acquire_s": lambda r: r.acquire_s if r.acquire_s is not None else float("nan"),
    "mean_miss_m": lambda r: r.mean_miss_m,
    "peak_miss_m": lambda r: r.peak_miss_m,
}

METRIC_NAMES = list(_METRICS)


def _value(r, fn) -> float:
    """A failed cell has no valid metric — contribute NaN so it neither skews the
    scatter/heatmap average nor reads as a real (e.g. zero-water) result."""
    return float("nan") if getattr(r, "failed", False) else fn(r)


class SweepResultsView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.data: dict | None = None

        self._stack = QStackedLayout(self)
        self._empty = QLabel("Run a sweep (Run ▸ Batch sweep…)")
        self._empty.setAlignment(self._empty.alignment())
        self._stack.addWidget(self._empty)

        self._plot = pg.PlotWidget()
        self._scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(ACCENT))
        self._plot.addItem(self._scatter)
        self._image = pg.ImageItem()
        self._image.setColorMap(pg.colormap.get("viridis"))
        self._plot.addItem(self._image)
        self._stack.addWidget(self._plot)
        self._stack.setCurrentWidget(self._empty)

    def set_results(self, spec, results, metric: str = "water_l") -> None:
        fn = _METRICS[metric]
        if len(spec.axes) == 1:
            self._render_1d(spec, results, metric, fn)
        else:
            self._render_2d(spec, results, metric, fn)
        self._stack.setCurrentWidget(self._plot)

    def _render_1d(self, spec, results, metric, fn) -> None:
        key = f"{spec.axes[0].group}.{spec.axes[0].name}"
        groups: dict = {}
        for r in results:
            groups.setdefault(r.coords[key], []).append(_value(r, fn))
        xs = sorted(groups)
        ys = [float(np.nanmean(groups[x])) for x in xs]
        self._image.clear()
        self._scatter.setData(xs, ys)
        self._plot.setLabel("bottom", key)
        self._plot.setLabel("left", metric)
        self.data = {"x": xs, "y": ys}

    def _render_2d(self, spec, results, metric, fn) -> None:
        k0 = f"{spec.axes[0].group}.{spec.axes[0].name}"
        k1 = f"{spec.axes[1].group}.{spec.axes[1].name}"
        v0s = sorted({r.coords[k0] for r in results})
        v1s = sorted({r.coords[k1] for r in results})
        i0 = {v: i for i, v in enumerate(v0s)}
        i1 = {v: i for i, v in enumerate(v1s)}
        sums = np.zeros((len(v0s), len(v1s)))
        counts = np.zeros_like(sums)
        for r in results:
            v = _value(r, fn)
            if v != v:  # NaN (failed cell): leave the sum/count untouched -> blank cell
                continue
            i, j = i0[r.coords[k0]], i1[r.coords[k1]]
            sums[i, j] += v
            counts[i, j] += 1
        grid = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        self._scatter.setData([], [])
        self._image.setImage(grid)
        # position the image over the REAL swept-parameter ranges so the axes read
        # the parameter values, not 0..N array indices (grid[i,j]: i->x=k0, j->y=k1)
        x0, x1 = v0s[0], v0s[-1]
        y0, y1 = v1s[0], v1s[-1]
        self._image.setRect(QRectF(x0, y0, (x1 - x0) or 1.0, (y1 - y0) or 1.0))
        self._plot.setLabel("bottom", k0)
        self._plot.setLabel("left", k1)
        self.data = {"grid": grid, "v0": v0s, "v1": v1s, "metric": metric}
