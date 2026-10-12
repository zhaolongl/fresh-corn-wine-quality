"""Millimetre-based journal layout with geometry checks; no file output at import."""
import numpy as np
import matplotlib
from matplotlib.transforms import Bbox
from .style import plt, MM
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
FS = dict(tick=7.0, label=8.0, panel=9.0, legend=7.0, annot=6.5, title=7.5, small=6.5)
GRID_STYLE = {'xtick.labelsize': FS['tick'], 'ytick.labelsize': FS['tick'], 'axes.labelsize': FS['label'], 'legend.fontsize': FS['legend'], 'axes.titlesize': FS['title'], 'axes.titleweight': 'bold', 'legend.title_fontsize': FS['legend'], 'axes.labelpad': 2.5, 'xtick.major.pad': 2.0, 'ytick.major.pad': 2.0}
W2, W15, W1 = (160.0, 120.0, 78.0)

class Cell:

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = (x, y, w, h)

    def __repr__(self):
        return f'Cell({self.x:.1f}, {self.y:.1f}, {self.w:.1f}, {self.h:.1f})'

def cells(x0, y0, widths, heights, xgap=0.0, ygap=0.0):
    """grid of cells (rows x cols) with the given column widths / row heights (mm); returns a 2-D list [row][col]"""
    out, y = ([], y0)
    for h in heights:
        row, x = ([], x0)
        for w in widths:
            row.append(Cell(x, y, w, h))
            x += w + xgap
        out.append(row)
        y += h + ygap
    return out

class Canvas:

    def __init__(self, width=W2, height=100.0):
        self.W, self.H = (float(width), float(height))
        self.fig = Figure(figsize=(self.W * MM, self.H * MM))
        FigureCanvasAgg(self.fig)
        self.axes = []

    def ax(self, x, y, w, h, polar=False, overlap_ok=False, **kw):
        """axes whose plotting area has its top-left corner at (x, y) mm (measured from the canvas' top-left)"""
        a = self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], polar=polar, **kw)
        a._fg_overlap_ok = overlap_ok
        self.axes.append(a)
        return a

    def panel(self, cell, letter=None, L=13.0, T=5.0, R=2.0, B=10.0, polar=False, overlap_ok=False, **kw):
        """axes inside a cell with reserved margins (mm): L for y tick labels + y title, B for x tick labels + x title,
        T for the panel letter, R on the right; the letter sits at the cell's top-left corner"""
        a = self.ax(cell.x + L, cell.y + T, cell.w - L - R, cell.h - T - B, polar=polar, overlap_ok=overlap_ok, **kw)
        if letter:
            self.letter(cell.x, cell.y, letter)
        a._fg_cell = cell
        return a

    def letter(self, x, y, s):
        self.fig.text(x / self.W, 1 - y / self.H, f'({s})', fontsize=FS['panel'], fontweight='bold', va='top', ha='left')

    def text(self, x, y, s, **kw):
        kw.setdefault('fontsize', FS['annot'])
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def cbar(self, ax, mappable, label=None, w=2.0, pad=1.5, ticks=None, ticklabels=None, frac=1.0, extend=None):
        """vertical colour bar right of ax, same height (or a centred fraction of it)"""
        bb = ax.get_position()
        x = bb.x1 * self.W + pad
        hh = bb.height * self.H * frac
        y = (1 - bb.y1) * self.H + bb.height * self.H * (1 - frac) / 2
        cax = self.ax(x, y, w, hh, overlap_ok=True)
        cb = self.fig.colorbar(mappable, cax=cax, extend=extend) if extend else self.fig.colorbar(mappable, cax=cax)
        cb.ax.tick_params(labelsize=FS['small'], length=2, width=0.6, pad=1.5)
        cb.outline.set_linewidth(0.6)
        if ticks is not None:
            cb.set_ticks(ticks)
            if ticklabels is not None:
                cb.set_ticklabels(ticklabels)
            cb.ax.minorticks_off()
        if label:
            cb.set_label(label, fontsize=FS['small'], fontweight='normal', labelpad=2)
        return cb

    def hcbar(self, x, y, w, mappable, label=None, h=2.0, ticks=None, ticklabels=None):
        cax = self.ax(x, y, w, h, overlap_ok=True)
        cb = self.fig.colorbar(mappable, cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=FS['small'], length=2, width=0.6, pad=1.5)
        cb.outline.set_linewidth(0.6)
        if ticks is not None:
            cb.set_ticks(ticks)
            if ticklabels is not None:
                cb.set_ticklabels(ticklabels)
            cb.ax.minorticks_off()
        if label:
            cb.set_label(label, fontsize=FS['small'], labelpad=2)
        return cb

    def legend(self, handles, x, y, ncol=1, anchor='upper center', **kw):
        """figure-level legend anchored at (x, y) mm"""
        kw.setdefault('fontsize', FS['legend'])
        kw.setdefault('frameon', False)
        kw.setdefault('handlelength', 1.8)
        kw.setdefault('columnspacing', 1.4)
        kw.setdefault('handletextpad', 0.5)
        return self.fig.legend(handles=handles, loc=anchor, bbox_to_anchor=(x / self.W, 1 - y / self.H), ncol=ncol, **kw)

    def check(self):
        """warn about text outside the canvas and overlapping panel boxes (tick labels / titles included)"""
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        fb = self.fig.bbox
        tol = 0.4 / 25.4 * self.fig.dpi
        issues = []
        hidden = set()
        for a in self.fig.axes:
            for axis in (a.xaxis, a.yaxis):
                lo, hi = sorted(axis.get_view_interval())
                for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                    loc = tk.get_loc()
                    if loc is None or loc < lo - 1e-09 * max(1, abs(lo)) or loc > hi + 1e-09 * max(1, abs(hi)):
                        hidden.update({id(tk.label1), id(tk.label2)})
        for t in self.fig.findobj(matplotlib.text.Text):
            if id(t) in hidden or not t.get_visible() or (not t.get_text().strip()):
                continue
            try:
                bb = t.get_window_extent(r)
            except Exception:
                continue
            if bb.x0 < fb.x0 - tol or bb.x1 > fb.x1 + tol or bb.y0 < fb.y0 - tol or (bb.y1 > fb.y1 + tol):
                issues.append(f"outside canvas: '{t.get_text()[:30]}'")
        boxes = []
        for a in self.axes:
            if getattr(a, '_fg_overlap_ok', False) or not a.get_visible():
                continue
            try:
                boxes.append((a, a.get_tightbbox(r)))
            except Exception:
                pass
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                b1, b2 = (boxes[i][1], boxes[j][1])
                ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                if ix > tol and iy > tol:
                    l1 = getattr(boxes[i][0], '_fg_cell', None)
                    l2 = getattr(boxes[j][0], '_fg_cell', None)
                    issues.append(f'panels overlap by {ix / self.fig.dpi * 25.4:.1f} x {iy / self.fig.dpi * 25.4:.1f} mm: {l1} / {l2}')
        return issues

def leg(ax, **kw):
    """axes legend in the house style"""
    kw.setdefault('fontsize', FS['legend'])
    kw.setdefault('frameon', True)
    kw.setdefault('fancybox', False)
    kw.setdefault('edgecolor', 'black')
    kw.setdefault('framealpha', 1.0)
    kw.setdefault('handlelength', 1.6)
    kw.setdefault('borderpad', 0.35)
    kw.setdefault('labelspacing', 0.3)
    kw.setdefault('handletextpad', 0.5)
    lg = ax.legend(**kw)
    lg.get_frame().set_linewidth(0.6)
    return lg

def frame(ax, lw=0.8):
    for s in ax.spines.values():
        s.set_linewidth(lw)
    ax.tick_params(width=lw, length=3)
