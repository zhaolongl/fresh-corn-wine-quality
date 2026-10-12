"""Reusable plotting primitives for the final English manuscript figures."""
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.transforms as mtrans
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial import ConvexHull
from scipy.stats import f as fdist
from .layout import FS, leg, plt

SHAP_CMAP = mcolors.LinearSegmentedColormap.from_list('lohi', ['#2166AC', '#9EC5E0', '#F2E6D8', '#E8957A', '#B2182B'])
DRY_C, WET_C = '#D9603F', '#2F6DB5'


def class_totals(v, keys, classes):
    """Descriptive sums and RSS dispersion, assuming independent compounds.

    Replicate covariance is unavailable. No inferential test is performed on
    aggregate classes, and no class-level significance letters are generated.
    """
    totals = {}
    for category in ['总量'] + list(classes):
        subset = v if category == '总量' else v[v['类别'] == category]
        means = [float(subset[k + '_均值'].fillna(0).sum()) for k in keys]
        sds = [float(np.sqrt((subset[k + '_SD'].fillna(0) ** 2).sum())) for k in keys]
        counts = [int(subset[k + '_均值'].notna().sum()) for k in keys]
        totals[category] = {'均值': means, 'SD': sds,
                            '字母': ['nd' if m == 0 else '' for m in means],
                            '数量': counts}
    return totals


def fmt(x, nd=2):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def below(ax, pts):
    """transform: x in data units, y = axis bottom shifted down by pts points"""
    return mtrans.offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=-pts, units="points")


def mk(marker, color, label, ls="none", lw=1.0, ms=4.0, mec="black"):
    return plt.Line2D([], [], marker=marker, ls=ls, lw=lw, color=color, mec=mec if marker else None, mew=0.4, ms=ms, label=label)


def gbars(ax, groups, ylabel, ylim=None, sub_fs=FS["small"], glab_pts=None, width=0.74, letter_fs=FS["annot"]):
    """groups: list of dict(label, means, sds, letters, colors, sub)"""
    x = 0.0; pos, bounds, hi = [], [], []
    for g in groups:
        n = len(g["means"]); xs = x + np.arange(n); sds = np.nan_to_num(np.asarray(g["sds"], float))
        ax.bar(xs, g["means"], width=width, color=g["colors"], edgecolor="black", linewidth=0.5, zorder=2)
        ax.errorbar(xs, g["means"], yerr=sds, fmt="none", ecolor="black", elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3)
        hi += list(np.asarray(g["means"], float) + sds); pos.append(xs); bounds.append((x - 0.5, x + n - 0.5)); x += n + 0.6
    top = max(hi); ylim = ylim or (0, top * 1.2); ax.set_ylim(*ylim); r_ = ylim[1] - ylim[0]
    bb = ax.get_position(); fw, fh = ax.figure.get_size_inches() * 25.4
    mm_x = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (bb.width * fw) if ax.get_xlim()[1] > ax.get_xlim()[0] else 1.0
    x0_, x1_ = bounds[0][0] - 0.3, bounds[-1][1] + 0.3; mm_x = (x1_ - x0_) / (bb.width * fw); mm_y = r_ / (bb.height * fh)
    cw, th = 0.42 * letter_fs * 25.4 / 72, 0.95 * letter_fs * 25.4 / 72      # character width and text height (mm)
    for g, xs in zip(groups, pos):
        prev = None
        for xx, m, s, l in zip(xs, g["means"], np.nan_to_num(np.asarray(g["sds"], float)), g["letters"]):
            if not l:
                prev = None; continue
            yy = m + s + r_ * 0.015; w = len(l) * cw * mm_x
            if prev is not None and (w + prev[1]) / 2 > (xx - prev[0]) * 0.98 and abs(yy - prev[2]) < th * mm_y:
                yy = prev[2] + th * mm_y                       # stagger a letter group that would collide with its neighbour
            ax.text(xx, yy, l, ha="center", va="bottom", fontsize=letter_fs, fontweight="bold")
            prev = (xx, w, yy)
    for (a, b), (a2, _) in zip(bounds[:-1], bounds[1:]):
        ax.axvline((b + a2) / 2, color="black", lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(bounds[0][0] - 0.3, bounds[-1][1] + 0.3); ax.set_ylabel(ylabel)
    if groups[0].get("sub") is not None:
        ax.set_xticks(np.concatenate(pos)); ax.set_xticklabels(sum([list(g["sub"]) for g in groups], []), fontsize=sub_fs)
        ax.tick_params(axis="x", length=2, pad=1.5)
    else:
        ax.set_xticks([])
    gp = glab_pts if glab_pts is not None else (11 if groups[0].get("sub") is not None else 3)
    for g, (a, b) in zip(groups, bounds):
        ax.text((a + b) / 2, 0, g["label"], transform=below(ax, gp), ha="center", va="top", fontsize=FS["small"], fontweight="bold")
    return pos


def class_groups(v, keys, sub, colors, classes):
    ct = class_totals(v, keys, sorted({c for c in classes if c != "总量"}))
    return [dict(label=c, means=ct[c]["均值"], sds=ct[c]["SD"], letters=ct[c]["字母"], colors=[colors[k] for k in keys], sub=[sub[k] for k in keys]) for c in classes], ct


def radar(ax, df, cols, colors, markers, labels, ticks, log=False, lab_off=0.16):
    attrs = list(df.index); n = len(attrs)
    if log:
        floor = ticks[0]; df = df.copy().astype(float).apply(lambda s: np.log10(np.clip(s, floor, None)))
        tl = ticks; ticks = [np.log10(t) for t in ticks]
    else:
        tl = ticks
    rmin, rmax = ticks[0], ticks[-1]
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False); aa = np.concatenate([ang, ang[:1]])
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1); ax.set_ylim(rmin, rmax)
    ax.grid(False); ax.spines["polar"].set_visible(False); ax.set_yticks([]); ax.set_xticks([])
    for t, l in zip(ticks, tl):
        ax.plot(aa, [t] * (n + 1), color="#BDBDBD", lw=0.5, zorder=0)
        if t > rmin:
            ax.text(-np.pi / n, t, f"{l:g}", fontsize=FS["small"], color="#555555", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=0.2))
    for a in ang:
        ax.plot([a, a], [rmin, rmax], color="#BDBDBD", lw=0.5, zorder=0)
    for c in cols:
        v = df[c].values.astype(float); vv = np.concatenate([v, v[:1]])
        ax.plot(aa, vv, color=colors[c], lw=1.1, marker=markers[c], ms=3, mec="white", mew=0.4, label=labels[c], zorder=3)
        ax.fill(aa, vv, color=colors[c], alpha=0.08, zorder=2)
    for a, lab in zip(ang, attrs):
        ax.text(a, rmax + (rmax - rmin) * lab_off, lab, ha="center", va="center", fontsize=FS["small"], fontweight="bold")


def pca_scatter(ax, df, colors, markers, labels, xlab, ylab):
    t1 = df["PC1"].values; t2 = df["PC2"].values; n = len(t1); A = 2
    s1 = np.sqrt(np.sum((t1 - t1.mean()) ** 2) / (n - 1)); s2 = np.sqrt(np.sum((t2 - t2.mean()) ** 2) / (n - 1))
    T2 = A * (n ** 2 - 1) / (n * (n - A)) * fdist.ppf(0.95, A, n - A)
    ax.add_patch(Ellipse((t1.mean(), t2.mean()), 2 * s1 * np.sqrt(T2), 2 * s2 * np.sqrt(T2), fc="none", ec="#9E9E9E", lw=0.8))
    ax.axhline(0, color="#9E9E9E", lw=0.6, zorder=0); ax.axvline(0, color="#9E9E9E", lw=0.6, zorder=0)
    for g, sub in df.groupby("样品", sort=False):
        ax.scatter(sub.PC1, sub.PC2, s=22, marker=markers[g], c=colors[g], edgecolors="black", linewidths=0.4, zorder=3, label=labels[g])
    rx = s1 * np.sqrt(T2) * 1.12; ry = s2 * np.sqrt(T2) * 1.12
    ax.set_xlim(t1.mean() - rx, t1.mean() + rx); ax.set_ylim(t2.mean() - ry, t2.mean() + ry)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)


def hull(ax, P, color):
    if len(P) >= 3:
        h = ConvexHull(P); ax.add_patch(mpatches.Polygon(P[h.vertices], closed=True, fc=color, ec=color, alpha=0.13, lw=0.8, zorder=1))


def cluster_heatmap(cv, x, y, w, h, M, rowlabels, collabels, rowstrip, strip_colors, cmap, norm, annot=None, side=None, side_label="", nan_color="#E0E0E0"):
    """mm-based clustered heatmap: row dendrogram | class strip | heatmap | (side bars) | row labels; column dendrogram on top"""
    dw, sw, hw = 8.0, 2.2, 6.6 * M.shape[1]
    Mc = np.nan_to_num(M, nan=np.nanmin(M) - 0.5)
    ro = dendrogram(linkage(Mc, "average"), no_plot=True)["leaves"][::-1]; co = dendrogram(linkage(Mc.T, "average"), no_plot=True)["leaves"]
    top = 5.0
    axd = cv.ax(x, y + top, dw, h - top, overlap_ok=True); dendrogram(linkage(Mc, "average"), ax=axd, orientation="left", no_labels=True, link_color_func=lambda k: "black"); axd.axis("off")
    axc = cv.ax(x + dw + sw, y, hw, top - 0.6, overlap_ok=True); dendrogram(linkage(Mc.T, "average"), ax=axc, no_labels=True, link_color_func=lambda k: "black"); axc.axis("off")
    axs = cv.ax(x + dw, y + top, sw * 0.85, h - top, overlap_ok=True)
    axs.imshow(np.arange(len(ro))[:, None], aspect="auto", cmap=mcolors.ListedColormap([strip_colors[rowstrip[r]] for r in ro])); axs.set_xticks([]); axs.set_yticks([])
    axh = cv.ax(x + dw + sw, y + top, hw, h - top, overlap_ok=True)
    Mo = M[np.ix_(ro, co)]
    im = axh.imshow(np.ma.masked_invalid(Mo), aspect="auto", cmap=cmap, norm=norm)
    for i in range(Mo.shape[0]):
        for j in range(Mo.shape[1]):
            if np.isnan(Mo[i, j]):
                axh.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fc=nan_color, ec="white", lw=0.3))
            if annot is not None and annot[ro[i]][co[j]]:
                v = Mo[i, j]; dark = (not np.isnan(v)) and norm(v) > 0.62
                axh.text(j, i, annot[ro[i]][co[j]], ha="center", va="center", fontsize=FS["small"], fontweight="bold", color="white" if dark else "black")
    axh.set_xticks(range(len(co))); axh.set_xticklabels([collabels[c] for c in co], fontsize=FS["small"], fontweight="bold")
    axh.set_yticks([]); axh.tick_params(length=0)
    for i in range(Mo.shape[0] + 1):
        axh.axhline(i - 0.5, color="white", lw=0.4)
    for j in range(Mo.shape[1] + 1):
        axh.axvline(j - 0.5, color="white", lw=0.4)
    xl = x + dw + sw + hw + 0.8
    if side is not None:
        axb = cv.ax(xl, y + top, 12, h - top, overlap_ok=True)
        axb.barh(range(len(ro)), [side[r] for r in ro], color="#9E9E9E", edgecolor="black", linewidth=0.4, height=0.7)
        axb.set_ylim(len(ro) - 0.5, -0.5); axb.set_yticks([]); axb.axvline(1, color="black", lw=0.6, ls=(0, (3, 2)))
        axb.set_xlabel(side_label, fontsize=FS["small"]); axb.tick_params(axis="x", labelsize=FS["small"])
        for s in ("top", "right"):
            axb.spines[s].set_visible(False)
        xl += 13
    axl = cv.ax(xl, y + top, max(1.0, x + w - xl), h - top, overlap_ok=True); axl.axis("off"); axl.set_ylim(len(ro) - 0.5, -0.5); axl.set_xlim(0, 1)
    for i, r in enumerate(ro):
        axl.text(0.0, i, rowlabels[r], ha="left", va="center", fontsize=FS["small"])
    return im, axh


def bar_panel(ax, v, code, samples, labels, colors, title, ylabel=None):
    r = v[v["编号"] == code].iloc[0]
    m = [r[s + "_均值"] if r[s + "_均值"] == r[s + "_均值"] else 0.0 for s in samples]
    sd = [r[s + "_SD"] if r[s + "_SD"] == r[s + "_SD"] else 0.0 for s in samples]
    lt = [r[s + "_字母"] if isinstance(r[s + "_字母"], str) else "" for s in samples]
    x = np.arange(len(samples))
    ax.bar(x, m, width=0.64, color=[colors[s] for s in samples], edgecolor="black", linewidth=0.5, zorder=2)
    ax.errorbar(x, m, yerr=sd, fmt="none", ecolor="black", elinewidth=0.6, capsize=1.8, capthick=0.6, zorder=3)
    top = max(np.array(m) + np.array(sd)); ax.set_ylim(0, top * 1.2 if top > 0 else 1)
    for xi, mi, si, li in zip(x, m, sd, lt):
        if mi == 0:
            ax.text(xi, top * 0.03, "ND", ha="center", va="bottom", fontsize=FS["small"], color="#666666")
        else:
            ax.text(xi, mi + si + top * 0.03, li, ha="center", va="bottom", fontsize=FS["annot"], fontweight="bold")
    for xi, s, mi in zip(x, samples, m):
        o = r[s + "_OAV"]
        txt = "—" if (o != o or mi == 0) else (fmt(o, 2) if o < 1 else fmt(o, 1) if o < 10 else fmt(o, 0))
        ax.text(xi, 0, txt, transform=below(ax, 10.5), ha="center", va="top", fontsize=FS["small"], color="#8C1C13", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([labels[s] for s in samples], fontsize=FS["small"], fontweight="bold"); ax.tick_params(axis="x", pad=1.5)
    ax.set_xlim(-0.6, len(samples) - 0.4)
    ax.set_title(title, fontsize=FS["small"], fontweight="bold", pad=2, linespacing=1.15)
    if ylabel:
        ax.set_ylabel(ylabel)


def perm_panel(ax, P):
    perm = P["perm"]
    ax.scatter(perm[:, 0], perm[:, 1], s=7, c="#2E9A6E", edgecolors="none", alpha=0.75, label="R²")
    ax.scatter(perm[:, 0], perm[:, 2], s=7, c="#2F6DB5", edgecolors="none", alpha=0.75, label="Q²")
    ax.scatter([1], [P["R2Y"]], s=20, c="#2E9A6E", edgecolors="black", linewidths=0.5); ax.scatter([1], [P["Q2"]], s=20, c="#2F6DB5", edgecolors="black", linewidths=0.5)
    for k, col in ((1, "#2E9A6E"), (2, "#2F6DB5")):
        xs_ = np.r_[perm[:, 0], 1.0]; ys_ = np.r_[perm[:, k], P["R2Y"] if k == 1 else P["Q2"]]
        b = np.polyfit(xs_, ys_, 1); xx = np.array([0, 1]); ax.plot(xx, b[1] + b[0] * xx, color=col, lw=0.9, ls=(0, (4, 2)))
    ax.axhline(0, color="#9E9E9E", lw=0.6)
    ax.set_xlabel("Correlation between permuted and original Y"); ax.set_ylabel("R² / Q²")
    leg(ax, loc="lower right", handletextpad=0.1)
