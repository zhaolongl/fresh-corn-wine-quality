"""Small helpers retained from the original plotting implementation.

Data are passed explicitly; this module neither loads bundles nor writes files.
"""
import re
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse
from .statistics import duncan_letters

VAR = ['HN', 'BT', 'HT']
HEAT = ['WB', 'HK', 'ZZ', 'SZ']
SERIES = ['果香', '花香', '甜香', '青草/蘑菇', '辛香/其他']
SER_COL = {'果香': '#D9603F', '花香': '#C2185B', '甜香': '#E3A21A', '青草/蘑菇': '#2E9A6E', '辛香/其他': '#5D6D7E'}
TIMES = {'WB': ['1.0', '1.5', '2.0', '2.5', '3.0'], 'HK': ['30', '40', '50', '60', '70'], 'ZZ': ['30', '40', '50', '60', '70'], 'SZ': ['30', '40', '50', '60', '70']}
HEATL = {"WB":"Microwave", "HK":"Baking", "ZZ":"Steaming", "SZ":"Boiling"}

def parse_ms(s):
    m = re.match(r"^([\d.]+)±([\d.]+)([a-z]*)$", str(s))
    return (float(m.group(1)), float(m.group(2)), m.group(3)) if m else (float(s), 0.0, "")


def venn3(ax, regions, keys, colors, names):
    pos = {keys[0]: (-0.55, 0.35), keys[1]: (0.55, 0.35), keys[2]: (0.0, -0.55)}
    for k in keys:
        ax.add_patch(Circle(pos[k], 1.0, fc=colors[k], alpha=0.22, lw=0))
        ax.add_patch(Circle(pos[k], 1.0, fc="none", ec=colors[k], lw=1.1))
    a, b, c = keys
    txt = {a: (-1.0, 0.7), b: (1.0, 0.7), c: (0.0, -1.1), f"{a}&{b}": (0.0, 0.85), f"{a}&{c}": (-0.62, -0.38), f"{b}&{c}": (0.62, -0.38), f"{a}&{b}&{c}": (0.0, 0.05)}
    for key, (x, y) in txt.items():
        ax.text(x, y, str(regions[key]), ha="center", va="center", fontsize=8, fontweight="bold")
    for k, (x, y) in {a: (-1.3, 1.55), b: (1.3, 1.55), c: (0.0, -1.8)}.items():
        ax.text(x, y, names[k], ha="center", va="center", fontsize=7, fontweight="bold", color=colors[k])
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-2.0, 1.8); ax.set_aspect("equal"); ax.axis("off")


def venn4(ax, regions, keys, colors, names):
    E = [(0.350, 0.400, 140.0), (0.450, 0.500, 140.0), (0.544, 0.500, 40.0), (0.644, 0.400, 40.0)]
    for k, (x, y, ang) in zip(keys, E):
        ax.add_patch(Ellipse((x, y), 0.72, 0.45, angle=ang, fc=colors[k], alpha=0.18, lw=0))
        ax.add_patch(Ellipse((x, y), 0.72, 0.45, angle=ang, fc="none", ec=colors[k], lw=1.1))
    P = {"0001": (0.85, 0.42), "0010": (0.68, 0.72), "0011": (0.77, 0.59), "0100": (0.32, 0.72), "0101": (0.71, 0.30), "0110": (0.50, 0.66),
         "0111": (0.65, 0.50), "1000": (0.14, 0.42), "1001": (0.50, 0.17), "1010": (0.29, 0.30), "1011": (0.39, 0.24), "1100": (0.23, 0.59),
         "1101": (0.61, 0.24), "1110": (0.35, 0.50), "1111": (0.50, 0.38)}
    for bits, (x, y) in P.items():
        comb = "&".join(k for k, bb in zip(keys, bits) if bb == "1")
        ax.text(x, y, str(regions.get(comb, 0)), ha="center", va="center", fontsize=7.5, fontweight="bold")
    for k, (x, y) in zip(keys, [(0.10, 0.12), (0.17, 0.88), (0.83, 0.88), (0.90, 0.12)]):
        ax.text(x, y, names[k], ha="center", va="center", fontsize=7, fontweight="bold", color=colors[k])
    ax.set_xlim(0, 1); ax.set_ylim(0.02, 0.96); ax.set_aspect("equal"); ax.axis("off")


def heat_groups(d, ind, letters_from="thesis"):
    hs = d["heat_series"]
    groups = []
    for k in HEAT:
        s = hs[(hs["指标"] == ind) & (hs["处理"] == k)]
        means = s["均值"].values; sds = s["SD"].fillna(0).values
        if letters_from == "thesis":
            L = list(s["字母"])
        else:
            L, _, _ = duncan_letters(means, sds)
        groups.append((k, HEATL[k], TIMES[k], means, sds, L))
    return groups


def tier_effects(d):
    """Within-stage descriptive maximum/minimum ratios, not causal effects."""
    hs = d["heat_series"]
    tv = d["tab_wine_var"].set_index("指标"); tm = d["tab_mode"].set_index("指标")
    def r_(vals):
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return max(vals) / min(vals)
    def fromtab(t, row, cols):
        return [parse_ms(t.loc[row, c])[0] for c in cols]
    H = lambda ind: hs[hs["指标"] == ind]["均值"].values
    rows = {
        "酒精度": (r_(fromtab(tv, "酒精度（% vol）", VAR)), r_(H("酒精度")), r_(fromtab(tm, "酒精度（% vol）", ["全浆发酵", "清汁发酵", "浸渍后发酵"]))),
        "总酚": (r_(fromtab(tv, "总酚（mg GAE/L）", VAR)), r_(H("总酚")), r_(fromtab(tm, "总酚（mg GAE/L）", ["全浆发酵", "清汁发酵", "浸渍后发酵"]))),
        "总黄酮": (r_(fromtab(tv, "总黄酮（mg RE/L）", VAR)), r_(H("总黄酮")), r_(fromtab(tm, "总黄酮（mg RE/L）", ["全浆发酵", "清汁发酵", "浸渍后发酵"]))),
        "可溶性蛋白": (r_(fromtab(tv, "可溶性蛋白（mg BSA/L）", VAR)), r_(H("可溶性蛋白")), r_(fromtab(tm, "可溶性蛋白（mg BSA/L）", ["全浆发酵", "清汁发酵", "浸渍后发酵"]))),
        "DPPH 清除能力": (r_(fromtab(tv, "DPPH 自由基清除能力（μg VCE/mL）", VAR)), r_(H("DPPH")), r_(fromtab(tm, "DPPH 自由基清除能力（μg VCE/mL）", ["全浆发酵", "清汁发酵", "浸渍后发酵"]))),
        "总还原能力": (r_(fromtab(tv, "总还原能力（μg VCE/mL）", VAR)), r_(H("总还原能力")), r_(fromtab(tm, "总还原能力（μg VCE/mL）", ["全浆发酵", "清汁发酵", "浸渍后发酵"]))),
        "挥发性化合物总量": (r_([d["cls_var"][k + "_含量"].sum() for k in VAR]), r_([d["cls_heat"][k + "_含量"].sum() for k in HEAT]), np.nan),
        "关键香气化合物数": (r_([int((d["vol_var"][k + "_OAV"] >= 1).sum()) for k in VAR]), r_([int((d["vol_heat"][k + "_OAV"] >= 1).sum()) for k in HEAT]), np.nan),
        "感官评分": (r_(d["qda_var"].sum().values), r_(d["qda_heat"].sum().values), r_([76.5, 74.5, 70.3])),
        "非糖固形物": (np.nan, np.nan, r_(fromtab(tm, "非糖固形物（g/L）", ["全浆发酵", "清汁发酵", "浸渍后发酵"]))),
        "氨基酸态氮": (np.nan, np.nan, r_(fromtab(tm, "氨基酸态氮（g/L）", ["全浆发酵", "清汁发酵", "浸渍后发酵"]))),
    }
    return pd.DataFrame(rows, index=["品种", "热预处理", "发酵方式"]).T
