"""Reproduce the final manuscript figures from the reviewed numeric bundles.

Fig1-7 and FigS2-S4 are redrawn; FigS1 is a supplied SIMCA-derived asset.
These routines reuse archived model results and do not retrain models.
"""
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib import font_manager

from ..dataio import load_bundle, resource_path
from . import primitives as P
from . import legacy_helpers as mf
from . import layout as G
from .layout import Canvas, cells, leg, FS, plt, GRID_STYLE
from .style import (STYLE, ramp, HEAT_BASE, HEAT_SOLID, HEATM, VAR_SOLID,
                    VARM, MODE_SOLID, MODEM, CLASS_COL, RDBU, REDS)

LOGGER = logging.getLogger(__name__)
FIGURE_NAMES = ('Fig1', 'Fig2', 'Fig3', 'Fig4', 'Fig5', 'Fig6', 'Fig7',
                'FigS1', 'FigS2', 'FigS3', 'FigS4')

VAR = ['HN', 'BT', 'HT']
HEAT = ['WB', 'HK', 'ZZ', 'SZ']
HEATL = {'WB': 'Microwave', 'HK': 'Baking', 'ZZ': 'Steaming', 'SZ': 'Boiling'}
HEATS = {'WB': 'MW', 'HK': 'BK', 'ZZ': 'ST', 'SZ': 'BO'}
VARL = {'HN': 'Black waxy (HN)', 'BT': 'White sweet (BT)', 'HT': 'Yellow sweet (HT)'}
MODEL = {'全浆发酵': 'Whole-mash fermentation', '浸渍后发酵': 'Post-maceration fermentation', '清汁发酵': 'Clear-juice fermentation'}
MODES = {'全浆发酵': 'WMF', '浸渍后发酵': 'PMF', '清汁发酵': 'CJF'}
CLS = {'总量': 'Total', '醇类': 'Alcohols', '酯类': 'Esters', '酸类': 'Acids', '醛酮类': 'Aldehydes &\nketones', '其它类': 'Others', '萜烯类': 'Terpenes', '烷烃类': 'Alkanes'}
CLS1 = {k: v.replace('\n', ' ') for k, v in CLS.items()}
SER = {'果香': 'Fruity', '花香': 'Floral', '甜香': 'Sweet', '青草/蘑菇': 'Green/mushroom', '辛香/其他': 'Spicy/other'}
SER2 = {'果香': 'Fruity', '花香': 'Floral', '甜香': 'Sweet', '青草/蘑菇': 'Green/\nmushroom', '辛香/其他': 'Spicy/\nother'}
SHORT = {'1,1,6-Trimethyl-1,2-dihydronaphthalene': 'TDN', '1,2,3,4-Tetrahydro-1,1,6-trimethylnaphthalene': 'α-Ionene'}
FL = {'酒精度': 'Alcohol', '还原糖': 'Reducing sugar', '总酸': 'Total acid', 'pH': 'pH', '总酚': 'Total phenolics', '总黄酮': 'Total flavonoids', '可溶性蛋白': 'Soluble protein', 'DPPH': 'DPPH scavenging', '总还原能力': 'Reducing power'}
SENS_VAR = {'澄清度': 'Clarity', '色泽': 'Color', '香气': 'Aroma', '口感': 'Taste', '典型性': 'Typicality'}
QDA = {'色泽': 'Color', '澄清度': 'Clarity', '香气纯正度': 'Aroma purity', '香气浓郁度': 'Aroma intensity', '香气质量': 'Aroma quality', '口感浓郁度': 'Palate fullness', '口感持久性': 'Persistence', '口感涩度': 'Astringency', '整体风格': 'Typicality', '整体平衡性': 'Overall balance'}
VAR_CODES = [('A48', 'Ethyl cinnamate\n(cinnamon, fruity)'), ('A33', 'Ethyl octanoate\n(pineapple, sweet)'), ('A16', 'Ethyl hexanoate\n(pineapple, fruity)'), ('A10', 'Isoamyl acetate\n(banana)')]
LABPOS = {'酒精度': (-0.5, 0.66, 'right', 'Alcohol'), '总黄酮': (-0.5, 0.49, 'right', 'Total\nflavonoids'), '还原糖': (-0.5, 0.27, 'right', 'Reducing\nsugar'), 'DPPH': (-0.5, 0.1, 'right', 'DPPH'), '总酸': (-0.5, -0.03, 'right', 'Total acid'), 'pH': (0.42, 0.42, 'center', 'pH'), '总酚': (-0.5, -0.3, 'right', 'Total\nphenolics'), '可溶性蛋白': (0.06, -0.72, 'left', 'Soluble\nprotein'), '总还原能力': (-0.36, -0.66, 'right', 'Reducing\npower')}
HEAT_CODES = [('A3', 'Isoamyl acetate\n(banana)'), ('A6', 'Ethyl propanoate\n(pineapple)'), ('A20', 'Ethyl hexanoate\n(pineapple, fruity)'), ('A11', 'Isoamyl butyrate\n(fruity, tropical)'), ('B25', '1-Octen-3-ol\n(mushroom)')]
TE_ROW = {'酒精度': 'Alcohol', '总酚': 'Total phenolics', '总黄酮': 'Total flavonoids', '可溶性蛋白': 'Soluble protein', 'DPPH 清除能力': 'DPPH scavenging', '总还原能力': 'Reducing power', '挥发性化合物总量': 'Total volatiles', '关键香气化合物数': 'No. of compounds with OAV ≥ 1', '感官评分': 'Sensory score', '非糖固形物': 'Non-sugar solids', '氨基酸态氮': 'Amino acid nitrogen'}
TE_COL = {'品种': 'Cultivar', '热预处理': 'Thermal\npretreatment', '发酵方式': 'Fermentation\nmethod'}
SPEC9 = [('酒精度', 'Alcohol (% vol)', (0, 13.2), 'calc'), ('还原糖', 'Reducing sugar (g/L)', (0, 11), 'calc'), ('总酸', 'Total acid (g/L)', (0, 4.6), 'calc'), ('pH', 'pH', (0, 4.6), 'calc'), ('总酚', 'Total phenolics\n(mg GAE/L)', (0, 150), 'thesis'), ('总黄酮', 'Total flavonoids\n(mg RE/L)', (0, 95), 'thesis'), ('可溶性蛋白', 'Soluble protein\n(mg BSA/L)', (0, 180), 'thesis'), ('DPPH', 'DPPH scavenging\n(μg VCE/mL)', (0, 65), 'thesis'), ('总还原能力', 'Reducing power\n(μg VCE/mL)', (0, 90), 'thesis')]
DRY_C, WET_C = P.DRY_C, P.WET_C
fmt, mk, hull, below = P.fmt, P.mk, P.hull, P.below

class FigureBuilder:
    """A figure-generation session with independent inputs and output paths."""

    def __init__(self, output_dir, dpi=300):
        self.output_dir = Path(output_dir).expanduser().resolve()
        if isinstance(dpi, bool) or int(dpi) != dpi or dpi <= 0:
            raise ValueError('dpi must be a positive integer')
        self.dpi = int(dpi)
        self.d = load_bundle('paper')
        self.amap = load_bundle('aroma')
        self.ml = load_bundle('ml')
        self.ka = load_bundle('aroma_candidates')
        self.W, self.FEAT, self.METHODS = self.ml['W'], self.ml['FEAT'], self.ml['METHODS']
        self.EN = json.loads(resource_path('compound_names.json').read_text(encoding='utf-8-sig'))
        self.files = []
        self.layout_issues = {}

    def save(self, fig, name):
        """Write PNG and vector PDF at fixed physical dimensions."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            for suffix in ('.png', '.pdf'):
                destination = self.output_dir / (name + suffix)
                options = {'dpi': self.dpi} if suffix == '.png' else {}
                fig.savefig(destination, facecolor='white', **options)
                self.files.append(destination)
        finally:
            plt.close(fig)

    def out(self, canvas, name):
        self.layout_issues[name] = canvas.check()
        for issue in self.layout_issues[name]:
            LOGGER.warning('%s layout: %s', name, issue)
        self.save(canvas.fig, name)

    def figS1(self):
        """Copy the supplied SIMCA image/PDF; do not imply model reproduction."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for suffix in ('.png', '.pdf'):
            source = resource_path('assets/FigS1' + suffix)
            destination = self.output_dir / ('FigS1' + suffix)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            self.files.append(destination)


    def en_groups(self, groups):
        for g in groups:
            g['label'] = CLS[g['label']]
        return groups

    def key_bars(self, cv, cells_, codes, v, samples, labels, colors, letter):
        """Tier I key aroma compounds: one bar panel per compound, OAV under the x axis; one panel letter for the whole row"""
        for i, (code, title) in enumerate(codes):
            ax = cv.panel(cells_[i], letter if i == 0 else None, L=14 if i == 0 else 10, B=12, R=2, T=10)
            ax.set_label(f'{letter}{i + 1}')
            P.bar_panel(ax, v, code, samples, labels, colors, title, ylabel='Content (μg/L)' if i == 0 else None)
            if i == 0:
                ax.text(-0.5, 0, 'OAV', transform=below(ax, 10.5), ha='right', va='top', fontsize=FS['small'], color='#8C1C13', fontweight='bold')

    def fig1(self):
        v1 = self.d['vol_var']
        cv = Canvas(180, 220)
        sub = {k: k for k in VAR}
        gA1, ct = P.class_groups(v1, VAR, sub, VAR_SOLID, ['总量', '醇类', '酯类'])
        gA2, _ = P.class_groups(v1, VAR, sub, VAR_SOLID, ['酸类', '醛酮类', '其它类'])
        cv.letter(0, 1, 'A')
        a1 = cv.ax(16, 7, 92, 23)
        a1.set_label('A1')
        P.gbars(a1, self.en_groups(gA1), 'Content (μg/L)')
        a2 = cv.ax(16, 43, 92, 20)
        a2.set_label('A2')
        P.gbars(a2, self.en_groups(gA2), 'Content (μg/L)')
        for a in (a1, a2):
            a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f'{v:g}'))
        ax = cv.panel(G.Cell(114, 1, 66, 73), 'B', L=14, B=10, R=3, T=6)
        ax.set_label('B')
        P.pca_scatter(ax, self.d['pca_var'], VAR_SOLID, VARM, sub, 'PC1 (49.9%)', 'PC2 (42.0%)')
        leg(ax, loc='upper right')
        cv.letter(0, 79, 'C')
        axV = cv.ax(1, 90, 48, 50)
        axV.set_label('C')
        mf.venn3(axV, self.d['venn_var'], VAR, VAR_SOLID, sub)
        for t in axV.texts:
            t.set_fontsize(FS['label'] if t.get_text() in VAR else FS['annot'] + 0.5)
        vip = self.d['vip_var'].copy()
        M = vip[VAR].values.astype(float)
        M0 = np.nan_to_num(M)
        Mz = (M0 - M0.mean(1, keepdims=True)) / (M0.std(1, ddof=0, keepdims=True) + 1e-09)
        cls_of = vip['编号'].map(self.d['vol_var'].set_index('编号')['类别']).tolist()
        cv.letter(52, 79, 'D')
        im, axh = P.cluster_heatmap(cv, 54, 84, 76, 60, Mz, [self.EN[n] for n in vip['化合物']], dict(enumerate(VAR)), cls_of, CLASS_COL, RDBU, mcolors.Normalize(-1.5, 1.5), side=vip['VIP'].tolist(), side_label='VIP')
        axh.set_label('D')
        cv.hcbar(58, 154.5, 22, im, label='Z-score of content', ticks=[-1.5, 0, 1.5])
        hs_ = [mpatches.Patch(fc=CLASS_COL[c], ec='black', lw=0.4, label=CLS1[c]) for c in ['酯类', '醇类', '酸类', '醛酮类', '其它类']]
        cv.legend(hs_, 108, 152.5, ncol=2, anchor='upper center', handlelength=1.0, columnspacing=0.8, fontsize=FS['small'], labelspacing=0.25)
        ser = pd.DataFrame(index=mf.SERIES)
        for k in VAR:
            ser[k] = [v1[v1['编号'].isin([c for c, s in self.amap['var'].items() if s == S])][k + '_OAV'].pipe(lambda x: x[x >= 0.1].sum()) for S in mf.SERIES]
        ser.index = [SER2[s] for s in ser.index]
        cv.letter(132, 79, 'E')
        axE = cv.ax(137, 93, 34, 34, polar=True)
        axE.set_label('E')
        P.radar(axE, ser, VAR, VAR_SOLID, VARM, sub, [0.1, 1, 10, 100], log=True, lab_off=0.26)
        cv.legend([mk(VARM[k], VAR_SOLID[k], VARL[k], ls='-', lw=1.1, ms=3.5, mec='white') for k in VAR], 156, 139.5, ncol=1, anchor='upper center', labelspacing=0.25)
        r3 = cells(0, 165, [48, 44, 44, 44], [55])
        self.key_bars(cv, r3[0], VAR_CODES, self.ka['var'], VAR, sub, VAR_SOLID, 'F')
        self.out(cv, 'Fig1')
        return (ct, ser)

    def vip_panel(self, ax, vip, col, labels=True):
        order = np.argsort(vip)
        ax.barh(range(len(self.FEAT)), vip[order], color=[col if v > 1 else '#BDBDBD' for v in vip[order]], edgecolor='black', linewidth=0.4, height=0.65, zorder=2)
        ax.set_yticks(range(len(self.FEAT)))
        ax.set_yticklabels([FL[self.FEAT[i]] for i in order], fontsize=FS['small'])
        ax.axvline(1, color='black', lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xlabel('VIP')
        ax.set_xlim(0, vip.max() * 1.25)
        if labels:
            for yi, v in enumerate(vip[order]):
                ax.text(v + 0.03, yi, fmt(v), va='center', fontsize=FS['small'], zorder=3)

    def perm_panel(self, ax, Pm):
        P.perm_panel(ax, Pm)
        ax.set_xlabel('Correlation between permuted and original Y')
        ax.text(0.97, 0.04, f"R² intercept = {fmt(Pm['R2_int'])}\nQ² intercept = {fmt(Pm['Q2_int'])}\nP = {Pm['p']:.3f}".replace('-', '−'), transform=ax.transAxes, fontsize=FS['small'], ha='right', va='bottom')
        lg = ax.get_legend()
        if lg is not None:
            lg.remove()
        leg(ax, loc='lower right', bbox_to_anchor=(1.0, 0.27), frameon=False, handletextpad=0.1)

    def fig2(self):
        cv = Canvas(160, 126)
        r1 = cells(0, 1, [52, 58, 50], [62])
        r2 = cells(0, 64, [54, 52, 54], [61])
        pca = self.ml['pca']
        sc = pca['scores']
        ev = pca['evr']
        Lo = pca['loadings']
        ax = cv.panel(r1[0][0], 'A', L=13, B=10, R=2, T=11)
        ax.set_label('A')
        for m in self.METHODS:
            idx = (self.W['处理'] == m).values
            hull(ax, sc[idx, :2], HEAT_SOLID[m])
            ax.scatter(sc[idx, 0], sc[idx, 1], s=20, marker=HEATM[m], c=HEAT_SOLID[m], edgecolors='black', linewidths=0.4, zorder=3, label=HEATL[m])
        ax.axhline(0, color='#9E9E9E', lw=0.6, zorder=0)
        ax.axvline(0, color='#9E9E9E', lw=0.6, zorder=0)
        ax.set_xlabel(f'PC1 ({100 * ev[0]:.1f}%)')
        ax.set_ylabel(f'PC2 ({100 * ev[1]:.1f}%)')
        ax.set_xlim(-5.2, 5.2)
        ax.set_ylim(-4.2, 4.6)
        leg(ax, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, columnspacing=0.6, handletextpad=0.1, frameon=False, borderaxespad=0.2)
        ax = cv.panel(r1[0][1], 'B', L=13, B=10, R=2, T=11)
        ax.set_label('B')
        for i, f in enumerate(self.FEAT):
            ax.annotate('', xy=(Lo[i, 0], Lo[i, 1]), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color='#2F6DB5', lw=0.9, mutation_scale=7))
            lx, ly, ha, lab = LABPOS[f]
            ax.annotate(lab, xy=(Lo[i, 0], Lo[i, 1]), xytext=(lx, ly), fontsize=FS['small'] - 0.3, ha=ha, va='center', linespacing=0.95, arrowprops=dict(arrowstyle='-', color='#9E9E9E', lw=0.45, shrinkA=0.5, shrinkB=1.5))
        ax.set_xlim(-1.05, 0.75)
        ax.set_ylim(-0.85, 0.85)
        ax.axhline(0, color='#9E9E9E', lw=0.6, zorder=0)
        ax.axvline(0, color='#9E9E9E', lw=0.6, zorder=0)
        ax.add_patch(mpatches.Circle((0, 0), np.sqrt(2 / len(self.FEAT)), fc='none', ec='#9E9E9E', lw=0.6, ls=(0, (3, 2))))
        ax.set_xlabel('PC1 loading')
        ax.set_ylabel('PC2 loading')
        Pm = self.ml['plsda_med']
        t = Pm['scores']
        y = self.W['处理'].isin(['WB', 'HK']).values
        ax = cv.panel(r1[0][2], 'C', L=13, B=10, R=2, T=11)
        ax.set_label('C')
        for flag, lab, col, m_ in ((True, 'Dry heat (MW, BK)', DRY_C, 'o'), (False, 'Moist heat (ST, BO)', WET_C, 's')):
            ax.scatter(t[y == flag, 0], t[y == flag, 1], s=20, marker=m_, c=col, edgecolors='black', linewidths=0.4, zorder=3, label=lab)
            hull(ax, t[y == flag, :2], col)
        ax.axhline(0, color='#9E9E9E', lw=0.6, zorder=0)
        ax.axvline(0, color='#9E9E9E', lw=0.6, zorder=0)
        ax.set_xlabel('t[1]')
        ax.set_ylabel('t[2]')
        ax.text(0.03, 0.97, f"R²X = {Pm['R2X']:.3f}\nR²Y = {Pm['R2Y']:.3f}\nQ² = {Pm['Q2']:.3f}", transform=ax.transAxes, va='top', fontsize=FS['small'])
        leg(ax, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=1, frameon=False, borderaxespad=0.2, handletextpad=0.2, labelspacing=0.2)
        ax = cv.panel(r2[0][0], 'D', L=13, B=10, R=2, T=6)
        ax.set_label('D')
        self.perm_panel(ax, Pm)
        ax = cv.panel(r2[0][1], 'E', L=22, B=10, R=2, T=6)
        ax.set_label('E')
        self.vip_panel(ax, Pm['vip'], DRY_C)
        ax = cv.panel(r2[0][2], 'F', L=22, B=10, R=3, T=6)
        ax.set_label('F')
        co = Pm['coef'][:, 0]
        o2 = np.argsort(co)
        ax.barh(range(len(self.FEAT)), co[o2], color=[DRY_C if c > 0 else WET_C for c in co[o2]], edgecolor='black', linewidth=0.4, height=0.65, zorder=2)
        ax.set_yticks(range(len(self.FEAT)))
        ax.set_yticklabels([FL[self.FEAT[i]] for i in o2], fontsize=FS['small'])
        ax.axvline(0, color='black', lw=0.7)
        ax.set_xlabel('Coefficient (> 0: dry heat)')
        self.out(cv, 'Fig2')

    def fig4(self):
        v2 = self.d['vol_heat']
        cv = Canvas(180, 226)
        gA1, ct = P.class_groups(v2, HEAT, HEATS, HEAT_SOLID, ['总量', '醇类', '酯类'])
        gA2, _ = P.class_groups(v2, HEAT, HEATS, HEAT_SOLID, ['其它类', '酸类', '醛酮类', '萜烯类', '烷烃类'])
        for g in gA1 + gA2:
            g['sub'] = None
        cv.letter(0, 1, 'A')
        cv.legend([mpatches.Patch(fc=HEAT_SOLID[k], ec='black', lw=0.4, label=HEATL[k]) for k in HEAT], 62, 1.0, ncol=4, anchor='upper center', handlelength=1.0, columnspacing=0.9)
        a1 = cv.ax(16, 9, 92, 24)
        a1.set_label('A1')
        P.gbars(a1, self.en_groups(gA1), 'Content (μg/L)')
        a2 = cv.ax(16, 44, 92, 22)
        a2.set_label('A2')
        P.gbars(a2, self.en_groups(gA2), 'Content (μg/L)')
        for a in (a1, a2):
            a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f'{v:g}'))
        ax = cv.panel(G.Cell(114, 1, 66, 73), 'B', L=14, B=10, R=3, T=6)
        ax.set_label('B')
        P.pca_scatter(ax, self.d['pca_heat'], HEAT_SOLID, HEATM, HEATL, 'PC1 (35.4%)', 'PC2 (31.0%)')
        leg(ax, loc='upper right', ncol=1, handletextpad=0.2, labelspacing=0.2)
        cv.letter(0, 79, 'C')
        axV = cv.ax(5, 92, 46, 44)
        axV.set_label('C')
        mf.venn4(axV, self.d['venn_heat'], HEAT, HEAT_SOLID, HEATL)
        for t in axV.texts:
            t.set_fontsize(FS['small'] if t.get_text() in HEATL.values() else FS['annot'])
        codes = list(self.amap['heat'].keys())
        sub = v2[v2['编号'].isin(codes)].reset_index(drop=True)
        M = sub[[k + '_OAV' for k in HEAT]].values.astype(float)
        with np.errstate(divide='ignore'):
            Lg = np.log10(M)
        ann = [[(fmt(v, 1) if v >= 1 else '') if not np.isnan(v) else '' for v in row] for row in M]
        cv.letter(52, 79, 'D')
        rows = [SHORT.get(self.EN[n], self.EN[n]) for n in sub['化合物']]
        im, axh = P.cluster_heatmap(cv, 54, 84, 76, 66, Lg, rows, {i: HEATS[k] for i, k in enumerate(HEAT)}, [self.amap['heat'][c] for c in sub['编号']], mf.SER_COL, REDS, mcolors.Normalize(-1, 1.6), annot=ann)
        axh.set_label('D')
        cv.hcbar(58, 160, 22, im, label='lg OAV (grey: not detected)', ticks=[-1, 0, 1])
        hs_ = [mpatches.Patch(fc=mf.SER_COL[s], ec='black', lw=0.4, label=SER[s]) for s in mf.SERIES]
        cv.legend(hs_, 108, 158, ncol=2, anchor='upper center', handlelength=1.0, columnspacing=0.8, fontsize=FS['small'], labelspacing=0.25)
        ser = pd.DataFrame(index=mf.SERIES)
        for k in HEAT:
            ser[k] = [v2[v2['编号'].isin([c for c, s in self.amap['heat'].items() if s == S])][k + '_OAV'].pipe(lambda x: x[x >= 0.1].sum()) for S in mf.SERIES]
        ser.index = [SER2[s] for s in ser.index]
        cv.letter(132, 79, 'E')
        axE = cv.ax(137, 94, 34, 34, polar=True)
        axE.set_label('E')
        P.radar(axE, ser, HEAT, HEAT_SOLID, HEATM, HEATL, [0.1, 1, 10, 100], log=True, lab_off=0.26)
        cv.legend([mk(HEATM[k], HEAT_SOLID[k], HEATL[k], ls='-', lw=1.1, ms=3.5, mec='white') for k in HEAT], 156, 140.5, ncol=2, anchor='upper center', columnspacing=0.8)
        r3 = cells(0, 170, [40, 35, 35, 35, 35], [56])
        self.key_bars(cv, r3[0], HEAT_CODES, self.ka['heat'].drop_duplicates('编号'), HEAT, HEATS, HEAT_SOLID, 'F')
        self.out(cv, 'Fig4')
        return (ct, ser)

    def fig5(self):
        t = self.d['tab_mode'].set_index('指标')
        order = ['全浆发酵', '浸渍后发酵', '清汁发酵']
        specs = [('非糖固形物（g/L）', 'Non-sugar solids (g/L)', 13), ('氨基酸态氮（g/L）', 'Amino acid nitrogen (g/L)', 0.3), ('总酚（mg GAE/L）', 'Total phenolics\n(mg GAE/L)', 150), ('总黄酮（mg RE/L）', 'Total flavonoids\n(mg RE/L)', 85), ('DPPH 自由基清除能力（μg VCE/mL）', 'DPPH scavenging\n(μg VCE/mL)', 55), ('总还原能力（μg VCE/mL）', 'Reducing power\n(μg VCE/mL)', 95)]
        cv = Canvas(160, 102)
        cc = cells(0, 1, [53.3, 53.3, 53.4], [50, 50])
        for i, ((row, yl, ymax), s) in enumerate(zip(specs, 'ABCDEF')):
            ax = cv.panel(cc[i // 3][i % 3], s, L=15, B=8, R=2, T=5)
            ax.set_label(s)
            vals = [mf.parse_ms(t.loc[row, o]) for o in order]
            x = np.arange(3)
            ax.bar(x, [v[0] for v in vals], width=0.62, color=[MODE_SOLID[o] for o in order], edgecolor='black', linewidth=0.5, zorder=2)
            ax.errorbar(x, [v[0] for v in vals], yerr=[v[1] for v in vals], fmt='none', ecolor='black', elinewidth=0.6, capsize=2.0, capthick=0.6, zorder=3)
            for xi, (m, sd, l) in zip(x, vals):
                ax.text(xi, m + sd + ymax * 0.015, l, ha='center', va='bottom', fontsize=FS['annot'], fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([MODES[o] for o in order], fontweight='bold', fontsize=FS['small'])
            ax.set_ylim(0, ymax)
            ax.set_ylabel(yl)
            ax.yaxis.label.set_linespacing(1.1)
        self.out(cv, 'Fig5')

    def radar_q(self, ax, df, cols, colors, markers, labels, ticks, lab_off=0.24):
        ticks = [0, 4, 8, 12, 16, 20] if ticks[-1] == 20 else [0, 2, 4, 6, 8, 10]
        if np.nanmin(df.values.astype(float)) < 0 or np.nanmax(df.values.astype(float)) > ticks[-1]:
            raise ValueError('Sensory values lie outside the documented scoring scale')
        attrs = list(df.index)
        n = len(attrs)
        rmin, rmax = (ticks[0], ticks[-1])
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        aa = np.concatenate([ang, ang[:1]])
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(rmin, rmax)
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        for t in ticks:
            ax.plot(aa, [t] * (n + 1), color='#BDBDBD', lw=0.5, zorder=0)
            if rmin < t < rmax:
                ax.text(np.pi / n, t, f'{t:g}', fontsize=FS['small'] - 0.5, color='#555555', ha='center', va='center', bbox=dict(fc='white', ec='none', pad=0.2))
        for a in ang:
            ax.plot([a, a], [rmin, rmax], color='#BDBDBD', lw=0.5, zorder=0)
        for c in cols:
            v = df[c].values.astype(float)
            vv = np.concatenate([v, v[:1]])
            ax.plot(aa, vv, color=colors[c], lw=1.1, marker=markers[c], ms=3, mec='white', mew=0.4, label=labels[c], zorder=3)
            ax.fill(aa, vv, color=colors[c], alpha=0.08, zorder=2)
        for a, lab in zip(ang, attrs):
            ax.text(a, rmax + (rmax - rmin) * lab_off, lab, ha='center', va='center', fontsize=FS['small'], fontweight='bold', linespacing=0.95)

    def fig6(self):
        cv = Canvas(160, 80)
        cc = cells(0, 1, [53.3, 53.3, 53.4], [78])
        qv = self.d['qda_var'].copy()
        qv.index = [SENS_VAR[i] for i in qv.index]
        two = lambda s: s.replace('Aroma ', 'Aroma\n').replace('Overall ', 'Overall\n').replace('Palate ', 'Palate\n')
        qh = self.d['qda_heat'].copy()
        qh.index = [two(QDA[i]) for i in qh.index]
        qm = self.d['qda_mode'].copy()
        qm.index = [two(QDA[i]) for i in qm.index]
        modes = list(MODE_SOLID)
        specs = [(qv, VAR, VAR_SOLID, VARM, VARL, [8, 12, 16, 20], 1), (qh, HEAT, HEAT_SOLID, HEATM, {k: f'{HEATL[k]} {s}' for k, s in {'WB': '1.5 min', 'HK': '50 min', 'ZZ': '50 min', 'SZ': '50 min'}.items()}, [2, 4, 6, 8, 10], 2), (qm, modes, MODE_SOLID, MODEM, MODEL, [2, 4, 6, 8, 10], 1)]
        for i, ((df, cols, colors, marks, labs, ticks, nc), s) in enumerate(zip(specs, 'ABC')):
            c = cc[0][i]
            cv.letter(c.x, c.y, s)
            ax = cv.ax(c.x + 12.5, c.y + 12, 29, 29, polar=True)
            ax.set_label(s)
            self.radar_q(ax, df, cols, colors, marks, labs, ticks, lab_off=0.27)
            cv.legend([mk(marks[k], colors[k], labs[k], ls='-', lw=1.1, ms=3.5, mec='white') for k in cols], c.x + c.w / 2, c.y + 57, ncol=nc, anchor='upper center', columnspacing=0.8, fontsize=FS['small'])
        self.out(cv, 'Fig6')

    def fig7(self):
        z = self.d['z_heat'].copy()
        cv = Canvas(176, 118)
        cv.letter(0, 1, 'A')
        axA = cv.ax(26, 14, 40, 72)
        axA.set_label('A')
        cols = ['Z_酒精度', 'Z_总酚', 'Z_DPPH', 'Z_总还原能力', 'Z_感官']
        im = axA.imshow(z[cols].values, cmap=RDBU, vmin=-2.2, vmax=2.2, aspect='auto')
        axA.set_xticks(range(5))
        axA.set_xticklabels(['Alcohol', 'Total phenolics', 'DPPH', 'Reducing power', 'Sensory score'], fontsize=FS['small'], fontweight='bold', rotation=45, ha='right', rotation_mode='anchor')
        axA.set_yticks(range(len(z)))
        axA.set_yticklabels([f"{HEATS[s.split('-')[0]]} {s.split('-')[1]} min" for s in z['样品']], fontsize=FS['small'])
        axA.tick_params(length=0)
        for j, w_ in enumerate(self.d['z_heat_w']['权重']):
            axA.text(j, -0.8, f'{w_:.2f}', ha='center', va='bottom', fontsize=FS['small'], fontweight='bold')
        axA.text(-0.75, -0.8, 'Weight', ha='right', va='bottom', fontsize=FS['small'], fontweight='bold')
        for i in (4.5, 9.5, 14.5):
            axA.axhline(i, color='black', lw=0.8)
        axL = cv.ax(68, 14, 24, 72)
        axL.set_label('A2')
        yv = np.arange(len(z))
        cs = [HEAT_SOLID[s.split('-')[0]] for s in z['样品']]
        axL.hlines(yv, 0, z['综合评分'], color='#9E9E9E', lw=1.0)
        axL.scatter(z['综合评分'], yv, c=cs, s=16, edgecolors='black', linewidths=0.4, zorder=3)
        best = set(z.loc[z.groupby(z['样品'].str[:2])['综合评分'].idxmax(), '样品'])
        for yi, (_, r) in zip(yv, z.iterrows()):
            if r['样品'] in best:
                v = r['综合评分']
                axL.text(v + (0.12 if v >= 0 else -0.12), yi, fmt(v).replace('-', '−'), va='center', ha='left' if v >= 0 else 'right', fontsize=FS['small'], fontweight='bold')
        axL.axvline(0, color='black', lw=0.6)
        axL.set_ylim(len(z) - 0.5, -0.5)
        axL.set_yticks([])
        axL.set_xlim(-2.1, 1.6)
        axL.set_xlabel('Composite score')
        for i in (4.5, 9.5, 14.5):
            axL.axhline(i, color='black', lw=0.6, ls=(0, (4, 3)))
        cv.hcbar(32, 104.5, 30, im, label='Z-score', ticks=[-2, 0, 2])
        te = mf.tier_effects(self.d)
        Mx = te.values.astype(float)
        cv.letter(98, 1, 'B')
        axB = cv.ax(136, 14, 24, 72)
        axB.set_label('B')
        lnorm = mcolors.LogNorm(1, 3.6)
        imB = axB.imshow(Mx, cmap=REDS, norm=lnorm, aspect='auto')
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                if np.isnan(Mx[i, j]):
                    axB.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fc='#EEEEEE', ec='white'))
                    axB.text(j, i, '—', ha='center', va='center', fontsize=FS['small'], color='#777777')
                else:
                    axB.text(j, i, fmt(Mx[i, j]), ha='center', va='center', fontsize=FS['small'], fontweight='bold', color='white' if Mx[i, j] > 2.2 else 'black')
        axB.set_xticks(range(3))
        axB.set_xticklabels([TE_COL[c].replace('\n', ' ') for c in te.columns], fontsize=FS['small'], fontweight='bold', rotation=45, ha='right', rotation_mode='anchor')
        axB.set_yticks(range(len(te)))
        axB.set_yticklabels([TE_ROW[r] for r in te.index], fontsize=FS['small'])
        axB.tick_params(length=0)
        for i in range(len(te) + 1):
            axB.axhline(i - 0.5, color='white', lw=1)
        for j in range(4):
            axB.axvline(j - 0.5, color='white', lw=1)
        sm = plt.cm.ScalarMappable(cmap=REDS, norm=lnorm)
        cb = cv.cbar(axB, sm, 'R$_{max/min}$', ticks=[1, 1.5, 2, 3], ticklabels=['1', '1.5', '2', '3'], frac=0.8)
        self.out(cv, 'Fig7')
        return te

    def figS2(self):
        cv = Canvas(132, 120)
        ax = cv.ax(26, 24, 88, 88)
        ax.set_label('corr')
        names = ['Alcohol', 'Reducing sugar', 'Total acid', 'pH', 'Total phenolics', 'Total flavonoids', 'Soluble protein', 'DPPH', 'Reducing power', 'Sensory score']
        Rm, Pm = (self.d['corr_r'].values, self.d['corr_p'].values)
        n = len(names)
        for i in range(n):
            for j in range(n):
                ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fc='white', ec='#BDBDBD', lw=0.4))
                if j > i:
                    r = Rm[i, j]
                    s = np.sqrt(abs(r)) * 0.92
                    ax.add_patch(mpatches.Rectangle((j - s / 2, i - s / 2), s, s, fc=RDBU((r + 1) / 2), ec='none'))
                    star = '***' if Pm[i, j] < 0.001 else '**' if Pm[i, j] < 0.01 else '*' if Pm[i, j] < 0.05 else ''
                    if star:
                        ax.text(j, i, star, ha='center', va='center', fontsize=FS['small'], fontweight='bold', color='white' if abs(r) > 0.7 else 'black')
                elif j < i:
                    ax.text(j, i, f'{Rm[i, j]:.2f}'.replace('-', '−'), ha='center', va='center', fontsize=FS['small'] - 0.3, color='#B2182B' if Rm[i, j] >= 0.5 else '#2166AC' if Rm[i, j] <= -0.5 else '#333333', fontweight='bold')
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=90, fontsize=FS['small'], fontweight='bold')
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=FS['small'], fontweight='bold')
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        sm = plt.cm.ScalarMappable(cmap=RDBU, norm=plt.Normalize(-1, 1))
        cv.cbar(ax, sm, 'Pearson r', frac=0.8)
        self.out(cv, 'FigS2')

    def figS3(self):
        cv = Canvas(176, 196)
        cc = cells(0, 1, [88, 88], [39] * 5)
        for i, ((ind, yl, ylim, src), s) in enumerate(zip(SPEC9, 'ABCDEFGHI')):
            ax = cv.panel(cc[i // 2][i % 2], s, L=15, B=7, R=2, T=5)
            ax.set_label(s)
            groups = [dict(label=HEATL[k], means=m, sds=sd, letters=L, colors=ramp(*HEAT_BASE[k], 5), sub=None) for k, _, _, m, sd, L in mf.heat_groups(self.d, ind, src)]
            P.gbars(ax, groups, yl, ylim=ylim, glab_pts=2.5)
            ax.yaxis.label.set_linespacing(1.1)
        c = cc[4][1]
        ax = cv.ax(c.x + 4, c.y + 3, c.w - 6, c.h - 5)
        ax.set_xlim(0, 74)
        ax.set_ylim(0, 34)
        ax.axis('off')
        ax.set_label('key')
        for gi, k in enumerate(HEAT):
            cols = ramp(*HEAT_BASE[k], 5)
            for j in range(5):
                ax.add_patch(mpatches.Rectangle((2 + gi * 18 + j * 2.6, 22), 2.2, 8, fc=cols[j], ec='black', lw=0.4))
            ax.text(2 + gi * 18 + 6.3, 19.5, HEATL[k], ha='center', va='top', fontsize=FS['small'], fontweight='bold')
        ax.text(0, 11.5, 'Bars from dark to light: treatment time from short to long\n(microwave 1.0–3.0 min; others 30–70 min)', ha='left', va='center', fontsize=FS['small'], linespacing=1.3)
        ax.text(0, 2.5, "Letters: Duncan's test among the five times of one method", ha='left', va='center', fontsize=FS['small'])
        self.out(cv, 'FigS3')

    def beeswarm(self, ax, S, Xv, order, title=None):
        for row, j in enumerate(order):
            sv = S[:, j]
            xv = Xv[:, j]
            c = (xv - xv.min()) / (xv.max() - xv.min() + 1e-12)
            jit = np.zeros(len(sv))
            bins = np.round(sv / (np.ptp(S[:, order]) / 40 + 1e-12))
            for b in np.unique(bins):
                ids = np.where(bins == b)[0]
                ids = ids[np.argsort(xv[ids])]
                step = min(0.075, 0.7 / max(len(ids) - 1, 1))
                jit[ids] = (np.arange(len(ids)) - (len(ids) - 1) / 2) * step
            ax.scatter(sv, row + jit, c=P.SHAP_CMAP(c), s=10, edgecolors='black', linewidths=0.25, zorder=3)
        ax.axvline(0, color='#9E9E9E', lw=0.7, zorder=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([FL[self.FEAT[j]] for j in order], fontsize=FS['small'])
        ax.set_ylim(-0.7, len(order) - 0.3)
        if title:
            ax.set_title(title, fontsize=FS['small'], fontweight='bold', pad=2)

    def fig3(self):
        Xv = self.W[self.FEAT].values.astype(float)
        mb = self.ml['xgb_med']
        mm_ = self.ml['xgb_met']
        cv = Canvas(160, 128)
        r1 = cells(0, 1, [80, 80], [62])
        r2 = cells(0, 64, [80, 80], [63])
        ax = cv.panel(r1[0][0], 'A', L=12, B=8, R=2, T=12)
        ax.set_label('A')
        names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        nl = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        x = np.arange(5)
        wb = 0.36
        b1 = [mb['metrics'][n] for n in names]
        b2 = [mm_['metrics'][n] for n in names]
        ax.bar(x - wb / 2, b1, width=wb, color=DRY_C, edgecolor='black', linewidth=0.5, label=f"Dry/moist heat (permutation P = {mb['p_acc']:.3f})", zorder=2)
        ax.bar(x + wb / 2, b2, width=wb, color=WET_C, edgecolor='black', linewidth=0.5, label=f"Four methods (permutation P = {mm_['p_acc']:.3f})", zorder=2)
        for xi, v1, v2 in zip(x, b1, b2):
            ax.text(xi - wb / 2, v1 + 0.015, fmt(v1), ha='center', va='bottom', fontsize=FS['small'] - 0.5)
            ax.text(xi + wb / 2, v2 + 0.015, fmt(v2), ha='center', va='bottom', fontsize=FS['small'] - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(nl, fontsize=FS['small'], fontweight='bold')
        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Nested LOOCV performance')
        leg(ax, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=1, frameon=False, fontsize=FS['small'], labelspacing=0.2)
        S = mb['shap'][:, :-1]
        ms = np.abs(S).mean(0)
        order = np.argsort(ms)
        ax = cv.panel(r1[0][1], 'B', L=22, B=10, R=3, T=12)
        ax.set_label('B')
        ax.barh(range(len(self.FEAT)), ms[order], color=DRY_C, edgecolor='black', linewidth=0.4, height=0.62, zorder=2)
        for yi, v in enumerate(ms[order]):
            ax.text(v + 0.02, yi, fmt(v), va='center', fontsize=FS['small'])
        ax.set_yticks(range(len(self.FEAT)))
        ax.set_yticklabels([FL[self.FEAT[i]] for i in order], fontsize=FS['small'])
        ax.set_xlabel('mean |SHAP| (dry/moist-heat model)')
        ax.set_xlim(0, ms.max() * 1.25)
        ax = cv.panel(r2[0][0], 'C', L=22, B=10, R=12, T=6)
        ax.set_label('C')
        used = [j for j in np.argsort(-ms) if ms[j] > 0][::-1]
        self.beeswarm(ax, S, Xv, used)
        ax.set_xlabel('SHAP value (> 0: towards dry heat)')
        sm = plt.cm.ScalarMappable(cmap=P.SHAP_CMAP, norm=plt.Normalize(0, 1))
        cv.cbar(ax, sm, 'Feature value', ticks=[0, 1], ticklabels=['Low', 'High'], frac=0.8)
        Sm = mm_['shap'][:, :, :-1]
        Mh = np.zeros((len(self.FEAT), 4))
        Dir = np.zeros((len(self.FEAT), 4))
        for ci in range(4):
            Mh[:, ci] = np.abs(Sm[:, ci, :]).mean(0)
            for j in range(len(self.FEAT)):
                if np.std(Sm[:, ci, j]) > 1e-09:
                    Dir[j, ci] = np.corrcoef(Xv[:, j], Sm[:, ci, j])[0, 1]
        rowo = np.argsort(-Mh.sum(1))
        ax = cv.panel(r2[0][1], 'D', L=22, B=10, R=14, T=8)
        ax.set_label('D')
        im = ax.imshow(Mh[rowo], cmap=REDS, aspect='auto', vmin=0, vmax=Mh.max())
        for i, j_ in enumerate(rowo):
            for ci in range(4):
                v = Mh[j_, ci]
                if v > 0.05:
                    ax.text(ci, i, f"{fmt(v)}{('↑' if Dir[j_, ci] > 0 else '↓')}", ha='center', va='center', fontsize=FS['small'], fontweight='bold', color='white' if v > Mh.max() * 0.55 else 'black')
        ax.set_xticks(range(4))
        ax.set_xticklabels([HEATL[m] for m in self.METHODS], fontsize=FS['small'], fontweight='bold')
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(self.FEAT)))
        ax.set_yticklabels([FL[self.FEAT[j]] for j in rowo], fontsize=FS['small'])
        ax.tick_params(length=0)
        for k in range(len(self.FEAT) + 1):
            ax.axhline(k - 0.5, color='white', lw=0.8)
        for k in range(5):
            ax.axvline(k - 0.5, color='white', lw=0.8)
        cv.cbar(ax, im, 'mean |SHAP| (four-class model)')
        self.out(cv, 'Fig3')

    def figS4(self):
        Xv = self.W[self.FEAT].values.astype(float)
        Sm = self.ml['xgb_met']['shap'][:, :, :-1]
        cv = Canvas(160, 112)
        cc = cells(0, 1, [80, 80], [55, 55])
        for ci, (m, s) in enumerate(zip(self.METHODS, 'ABCD')):
            ax = cv.panel(cc[ci // 2][ci % 2], s, L=22, B=10, R=3 if ci % 2 == 0 else 16, T=7)
            ax.set_label(s)
            S_ = Sm[:, ci, :]
            ms_ = np.abs(S_).mean(0)
            used = [j for j in np.argsort(-ms_) if ms_[j] > 0.01][:6][::-1]
            self.beeswarm(ax, S_, Xv, used, title=f'{HEATL[m]} (> 0: towards {HEATL[m].lower()})')
            ax.set_xlabel('SHAP value (raw class score)')
            if ci == 3:
                sm = plt.cm.ScalarMappable(cmap=P.SHAP_CMAP, norm=plt.Normalize(0, 1))
                cv.cbar(ax, sm, 'Feature value', ticks=[0, 1], ticklabels=['Low', 'High'], frac=0.8)
        self.out(cv, 'FigS4')

def generate_figures(output_dir, names=None, dpi=300):
    """Generate selected final figures and return PNG/PDF paths.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Destination directory, created only when a figure is written.
    names : iterable of str or str, optional
        Final manuscript names, e.g. ``['Fig1', 'FigS4']``. ``None`` selects
        all seven main and four supplementary figures. Short names such as
        ``'1'`` and ``'S4'`` are also accepted.
    dpi : int
        PNG resolution (default 300). PDFs retain vector content. The supplied
        FigS1 is copied unchanged, so its original resolution is preserved.
    """
    selected = FIGURE_NAMES if names is None else ([names] if isinstance(names, str) else list(names))
    normalized = []
    for name in selected:
        value = str(name)
        if not value.startswith('Fig'):
            value = 'Fig' + value
        if value not in FIGURE_NAMES:
            raise ValueError(f'Unknown figure {name!r}; choose from {", ".join(FIGURE_NAMES)}')
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        return []
    settings = dict(STYLE, **GRID_STYLE)
    try:
        font_manager.findfont('Arial', fallback_to_default=False)
    except ValueError:
        settings['font.family'] = ['DejaVu Sans']
    else:
        settings['font.family'] = ['Arial', 'DejaVu Sans']
    builder = FigureBuilder(output_dir, dpi=dpi)
    with matplotlib.rc_context(settings):
        for name in normalized:
            getattr(builder, name[0].lower() + name[1:])()
    return builder.files
