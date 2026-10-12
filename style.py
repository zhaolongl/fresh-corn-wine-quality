"""Colours and typography shared by the publication figures."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

STYLE = {'font.family': ['DejaVu Sans'], 'font.size': 7.5, 'axes.labelsize': 8, 'axes.labelweight': 'bold', 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8, 'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.direction': 'out', 'ytick.direction': 'out', 'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.spines.top': True, 'axes.spines.right': True, 'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'axes.unicode_minus': False, 'mathtext.fontset': 'dejavusans'}
MM = 1 / 25.4
HEAT_BASE = {'WB': ('#2F6DB5', '#CFE2F5'), 'HK': ('#D9603F', '#FAD6C7'), 'ZZ': ('#2E9A6E', '#CDEBDD'), 'SZ': ('#6A57A8', '#DCD5EE')}
HEAT_SOLID = {'WB': '#2F6DB5', 'HK': '#D9603F', 'ZZ': '#2E9A6E', 'SZ': '#6A57A8'}
HEATM = {'WB': 'o', 'HK': 's', 'ZZ': '^', 'SZ': 'D'}
VAR_SOLID = {'HN': '#6A57A8', 'BT': '#3E9B5F', 'HT': '#E3A21A'}
VARM = {'HN': 'o', 'BT': 's', 'HT': '^'}
MODE_SOLID = {'全浆发酵': '#D9603F', '浸渍后发酵': '#2F6DB5', '清汁发酵': '#E3A21A'}
MODEM = {'全浆发酵': 'o', '浸渍后发酵': 's', '清汁发酵': '^'}
CLASS_COL = {'酯类': '#8C1C13', '醇类': '#C0392B', '酸类': '#E67E22', '萜烯类': '#F5B041', '醛酮类': '#F9D5A7', '烷烃类': '#A9CCE3', '其它类': '#5D6D7E'}
RDBU = LinearSegmentedColormap.from_list('rdbu', ['#2166AC', '#67A9CF', '#D1E5F0', '#F7F7F7', '#FDDBC7', '#EF8A62', '#B2182B'])
REDS = LinearSegmentedColormap.from_list('reds', ['#FFF5EB', '#FDD0A2', '#FD8D3C', '#D94801', '#7F2704'])

def ramp(c_dark, c_light, n):
    a = np.array(to_rgb(c_dark)); b = np.array(to_rgb(c_light))
    return [tuple(a + (b - a) * t) for t in np.linspace(0, 1, n)]
