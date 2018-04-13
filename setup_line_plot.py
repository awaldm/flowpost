#!/usr/bin/python
# -*- coding: utf-8 -*-


from matplotlib import rcParams
import matplotlib.pyplot as plt

import itertools

# allgemein fürs ganze skript:
# plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) +
#                           cycler('linestyle', ['-', '--', ':', '-.'])))
# nur für ein axes object
# ax1.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']) +
#                   cycler('lw', [1, 2, 3, 4]))

# choose plot type (only JOURNAL_1COL and SCREEN implemented so far
def setup_plot(target, width, height):
    inches_per_pt = 1.0/72.27
    line_style=itertools.cycle(["-", "--", ":"])
    #markers = mpl.markers.MarkerStyle.filled_markers
#    marker_style = itertools.cycle(('v', 'o', 's', 'd', '*', '^', 'p', '+'))
    _markers = ['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h']
    marker_style = itertools.cycle(_markers)

    if target=='SCREEN':
        axesFontSize = 14
        titleFontSize = 14
        xlabelpad = -2
        ylabelpad = -4

#        width = 6.5
#        height = 6 * (6.5 / 8)
        params = {
            'figure.figsize': (width, height),
            'axes.labelsize': 18,
            'font.size': 12,
            'font.family': 'CMU Serif',
            'font.weight': 'normal',
            'font.style': 'normal',
            'legend.fontsize': 16,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'lines.linewidth': 2,
            'lines.markersize': 10,
            'text.usetex': True,
            #   'figure.figsize': [7, 4] # instead of 4.5, 4.5
        }

    elif target=='JOURNAL_1COL':
#        width = 3.25*2/3 #in
#        height = 6 * (width / 8)
        axesFontSize = 14
        titleFontSize = 14
        xlabelpad = -3
        ylabelpad = -6
        params = {
            'figure.figsize': (width, height),
            'axes.labelsize': 8,
            'font.size': 8,
            'font.family': 'Times New Roman',
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'lines.linewidth': 1,
            'lines.markersize': 8,
            'xtick.major.size': 3,
            'xtick.minor.size': 3,
            'xtick.major.width': 1,
            'xtick.minor.width': 1,
            'ytick.major.size': 3,
            'ytick.minor.size': 3,
            'ytick.major.width': 1,
            'ytick.minor.width': 1,
            'legend.frameon': True,
            'legend.loc': 'center left',
            'text.usetex': True,
            'figure.subplot.right': 0.95,
            'figure.subplot.top': 0.95,
            'figure.subplot.bottom': 0.1

            #   'figure.figsize': [7, 4] # instead of 4.5, 4.5
        }

    rcParams.update(params)
    return line_style, marker_style

# from
# http://bikulov.org/blog/2013/10/03/creation-of-paper-ready-plots-with-matlotlib/


#    plt.gca().spines['right'].set_color('none')
#    plt.gca().spines['top'].set_color('none')
#    plt.gca().xaxis.set_ticks_position('bottom')
#    plt.gca().yaxis.set_ticks_position('left')

def get_brewer_colors():
    import brewer2mpl
    bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
    colors = bmap.mpl_colors
    return colors



