# /usr/bin/python
import numpy as np
import re
import pandas as pd
import pyTecIO_AW.tec_modules as tec
import sys
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import rc
import matplotlib as mpl
import wake_functions as wf
import pyTecIO_AW.parse_tecplot_helpers as parser
from mpl_toolkits import axes_grid1
#import pyTecIO_AW.read_2d_wake_timeseries as reader

from utils import *

# for clipping the wing outline
from matplotlib.path import Path
from matplotlib.patches import PathPatch

width = 3.5

try:
    from setup_plot import * # custom plot setup file
    line_style, markers = setup_plot('JOURNAL_1COL', width, width*0.8)

except ImportError as e:
    print 'setup_plot not found, setting default line styles...'
    line_style=itertools.cycle(["-", "--", ":"])
    marker_style = itertools.cycle(['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h'])

print 'matplotlib version: ' + mpl.__version__

def plot_surface(x, y, data, kind='OAT_cpmean', shape=None, struct=False):
    """
    create colored contour plot of the cp distribution on a surface
    capable of struct and unstruct data
    """
    # setup the color range
    if kind=='OAT_cpmean':
        cmap = plt.cm.seismic
        minval = np.min(data) # -1.5
        maxval = np.max(data) # 1
        minval = -1.5
        maxval = 0.8
        color_delta = 0.005
    elif kind == 'OAT_cprms':
        cmap = plt.cm.hot_r
        minval = 0
        maxval = 0.4
        color_delta = 0.001
    else:
        sys.exit('unknown type')

    vals = np.arange(minval, maxval+color_delta, color_delta)

    norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)

    #triang = tri.Triangulation(x, y)
    fig, ax = plt.subplots(1,1)
    if struct is False:
        if shape is not None:
            x = x.reshape(shape[0]*shape[1])
            y = y.reshape(shape[0]*shape[1])
            data = data.reshape(shape[0]*shape[1])
        contour = plt.tricontourf(x, y, data, vals, norm=norm, cmap=cmap)
    else:
        contour = plt.contourf(x, y, data, vals, norm=norm, cmap=cmap)
    # clip the wing outside of a manually-defined boundary
    # https://stackoverflow.com/questions/44082717/limit-mask-matplotlib-contour-to-data-area
    # https://stackoverflow.com/questions/42426095/matplotlib-contour-contourf-of-concave-non-gridded-data
    # http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/

    # the clipping path coordinates need to be swapped, as we are drawing in (y,-x) plane
    clip_x = [0.68088,1.2213,1.29472,1.03667,1.00037,0.68088]
    clip_x = [-1 * i for i in clip_x]
    clip_y = [0.08138, 0.7933, 0.7933, 0.29352, 0.08138,0.08138]


    clippath = Path(np.c_[clip_y, clip_x])
    patch = PathPatch(clippath, facecolor='none', edgecolor='none') # make the outer boundary invisible
    ax.add_patch(patch)
    for c in contour.collections:
        c.set_clip_path(patch)
    #ax.scatter(clip_x, clip_y, color='k') # The original data points
    #ax.plot(clip_x, clip_y, color="crimson")


    #line_vals = np.arange(minval, maxval+color_delta, 0.1)
    plt.gca().set_aspect('equal')
    #lines = plt.tricontour(x, y, data, levels=line_vals, linewidths=0.5, colors=[(0,0,0,0.5)])
    # setup colorbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)
    if kind=='OAT_cpmean':
        cbar = fig.colorbar(contour, ticks=[minval, 0, maxval], cax=cax1)
        cbar.ax.set_yticklabels([str(minval), '0', str(maxval)])  # vertically oriented colorbar
    elif kind == 'OAT_cprms':
        cbar = fig.colorbar(contour, ticks=[0, maxval], cax=cax1)
        cbar.ax.set_yticklabels([str(minval), str(maxval)])  # vertically oriented colorbar

    return fig, ax

##########################################################
# read surface time series from plt files
# create contour plots and cp distributions
# problem with PLTs: if anything changes during the computation we get varying zone lists, we
# are not capable of handling this yet
##########################################################
if __name__ == "__main__":
    
    rows=65
    cols=259
    in_plt = '/home/andreas/hazelhen_WS1/M085/RANS/v51_new/SSG/sol/CRM_v51_WT_defa45_M085_SSG_a1.0.surface.pval.70000.plt'
    in_plt = 'CRM_v52_WT_SAO_a05.surface.pval.10000.plt'

    data,varpos = reader.get_plane_PLT(in_plt, zone_no=1, verbose=False, force_struct=False) # formerly get_data
    cp_var = varpos.index('cp')
    x_var = varpos.index('X')
    y_var = varpos.index('Y')
    print str(cp_var)
    x = data[0, x_var, :]
    y = data[0, y_var, :]
    cp = data[0, cp_var, :]

    print 'min cp: ' + str(np.min(cp))
    print 'max cp: ' + str(np.max(cp))


    fig,ax = plot_surface(y, -x, cp, kind='OAT_cpmean', shape=None, struct=False)
    plt.savefig('wing.png', dpi=600)

