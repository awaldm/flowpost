#/usr/bin/python
"""
Read a time series of structured wake snapshots and create a spacetime plot
"""

import os, sys
import time
import shutil
import numpy as np
import matplotlib.mlab as mlab
import matplotlib as mpl
import pyTecIO_AW.tec_modules as tec
import matplotlib.pyplot as plt
import helpers.wake_stats as wstat
import pandas as pd
# plot size
width = float(7)/float(2)
try:
    from setup_line_plot import * # custom plot setup file
    line_style, markers = setup_plot('JOURNAL_1COL', width, 0.9*width)
except ImportError as e:
    print 'setup_plot not found, setting default line styles...'
    line_style=itertools.cycle(["-", "--", ":"])
    marker_style = itertools.cycle(['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h'])



######################################################################
if __name__ == "__main__":
    #data_type = 'CRM_plane'
    data_type = 'OAT15A'
    #plane = 'eta0201'
    #plane = 'wake_x3'
    plane = 'symm'

    zonelist = [0] # 12
    planelist = ['symm']

    plt_path = '/home/andreas/laki_ehrle/AW_tests/AZDES-SSG/AZDES-SSG_dt2e5_turbRoe2nd_SGDH_Lc00161/profile/'
    case_name = 'OAT15A_AZDES-SSG'
    zone_no = 0
    start_i = 15490
    end_i = 17450

    displaynames = {}

    casenames = ['URANS', 'AZDES', 'AZDESlowdiss']
    displaynames['URANS'] = r'URANS'
    displaynames['AZDES'] = r'AZDES $k^{(4)} = 1/64$'
    displaynames['AZDESlowdiss'] = r'AZDES $k^{(4)}=1/512$'

    filename = 'OAT15A_AZDES_SSG_buffet_points_profile_'

    pointdata = []
    varpos = []


    for point_num in range(5):
        dataset = {}
        print os.path.join(plt_path, filename + str(point_num) + '.dat')
        zones, dfs, els = tec.readnew(os.path.join(plt_path, filename + str(point_num) + '.dat'), verbose=0)
        varpos.append(list(dfs[0]))
            # dataset is a dict with the keys casenames, each value being a list of pandas dataframes
            # in this particular case each list has only one element
        pointdata.append(pd.to_numeric(dfs[0]['"cp"']))

    t= pd.to_numeric(dfs[0]['"time"'])
    print t


    # dataset  is
    print type(pointdata)
    print len(pointdata[4])

    img_folder = './images/'
    lines = np.zeros([len(pointdata[0]),5])
    for point in range(len(pointdata)):
        lines[:,point] = pointdata[point]
    saveFile = 'pointresults'
    np.savez(saveFile, lines=lines)


