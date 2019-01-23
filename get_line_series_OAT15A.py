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
import pyTecIO_AW.tecreader as tecreader
import matplotlib.pyplot as plt
import helpers.wake_stats as wstat
import scipy.interpolate

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

    plt_path = '/home/andreas/laki_ehrle/AW_tests/AZDES-SSG/AZDES-SSG_dt2e5_turbRoe2nd_SGDH_Lc00161/sol/surface/'
    case_name = 'OAT15A_AZDES-SSG'
    zone_no = 0
    start_i = 15490
    end_i = 17450

    # URANS SSG output period 4
    plt_path = '/home/andreas/laki_ehrle/AW_tests/URANS-SSG/URANS-SSG_dt2e5_2016.2_turbRoe2nd_SGDH/sol/surface/'
    case_name = 'OAT15A_URANS-SSG_dt4'
    zone_no = 0
    start_i = 13300
    end_i = 14800
    di = 4

    plt_path = '/home/andreas/laki_WS_OAT/URANS-JHhV2/URANS-JHh-v2_dt2e5_2016.2_turbRoe_CFL1_2v_SSK/sol/surface/plt/'
    case_name = 'OAT15A_URANS-JHh_turbRoe'
    zone_no = 0
    start_i = 3910
    end_i = 8100
    di = 10
    

    plt_path = '/home/andreas/NAS_sebi/OAT15A/AZDES-final-mesh/JHhv2-matrix-no-hyperflex/sol/field/plt/'
    case_name = 'OAT15A_URANS_JHh_illi'
    zone_no = 2
    start_i = 10100
    end_i = 19990
    di = 10

    plt_path = '/home/andreas/NAS_sebi/OAT15A/AZDES-final-mesh/JHhv2-matrix/sol/field/plt/'
    case_name = 'OAT15A_URANS_JHh_illi_hyperflex'
    zone_no = 4 # 2: symm, 4: airfoil_ss
    start_i = 10100
    end_i = 19630
    di = 10

    plt_path = '/home/andreas/laki_WS_OAT/URANS-JHhV2/URANS-JHh-v2_dt1e5_2016.2_turbRoe_CFL1_2v_SSK/sol/surface/plt/'
    case_name = 'OAT_URANS-JHh-v2_dt1e5_2016.2_turbRoe_CFL1_2v_SSK'
    zone_no = 0
    start_i = 2700
    end_i = 5570
    di = 10

    plt_path = '/home/andreas/laki_WS_OAT/URANS-JHhV2/URANS-JHh-v2_dt1e5_2016.2_turbRoe_CFL1_2v_SSK/sol/surface/plt/'
    case_name = 'OAT_URANS-JHh-v2_dt1e5_2016.2_turbRoe_CFL1_2v_SSK_dt1'
    zone_no = 0
    start_i = 6601
    end_i = 8031
    di = 1




#    for zone in range(len(zonelist)):
    varnames = ['cp']
    #varnames = ['pressure']
    filelist, num_files = tecreader.get_sorted_filelist(plt_path, 'plt', stride=di)
    filelist = tecreader.get_cleaned_filelist(filelist,start_i=start_i, end_i=end_i)
    num_files = len(filelist)

    if isinstance(zone_no,(int, long)):
        zone_no = [zone_no]
    else:
        zone_no = zone_no

    in_data = tecreader.read_series_parallel([plt_path + s for s in filelist], zone_no, varnames, 3)
    print('data shape: ' + str(in_data.shape))

    import tecplot as tp
    dataset = tp.data.load_tecplot(plt_path + filelist[0], zones=zone_no, variables=['X', 'Y', 'Z'], read_data_option = tp.constant.ReadDataOption.Replace)

    for zone in dataset.zones():
        array = zone.values('X')
        x = np.array(array[:]).T
        array = zone.values('Y')
        y = np.array(array[:]).T
        array = zone.values('Z')
        z = np.array(array[:]).T


    num_points = 100
    cref = 0.23
    #x0, z0 = 0.7*cref, 0.1*cref
    #x1, z1 = 0.9*cref, 0.1*cref
    #xi, zi = np.linspace(x0, x1, num_points), np.linspace(z0, z1, num_points)
    num_points = 500 # Illi: 500
    x0, z0 = 0.15, 0.04
    x1, z1 = 0.24, 0.04
    #x0, z0 = 0.1, 0.04 # surface
    #x1, z1 = 0.23, 0.04 # surface

    xi, zi = np.linspace(x0, x1, num_points), np.linspace(z0, z1, num_points)
    

    lines = np.zeros([in_data.shape[1], num_points])

    for i in range(in_data.shape[1]):
        lines[i,:] = scipy.interpolate.griddata((x, z), in_data[:, i, 0], (xi, zi), method='cubic')


    print('lines shape: ' + str(lines.shape))

    saveFile = case_name + '_z004_lineresults'
    #saveFile = case_name + '_surface_lineresults'
    np.savez(saveFile, lines=lines)
