#/usr/bin/python
"""
26 Sep 2018: currently, still issues with input/output of multiple planes/zones per files
(no idea why it works for the surface version)
however, no problem with single zones
"""

import os, sys
import time
import shutil
import numpy as np
import matplotlib.mlab as mlab
import matplotlib as mpl
import pyTecIO_AW.tecreader as tecreader
#import tecreader as tecreader
import matplotlib.pyplot as plt
import helpers.wake_stats as ws
import helpers.plotters as wplot
# plot size
width = float(7)/float(2)
try:
    from setup_line_plot import * # custom plot setup file
    line_style, markers = setup_plot('JOURNAL_1COL', width, 0.9*width)
except ImportError as e:
    print 'setup_plot not found, setting default line styles...'
    line_style=itertools.cycle(["-", "--", ":"])
    marker_style = itertools.cycle(['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h'])


from wake_config import *


def tec_get_dataset(filename, zone_no=None, variables=['X', 'Y', 'Z']):
    import tecplot as tp
    '''
    Obtain a single Tecplot zone and its coordinates.
    Useful in a situation when further processing yields some data.
    '''
    if isinstance(zone_no,(int, long)):
        zones = [zone_no]
    else:
        zones = zone_no
    dataset = tp.data.load_tecplot(filename, zones=zone_no, variables=['X', 'Y', 'Z'], read_data_option = tp.constant.ReadDataOption.Replace)
    # even though we loaded only a single zone, we need to remove the preceding zombie zones
    # otherwise they will get written and we have no chance to read properly (with paraview)
    # this works only (and is necessary) when using the single zone/no extra datasetfile approach
    for i in range(zones[0]):
        dataset.delete_zones(0)

    return dataset




def process_wake(plt_path, case_name, plane, zone_no, start_i=None, end_i=None, verbosename=False, datasetfile=None):
    print '\nreading wake...'
    print 80*'-'

    varnames = None
    filelist, num_files = tecreader.get_sorted_filelist(plt_path, 'plt')

    filelist = tecreader.get_cleaned_filelist(filelist,start_i=start_i, end_i=end_i)

    num_files = len(filelist)

    if isinstance(zone_no,(int, long)):
        zone_no = [zone_no]
    else:
        zone_no = zone_no

    in_data = tecreader.read_series_parallel([plt_path + s for s in filelist], zone_no, varnames, 3)

    print('data shape: ' + str(in_data.shape))
    u = in_data[:,:,0]
    v = in_data[:,:,1]
    w = in_data[:,:,2]
    print('u shape: ' + str(u.shape))

    import tecplot as tp
    if datasetfile is None:
        dataset = tec_get_dataset(plt_path + filelist[0], zone_no = zone_no)
    else:
        dataset = tec_get_dataset(datasetfile)

    return u,v,w,dataset





######################################################################
if __name__ == "__main__":
    data_type = 'CRM_plane'
    data_type = 'NACA0012'
    #data_type = 'OAT15A'
    #plane = 'eta0131'
    #plane = 'wake_4'
    #plane = 'wake_x3'
    #plane = 'symm'

    # CRM
    if data_type == 'CRM_plane':
        dt = 0.0003461
        MAC = 0.189144
        u_inf = 54.65
        aoa = 18
        x_PMR = 0.90249
        z_PMR = 0.14926


    # OAT15A
    if data_type == 'OAT15A':
        dt = 2e-5
        MAC = 0.23
        u_inf = 241
    elif data_type == 'NACA0012':
        dt = 1e-5
        MAC = 0.165
        u_inf = 85.7145
        x_PMR = MAC/4
        z_PMR = 0
        aoa = 19


    out_folder = './results/'

    u,v,w,dataset = process_wake(plt_path, case_name, planelist, zonelist, start_i, end_i, verbosename=True)

    u, w = ws.rotate_velocities(u,v,w, x_PMR, z_PMR, alpha=aoa)

    from scipy.stats import kurtosis, skew
    skew_u = skew(u, axis=-1)
    skew_v = skew(v, axis=-1)
    skew_w = skew(w, axis=-1)

    #Sr = freq * MAC / u_inf


    newvar=dict()
    newvar['skew-u'] = skew_u
    newvar['skew-v'] = skew_v
    newvar['skew-w'] = skew_w

    print('computed wake skewness, result shape: ' + str(skew_u.shape))

    for zone in dataset.zones():
        array = zone.values('X')
        x = np.array(array[:]).T
        array = zone.values('Y')
        y = np.array(array[:]).T
        array = zone.values('Z')
        z = np.array(array[:]).T


    for keys, _ in newvar.iteritems():
        dataset.add_variable(keys)

    varnames = ['X', 'Y', 'Z']
    print varnames
    print newvar.keys()
    for key in newvar.keys():
        varnames.append(key)
    print varnames
    x, z = ws.transform_wake_coords(x, z, x_PMR, z_PMR, aoa)
    newvar['X'] = x
    newvar['Y'] = y
    newvar['Z'] = z

    offset = 0


    for zone in dataset.zones():
        zone_points = zone.num_points
        print('zone has ' + str(zone_points) + ' points')
        for var in varnames:
            print(var)
            new_array = list(newvar[var][offset:offset+zone_points])
            zone.values(var)[:] = new_array
        offset = offset + zone_points - 1
    verbosename = True
    if verbosename:
        filename = out_folder + case_name+'_'+plane+'_skewness_rotated_'+str(start_i)+'_'+str(end_i)+'.plt'
    else:
        filename = out_folder + case_name+'_'+plane+'_skewness_rotated.plt'
    import tecplot as tp

    tp.data.save_tecplot_plt(filename, dataset=dataset)
