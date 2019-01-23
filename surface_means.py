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
#import tecreader as tecreader
import matplotlib.pyplot as plt
import helpers.wake_stats as wstat

from wake_config import *

# plot size
width = float(7)/float(2)
try:
    from setup_line_plot import * # custom plot setup file
    line_style, markers = setup_plot('JOURNAL_1COL', width, 0.9*width)
except ImportError as e:
    print 'setup_plot not found, setting default line styles...'
    line_style=itertools.cycle(["-", "--", ":"])
    marker_style = itertools.cycle(['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h'])




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
    #for i in range(zones[0]):
    #    dataset.delete_zones(0)

    return dataset




def process_series(plt_path, case_name, zone_name, zone_no, start_i=None, end_i=None, verbosename=False, datasetfile=None):
    print '\nreading wake...'
    print 80*'-'

    out_folder = './results/'

    varnames = ['cp']
    filelist, num_files = tecreader.get_sorted_filelist(plt_path, 'plt')

    filelist = tecreader.get_cleaned_filelist(filelist,start_i=start_i, end_i=end_i)

    num_files = len(filelist)

    if isinstance(zone_no,(int, long)):
        zone_no = [zone_no]
    else:
        zone_no = zone_no

    #in_data = tecreader.read_series_parallel([plt_path + s for s in filelist], zone_no, varnames, 3)
    in_data = tecreader.read_series([plt_path+s for s in filelist], zone_no, varnames, szplt=False)

    print('data shape: ' + str(in_data.shape))
    cp = in_data[:,:,0]
    print('cp shape: ' + str(cp.shape))

    import tecplot as tp
    if datasetfile is None:
        dataset = tec_get_dataset(plt_path + filelist[0], zone_no = zone_no)
    else:
        dataset = tec_get_dataset(datasetfile)

    for zone in dataset.zones():
        array = zone.values('X')
        x = np.array(array[:]).T
        array = zone.values('Y')
        y = np.array(array[:]).T
        array = zone.values('Z')
        z = np.array(array[:]).T
    #x = None
    #y = None
    #z = None
    #dataset = None

    return x,y,z,cp, dataset



######################################################################
if __name__ == "__main__":
    data_type = 'CRM_plane'
    data_type = 'OAT15A'
    plane = 'eta0201'
    #plane = 'wake_x3'
    plane = 'symm'
    x,y,z,cp,dataset = process_series(plt_path, case_name, zone_name, zonelist, start_i, end_i, verbosename=True, datasetfile=datasetfile)





    print 'done reading'
    print cp.shape
    mean_cp = np.mean(cp, axis=-1)
    var_cp = np.var(cp, axis=-1)

    import tecplot as tp
    #dataset = tec_get_dataset(plt_path + 'CRM_v51_WT_defa45_M085_URANS-SAQCR.surface.pval.unsteady_i=4190_t=5.805835979e-02.plt', zone_no = zonelist)

    #####################################################################################################
    # this is for reading coordinates and acquiring a dataset for writing PLT
    #dataset = tp.data.load_tecplot(plt_path + 'CRM_v51_WT_defa45_M085_URANS-SAQCR.surface.pval.unsteady_i=4190_t=5.805835979e-02.plt', variables=['X', 'Y', 'Z'], zones=zonelist, read_data_option = tp.constant.ReadDataOption.Replace)
    #dataset = tp.data.load_tecplot(plt_path + 'CRM_v51_WT_defa45_M085_URANS-SAQCR.surface.pval.unsteady_i=4190_t=5.805835979e-02.plt', variables=['X', 'Y', 'Z'], read_data_option = tp.constant.ReadDataOption.Replace)

    #dataset = tp.data.load_tecplot(source_path + filelist[0], zones=zone_no, variables=['X', 'Y', 'Z'], read_data_option = tp.constant.ReadDataOption.Replace)
    # this is necessary, because in case of szplt we can not choose to load only a few zones
    #dataset.delete_zones([dataset.zone('symleft'), dataset.zone('symleft'), dataset.zone('upperside'),dataset.zone('trailingedge'),dataset.zone('lowerside'), dataset.zone('farfield')])
    #dataset.delete_zones([dataset.zone('symmetry'), dataset.zone('fuselage')])

    #nonwingzones = [Z for Z in dataset.zones() if 'wing' is not in Z.name]
    #print(str(wing))
    #dataset.delete_zones(nonwingzones)
    #dataset.delete_zones(dataset.zone('symmetrieebene'))
    '''
    for zone in dataset.zones():
        print('found zone: ' + str(zone.name))
        if 'fl' not in zone.name:
            print('deleting zone: ' + str(zone.name))
            dataset.delete_zones([dataset.zone(zone.name)])
    for zone in dataset.zones():
        print('found zone: ' + str(zone.name))
        if 'wing' not in zone.name:
            print('deleting zone: ' + str(zone.name))
            dataset.delete_zones([dataset.zone(zone.name)])
    for zone in dataset.zones():
        print('found zone: ' + str(zone.name))
        if 'wing' not in zone.name:
            print('deleting zone: ' + str(zone.name))
            dataset.delete_zones([dataset.zone(zone.name)])
    for zone in dataset.zones():
        print('found zone: ' + str(zone.name))
        if 'wake' in zone.name:
            print('deleting zone: ' + str(zone.name))
            dataset.delete_zones([dataset.zone(zone.name)])
    for zone in dataset.zones():
        print('found zone: ' + str(zone.name))
        if 'eta' in zone.name:
            print('deleting zone: ' + str(zone.name))
            dataset.delete_zones([dataset.zone(zone.name)])

    print('deleted 2D zones, what remains is')
    print('number of zones: ' + str(dataset.num_zones))
    zonecount = 0
    for zone in dataset.zones():
        print(zone.name)
        print(zonecount)
        zonecount += 1
    '''
    #print('remaining zones:')
    #for zone in dataset.zones():
    #    print(zone.name)

    #delvars=[]
    #for v in dataset.variables():
    #    if 'X' is not v.name and 'Y' is not v.name and 'Z' is not v.name:
    #        delvars.append(v)
    #dataset.delete_variables(delvars)
    print([v.name for v in dataset.variables()])


    #for zone in dataset.zones():
    #    array = zone.values('X')
    #    x = np.array(array[:]).T
    #    array = zone.values('Y')
    #    y = np.array(array[:]).T
    #    array = zone.values('Z')
    #    z = np.array(array[:]).T

    newvar=dict()
    newvar['mean-cp'] = mean_cp
    newvar['var-cp'] = var_cp
    varnames = newvar.keys()

    offset = 0

    for keys, _ in newvar.iteritems():
        dataset.add_variable(keys)

    for zone in dataset.zones():
        #print('adding variable ' + varname1 + ' to zone ' + zone.name)
        zone_points = zone.num_points
        print('zone has ' + str(zone_points) + ' points')
        #print(len(array1[offset:offset+zone_points]))
        for var in varnames:
            print(var)
            zone.values(var)[:] = newvar[var][offset:offset+zone_points].ravel() # this is what the pytecplot example uses

        offset = offset + zone_points
#    if verbosename:
#        filename = out_folder + case_name+'_'+plane+'_rstresses_'+str(start_i)+'_'+str(end_i)+'.plt'
#    else:
    filename = case_name+'_meancp.plt'
    tp.data.save_tecplot_plt(filename, dataset=dataset)
