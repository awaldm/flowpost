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
import tecreader as tecreader
#import tecreader as tecreader
import matplotlib.pyplot as plt


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
    for i in range(zones[0]):
        dataset.delete_zones(0)

    return dataset





######################################################################
if __name__ == "__main__":
    print 'creating spacetime plot...'
    print 80*'-'

    ######################################################################
    # USER INPUT
    args = sys.argv
    print 'obtained ' + str(len(args)-1) + ' arguments'
#    print os.path.isfile(args[1])
    if len(args) == 3: # case name and plane given
        case = args[1]
        plane = args[2]
    elif len(args) == 1: #nothing given
        case = 'DDES_v38h_dt100_md'
        plane = 'eta0283'
        zone_no = [9,10,11,12,13]
        zone_no = 11
    else:
        print 'wrong option'
        sys.exit(0)

    # END USER INPUT
    ######################################################################
    print '\nreading wake...'
    print 80*'-'
    plt_path = '/home/andreas/hazelhen_WS1/v38_hex/DDES_dt100_md_CFL2_eigval015/sol/surface/plt/'
    plt_path = './'
    varnames = None
    filelist, num_files = tecreader.get_sorted_filelist(plt_path, 'plt')
    #start_i, end_i, num_i = cleanlist(start_i=start_i, end_i=end_i)
    start_i = 2200
    end_i = 5200
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
    mean_u = np.mean(u, axis=-1)
    mean_v = np.mean(v, axis=-1)
    mean_w = np.mean(w, axis=-1)
    #uu = np.var(u, axis=-1)
    #ww = np.var(w, axis=-1)

    #uu, vv, ww, uv, uw, vw, kt = ws.compute_rstresses_1D(u,v,w)
    uu = np.var(u, axis=-1)
    ww = np.var(w, axis=-1)
    uw = np.zeros(mean_u.shape)
    #aaa = (u-np.mean(u, keepdims=True))
    for i in range(len(mean_u)):
        uflu = u[i,:] - np.mean(u[i,:], keepdims = True)
        wflu = w[i,:] - np.mean(w[i,:], keepdims = True)
        uw[i] = np.mean(np.multiply(uflu, wflu))
    print('uw shape: ' + str(uw.shape))

    #uw = np.mean( np.multiply( (u - np.mean(u, keepdims=True)), (w-np.mean(w,keepdims=True))), axis=-1)

    #utils.ensure_dir(img_path)
    #utils.ensure_dir(phi_path)


    dt = 0.0003461


    MAC = 0.189144
    u_inf = 54.65

    #Sr = freq * MAC / u_inf


    newvar=dict()
    newvar['mean-u'] = mean_u
    newvar['mean-v'] = mean_v
    newvar['mean-w'] = mean_w
    newvar['resolved-uu'] = uu
    newvar['resolved-ww'] = ww
    newvar['resolved-uw'] = uw
    varnames = newvar.keys()

    print(ww.shape)

    import tecplot as tp
    dataset = tec_get_dataset(plt_path + filelist[0], zone_no = zone_no)

    for zone in dataset.zones():
        array = zone.values('X')
        x = np.array(array[:]).T
        array = zone.values('Y')
        y = np.array(array[:]).T
        array = zone.values('Z')
        z = np.array(array[:]).T


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
            new_array = list(newvar[var][offset:offset+zone_points])
            zone.values(var)[:] = new_array
        offset = offset + zone_points - 1
    filename = 'result.plt'
    tp.data.save_tecplot_plt(filename, dataset=dataset)

