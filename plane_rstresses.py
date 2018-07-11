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




def process_wake(plt_path, case_name, plane, zone_no, start_i=None, end_i=None, verbosename=False):
    print '\nreading wake...'
    print 80*'-'

    out_folder = './results/'

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
    mean_u, mean_v, mean_w = wstat.compute_means(u,v,w)
    uu,vv,ww,uv,uw,vw = wstat.calc_rstresses(u,v,w)

    mean_u = np.mean(u, axis=-1)
    mean_v = np.mean(v, axis=-1)
    mean_w = np.mean(w, axis=-1)
    '''
    uu = np.var(u, axis=-1)
    ww = np.var(w, axis=-1)
    uw = np.zeros(mean_u.shape)
    #aaa = (u-np.mean(u, keepdims=True))
    for i in range(len(mean_u)):
        uflu = u[i,:] - np.mean(u[i,:], keepdims = True)
        wflu = w[i,:] - np.mean(w[i,:], keepdims = True)
        uw[i] = np.mean(np.multiply(uflu, wflu))
    print('uw shape: ' + str(uw.shape))
    '''

    #utils.ensure_dir(img_path)
    #utils.ensure_dir(phi_path)

    # CRM
    if data_type == 'CRM_plane':
        dt = 0.0003461
        MAC = 0.189144
        u_inf = 54.65

    # OAT15A
    if data_type == 'OAT15A':
        dt = 2e-5
        MAC = 0.23
        u_inf = 241

    #Sr = freq * MAC / u_inf


    newvar=dict()
    newvar['mean-u'] = mean_u
    newvar['mean-v'] = mean_v
    newvar['mean-w'] = mean_w
    newvar['resolved-uu'] = uu
    newvar['resolved-vv'] = vv
    newvar['resolved-ww'] = ww
    newvar['resolved-uv'] = uv
    newvar['resolved-uw'] = uw
    newvar['resolved-vw'] = vw

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
    if verbosename:
        filename = out_folder + case_name+'_'+plane+'_rstresses_'+str(start_i)+'_'+str(end_i)+'.plt'
    else:
        filename = out_folder + case_name+'_'+plane+'_rstresses.plt'
    tp.data.save_tecplot_plt(filename, dataset=dataset)


######################################################################
if __name__ == "__main__":
    data_type = 'CRM_plane'
    data_type = 'OAT15A'
    plane = 'eta0201'
    #plane = 'wake_x3'
    plane = 'symm'

    plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/DDES_dt100_md_cK/sol/surface/plt/'
    case_name = 'DDES_v38_dt100_md_cK'
    zonelist = [12] # 12
    planelist = ['eta0283']
    start_i = 28500 # 27120
    end_i = 31800

    plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/DDES_dt100_md_CFL2_eigval015_pswitch1_symm_tau2017/sol/surface/plt/'
    case_name = 'DDES_v38h_dt100_md_CFL2_eigval015_pswitch1_symm_tau2017'
    zonelist = [11]
    planelist = ['eta0283']
    start_i = 7400
    end_i = 9990

    plt_path = '/home/andreas/hazelhen_WS1/v38_hex/DDES_dt100_md_CFL2_eigval02_pswitch1_tau2014/sol/surface/plt/'
    case_name = 'DDES_v38h_dt100_md_CFL2_eigval02_pswitch1_symm_tau2014'
    zonelist = [10]
    planelist = ['eta0201']
    start_i = 1000
    end_i = 5600

    plt_path = '/home/andreas/hazelhen_WS1/v38_hex/DDES_dt100_md_CFL2_eigval02_pswitch1_chim_tau2014/sol/surface/plt/'
    case_name = 'DDES_v38h_dt100_md_CFL2_eigval02_pswitch1_chim_tau2014'
    zonelist = [11]
    planelist = ['eta0283']
    start_i = 1200
    end_i = 5600

    plt_path = '/home/andreas/hazelhen_WS1/v38_hex/DDES_dt100_md_CFL2_eigval02_pswitch4_tau2014/sol/surface/plt/'
    case_name = 'DDES_v38h_dt100_md_CFL2_eigval02_pswitch4_symm_tau2014'
    zonelist = [11]
    planelist = ['eta0283']
    start_i = 2200
    end_i = 7600

    

    #plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38t/DDES_dt100_md_chim_tau2017/sol/surface/plt/'
    #case_name = 'DDES_v38t_dt100_md_chim_tau2017'
    #zonelist = [11]
    #planelist = ['eta0283']
    #start_i = 2800
    #end_i = 6500

    #plt_path = '/home/andreas/hazelhen_WS1/v38_tet/DDES/DDES_dt100_md_symm_tau2014/sol/surface/plt/'
    #case_name = 'DDES_v38t_dt100_md_symm_tau2014'
    #zonelist = [10]
    #planelist = ['eta0201']
    #start_i = 1100
    #end_i = 2800

    
    '''
    plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/DDES_dt180_md/sol/surface/plt/'
    case_name = 'DDES_v38_dt100md'
    zone_no = 11
    start_i = 26000
    end_i = 27510
    #zone_no = 10 # 11
    #start_i = 5000
    #end_i = 7000
    #plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/DDES_v38h05_dt50/sol/surface/plt/'
    #case_name = 'DDES_v38h_dt50md_symm_tau2014'
    #zone_no = 21 # 10 # 11
    #start_i = 2050
    #end_i = 5600
    '''
    #zone_no = 10 # 11
    #start_i = 1200
    #end_i = 3200
    plt_path = '/home/andreas/laki_ehrle/AW_tests/AZDES-SSG/AZDES-SSG_dt2e5_turbRoe2nd_SGDH_Lc00161/sol/surface/'
    case_name = 'OAT15A_AZDES-SSG'
    zone_no = 0
    zonelist = [0]
    start_i = 15490
    end_i = 17450
    '''
    plt_path = '/home/andreas/laki_ehrle/AW_tests/URANS-SSG/URANS-SSG_dt2e5_2016.2_turbRoe2nd_SGDH/sol/surface/'
    case_name = 'OAT15A_URANS-SSG'
    zone_no = 0
    zonelist = [0]
    start_i = 9640
    end_i = 12880


    plt_path = '/home/andreas/hazelhen_WS1/v38_tet/DDES/DDES_dt100_md_symm_tau2014/sol/surface/plt/'
    case_name = 'DDES_v38t_dt100md_symm_tau2014'
    zone_no = 10 # 11
    start_i = 1200
    end_i = 3100
    #start_i = 3200
    #end_i = 5600
    zonelist = [10,11,12,13]


    plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/URANS-SSG_turbRoe2nd/sol/surface/plt/'
    case_name = 'URANS-SSG_turbRoe2nd'
    zonelist = [12,13,14,15]

    planelist= ['eta0201', 'eta0283', 'eta0397', 'eta0603']
    zonelist = [13]
    planelist = ['eta0283']

    plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/URANS-SAO/sol/surface/plt/surfaces/'
    case_name = 'URANS-SAO'
    start_i = 10000
    end_i = 17260
    zonelist = [4]
    planelist = ['eta0283']
    '''


    for zone in range(len(zonelist)):
        process_wake(plt_path, case_name, planelist[zone], zonelist[zone], start_i, end_i, verbosename=True)


#    process_wake(plt_path, case_name, plane, zone_no, start_i, end_i)

