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
import helpers.wake_stats as wstat
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


def get_vert_line(xpos, x_WT, z_WT, u_WT, uu_WT, w_WT, ww_WT, ind_samples_u, ind_samples_w, num=250, zmin = -0.15, zmax=0.3):
    # xpos can be a list. if so, then the returned values are 2D arrays
    print('extracting ' + str(len(xpos)) + ' lines from ' + str(len(u_WT)) + ' cases')
    if type(xpos) is not list and len(xpos) <= 1: xpos = [ xpos ]

    ui = np.zeros([num, len(xpos)])
    samp_u = np.zeros([num, len(xpos)])
    samp_w = np.zeros([num, len(xpos)])
    uu = np.zeros([num, len(xpos)])
    wi = np.zeros([num, len(xpos)])
    ww = np.zeros([num, len(xpos)])


    # iterate over positions
    for pos in range(len(xpos)):
        print xpos[0]
        x0, z0 = xpos[pos], zmin
        x1, z1 = xpos[pos], zmax
        print('extracting line from ('+str(x0)+', '+str(z0)+') to ('+str(x1)+', '+str(z1)+')')

        # create line as interpolation targeti
        xi, zi = np.linspace(x0, x1, num), np.linspace(z0, z1, num)
        print('obtained line of shape ' + str(xi.shape))

        # iterate over cases
        ui[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), u_WT, (xi, zi), method='cubic')
        uu[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), uu_WT, (xi, zi), method='cubic')
        wi[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), w_WT, (xi, zi), method='cubic')
        ww[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), ww_WT, (xi, zi), method='cubic')


        samp_u[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), ind_samples_u, (xi, zi), method='cubic')
        samp_w[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), ind_samples_w, (xi, zi), method='cubic')

    return xi, zi, ui, uu, wi, ww, samp_u, samp_w

######################################################################
if __name__ == "__main__":
    data_type = 'CRM_plane'
    #data_type = 'OAT15A'
    #plane = 'eta0131'
    #plane = 'wake_4'
    #plane = 'wake_x3'
    #plane = 'symm'
    out_folder = './results/'

    '''
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

    plt_path = '/home/andreas/hazelhen_WS1/v38_tet/DDES/DDES_dt100_ldDLR/sol/surface/plt/'
    case_name = 'DDES_v38t_dt100_ldDLR_chim_tau2017'
    zonelist = [10]
    planelist = ['eta0201']
    start_i = 2250
    end_i = 4800

    plt_path = '/home/andreas/NAS_CRM/M025/AoA18/v38_h05/DDES_dt100_ldDLR_CFL2_eigval02_pswitch4_tau2014/sol/surface/plt/'
    case_name = 'DDES_v38h_dt100_ldDLR_CFL2_eigval02_pswitch1_symm_tau2014'
    zonelist = [11]
    planelist = ['eta0283']
    start_i = 3330
    end_i = 6800

    plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/IDDES_dt200_ldDLR/sol/surface/plt/'
    case_name = 'CRM_v38_IDDES_dt200_ldDLR'
    zonelist = [12]
    planelist = ['eta0283']
    start_i = 26000
    end_i = 28400
    zone_name ='wing_ss'

    plt_path = '/home/andreas/hazelhen_WS1/v52/DDES-SAO/dt200_LD2/sol/surface/plt/'
    case_name = 'CRM_v52_dt200_LD2'
    zonelist = [9,10,11,12,13,14,15,16,17]
    start_i = 3000
    end_i = 4700
    planelist = ['eta0131','eta0201','eta0283','eta0397','eta0501','eta0603','eta0727','eta0848','eta0950']
    zone_name ='wing_ss'
    planelist = [planelist[0]]
    zonelist = [zonelist[0]]

    plt_path = '/home/andreas/hazelhen_WS1/v38_hex/DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017/sol/surface/plt/'
    case_name = 'DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017'
    zonelist = [11]
    planelist = ['eta0283']
    start_i = 5410
    end_i = 5430


    plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/URANS-SST/sol/surface/plt/'
    case_name = 'URANS-SST'
    zonelist = [13]
    planelist = ['eta0283']
    start_i = 3500
    end_i = 5100




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
    #zone_no = 10 # 11
    #start_i = 1200
    #end_i = 3200
    plt_path = '/home/andreas/laki_ehrle/AW_tests/AZDES-SSG/AZDES-SSG_dt2e5_turbRoe2nd_SGDH_Lc00161/sol/surface/'
    case_name = 'OAT15A_AZDES-SSG'
    zone_no = 0
    zonelist = [0]
    start_i = 15490
    end_i = 17450
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

    plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/AoA16_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017/sol/surface/plt/'
    case_name = 'CRM_v38h_AoA16_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017'
    #zonelist = [3]
    zonelist = [9,10,11,12,13,14]
    zonelist = [11]
    start_i = 2200
    end_i = 4800
    planelist= ['eta0131', 'eta0201', 'eta0283', 'eta0397', 'eta0501', 'eta0603']
    planelist= ['eta0283']
    plane = planelist[0]
    datasetfile = 'v38h_etaplanes.plt'


    plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2/sol/surface/plt/'
    case_name = 'CRM_v38h_AoA18_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2'
    zonelist = [11]
    start_i = 1000
    end_i = 3000
    planelist= ['eta0283']
    plane = planelist[0]
    datasetfile = 'v38h_etaplanes.plt'


    plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017/sol/surface/plt/'
    case_name = 'CRM_v38h_AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017'
    #zonelist = [3]
    zonelist = [9,10,11,12,13,14]
    zonelist = [11]
    start_i = 4000
    end_i = 6400
    planelist= ['eta0131', 'eta0201', 'eta0283', 'eta0397', 'eta0501', 'eta0603']
    planelist= ['eta0283']
    datasetfile = 'v38h_etaplanes.plt'

    plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/AoA16_Re172_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017/sol/surface/plt/'
    case_name = 'CRM_v38h_AoA16_Re172_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017'
    #zonelist = [3]
    zonelist = [9,10,11,12,13,14]
    zonelist = [11]
    start_i = 2000
    end_i = 4600
    planelist= ['eta0131', 'eta0201', 'eta0283', 'eta0397', 'eta0501', 'eta0603']
    planelist= ['eta0283']
    datasetfile = 'v38h_etaplanes.plt'


    plt_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/AoA18_Re172_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017/sol/surface/plt/'
    case_name = 'CRM_v38h_AoA18_Re172_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017'
    #zonelist = [3]
    zonelist = [9,10,11,12,13,14]
    zonelist = [11]
    start_i = 4500
    end_i = 7400
    planelist= ['eta0131', 'eta0201', 'eta0283', 'eta0397', 'eta0501', 'eta0603']
    planelist= ['eta0283']
    datasetfile = 'v38h_etaplanes.plt'
    '''

    #for zone in range(len(zonelist)):
    #u,v,w,dataset = process_wake(plt_path, case_name, planelist, zonelist, start_i, end_i, verbosename=True, datasetfile=datasetfile)
    u,v,w,dataset = process_wake(plt_path, case_name, planelist, zonelist, start_i, end_i, verbosename=True)
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
    verbosename = True
    if verbosename:
        filename = out_folder + case_name+'_'+plane+'_rstresses_'+str(start_i)+'_'+str(end_i)+'.plt'
    else:
        filename = out_folder + case_name+'_'+plane+'_rstresses.plt'
    import tecplot as tp

    tp.data.save_tecplot_plt(filename, dataset=dataset)
    print u[:,1]

    fig, ax = wplot.plot_velocity(x, z, u[:,1], kind='CRM_wake', shape=None, struct=False)
    if 'eta' in plane:
        ax.set_xlim(0.7, 1.8)
        ax.set_ylim(-0.1, 0.6)
    ax.set_aspect('equal')
    plt.savefig(case_name + '_' + plane + '_inst_xvel.png', dpi=600)
