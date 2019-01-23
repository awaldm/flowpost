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
import plane_rstresses as prs
import main.wake.funcs.wake_tscale as wt
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



######################################################################
if __name__ == "__main__":
    data_type = 'CRM_plane'
    #data_type = 'OAT15A'
    #plane = 'eta0131'
    #plane = 'wake_4'
    #plane = 'wake_x3'
    #plane = 'symm'
    out_folder = './results/'


    u,v,w,dataset = prs.process_wake(plt_path, case_name, planelist, zonelist, start_i, end_i, verbosename=True)

    for zone in dataset.zones():
        array = zone.values('X')
        x = np.array(array[:]).T
        array = zone.values('Y')
        y = np.array(array[:]).T
        array = zone.values('Z')
        z = np.array(array[:]).T

    mean_u, mean_v, mean_w = wstat.compute_means(u,v,w)
    uflu = u - np.mean(u, axis=-1, keepdims=True)
    wflu = w - np.mean(w, axis=-1, keepdims=True)

    acf_u = wt.compute_field_acf(uflu, 300)
    acf_w = wt.compute_field_acf(wflu, 300)

    #print acf.shape
    ind_u = wt.compute_field_acf_index(acf_u)
    ind_w = wt.compute_field_acf_index(acf_w)

    n_samples = u.shape[1]
    fig,ax  = wt.contour_plot_index(x,z,n_samples/(2*ind_u), struct=False)
    plt.title('number of independent samples')
    ax.set_xlim(0.8, 1.8)
    ax.set_ylim(0, 0.3)
    ax.set_aspect('equal')
    image_name = 'CRM_'+case_name+'_ind_samples_u'
    print("exporting image " + image_name)
    fig.savefig(image_name + '.png', dpi=600)
    plt.close(fig)


    newvar=dict()
    newvar['n_eff_u'] = n_samples/(2*ind_u)
    newvar['n_eff_w'] = n_samples/(2*ind_w)


    varnames = newvar.keys()

    for keys, _ in newvar.iteritems():
        dataset.add_variable(keys)

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
        filename = out_folder + case_name+'_'+plane+'_ind_samples_'+str(start_i)+'_'+str(end_i)+'.plt'
    else:
        filename = out_folder + case_name+'_'+plane+'_ind_samples.plt'
    import tecplot as tp

    tp.data.save_tecplot_plt(filename, dataset=dataset)



