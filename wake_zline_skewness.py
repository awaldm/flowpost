#/usr/bin/python
"""
26 Sep 2018: currently, still issues with input/output of multiple planes/zones per files
(no idea why it works for the surface version)
however, no problem with single zones
"""
from __future__ import division
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
import tecplot as tp
import helpers.wake_stats as ws
import scipy.interpolate

plt.style.use('seaborn-paper')

# plot size
width = float(7)/float(2)
try:
    from setup_line_plot import * # custom plot setup file
    line_style, markers = setup_plot('JOURNAL_1COL', width, 0.9*width)
except ImportError as e:
    print 'setup_plot not found, setting default line styles...'
    line_style=itertools.cycle(["-", "--", ":"])
    marker_style = itertools.cycle(['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h'])



data_type = 'CRM_plane'
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
    x_TE = 1


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
    aoa = 19.
    x_TE = 0.165
#aoa = {}

case_name = 'DDES_v38h_dt100md_symm_tau2014'
case_name = 'NACA0012_AoA19_DDES_SAO_dt1e5_k1024_turbAoF'
case_name = 'DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2'
def get_result_file(in_path, load_vars):

    data = {}
    dataset = tp.data.load_tecplot(in_path, zones=[0], variables = load_vars, read_data_option = tp.constant.ReadDataOption.Replace)
    for zone in dataset.zones():
        for variable in load_vars:
            print variable
            array = zone.values(variable)
            data[variable] = np.array(array[:]).T
    return data

def get_vert_line(xpos, x_WT, z_WT, skew_u, skew_v, skew_w, num=150, zmin = -0.15, zmax=0.3):
    # xpos can be a list. if so, then the returned values are 2D arrays
    #print('extracting ' + str(len(xpos)) + ' lines from ' + str(len(u_WT)) + ' cases')
    if type(xpos) is not list and len(xpos) <= 1: xpos = [ xpos ]
    zmin = -0.2
    zmax = 0.5
    #zmin = -0.2
    #zmax = 0.25

    #ui = np.zeros([num, len(xpos)])
    #samp_u = np.zeros([num, len(xpos)])
    #samp_w = np.zeros([num, len(xpos)])
    #uu = np.zeros([num, len(xpos)])
    #wi = np.zeros([num, len(xpos)])
    #ww = np.zeros([num, len(xpos)])
    skew_u_line = np.zeros([num, len(xpos)])
    skew_v_line = np.zeros([num, len(xpos)])
    skew_w_line = np.zeros([num, len(xpos)])



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

        skew_u_line[:,pos] = scipy.interpolate.griddata((x_WT, z_WT), skew_u, (xi, zi), method='cubic')
        skew_v_line[:,pos] = scipy.interpolate.griddata((x_WT, z_WT), skew_v, (xi, zi), method='cubic')
        skew_w_line[:,pos] = scipy.interpolate.griddata((x_WT, z_WT), skew_w, (xi, zi), method='cubic')

        #uu[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), uu_WT, (xi, zi), method='cubic')
        #wi[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), w_WT, (xi, zi), method='cubic')
        #ww[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), ww_WT, (xi, zi), method='cubic')


        #samp_u[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), ind_samples_u, (xi, zi), method='cubic')
        #samp_w[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), ind_samples_w, (xi, zi), method='cubic')

    return xi, zi, skew_u_line, skew_v_line, skew_w_line

label2 = 'AoA18'
plane = 'periodic'
plane = 'eta0283'

skewness_file = './results/NACA0012_AoA19_DDES_SAO_dt1e5_k1024_turbAoF_'+plane+'_skewness_rotated_10000_24000.plt'
skewness_file = './results/CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2_'+plane+'_skewness_rotated_5000_16400.plt'
dataset = get_result_file(skewness_file, ['X', 'Y', 'Z', 'skew-u', 'skew-v', 'skew-w'])
skew_u = dataset['skew-u']
skew_v = dataset['skew-v']
skew_w = dataset['skew-w']



#plane = 'eta0283'
if plane is 'eta0201':
    xlim = (1, 1.6)
elif plane is 'eta0283':
    xlim = (1.05, 1.65)
elif plane is 'eta0603':
    xlim = (1.15, 1.75)
elif plane is 'eta0131':
    xlim = (1, 1.6)
elif plane is 'periodic':
    xlim = (0.15, 0.5)

x_TE = 1
xpos_list = [1.3, 1.4, 1.5, 1.6]
xref_list = [1, 1.5, 2, 2.5, 3]
#xref_list = [1,2,3]
xpos_list = (np.asarray(xref_list) * MAC) + x_TE
print xpos_list


x = dataset['X']
z = dataset['Z']
xi, zi, skew_u, skew_v, skew_w = get_vert_line(xpos_list, x, z, skew_u, skew_v, skew_w)

fig, ax = plt.subplots(1, len(xref_list), figsize=(width,0.6*width), sharey=True)
for i in range(len(xpos_list)):
    #col = line_color.next()
    ax[i].plot(skew_u[:,i], zi, label=str(i))
    ax[i].plot(skew_w[:,i], zi, label=str(i))
    #ax[i].plot(CI_hi, zi, linestyle='--', color='darkgrey')
    #ax[i].plot(CI_lo, zi, linestyle='--', color='darkgrey')
    #if i == 2:
    #    ax[i].set_xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[i].grid(True)
    ax[0].set_ylabel('$z [m]$', labelpad=-4.5)

    #ax[i].xaxis.set_ticklabels([-3, 0, 3])
    ax[i].set_xticks([-3, 0, 3])
    ax[i].set_xticklabels([-3, 0, 3])
    
    ax[i].grid(True)
#    plt.xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[i].set_xlim(-4, 4)
    #plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    #plt.ylim(-0.1,0.17)
    #plt.legend(loc='best')
ax[2].set_xlabel('skewness', labelpad=-2)
adjustprops = dict(left=0.13, bottom=0.16, right=0.95, top=0.97, wspace=0.0, hspace=0)
plt.subplots_adjust(**adjustprops)
plt.savefig(case_name+'_'+plane+'_vertlines_skewness.pdf')
plt.close()
