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




aoa = {}
x_PMR = 0.90249
z_PMR = 0.14926
u_inf = 54.65
case_name = 'DDES_v38h_dt100md_symm_tau2014'

in_path = case_name + '_result.plt'
in_vars = ['X', 'Y', 'Z', 'mean-u', 'mean-v', 'mean-w', 'resolved-uu', 'resolved-uw', 'resolved-ww', 'resolved-vv']

def get_result_file(in_path, load_vars):

    data = {}
    dataset = tp.data.load_tecplot(in_path, zones=[0], variables = load_vars, read_data_option = tp.constant.ReadDataOption.Replace)
    for zone in dataset.zones():
        for variable in load_vars:
            print variable
            array = zone.values(variable)
            data[variable] = np.array(array[:]).T
    return data

def get_vert_line(xpos, x_WT, z_WT, df_plane, sf_plane, num=100, zmin = -0.15, zmax=0.3):
    # xpos can be a list. if so, then the returned values are 2D arrays
    #print('extracting ' + str(len(xpos)) + ' lines from ' + str(len(u_WT)) + ' cases')
    if type(xpos) is not list and len(xpos) <= 1: xpos = [ xpos ]
    zmin = 0
    zmax = 0.5

    #ui = np.zeros([num, len(xpos)])
    #samp_u = np.zeros([num, len(xpos)])
    #samp_w = np.zeros([num, len(xpos)])
    #uu = np.zeros([num, len(xpos)])
    #wi = np.zeros([num, len(xpos)])
    #ww = np.zeros([num, len(xpos)])
    df = np.zeros([num, 4, len(xpos)])
    sf = np.zeros([num, 4, len(xpos)])


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
        for i in range(4):
            df[:,i,pos] = scipy.interpolate.griddata((x_WT, z_WT), df_plane[:,i], (xi, zi), method='cubic')
            sf[:,i,pos] = scipy.interpolate.griddata((x_WT, z_WT), sf_plane[:,i], (xi, zi), method='cubic')
        #uu[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), uu_WT, (xi, zi), method='cubic')
        #wi[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), w_WT, (xi, zi), method='cubic')
        #ww[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), ww_WT, (xi, zi), method='cubic')


        #samp_u[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), ind_samples_u, (xi, zi), method='cubic')
        #samp_w[:, pos] = scipy.interpolate.griddata((x_WT, z_WT), ind_samples_w, (xi, zi), method='cubic')

    return xi, zi, df, sf

label2 = 'AoA18'
plane = 'eta0283'
samp_file_u = './results/CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2_'+plane+'_ind_samples_5000_13000.plt'
samp_file_w = './results/CRM_v38h_AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_k1024_'+plane+'_ind_samples_w_5000_15600.plt'
stress_file = './results/CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2_'+plane+'_rstresses_5000_14400.plt'

QA_file = './results/CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2_'+plane+'_QA_rotated_5000_16400.plt'
aoa = 18.

dataset = get_result_file(QA_file, ['X', 'Y', 'Z', 'df1', 'df2', 'df3', 'df4', 'sf1', 'sf2', 'sf3', 'sf4'])
#samples_u = dataset['ind_samples']
df_plane = np.c_[dataset['df1'], dataset['df2'], dataset['df3'], dataset['df4'] ]
sf_plane = np.c_[dataset['sf1'], dataset['sf2'], dataset['sf3'], dataset['sf4'] ]

print('sf matrix has shape ' + str(sf_plane.shape))

#dataset = get_result_file(stress_file, in_vars)


#x_WT, z_WT = ws.transform_wake_coords(dataset['X'], dataset['Z'], x_PMR, z_PMR, aoa)
#u_WT, w_WT = ws.rotate_velocities(dataset['mean-u'], dataset['mean-v'], dataset['mean-w'], x_PMR, z_PMR, aoa)
#uu_WT, _, ww_WT, uv_WT, uw_WT, vw_WT = ws.rotate_stresses(dataset['resolved-uu'],None,dataset['resolved-ww'], uw=dataset['resolved-uw'], x_PMR=x_PMR, z_PMR=z_PMR, alpha=aoa)
#vv_WT = dataset['resolved-vv']
# end loading 3 cases
#######################


plane = 'eta0283'
if plane is 'eta0201':
    xlim = (1, 1.6)
elif plane is 'eta0283':
    xlim = (1.05, 1.65)
elif plane is 'eta0603':
    xlim = (1.15, 1.75)
elif plane is 'eta0131':
    xlim = (1, 1.6)


xpos_list = [1.3, 1.4, 1.5, 1.6]
xref_list = [1, 1.5, 2, 2.5, 3]
#xref_list = [2]
xpos_list = (np.asarray(xref_list) * 0.189144) + 1
print xpos_list


x = dataset['X']
z = dataset['Z']
xi, zi, df, sf = get_vert_line(xpos_list, x, z, df_plane, sf_plane)

print('shape of resulting df: ' + str(df.shape))
u_inf = 54.65

fig, ax = plt.subplots(1, len(xref_list), figsize=(width,0.6*width), sharey=True)
for i in range(len(xpos_list)):
    #col = line_color.next()
    for j in range(4):
        ax[i].plot(df[:,j,i], zi, label=str(j+1))
    #ax[i].plot(CI_hi, zi, linestyle='--', color='darkgrey')
    #ax[i].plot(CI_lo, zi, linestyle='--', color='darkgrey')
    #if i == 2:
    #    ax[i].set_xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[i].grid(True)

#        axes.get_yaxis().set_visible(False)
#        axes.yaxis.set_ticklabels([])
#    plt.grid(True)
#    plt.xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[i].set_ylabel('$z [m]$', labelpad=-4.5)
    ax[i].set_xlim(0.0, 1)
    #plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    #plt.xlim(0,0.05)
    #plt.ylim(-0.1,0.17)
    plt.legend(loc='best')
adjustprops = dict(left=0.15, bottom=0.15, right=0.95, top=0.97, wspace=0.1, hspace=0)
plt.subplots_adjust(**adjustprops)
plt.savefig('QA_'+plane+'_df.pdf')
plt.close()
fig, ax = plt.subplots(1, len(xref_list), figsize=(width,0.6*width), sharey=True)
for i in range(len(xpos_list)):
    #col = line_color.next()
    for j in range(4):
        ax[i].plot(sf[:,j,i] / (u_inf*u_inf), zi, label=str(j+1))
    #ax[i].plot(CI_hi, zi, linestyle='--', color='darkgrey')
    #ax[i].plot(CI_lo, zi, linestyle='--', color='darkgrey')
    #if i == 2:
    #    ax[i].set_xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[i].grid(True)

#        axes.get_yaxis().set_visible(False)
#        axes.yaxis.set_ticklabels([])
#    plt.grid(True)
#    plt.xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[i].set_ylabel('$z [m]$', labelpad=-4.5)
    #ax.set_xlim(0.0, 1)
    #plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    #plt.xlim(0,0.05)
    #plt.ylim(-0.1,0.17)
    plt.legend(loc='best')
adjustprops = dict(left=0.15, bottom=0.15, right=0.95, top=0.97, wspace=0.1, hspace=0)
plt.subplots_adjust(**adjustprops)
plt.savefig('QA_'+plane+'_sf.pdf')
plt.close()

