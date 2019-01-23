#/usr/bin/python
"""
26 Sep 2018: currently, still issues with input/output of multiple planes/zones per files
(no idea why it works for the surface version)
however, no problem with single zones
"""

import os, sys
import matplotlib
matplotlib.use('Agg')

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

label2 = 'AoA18'
plane = 'eta0283'
samp_file_u = './results/CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2_'+plane+'_ind_samples_5000_13000.plt'
samp_file_w = './results/CRM_v38h_AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_k1024_'+plane+'_ind_samples_w_5000_15600.plt'
stress_file = './results/CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2_'+plane+'_rstresses_5000_14400.plt'

aoa = 18.

dataset = get_result_file(samp_file_u, ['X', 'Y', 'Z', 'ind_samples'])
samples_u = dataset['ind_samples']
dataset = get_result_file(samp_file_w, ['X', 'Y', 'Z', 'ind_samples'])
samples_w = dataset['ind_samples']


dataset = get_result_file(stress_file, in_vars)


x_WT, z_WT = ws.transform_wake_coords(dataset['X'], dataset['Z'], x_PMR, z_PMR, aoa)
u_WT, w_WT = ws.rotate_velocities(dataset['mean-u'], dataset['mean-v'], dataset['mean-w'], x_PMR, z_PMR, aoa)
uu_WT, _, ww_WT, uv_WT, uw_WT, vw_WT = ws.rotate_stresses(dataset['resolved-uu'],None,dataset['resolved-ww'], uw=dataset['resolved-uw'], x_PMR=x_PMR, z_PMR=z_PMR, alpha=aoa)
#ivv_WT = dataset['resolved-vv']
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
samples_u = np.clip(samples_u, 0, 300)
samples_w = np.clip(samples_w, 0, 300)



xi, zi, ui, uu, wi, ww, samp_u, samp_w = get_vert_line(xpos_list, x_WT, z_WT, u_WT, uu_WT, w_WT, ww_WT, samples_u, samples_w)
u_inf = 54.65


fig, ax = plt.subplots(1, len(xref_list), figsize=(width,0.6*width), sharey=True)
for i in range(len(xpos_list)):
    #col = line_color.next()
    ax[i].plot(ui[:,i]/u_inf, zi, label=str(i))
    SE_mean_eff = np.sqrt(uu[:,i]) / np.sqrt(samp_u[:,i])

    CI_hi = (1.96 * SE_mean_eff + ui[:,i]) / 54.65
    CI_lo = (ui[:,i] - 1.96 * SE_mean_eff) / 54.65
    #ax[i].plot(ui[:,i], zi)
    ax[i].plot(CI_hi, zi, linestyle='--', color='darkgrey')
    ax[i].plot(CI_lo, zi, linestyle='--', color='darkgrey')
    if i == 2:
        ax[i].set_xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[i].grid(True)

#        axes.get_yaxis().set_visible(False)
#        axes.yaxis.set_ticklabels([])
#    plt.grid(True)
#    plt.xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[0].set_ylabel('$z [m]$', labelpad=-4.5)
    ax[i].set_xlim(0.3, 1)
    #plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    #plt.xlim(0,0.05)
    #plt.ylim(-0.1,0.17)
    #plt.legend(loc='best')
    adjustprops = dict(left=0.15, bottom=0.16, right=0.95, top=0.97, wspace=0.1, hspace=0)
    plt.subplots_adjust(**adjustprops)

plt.savefig('samples_umean_'+plane+'.pdf')
plt.savefig('samples_umean_'+plane+'.png', dpi=600)
plt.close()


fig, ax = plt.subplots(1, len(xref_list), figsize=(width,0.6*width), sharey=True)
for i in range(len(xpos_list)):
    #col = line_color.next()
    ax[i].plot(wi[:,i]/u_inf, zi, label=str(i))
    SE_mean_eff = np.sqrt(ww[:,i]) / np.sqrt(samp_w[:,i])

    CI_hi = (1.96 * SE_mean_eff + wi[:,i]) / u_inf
    CI_lo = (wi[:,i] - 1.96 * SE_mean_eff) / u_inf
    #ax[i].plot(ui[:,i], zi)
    ax[i].plot(CI_hi, zi, linestyle='--', color='darkgrey')
    ax[i].plot(CI_lo, zi, linestyle='--', color='darkgrey')
    if i == 2:
        ax[i].set_xlabel('$\overline{w} / u_{\infty}$', labelpad=0)
    ax[i].grid(True)
    ax[i].set_xticks([-0.25, 0])
    ax[i].set_xticklabels([-0.25, 0])

#        axes.get_yaxis().set_visible(False)
#        axes.yaxis.set_ticklabels([])
#    plt.grid(True)
#    plt.xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[0].set_ylabel('$z [m]$', labelpad=-4.5)
    ax[i].set_xlim(-0.4, 0.1)
    #plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    #plt.xlim(0,0.05)
    #plt.ylim(-0.1,0.17)
    #plt.legend(loc='best')
    adjustprops = dict(left=0.15, bottom=0.16, right=0.95, top=0.97, wspace=0.1, hspace=0)
    plt.subplots_adjust(**adjustprops)

plt.savefig('samples_wmean_'+plane+'.pdf')
plt.savefig('samples_wmean_'+plane+'.png', dpi=600)
plt.close()

fig, ax = plt.subplots(1, len(xref_list), figsize=(width,0.6*width), sharey=True)
for i in range(len(xpos_list)):
    ax[i].plot(uu[:,i]/(u_inf**2), zi, label=str(i))
    var_var = (2.0 / (samp_u[:,i]*2)) * uu[:,i]*uu[:,i]
    std_var = np.sqrt(var_var)
    CI_hi = (1.96 * std_var + uu[:,i]) / (54.65*u_inf)
    CI_lo = (uu[:,i] - 1.96 * std_var) / (54.65*u_inf)
    ax[i].plot(CI_hi, zi, linestyle='--', color='darkgrey')
    ax[i].plot(CI_lo, zi, linestyle='--', color='darkgrey')
    if i == 2:
        ax[i].set_xlabel('$\overline{u^{\prime}u^{\prime}} / u_{\infty}^2$', labelpad=1)
    ax[i].grid(True)

#        axes.get_yaxis().set_visible(False)
#        aset_aspectklabels([])
#    plt.grid(True)
#    plt.xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[0].set_ylabel('$z [m]$', labelpad=-4.5)
    ax[i].set_xlim(0.0, 0.08)
    ax[i].set_xticks([0, 0.05])
    ax[i].set_xticklabels([0, 0.05])

    #plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    #plt.xlim(0,0.05)
    #plt.ylim(-0.1,0.17)
    #plt.legend(loc='best')
    adjustprops = dict(left=0.15, bottom=0.16, right=0.95, top=0.97, wspace=0.1, hspace=0)       # Subplot properties
    plt.subplots_adjust(**adjustprops)

plt.savefig('samples_uu_'+plane+'.pdf')
plt.savefig('samples_uu_'+plane+'.png', dpi=600)
plt.close()

fig, ax = plt.subplots(1, len(xref_list), figsize=(width,0.6*width), sharey=True)
for i in range(len(xpos_list)):
    ax[i].plot(ww[:,i]/(u_inf**2), zi, label=str(i))
    var_var = (2.0 / (samp_w[:,i]*2)) * ww[:,i]*ww[:,i]
    std_var = np.sqrt(var_var)
    CI_hi = (1.96 * std_var + ww[:,i]) / (54.65*u_inf)
    CI_lo = (ww[:,i] - 1.96 * std_var) / (54.65*u_inf)
    ax[i].plot(CI_hi, zi, linestyle='--', color='darkgrey')
    ax[i].plot(CI_lo, zi, linestyle='--', color='darkgrey')
    if i == 2:
        ax[i].set_xlabel('$\overline{w^{\prime}w^{\prime}} / u_{\infty}^2$', labelpad=1)
    ax[i].grid(True)
    #ax[i].set_xticks([-3, 0, 3])
    #ax[i].set_xticklabels([-3, 0, 3])

#        axes.get_yaxis().set_visible(False)
#        axes.yaxis.set_ticklabels([])
#    plt.grid(True)
#    plt.xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[0].set_ylabel('$z [m]$', labelpad=-4.5)
    ax[i].set_xlim(0.0, 0.15)
    #plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    #plt.xlim(0,0.05)
    #plt.ylim(-0.1,0.17)
    #plt.legend(loc='best')
    adjustprops = dict(left=0.15, bottom=0.16, right=0.95, top=0.97, wspace=0.1, hspace=0)       # Subplot properties
    plt.subplots_adjust(**adjustprops)

plt.savefig('samples_ww_'+plane+'.pdf')
plt.savefig('samples_ww_'+plane+'.png', dpi=600)
plt.close()

sys.exit()
x0, z0 = xlim[0], -0.05
x1, z1 = xlim[1], 0.25

xi, zi = np.linspace(x0, x1, 200), np.linspace(z0, z1, 250)
xmesh, zmesh = np.meshgrid(xi,zi)

print('shape of xi: ' + str(xi.shape))
print('shape of zi: ' + str(zi.shape))
print('mesh shape: ' + str(xmesh.shape))


print samples
#print ui
#fig,ax  = wt.contour_plot_index(dataset['X'], dataset['Z'],dataset['mean-v'], struct=False)
#plt.show()
samp = scipy.interpolate.griddata((x_WT, z_WT), samples, (xmesh, zmesh), method='cubic')
print samp
fig,ax = plt.subplots(1,1)
color_delta = 1 # (max(data) - min(data)) / 200
minval = 0
maxval = 150
#minval = np.min(samp)
#maxval = np.max(samp)
vals = np.arange(minval, maxval+color_delta, color_delta)

cmap=plt.cm.bone


norm = mpl.colors.Normalize(vmin=minval, vmax=maxval, clip=False)

contour = plt.contourf(xmesh, zmesh, samp, vals, norm=norm, cmap=cmap)
plt.gca().set_aspect('equal')
cbar = wplot.add_colorbar(ax, contour)
cbar.set_ticks(np.linspace(minval,maxval,5))
plt.savefig('indsamp.png', dpi=600)
plt.close()

line = 50
fig, ax = plt.subplots(1,1)
plt.plot(samp[:,line], zi, color='k', label=str(line).replace("_", "\_"))
plt.legend(loc='best')
plt.show()
