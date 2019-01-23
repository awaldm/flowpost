# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import tecplot as tp
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import helpers.wake_stats as ws
import itertools
width = 3.5
from setup_line_plot import *
line_style, markers = setup_plot('JOURNAL_1COL', width, width*2/3)


aoa = {}
x_PMR = 0.90249
z_PMR = 0.14926
u_inf = 54.65
case_name = 'DDES_v38t_dt100md_symm_tau2014'
case_name = 'DDES_v38_dt100md'
case_name = 'DDES_v38h_dt100md_symm_tau2014'

in_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/DDES_dt100_md_CFL2_eigval02_pswitch4_tau2014/sol/slices/CRM_v38h05_WT_DDES_SAO_a18.pval.unsteady_i=7000_t=3.668660000e-01_slc_eta_0.283.plt'
in_path = case_name + '_result.plt'
in_vars = ['X', 'Y', 'Z', 'mean-u', 'mean-v', 'mean-w', 'resolved-uu', 'resolved-uw', 'resolved-ww', 'resolved-vv']

def get_result_file(in_path):

    data = {}
    dataset = tp.data.load_tecplot(in_path, zones=[0], variables = in_vars, read_data_option = tp.constant.ReadDataOption.Replace)
    for zone in dataset.zones():
        for variable in in_vars:
            print variable
    #        variable = "'" +variable + "'"
            print variable
            array = zone.values(variable)
            data[variable] = np.array(array[:]).T
    return data



def rotate_stresses(uu,vv,ww,uv=None, uw=None, vw=None, x_PMR=None, z_PMR=None, alpha=18.):
    uu_WT =  uu * (np.cos(-1*np.radians(alpha)))**2 - 2 * uw * np.sin(-1*np.radians(alpha)) * np.cos(-1*np.radians(alpha)) + uw * (-1*np.radians(alpha))**2
    return uu_WT



import main.wake.funcs.plotters as wplot


plane = 'eta0603'
if plane is 'eta0201':
    xlim = (1, 1.6)
elif plane is 'eta0283':
    xlim = (1.05, 1.65)
elif plane is 'eta0603':
    xlim = (1.15, 1.75)
elif plane is 'eta0131':
    xlim = (1, 1.6)


###################
# load 3 cases

files = dict()

#files['DDES_v38_dt100md'] = './results/DDES_v38_dt100_md_cK_eta0283_rstresses_28500_31800.plt'
#files['DDES_v38h_dt100md_symm_tau2014'] = './results/DDES_v38h_dt100_md_CFL2_eigval02_pswitch1_symm_tau2014_eta0283_rstresses_1000_5600.plt'
#files['DDES_v38t_dt100md_symm_tau2014'] = './results/DDES_v38t_dt100md_symm_tau2014_eta0283_rstresses_1100_2800.plt'
label1 = 'AoA16'
label2 = 'AoA18'
label3 = 'AoA20'
label1 = r'$\alpha=16^{\circ}$'
label2 = r'$\alpha=18^{\circ}$'
label3 = r'$\alpha=20^{\circ}$'

files[label1] = './results/CRM_v38h_AoA16_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_'+plane+'_rstresses_3000_10400.plt'
files[label1] = './results/CRM_v38h_AoA16_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_'+plane+'_rstresses_5000_12000.plt'
files[label2] = './results/CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2_'+plane+'_rstresses_5000_13000.plt'
files[label2] = './results/CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2_'+plane+'_rstresses_5000_14400.plt'
files[label3] = './results/CRM_v38h_AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_k1024_'+plane+'_rstresses_5000_12800.plt'
files[label3] = './results/CRM_v38h_AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_k1024_'+plane+'_rstresses_5000_15000.plt'

aoa[label1] = 16.
aoa[label2] = 18.
aoa[label3] = 20.
dataset={}
t={}
CL={}

for case, path in files.iteritems():
    dataset[case] = get_result_file(path)


#case_name = 'DDES_v38t_dt100md_symm_tau2014'
#case_name = 'DDES_v38_dt100md'
#case_name = 'DDES_v38h_dt100md_symm_tau2014'

case_name = 'DDES_v38_dt100md'
case_name ='CRM_DDES_v38h_3AoA'

#data_v38 = get_result_file('./results/DDES_v38_dt100_md_cK_eta0283_rstresses_28500_31800.plt')
#case_name = 'DDES_v38h_dt100md_symm_tau2014'
#data_v38h = get_result_file('./results/DDES_v38h_dt100_md_CFL2_eigval02_pswitch1_symm_tau2014_eta0283_rstresses_1000_5600.plt')
#case_name = 'DDES_v38t_dt100md_symm_tau2014'
#data_v38t = get_result_file('./results/DDES_v38t_dt100md_symm_tau2014_eta0283_rstresses_1100_2800.plt')

#case_name = 'DDES_v38h_dt50md_symm_tau2014'
#data_v38h_dt50 = get_result_file(case_name + '_'+plane+ '_result.plt')

#case_name = 'DDES_v38t05_dt100md_symm_tau2014'
#data_v38t05 = get_result_file(case_name + '_'+plane+ '_result.plt')

x_WT = {}
z_WT = {}
u_WT = {}
w_WT = {}
uu_WT = {}
vv_WT = {}
ww_WT = {}
uv_WT = {}
uw_WT = {}
vw_WT = {}
for case, values in dataset.iteritems():
    x_WT[case], z_WT[case] = ws.transform_wake_coords(values['X'], values['Z'], x_PMR, z_PMR, aoa[case])
    u_WT[case], w_WT[case] = ws.rotate_velocities(values['mean-u'], values['mean-v'], values['mean-w'], x_PMR, z_PMR, aoa[case])
    uu_WT[case], _, ww_WT[case], uv_WT[case], uw_WT[case], vw_WT[case] = ws.rotate_stresses(values['resolved-uu'],None,values['resolved-ww'], uw=values['resolved-uw'], x_PMR=x_PMR, z_PMR=z_PMR, alpha=aoa[case])
    vv_WT[case] = values['resolved-vv']
# end loading 3 cases
#######################


x0, z0 = xlim[0], -0.05
x1, z1 = xlim[1], 0.25

xi, zi = np.linspace(x0, x1, 100), np.linspace(z0, z1, 250)
xmesh, zmesh = np.meshgrid(xi,zi)

print('shape of xi: ' + str(xi.shape))
print('shape of zi: ' + str(zi.shape))
print('mesh shape: ' + str(xmesh.shape))


ui = {}
wi = {}
uu = {}
vv = {}
ww = {}
uw = {}
for case, _ in files.iteritems():
    ui[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), u_WT[case], (xmesh, zmesh), method='cubic')
    wi[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), w_WT[case], (xmesh, zmesh), method='cubic')
    uu[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), uu_WT[case], (xmesh, zmesh), method='cubic')
    vv[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), vv_WT[case], (xmesh, zmesh), method='cubic')
    ww[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), ww_WT[case], (xmesh, zmesh), method='cubic')
    uw[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), uw_WT[case], (xmesh, zmesh), method='cubic')

print('shape of interpolated ui: ' + str(ui[label1].shape))

umin = {}
us = {}
pos = {}

###############################################################################
# Plot: position of wake center (z location of us) over x for all cases
###############################################################################

fig, ax = plt.subplots(1, 1)
line_color = itertools.cycle(["k", "b", "r", "g"])
for case, _ in files.iteritems():
    pos = np.argmin(np.where(ui[case]==0, ui[case].max(), ui[case]), axis=0)
    print('shape of xi:' + str(xi.shape))
    print('shape of zi:' + str(zi[pos].shape))
    plt.plot(xi, zi[pos], color=line_color.next(), label=case) # , linestyle=line_style.next(), marker=markers.next(), mew=1, ms=2, markevery=10)
    np.savez(case+'_' + plane + '_wake_min_pos', xi=xi, zi=zi[pos])

plt.legend(loc='best')

image_name = case_name+'_wake_center'
#    print("exporting image " + image_name)
plt.ylim(-0.05, 0.25)
fig.savefig(image_name + '.png', dpi=600)
plt.close(fig)



##################################
# compute the maximum and the vertically averaged flow angle (time average)
def get_wake_grid_flow_angle(u, w):
    angle = np.arctan2(w,u) * 180 / np.pi
    return angle
angle = {}
angmin = {}
angmean = {}
for case, _ in files.iteritems():
    angle[case] = get_wake_grid_flow_angle(ui[case], wi[case])
    angmin[case] = np.min(angle[case], axis=0)
    angmean[case] = np.mean(angle[case], axis=0)

line_style, markers = setup_plot('JOURNAL_1COL', width, width*2./3.)

fig, ax = plt.subplots(1,1)
for case, _ in files.iteritems():
    col = line_color.next()
    mark = markers.next()
    plt.plot(xi, angmin[case], color=col, label=case.replace("_", "\_"), marker=mark, mew=1, ms=2, markevery=10)


plt.legend(loc='best')
plt.ylim(-25,0)
plt.savefig('angmin_'+plane+'.pdf')
plt.close()

labels = {}
#labels[] = 'Grid A, $\Delta t_{red} = 0.01$'
line_style, markers = setup_plot('JOURNAL_1COL', width, width*2./3.)

fig, ax = plt.subplots(1,1)
line_color = itertools.cycle(["k", "b", "r", "g"])

for case in sorted(files):
    col = line_color.next()
    mark = markers.next()
    plt.plot(xi, angmean[case], color=col, label=case, marker=mark, mew=1, ms=2, markevery=10)
ax.set_xlim(xlim)
ax.set_ylim(-20, 5)
plt.xlabel('$x [m]$', labelpad=-1)
plt.ylabel('$\gamma [^{\circ}]$', labelpad=-1)

plt.legend(loc='best')
adjustprops = dict(left=0.12, bottom=0.13, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
plt.subplots_adjust(**adjustprops)
plt.grid(True)
plt.savefig('angave_'+plane+'.pdf')


import main.wake.funcs.wake_structure as wstruct


###############################################################################
# Plot: position of wake center over x for all cases
###############################################################################

line_style, markers = setup_plot('JOURNAL_1COL', width, width*2./3.)
fig, ax = plt.subplots(1,1)
import matplotlib.tri as tri
line_color = itertools.cycle(["k", "b", "r", "g"])


for case in sorted(files):
    z_upper, z_lower, z_center, umin = wstruct.compute_wake_scale(ui[case], xi, zi)
    u_inf = 54.65
    half = ((u_inf - umin) / 2) + umin
    col = line_color.next()
    mark = markers.next()
    #ax.plot(xi, savgol_filter(z[center_pos], 15, 2), label='pos of minimum')
    ax.plot(xi, z_center, color=col, label=case, marker=mark, mew=1, ms=2, markevery=10)

adjustprops = dict(left=0.12, bottom=0.13, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
plt.subplots_adjust(**adjustprops)

ax.set_xlim(xlim)
plt.legend(loc='best')
plt.savefig(case_name + '_wake_center_pos_'+plane+'.png', dpi=600)
plt.savefig(case_name + '_wake_center_pos_'+plane+'.pdf')


###############################################################################
# Plot: position of minimum velocity
###############################################################################

line_style, markers = setup_plot('JOURNAL_1COL', width, width*2./3.)
fig, ax = plt.subplots(1,1)
import matplotlib.tri as tri
line_color = itertools.cycle(["k", "b", "r", "g"])


for case in sorted(files):
    z_upper, z_lower, z_center, umin = wstruct.compute_wake_scale(ui[case], xi, zi)
    u_inf = 54.65
    mark = markers.next()
    col = line_color.next()
    ax.plot(xi, umin/u_inf, color=col, label=case, marker=mark, mew=1, ms=2, markevery=10)

adjustprops = dict(left=0.12, bottom=0.13, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
plt.subplots_adjust(**adjustprops)
plt.xlabel('$x [m]$', labelpad=-1)
plt.ylabel('$z [m]$', labelpad=-3)

#plt.xlim(1,1.63)
ax.set_xlim(xlim)
plt.ylim(0, 1)

plt.legend(loc='best')
plt.savefig(case_name + '_wake_umin_'+plane+'.png', dpi=600)
plt.savefig(case_name + '_wake_umin_'+plane+'.pdf')


###############################################################################
# Plot: position of (upper and lower) half-width ls over x for all cases
###############################################################################

line_style, markers = setup_plot('JOURNAL_1COL', width, width*2./3.)
fig, ax = plt.subplots(1,1)
import matplotlib.tri as tri
line_color = itertools.cycle(["k", "b", "r", "g"])


for case in sorted(files):
    z_upper, z_lower, z_center, umin = wstruct.compute_wake_scale(ui[case], xi, zi)
    u_inf = 54.65
    mark = markers.next()
    half = ((u_inf - umin) / 2) + umin
    col = line_color.next()
    #ax.plot(xi, z_center, label=case, color=col)
    #ax[2].plot(x, savgol_filter(z[center_pos], 15, 2), label='pos of minimum')
    ax.plot(xi, z_upper, color=col, label=case, marker=mark, mew=1, ms=2, markevery=10)
    ax.plot(xi, z_lower, color=col, marker=mark, mew=1, ms=2, markevery=10)
    '''
    x_wing, z_wing, x_HTP, z_HTP = wplot.get_wing_HTP_cuts(plane=plane, rotated=aoa[case])
    triang = tri.Triangulation(x_wing,z_wing)
    plt.tricontourf(triang, x_wing, colors='black')
    triang = tri.Triangulation(x_HTP,z_HTP)
    plt.tricontourf(triang, x_HTP, colors='black')
    '''

adjustprops = dict(left=0.12, bottom=0.13, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
plt.subplots_adjust(**adjustprops)
plt.xlabel('$x [m]$', labelpad=-1)
plt.ylabel('$z [m]$', labelpad=-3)

ax.set_xlim(xlim)
plt.legend(loc='best')
plt.savefig(case_name + '_wake_halfwidth_pos_'+plane+'.png', dpi=600)
plt.savefig(case_name + '_wake_halfwidth_pos_'+plane+'.pdf')


###############################################################################
# Plot: average half-width ls over x for all cases
###############################################################################

#line_style, markers = setup_plot('JOURNAL_1COL', width, width*2./3.)
#fig, ax = plt.subplots(1,1)
#import matplotlib.tri as tri
#line_color = itertools.cycle(["k", "b", "r", "g"])


#for case in sorted(files):
#    z_upper, z_lower, z_center, umin = wstruct.compute_wake_scale(ui[case], xi, zi)
#    u_inf = 54.65
#    half = ((u_inf - umin) / 2) + umin
#    col = line_color.next()
#    halfwidth = (z_upper + z_lower) / 2.0
#    ax.plot(xi, halfwidth, color=col, label=case)


#plt.xlim(1.05, 1.60)
#plt.legend(loc='best')
#plt.grid(True)
#plt.savefig(case_name + '_wake_average_halfwidth_pos_'+plane+'.png', dpi=600)
#plt.savefig(case_name + '_wake_average_halfwidth_pos_'+plane+'.pdf')



###############################################################################
# Plot: average half-width ls over x for all cases
###############################################################################

fig, ax = plt.subplots(1,1)
line_color = itertools.cycle(["k", "b", "r", "g"])
for case in sorted(files):
    mark = markers.next()
    lsmean = wstruct.compute_halfwidth(ui[case], xi, zi, u_inf = 54.65, asymm=False)
    col = line_color.next()
    ax.plot(xi, lsmean, color=col, label=case, marker=mark, mew=1, ms=2, markevery=10)

adjustprops = dict(left=0.12, bottom=0.13, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
plt.subplots_adjust(**adjustprops)
plt.xlabel('$x [m]$', labelpad=-1)
plt.ylabel('$l_s [m]$', labelpad=-3)

ax.set_xlim(xlim)
ax.set_ylim(0, 0.1)
plt.grid(True)
plt.legend(loc='best')
plt.savefig(case_name + '_wake_avg_halfwidth_'+plane+'.png', dpi=600)
plt.savefig(case_name + '_wake_avg_halfwidth_'+plane+'.pdf')


###############################################################################
# Plot: wake deficit us over x for all cases
###############################################################################

line_style, markers = setup_plot('JOURNAL_1COL', width, width*2/3)

fig, ax = plt.subplots(1,1)
line_color = itertools.cycle(["k", "b", "r", "g"])
for case in sorted(files):
    col = line_color.next()
    mark = markers.next()
    umin = np.min(ui[case], axis=0)
    us = 54.65 - umin
    ax.plot(xi, us / 54.65, color=col, label=case, marker=mark, mew=1, ms=2, markevery=10)
adjustprops = dict(left=0.13, bottom=0.13, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
plt.subplots_adjust(**adjustprops)
plt.xlabel('$x [m]$', labelpad=-1)
plt.ylabel('$u_d / \overline{u_{\infty}}$', labelpad=-1)

ax.set_xlim(xlim)
plt.grid(True)
plt.ylim(0, 1.5)
plt.legend(loc='best')
plt.savefig(case_name + '_wake_deficit_'+plane+'.png', dpi=600)
plt.savefig(case_name + '_wake_deficit_'+plane+'.pdf')


###############################################################################
# Plot: max magnitude of uu over x for all cases
###############################################################################

line_style, markers = setup_plot('JOURNAL_1COL', width, width*2/3)
fig, ax = plt.subplots(1,1)
line_color = itertools.cycle(["k", "b", "r", "g"])
for case in sorted(files):
    col = line_color.next()
    mark = markers.next()
    uumax = np.max(uu[case], axis=0)
    ax.plot(xi, uumax / (54.65*54.65), color=col, label=case, marker=mark, mew=1, ms=2, markevery=10)

adjustprops = dict(left=0.15, bottom=0.13, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
plt.subplots_adjust(**adjustprops)
plt.xlabel('$x [m]$', labelpad=-1)
plt.ylabel('$\overline{u^{\prime}u^{\prime}/\overline{u_{\infty}}}_{max}$', labelpad=0)

ax.set_xlim(xlim)
plt.grid(True)
plt.ylim(0, 0.15)
plt.legend(loc='best')
plt.savefig(case_name + '_uumax_'+plane+'.png', dpi=600)
plt.savefig(case_name + '_uumax_'+plane+'.pdf')

###############################################################################
# Plot: magnitude of uw over x for all cases
###############################################################################

line_style, markers = setup_plot('JOURNAL_1COL', width, width*2/3)
fig, ax = plt.subplots(1,1)
line_color = itertools.cycle(["k", "b", "r", "g"])
for case in sorted(files):
    col = line_color.next()
    mark = markers.next()
    uwmin = np.min(uw[case], axis=0)
    uwmax = np.max(uw[case], axis=0)
    ax.plot(xi, uwmin / (54.65*54.65), color=col, linestyle='--', marker=mark, mew=1, ms=2, markevery=10)
    ax.plot(xi, uwmax / (54.65*54.65), color=col, label=case, marker=mark, mew=1, ms=2, markevery=10)

adjustprops = dict(left=0.15, bottom=0.13, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
plt.subplots_adjust(**adjustprops)
plt.xlabel('$x [m]$', labelpad=-1)
plt.ylabel('$\overline{u^{\prime}w^{\prime}/\overline{u_{\infty}}}_{max}$', labelpad=-5)
plt.grid(True)
ax.set_xlim(xlim)
#plt.ylim(-3, 0)
plt.legend(loc='best')
plt.savefig(case_name + '_uwmax_'+plane+'.png', dpi=600)
plt.savefig(case_name + '_uwmax_'+plane+'.pdf')

###############################################################################
# Plot: magnitude of uw/k over x for all cases
###############################################################################

line_style, markers = setup_plot('JOURNAL_1COL', width, width*2/3)
fig, ax = plt.subplots(1,1)
line_color = itertools.cycle(["k", "b", "r", "g"])
for case in sorted(files):
    col = line_color.next()
    mark = markers.next()
    uwmin = np.min(uw[case], axis=0)
    indmin =  np.argmin(uw[case], axis=0)
    uwmax = np.max(uw[case], axis=0)
    indmax = np.argmax(uw[case], axis=0)
    print('shape of selection: ' + str(uu[case][indmax, np.arange(len(indmax))].shape))
    kt_uwmin = 0.5 * (uu[case][indmin, np.arange(len(indmin))] +vv[case][indmin, np.arange(len(indmin))] + ww[case][indmin, np.arange(len(indmin))])
    kt_uwmax = 0.5 * (uu[case][indmax, np.arange(len(indmax))] +vv[case][indmax, np.arange(len(indmax))] + ww[case][indmax, np.arange(len(indmax))])
    print('shape of uwmax: ' + str(uwmax.shape))
    print('shape of indmax: ' + str(indmax.shape))
    print('shape of kt_uwmax: ' + str(kt_uwmax.shape))
    ax.plot(xi, uwmin / (kt_uwmin), color=col, linestyle='--', marker=mark, mew=1, ms=2, markevery=10)
    ax.plot(xi, uwmax / (kt_uwmax), color=col, label=case, marker=mark, mew=1, ms=2, markevery=10)

adjustprops = dict(left=0.13, bottom=0.13, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
plt.subplots_adjust(**adjustprops)
plt.xlabel('$x [m]$', labelpad=-1)
plt.ylabel('$\overline{u^{\prime}w^{\prime}/{k_t}$', labelpad=-5)

ax.set_xlim(xlim)
plt.grid(True)
#plt.ylim(-3, 0)
plt.legend(loc='best')
plt.savefig(case_name + '_uwmax_kt_'+plane+'.png', dpi=600)
plt.savefig(case_name + '_uwmax_kt_'+plane+'.pdf')


###############################################################################
# Plot: compute wake properties for each case separately
###############################################################################
separate = False
if separate:

    line_color = itertools.cycle(["k", "b", "r", "g"])
    for case, _ in files.iteritems():
        #data=np.load(case + '_' + plane + '_wake_center_linear_fit.npz')
        #x = data['x']
        zcenter = data['z']
        col = line_color.next()
        wstruct.plot_wake_strength(ui[case], xi, zi, case+'_'+plane)
        #wstruct.plot_wake_scale(ui[case], xi, zi, case+'_'+plane, fitcenter = zcenter)
        wstruct.plot_wake_center(ui[case], xi, zi, case+'_'+plane)
