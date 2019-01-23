# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import tecplot as tp
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import helpers.wake_stats as ws

width = 1.75
from setup_line_plot import *
line_style, markers = setup_plot('JOURNAL_1COL', width, width*3/2)


aoa = {}
x_PMR = 0.90249
z_PMR = 0.14926
u_inf = 54.65
in_vars = ['X', 'Y', 'Z', 'mean-u', 'mean-v', 'mean-w', 'resolved-uu', 'resolved-uw', 'resolved-ww', 'resolved-vv']

def get_result_file(in_path):

    data = {}
    dataset = tp.data.load_tecplot(in_path, zones=[0], variables = in_vars, read_data_option = tp.constant.ReadDataOption.Replace)
    for zone in dataset.zones():
        for variable in in_vars:
            #print variable
    #        variable = "'" +variable + "'"
            #print variable
            array = zone.values(variable)
            data[variable] = np.array(array[:]).T
    return data



def rotate_stresses(uu,vv,ww,uv=None, uw=None, vw=None, x_PMR=None, z_PMR=None, alpha=18.):
    uu_WT =  uu * (np.cos(-1*np.radians(alpha)))**2 - 2 * uw * np.sin(-1*np.radians(alpha)) * np.cos(-1*np.radians(alpha)) + ww * np.sin(-1*np.radians(alpha))**2
    ww_WT =  uu * (np.sin(-1*np.radians(alpha)))**2 + 2 * uw * np.sin(-1*np.radians(alpha)) * np.cos(-1*np.radians(alpha)) + ww * np.cos(-1*np.radians(alpha))**2
#  EQUATION = '{MSMT_uu} = {resolved-uu}*(COS(-|alpha|/180*PI))**2 -2*{resolved-uw}*SIN(-|alpha|/180*PI)*COS(-|alpha|/180*PI) + {resolved-ww}*(SIN(-|alpha|/180*PI))**2'
#  EQUATION = '{MSMT_ww} = {resolved-uu}*(SIN(-|alpha|/180*PI))**2 +2*{resolved-uw}*SIN(-|alpha|/180*PI)*COS(-|alpha|/180*PI) + {resolved-ww}*(COS(-|alpha|/180*PI))**2'

    return uu_WT, ww_WT



import main.wake.funcs.plotters as wplot


plane = 'eta0131'

x_wing, z_wing, x_HTP, z_HTP = wplot.get_wing_HTP_cuts(plane=plane)


###################
files = dict()

label1 = 'AoA16'
label2 = 'AoA18'
label3 = 'AoA20'
files['AoA16'] = './results/CRM_v38h_AoA16_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_'+plane+'_rstresses_2500_9400.plt'
files['AoA18'] = './results/CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2_'+plane+'_rstresses_2000_8400.plt'
files['AoA20'] = './results/CRM_v38h_AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_k1024_'+plane+'_rstresses_5000_10800.plt'

aoa[label1] = 16.
aoa[label2] = 18.
aoa[label3] = 20.

dataset={}
t={}
CL={}

for case, path in files.iteritems():
    dataset[case] = get_result_file(path)
#######################

##################################
# compute the maximum and the vertically averaged flow angle (time average)
def get_wake_grid_flow_angle(u, w):
    angle = np.arctan2(w,u) * 180 / np.pi
    return angle

x_WT = {}
z_WT = {}
u_WT = {}
w_WT = {}
uu_WT = {}
vv_WT = {}
ww_WT = {}

for case, values in dataset.iteritems():
    x_WT[case], z_WT[case] = ws.transform_wake_coords(values['X'], values['Z'], x_PMR, z_PMR, aoa[case])
    u_WT[case], w_WT[case] = ws.rotate_velocities(values['mean-u'], values['mean-v'], values['mean-w'], x_PMR, z_PMR, aoa[case])
    uu_WT[case], ww_WT[case] = rotate_stresses(values['resolved-uu'],values['resolved-vv'],values['resolved-ww'], uw=values['resolved-uw'], x_PMR=x_PMR, z_PMR=z_PMR, alpha=aoa[case])
    vv_WT[case] = values['resolved-vv']

xpos_list = [1.2, 1.3, 1.4, 1.5, 1.6]

def get_line(xpos, x_WT, z_WT, u_WT, w_WT, uu_WT, vv_WT, ww_WT, num=500):
    #xpos = 1.45
    x0, z0 = xpos, -0.15
    x1, z1 = xpos, 0.25

    xi, zi = np.linspace(x0, x1, num), np.linspace(z0, z1, num)

    ui = {}
    wi = {}
    uu = {}
    vv = {}
    ww = {}
    for case, _ in files.iteritems():
        ui[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), u_WT[case], (xi, zi), method='cubic')
        wi[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), w_WT[case], (xi, zi), method='cubic')
        uu[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), uu_WT[case], (xi, zi), method='cubic')
        vv[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), vv_WT[case], (xi, zi), method='cubic')
        ww[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), ww_WT[case], (xi, zi), method='cubic')
    return xi, zi, ui, wi, uu, vv, ww

def plot_stations(xi, zi, ui, wi, uu, vv, ww, xpos, plane):

    #label1 = r'$\Delta t = 0.02t_{\infty}$'
    #label2 = r'$\Delta t = 0.01t_{\infty}$'
    angle = {}
    kt = {}
    for case, _ in files.iteritems():
        angle[case] = get_wake_grid_flow_angle(ui[case], wi[case])
        kt[case] = 0.5*(uu[case] + vv[case] + ww[case])



    fig, ax = plt.subplots(1, 1)
    line_color = itertools.cycle(["k", "b", "r", "g"])
    #for case, _ in files.iteritems():
    for case in sorted(files):
        col = line_color.next()
        plt.plot(angle[case], zi, color=col, label=case.replace("_", "\_"))

    plt.legend(loc='best')
    plt.ylim(-0.1,0.17)
    plt.xlabel('$\gamma [^{\circ}]$', labelpad=0)
    plt.ylabel('$z [m]$', labelpad=-4.5)
    adjustprops = dict(left=0.25, bottom=0.11, right=0.95, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
    plt.subplots_adjust(**adjustprops)
    plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)

    plt.savefig('angle_plane_'+str(xpos).replace(".","")+'_'+plane+'.pdf')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    line_color = itertools.cycle(["k", "b", "r", "g"])
    for case in sorted(files):
        col = line_color.next()
        plt.plot(ui[case] / u_inf, zi, color=col, label=case.replace("_", "\_"))
    plt.xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    plt.ylabel('$z [m]$', labelpad=-4.5)
    plt.legend(loc='best')
    adjustprops = dict(left=0.25, bottom=0.11, right=0.95, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
    plt.subplots_adjust(**adjustprops)
    plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    plt.ylim(-0.1,0.17)
    plt.xlim(0.6, 1)
    plt.savefig('umean_plane_'+str(xpos).replace(".","")+'_'+plane+'.pdf')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    line_color = itertools.cycle(["k", "b", "r", "g"])
    for case in sorted(files):
        col = line_color.next()
        plt.plot(uu[case] / (u_inf*u_inf), zi, color=col, label=case.replace("_", "\_"))
    plt.xlabel('$\overline{u^{\prime}u^{\prime}} / u_{\infty}^2$', labelpad=0)
    plt.ylabel('$z [m]$', labelpad=-4.5)

    plt.ylim(-0.1,0.17)
    plt.legend(loc='best')
    adjustprops = dict(left=0.25, bottom=0.11, right=0.95, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
    plt.subplots_adjust(**adjustprops)
    plt.xlim(0,0.03)
    plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    plt.savefig('uu_plane_'+str(xpos).replace(".","")+'_'+plane+'.pdf')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    line_color = itertools.cycle(["k", "b", "r", "g"])
    for case in sorted(files):
        col = line_color.next()
        plt.plot(ww[case] / (u_inf*u_inf), zi, color=col, label=case.replace("_", "\_"))
    plt.xlabel('$\overline{w^{\prime}w^{\prime}} / u_{\infty}^2$', labelpad=0)
    plt.ylabel('$z [m]$', labelpad=-4.5)
    plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    plt.xlim(0,0.05)
    plt.ylim(-0.1,0.17)
    plt.legend(loc='best')
    adjustprops = dict(left=0.25, bottom=0.11, right=0.95, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
    plt.subplots_adjust(**adjustprops)

    plt.savefig('ww_plane_'+str(xpos).replace(".","")+'_'+plane+'.pdf')
    plt.close()

    fig, ax = plt.subplots(1, 1)
    line_color = itertools.cycle(["k", "b", "r", "g"])
    for case in sorted(files):
        col = line_color.next()
        plt.plot(kt[case] / (u_inf*u_inf), zi, color=col, label=case.replace("_", "\_"))
    plt.xlabel('$k_t / u_{\infty}^2$', labelpad=0)
    plt.ylabel('$z [m]$', labelpad=-4.5)
    plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    plt.xlim(0,0.15)
    plt.ylim(-0.1,0.23)
    plt.legend(loc='best')
    adjustprops = dict(left=0.25, bottom=0.11, right=0.95, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
    plt.subplots_adjust(**adjustprops)

    plt.savefig('kt_plane_'+str(xpos).replace(".","")+'_'+plane+'.pdf')
    plt.close()

for xpos in xpos_list:
    print(xpos)
    xi, zi, ui, wi, uu, vv, ww = get_line(xpos, x_WT, z_WT, u_WT, w_WT, uu_WT, vv_WT, ww_WT)

    plot_stations(xi, zi, ui, wi, uu, vv, ww, xpos, plane)
