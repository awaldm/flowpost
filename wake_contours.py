# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import tecplot as tp
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

import helpers.wake_stats as ws

width = 3.5
from setup_line_plot import *
line_style, markers = setup_plot('JOURNAL_1COL', width, width*2/3)


alpha = 16.
x_PMR = 0.90249
z_PMR = 0.14926
u_inf = 54.65
plane = 'eta0283'
case_name = 'DDES_v38t_dt100md_symm_tau2014'
#case_name = 'DDES_v38_dt100md'
case_name = 'DDES_v38h_dt100md_symm_tau2014'
#case_name = 'DDES_v38h_dt50md_symm_tau2014'
#case_name = 'DDES_v38t05_dt100md_symm_tau2014'
in_path = case_name + '_' + plane + '_result.plt'
in_vars = ['X', 'Y', 'Z', 'mean-u', 'mean-v', 'mean-w', 'resolved-uu', 'resolved-uw', 'resolved-ww']
case_name = 'CRM_v38h_AoA16_Re172_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_eta0201'
#in_path = './results/' + case_name + '_' + plane + '_result.plt'
in_path = './results/CRM_v38h_AoA16_Re172_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_eta0201_rstresses_2000_4600.plt'
data = {}

dataset = tp.data.load_tecplot(in_path, zones=[0], variables = in_vars, read_data_option = tp.constant.ReadDataOption.Replace)
for zone in dataset.zones():
    for variable in in_vars:
        print variable
#        variable = "'" +variable + "'"
        print variable
        array = zone.values(variable)
        data[variable] = np.array(array[:]).T


#    u_WT = u*np.cos(-1*np.radians(alpha)) - w * np.sin(-1*np.radians(alpha))

#    uu_WT =  uu * (np.cos(-1*np.radians(alpha)))**2 - 2 * uw * np.sin(-1*np.radians(alpha)) * np.cos(-1*np.radians(alpha)) + uw * (-1*np.radians(alpha))**2
  #EQUATION = '{MSMT_uu} = {resolved-uu}*(COS(-|alpha|/180*PI))**2 -2*{resolved-uw}*SIN(-|alpha|/180*PI)*COS(-|alpha|/180*PI) + {resolved-ww}*(SIN(-|alpha|/180*PI))**2'


def rotate_stresses(uu,vv,ww,uv=None, uw=None, vw=None, x_PMR=None, z_PMR=None, alpha=alpha):
    uu_WT =  uu * (np.cos(-1*np.radians(alpha)))**2 - 2 * uw * np.sin(-1*np.radians(alpha)) * np.cos(-1*np.radians(alpha)) + uw * (-1*np.radians(alpha))**2
    return uu_WT



import main.wake.funcs.plotters as wplot
x_wing, z_wing, x_HTP, z_HTP = wplot.get_wing_HTP_cuts(plane=plane)

x_WT, z_WT = ws.transform_wake_coords(data['X'], data['Z'], x_PMR, z_PMR, alpha)
u_WT, w_WT = ws.rotate_velocities(data['mean-u'], data['mean-v'], data['mean-w'], x_PMR, z_PMR, alpha)
uu_WT = rotate_stresses(data['resolved-uu'],None,data['resolved-ww'], uw=data['resolved-uw'], x_PMR=x_PMR, z_PMR=z_PMR, alpha=alpha)


#fig, ax = wplot.plot_velocity(x, z, uw, kind='CRM_wake', shape=None, struct=False, v=None, streamlines=False)
fig, ax = plt.subplots(1,1)
cmap = plt.cm.hot_r

minval = 0
maxval = 0.15
color_delta = 0.001 # 0.001
#minval = np.amin(data)
#maxval = np.amax(data)
vals = np.arange(minval, maxval+color_delta, color_delta)
import matplotlib.colors as mpcolors
norm = mpcolors.Normalize(vmin=minval, vmax=maxval)
contour = plt.tricontourf(x_WT, z_WT, uu_WT/(u_inf * u_inf), vals, norm=norm, cmap=cmap)

import matplotlib.tri as tri
triang = tri.Triangulation(x_wing,z_wing)
plt.tricontourf(triang, x_wing, colors='black')
triang = tri.Triangulation(x_HTP,z_HTP)
plt.tricontourf(triang, x_HTP, colors='black')

plt.savefig(case_name+'_'+plane+'_wake_uu.png', dpi=600)
plt.close()

