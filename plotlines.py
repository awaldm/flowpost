# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import tecplot as tp
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


alpha = 18.
x_PMR = 0.90249
z_PMR = 0.14926
u_inf = 54.65
case_name = 'DDES_v38t_dt100md_symm_tau2014'
case_name = 'DDES_v38_dt100md'
in_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/v38_h05/DDES_dt100_md_CFL2_eigval02_pswitch4_tau2014/sol/slices/CRM_v38h05_WT_DDES_SAO_a18.pval.unsteady_i=7000_t=3.668660000e-01_slc_eta_0.283.plt'
in_path = 'DDES_v38t_dt100md_symm_tau2014_result.plt'
in_vars = ['X', 'Y', 'Z', 'mean-u', 'mean-v', 'mean-w', 'resolved-uu', 'resolved-uw', 'resolved-ww']

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


def rotate_stresses(uu,vv,ww,uv=None, uw=None, vw=None, x_PMR=None, z_PMR=None, alpha=18.):
    uu_WT =  uu * (np.cos(-1*np.radians(alpha)))**2 - 2 * uw * np.sin(-1*np.radians(alpha)) * np.cos(-1*np.radians(alpha)) + uw * (-1*np.radians(alpha))**2
    return uu_WT



import main.wake.funcs.plotters as wplot

xslice, yslice, zslice = wplot.get_slice_boundaries(plane='eta0283')
xslice, zslice = transform_wake_coords(xslice, zslice, x_PMR, z_PMR, alpha)
zslice = np.delete(zslice,np.where(xslice>1.2))
yslice = np.delete(yslice,np.where(xslice>1.2))
xslice = np.delete(xslice,np.where(xslice>1.2))

x_WT, z_WT = transform_wake_coords(data['X'], data['Z'], x_PMR, z_PMR, alpha)
u_WT, w_WT = rotate_velocities(data['mean-u'], data['mean-v'], data['mean-w'], x_PMR, z_PMR, alpha)
uu_WT = rotate_stresses(data['resolved-uu'],None,data['resolved-ww'], uw=data['resolved-uw'], x_PMR=x_PMR, z_PMR=z_PMR, alpha=18.)


#fig, ax = wplot.plot_velocity(x, z, uw, kind='CRM_wake', shape=None, struct=False, v=None, streamlines=False)
fig, ax = plt.subplots(1,1)
cmap = plt.cm.hot_r

minval = 0
maxval = 0.3
color_delta = 0.001 # 0.001
#minval = np.amin(data)
#maxval = np.amax(data)
vals = np.arange(minval, maxval+color_delta, color_delta)
import matplotlib.colors as mpcolors
norm = mpcolors.Normalize(vmin=minval, vmax=maxval)
contour = plt.tricontourf(x_WT, z_WT, uu_WT/(u_inf * u_inf), vals, norm=norm, cmap=cmap)

import matplotlib.tri as tri
triang = tri.Triangulation(xslice,zslice)
plt.tricontourf(triang, xslice, colors='white')
plt.savefig('wake_uw.png', dpi=600)
plt.close()
sys.exit(0)
################ end slices

fig, ax = wplot.plot_velocity(x, z, uu, kind='CRM_wake', shape=None, struct=False, v=None, streamlines=False)
plt.savefig('wake_uu.png', dpi=600)
plt.close()
sys.exit(0)
# the line coordinates onto which to interpolate
x0, z0 = 1.4, -0.05
x1, z1 = 1.4, 0.25
num = 300

print('creating line using ' + str(num) + ' points...')
xi, zi = np.linspace(x0, x1, num), np.linspace(z0, z1, num)

ui = scipy.interpolate.griddata((x_WT, z_WT), uu_WT, (xi, zi), method='cubic')

fig,ax = plt.subplots(1,1)

plt.plot(ui/(u_inf*u_inf), zi)
plt.savefig('wake_line14.png', dpi=600)

x0, z0 = 1.5, -0.05
x1, z1 = 1.5, 0.25
num = 300

print('creating line using ' + str(num) + ' points...')
xi, zi = np.linspace(x0, x1, num), np.linspace(z0, z1, num)

ui = scipy.interpolate.griddata((x_WT, z_WT), uu_WT, (xi, zi), method='cubic')

fig,ax = plt.subplots(1,1)

plt.plot(ui/(u_inf*u_inf), zi)
plt.savefig('wake_line15.png', dpi=600)
x0, z0 = 1.6, -0.05
x1, z1 = 1.6, 0.25
num = 300

print('creating line using ' + str(num) + ' points...')
xi, zi = np.linspace(x0, x1, num), np.linspace(z0, z1, num)

ui = scipy.interpolate.griddata((x_WT, z_WT), uu_WT, (xi, zi), method='cubic')

fig,ax = plt.subplots(1,1)

plt.plot(ui/(u_inf*u_inf), zi)
plt.savefig('wake_line16.png', dpi=600)

