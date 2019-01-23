# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import tecplot as tp
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

width = 3.5
from setup_line_plot import *
line_style, markers = setup_plot('JOURNAL_1COL', width, width*2/3)
import main.wake.funcs.plotters as wplot



#    u_WT = u*np.cos(-1*np.radians(alpha)) - w * np.sin(-1*np.radians(alpha))

#    uu_WT =  uu * (np.cos(-1*np.radians(alpha)))**2 - 2 * uw * np.sin(-1*np.radians(alpha)) * np.cos(-1*np.radians(alpha)) + uw * (-1*np.radians(alpha))**2
  #EQUATION = '{MSMT_uu} = {resolved-uu}*(COS(-|alpha|/180*PI))**2 -2*{resolved-uw}*SIN(-|alpha|/180*PI)*COS(-|alpha|/180*PI) + {resolved-ww}*(SIN(-|alpha|/180*PI))**2'



def rotate_stresses(uu,vv,ww,uv=None, uw=None, vw=None, x_PMR=None, z_PMR=None, alpha=18.):
    uu_WT =  uu * (np.cos(-1*np.radians(alpha)))**2 - 2 * uw * np.sin(-1*np.radians(alpha)) * np.cos(-1*np.radians(alpha)) + uw * (-1*np.radians(alpha))**2
    return uu_WT


def main():
    alpha = 18.
    x_PMR = 0.23/4
    z_PMR = 0
    u_inf = 241
    case_name = 'OAT15A_AZDES-SSG'
    plane = 'symm'
    #case_name = 'DDES_v38t_dt100md_symm_tau2014'
    #case_name = 'DDES_v38_dt100md'
    #case_name = 'DDES_v38h_dt100md_symm_tau2014'
    #case_name = 'DDES_v38h_dt50md_symm_tau2014'
    #case_name = 'DDES_v38t05_dt100md_symm_tau2014'

    in_path = case_name + '_' + plane + '_result.plt'
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

    uu = data['resolved-uu']
    x = data['X']
    z = data['Z']
    #########################################
    #plot velocity
    import matplotlib.tri as tri

    triang = tri.Triangulation(x, z)
    fig, ax = plt.subplots(1,1)
    minval = 0
    maxval = 0.15
    color_delta = 0.001 # 0.001

    vals = np.arange(minval, maxval+color_delta, color_delta)
    import matplotlib.colors as mpcolors
    cmap = plt.cm.seismic
    norm = mpcolors.Normalize(vmin=minval, vmax=maxval)


    contour = plt.tricontourf(x, z, uu/(u_inf * u_inf), vals, norm=norm, cmap=cmap)
    #plt.savefig('OAT1.png', dpi=600)

    plt.gca().set_aspect('equal')

    plt.xlim(0.5*0.23, 1.5*0.23)
    plt.ylim(-0.2*0.23, 0.4*0.23)


    #levels = np.linspace(varmin,varmax,100)
    #cmap = plt.cm.viridis
    #plotvar = u
    #norm = mpl.colors.Normalize(vmin=varmin, vmax=varmax)

#    cbar = wplot.add_colorbar(ax, contour)
#    cbar.set_ticks(np.linspace(minval,maxval,6))


    # setup colorbar
#    cbar = fig.colorbar(contour, ticks=[minval, maxval])
#    cbar.ax.set_yticklabels(['-1', '0', '1'])  # vertically oriented colorbar
    adjustprops = dict(bottom=0.15, right=0.92)       # Subplot properties
    plt.subplots_adjust(**adjustprops)


    #contour_filled = plt.tricontourf(x, z, plotvar, levels, cmap=cmap)
    plt.savefig('OAT.png', dpi=600)


if __name__ == "__main__":
    main()


#import main.wake.funcs.plotters as wplot
#x_wing, z_wing, x_HTP, z_HTP = wplot.get_wing_HTP_cuts(plane=plane)

#x_WT, z_WT = transform_wake_coords(data['X'], data['Z'], x_PMR, z_PMR, alpha)
#u_WT, w_WT = rotate_velocities(data['mean-u'], data['mean-v'], data['mean-w'], x_PMR, z_PMR, alpha)
#uu_WT = rotate_stresses(data['resolved-uu'],None,data['resolved-ww'], uw=data['resolved-uw'], x_PMR=x_PMR, z_PMR=z_PMR, alpha=18.)


#fig, ax = wplot.plot_velocity(x, z, uw, kind='CRM_wake', shape=None, struct=False, v=None, streamlines=False)
#cmap = plt.cm.hot_r


#triang = tri.Triangulation(x_wing,z_wing)
#plt.tricontourf(triang, x_wing, colors='black')

