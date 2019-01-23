# /usr/bin/python
"""
read slices of multiple cases, as indicated in cases_slices.txt
plot velocity contours of these cases in one image
"""
import numpy as np
import re
import pandas as pd
import pyTecIO_AW.tec_modules as tec

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import rc
import matplotlib as mpl
#import seaborn as sns
#plt.style.use('seaborn')
#import pyTecIO_AW.read_unstruct_plt as reader
import plotters as wplot
import pyTecIO_AW.parse_tecplot_helpers as parser
import helpers as wf
import main.wake.funcs.plotters as wplot
import tecplot as tp
import helpers.wake_stats as ws

width = 3.5

try:
    from setup_plot import * # custom plot setup file
    line_style, markers = setup_plot('JOURNAL_1COL', width, width*2/5)

except ImportError as e:
    print 'setup_plot not found, setting default line styles...'
    line_style=itertools.cycle(["-", "--", ":"])
    marker_style = itertools.cycle(['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h'])

print 'matplotlib version: ' + mpl.__version__

def triangulate_wake(x,z):
    triang = tri.Triangulation(x, z)
    return triang


x_PMR = 0.90249
z_PMR = 0.14926
u_inf = 54.65
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


###################
# load 3 cases
in_vars = ['X', 'Y', 'Z', 'mean-u', 'mean-v', 'mean-w']

files = dict()
aoa={}
label1 = 'AoA16'
label2 = 'AoA18'
label3 = 'AoA20'
label1 = r'$\alpha=16^{\circ}$'
label2 = r'$\alpha=18^{\circ}$'
label3 = r'$\alpha=20^{\circ}$'

plane = 'eta0603'

files[label1] = './results/CRM_v38h_AoA16_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_'+plane+'_rstresses_3000_10400.plt'
files[label2] = './results/CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2_'+plane+'_rstresses_5000_13000.plt'
files[label3] = './results/CRM_v38h_AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_k1024_'+plane+'_rstresses_5000_12800.plt'


aoa[label1] = 16.
aoa[label2] = 18.
aoa[label3] = 20.
dataset={}
t={}
CL={}


for case, path in files.iteritems():
    dataset[case] = get_result_file(path)

x_wing, z_wing, x_HTP, z_HTP = wplot.get_wing_HTP_cuts(plane=plane)


case_name ='CRM_DDES_v38h_3AoA'

x_WT = {}
z_WT = {}
u_WT = {}
w_WT = {}
uu_WT = {}
for case, values in dataset.iteritems():
    x_WT[case], z_WT[case] = ws.transform_wake_coords(values['X'], values['Z'], x_PMR, z_PMR, aoa[case])
    u_WT[case], w_WT[case] = ws.rotate_velocities(values['mean-u'], values['mean-v'], values['mean-w'], x_PMR, z_PMR, aoa[case])




n_cases = len(aoa)
#print len(x)

#levels = [0, 50,100, 150, 200] # NACA0012
#levels = [0, 10, 20, 30, 40, 50, 60]
levels = [0.05*54.65]
#labels = ['k', 'b','r']


fig, ax = plt.subplots(1,1)
line_style = itertools.cycle(["-", "--", "-."])
line_color = itertools.cycle(["k", "b", "r"])
clabels =False
leg=True
zo = 0
recirc_x = {}
recirc_z = {}
for case in sorted(files):
    col = line_color.next()
    print 'handling case ' + str(case)
    triang = triangulate_wake(x_WT[case],z_WT[case])
    contour = ax.tricontour(triang, u_WT[case], levels, colors=col)
    x_wing, z_wing, x_HTP, z_HTP = wplot.get_wing_HTP_cuts(plane=plane, rotated=aoa[case])
    print('wing max: ' + str(max(z_wing)))
    print('wing min: ' + str(min(z_wing)))
    print('bluff body projected height: ' + str(max(z_wing) - min(z_wing)))
    triang = tri.Triangulation(x_wing,z_wing)
    ax.tricontourf(triang, x_wing, colors='black', zorder=zo)
    zo = zo + 10

    p = contour.collections[0].get_paths()[0]
    v = p.vertices
    recirc_x[case] = v[:,0]
    recirc_z[case] = v[:,1]


    print recirc_x[case].shape
    max_x = np.max(recirc_x[case])
    print(np.max(recirc_x[case]))
    print(np.max(recirc_z[case]))
    print(np.min(recirc_z[case]))
    plt.axvline(max_x, color=col, linestyle='--', zorder=zo+10)
    if leg is True:
        contour.collections[0].set_label(case)

    if clabels is True:
        # clabel needs to occur AFTER limits setting: label positions are calculated depending on visible area
        plt.clabel(contour, inline=1, inline_spacing=0, fontsize=8, fmt='%i')
#plt.legend(loc='best')
ax.set_aspect('equal')

plt.xlabel('x [m]', labelpad=-1)
plt.ylabel('z [m]', labelpad=1)
plt.xlim(0.55, 1.3) # CRM close to wing
plt.ylim(-0.05, 0.2)
adjustprops = dict(left=0.11, bottom=0.13, right=0.97, top=0.97)       # Subplot properties
plt.subplots_adjust(**adjustprops)


img_name = case_name + '_' + plane + '_recirculation_contours'
plt.savefig(img_name+'.png', dpi=600)
plt.savefig(img_name+'.pdf')




###############################################################
# filled contours, i.e. colors

#contour_filled = plt.tricontourf(x, z, u, levels, cmap=cmap)
#cmap = plt.cm.viridis
#norm = mpl.colors.Normalize(vmin=min(u), vmax=max(u))
#contour_filled = plt.tricontourf(x, z, u, 100,norm=norm, cmap=cmap)

#plt.colorbar(contour_filled)
#plt.show()

##############################################################
# save image
#img_name = 'DDES_umean_3diss_contours_eta0283'
#print "writing : " + img_name + ".pdf"
#plt.savefig(img_name+'.png', dpi=600)
#plt.savefig(img_name+'.pdf', dpi=600)
