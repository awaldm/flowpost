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
import matplotlib.tri as tri

from setup_line_plot import *
line_style, markers = setup_plot('JOURNAL_1COL', width, width*2/3)

aoa = {}
x_PMR = 0.165/4
z_PMR = 0.0
u_inf = 85.7
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
plane = 'periodic1'
if plane is 'eta0201':
    xlim = (1, 1.6)
elif plane is 'eta0283':
    xlim = (1.05, 1.65)
elif plane is 'eta0603':
    xlim = (1.15, 1.75)
elif plane is 'eta0131':
    xlim = (1, 1.6)
elif plane is 'periodic1':
    xlim = (0, 0.15)
###################
# load 3 cases

files = dict()

#files['DDES_v38_dt100md'] = './results/DDES_v38_dt100_md_cK_eta0283_rstresses_28500_31800.plt'
#files['DDES_v38h_dt100md_symm_tau2014'] = './results/DDES_v38h_dt100_md_CFL2_eigval02_pswitch1_symm_tau2014_eta0283_rstresses_1000_5600.plt'
#files['DDES_v38t_dt100md_symm_tau2014'] = './results/DDES_v38t_dt100md_symm_tau2014_eta0283_rstresses_1100_2800.plt'
label1 = 'AoA19'
#label2 = 'AoA18'
#label3 = 'AoA20'
#label1 = r'$\alpha=16^{\circ}$'
#label2 = r'$\alpha=18^{\circ}$'
#label3 = r'$\alpha=20^{\circ}$'

files[label1] = './results/NACA0012_AoA19_DDES_SAO_dt1e5_k1024_turbAoF_eta0283_rstresses_10000_24000.plt'
files[label1] = './results/NACA0012_AoA19_URANS_SAO_dt1e5_k1024_turbAoF_eta0283_rstresses_17000_20000.plt'

aoa[label1] = 19.
#aoa[label2] = 18.
#aoa[label3] = 20.
dataset={}
t={}
CL={}

for case, path in files.iteritems():
    dataset[case] = get_result_file(path)


#case_name = 'DDES_v38t_dt100md_symm_tau2014'
#case_name = 'DDES_v38_dt100md'
#case_name = 'DDES_v38h_dt100md_symm_tau2014'

case_name = 'DDES_v38_dt100md'
case_name ='NACA0012'

x_WT = {}
z_WT = {}
u_WT = {}

for case, values in dataset.iteritems():
    x_WT[case], z_WT[case] = ws.transform_wake_coords(values['X'], values['Z'], x_PMR, z_PMR, aoa[case])
    u_WT[case], _ = ws.rotate_velocities(values['mean-u'], values['mean-v'], values['mean-w'], x_PMR, z_PMR, aoa[case])
    #uu_WT[case], _, ww_WT[case], uv_WT[case], uw_WT[case], vw_WT[case] = ws.rotate_stresses(values['resolved-uu'],None,values['resolved-ww'], uw=values['resolved-uw'], x_PMR=x_PMR, z_PMR=z_PMR, alpha=aoa[case])
    #vv_WT[case] = values['resolved-vv']
# end loading 3 cases
#######################


x0, z0 = xlim[0], -0.05
x1, z1 = xlim[1], 0.1

xi, zi = np.linspace(x0, x1, 100), np.linspace(z0, z1, 1000)
xmesh, zmesh = np.meshgrid(xi,zi)

print('shape of xi: ' + str(xi.shape))
print('shape of zi: ' + str(zi.shape))
print('mesh shape: ' + str(xmesh.shape))

n_cols = xi.shape[0]
n_rows = zi.shape[0]
ui = {}

for case, _ in files.iteritems():
    ui[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), u_WT[case], (xmesh, zmesh), method='linear')

print('size of interpolated velocity ui: ' + str(ui[label1].shape))
dx = xi[1] - xi[0]
dz = zi[1] - zi[0]
print('dx: ' + str(dx))
print('dz: ' + str(dz))
uy, ux = np.gradient(ui[label1], dz, dx)
print('shape of gradient: ' + str(uy.shape))
max_u = np.max(ui[label1], axis=0)
min_u = np.min(ui[label1], axis=0)
print('size of max_u: ' + str(max_u.shape))
vort_thick = np.zeros_like(max_u)
ind = np.zeros_like(max_u, dtype=np.int64)
max_grad_u = np.zeros_like(max_u)
vort_thick = np.zeros_like(max_u)
max_grad_pos = np.zeros_like(max_u, dtype=np.int64)

for j in range(n_cols):
    indtemp = np.argwhere(ui[label1][:,j] == 0)

    if len(indtemp) > 0:
        ind[j] = np.max(indtemp)
    #print ind[j]
    max_grad_u[j] = np.max(uy[ind[j]:-1, j])
    max_grad_pos[j] = np.argmax(uy[ind[j]:-1, j]) + ind[j]
    vort_thick[j] = ((max_u[j] - min_u[j]) / max_grad_u[j]) / 0.165
    #print max_grad_pos[j]
print('size of max_grad_pos: ' + str(max_grad_pos.shape))

fig, ax = plt.subplots(1,1)
plt.plot(xi, zi[ind])
fig.savefig('umeans_ind.pdf')
plt.close()


fig, ax = plt.subplots(1,1)
minval = -50
maxval = 100
color_delta = 1
vals = np.arange(minval, maxval+color_delta, color_delta)

norm = matplotlib.colors.Normalize(vmin=minval, vmax=maxval)
xslice, zslice = wplot.get_NACA0012_airfoil_shape(rotated=aoa[label1])

contour = plt.contourf(xmesh, zmesh, ui[label1], vals, norm=norm, extend='max')#, vals, norm, cmap=cmap)
for c in contour.collections:
    c.set_edgecolor("face")
#plt.clim(minval, maxval)
triang = tri.Triangulation(xslice,zslice)
ax.tricontourf(triang, xslice, colors='black', zorder=10)
fig.savefig('umeans_contour.pdf')
fig.savefig('umeans_contour.png', dpi=600)
plt.close()

fig, ax = plt.subplots(1,1)
plt.plot(xi, max_u)
fig.savefig('umeans_max.pdf')
fig, ax = plt.subplots(1,1)
plt.plot(xi, min_u)
fig.savefig('umeans_min.pdf')
plt.close()

fig, ax = plt.subplots(1,1)
plt.plot(xi, zi[max_grad_pos])
fig.savefig('max_grad_pos.pdf')
plt.close()

fig, ax = plt.subplots(1,1)
plt.plot(xi, max_grad_u)
plt.ylim(0, 10000)
fig.savefig('max_grad.pdf')
plt.close()
fig, ax = plt.subplots(1, 1)
plt.plot(xi, vort_thick)
plt.ylim(0, 0.3)
#plt.xlim(0, 0.115)
plt.ylabel(r'$\delta_{\omega}$', labelpad=-2)
plt.xlabel('x [m]')
plt.grid(True)
fig.savefig('vortthick.pdf')


xpos_list = [1.3, 1.4, 1.5, 1.6]
xref_list = [1, 1.5, 2, 2.5, 3]
#xref_list = [2]
xpos_list = (np.asarray(xref_list) * 0.189144) + 1
print xpos_list



#xi, zi, ui, uu, wi, ww, samp_u, samp_w = get_vert_line(xpos_list, x_WT, z_WT, u_WT, uu_WT, w_WT, ww_WT, samples_u, samples_w)
u_inf = 54.65

xpos_list = [50, 100, 150, 200, 249]
xpos_list = [20, 40, 60, 80, 99]

fig, ax = plt.subplots(1, len(xpos_list), figsize=(width,0.6*width), sharey=True)
for xpos in range(len(xpos_list)):
    #col = line_color.next()
    i = xpos_list[xpos]
    print i
    #print(ui[:,i].shape)
    #print(zi.shape)
    ax[xpos].plot(ui[label1][ind[i]:-1,i], zi[ind[i]:-1], label=str(np.max(ui[label1][:,i])))
    #ax[xpos].axhline(zi[ind[i]], color='k')
    ax[xpos].axhline(zi[max_grad_pos[i]], color='r')

    print(str(np.max(ui[label1][:,i])))
    print('pos of grad maximum: ' + str(max_grad_pos[i]))
#        axes.get_yaxis().set_visible(False)
#        axes.yaxis.set_ticklabels([])
#    plt.grid(True)
#    plt.xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[0].set_ylabel('$z [m]$', labelpad=-4.5)
#    ax[i].set_xlim(0.3, 1)
    #plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    #plt.xlim(0,0.05)
    #plt.ylim(-0.1,0.17)
    plt.legend(loc='best')
    adjustprops = dict(left=0.15, bottom=0.16, right=0.95, top=0.97, wspace=0.1, hspace=0)
    plt.subplots_adjust(**adjustprops)

plt.savefig('umeans_'+plane+'.pdf')
plt.savefig('umeans_'+plane+'.png', dpi=600)
plt.close()

fig, ax = plt.subplots(1, len(xpos_list), figsize=(width,0.6*width), sharey=True)
for xpos in range(len(xpos_list)):
    #col = line_color.next()
    i = xpos_list[xpos]
    #print(ui[:,i].shape)
    #print(zi.shape)
    ax[xpos].plot(uy[ind[i]:-1,i], zi[ind[i]:-1], label=str(np.max(ui[label1][:,i])))
#    ax[xpos].axhline(zi[ind[i]])
#        axes.get_yaxis().set_visible(False)
#        axes.yaxis.set_ticklabels([])
#    plt.grid(True)
#    plt.xlabel('$\overline{u} / u_{\infty}$', labelpad=0)
    ax[0].set_ylabel('$z [m]$', labelpad=-4.5)
    ax[xpos].set_xlim(-1000, 15000)
    #plt.axhspan(-0.036,-0.033, color='gray', alpha=0.35)
    #plt.ylim(-0.1,0.17)
    #plt.legend(loc='best')
    adjustprops = dict(left=0.15, bottom=0.16, right=0.95, top=0.97, wspace=0.1, hspace=0)
    plt.subplots_adjust(**adjustprops)

plt.savefig('umeans_'+plane+'_gradients.pdf')
plt.savefig('umeans_'+plane+'_gradients.png', dpi=600)
plt.close()


'''
fig, ax = plt.subplots(1, 1)
line_color = itertools.cycle(["k", "b", "r", "g"])
for case, _ in files.iteritems():
    pos = np.argmin(np.where(ui[case]==0, ui[case].max(), ui[case]), axis=0)
    print('shape of xi:' + str(xi.shape))
    print('shape of zi:' + str(zi[pos].shape))
    plt.plot(xi, zi[pos], color=line_color.next(), label=case) # , linestyle=line_style.next(), marker=markers.next(), mew=1, ms=2, markevery=10)
    np.savez(case+'_' + plane + '_wake_min_pos', xi=xi, zi=zi[pos])

plt.legend(loc='best')

image_name = case_name+'_vt'
#    print("exporting image " + image_name)
plt.ylim(-0.05, 0.25)
fig.savefig(image_name + '.png', dpi=600)
plt.close(fig)



[max_u, index_max_u] = max(interp_u);
[min_u, index_min_u] = min(interp_u);

index_max_grad_u = linspace(0,0,n);
real_index_max_grad_u = linspace(0,0,n);
vort_thick = linspace(0,0,n);

delta_u = zeros(m,n);
delta_z = zeros(m,n);

for j=1:n

    for i=1:m-1
        delta_u(i,j) = (interp_u(i+1,j)-interp_u(i,j));
        delta_z(i,j) = (interp_z(i+1,j)-interp_z(i,j));
    end

end

for j=1:n
    [sel index_low_profile_bound] = max(interp_u==0, [], 1);
    [max_grad_u(j) index_max_grad_u(j)] = max(gradient(interp_u(index_low_profile_bound(j):end,j))/z_spacing);
    real_index_max_grad_u(j) = index_max_grad_u(j) + index_low_profile_bound(j);
    vort_thick(j) = ((max_u(j)-min_u(j))/max_grad_u(j))./data.c;

end

%gradient_u = delta_u./delta_z;
max_grad_zero_u = max(delta_u./delta_z);
vort_thick_zero_u = (max_u./max_grad_zero_u)./data.c;
'''
