#/usr/bin/python
"""
from https://github.com/richardagalvez/Vortices-Python/blob/master/Vortex-FTLE.ipynb

Andreas Waldmann, May 2017
"""
from __future__ import division
import os
import time
import shutil
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib as mpl
import h5py
import matplotlib.pyplot as plt

#import VortexFitting.src.schemes as schemes
#import VortexFitting.src.detection as detection
#import VortexFitting.src.tools as tools
#import VortexFitting.src.fitting as fitting
#import VortexFitting.src.plot as plot


import pyTecIO_AW.read_2d_wake_timeseries as tecread


def write_FTLE_ASCII(filename, x, z, ftle, shape, title, variables):
    '''
    Write 2D data modes to a ASCII text files in Tecplot format
    inputs:
        mode : dict of mode with their respective variable names, i.e. 'u', 'v' etc
        shape : if None then dataset is assumed to be unstructured
        variables : list of strings
    '''
    #so far structured only
    #mode = unstack_DMD_mode(mode, variables)
    ftle = ftle.reshape(shape[0]*shape[1])
    # modes is now a dict
    x = x.reshape(shape[0]*shape[1])
    z = z.reshape(shape[0]*shape[1])

    header_str = 'TITLE = ' + title + 'mode\n'
    header_str += 'VARIABLES = "x" "z" "FTLE"'
    header_str += '\n'        
    header_str += 'ZONE T = "'+title+' results"\n'
    header_str += 'I='+str(shape[1])+', J=1, K='+str(shape[0])+', ZONETYPE=Ordered'
    header_str += ' DATAPACKING=POINT\n'
    header_str += 'DT=( DOUBLE DOUBLE DOUBLE '
    header_str +=')'
    
    n_vars = len(variables)
    data_matrix = np.c_[x, z, ftle]
    #np.c_[mode[:,0]], \
    #out_path + title+ '_mode.dat', \
    np.savetxt(filename, \
        data_matrix, \
        delimiter = ' ', \
        header = header_str,
        comments='')  


# plot size
width = float(7)/float(2)
'''
try:
    from setup_plot import * # custom plot setup file
    line_style, markers = setup_plot('JOURNAL_1COL', width, 0.9*width)
except ImportError as e:
    print('setup_plot not found, setting default line styles...')
    line_style=itertools.cycle(["-", "--", ":"])
    marker_style = itertools.cycle(['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h'])
'''
out_folder = './'

def field_geometry_info(x, z):
    print('rectangular field: \n')
    print('X extents from x = ' + str(x[1, 1]) + ' to x = ' +  str(x[1, -1]))
    print('Z extents from z = ' + str(z[1, 1]) + ' to z = ' +  str(z[-1, 1]))

    dx = x[1, 2] - x[1, 1]
    dz = z[2, 1] - z[1, 1]

    print('dx = ' + str(dx) + ', dz = ' + str(dz))
class VelocityField():
    """NetCDF file

    Loads the input file with the NetCFD (.nc) format and
    initialize the variables.

    """
    def __init__(self, path="/", time=0, meanfilepath="/", rows=0, cols=0,x=None,z=None,u=None,w=None):

        self.sizex=cols
        self.sizey=rows
        y = z
        print(str(cols) + ' cols by ' + str(rows) + ' rows')
        skip = 0
        self.u  = np.array(u).reshape(self.sizey,self.sizex)
        self.v  = np.array(w).reshape(self.sizey,self.sizex)
        self.u = self.u[:,skip:]
        self.v = self.v[:,skip:]
        print('u shape: ' + str(u.shape))

        '''
        if (self.meanfilepath != '/' ):
            print("subtracting mean file")
            grp2=np.loadtxt(meanfilepath,delimiter=" ",dtype=float,skiprows=3) #mean data
            self.uMean  = np.array(grp2[:,index_u]).reshape(self.sizex,self.sizey)
            self.vMean  = np.array(grp2[:,index_v]).reshape(self.sizex,self.sizey)
            self.u = self.u - self.uMean
            self.v = self.v - self.vMean
        '''
        self.samples = self.u.shape[1]

        tmp_x  = np.array(x).reshape(self.sizey,self.sizex)
        tmp_y  = np.array(y).reshape(self.sizey,self.sizex)
        tmp_x = tmp_x[:,skip:]
        tmp_y = tmp_y[:,skip:]

        print('tmp_x shape:' + str(tmp_x.shape))

        self.dx = np.linspace(0, np.max(tmp_x)-np.min(tmp_x), self.u.shape[1])
        self.dy = np.linspace(0, np.max(tmp_y)-np.min(tmp_y), self.u.shape[0])
        print('dy shape: ' + str(self.dy.shape))
        print('dx shape: ' + str(self.dx.shape))
        #print(tmp_y[:,0])
        #print(tmp_x[:,0])


        self.step_dx=round((np.max(self.dx)-np.min(self.dx)) / (np.size(self.dx)-1) ,6)
        self.step_dy=round((np.max(self.dy)-np.min(self.dy)) / (np.size(self.dy)-1) ,6)
        self.norm = False
        self.normdir = 'x'

        #COMMON TO ALL DATA
        self.derivative = {'dudx': np.zeros_like(self.u),
                           'dudy': np.zeros_like(self.u),
                           'dudz': np.zeros_like(self.u),
                           'dvdx': np.zeros_like(self.u),
                           'dvdy': np.zeros_like(self.u),
                           'dvdz': np.zeros_like(self.u),
                           'dwdx': np.zeros_like(self.u),
                           'dwdy': np.zeros_like(self.u),
                           'dwdz': np.zeros_like(self.u)}

def bilinear_interpolation(X, Y, f, x, y):
    """Returns the approximate value of f(x,y) using bilinear interpolation.

    Arguments
    ---------
    X, Y -- mesh grid.
    f -- the function f that should be an NxN matrix.
    x, y -- coordinates where to compute f(x,y)

    """

    N = np.shape(X[:,0])[0]

    dx, dy = X[0,1] - X[0,0], Y[1,0] - Y[0,0]
    x_start, y_start = X[0,0], Y[0,0]
    #print('x_start: ' + str(x_start))
    #print('dx: ' + str(dx))
    #print('X shape: ' + str(X.shape))
    rows = X.shape[0]
    cols = X.shape[1]



    i1, i2 = int ((x - x_start)/dx) , int((x - x_start)/dx) + 1
    j1, j2 = int ((y - y_start)/dy) , int((y - y_start)/dy) + 1


    # Take care of boundaries

    # 1. Right boundary

    if i1 >= cols-1 and j1 <= rows-1 and j1 >= 0:
        return f[j1, cols-1]
    if i1 >= cols-1 and j1 <= 0 :
        return f[0, cols-1]
    if i1 >= cols-1 and j1 >= rows-1 :
        return f[rows-1, cols-1]

    # 2. Left boundary

    if i1 <= 0 and j1 <= rows-1 and j1 >= 0:
        return f[j1, 0]
    if i1 <= 0 and j1 <= 0 :
        return f[0, 0]
    if i1 <= 0 and j1 >= rows-1 :
        return f[rows-1, 0]

    # 3. Top boundary

    if j1 >= rows-1 and i1<=cols-1 and i1>=0:
        return f[rows-1, i1]
    if j1 >= rows-1 and i1 <= 0 :
        return f[rows-1, 0]

    # 3. Bottom boundary

    if j1 <= 0 and i1<=cols-1 and i1>=0:
        return f[0, i1]
    if j1 <= 0 and i1 >= cols-1 :
        return f[rows-1, 0]


    x1, x2 = X[j1,i1], X[j2,i2]
    y1, y2 = Y[j1,i1], Y[j2,i2]
    #print(type(x1))
    #print(type(f))

    f_interpolated = ( 1/(x2-x1)*1/(y2-y1) *
                      ( f[j1,i1]*(x2-x)*(y2-y) + f[j1,i2]*(x-x1)*(y2-y)
                      + f[j2,i1]*(x2-x)*(y-y1) + f[j2,i2]*(x-x1)*(y-y1)) )

    return f_interpolated

def rk4(X, Y, x, y, f, h, dim):
    """Returns the approximate value of f(x,y) using bilinear interpolation.

    Arguments
    ---------
    X, Y -- mesh grid.
    x, y -- coordinates where to begin the evolution.
    f -- the function f that will be evolved.
    h -- the time step (usually referred to this as dt.)
    dim -- 0 for x and 1 for y.

    """

    k1 = h * bilinear_interpolation(X, Y, f, x, y)
    k2 = h * bilinear_interpolation(X, Y, f, x + 0.5 * h, y + 0.5 * k1)
    k3 = h * bilinear_interpolation(X, Y, f, x + 0.5 * h, y + 0.5 * k2)
    k4 = h * bilinear_interpolation(X, Y, f, x + h      , y + k3)

    if dim == 0:
        return x + 1./6 * k1 + 1./3 * k2 + 1./3 * k3 + 1./6 * k4
    elif dim == 1:
        return y + 1./6 * k1 + 1./3 * k2 + 1./3 * k3 + 1./6 * k4
    else:
        #print 'invalid dimension parameter passed to rk4, exiting'
        sys.exit()


def test_trajectory(X, Y, i, j, u,v,integration_time, dt):
    """ Plots the trajectories of a few particles

    Arguments
    ---------
    X, Y -- mesh grid.
    i, j -- indices of the first particle on the mesh.
    integration_time -- the duration of the integration
    dt -- the finess of the time integration space.

    """
    x_start = X[0,0]
    x_end = X[0,-1]
    print('x_end: ' + str(x_end))
    y_start = Y[0,0]
    y_end = Y[-1,0]
    size = 12
    #fig = plt.figure(figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
    fig, ax = plt.subplots(1,1)
    fileName = out_folder + 'vortex1_sample_traj.pdf'

    #plt.xlim(x_start*1.5, x_end*1.5)
    #plt.ylim(y_start*1.5, y_end*1.5)

    xs, ys = X[j,i], Y[j, i]

    traj_x , traj_y = np.zeros((X.shape[0],X.shape[1])), np.zeros((X.shape[0],X.shape[1]))

    traj_x[j][i], traj_y[j][i] = xs, ys

    print '(x0s, y0s) ->', xs, ys

    print 'begining trajectory calculation'

    colors = ['r','k','c'] # To plot more particles, add more colors

    for l, c in enumerate(colors):

        for k in xrange(0, int(integration_time/dt)):

            xs, ys = rk4(X, Y, xs, ys, u, dt, 0), rk4(X, Y, xs, ys, v, dt, 1)
            traj_x[j][i] += xs
            traj_y[j][i] += ys

            #print '(xs, ys) ->', xs, ys
            plt.scatter(xs, ys, s=5, color = c)
            #print k*dt

        i += 10
        j += 10

        xs, ys = X[j,i], Y[j, i]
    #print('X: ' + str(X[0,:]) + ', Y: ' + str(Y[:,0]))
    plt.streamplot(X, Y, u, v, density=2, linewidth=1, arrowsize=1, arrowstyle='->', color='k')
    #plt.scatter(x_vortex, y_vortex, color='#CD2305', s=25, marker='o')
    fig.savefig(fileName)

    return None

def get_traj(X, Y, u, v, integration_time, dt):
    """ Returns the FTLE particle trajectory

    Arguments
    ---------
    x, y -- mesh grid coordinates
    dt -- integral time step
    """
    N = np.shape(X[:,1])[0]
    x = X[0,:]
    y = Y[:,0]
    rows = X.shape[0]
    cols = X.shape[1]
    print(len(x))
    print(len(y))

    traj_x = np.zeros((rows,cols), dtype=np.float)
    traj_y = np.zeros((rows,cols), dtype=np.float)

    for i, xx in enumerate(x):
        for j, yy in enumerate(y):
            xs, ys = xx, yy
            #print('xx: ' +str(xx) + ', yy: ' + str(yy))
            traj_x[j][i], traj_y[j][i] = xs, ys
            for k in xrange(0, int(integration_time/dt)):

                xs, ys = rk4(X, Y, xs, ys, u, dt, 0), rk4(X, Y, xs, ys, v, dt, 1)
                traj_x[j][i] += xs
                traj_y[j][i] += ys

    return traj_x, traj_y

def get_ftle(traj_x, traj_y, X, Y, integration_time):
    """ Returns the FTLE scalar field

    Mostly adapted from Steven's FTLE code (GitHub user stevenliuyi)

    Arguments
    ---------
    traj_x, traj_y -- The trajectories of the FTLE particles
    X, Y -- Meshgrid
    integration_time -- the duration of the integration time

    """

    #dx, dy = X[0,1] - X[0,0], Y[1,0] - Y[0,0]
    #x_start, y_start = X[0,0], Y[0,0]
    #print('x_start: ' + str(x_start))
    #print('dx: ' + str(dx))
    #print('X shape: ' + str(X.shape))
    rows = X.shape[0]
    cols = X.shape[1]

    N = np.shape(X[:,0])[0]
    ftle = np.zeros((rows,cols))

    for i in range(0,cols):
        for j in range(0,rows):
            # index 0:left, 1:right, 2:down, 3:up
            xt = np.zeros(4); yt = np.zeros(4)
            xo = np.zeros(2); yo = np.zeros(2)

            if (i==0):
                xt[0] = traj_x[j][i]; xt[1] = traj_x[j][i+1]
                yt[0] = traj_y[j][i]; yt[1] = traj_y[j][i+1]
                xo[0] = X[j][i];      xo[1] = X[j][i+1]
            elif (i==cols-1):
                xt[0] = traj_x[j][i-1]; xt[1] = traj_x[j][i]
                yt[0] = traj_y[j][i-1]; yt[1] = traj_y[j][i]
                xo[0] = X[j][i-1]; xo[1] = X[j][i]
            else:
                xt[0] = traj_x[j][i-1]; xt[1] = traj_x[j][i+1]
                yt[0] = traj_y[j][i-1]; yt[1] = traj_y[j][i+1]
                xo[0] = X[j][i-1]; xo[1] = X[j][i+1]

            if (j==0):
                xt[2] = traj_x[j][i]; xt[3] = traj_x[j+1][i]
                yt[2] = traj_y[j][i]; yt[3] = traj_y[j+1][i]
                yo[0] = Y[j][i]; yo[1] = Y[j+1][i]
            elif (j==rows-1):
                xt[2] = traj_x[j-1][i]; xt[3] = traj_x[j][i]
                yt[2] = traj_y[j-1][i]; yt[3] = traj_y[j][i]
                yo[0] = Y[j-1][i]; yo[1] = Y[j][i]
            else:
                xt[2] = traj_x[j-1][i]; xt[3] = traj_x[j+1][i]
                yt[2] = traj_y[j-1][i]; yt[3] = traj_y[j+1][i]
                yo[0] = Y[j-1][i]; yo[1] = Y[j+1][i]

            lambdas = eigs(xt, yt, xo, yo)
            if lambdas=='nan':
                ftle[j][i] = float('nan')
            else:
                # why not sqrt? maybe its just scaling and doesnt matter much. why 0.5 though?
#                ftle[j][i] = .5*np.log(max(lambdas))/(integration_time)
                ftle[j][i] = np.sqrt(np.log(max(lambdas))) / integration_time# .5*np.log(max(lambdas))/(integration_time)


    return ftle
# calculate eigenvalues of [dx/dx0]^T[dx/dx0]
def eigs(xt, yt, xo, yo):
    ftlemat = np.zeros((2,2))
    ftlemat[0][0] = (xt[1]-xt[0])/(xo[1]-xo[0])
    ftlemat[1][0] = (yt[1]-yt[0])/(xo[1]-xo[0])
    ftlemat[0][1] = (xt[3]-xt[2])/(yo[1]-yo[0])
    ftlemat[1][1] = (yt[3]-yt[2])/(yo[1]-yo[0])

    if (True in np.isnan(ftlemat)):
        return 'nan'

    # is this the cauchy-green deformation tensor? Eq. 9 in Steinfurth/Haucke
    ftlemat = np.dot(ftlemat.transpose(), ftlemat)
    w, v = np.linalg.eig(ftlemat)

    return w

# Wang: "Optimal stretching in the reacting wake of a bluff body" say that "Lagrangian stretching S" and FTLE are related.
# For them, Lagrangian stretching is the maximum eigenvalue of the right Cauchy-Green Tensor
# Basically, it is the same thing as far as I understand.

# BIG TODO: check out what happens if we take SQRT!


######################################################################
# USER INPUT
case = 'DDES_dt200_ldDLR'
#case = 'URANS-SAO'
case = 'DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2'
case = 'AoA20_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_k1024'
plane = 'eta0283_struct_front'

source_path = '/home/andreas/NAS_CRM/CRM/M025/AoA18/' + case + '/sol/surface/ascii/'+plane+'/'
source_path = '/home/andreas/NAS_CRM2/CRM/M025/v38_h2/'+ case +'/sol/surface/ascii/'+plane+'/'

img_path = './'
case_name = case + '_eta0283'
case_name = case + '_' + plane
case_name = case_name + '_sqrt'

ts = 9760
ts = 14000
dt=0.0003461
c_ref = 0.189144
u_inf = 54.65
t_inf = c_ref / u_inf
#dt = 0.0003461 / 2


rows = 300
cols = 435
rows = 265
cols = 450
streamplot = False
forward = False
if forward:
    direction_string = 'fwd'
else:
    direction_string = 'back'

# END USER INPUT
######################################################################

in_file = source_path + 'CRM_v38_WT_SAO_a18_DES_'+plane+'_i='+str(ts)+'.dat'
#in_file = source_path + 'CRM_v38_WT_SAO_a18_URANS_eta0283_i='+str(ts)+'.dat'
x,y,z,data = tecread.read_ASCII_file(in_file, varnames=['"x_velocity"', '"y_velocity"', '"z_velocity"'], shape=None, verbose=True)

#x, z, u, w = get_struct_wake(source_path, matfilepath, rows, cols)
u = data['"x_velocity"']
v = data['"y_velocity"']
w = data['"z_velocity"']

print("type of umat: " + str(type(u)))
print("shape of loaded u velocity matrix: " + str(u.shape))


#print("Opening file:",args.infilename.format(time_step),args.meanfilename)
a = VelocityField('aaa',10,'args.meanfilename', rows, cols, x,z,np.nan_to_num(u),np.nan_to_num(w))
fig, ax = plt.subplots(1,1)

minval = -20 # np.amin(u)
maxval = 90 # np.amax(u)
color_delta = 1
cmap = plt.cm.seismic
cmap=plt.cm.bone
vals = np.arange(minval, maxval+color_delta, color_delta)
norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
xgrid,ygrid = np.meshgrid(a.dx, a.dy)
print('xgrid shape: ' + str(xgrid.shape))
print('u shape: ' + str(a.u.shape))
contour = plt.contourf(xgrid, ygrid, a.u, vals, norm=norm, cmap=cmap)
plt.gca().set_aspect('equal')

fig.savefig(out_folder + case_name + '_velocityfield_'+str(ts)+'.png', dpi=600)

if streamplot:
#print('dx: ' + str(a.dx) + ', dy:' + str(a.dy))

#xgrid=xgrid.T
#ygrid=ygrid.T
#print('xgrid: ' + str(xgrid[0,:]))
    print(np.allclose(xgrid[0,:], xgrid))
    size = 12 # Size of the image displayed only in this notebook.

    fileName = out_folder + 'central_vortex_flow_streamplot'+str(ts)+'.pdf'

    #fig = plt.figure(figsize=(size, (y_end-y_start)/(x_end-x_start)*size))
    fig, ax = plt.subplots(1,1)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    #plt.xlim(x_start, x_end)
    #plt.ylim(y_start, y_end)
    #plt.streamplot(a.dx, a.dy, a.u, a.v, density=2, linewidth=1, arrowsize=1, arrowstyle='->', color='k')
    plt.streamplot(a.dx, a.dy, a.u, a.v, density=2, linewidth=1, arrowsize=1, arrowstyle='->', color='k')

    #plt.scatter(x_vortex, y_vortex, color='#CD2305', s=40, marker='o')
    plt.draw()
    fig.savefig(fileName)




integration_time = 0.05
integration_time = t_inf * 2
dt_factor = 1
if not forward:
    dt = -1.0 * dt
    integration_time = -1.0 * integration_time

#dt=0.002
#integration_time = 0.000
#test_trajectory(xgrid, ygrid, 150,150, a.u, a.v,-0.01, 0.00005)
#test_trajectory(xgrid, ygrid, 150,150, a.u, a.v,integration_time, dt)
factors = [0.1, 0.25, 0.5, 1]
factors = [0.5]
for factor in factors:
    ftle_dt = dt * factor
    print('computing FTLE field with dt ' + str(ftle_dt) +' and integration time ' + str(integration_time))
    traj_x, traj_y = get_traj(xgrid, ygrid, a.u, a.v, integration_time, ftle_dt)
    ftle = get_ftle(traj_x, traj_y, xgrid, ygrid, integration_time)

    print ftle.shape
    print('min FTLE: ' + str(np.min(ftle)))
    print('max FTLE: ' + str(np.max(ftle)))
    ftle = np.abs(ftle)

    fileName = out_folder + case_name + '_ftle'+str(ts)+'_inttime'+str(round(integration_time,3))+'_dtfactor'+str(factor)+'_'+direction_string+'.png'
    fig, ax = plt.subplots(1,1)
    x_start = xgrid[0,0]
    x_end = xgrid[0,-1]
    y_start = ygrid[0,0]
    y_end = ygrid[-1,0]

    #    plt.xlim(x_start*1.5, x_end*1.5)
    #    plt.ylim(y_start*1.5, y_end*1.5)

    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    #plt.xlim(x_start, x_end)
    #plt.ylim(y_start, y_end)
    cmap=plt.cm.jet
    cmap=plt.cm.bone_r
    color_delta = 10
    maxval = 1000
    maxval = np.max(ftle)
    minval = np.min(ftle)
    vals = np.arange(minval, maxval+color_delta, color_delta)
    norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)

    contf = plt.contourf(xgrid, ygrid, ftle, vals,norm=norm,extend='both',cmap=cmap)
    cbar = plt.colorbar(contf)
    plt.gca().set_aspect('equal')

    cbar.set_label('$FTLE$', fontsize=16)
    #plt.streamplot(xgrid, ygrid, a.u, a.v, density=2, linewidth=1, arrowsize=1, arrowstyle='->', color='k')
    #plt.scatter(x_vortex, y_vortex, color='#CD2305', s=40, marker='o')
    plt.draw()

    fig.savefig(fileName, dpi=600)
    write_FTLE_ASCII(out_folder + case_name + '_ftle'+str(ts)+'_inttime'+str(round(integration_time,3))+'_dtfactor'+str(factor)+'_'+direction_string+'.dat', x, z, ftle, shape=(rows,cols), title='FTLE', variables=['x', 'z', 'FTLE'])

