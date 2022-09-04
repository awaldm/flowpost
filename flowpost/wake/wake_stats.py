#/usr/bin/python
'''
This is a collection of various analysis functions for 2D wake plane time series data. They represent a mix of actual computation functions and IO/data handling routines.


'''

from __future__ import division

import numpy as np
#import h5py as hdf
import os, sys
import pandas as pd
#import tecplot as tp
#import pyTecIO_AW.tecreader as tecreader
import flowpost.autocorr as ac

############################################################################


def vorticity2D(velocity, dx, dy):
    ux, uy = np.gradient(velocity[:, :, 0], dx, dy)
    vx, vy = np.gradient(velocity[:, :, 1], dx, dy)
    return np.array(vx - uy).T


def rotate_dataset(dataset, x_PMR, z_PMR, alpha):
    offset = 0
    for zone in dataset.zones():
        array = zone.values('X')
        x = np.array(array[:]).T
        array = zone.values('Y')
        y = np.array(array[:]).T
        array = zone.values('Z')
        z = np.array(array[:]).T
    x, z = transform_wake_coords(x, z, x_PMR, z_PMR, alpha)


    coordvars = ['X', 'Y', 'Z']
    for zone in dataset.zones():
        zone_points = zone.num_points
        print('zone has ' + str(zone_points) + ' points')
        new_x = list(x[offset:offset+zone_points])
        new_y = list(y[offset:offset+zone_points])
        new_z = list(z[offset:offset+zone_points])
        zone.values('X')[:] = new_x
        zone.values('Y')[:] = new_y
        zone.values('Z')[:] = new_z
        offset = offset + zone_points - 1



def transform_wake_coords(x_old, z_old, x_PMR, z_PMR, alpha, axis='y'):
    '''
    transform coordinates of a plane, rotate by alpha
    currently only aoa rotation about y is implemented

    Parameters
    ----------
    x,y,z : float numpy arrays
        current coordinates
    x_PMR, z_PMR : floats
        coordinates of rotation point
    alpha : float
        angle in degrees

    Returns
    -------
    x_WT, z_WT : float numpy arrays
        rotated coordinates
    '''
    if axis is 'y':
        x_WT = (x_old - x_PMR) * np.cos(-1*np.radians(alpha)) - (z_old-z_PMR)* (np.sin(-1*np.radians(alpha))) + x_PMR
        z_WT = (x_old - x_PMR) * np.sin(-1*np.radians(alpha)) + (z_old-z_PMR)* (np.cos(-1*np.radians(alpha))) + z_PMR
        return x_WT, z_WT

def rotate_velocities(u,v,w, x_PMR, z_PMR, alpha):
    u_WT = u*np.cos(-1*np.radians(alpha)) - w * np.sin(-1*np.radians(alpha))
    w_WT = u*np.sin(-1*np.radians(alpha)) + w * np.cos(-1*np.radians(alpha))

    return u_WT, w_WT



def rotate_stresses(uu,vv,ww,uv=None, uw=None, vw=None, x_PMR=None, z_PMR=None, alpha=18.):
    uu_WT =  uu * (np.cos(-1*np.radians(alpha)))**2 - 2 * uw * np.sin(-1*np.radians(alpha)) * np.cos(-1*np.radians(alpha)) + ww * np.sin(-1*np.radians(alpha))**2
    vv_WT =  vv
    ww_WT =  uu * (np.sin(-1*np.radians(alpha)))**2 + 2 * uw * np.sin(-1*np.radians(alpha)) * np.cos(-1*np.radians(alpha)) + ww * np.cos(-1*np.radians(alpha))**2
    uv_WT = 0
    uw_WT = 0
    vw_WT = 0
    if uv is not None and vw is not None:
        uv_WT = uv * (np.cos(-1*np.radians(alpha))) - vw*np.sin(-1*np.radians(alpha))
        vw_WT = uv * np.sin(-1*np.radians(alpha)) + vw*np.cos(-1*np.radians(alpha))
    if uw is not None:
        uw_WT = uu*np.sin(-1*np.radians(alpha))*np.cos(-1*np.radians(alpha)) + uw * np.cos(-1*np.radians(alpha))**2 -np.sin(-1*np.radians(alpha))**2 - ww * np.sin(-1*np.radians(alpha))*np.cos(-1*np.radians(alpha))
    return uu_WT, vv_WT, ww_WT, uv_WT, uw_WT, vw_WT

def rotate_gradients(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, alpha):
    dudx_rot = dudx*np.cos(-1*np.radians(alpha)) - dudz * np.sin(-1*np.radians(alpha))
    dudy_rot = dudy
    dudz_rot = dudx*np.sin(-1*np.radians(alpha)) + dudz * np.cos(-1*np.radians(alpha))
    dvdx_rot = dvdx*np.cos(-1*np.radians(alpha)) - dvdz * np.sin(-1*np.radians(alpha))
    dvdy_rot = dvdy
    dvdz_rot = dvdx*np.sin(-1*np.radians(alpha)) + dvdz * np.cos(-1*np.radians(alpha))
    dwdx_rot = dwdx*np.cos(-1*np.radians(alpha)) - dwdz * np.sin(-1*np.radians(alpha))
    dwdy_rot = dwdy
    dwdz_rot = dwdx*np.sin(-1*np.radians(alpha)) + dwdz * np.cos(-1*np.radians(alpha))
    return dudx_rot, dudy_rot, dudz_rot, dvdx_rot, dvdy_rot, dvdz_rot, dwdx_rot, dwdy_rot, dwdz_rot
#$!AlterData
#  Equation = '{MSMT_uw} = {resolved-uu}*SIN(-18/180*PI)*COS(-18/180*PI) +{resolved-uw}*((COS(-18/180*PI))**2 -(SIN(-18/180*PI))**2) -{resolved-ww}*SIN(-18/180*PI)*COS(-18/180*PI)'
#$!AlterData
#  Equation = '{MSMT_vw} = {resolved-uv}*SIN(-18/180*PI) +{resolved-vw}*COS(-18/180*PI)'




def compute_fluctuations(u,v,w):
    '''
    subtract temporal mean from a data set

    Parameters
    ----------
    u,v,w : float numpy arrays
        assumes that the last dimension is the time

    Returns
    -------
    u_flu, v_flu, w_flu : float numpy arrays
        same shapes as the input
    '''

    time_dim = u.ndim - 1

    # without keepdims, mean reduces the dimensions and we cannot subtract due
    # to dimension mismatch
    u_flu = u - np.mean(u, axis=time_dim,keepdims=True)
    v_flu = v - np.mean(v, axis=time_dim,keepdims=True)
    w_flu = w - np.mean(w, axis=time_dim,keepdims=True)
    return u_flu, v_flu, w_flu

def compute_means(u,v,w):
    u_mean = np.mean(u, axis=-1,keepdims=True)
    v_mean = np.mean(v, axis=-1,keepdims=True)
    w_mean = np.mean(w, axis=-1,keepdims=True)
    return u_mean, v_mean, w_mean


def calc_gradient_means(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, dpdx=None, dpdy=None, dpdz=None):
    '''
    This computes the gradients of the mean velocity from time-resolved gradients
    '''
    mean_dudx = np.mean(dudx, axis=-1,keepdims=True)
    mean_dudy = np.mean(dudy, axis=-1,keepdims=True)
    mean_dudz = np.mean(dudz, axis=-1,keepdims=True)
    mean_dvdx = np.mean(dvdx, axis=-1,keepdims=True)
    mean_dvdy = np.mean(dvdy, axis=-1,keepdims=True)
    mean_dvdz = np.mean(dvdz, axis=-1,keepdims=True)
    mean_dwdx = np.mean(dwdx, axis=-1,keepdims=True)
    mean_dwdy = np.mean(dwdy, axis=-1,keepdims=True)
    mean_dwdz = np.mean(dwdz, axis=-1,keepdims=True)
    if dpdx is not None:
        mean_dpdx = np.mean(dpdx, axis=-1,keepdims=True)
    else:
        mean_dpdx = None
    if dpdy is not None:
        mean_dpdy = np.mean(dpdy, axis=-1,keepdims=True)
    else:
        mean_dpdy = None
    if dpdz is not None:
        mean_dpdz = np.mean(dpdz, axis=-1,keepdims=True)
    else:
        mean_dpdz = None
    return mean_dudx, mean_dudy, mean_dudz, mean_dvdx, mean_dvdy, mean_dvdz, mean_dwdx, mean_dwdy, mean_dwdz, mean_dpdx, mean_dpdy, mean_dpdz

def TKE_production(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, uu,vv,ww,uv,uw,vw, from_rstresses = True):
    # requires mean gradients/gradients of mean velocities
    #if mean_gradients_exist:
    print('shape of uu: '+str(uu.shape))
    print('shape of dudx: '+str(dudx.shape))
    #first_product = np.multiply(uu,dudx)
    #print('shape of first_product: '+str(first_product.shape))
    pk_normal = -1*(np.multiply(uu,dudx)+np.multiply(vv,dvdy)+np.multiply(ww,dwdz))
    pk_shear = -1*(np.multiply(uv,(dudy+dvdx)) + np.multiply(uw, (dudz+dwdx)) + np.multiply(vw, (dvdz+dwdy)))
    pk_total = pk_normal + pk_shear

    return pk_total, pk_normal, pk_shear



def calc_rstresses(u,v,w, return_dict=False):
    reshaped = False
    shape = u.shape
    newshape = shape[0:-1]
    if len(shape) > 2:
        u.reshape(shape[0]*shape[1],shape[2])
        v.reshape(shape[0]*shape[1],shape[2])
        w.reshape(shape[0]*shape[1],shape[2])
        reshaped = True

    uu = np.var(u, axis=-1, keepdims = True)
    ww = np.var(w, axis=-1, keepdims = True)
    vv = np.var(v, axis=-1, keepdims = True)
    print('uu shape: ' + str(uu.shape))
    uw = np.zeros(uu.shape)
    uv = np.zeros(uu.shape)
    vw = np.zeros(uu.shape)



    #aaa = (u-np.mean(u, keepdims=True))
    for i in range(uu.shape[0]):
        uflu = u[i,:] - np.mean(u[i,:], keepdims = True)
        vflu = v[i,:] - np.mean(v[i,:], keepdims = True)
        wflu = w[i,:] - np.mean(w[i,:], keepdims = True)
        uw[i] = np.mean(np.multiply(uflu, wflu), keepdims = True)
        vw[i] = np.mean(np.multiply(vflu, wflu), keepdims = True)
        uv[i] = np.mean(np.multiply(uflu, vflu), keepdims = True)

    print('uw shape: ' + str(uw.shape))
    if reshaped:
        uu.reshape(newshape)
        vv.reshape(newshape)
        ww.reshape(newshape)
        uv.reshape(newshape)
        uw.reshape(newshape)
        vw.reshape(newshape)
    if return_dict:
        rs = {}
        rs['uu'] = uu
        rs['vv'] = vv
        rs['ww'] = ww
        rs['uv'] = uv
        rs['uw'] = uw
        rs['vw'] = vw
        
        return rs
    else:
        return uu,vv,ww,uv,uw,vw

def get_rstresses(in_path, case=None, name=None, force=False, shape=(300,435), plane='eta0283_struct', source_path=None):
    '''
    returns a dict containing the Reynolds stresses of one case
    first, look for existing .mat files containing the Reynolds stresses
    if recompute is true, then the time series is loaded and the stresses are computed
    '''
    rstress = dict()
    stress_list = ['uu', 'vv', 'ww', 'uv', 'uw', 'vw', 'kt']
#    print 'looking for wake data in ', in_path, ' ...'
    recompute  = False

    if force == False:
        # first check if ANY rstress component is mixing
        # if yes -> recompute all
        matfile = in_path + plane + '_rstresses.mat'
        print('looking for existing Reynolds stresses file ' + matfile +' ...')
        if os.path.isfile(matfile) == False:
            recompute = True
            print('file not found, recomputing...')
        else:
            print('file found!')
        #for comp in stress_list:
        #    matfile = in_path + plane+'_'+comp+'.mat'
        #    if os.path.isfile(matfile) == False:
        #        recompute = True
        #        break
    else: # doesnt matter, compute anew in any case
        recompute = True

    if recompute:
        print('Reynolds stresses not computed yet, computing from raw time series data...')
        # caution, this takes the paths and shapes from the present script namespace.
        # this is not necessarily the safest option
#        x, z, u, v, w = reader.get_struct_wake(source_path, in_path, name, rows, cols)
        #x, z, u, v, w = reader.get_struct_wake(source_path, name, shape=shape)
        import pyTecIO_AW.read_2d_wake_timeseries as reader
        x,z,u,v,w = reader.get_struct_wake(in_path, case=case, name=name, shape=(300,435), coord_path=None, out_path=None)

        u,v,w = compute_fluctuations(u,v,w)
        uu,vv,ww,uv,uw,vw,kt = compute_rstresses(u,v,w)
        rstress['uu'] = uu
        rstress['vv'] = vv
        rstress['ww'] = ww
        rstress['uv'] = uv
        rstress['uw'] = uw
        rstress['vw'] = vw
        rstress['kt'] = kt

    '''
    else:
        print('loading Reynolds stress data from HDF5 file...')
        f = hdf.File(matfile, 'r')
        for name in f:
            print('variable '+name+' found.')
        #    for comp in stress_list:
            var = f.get(name)
            var = np.array(var)
            rstress[name] = var
#            print 'shape of '+name+': ' + str(rstress[name].shape)
        f.close()
        print('done loading existing Reynolds stresses')
    '''

#            rstress
#        for comp in stress_list:
#            matfile = in_path + plane+'_'+comp+'.mat'
#            f = hdf.File(matfile, 'r')
#            for name in f:
#                print 'variable '+name+' found.'
#
#            var = f.get(comp)
#            var = np.array(var)
#            rstress[comp] = var
#            print 'shape of '+comp+': ' + str(rstress[comp].shape)
#            f.close()

    return rstress


def compute_rstresses_1D(u,v,w):
    uu = np.mean(u**2, axis=-1)
    vv = np.mean(v**2, axis=-1)
    ww = np.mean(w**2, axis=-1)
    uv = np.mean(u*v, axis=-1)
    uw = np.mean(u*w, axis=-1)
    vw = np.mean(v*w, axis=-1)
    kt = 0.5*(uu+vv+ww)
    return uu, vv, ww, uv, uw, vw, kt




def compute_rstresses(u,v,w):
    rst_dims = u.ndim - 1
    rst_shape = u.shape[:-1]
    uu = np.zeros([u.shape[0], u.shape[1]])
    vv = np.zeros([u.shape[0], u.shape[1]])
    ww = np.zeros([u.shape[0], u.shape[1]])
    uv = np.zeros([u.shape[0], u.shape[1]])
    uw = np.zeros([u.shape[0], u.shape[1]])
    vw = np.zeros([u.shape[0], u.shape[1]])
    kt = np.zeros([u.shape[0], u.shape[1]])

    for r in range(u.shape[0]):
        for c in range(u.shape[1]):
            uu[r,c] = np.mean(np.multiply(u[r,c,:], u[r,c,:]))
            vv[r,c] = np.mean(np.multiply(v[r,c,:], v[r,c,:]))
            ww[r,c] = np.mean(np.multiply(w[r,c,:], w[r,c,:]))
            uv[r,c] = np.mean(np.multiply(u[r,c,:], v[r,c,:]))
            uw[r,c] = np.mean(np.multiply(u[r,c,:], w[r,c,:]))
            vw[r,c] = np.mean(np.multiply(v[r,c,:], w[r,c,:]))
            kt[r,c] = 0.5 * (uu[r,c] + vv[r,c] + ww[r,c])

    return uu, vv, ww, uv, uw, vw, kt



import math


'''
n_samples = size(vel.u,1);
n_points = size(vel.u,2);

QA = zeros(n_samples, n_points);
%S = zeros(n_samples, n_points, 4);
S = zeros(n_points, 4);
for point=1:n_points
    for sample=1:n_samples
        %        prod = vel.u(sample, point) * vel.w(sample, point);
        if (vel.u(sample,point) > 0 && vel.w(sample,point) > 0) % Q1, outward
            QA(sample, point) = 1;
            %S(sample, point, 1) = S(sample, point, 1) + vel.u(sample,point) * vel.w(sample,point);
        elseif (vel.u(sample,point) < 0 && vel.w(sample,point) > 0) % Q2, ejection
            QA(sample, point) = 2;
            %S(sample, point, 1) = S(sample, point, 2) + vel.u(sample,point) * vel.w(sample,point);
        elseif (vel.u(sample,point) < 0 && vel.w(sample,point) < 0) % Q3, inward
            QA(sample, point) = 3;
            %S(sample, point, 1) = S(sample, point, 3) + vel.u(sample,point) * vel.w(sample,point);
        else
            QA(sample, point) = 4; % Q4, sweep
            %S(sample, point, 1) = S(sample, point, 4) + vel.u(sample,point) * vel.w(sample,point);
        end;
    end;

    for q=1:4
        ind = find(QA(:,point) == q);
        S(point,q) = sum(vel.u(ind,point) .* vel.w(ind,point)) / n_samples;
    end;
end;

% df = duration fraction
df = zeros(4, n_points);
for point=1:n_points
    for quadrant=1:4
        df(quadrant, point) = sum(QA(:,point) == quadrant);
    end;
end;
'''

def get_quadrants(data_u, data_w):
    '''
    data_u is a mean-subtracted data set of shape (n_points, n_samples)
    we need a duration fraction and a stress fraction and exuberance
    for that, we compute at each point and time step the QA value which is 1,2,3 or 4
    '''
    n_points = data_u.shape[0]
    n_samples = data_w.shape[1]
    #print('spatial points: ' + str(n_points))
    #print('temporal samples: ' + str(n_samples))
    shear = np.zeros(n_samples)
    quadrant = np.zeros(n_samples)
    q = np.zeros([n_points, n_samples])
    sf = np.zeros([n_points, 4])
    df = np.zeros([n_points, 4])

    for p in range(n_points):
        quadrant = np.zeros(n_samples)

        # time series at each point
        #u = data_u[p,:]
        #w = data_w[p,:]


        for t in range(n_samples):

            u = data_u[p,t]
            w = data_w[p,t]
            if u > 0 and w > 0: # Q1, outward interaction
                quadrant[t] = 1
                #q[0,t] = QA[0,t] + u*w
            elif u < 0 and w > 0: # Q2, ejection
                #QA[1,t] = QA[1,t] + u*w
                quadrant[t] = 2
            elif u < 0 and w < 0: # Q3, inward interaction
                quadrant[t] = 3
                #QA[2,t] = QA[2,t] + u*w
            else: # Q4, sweep
                quadrant[t] = 4
                #QA[3,t] = QA[3,t] + u*w
        # time series at each point
        u = data_u[p,:]
        w = data_w[p,:]
        uw = np.mean(np.multiply(u,w))
        #uw[i] = np.mean(np.multiply(uflu, wflu), keepdims = True)

        for q in range(4):
            #QA[q,:] = QA[q,:] / n_points
            #df[p,q] = np.sum(quadrant[np.where(quadrant == q+1)]) * shear[t] / n_samples
            # duration fraction: fraction of time spent in each quadrant
            q_ind = np.where(quadrant == q+1)
            df[p,q] = len(quadrant[q_ind]) / n_samples
            sf[p,q] = np.sum(np.multiply(u[q_ind], w[q_ind])) / n_samples
            
            #if p==5500:
                #print(uw)
                #print(str(np.mean(np.multiply(u[q_ind], w[q_ind]))))
                #print(str(sf[p,q]))            
            #sf[p,q] = uw

    return df, sf

def compute_field_acf(data, maxlags=300):
    '''
    compute autocorrelation for entire field
    reshape if necessary, return in original shape
    '''

    if data.ndim == 3:
        #fields = np.zeros([rows*cols, n_samples, 1])
        shape_flat = (data.shape[0] * data.shape[1], data.shape[2])
        field = data.reshape(shape_flat)
    else:
        field = data

    n_points = field.shape[0]
    n_samples = field.shape[1]
    print('n_points: ' + str(n_points))


    acf = np.zeros([n_points, n_samples])
    for i in range(n_points):
        acf[i,:] = ac.autocorr(field[i,:])
    return acf.reshape(data.shape)

def compute_field_acf_index(data, threshold=0.2):
    '''
    compute for each field point the index at which the threshold drops below the specified value
    this is basically part of the timescale computation, however for the purpose of determination of independent
    samples this has its own purpose
    '''
    if data.ndim == 3:
        #fields = np.zeros([rows*cols, n_samples, 1])
        shape_flat = (data.shape[0] * data.shape[1], data.shape[2])
        field = data.reshape(shape_flat)
    else:
        field = data
    n_points = field.shape[0]
    n_samples = field.shape[1]
    print('n_points: ' + str(n_points))

    ind = np.zeros([n_points])
    for i in range(n_points):
        ind[i] = np.argmax(field[i,:] <= threshold)
    if data.ndim == 3:
        ind = ind.reshape(data.shape[0], data.shape[1])
    return ind

