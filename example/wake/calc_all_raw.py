#/usr/bin/python
"""
Compute turbulence anisotropy quantities for a 2D planar time series from a TAU solution

Data input: consecutive time series data in plt format

Data output: one plt file each containing the anisotropy tensor, the eigenvalues and the barycentric coordinates

Required input:

case_name: string
    input for conf.WakeCaseParams() in order to obtain the parameters for the data reader
plane: string
    name of the plane to be read from the input data files
data_type: str
    typically just CRM_LSS
WCalign: bool
    set to True to rotate by angle of attack, set to False to align with precomputed wake centerline direction

Andreas Waldmann, 2020

"""


import os, sys

import time
import shutil
import numpy as np
import matplotlib.mlab as mlab
import matplotlib as mpl
import TAUpost.pyTecIO.tecreader as tecreader
import matplotlib.pyplot as plt
import scipy.signal
import TAUpost.wake.helpers.wake_stats as ws
from wake_config import WakeCaseParams
from TAUpost.wake.helpers.data_class import FieldSeries, WakeField


def get_rawdata(case_name, plane_name, case_type):

    param = WakeCaseParams(case_name, plane_name, case_type)
    print(param)
    # Get parameter dict from config file based on above input
    param.end_i = 5500

    # Get the data time series. The uvw data are arrays of shape (n_points, n_samples). dataset is a Tecplot dataset.
    in_data,dataset = tecreader.get_series(param.plt_path, param.zonelist, param.start_i, param.end_i, \
        read_velocities=True,read_cp=False, read_vel_gradients=False, stride = param.di, \
        parallel=False, verbose=True)

    # Create a FieldSeries object to hold the velocity vectors
    vel  = FieldSeries()
    vel.set_velocities(in_data['u'], in_data['v'], in_data['w'])

    print('done reading. shape of u: ' + str(in_data['u'].shape))
    # Get the coordinates as arrays and add them to the velocity data
    x,y,z = tecreader.get_coordinates(dataset, caps=True)

    vel.set_coords(x,y,z)
    wake = WakeField()
    wake.vel = vel
    wake.dataset = dataset
    wake.param = param
    wake.set_coords(x,y,z)


    # Return the FieldSeries object and the Tecplot dataset
    return wake


######################################################################
if __name__ == "__main__":
    out_folder = './results/'
    plane_name = 'eta0603'
    case_type = 'CRM_LSS'
    case_name = 'CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2'
    #par = WakeCaseParams(case_name, plane_name, case_type)

    wake = get_rawdata(case_name, plane_name, case_type)
    par = wake.param
    # Get the coordinates as arrays
    #x,y,z = tecreader.get_coordinates(dataset, caps=True)
    vel = wake.vel

    # Carry out rotation to the inflow direction
    print('point of model rotation:')
    print('x_PMR: ' +str(wake.param.x_PMR))
    print('z_PMR: ' +str(wake.param.z_PMR))

    wake.rotate_CS(18, 'WT')





    #wake.set_coords(x_WT, None, z_WT)
    #wake.set_coords(x_WT, None, z_WT)
    #vel.cs = 'WT'


    wake.compute_rstresses(do_save = True)
    wake.compute_anisotropy(do_save=True)


    #res.save_anisotropy()
    sys.exit(0)
    #anisotropy.compute(u_WT,v,w_WT)

    file_prefix = case_name+'_'+par.plane
    anisotropy.save(par.res_path, file_prefix)

    sys.exit(0)
    n_samples = u.shape[-1]



    '''
    uu = np.var(u, axis=-1)
    ww = np.var(w, axis=-1)
    uw = np.zeros(mean_u.shape)
    #aaa = (u-np.mean(u, keepdims=True))
    for i in range(len(mean_u)):
        uflu = u[i,:] - np.mean(u[i,:], keepdims = True)
        wflu = w[i,:] - np.mean(w[i,:], keepdims = True)
        uw[i] = np.mean(np.multiply(uflu, wflu))
    print('uw shape: ' + str(uw.shape))
    '''


    # Get the coordinates as arrays
    x,y,z = tecreader.get_coordinates(dataset, caps=True)


    # Carry out rotation to the inflow direction
    aoa = par.aoa
    x_PMR = par.x_PMR
    z_PMR = par.z_PMR
    print('x_PMR: ' +str(x_PMR))
    print('z_PMR: ' +str(z_PMR))

    ws.rotate_dataset(dataset, x_PMR, z_PMR, aoa)
    x_WT, z_WT = ws.transform_wake_coords(x,z, x_PMR, z_PMR, par.aoa)
    u_WT, w_WT = ws.rotate_velocities(u,v,w, x_PMR, z_PMR, par.aoa)

    # We need reynolds stresses for anisotropy calculation
    uu,vv,ww,uv,uw,vw = ws.calc_rstresses(u_WT,v,w_WT)
    mean_u = np.mean(u_WT, axis=-1)
    mean_v = np.mean(v, axis=-1)
    mean_w = np.mean(w_WT, axis=-1)

    kt = 0.5* (uu + vv + ww)

    # Compute the anisotropy tensor
    a_uu, a_vv, a_ww, a_uv, a_uw, a_vw = ws.compute_atensor(uu, vv, ww, uv, uw, vw, kt)
    # Compute second and third invariants of the anisotropy tensor
    invar2, invar3, ev = ws.compute_anisotropy_invariants(a_uu, a_vv, a_ww, a_uv, a_uw, a_vw)
    # Compute barycentric coordinates
    C, xb, yb = ws.compute_anisotropy_barycentric(ev)


    print('shape of C: ' + str(C.shape))



    # Save the results

    newvar=dict()
    newvar['a_uu'] = a_uu
    newvar['a_vv'] = a_vv
    newvar['a_ww'] = a_ww
    newvar['a_uv'] = a_uv
    newvar['a_uw'] = a_uw
    newvar['a_vw'] = a_vw
    varnames = newvar.keys()
    filename = par.res_path + case_name+'_'+par.plane+'_anisotropy_tensor.plt'
    save_plt(newvar, dataset, filename, addvars = True, removevars = True)

    newvar=dict()
    newvar['ev1'] = ev[0,:]
    newvar['ev2'] = ev[1,:]
    newvar['ev3'] = ev[2,:]
    varnames = newvar.keys()
    filename = par.res_path + case_name+'_'+par.plane+'_anisotropy_eigvals.plt'
    save_plt(newvar, dataset, filename, addvars = True, removevars = True)


    newvar = dict()
    newvar['C1'] = C[0,:]
    newvar['C2'] = C[1,:]
    newvar['C3'] = C[2,:]
    varnames = newvar.keys()
    filename = par.res_path + case_name+'_'+par.plane+'_anisotropy_components.plt'
    save_plt(newvar, dataset, filename, addvars = True, removevars = True)
