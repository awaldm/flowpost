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
import flowpost.IO.pyTecIO.tecreader as tecreader
import matplotlib.pyplot as plt
import scipy.signal
import flowpost.wake.helpers.wake_stats as ws
from wake_config import WakeCaseParams
from flowpost.wake.helpers.data_class import FieldSeries, WakeField


def get_rawdata(case_name, plane_name, case_type):

    param = WakeCaseParams(case_name, plane_name, case_type)
    print(param)
    # Get parameter dict from config file based on above input
    param.end_i = 5500
    param.plt_path = '/home/andreas/data/CRM_example_data/low_speed/' 
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

    wake.rotate_CS('WT')





    #wake.set_coords(x_WT, None, z_WT)
    #wake.set_coords(x_WT, None, z_WT)
    #vel.cs = 'WT'


    wake.vel.n_samples = wake.vel.u.shape[-1]
    print(wake.stats.__dict__)
    #wake.compute_rstresses(do_save = True)
    wake.compute_anisotropy(do_save = True)
    #wake.compute_independent_samples(do_save = True)
    #wake.compute_PSD([], do_save = True)
    #wake.compute_skew_kurt(do_save = True)


    #res.save_anisotropy()
