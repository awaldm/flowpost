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
#os.environ["LD_LIBRARY_PATH"]="/usr/local/tecplot/360ex_2021r1/bin/llvm:/usr/local/tecplot/360ex_2021r1/bin/osmesa:/usr/local/tecplot/360ex_2021r1/bin:/usr/local/tecplot/360ex_2021r1/bin/sys-util:/usr/local/tecplot/360ex_2021r1/bin/Qt${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
#os.environ["LD_LIBRARY_PATH"]="/usr/local/tecplot/360ex_2021r1/bin/llvm:/usr/local/tecplot/360ex_2021r1/bin/osmesa:/usr/local/tecplot/360ex_2021r1/bin:/usr/local/tecplot/360ex_2021r1/bin/sys-util:/usr/local/tecplot/360ex_2021r1/bin/Qt" + ":" + os.path.expandvars("$LD_LIBRARY_PATH")
#os.environ["TECPLOT_SDK_CONFIG"]="--osmesa"

import time
import shutil
import numpy as np
import matplotlib.mlab as mlab
import matplotlib as mpl
import tecreader as tecreader
import matplotlib.pyplot as plt
import scipy.signal
import flowpost.wake.wake_stats as ws
from flowpost.configs.wake_config import WakeCaseParams
from flowpost.wake.data import FieldSeries, WakeField
import extract_centerline as ex
import flowpost.utils.loaders




######################################################################
if __name__ == "__main__":
    # We sometimes need to create and destroy a figure BEFORE loading any Tecplot data
    fig, ax = plt.subplots(1,1)
    plt.close()

    out_folder = './results/'
    plane_name = 'eta0603'
    case_type = 'CRM_LSS'
    case_name = 'CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2'
    #par = WakeCaseParams(case_name, plane_name, case_type)

    wake = flowpost.utils.loaders.get_rawdata(case_name, plane_name, case_type)

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

    print(wake.x)
    wake.vel.n_samples = wake.vel.u.shape[-1]
    print(wake.vel_stats.__dict__)

    # compute velocty means
    wake.compute_means()
    wake.save_means()

    # interpolate mean velocity onto a regular grid
    #xi, zi, ui = ex.interpolate_struct_data(wake.x, wake.z, wake.stats.mean['u'])

    # get wake minimum position
    #wake_min_pos = ex.wake_min_pos(xi,zi,ui)
    #print(xi)
    #plt.plot(xi, wake_min_pos)

    # fit a rough linear approximation of the wake centerline
    #_, fit_line = ex.fit_line(xi, wake_min_pos)

    #plt.plot(xi, fit_line)
    #plt.show()
    #plt.close()
    #wake.compute_rstresses(do_save = True)
    #wake.save_rstresses(wake.stats.rs.as_dict())

    # get the anisotropy data including the barycentric coordinates
    #wake.compute_anisotropy(do_save = True)

    #from plot_anisotropy import draw_barycentric
    # draw anisotropy along centerline in barycentric coordinates
    #draw_barycentric(wake.stats.an.C, wake.stats.an.xb, wake.stats.an.yb, wake.x, wake.z, xi, wake_min_pos)

    #wake.compute_independent_samples(do_save = True)
    #wake.compute_PSD([], do_save = True)
    #wake.compute_skew_kurt(do_save = True)


    #res.save_anisotropy()
    """
    ## Run POD
    from flowpost.wake.modal import POD_utils, modal_utils

    # Set the number of modes to keep
    num_modes = 10



    # Set what to feed into POD algorithm, i.e. one or multiple variables

    # Get the fluctuations
    uflu = wake.vel.u - wake.stats.mean['u']
    vflu = wake.vel.v - wake.stats.mean['v']
    wflu = wake.vel.w - wake.stats.mean['w']

    indata = np.vstack((uflu, vflu, wflu))

    # Run POD via own hand-coded numpy implementation
    modes, eigvals, eigvecs, coeffs = POD_utils.compute_POD(indata, num_modes)

    # TODO: get this from somewhere else
    varnames = ['u','v','w']

    # Write modes to Tecplot binary files
    newvar = modal_utils.unstack_mode(modes, varnames)
    for i in range(num_modes):
        POD_utils.write_mode_plt(newvar, wake.dataset, i, wake.param.res_path, 'mode')

    # Create time vector for the POD coefficients.
    POD_time = np.linspace(wake.param.start_t,wake.vel.n_samples*wake.param.dt + wake.param.start_t, wake.vel.n_samples,endpoint=True)

    POD_utils.write_POD_coeffs(wake.param.res_path, case_name, POD_time, coeffs[:,0:num_modes])
    POD_utils.write_POD_eigvals(wake.param.res_path, case_name, eigvals)

    """
