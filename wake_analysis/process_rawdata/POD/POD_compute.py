'''
Carry out POD analysis on Tecplot-formatted binary time series flow data from
TAU. This is basically a wrapper around modred's POD computation functions
for work with TAU time series.

Currently, velocity input data is mostly hard-coded, but any sort of data can be entered.

Tested with modred 2.0.4

Andreas Waldmann, 2018
'''
import numpy as np
import os, sys
import tec_series.tecreader as tecreader
import wake_analysis.helpers.wake_stats as ws
import wake_analysis.wake_config as conf
import modred as mr
import tecplot as tp
from memory_profiler import profile

import POD_utils as utils



def meansubtract(u):
    return u - np.mean(u, axis=1, keepdims=True)

#@profile(precision=4)
def main():


    # Select the variables
    var = 'uvw'

    # Select the case name for lookup in the config file
    case_name = 'CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2'

    # Plane name
    plane = 'eta0283'

    # Case type, mainly for reference attributes
    case_type = 'CRM_LSS'

    #di = 10

    ##################################################

    # Get parameter dict from config file based on above input
    par = conf.WakeCaseParams(case_name, plane, case_type)


    # Number of modes to retain during computation
    num_modes = 50
    num_modes = 20

    # Number of mode shapes to write to disk
    output_modes = 10


    # Select variables by name
    varnames = ['u','v','w']
    #varnames = ['w']
    #varnames = ['cp']
    #u,_,w,dataset = process_wake(plt_path, case_name, planelist[0], zonelist[0], start_i, end_i, verbosename=True)
    di = 10

    out_path = par.res_path + str('POD/')
    print('will write to ' + out_path)


    if not os.path.isdir(out_path):
        print('directory not found, exiting')
        os.makedirs(out_path)
    print('reading data series...')
    #u,v,w,dataset = tecreader.get_series(par.plt_path, zonelist[0], start_i, end_i, read_velocities=True,read_cp=False, read_vel_gradients=False, stride = di)
    timeseries, dataset = tecreader.get_series(par.plt_path, par.zonelist, par.start_i, par.end_i, read_velocities=True,read_cp=False, read_vel_gradients=False, stride = par.di, parallel=False, verbose=True)

    print('done reading')

    u = timeseries['u']
    v = timeseries['v']
    w = timeseries['w']
    u, w = ws.rotate_velocities(u,None,w, par.x_PMR, par.z_PMR, par.aoa)

    # POD employs only the fluctuating quantities, but we still need the means to obtain sensible reconstruction later
    umean = np.mean(u, axis=1, keepdims=True)
    vmean = np.mean(v, axis=1, keepdims=True)
    wmean = np.mean(w, axis=1, keepdims=True)
    u = meansubtract(u)
    v = meansubtract(v)
    w = meansubtract(w)

    # u,v,w now contain fluctuating quantities

    #cpmean = np.mean(cp, axis=1, keepdims=True)
    #cpflu = cp - cpmean

    #del u,v,w

    #wflu = w - np.mean(w, axis=1, keepdims=True)
    #data = np.vstack((uflu,wflu))
    #varnames = ['u', 'w']
    #varnamestring =  '_'.join(varnames) # wow such pythonic
    #indata = wflu

    print('Stacking input data matrix')
    indata = np.vstack((u,v,w))
    print('Input data has shape ' + str(indata.shape))

    print('numpy flags of POD input data:')
    print(indata.flags)

    # Start POD algorithm
    modes, eigvals, eigvecs, coeffs = utils.compute_POD_modred(indata, num_modes)

    n_samples = indata.shape[-1]

    print('POD coefficient matrix shape: ' + str(coeffs.shape))
    #eigvals = eigvals / (n_samples * n_samples)

    #eigvals = np.sqrt(eigvals) / n_samples
    #eigvals = np.sqrt(eigvals)



    # this needs to be done only once, otherwise we get many variables with the same name
    ws.rotate_dataset(dataset, par.x_PMR, par.z_PMR, par.aoa)

    newvar=dict()
    newvar['umean'] = umean
    newvar['vmean'] = vmean
    newvar['wmean'] = wmean
    # save mean variables to file
    tecreader.save_plt(newvar, dataset, os.path.join(out_path,'means.plt'), addvars = True, removevars = True)


    import modal_utils as dutils
    newvar = dutils.unstack_mode(modes, varnames)

    #newvar['phi_w'] = modes[:,0]
    print('writing modes to ' + out_path + ' ...')

    # Write the first output_modes POD modes as Tecplot binary data using the existing dataset
    utils.write_mode_plt(newvar, dataset, out_path, 'mode', output_modes)


    # Write POD coefficient time series to an ASCII file
    time = np.linspace(par.start_t,par.end_t, n_samples,endpoint=True)
    utils.write_POD_coeffs(out_path, 'CRM', time, coeffs[:,0:num_modes])

    # Write POD mode eigenvalues to an ASCII file
    utils.write_POD_eigvals(out_path, 'CRM', eigvals)



if __name__ == "__main__":
    #rc('text', usetex=False)
    main()
