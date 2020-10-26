'''
Utilities for POD computation in python. These need a lot of cleanup.


Andreas Waldmann, 2017
'''


import tecplot as tp
import numpy as np
import pyTecIO_AW.tecreader as tecreader
import pyTecIO_AW.tec_modules as tec
import os
import modred as mr

def write_mode_plt(out_data, dataset, out_folder, filename, output_modes):
    for i in range(output_modes):
        mode = dict()
        for var in out_data.keys():
            mode[var] = np.squeeze(np.asarray(out_data[var][:,i])) # need to cast to array, otherwise numpy's old matrix primitive is used which is incompatible with ravel()
        tecreader.save_plt(mode, dataset, out_folder+filename+'_'+str(i)+'.plt', addvars = True, removevars = True)



def compute_POD(indata):
    '''
    Straightforward implementation of the POD algorithm using numpy only. I recommend using compute_POD_modred

    Parameters
    ----------

    indata : numpy array
        m, n input matrix with m points and n snapshots

    Raises
    ------


    Returns
    -------
    modes : (m, n) numpy array
        POD modes, each column is a mode
    eigvals : numpy array
        holds the vector of POD eigenvalues
    eigvevs : numpy array
        the matrix of POD eigenvectors
    coeffs : numpy array
        the vector of POD coefficients
        


    '''
    # Temporal samples
    n_samples = indata.shape[-1]

    # Assemble the correlation matrix C
    corrmat = np.dot(indata.T, indata) / n_samples
    print('shape of correlation matrix: ' + str(corrmat.shape))

    # Carry out eigendecomposition of C
    eigvals, eigvecs = np.linalg.eig(corrmat)
    eigvals[::-1].sort()

    # Compute POD modes from dot product of data and the eigenvectors: phi = Uv
    modes = np.dot(indata, eigvecs)
    eigvecs = eigvecs.T

    # Normalize the modes
    for i in range(num_modes):
        #coeffs[:,i] = eigvecs[:,i] * np.sqrt(eigvals[i] * n_samples)
        #modes[:,i] = modes[:,i] / np.sqrt(eigvals[i] * n_samples)
        modes[:,i] = modes[:,i] / np.linalg.norm(modes[:,i], ord=2)

    # Compute the POD mode coefficients
    coeffs = compute_coeffs(modes, indata)

    return modes, eigvals, eigvecs, coeffs


def compute_POD_modred(indata, num_modes):

    n_samples = indata.shape[-1]

    print("using modred version ", mr.__version__)
    #modes_, ritz_vals, mode_norms, build_coeffs = mr.compute_DMD_matrices_snaps_method(fields.T, [], inner_product_weights=np.ones(fields.shape[1]), return_all=True)

    #modes, eigvals = mr.pod.compute_POD_matrices_direct_method(wflu, range(50), inner_product_weights=None, atol=1e-13, rtol=None, return_all=False)
    #modes, eigvals = mr.pod.compute_POD_matrices_snaps_method(wflu, range(50), inner_product_weights=None, atol=1e-13, rtol=None, return_all=False)
    #modes, eigvals = mr.pod.compute_POD_matrices_snaps_method(wflu, range(num_modes), inner_product_weights=None, atol=None, rtol=None, return_all=False)
    atol = 1e-13 # N.B.: the version used on miracoolix with python 2.7 had atol set to None
    modes, eigvals, eigvecs, _ = mr.pod.compute_POD_matrices_snaps_method(indata, range(num_modes), inner_product_weights=None, atol=atol, rtol=None, return_all=True)
    # this should yield a correct correlation matrix
    #corrmat = np.dot(indata, indata.T)
    #coeffs = compute_coeffs(modes, indata)


    #import matplotlib.pyplot as plt
    #plt.scatter(range(20),eigvals[0:20])
    #plt.savefig('modes.png')

    # problem: no matter whether np.dot or modred, we get three large eigenvalues in numpy vs. only two in matlab. whats the difference? the correlation matrix or the data ordering?
    # why is data ordering important?
    print('modes shape: ' + str(modes.shape))
    print('eigenvectors shape: ' + str(eigvecs.shape))
    print('eigenvalues shape: ' + str(eigvals.shape))
    coeffs = np.zeros([n_samples, num_modes])
    coeffs = np.zeros_like(eigvecs)
    coeffs = compute_coeffs(modes, indata)


    return modes, eigvals, eigvecs, coeffs


def POD_reconstruct(coeffs, modes, timesteps=10):
    if isinstance(timesteps, int):
        num_timesteps = timesteps
        coeff_offset = 2
        time_offset = 0
    elif isinstance(timesteps, tuple):
        num_timesteps = timesteps[1] - timesteps[0]
        coeff_offset = 2
        time_offset = timesteps[0]
    num_modes = modes.shape[-2]

    print('using ' + str(num_modes) + ' modes for reconstruction')
    print('shape of modes: ' + str(modes.shape))
    print('shape of coeffs: ' + str(coeffs.shape))
    #coeff_offset = 2

    reconstructed = np.zeros([modes.shape[0], num_timesteps])
    for t in range(num_timesteps):
        recon = np.zeros(modes.shape[0])
        for mode in range(num_modes):
            #print coeffs[t, mode + coeff_offset]**2
            #print recon.shape
            #print modes[:,mode,0].shape


            recon = recon + coeffs[t, mode + coeff_offset] * modes[:,mode,0]
        #print reconstructed.shape
        reconstructed[:,t] = recon
    A = coeffs[:,coeff_offset:]
    print('size of A: ' + str(A.shape))

    rec = np.zeros([modes.shape[0], num_timesteps, modes.shape[-1]])
    #rec = np.dot(modes[:,0:num_modes-1,0], A[time_offset:time_offset+num_timesteps, 0:num_modes-1].T)
    for i in range(modes.shape[-1]):
        rec[:,:,i] = np.dot(modes[:,0:num_modes-1,i], A[time_offset:time_offset+num_timesteps, 0:num_modes-1].T)
    
    
    print('size of rec: ' + str(rec.shape))
    return rec

def get_means(read_path = './', meanfilename = 'means.plt', meanvars = 'umean'):
    '''
    Read mean data set from Tecplot format binary file
    Typically used for reconstruction

    Parameters
    ----------

    read_path : string
        folder containing the mean file
        
    meanfilename : string
        file name

    meanvars : list of strings
        names of the mean variables to load

    '''
    
    data = {}
    print('reading mean variables ' + str(meanvars))
    meandataset = tp.data.load_tecplot(read_path + 'means.plt', zones=[0], variables = meanvars, read_data_option = tp.constant.ReadDataOption.Replace)
    for zone in meandataset.zones():
        for variable in meanvars:
            #print variable
            array = zone.values(variable)
            data[variable] = np.array(array[:]).T
    #means = tec_get_dataset(read_path + 'umean.plt', [0], varnames=[varnames], szplt=False)
    return data


def get_modes(read_path = './', modelist = [0,1], varname = 'phi_u'):
    mode_filelist = [read_path + 'mode_' + str(x) + '.plt' for x in modelist]
    modes = tecreader.read_series(mode_filelist, [0], varnames=varname, szplt=False)

    #for mode in modelist:
    #    if mode == 0
    print('getting dataset from ' + mode_filelist[0])
    dataset = tecreader.tec_get_dataset(mode_filelist[0], zone_no = [0], dataset_only=True)
    print(dataset)
    return modes, dataset


def read_coeff(read_path = './AoA18_uflu/'):

    vars, data= tec.read1D(os.path.join(read_path, 'CRM_POD_coeffs.dat'), verbose=True)

    print(vars)
    #print data.shape
    return data, vars

def compute_coeffs(modes, indata):
    '''
    Compute POD coefficients from modes, i.e. performs the operation
    U.T \cdot phi

    The result is a 2D numpy array
    '''
    return np.dot(indata.T, modes)


def write_POD_eigvals(out_path, title, eigvals):

    num_vals = len(eigvals)
    #sample = np.array(range(coeffs.shape[0]))
    # Generate Tecplot-readable header
    header_str = 'TITLE = ' + title + ' results\n'
    header_str += 'VARIABLES = "mode_no" "eigenvalue"'
    header_str += '\n'
    header_str += 'ZONE T = "'+title+' results"'
    sample = np.array(range(len(eigvals)))

    txt_matrix = np.c_[ sample, eigvals ]
    np.savetxt(os.path.join(out_path,title+'_POD_eigvals.dat'), \
        txt_matrix, \
        delimiter = ' ', \
        header = header_str,
        comments='')

def write_POD_coeffs(out_path, title, time, coeffs):
    '''
    Write DMD results to an ASCII text file in Tecplot format

    Parameters
    ----------

    out_path : string
        the folder where the results are written to

    title : string
        identified of the case, used in file name and title

    eigvals : complex
        array of complex eigenvalues

    growthrate : double
        mode growth rate, i.e. real part of lambda_dmd

    frequency : double
        mode frequency in Hz, i.e. imaginary part of lambda_dmd divided by 2pi

    SV : double
        singular values of the mode

    b : dict of numpy arrays
        mode amplitudes, variable number of fields
        all are written to file
    '''

    num_coeffs = coeffs.shape[1]
    sample = np.array(range(coeffs.shape[0]))
    # Generate Tecplot-readable header
    header_str = 'TITLE = ' + title + ' results\n'
    header_str += 'VARIABLES = "sample" "time"'
    for num in range(num_coeffs):
        header_str += ' "a' + str(num+1).zfill(2) + '"'
    header_str += '\n'
    header_str += 'ZONE T = "'+title+' results"'


    #print('shape of SV: ' + str(SV.shape))
    #print('shape of freq: ' + str(freq.shape))
    #print('shape of growthrate: ' + str(growthrate.shape))


    # Generate the data table
    print('shape of coeffs: ' + str(coeffs.shape))
    print('shape of time: ' + str(time.shape))
    print('shape of sample: ' + str(sample.shape))

    txt_matrix = np.c_[ sample, time ]


    # Append the appropriate number of columns containing the various amplitude definitions
    #for key, value in :
    #    print('shape of b '+str(key) + ' : ' + str(value.shape))
    #    txt_matrix = np.c_[txt_matrix, value]
    txt_matrix = np.c_[txt_matrix, coeffs]
    # write to file
    np.savetxt(os.path.join(out_path, title + '_POD_coeffs.dat'), \
        txt_matrix, \
        delimiter = ' ', \
        header = header_str,
        comments='')    
