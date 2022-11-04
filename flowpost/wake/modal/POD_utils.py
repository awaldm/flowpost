import tecplot as tp
import numpy as np
import tecreader as tecreader
import flowpost.IO.pyTecIO.tec_modules as tec

import os
try:
    import modred as mr
except:
    print('modred not available')

# This should not be here
def tec_get_dataset(filename, zone_no=None, variables=['X', 'Y', 'Z']):
    import tecplot as tp
    '''
    Obtain a single Tecplot zone and its coordinates.
    Useful in a situation when further processing yields some data.
    '''
    if isinstance(zone_no,int):
        zones = [zone_no]
    else:
        zones = zone_no
    dataset = tp.data.load_tecplot(filename, zones=zone_no, variables=['X', 'Y', 'Z'], read_data_option = tp.constant.ReadDataOption.Replace)
    # even though we loaded only a single zone, we need to remove the preceding zombie zones
    # otherwise they will get written and we have no chance to read properly (with paraview)
    # this works only (and is necessary) when using the single zone/no extra datasetfile approach
    for i in range(zones[0]):
        dataset.delete_zones(0)

    return dataset



def write_mode_plt(newvar, dataset, mode_no, out_folder, filename):
#    newvar=dict()
#    newvar['phi_w'] = modes[:,mode_no]

    varnames = list(newvar.keys())


    offset = 0
    for keys, _ in newvar.items():
        dataset.add_variable(keys)

    #for keys, _ in newvar.iteritems():
    #    dataset.add_variable(keys)

    for zone in dataset.zones():
        zone_points = zone.num_points
        print(('zone has ' + str(zone_points) + ' points'))
        for var in varnames:
            #print var
            #print newvar[var].shape
            new_array = list(newvar[var][offset:offset+zone_points, mode_no])
            #print len(new_array)
            zone.values(var)[:] = new_array
        offset = offset + zone_points - 1
    verbosename = True

    outname = os.path.join(out_folder, filename + '_' + str(mode_no) + '.plt')
    print(('writing ' + str(outname)))
    tp.data.save_tecplot_plt(outname, dataset=dataset)
    for keys, _ in newvar.items():
        dataset.delete_variables(dataset.variable(keys))



def compute_POD(indata, num_modes):
    '''
    Compute the Proper Orthogonal Decomposition on the indata array using numpy

    '''
    n_samples = indata.shape[-1]

    # Obtain the correlation matrix C
    corrmat = np.dot(indata.T, indata) / n_samples
    print('shape of correlation matrix: ' + str(corrmat.shape))

    # Solve the eigenvalue problem for C: C v = lambda v
    eigvals, eigvecs = np.linalg.eig(corrmat)

    # Sort the eigenvalues by magnitude:
    # [::-1] inverts the order
    #eigvals[::-1].sort()

    # Compute the POD modes via dot product C*v
    modes = np.dot(indata, eigvecs)
    eigvecs = eigvecs.T

    # Apply a norm to the modes
    for i in range(num_modes):
        #coeffs[:,i] = eigvecs[:,i] * np.sqrt(eigvals[i] * n_samples)
        #modes[:,i] = modes[:,i] / np.sqrt(eigvals[i] * n_samples)
        modes[:,i] = modes[:,i] / np.linalg.norm(modes[:,i], ord=2)

    # Get coefficients a_i via a_i = phi * indata
    coeffs = compute_coeffs(modes, indata)

    return modes, eigvals, eigvecs, coeffs


def compute_POD_modred(indata, num_modes):
    """
    Compute POD using modred
    """

    n_samples = indata.shape[-1]

    print("using modred version ", mr.__version__)
    #modes_, ritz_vals, mode_norms, build_coeffs = mr.compute_DMD_matrices_snaps_method(fields.T, [], inner_product_weights=np.ones(fields.shape[1]), return_all=True)

    #modes, eigvals = mr.pod.compute_POD_matrices_direct_method(wflu, range(50), inner_product_weights=None, atol=1e-13, rtol=None, return_all=False)
    #modes, eigvals = mr.pod.compute_POD_matrices_snaps_method(wflu, range(50), inner_product_weights=None, atol=1e-13, rtol=None, return_all=False)
    #modes, eigvals = mr.pod.compute_POD_matrices_snaps_method(wflu, range(num_modes), inner_product_weights=None, atol=None, rtol=None, return_all=False)
    modes, eigvals, eigvecs, _ = mr.pod.compute_POD_matrices_snaps_method(indata, list(range(num_modes)), inner_product_weights=None, atol=None, rtol=None, return_all=True)
    # this should yield a correct correlation matrix
    #corrmat = np.dot(indata, indata.T)
    #coeffs = compute_coeffs(modes, indata)

    #print corrmat[0,:]


    #print(eigvals)

    #import matplotlib.pyplot as plt
    #plt.scatter(range(20),eigvals[0:20])
    #plt.savefig('modes.png')

    # problem: no matter whether np.dot or modred, we get three large eigenvalues in numpy vs. only two in matlab. whats the difference? the correlation matrix or the data ordering?
    # why is data ordering important?
    print(('modes shape: ' + str(modes.shape)))
    print(('eigenvectors shape: ' + str(eigvecs.shape)))
    print(('eigenvalues shape: ' + str(eigvals.shape)))
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

    print(('using ' + str(num_modes) + ' modes for reconstruction'))
    print(('shape of modes: ' + str(modes.shape)))
    print(('shape of coeffs: ' + str(coeffs.shape)))
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
    print(('size of A: ' + str(A.shape)))

    rec = np.zeros([modes.shape[0], num_timesteps, modes.shape[-1]])
    #rec = np.dot(modes[:,0:num_modes-1,0], A[time_offset:time_offset+num_timesteps, 0:num_modes-1].T)
    for i in range(modes.shape[-1]):
        rec[:,:,i] = np.dot(modes[:,0:num_modes-1,i], A[time_offset:time_offset+num_timesteps, 0:num_modes-1].T)


    print(('size of rec: ' + str(rec.shape)))
    return rec

def get_means(read_path = './', meanfilename = 'means.plt', meanvars = 'umean'):
    '''
    Read precomputed time-averaged data set from a Tecplot format binary file
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
    print(('reading mean variables ' + str(meanvars)))
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
    print(('getting dataset from ' + mode_filelist[0]))
    dataset = tecreader.tec_get_dataset(mode_filelist[0], zone_no = [0], dataset_only=True)
    print(dataset)
    return modes, dataset


def read_coeff(read_path = './AoA18_uflu/'):
    import pyTecIO_AW.tec_modules as tec

    vars, data= tec.read1D(os.path.join(read_path, 'CRM_POD_coeffs.dat'), verbose=True)

    print(vars)
    #print data.shape
    return data, vars



def POD_coeffs_from_python_folder(filename, skip=0):
    print('looking for POD coefficients in ' + filename)
    zones, dfs, _ = tec.readnew(filename, verbose=0)
    print(zones)
    data = dfs[0].to_numpy()
    sample = data[:,0]
    coeff = data[:,2:]
    print(coeff.shape)
    # Datei-Count
    coeff = coeff.astype(float) / len(sample)


    mode_arr = np.linspace(1,20,20)
    mode_arr = map(int, mode_arr)

    return sample, coeff.T, mode_arr



def compute_coeffs(modes, indata):
    return np.dot(indata.T, modes)


def write_POD_eigvals(out_path, title, eigvals):

    num_vals = len(eigvals)
    #sample = np.array(range(coeffs.shape[0]))
    # Generate Tecplot-readable header
    header_str = 'TITLE = ' + title + ' results\n'
    header_str += 'VARIABLES = "mode_no" "eigenvalue"'
    header_str += '\n'
    header_str += 'ZONE T = "'+title+' results"'
    sample = np.array(list(range(len(eigvals))))

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
    sample = np.array(list(range(coeffs.shape[0])))
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
    print(('shape of coeffs: ' + str(coeffs.shape)))
    print(('shape of time: ' + str(time.shape)))
    print(('shape of sample: ' + str(sample.shape)))

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
