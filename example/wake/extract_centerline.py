import numpy as np
import numpy.polynomial.polynomial as poly
import scipy.interpolate
"""
def interpolate_struct_data(x, z, var):
    xlim = (1.15, 1.75) # eta0603

    x0, z0 = xlim[0], -0.05
    x1, z1 = xlim[1], 0.25

    xi, zi = np.linspace(x0, x1, 100), np.linspace(z0, z1, 250)
    xmesh, zmesh = np.meshgrid(xi,zi)

    print('shape of xi: ' + str(xi.shape))
    print('shape of zi: ' + str(zi.shape))
    print('mesh shape: ' + str(xmesh.shape))

    return xi, zi, scipy.interpolate.griddata((x, z), var, (xmesh, zmesh), method='linear')
    #wi[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), w_WT[case], (xmesh, zmesh), method='cubic')
    #uu[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), uu_WT[case], (xmesh, zmesh), method='cubic')
    #vv[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), vv_WT[case], (xmesh, zmesh), method='cubic')
    #ww[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), ww_WT[case], (xmesh, zmesh), method='cubic')
    #uw[case] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), uw_WT[case], (xmesh, zmesh), method='cubic')
"""



# need means!
def fit_line(x, z):
    start = 0
    end = -1
    coefs = poly.polyfit(x[start:end], z[start:end], 1)
    ffit = poly.polyval(x, coefs)
    print(x)
    print(ffit)
    #np.savez(in_path + run_names[case]+'_wake_center_linear_fit', x=x, z=ffit)
    return x, np.squeeze(ffit)

