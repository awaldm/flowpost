import numpy as np
import matplotlib.pyplot as plt
import flowpost.common.data_helpers as dh
import logging
logger = logging.getLogger(__name__)


def lumley_map_boundaries():
    """Create a set of boundaries for Lumley's anisotropy invariant map.
    This is mainly for plotting

    :return: Jxn, Jyn: the x and y coordinate arrays of each of the three boundaries.
    These are in IIIa (x) and IIa (y) coordinates
    """
    #rechts: axisymmetry
    Jx1=np.linspace(0,2./9.,300)
    Jy1=(3./2.) * (Jx1 *(4./3.))**(2./3.)
    #% links: axisymmetry
    Jx2=np.linspace(0,-1./36.,50)
    Jy2=3./2. * (-Jx2  *4.0/3.0)**(2./3.)
    #% oben: 2-component-turbulence
    Jx3=np.linspace(-1./36.,2./9.,300)
    Jy3=2./9. + Jx3*2
    return Jx1, Jy1, np.flipud(Jx2), np.flipud(Jy2), Jx3, Jy3



def draw_lumley_map_1D(x, z, xlin, zlin, invar2, invar3, width=3, plot_target = 'SCREEN'):
    _, _,I2 = dh.interpolate_line(wake.x, wake.z,  None, None, in_data = invar2, num=None, xi = xlin, zi = zlin)
    _, _,I3 = dh.interpolate_line(wake.x, wake.z,  None, None, in_data = invar2, num=None, xi = xlin, zi = zlin)

    print('line x is from ' + str(xlin[0]) + ' to ' + str(xlin[-1]))
    
    Jx1, Jy1, Jx2, Jy2, Jx3, Jy3  = plot_an.lumley_map_boundaries()
    line_style, markers = setup_plot(plot_target, width, width*0.8)
    fig, ax = plt.subplots(1,1)
    plt.plot(Jx1, Jy1, Jx2, Jy2, Jx3, Jy3, color='k')
    pcm = plt.scatter(I3, I2, s=30, c = (xlin - x_TE) / c_local, cmap = plt.cm.viridis)
    cbar = plt.colorbar(pcm)#, ticks=[xlim[0], xlim[1]])
    cbar.set_label('$(x-x_{TE})/c_{front}$', labelpad=-2)
    #cbar.ax.set_yticklabels(['0', '3'])
    #cbar.ax.set_yticklabels([xlim[0], xlim[1]])
    adjustprops = dict(left=0.12, bottom=0.12, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
    #adjustprops = dict(left=0.14, bottom=0.2, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
    plt.subplots_adjust(**adjustprops)
    plt.xlabel(r'$III_a$', labelpad = 5)
    plt.ylabel(r'$II_a$', labelpad=-9)
    print('xlin0: ' + str(xlin[0]))
    print(I3[0])
    print(I2[0])


    return fig, ax, cbar
