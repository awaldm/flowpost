import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import tecplot as tp
import wake_config as conf

import matplotlib.ticker as ticker

width = 3
#from helpers.setup_plot import *
#plot_style = 'THESIS'
#line_style, markers = setup_plot('THESIS', width, width*0.8)



def lumley_map():
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




def draw_map(x, z, xlin, zlin, invar2, invar3):
#xi, zi = np.linspace(x0, x0, num), np.linspace(z0, z1, num)
    I2 = scipy.interpolate.griddata((x, z), invar2, (xlin, zlin), method='linear')
    I3 = scipy.interpolate.griddata((x, z), invar3, (xlin, zlin), method='linear')

    print('line x is from ' + str(xlin[0]) + ' to ' + str(xlin[-1]))

    Jx1, Jy1, Jx2, Jy2, Jx3, Jy3  = lumley_map()
    width = 3
    line_style, markers = setup_plot('THESIS', width, width*0.8)

    fig, ax = plt.subplots(1,1)
    plt.plot(Jx1, Jy1, Jx2, Jy2, Jx3, Jy3, color='k')
    pcm = plt.scatter(I3, I2, s=2, c = xlin, cmap = plt.cm.viridis)
    #cbar = plt.colorbar(pcm, ticks=[1.05, 1.65])
    #cbar.set_label('$(x-x_{TE})/c_{local}$', labelpad=-5)
    #cbar.ax.set_yticklabels(['0', '3'])
    adjustprops = dict(left=0.12, bottom=0.12, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
    adjustprops = dict(left=0.14, bottom=0.2, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
    plt.subplots_adjust(**adjustprops)
    plt.xlabel(r'$III_a$', labelpad = 5)
    plt.ylabel(r'$II_a$', labelpad=-9)

    img_name = 'wake_centerline_anisotropy_lumley.pdf'
    plt.savefig(img_name)
    print('written ' + img_name)

    plt.close()

def draw_barycentric(C, xb, yb, x, z, xlin, zlin):
    zlin = np.squeeze(zlin)

    do_annotate = False
    xbi = scipy.interpolate.griddata((x, z), xb, (xlin, zlin), method='linear')
    ybi = scipy.interpolate.griddata((x, z), yb, (xlin, zlin), method='linear')
    print(x.shape)
    print(xlin.shape)
    print(zlin.shape)
    print(xbi.shape)

    width = 2
    fig, ax = plt.subplots(1,1)
#connectpoints()
    x1, y1 = [0, 0.5], [0, np.sqrt(3.0)/2.0]
    x2, y2 = [0, 1], [0, 0]
    x3, y3 = [1, 0.5], [0, np.sqrt(3.0)/2.0]
    plt.plot(x1, y1, x2, y2, x3, y3, color='k')
#plt.xlim([0,1])
#plt.ylim([0,1])
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    plt.axis('off')
    adjustprops = dict(left=0.05, bottom=0.05, right=0.97, top=0.97, wspace=0.2, hspace=0.2)       # Subplot properties
    plt.subplots_adjust(**adjustprops)

    plt.scatter(xbi, ybi, s=2, c = xlin, cmap=plt.cm.viridis)

    if do_annotate:
        ax.annotate('upstream', xy=(xbi[0], ybi[0]), xytext=(-20,-20),
                            textcoords='offset points', ha='center', va='top',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=1),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                                                        color='black'), fontsize=11)

        ax.annotate('downstream', xy=(xbi[-1], ybi[-1]), xytext=(-40,40),
                            textcoords='offset points', ha='center', va='top',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=1),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.5',
                                                                        color='black'), fontsize=11)

#ax.text(0, 0, r'$x_{2c}$', color='black' , fontsize=12, ha='right', va='top')
#ax.text(x2[1], y2[1], r'$x_{1c}$', color='black' , fontsize=12, ha='left', va='top')
#ax.text(x1[1], y1[1]+0.03, r'$x_{3c}$', color='black' , fontsize=12, ha='center', va='bottom')

    ax.text(0, 0, r'$2c$', color='black' , fontsize=12, ha='right', va='top')
    ax.text(x2[1], y2[1], r'${1c}$', color='black' , fontsize=12, ha='left', va='top')
    ax.text(x1[1], y1[1]+0.03, r'${3c}$', color='black' , fontsize=12, ha='center', va='bottom')

# Rotate angle
    angle = 45
#trans_angle = plt.gca().transData.transform_angles(np.array((45,)), l2.reshape((1, 2)))[0]
# Plot text
    th1 = plt.text(0.2, np.sqrt(3)/4.0+0.05, 'axisymmetric contraction', fontsize=11,  rotation=60, rotation_mode='anchor', ha='center', va='center')
    th2 = plt.text(0.8, np.sqrt(3)/4.0+0.05, 'axisymmetric expansion', fontsize=11,  rotation=-60, rotation_mode='anchor', ha='center', va='center')

    adjustprops = dict(left=0.05, bottom=0.07, right=0.95, top=0.93, wspace=0.2, hspace=0.2)       # Subplot properties
    plt.subplots_adjust(**adjustprops)
    plt.gca().set_aspect('equal')

    img_name = 'wake_centerline_anisotropy_barycentric.pdf'
    plt.savefig(img_name)
    print('written ' + img_name)

    plt.close()


