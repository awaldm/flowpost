import numpy as np
import scipy
import logging
logger = logging.getLogger(__name__)



def create_interpolation_line(x, zlim, num=100):

    
    x0, z0 = x, zlim[0]
    x1, z1 = x, zlim[1]
    logger.info('extracting line from ('+str(x0)+', '+str(z0)+') to ('+str(x1)+', '+str(z1)+')')

    xi, zi = np.linspace(x0, x1, num), np.linspace(z0, z1, num)
    return xi, zi


def interpolate_line(xfield, zfield, xpos = None, zlim = None, in_data = None, num = 100):
    xi, zi = create_interpolation_line(xpos, zlim, num=num)
    
    return xi, zi, scipy.interpolate.griddata((xfield,zfield), in_data, (xi, zi), method='cubic')



