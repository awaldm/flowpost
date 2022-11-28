#/usr/bin/python
'''


'''

from __future__ import division

import numpy as np
import os, sys
import pandas as pd


def rotate_dataset(dataset, x_PMR, z_PMR, alpha):
    """Rotate Tecplot dataset

    :param dataset: _description_
    :param x_PMR: _description_
    :param z_PMR: _description_
    :param alpha: _description_
    """

    offset = 0
    for zone in dataset.zones():
        array = zone.values('X')
        x = np.array(array[:]).T
        array = zone.values('Y')
        y = np.array(array[:]).T
        array = zone.values('Z')
        z = np.array(array[:]).T
    x, z = transform_wake_coords(x, z, x_PMR, z_PMR, alpha)


    coordvars = ['X', 'Y', 'Z']
    for zone in dataset.zones():
        zone_points = zone.num_points
        print('zone has ' + str(zone_points) + ' points')
        new_x = list(x[offset:offset+zone_points])
        new_y = list(y[offset:offset+zone_points])
        new_z = list(z[offset:offset+zone_points])
        zone.values('X')[:] = new_x
        zone.values('Y')[:] = new_y
        zone.values('Z')[:] = new_z
        offset = offset + zone_points - 1

def transform_wake_coords(x_old, z_old, x_PMR, z_PMR, alpha, axis='y'):

    '''
    transform coordinates of a plane, rotate by alpha
    currently only aoa rotation about y is implemented

    Parameters
    ----------
    x,y,z : float numpy arrays
        current coordinates
    x_PMR, z_PMR : floats
        coordinates of rotation point
    alpha : float
        angle in degrees

    Returns
    -------
    x_WT, z_WT : float numpy arrays
        rotated coordinates
    '''
    if axis is 'y':
        x_WT = (x_old - x_PMR) * np.cos(-1*np.radians(alpha)) - (z_old-z_PMR)* (np.sin(-1*np.radians(alpha))) + x_PMR
        z_WT = (x_old - x_PMR) * np.sin(-1*np.radians(alpha)) + (z_old-z_PMR)* (np.cos(-1*np.radians(alpha))) + z_PMR
        return x_WT, z_WT

def rotate_velocities(u,v,w, x_PMR, z_PMR, alpha):
    u_WT = u*np.cos(-1*np.radians(alpha)) - w * np.sin(-1*np.radians(alpha))
    w_WT = u*np.sin(-1*np.radians(alpha)) + w * np.cos(-1*np.radians(alpha))

    return u_WT, w_WT



def rotate_stresses(uu,vv,ww,uv=None, uw=None, vw=None, x_PMR=None, z_PMR=None, alpha=18.):
    uu_WT =  uu * (np.cos(-1*np.radians(alpha)))**2 - 2 * uw * np.sin(-1*np.radians(alpha)) * np.cos(-1*np.radians(alpha)) + ww * np.sin(-1*np.radians(alpha))**2
    vv_WT =  vv
    ww_WT =  uu * (np.sin(-1*np.radians(alpha)))**2 + 2 * uw * np.sin(-1*np.radians(alpha)) * np.cos(-1*np.radians(alpha)) + ww * np.cos(-1*np.radians(alpha))**2
    uv_WT = 0
    uw_WT = 0
    vw_WT = 0
    if uv is not None and vw is not None:
        uv_WT = uv * (np.cos(-1*np.radians(alpha))) - vw*np.sin(-1*np.radians(alpha))
        vw_WT = uv * np.sin(-1*np.radians(alpha)) + vw*np.cos(-1*np.radians(alpha))
    if uw is not None:
        uw_WT = uu*np.sin(-1*np.radians(alpha))*np.cos(-1*np.radians(alpha)) + uw * np.cos(-1*np.radians(alpha))**2 -np.sin(-1*np.radians(alpha))**2 - ww * np.sin(-1*np.radians(alpha))*np.cos(-1*np.       radians(alpha))
    return uu_WT, vv_WT, ww_WT, uv_WT, uw_WT, vw_WT

