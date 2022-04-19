# -*- coding: utf-8 -*-
import numpy as np

#import dmd_plotters as dmd_plot
import matplotlib.pyplot as plt
import os


def unstack_mode(mode, varnames):
    '''
    if there are multiple variables in the POD/DMD solution we need to unstack them
    before writing to ASCII
    '''
    n_points = int(mode.shape[0] / len(varnames))
    unstacked = dict()
    print('n_points: ' + str(n_points))
    print('unstacking mode with shape: ' + str(mode.shape) + ' and ' + str(len(varnames)) + ' variables')
    i = 0
    for var in varnames:
        unstacked[var] = mode[i*n_points:(i+1)*n_points] # theres probably a numpy function for this
        i = i + 1
    return unstacked

