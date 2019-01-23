#/usr/bin/python
"""
Read a time series of structured wake snapshots and create a spacetime plot
"""

import os, sys
import time
import shutil
import numpy as np
import matplotlib.mlab as mlab
import matplotlib as mpl
import matplotlib.pyplot as plt
import helpers.wake_stats as wstat

# plot size
width = float(7)/float(2)
try:
    from setup_line_plot import * # custom plot setup file
    line_style, markers = setup_plot('JOURNAL_1COL', width, 0.9*width)
except ImportError as e:
    print 'setup_plot not found, setting default line styles...'
    line_style=itertools.cycle(["-", "--", ":"])
    marker_style = itertools.cycle(['o', 's', 'v', '^', 'D', '<', '>', 'p', 'h'])



from scipy import signal
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def butter_highpass_filter(data, cutoff, fs, order=5):
    data_flu = data - np.mean(data)
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data_flu)
    return y + np.mean(data)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #y = signal.lfilter(b, a, data)
    y = signal.filtfilt(b, a, data)
    return y




def plot_variable_map(tcoords, xcoords, data, name):
    # create spatial and temporal coordinates
    tt, xx = np.meshgrid(tcoords, xcoords)
    fig, ax = plt.subplots(1, 1)
    #axes.set_aspect('equal')
    cmap = plt.cm.seismic
    maxval = np.max(np.max(data))
    minval = np.min(np.min(data))
    #print minval
    #print maxval
    # setup the color range
    color_delta = 0.001
    vals = np.arange(minval, maxval+color_delta, color_delta)

    norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
    contour_filled = plt.contourf(tt, xx, data.T, vals, norm=norm, cmap=cmap)
    plt.savefig(name + '_velocities.png', dpi=600)
    plt.close()




def compute_correl(data, xcoords, xref, maxlag = 50, dt = 1, verbose=False):
    '''
    Compute the correlation coefficients
    '''
    from scipy.signal import argrelextrema
    # possible temporal lags vs. the reference, in samples
    lag_range = np.arange(-maxlag, maxlag+1)
    # dtrange holds the possible values of t that are possible at that lag
    # this is necessary for correlation computation over all possible time steps separated
    # by the appropriate lag. dtrange simply holds all valid indices ensuring that
    # we do not run out of array bounds
    dtrange = np.arange(maxlag, data.shape[0]-maxlag)
    # rij holds the correlation coefficient between each point x and the reference xref for every lag
    rij = np.zeros([len(lag_range), len(xcoords)])
    maxima = np.zeros(len(xcoords))
    #maxima = np.zeros(len(lag_range))
    maxima_dt = np.zeros(len(xcoords))
    #maxima_dt = np.zeros(len(lag_range))

    print('possible combinations for time averaging: ' +str(len(dtrange)))
    for x in np.arange(len(xcoords)): # for each point x
        if verbose:
            print('correlating x ' + str(xcoords[x]) + 'for ' + str(len(lag_range)) + ' lags')
        for lag in np.arange(len(lag_range)): # for each temporal distance lag
    #        print dtrange
            if lag_range[lag] < 0: # negative lags
                dtrange = np.arange(lag_range[lag], data.shape[0])
            else: # positive lags
                dtrange = np.arange(0, data.shape[0]-lag_range[lag])
            # average correlation rij at that x position over all sample combinations that
            # are separated by this lag
            for t in dtrange:
    #            print 'correlating time ' + str(t) + ' with ' + str(t+lag) + ' for lag ' + str(lag)
                rij[lag,x] = (data[t+lag_range[lag], x] * data[t, xref]) + rij[lag,x]
            rij[lag,x] = rij[lag,x] / len(dtrange)
            rij[lag,x] = rij[lag,x] / np.sqrt((np.mean(data[:, x]*data[:,x], axis=0) * np.mean(data[:, xref]*data[:, xref], axis=0)))
            if lag_range[lag]==0 and x==xref:
                print 'reference rij: ' + str(rij[lag,x])
        # rij[:,x] holds the correlation for the given x at all lags
        #print rij[:,x]
        # maxInd holds the index of the next maximum
        # problem: what happens if it finds multiple maxima?
        maxInd = argrelextrema(rij[:,x], np.greater)
        #print maxInd[0][0]
        #print rij[maxInd,x]
        maxima[x] = maxInd[0][0]
        #print('length of maxima: ' + str(len(maxima)))
        maxima_dt[x] = lag_range[maxInd[0][0]]*dt

        #print str(lag_range[maxInd[0]])
        #print maxInd[0]
        #print maxima.shape
        poselement = np.extract(lag_range[maxInd[0]]>0,maxInd[0])
        #print poselement[0]
        #min_index = np.amin(x)
        #min_value = np.argmin(x)
    print('poselement:')
    print poselement
    print maxima_dt
    return rij, maxima, maxima_dt, lag_range, poselement



#%%
def write_rij_ASCII(filename, trange, xcoords, rij, shape, title):
    '''
    Write correlation coefficient matrix to a ASCII text files in Tecplot format
    inputs:
        filename : dict of mode with their respective variable names, i.e. 'u', 'v' etc
        rij: 2D correlation matrix, (lags, points)
        trange: vector of lags
        xcoords: vector of spatial coordinates
        shape : if None then dataset is assumed to be unstructured
        title : title string for the Tecplot file header
    '''
    # modes is now a dict
    print trange
    x = np.tile(xcoords, rij.shape[0])
    x = np.repeat(xcoords, rij.shape[0])
    trange = np.tile(trange, rij.shape[1])
    #trange = np.repeat(trange, rij.shape[1])
    rij = rij.T
    rij = rij.reshape(rij.shape[0]*rij.shape[1])


    header_str = 'TITLE = ' + title + 'mode\n'
    header_str += 'VARIABLES = "dt" "x" "rij"'
    header_str += '\n'
    header_str += 'ZONE T = "'+title+' results"\n'
    header_str += 'I='+str(shape[0])+', J='+str(shape[1])+', K=1, ZONETYPE=Ordered'
    header_str += ' DATAPACKING=POINT\n'
    header_str += 'DT=( DOUBLE DOUBLE DOUBLE '
    #for _ in mode.iteritems():
    #   header_str += 'DOUBLE DOUBLE '
    header_str +=')'

    #n_vars = len(variables)
    data_matrix = np.c_[trange, x, rij]
    #for key, value in mode.iteritems():
    #    data_matrix = np.c_[x, trange, rij]
    #np.c_[mode[:,0]], \
    #out_path + title+ '_mode.dat', \
    np.savetxt(filename, \
        data_matrix, \
        delimiter = ' ', \
        header = header_str,
        comments='')


# downstream is for low frequency, upstream for high frequency
downstream = False

########### case name
name = 'OAT15A_URANS_JHh_illi_z004'
#name = 'OAT15A_URANS-JHh_turbRoe_z004'
name ='URANS'
name = 'OAT15A_URANS-SSG_dt4_z004'
#name = 'OAT15A_URANS_JHh_illi_hyperflex_z004'
name = 'OAT_URANS-JHh-v2_dt1e5_2016.2_turbRoe_CFL1_2v_SSK_dt1_z004'
dt=1e-5
fs = 1./dt
xref = 74 # x/c=0.85 for 100 points
xref = 180 # 252 # x/c=0.85 for Illi's data, 500 points from 0.15 to 0.24
n_points = 500

in_file = np.load(name+'_lineresults.npz')
if downstream:
    name = name+'_ds'
else:
    name = name + '_us'

print(str(type(in_file)))
lines = in_file['lines']
print(lines.shape)

if downstream:
    xcoords = np.linspace(0.7, 0.9, n_points)


else:
    xcoords = np.linspace(0.15, 0.24, n_points)

tcoords = np.linspace(0,(dt*(lines.shape[0]-1)),lines.shape[0])
tcoords = tcoords - tcoords[100]

plot_variable_map(tcoords, xcoords, lines, name)

fig,ax=plt.subplots(1,1)
plt.plot(lines[:,xref], label='xref')
plt.plot(lines[:,xref+20], label='ref+20')
plt.plot(lines[:,xref-20], label='ref-20')
#plt.plot(lines[:,xref], label='xref')
plt.legend()
plt.savefig(name + '_unfiltered_signals.png', dpi=600)
plt.close()


#plot_velocity(tcoords, xcoords, lines, 'z005')
#print tcoords
#fig, ax = plt.subplots(1,1)
#plt.plot(lines[:,x])
#plt.show()



for x in np.arange(lines.shape[1]):
    lines[:,x] = lines[:,x] - np.mean(lines[:,x], axis=0)
    if downstream:
        lines[:,x] = butter_bandpass_filter(lines[:,x], 50,120, fs)
        maxlag = 40
    else:
        lines[:,x] = butter_bandpass_filter(lines[:,x], 2000,2600, fs, order=3)
        #lines[:,x] = butter_highpass_filter(lines[:,x], cutoff_freq, fs)
        maxlag = 200


fig,ax=plt.subplots(1,1)
plt.plot(lines[:,xref], label='xref')
plt.plot(lines[:,xref+200], label='ref+200')
#plt.plot(lines[:,xref], label='xref')
plt.legend()
plt.savefig(name+'_filtered_signals.png', dpi=600)

# correlate
rij, maxima, maxima_dt, lag_range, pos = compute_correl(lines, xcoords, xref, maxlag = maxlag, dt=dt)

print('shape of rij: ' + str(rij.shape))
write_rij_ASCII('newfile.dat', lag_range*dt, xcoords, rij, rij.shape, 'title')

'''
#print rij
fig, ax = plt.subplots(1, 1)
#axes.set_aspect('equal')
cmap = plt.cm.seismic
maxval = np.max(np.max(rij))
minval = np.min(np.min(rij))
print minval
print maxval
# setup the color range
color_delta = 0.001
vals = np.arange(minval, maxval+color_delta, color_delta)
#print vals
tt, xx = np.meshgrid(lag_range*dt, xcoords)

norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
contour_filled = plt.contourf(tt, xx, rij.T, vals, norm=norm, cmap=cmap)
ax.scatter(maxima_dt, xcoords)
#plt.xlim(-0.001, 0.001)
plt.savefig(name+'_correl.png', dpi=600)
#plt.show()
plt.close()
'''


pos = pos[0]

fig, ax = plt.subplots(1, 1)
#axes.set_aspect('equal')
cmap = plt.cm.seismic
maxval = np.max(np.max(rij))
minval = np.min(np.min(rij))
print minval
print maxval
# setup the color range
color_delta = 0.001
vals = np.arange(minval, maxval+color_delta, color_delta)
#print vals

tt, xx = np.meshgrid(lag_range*dt, xcoords)

norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
contour_filled = plt.contourf(tt, xx, rij.T, vals, norm=norm, cmap=cmap)
plt.ylim([0.161, 0.207])
plt.xlim(-0.001, 0.001)
plt.savefig(name+'_correl.png', dpi=600)
#plt.show()
plt.close()


# propagation velocity
if downstream:
    corr_dt = maxima_dt[xref] - maxima_dt[48]
    corr_ds = (xref-48)*(0.9-0.7)*0.23/100
    u = corr_ds/corr_dt
    print('downstream propagation velocity u: ' + str(u))
else:
    corr_dt = maxima_dt[xref] - maxima_dt[xref +200]
    print('corr_dt: ' + str(corr_dt))
    corr_ds = (xref-(xref+200))*(xcoords[-1]-xcoords[0])/n_points
    print('corr_ds: ' + str(corr_ds))
    u = corr_ds/corr_dt
    print('upstream propagation velocity u: ' + str(u))



#corr_dt = maxima_dt[2] - maxima_dt[0]
#corr_ds = 0.1*0.23



#tref = 70


#l,rij = compute_correl(lines, xref)
#print rij
#sys.exit(0)
#rij = np.zeros([lines.shape[0], lines.shape[1]])
#xcoords = [73,74,75]


np.savetxt('maxima.dat', maxima_dt)

#estimate downstream velocity
# choose two x locations and obtain the time delay of the correlation maximum
print maxima_dt[xref]
print maxima_dt[xref+200]





'''
fig,ax=plt.subplots(1,1)
plt.plot(rij[:,xref])
plt.plot(rij[:,xref-10])
plt.plot(rij[:,xref+10])
plt.plot(rij[:,xref+40])

plt.savefig('correl_plot.png', dpi=600)

'''
#ddsys.exit(0)
#
#
# in_file = np.load('OAT15A_URANS-SSG_lineresults.npz')
# print(str(type(in_file)))
# lines = in_file['lines']
# print(lines.shape)
# dt=2e-5 * 4
# fs = 1/dt
# xcoords = np.linspace(0.7, 0.9, 100)
# tcoords = np.linspace(0,(dt*(lines.shape[0]-1)),lines.shape[0])
# tcoords = tcoords - tcoords[150]
#
# plot_velocity(tcoords, xcoords, lines, 'dt4')
# #print tcoords
# for x in np.arange(lines.shape[1]):
#     lines[:,x] = lines[:,x] - np.mean(lines[:,x], axis=0)
#     lines[:,x] = butter_bandpass_filter(lines[:,x], 1800,2200, fs)
#     #lines[:,x] = butter_highpass_filter(lines[:,x], cutoff_freq, fs)
# maxlag = 10
#
# rij, maxima, maxima_dt, lag_range = compute_correl(lines, xcoords, xref, maxlag = maxlag, dt=dt)
#
# fig, ax = plt.subplots(1, 1)
# #axes.set_aspect('equal')
# cmap = plt.cm.seismic
# maxval = np.max(np.max(rij))
# minval = np.min(np.min(rij))
# print minval
# print maxval
# # setup the color range
# color_delta = 0.001
# vals = np.arange(minval, maxval+color_delta, color_delta)
# #print vals
#
# tt, xx = np.meshgrid(lag_range*dt, xcoords)
#
# norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
# contour_filled = plt.contourf(tt, xx, rij.T, vals, norm=norm, cmap=cmap)
# #plt.xlim(-0.001, 0.001)
# plt.savefig('correl_dt4.png', dpi=600)
# #plt.show()
# plt.close()
# fig,ax=plt.subplots(1,1)
# plt.plot(rij[:,xref])
# plt.plot(rij[:,xref-10])
# plt.plot(rij[:,xref+10])
# plt.plot(rij[:,xref+20])
#
# plt.savefig('correl_plot.png', dpi=600)
#
# #estimate upstream velocity
# # choose two x locations and obtain the time delay of the correlation maximum
# print 'maxima_dt:'
# print maxima_dt
# print maxima_dt[85]
# print maxima_dt[48]
# #>>> t=0.0004-(-0.0008)
# #>>> s=(85-48)*(0.9-0.7)*0.23/100
# fig,ax=plt.subplots(1,1)
# plt.plot(rij[:,85], label='85')
# plt.plot(rij[:,48], label='48')
# plt.plot(rij[:,xref], label='xref')
# plt.legend()
# plt.savefig('correlsdt4.png', dpi=600)
#
# fig,ax=plt.subplots(1,1)
# plt.plot(lines[:,85], label='85')
# plt.plot(lines[:,48], label='48')
# #plt.plot(lines[:,xref], label='xref')
# plt.legend()
# plt.savefig('filtered_signals_48_85_dt4.png', dpi=600)
# corr_dt = maxima_dt[85] - maxima_dt[48]
# corr_ds = (85-48)*(0.9-0.7)*0.23/100
#
# #corr_dt = maxima_dt[2] - maxima_dt[0]
# #corr_ds = 0.1*0.23
#
# u_u = corr_ds/corr_dt
# print('upstream propagation velocity u_u: ' + str(u_u))

#dt = 8e-5

#xcoords = np.linspace(0.7, 0.9, 300)




#estimate upstream velocity
# choose two x locations and obtain the time delay of the correlation maximum
print 'maxima_dt:'
#print maxima_dt
print maxima_dt[xref]
print maxima_dt[pos]
#>>> t=0.0004-(-0.0008)
#>>> s=(85-48)*(0.9-0.7)*0.23/100
fig,ax=plt.subplots(1,1)
plt.plot(rij[:,xref-50], label='ref-50')
plt.plot(rij[:,pos], label='pos')
plt.plot(rij[:,xref], label='xref')
plt.legend()
plt.savefig(name+'_correls.png', dpi=600)


# ########### points
# in_file = np.load('pointresults.npz')
# print(str(type(in_file)))
# lines = in_file['lines']
# print(lines.shape)
# dt = 2e-5
# fs = 1/dt
# xref = 3
# xcoords = np.linspace(0.7, 0.9, 5)
# tcoords = np.linspace(0,(dt*(lines.shape[0]-1)),lines.shape[0])
# tcoords = tcoords - tcoords[2000]
#
# fig,ax=plt.subplots(1,1)
# plt.plot(lines[:,xref])
# plt.plot(lines[:,xref-1])
# plt.plot(lines[:,xref+1])
#
# plt.savefig('unfiltered_signals_points.png', dpi=600)
#
#
# for x in np.arange(lines.shape[1]):
#     lines[:,x] = lines[:,x] - np.mean(lines[:,x], axis=0)
#     lines[:,x] = butter_bandpass_filter(lines[:,x], 1800,2200, fs, order=5)
#     #lines[:,x] = butter_highpass_filter(lines[:,x], cutoff_freq, fs)
#
# fig,ax=plt.subplots(1,1)
# plt.plot(lines[:,xref])
# plt.plot(lines[:,xref-1])
# plt.plot(lines[:,xref+1])
#
# plt.savefig('filtered_signals_points.png', dpi=600)
# plt.savefig('filtered_signals_points.pdf')
#
# maxlag = 50
# rij, maxima, maxima_dt = compute_correl(lines, xcoords, xref, maxlag = maxlag)
#
# fig, ax = plt.subplots(1, 1)
# #axes.set_aspect('equal')
# cmap = plt.cm.seismic
# maxval = np.max(np.max(rij))
# minval = np.min(np.min(rij))
# print minval
# print maxval
# # setup the color range
# color_delta = 0.001
# vals = np.arange(minval, maxval+color_delta, color_delta)
# #print vals
# tt, xx = np.meshgrid(lag_range*dt, xcoords)
#
# norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
# contour_filled = plt.contourf(tt, xx, rij.T, vals, norm=norm, cmap=cmap)
# #plt.xlim(-0.001, 0.001)
# plt.savefig('correl_points.png', dpi=600)
# #plt.show()
# plt.close()
#
# #estimate upstream velocity
# # choose two x locations and obtain the time delay of the correlation maximum
# print maxima_dt[1]
# print maxima_dt[2]
# #>>> t=0.0004-(-0.0008)
# #>>> s=(85-48)*(0.9-0.7)*0.23/100
#
# corr_dt = maxima_dt[2] - maxima_dt[0]
# corr_ds = 0.1*0.23
#
# u_u = corr_ds/corr_dt
# print('downstream propagation velocity u_u: ' + str(u_u))
#
# corr_dt = maxima_dt[1] - maxima_dt[0]
# corr_ds = 0.05*0.23
#
# u_u = corr_ds/corr_dt
# print('downstream propagation velocity u_u: ' + str(u_u))
#
# fig,ax=plt.subplots(1,1)
# plt.plot(rij[:,xref])
# plt.plot(rij[:,xref-1])
# plt.plot(rij[:,xref+1])
#
# plt.savefig('correl_plot_points.png', dpi=600)
