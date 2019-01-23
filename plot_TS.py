import numpy as np
from scipy import spatial
import helpers.wake_stats as ws
import matplotlib.pyplot as plt
aoa = 19.
x_PMR = 0.165/4
z_PMR = 0
dataset = np.load('NACA0012_AoA19_DDES_SAO_dt1e5_k1024_turbAoF_TS.npz')
x, z = ws.transform_wake_coords(dataset['x'], dataset['z'], x_PMR, z_PMR, aoa)
u, w = ws.rotate_velocities(dataset['u'], None, dataset['w'], x_PMR, z_PMR, aoa)

x = dataset['x']
z = dataset['z']

pt = [0.5, 0.1] # to find

print x.shape
print u.shape
A = np.c_[x, z]
print A.shape

#A[spatial.KDTree(A).query(pt)[1]] # <-- the nearest point 
distance,index = spatial.KDTree(A).query(pt)
print distance
print index
up = u[index] - np.mean(u[index])
wp = w[index] - np.mean(w[index])

#plt.scatter(up,wp)
fig, axes = plt.subplots(1,1)
plt.plot(wp)
plt.savefig('hist1_1.png', dpi=600)
plt.close()
fig, axes = plt.subplots(1,1)
n, bins, patches = axes.hist(wp, 50, normed=1, facecolor='green', alpha=0.75)

plt.savefig('hist1.png', dpi=600)
plt.close()

pt = [0.5, -0.1] # to find
distance,index = spatial.KDTree(A).query(pt)
wp = w[index] - np.mean(w[index])
fig, axes = plt.subplots(1,1)
plt.plot(wp)
plt.savefig('hist2_1.png', dpi=600)
plt.close()
fig, axes = plt.subplots(1,1)
n, bins, patches = axes.hist(wp, 50, normed=1, facecolor='green', alpha=0.75)
plt.savefig('hist2.png', dpi=600)
plt.close()



