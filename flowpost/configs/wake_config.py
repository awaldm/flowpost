#/usr/bin/python
import os
'''
This is the setup file used by most of the analysis scripts. This is a bunch of ifs
serving as case selector. Each case needs variables such as:
dt: physical time step in seconds
plane: plane name input
plt_path: the folder containing the time series of TAU results (in my case, all the surface/wake plane data)
start_i: first i to load from the folder
end_i: last i
aoa: angle of attack in degrees
res_path: output path for the results
x_PMR and z_PMR: point of model rotation, the fulcrum of the rotation for 
    coordinate system changes
zonelist: zone number list for the raw data reader. 

TODO: JSON or something similar would be a better choice for this file.

Andreas Waldmann, 2018
'''


class WakeCaseParams():
    def __init__(self, case_name, plane, type='CRM_LSS'):

        self.case_name = case_name
        self.plane_name = plane
        # Types of result dataset
        if type == 'CRM_LSS':
            self.x_PMR = 0.90249
            self.z_PMR = 0.14926
            self.MAC = 0.189144
            self.u_inf = 54.65
            
        elif type == 'NACA0012':
            self.x_PMR = 0.165/2
            self.z_PMR = 0
        elif type == 'OAT15A':
            self.x_PMR = 0.23/4
            self.z_PMR = 0

        # Set up the simulation data properties. Add cases as necessary
        if case_name == 'CRM_v38h_DDES_dt100_ldDLR_CFL2_eigval015_pswitch1_tau2017_2':
            self.plane = plane

            # Angle of attack
            self.aoa = 18.0
            
            # Time step size between two successive files
            self.dt = 3.461e-4

            # Stride
            self.di = 10

            # If applicable, pre-saved Tecplot formatted dataset
            self.datasetfile = 'v38h_etaplanes.plt'

            # Path of data time series (isurfaces and/or wake planes)
            self.plt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'example_data/')

            # Result output path
            self.res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'example_results/res/wake/'+plane+'/')

            # Desired start and end of time series
            self.start_i = 5000
            self.end_i = 6000

            self.start_t = 3.59944000e-01
            self.end_t = 3.94554000e-01
            # Zone list to be passed to reader function
            self.zonelist = [11]


        else:
            print('unknown case')

