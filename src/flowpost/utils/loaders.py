import tecreader as tecreader
from flowpost.wake.data import FieldSeries, WakeField
#from flowpost.configs.wake_config import WakeCaseParams
from flowpost.common.case import Case
import numpy as np
import logging
logger = logging.getLogger(__name__)

class Loader:
    def __init__(self):
        self.in_data = None
        self.case = None
        self.dataset = None
    def set_case(self, case):
        self.case = case


    def load_plt_series(self):
        logger.info(self.case.inputs.start_i)
        logger.info(self.case.inputs.end_i)
        logger.info(self.case.inputs.di)

        self.in_data, self.dataset = tecreader.get_series(self.case.inputs.in_path, self.case.inputs.zone_list, self.case.inputs.start_i, self.case.inputs.end_i, \
            read_velocities=True,read_cp=False, read_vel_gradients=False, stride = self.case.inputs.di, \
            parallel=False, verbose=True)


    def create_wake_object(self, in_data = None):
        """Generate wake object from the loader

        :return: _description_
        """

        # If new data is passed, set it as self.in_data
        if in_data is not None:
            self.in_data = {}
            self.in_data['u'] = in_data['u']
            self.in_data['v'] = in_data['v']
            self.in_data['w'] = in_data['w']

        vel  = FieldSeries()
        
        vel.set_velocities(self.in_data['u'], self.in_data['v'], self.in_data['w'])
        
        print('done reading. shape of u: ' + str(self.in_data['u'].shape))

        wake = WakeField()
        wake.vel = vel        

        # Get the coordinates as arrays and add them to the velocity data
        if self.dataset is not None:
            wake.dataset = self.dataset
            x,y,z = tecreader.get_coordinates(self.dataset, caps=True)
            #wake.param = param
            wake.set_coords(x,y,z)
        else:
            wake.set_coords(in_data['x'], np.zeros_like(in_data['x']), in_data['z'])
        return wake

def load_plt_series(plt_path, zonelist, start_i, end_i, read_velocities = True, read_cp=False, read_vel_gradients=False, stride = 10, parallel=False, verbose=False):

    in_data, dataset = tecreader.get_series(plt_path, zone_list, start_i, end_i, \
        read_velocities=True,read_cp=False, read_vel_gradients=False, stride = stride, \
        parallel=False, verbose=True)
    return in_data, dataset




def get_rawdata(case_name, plane_name, case_type):

    #param = WakeCaseParams(case_name, plane_name, case_type)
    print(param)
    # Get parameter dict from config file based on above input
    param.end_i = 5500
    param.plt_path = '/home/andreas/data/CRM_example_data/low_speed/'
    # Get the data time series. The uvw data are arrays of shape (n_points, n_samples). dataset is a Tecplot dataset.
    in_data, dataset = load_plt_series(param.plt_path, param.zonelist, param.start_i, param.end_i, \
        read_velocities=True,read_cp=False, read_vel_gradients=False, stride = param.di, \
        parallel=False, verbose=True)


    wake = create_wake_object(in_data, dataset)
    # Return the FieldSeries object and the Tecplot dataset
    return wake

