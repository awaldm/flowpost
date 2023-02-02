import tecreader as tecreader
from flowpost.wake.data import FieldData, FieldSeries, WakeField
import numpy as np
import logging
logger = logging.getLogger(__name__)

"""
TODO: onsider having Case be the central import after all,
with a read() method that calls appropriate loaders. If the users so desires, they can handle the loader directly.
"""


class Loader:
    def __init__(self):
        self.in_data = None
        self.case = None
        self.tec_dataset = None

        self.parallel = False
        self.verbose = False

        # Defaults what to read, for regular wake data
        self.read_velocities: bool = True
        self.read_cp: bool = False
        self.read_vel_gradients: bool = False
        
    def set_case(self, case):
        self.case = case

    def load_plt_series(self):
        """Load a series of Tecplot data files
        """
        logger.info(self.case.inputs.start_i)
        logger.info(self.case.inputs.end_i)
        logger.info(self.case.inputs.di)

        # Call the series reader involving pytecplot
        self.in_data, self.tec_dataset = tecreader.get_series(
            self.case.inputs.in_path,
            self.case.inputs.zone_list,
            self.case.inputs.start_i,
            self.case.inputs.end_i,
            read_velocities=self.read_velocities,
            read_cp=self.read_cp,
            read_vel_gradients=self.read_vel_gradients,
            stride=self.case.inputs.di,
            parallel=self.parallel,
            verbose=self.verbose)

    def create_dataset(self, in_data: dict = None) -> FieldData:

        # If new data is passed, set it as self.in_data
        if in_data is not None:
            self.in_data = {}
            
            for key in in_data.keys:
                self.in_data[key] = in_data[key]                    


        dataset = FieldData()
        
        
        if self.tec_dataset is not None:
            dataset.tec_dataset = self.tec_dataset
            x, y, z = tecreader.get_coordinates(self.tec_dataset, caps=True)
            dataset.set_coords(x,y,z)
        else:
            dataset.set_coords(in_data['x'], np.in_data['y'], in_data['z'])

        dataset.set_data(self.in_data)
        #dataset.set_data(in_data)

        return dataset

    def create_wake_object(self, in_data: dict = None) -> WakeField:
        """Generate wake object from the loader

        :return: _description_
        """

        # If new data is passed, set it as self.in_data
        if in_data is not None:
            self.in_data = {}
            for key in in_data.keys:
                self.in_data[key] = in_data[key]
                
        # Create an object
        vel = FieldSeries()

        # Set the newly variables as object attributes
        if self.read_velocities:
            vel.set_velocities(self.in_data['u'], self.in_data['v'], self.in_data['w'])
        
        logger.info('Done reading raw data time series. Shape of first variable: ' + str(self.in_data[list(self.in_data.keys())[0]].shape))
        # Create the WakeField object
        wake = WakeField()
        wake.vel = vel        

        # Get the coordinates as arrays and add them to the velocity data
        if self.tec_dataset is not None:
            wake.tec_dataset = self.tec_dataset
            x, y, z = tecreader.get_coordinates(self.tec_dataset, caps=True)
            wake.set_coords(x,y,z)
        else:
            wake.set_coords(in_data['x'], np.zeros_like(in_data['x']), in_data['z'])
        return wake


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

