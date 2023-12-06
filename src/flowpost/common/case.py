import json

# from flowpost.utils import loaders
import logging

logger = logging.getLogger(__name__)


class Input:
    def __init__(self):
        self.in_path = None
        self.start_i = None
        self.end_i = None
        self.zone_list = None
        self.di = None


class GeometryParams:
    def __init__(self, x_PMR=None, y_PMR=None, z_PMR=None, MAC=None):
        self.x_PMR = x_PMR
        self.y_PMR = y_PMR
        self.z_PMR = z_PMR
        self.MAC = MAC

    def from_json(self, filename):
        with open(filename) as json_file:
            data = json.load(json_file)
            self.MAC = data["MAC"]
            self.x_PMR = data["x_PMR"]
            self.z_PMR = data["z_PMR"]


class Case:
    inputs = None

    def __init__(self, case_name, plane, type="CRM_LSS", plt_path=None, res_path=None):
        # self.param = param
        self.case_name = case_name
        self.plane_name = plane
        self.type = "wall"  # surface or wall data?
        self.inputs = Input()

        # Path of data time series (isurfaces and/or wake planes)

        # Result output path
        self.res_path = res_path

    def from_json(self, filename):
        """
        Create a case object from a JSON filename

        Presently this does not alter the case and plane names
        """
        with open(filename) as json_file:
            data = json.load(json_file)
            # print("Type: " , str(type(data)))
            self.inputs.in_path = data["in_path"]
            self.inputs.start_i = data["start_i"]
            self.inputs.end_i = data["end_i"]
            self.inputs.di = data["di"]
            self.inputs.zone_list = data["zone_list"]
            self.res_path = data["res_path"]
            self.aoa = data["aoa"]

    def from_dict(self, data):
        self.inputs.in_path = data["in_path"]
        self.inputs.start_i = data["start_i"]
        self.inputs.end_i = data["end_i"]
        self.inputs.di = data["di"]
        self.inputs.zone_list = data["zone_list"]
        self.res_path = data["res_path"]
        self.aoa = data["aoa"]

    """
    def load_raw(self, raw_type='plt_series'):
        logger.info('loading zone number ' + str(self.inputs.zone_list))
        in_data, dataset = loaders.load_plt_series(self.inputs.in_path, self.inputs.zone_list, self.inputs.start_i, self.inputs.end_i, read_velocities = True, read_cp=False, read_vel_gradients=False, stride = self.inputs.di, parallel=False, verbose=False)

        self.wake = loaders.create_wake_object(in_data, dataset)
    """
