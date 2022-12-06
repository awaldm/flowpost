from dataclasses import dataclass
import flowpost.wake.wake_stats as ws
from flowpost.common.wake_config import WakeCaseParams
from flowpost.common.case import Case
#from flowpost.utils import loaders

import tecreader as tecreader
import os
import numpy as np
from flowpost.stats import VelocityStatistics, ReynoldsStresses
import flowpost.wake.wake_transform as wt
import scipy
import logging
logger = logging.getLogger(__name__)

###############################################################################
# Some data classes, never used
class DataField():
    def __init__(self):
        dims = 2
        struct_data = False

class Coordinates:
    def __init__(self, aoa=0, x=None,y=None,z=None):
        self.x = x
        self.y = y
        self.z = z



@dataclass(init=False)
class DataSeries():

    def get_vars(self, var_list):
        print(self.__dict__.keys())
        sys.exit(0)

        for v in var_list:
            value = getattr(self, v)
            print(value)

    def get_var(self, varname):

        if varname in self.__dict__.keys():
            return getattr(self, varname)

    def computeGradients(self):
        dudy, dudx = np.gradient(self.vx,-self.dy/1000,self.dx/1000)
        dvdy, dvdx = np.gradient(self.vy,-self.dy/1000,self.dx/1000)
        self.gradients['dudy']=dudy
        self.gradients['dudx']=dudx
        self.gradients['dvdy']=dvdy
        self.gradients['dvdx']=dvdx
        # TODO and so on

        #skip = 0
        #self.u  = np.array(u).reshape(self.sizey,self.sizex)
        #self.v  = np.array(w).reshape(self.sizey,self.sizex)
        #self.u = self.u[:,skip:]
        #self.v = self.v[:,skip:]


@dataclass(init=False)
class FieldSeries(DataSeries):
    """A class containing velocity time series data


    """

    u: np.ndarray
    v: np.ndarray
    w: np.ndarray

    """
    def __init__(self, time=0, x=None,y=None, z=None,u=None,v=None, w=None, struct_data=False, planar=True):
        #vel = VelocityField(self)
        #DataField.__init__(self)
        #self.coords = {}
        self.vel = {}
        self.set_velocities(u,v,w)

        self.planar = planar
                #self.sizex=cols
        #self.sizey=rows

        #print(str(cols) + ' cols by ' + str(rows) + ' rows')
        self.mean_u, self.mean_v, self.mean_w = compute_means(u,v,w)
        self.gradients = {}
        self.set_velocities(u,v,w)
    """
    def set_velocities(self ,u, v, w):
        self.u = u
        self.v = v
        self.w = w


"""
@dataclass
class ReynoldsStress():
    uu: np.ndarray = None
    vv: np.ndarray = None
    ww: np.ndarray = None
    uv: np.ndarray = None
    uw: np.ndarray = None
    vw: np.ndarray = None
    kt: np.ndarray = None

    def set_unnamed(self, initial_data):
        print(initial_data)
        for key in initial_data:
            setattr(self, key, initial_data[key])

    # https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
    def set_values(self, *initial_data, **kwargs):
        print('setting values')
        print(initial_data)
        for dictionary in initial_data:
            print(initial_data)
            for key in dictionary:
                print('setting ' + str(key))
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


"""



@dataclass
class WakeField():
    vel: FieldSeries = None
    vel_prime: FieldSeries = None
    data: DataSeries = None
    dataset = None  # Tecplot dataset
    # We needs statistics for velocities and other vars
    vel_stats = VelocityStatistics()
    #stats: FieldStats = None
    cs: str = 'AC'
    coords: Coordinates = None
    param: WakeCaseParams = None
    datatype: str = 'tecplot'
    aoa: float = 0
    #def set_coords(self, x, y, z):
    #    self.coords = Coordinates(x=x, y=y, z=z)
    case: Case = None
    def set_coords(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        self.set_PMR()
        # this does not work

    def set_PMR(self, x_PMR=0, z_PMR=0):
        self.x_PMR = x_PMR
        self.z_PMR = z_PMR


    def rotate_CS(self, CSname, aoa = None):
        """Rotate the coordinate system

        :param CSname: _description_
        """
        if aoa == None:
            aoa = self.aoa
        logger.info('rotating by ' + str(aoa))
        # Call the Tecplot dataset rotation function
        print(self.x_PMR)
        print(self.z_PMR)
        wt.rotate_dataset(self.dataset, self.x_PMR, self.z_PMR, aoa)
        x_WT, z_WT = wt.transform_wake_coords(self.x, self.z, self.x_PMR, self.z_PMR, aoa)
        u_WT, w_WT = wt.rotate_velocities(self.vel.u, self.vel.v, self.vel.w, self.x_PMR, self.z_PMR, aoa)
        self.vel.u = u_WT
        self.vel.w = w_WT
        self.cs = CSname
        self.set_coords(x_WT, self.y, z_WT)

    def compute_rstresses(self, do_save = False):
        self.stats.compute_rstresses(do_save = do_save, vel = self.vel)
        """
        #uu,vv,ww,uv,uw,vw = ws.calc_rstresses(u,v,w)
        #self.rstresses = ReynoldsStress
        #self.rstresses.set_values()
        self.stats.rstresses = ws.calc_rstresses(self.vel.u, self.vel.v, self.vel.w, return_dict=True)
        self.stats.rstresses['kt'] = 0.5* (self.rstresses['uu'] + self.rstresses['vv'] + self.rstresses['ww'])

        #print('d: ' + str(d))
        #self.rstresses.set_unnamed(d)
        #print(type(self.rstresses))
        #print(type(self.rstresses['uu']))
        #self.rstresses.uu,vv,ww,uv,uw,vw = ws.calc_rstresses(u_WT,v,w_WT)
        if do_save:
            self.save_rstresses(self.rstresses, res_path = self.param.res_path, file_prefix = self.param.case_name+'_'+ self.param.plane_name)
        """


    def interpolate_struct(self, coords = None, variables = ['u', 'v', 'w'], in_data = None):
        """Interpolate to a structured 2D dataset
    
        logger = fplog.get_main_app_logger()


        """
        if coords is None:
            x = self.x
            z = self.z

        
        if in_data is None:
            print('using default boundaries')
            u = self.vel.u
            v = self.vel.v
            w = self.vel.w
        xlim = (1.05, 1.65)
            
        x0, z0 = xlim[0], -0.05
        x1, z1 = xlim[1], 0.25

        xi, zi = np.linspace(x0, x1, 100), np.linspace(z0, z1, 250)
        xmesh, zmesh = np.meshgrid(xi,zi)

        print('shape of xi: ' + str(xi.shape))
        print('shape of zi: ' + str(zi.shape))
        print('mesh shape: ' + str(xmesh.shape))
        print(x)
        for var in variables:
            #self.struct[var] = scipy.interpolate.griddata((x_WT[case], z_WT[case]), u_WT[case], (xmesh, zmesh), method='cubic')
            #print(self.vel.get_var(var))
            return xi, zi, scipy.interpolate.griddata((x, z), self.vel.get_var(var), (xmesh, zmesh), method='linear')



    def compute_scales(self):
        pass

    def field_PSD(self, data, dt = 1, n_bins = 2, n_overlap = 0.5, window = 'hann'):
        import scipy
        n_points = data.shape[0]
        n_samples = data.shape[1]


        nperseg = np.round(n_samples / (1 + n_overlap*(n_bins - 1))) # this takes into account the overlap

        print('computing Welch PSD for all points...')
        print('temporal samples: ' + str(n_samples))
        print('points per segment: ' + str(nperseg))

        for point in range(n_points):
            f, PSD = scipy.signal.welch(data[point, :], fs = 1 / dt,
                                            window = window,
                                            nperseg = nperseg,
                                            scaling = 'density')
            if point == 0:
                f_mat = np.zeros([n_points, len(f)])
                PSD_mat = np.zeros([n_points, len(f)])
            f_mat[point, :] = f
            PSD_mat[point, :] = PSD

        return f_mat, PSD_mat

    def compute_PSD(self, data, dt = None, n_bins = 2, n_overlap = 0.5, do_save = False):
        if dt is None:
            dt = self.param.dt
        """
        if isinstance(data, list):
            print('is a list')
            print(self.vel.u.shape)
            in_data = []
            for entry in data:
                in_data.append(getattr(self.vel, entry))
        """
        # Compute the PSDs for each variable one by one and save them to disk
        # immediately, as the results have a significant memory footprint
        for var, name in zip([self.vel.u, self.vel.v, self.vel.w], ['u', 'v', 'w']):
            f, PSD = self.field_PSD(var, dt = dt, n_bins=n_bins, n_overlap = n_overlap)
            if do_save:
                file_prefix = self.param.case_name+'_' + self.param.plane_name
                filename = os.path.join(self.param.res_path, file_prefix + '_' + str(name) +'_PSD')
                np.savez(filename, x=self.x, y=self.y, z=self.z, f=f[0,:], PSD=PSD)



    def compute_skew_kurt(self, do_save = False):

        from scipy.stats import kurtosis, skew

        self.skew = {}
        self.kurt = {}
        for var, name in zip([self.vel.u, self.vel.v, self.vel.w], ['u', 'v', 'w']):
            self.skew[name] = skew(var, axis=-1)
            self.kurt[name] = kurtosis(var, axis=-1)


        res_path = self.param.res_path
        file_prefix = self.param.case_name+'_' + self.param.plane_name

        save_var= {'skew_u': self.skew['u'], 'skew_v': self.skew['v'], 'skew_w': self.skew['w']}

        filename = os.path.join(res_path, file_prefix + '_skewness.plt')
        tecreader.save_plt(save_var, self.dataset, filename, addvars = True, removevars = True)


        save_var= {'kurt_u': self.kurt['u'], 'kurt_v': self.kurt['v'], 'kurt_w': self.kurt['w']}

        filename = os.path.join(res_path, file_prefix + '_kurtosis.plt')
        tecreader.save_plt(save_var, self.dataset, filename, addvars = True, removevars = True)


    def write_stats(self, stat_name, file_prefix = 'test_data'):
        """
        Write stats. Select the specific variable using name.

        Parameters
        ----------

        name : str
            the 1D signal


        TODO: this can only work if means have been computed
        TODO: add a loop to write multiple variables
        TODO: understand how the logger works!
        TODO: set proper name for the data variables inside the resulting PLT file
        """
        out_path = self.cfg["case"]["res_path"]
        try:
            write_file(self.dataset, getattr(self.stats, stat_name), out_path, file_prefix)
        except:
            logger.error('No variable ' + str(stat_name) + ' in stats')
            print('no variable!')


    def save_means(self, res_path=None):
        if res_path is None:
            res_path = self.param.res_path
        file_prefix = self.case.case_name+'_' + self.case.plane_name

        filename = os.path.join(res_path, file_prefix + '_means.plt')
        #print(self.stats.mean)
        tecreader.save_plt(self.vel_stats.mean, self.dataset, filename, addvars = True, removevars = True)

    def save_rstresses(self, rstress, res_path = None, file_prefix = None):
        if res_path is None:
            res_path = self.param.res_path
        if file_prefix is None:
            file_prefix = self.param.case_name+'_' + self.param.plane_name
        # Save the results

        try:
            os.makedirs(res_path, mode = 0o777, exist_ok = True)
            print("Directory '%s' created successfully" %res_path)
        except:
            print("Directory '%s' can not be created"%res_path)

        save_var= {'uu': rstress['uu'], 'vv': rstress['vv'], 'ww': rstress['ww'], \
                'uv': rstress['uv'], 'uw': rstress['uw'], 'vw': rstress['vw'], 'kt': rstress['kt']}

        filename = os.path.join(res_path, file_prefix + '_rstresses.plt')
        tecreader.save_plt(save_var, self.dataset, filename, addvars = True, removevars = True)

    def compute_wake_quants(self):
        """
        The content of wake_stats_multiplecases.py goes here
        """
        pass

    def compute_fluctuations(self):
        self.vel.uprime, self.vel.vprime, self.vel.wprime = ws.compute_fluctuations(self.vel.u, self.vel.v, self.vel.w)

    def compute_anisotropy(self, do_save = False):
        self.vel_stats.compute_anisotropy(do_save = do_save, vel = self.vel)

    def compute_means(self):
        self.vel_stats.means(self.vel.u, 'u')
        self.vel_stats.means(self.vel.v, 'v')
        self.vel_stats.means(self.vel.w, 'w')

        '''
        mean_u = np.mean(u, axis=-1)
        mean_v = np.mean(v, axis=-1)
        mean_w = np.mean(w, axis=-1)
        '''
    def save_plt():
        pass


    def data_to_dict(**kwargs):
        out = {}
        for key, value in kwargs.items():
            out[key] = value
        return out

    def transform(self, aoa = None):
        if aoa is None:
            aoa = self.aoa
        logger.info('rotating by ' +str(aoa))
        ws.rotate_dataset(self.dataset, param.x_PMR, param.z_PMR, aoa)
        x_WT, z_WT = ws.transform_wake_coords(vel.x,vel.z, param.x_PMR, param.z_PMR, aoa)
        u_WT, w_WT = ws.rotate_velocities(vel.u, vel.v, vel.w, param.x_PMR, param.z_PMR, aoa)


    def save_anisotropy(self, atensor, ev, C, res_path = None, file_prefix = None):
        if res_path is None:
            res_path = self.param.res_path
        if file_prefix is None:
            file_prefix = self.param.case_name+'_' + self.param.plane_name
        # Save the results

        try:
            os.makedirs(res_path, mode = 0o777, exist_ok = True)
            print("Directory '%s' created successfully" %res_path)
        except:
            print("Directory '%s' can not be created"%res_path)

        save_var= {'a_uu': atensor['uu'], 'a_vv': atensor['vv'], 'a_ww': atensor['ww'], \
                'a_uv': atensor['uv'], 'a_uw': atensor['uw'], 'a_vw': atensor['vw']}

        filename = os.path.join(res_path, file_prefix + '_anisotropy_tensor.plt')
        print(filename)
        tecreader.save_plt(save_var, self.dataset, filename, addvars = True, removevars = True)

        save_var= {'ev1': ev[0,:], 'ev2': ev[1,:], 'ev3': ev[2,:]}

        filename = os.path.join(res_path, file_prefix + '_anisotropy_eigenvalues.plt')
        tecreader.save_plt(save_var, self.dataset, filename, addvars = True, removevars = True)

        save_var= {'C1': C[0,:], 'C2': C[1,:], 'C3': C[2,:]}

        filename = os.path.join(res_path, file_prefix + '_anisotropy_components.plt')
        tecreader.save_plt(save_var, self.dataset, filename, addvars = True, removevars = True)

    def compute_independent_samples(self, acf_maxlag = 10, do_save = False):
        # Compute the autocorrelation function at each point
        uprime, _, wprime = ws.compute_fluctuations(self.vel.u, self.vel.v, self.vel.w)
        self.acf_u = ws.compute_field_acf(uprime, acf_maxlag)
        self.acf_w = ws.compute_field_acf(wprime, acf_maxlag)

        # Obtain the number of independent samples based on the ACF
        ind_u = ws.compute_field_acf_index(self.acf_u)
        ind_w = ws.compute_field_acf_index(self.acf_w)
        self.n_eff_u = self.vel.n_samples/(2*ind_u)
        self.n_eff_w = self.vel.n_samples/(2*ind_w)

        if do_save:
            res_path = self.param.res_path
            file_prefix = self.param.case_name+'_' + self.param.plane_name
            save_var= {'n_eff_u': self.n_eff_u, 'n_eff_w': self.n_eff_w}


            filename = os.path.join(res_path, file_prefix + '_ind_samples.plt')
            tecreader.save_plt(save_var, self.dataset, filename, addvars = True, removevars = True)

    def load_raw(self, raw_type='plt_series'):
        logger.info('loading zone number ' + str(self.case.inputs.zone_list))
        in_data, dataset = loaders.load_plt_series(self.case.inputs.in_path, self.case.inputs.zone_list, self.case.inputs.start_i, self.case.inputs.end_i, read_velocities = True, read_cp=False, read_vel_gradients=False, stride = self.case.inputs.di, parallel=False, verbose=False)

        self.wake = loaders.create_wake_object(in_data, dataset)


class VelocityField(DataField):
    def __init__(self, x=None,z=None,v=None,u=None,w=None):
        DataField.__init__(self)
        self.coords = {}
        self.vel = {}
        self.set_velocities(u,v,w)

    def set_velocities(self,u,v,w):
        self.vel['u'] = u
        self.vel['v'] = v
        self.vel['w'] = w
