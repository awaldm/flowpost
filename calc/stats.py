import numpy as np
from dataclasses import dataclass

class Statistics():
    def __init__(self):
        # initialize defaults
        self.mean = {}
        self.variance = {}
        self.skew = {}
        self.kurt = {}
   
    def means(self, x, name_x):
        x_mean = np.mean(x, axis=-1,keepdims=True)
        self.mean[name_x] = x_mean   
 
    def variances(self, x, name_x):
        x_variance = np.var(x, axis=-1,keepdims=True)
        self.variance[name_x] = x_variance
 
    def skews(self, x, name_x):
        from scipy.stats import skew
        x_skew = skew(x, axis=-1)
        self.skew[name_x] = x_skew

    def kurts(self, x, name_x):
        from scipy.stats import kurtosis
        x_kurt = kurtosis(x, axis=-1)
        self.kurt[name_x] = x_kurt
    

class VelocityStatistics(Statistics):
    '''
    
    '''
    def __init__(self):
        # initialize defaults
        Statistics.__init__(self)
        self.rs: ReynoldsStresses = ReynoldsStresses()
        self.an: AnisotropyTensor = AnisotropyTensor()

    def compute_anisotropy(self, vel = None, do_save = False):
        #self.stats.atensor = AnisotropyData
        if self.rs.is_empty():
            self.rs = ReynoldsStresses()
            self.rs.calc_rstresses(vel.u,vel.v,vel.w)
            self.rs.kt = 0.5* (self.rs.uu + self.rs.vv + self.rs.ww)
        # Compute the anisotropy tensor
        print(self.rs.kt)
        a_uu, a_vv, a_ww, a_uv, a_uw, a_vw = self.an.compute_atensor(self.rs.uu, \
        self.rs.vv, \
        self.rs.ww, \
        self.rs.uv, \
        self.rs.uw, \
        self.rs.vw, \
        self.rs.kt)
        # Compute second and third invariants of the anisotropy tensor
        atensor = {'uu': a_uu, 'vv': a_vv, 'ww': a_ww, 'uv': a_uv, 'uw': a_uw, 'vw': a_vw}

        invar2, invar3, ev = self.an.compute_anisotropy_invariants(a_uu, a_vv, a_ww, a_uv, a_uw, a_vw)
        # Compute barycentric coordinates
        C, xb, yb = self.an.compute_anisotropy_barycentric(ev)

        if do_save:
            self.save_anisotropy(self.stats.atensor, ev, C, res_path = self.param.res_path, file_prefix = self.param.case_name+'_'+ self.param.plane_name)
        
@dataclass
class AnisotropyTensor():
    uu: np.ndarray = None
    vv: np.ndarray = None
    ww: np.ndarray = None
    uv: np.ndarray = None
    uw: np.ndarray = None
    vw: np.ndarray = None
    def set_values(a_uu, a_vv, a_ww, a_uv, a_uw, a_vw):
        self.uu = a_uu
        self.vv = a_vv
        self.ww = a_ww
        self.uv = a_uv
        self.uw = a_uw
        self.vw = a_vw



    def compute_atensor(self,uu, vv, ww, uv, uw, vw, kt, return_tensor = False):
        '''
        anisotropy tensor from reynolds stress tensor components
        '''
        a_uu = np.divide(uu, (2.*kt)) - (1/3)
        a_vv = np.divide(vv, (2.*kt)) - (1/3)
        a_ww = np.divide(ww, (2.*kt)) - (1/3)
        a_uv = np.divide(uv, (2.*kt))
        a_vw = np.divide(vw, (2.*kt))
        a_uw = np.divide(uw, (2.*kt))
        if return_tensor:
            return self.atensor_components_to_array(a_uu, a_vv, a_ww, a_uv, a_uw, a_vw)
        else:
            return a_uu, a_vv, a_ww, a_uv, a_uw, a_vw

    def atensor_components_to_array(self,a_uu, a_vv, a_ww, a_uv, a_uw, a_vw):
        '''
        build a matrix from the anisotropy tensor elements
        the spatial points a in the first axis (dim 0)

        squeeze is necessary to drop the last dimension of length 1
        transposition of points to dim 0 is necessary to parallelize eigenvalue computation
        rationale: np.linalg.eig computes eigenvalues
        '''
        atensor = np.array([[a_uu, a_uv, a_uw],[a_uv, a_vv, a_vw], [a_uw, a_vw, a_ww]])
        atensor = np.squeeze(atensor)
        return np.transpose(atensor, [2,0,1])

    def compute_anisotropy_invariants(self,a_uu, a_vv, a_ww, a_uv, a_uw, a_vw):
        print('shape of anisotropy component: ' + str(a_uu.shape))
        num_points = a_uu.shape[0]
        #ev = np.zeros([3,num_points])

        atensor = self.atensor_components_to_array(a_uu, a_vv, a_ww, a_uv, a_uw, a_vw)
        print('shape of atensor: ' + str(atensor.shape))

        ev, _ = np.linalg.eig(atensor)
        ev = np.transpose(ev, [1,0])
        for i in range(ev.shape[1]):
            #ev[::-1].sort()
            ev[:,i].sort()
            ev[:,i] = np.flip(ev[:,i])
            #if i == 0:
                #print(ev[:,i])

        print('shape of ev: ' + str(ev.shape))

        invar2 = ev[0,:]**2 + ev[1,:]**2 + np.multiply(ev[0,:],ev[1,:])
        invar3 = -1 * np.multiply(np.multiply(ev[0,:] , ev[1,:]), (ev[0,:] + ev[1,:])) # = det(b)

        invar2 = 2*invar2
        invar3 = 3*invar3
        return invar2, invar3, ev

    def compute_anisotropy_barycentric(self,ev):
        num_points = ev.shape[1]
        C = np.zeros([3, num_points])
        xb = np.zeros([1,num_points])
        yb = np.zeros([1,num_points])
        C[0,:] = ev[0,:] - ev[1,:]
        C[1,:] = 2*(ev[1,:] - ev[2,:])
        C[2,:] = 3*(ev[2,:]) + 1

        xb = C[0,:] + 0.5*C[2,:]
        yb = (np.sqrt(3.0)/2.0) * C[2,:]
        return C, xb, yb






class ReynoldsStresses():
    '''
    Maybe call this VectorStatistics in order to be less specific
    
    TODO: should this inherit from Statistics or be standalone?
    '''

    def __init__(self):
        # initialize defaults
        #VelocityStatistics.__init__(self)
        self.uu: np.ndarray = None
        self.uv: np.ndarray = None
        self.uw: np.ndarray = None
        self.vw: np.ndarray = None
        self.uv: np.ndarray = None
        self.vw: np.ndarray = None

    def is_empty(self):
        return self.uu == None

    def get_rstresses(self, u, v, w):
        '''
        handle the case where rstresses are already computed and no recomputation is necessary
        '''
        
        self.calc_rstresses(u, v, w)
    
    def calc_rstresses(self, u,v,w, return_dict=False):
        # Some shaping code, just in case the arrays had not been flattened before
        reshaped = False
        
        shape = u.shape
        newshape = shape[0:-1]
        
        if len(shape) > 2:
            u.reshape(shape[0]*shape[1],shape[2])
            v.reshape(shape[0]*shape[1],shape[2])
            w.reshape(shape[0]*shape[1],shape[2])
            reshaped = True

        # Normal components are just variances
        self.uu = np.var(u, axis=-1, keepdims = True)
        self.ww = np.var(w, axis=-1, keepdims = True)
        self.vv = np.var(v, axis=-1, keepdims = True)

        # Off-normal components are crosscorrelations, maybe there is a numpy
        # function that does this faster.
        self.uw = np.zeros(self.uu.shape)
        self.uv = np.zeros(self.uu.shape)
        self.vw = np.zeros(self.uu.shape)
        
        for i in range(self.uu.shape[0]):
            uflu = u[i,:] - np.mean(u[i,:], keepdims = True)
            vflu = v[i,:] - np.mean(v[i,:], keepdims = True)
            wflu = w[i,:] - np.mean(w[i,:], keepdims = True)
            self.uw[i] = np.mean(np.multiply(uflu, wflu), keepdims = True)
            self.vw[i] = np.mean(np.multiply(vflu, wflu), keepdims = True)
            self.uv[i] = np.mean(np.multiply(uflu, vflu), keepdims = True)

        if reshaped:
            self.uu.reshape(newshape)
            self.vv.reshape(newshape)
            self.ww.reshape(newshape)
            self.uv.reshape(newshape)
            self.uw.reshape(newshape)
            self.vw.reshape(newshape)
