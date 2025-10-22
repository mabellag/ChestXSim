from chestxsim.core  import  volumeData, xp, Geometry
import chestxsim.reconstruction.functional as F 
from typing import Union, Tuple, Optional 
from chestxsim.utility.filters import *
import copy 

import inspect



# class SampleProjections():
#     def __init__(self,
#                  nprojs: int, 
#                  sampling_rate: int):
        
#         self.nprojs = nprojs, 
#         self.sampling_rate = sampling_rate
    
#     def __call__(self, ct_data:volumeData):
#         """sample projections from full span acquisition 
#         for LSA reconstruction simulationc
        
#         """
#         volume = ct_data.volume
#         metadata = copy.deepcopy(ct_data.metadata)

#         processed_volume = F.sample_volume(volume, self.nprojs, self.sampling_rate)

#         metadata.dim = processed_volume.shape
#         metadata.step_outputs[self.__class__.__name__] = {
#             "sampled": self.sampling_rate
#         }
#         return volumeData(volume=processed_volume, metadata=metadata)
    

class BackProjector():
    def __init__(self, opt,
                 reco_dim_mm: Union[Tuple[int, int, int], str] = None,
                 vol_vx: Tuple[float, float, float] = None,
                 reco_dim_px: Tuple[int, int, int] = None,
                 channel_wise: bool = False,
                 geometry: Optional[Geometry] = None,
                 ):
        
        self.opt = opt
        self.reco_dim_mm = reco_dim_mm
        self.reco_vx = vol_vx
        self.reco_dim_px = reco_dim_px
        self.channel_wise = channel_wise
        self.geometry = geometry
        
    
    def __call__(self, ct_data:volumeData) -> volumeData:
        volume = ct_data.volume
        metadata = copy.deepcopy(ct_data.metadata)

        reco_dim = F.resolve_reco_dim(self.reco_dim_mm, self.reco_dim_px, self.reco_vx, self.geometry)
        if not self.channel_wise:
            volume = xp.sum(volume, axis=-1, keepdims=True)

        processed_volume = F.backProject(self.opt,
                                         volume,
                                         reco_dim,
                                         self.reco_vx)
        
        metadata.dim = processed_volume.shape
        metadata.step_outputs[self.__class__.__name__] = {
            "kernel": type(self.opt),
            "geometry": self.opt.geometry.to_dict() if hasattr(self.opt, "geometry") else None,
        }
        return volumeData(volume=processed_volume, metadata=metadata)
    

def instance_to_constructor_dict(instance):
    sig = inspect.signature(instance.__init__)
    params = list(sig.parameters.keys())[1:]  # skip 'self'
    return {
        name: getattr(instance, name, None)
        for name in params
    }

# class ExecutablesReconstruction:
#     def __init__(self, opt,
#                  reco_dim_mm: Union[Tuple[int, int, int], str] = None,
#                  reco_dim_px: Optional[Tuple[int, int, int]] = None,
#                  reco_vx: Optional[Tuple[float, float, float]] = None,
#                  **kwargs):
#         self.opt = opt
#         self.reco_dim_mm = reco_dim_mm
#         self.reco_dim_px = reco_dim_px
#         self.reco_vx = reco_vx
#         self.reco_params = kwargs  

#     def __call__(self, ct_data: volumeData) -> volumeData:
#         volume = ct_data.volume
#         metadata = copy.deepcopy(ct_data.metadata)

#         reco_dim = F.resolve_reco_dim(self.reco_dim_mm, self.reco_dim_px, self.reco_vx, self.opt.geometry)
#         print(reco_dim)
#         print(self.reco_params)

#         result = self.opt.adjoint_operator(volume, reco_dim, self.reco_vx, **self.reco_params)

#         metadata.dim = result.shape
#         metadata.voxel_size = self.reco_vx
#         metadata.step_outputs[self.__class__.__name__] = {
#             "kernel": type(self.opt).__name__,
#             "geometry": getattr(self.opt, "geometry", None),
#             "kernel": type(self.opt),
#             "params": instance_to_constructor_dict(self)
#         }

#         return volumeData(volume=result, metadata=metadata)

# class RaptorFDK(ExecutablesReconstruction):
#     """Step class for Raptor FDK"""
#     pass

# class FuximFDK(ExecutablesReconstruction):
#     """Step class for FuxSim FDK"""
#     pass 


class FDK:
    " built in implementation for FDK - for ASTRA "
    def __init__(self,
                 opt,
                 reco_dim_mm: Union[Tuple[int, int, int], str] = None,
                 reco_dim_px: Optional[Tuple[int, int, int]] = None,
                 reco_vx: Tuple[float, float, float] = (1.25, 5.0, 1.25),
                 match_input: bool = False, 
                 filter_type: str = "ramp",
                 offset_filter: float = 0.005,
                 axis: int = 1,
                 max_freq: float = 0.5,
                 padding: bool = True):
               
        self.opt = opt
        self.reco_dim_mm = reco_dim_mm
        self.reco_dim_px = reco_dim_px
        self.reco_vx = reco_vx
        self.match_input = match_input 
        self.filter_type = filter_type
        self.offset_filter = offset_filter
        self.axis = axis
        self.max_freq = max_freq
        self.padding = padding
     
    def __call__(self, ct_data: volumeData) -> volumeData:
        volume = ct_data.volume
        md = copy.deepcopy(ct_data.metadata)
        if self.match_input:
            # try : raise an error if not available data get original mm from metadata 
            init_shape = md.find("ct_orig_shape")
            init_vx    = md.find("ct_orig_vx")
            up_mm, down_mm = md.find("extension_mm")

            reco_dim_mm = [
            init_shape[0] * init_vx[0],
            init_shape[1] * init_vx[1],
            init_shape[2] * init_vx[2] + up_mm + down_mm
            ]
        
            self.reco_dim_mm = tuple(reco_dim_mm)
                
        print("mm", self.reco_dim_mm)

        reco_dim = F.resolve_reco_dim(self.reco_dim_mm, self.reco_dim_px, self.reco_vx, self.opt.geometry)
        print("reco_dim px",reco_dim)
        
        
        # print("fdk", volume[:3].shape)
        # volume = xp.squeeze(volume, axis=-1)
       
        # print(self.opt.DOD)
        if self.opt.DOD is None:
            try:
                self.opt.DOD = self.opt.set_DOD(md.find('ct_orig_shape'), md.find('ct_orig_vx'))
            except:
                self.opt.DOD = self.opt.bucky + 0.35 # default callback average obese patient ()

        result = F.fdk(
            volume,
            self.opt,
            reco_dim=reco_dim,
            reco_vx_size=self.reco_vx,
            filter_type=self.filter_type,
            offset_filter=self.offset_filter,
            axis=self.axis,
            max_freq=self.max_freq,
            padding=self.padding,
          
        )

        if self.match_input:
            # remove extended slices (extension was saved in mm as [up_mm, down_mm])
            up_mm, down_mm = md.find("extension_mm")
            # convert mm -> slices **in reconstructed grid**
            n_up   = int(up_mm   / self.reco_vx[2])   # or math.ceil(...) but match extender rounding
            n_down = int(down_mm / self.reco_vx[2])
            print(n_up, n_down)

            Z = result.shape[2]
            start = n_up
            stop  = Z - n_down
            # if start < 0: start = 0
            # if stop  < start: stop = start  # avoid empty/negative
            result = result[:, :, start:stop]
        
        md.dim = result.shape
        md.voxel_size = self.reco_vx
        md.step_outputs[self.__class__.__name__] = {
            "kernel": type(self.opt).__name__,
            "geometry": getattr(self.opt, "geometry", None),
            "kernel": type(self.opt),
            "params": instance_to_constructor_dict(self)
        }
        return volumeData(volume=result, metadata=md)

           
class SART():
    pass 

# class SART():
#     def __init__(self, opt,
#                 reco_dim:Union[Tuple[int, int, int], str],   # reco roi in mm 
#                 reco_vx:Tuple[float, float, float]= (1.25, 5.00, 1.25),
#                 lamb :int= 1, 
#                 n_iter:int= 20, 
#                 eps:int= 1e-10,
#                 x_0 = None): 
        
#         self.opt = opt
#         self.reco_dim= reco_dim
#         self.reco_vx = reco_vx
#         self.lamb= lamb
#         self.n_iter= n_iter
#         self.eps = eps
     
#     def __call__(self, ct_data:volumeData) -> volumeData:
#         volume = ct_data.volume
#         metadata = copy.deepcopy(ct_data.metadata)

#         try:
#             # coger el input vol de aqui 
#         except:

#         processed_volume = F.fdk(self.opt,
#                                 volume,
#                                 self.reco_dim, 
#                                 self.reco_vx,
#                                 self.filter_type,
#                                 self.offset_filter,
#                                 self.axis,
#                                 self.max_freq,
#                                 self.padding,
#                                 )
#         metadata.dim = processed_volume.shape
#         metadata.voxel_size = self.reco_vx

#         metadata.step_outputs[self.__class__.__name__] = {
#         }
#         return  volumeData(volume=processed_volume, metadata=metadata)
