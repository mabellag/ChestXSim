from chestxsim.core  import  volumeData, xp, Geometry
import chestxsim.reconstruction.functional as F 
from typing import Union, Tuple, Optional 
from chestxsim.utility.filters import *
import copy 

import inspect

__all__= [ 
    "FDK",
    "SART"
]
 

def instance_to_constructor_dict(instance):
    sig = inspect.signature(instance.__init__)
    params = list(sig.parameters.keys())[1:]  # skip 'self'
    return {
        name: getattr(instance, name, None)
        for name in params
    }


class _BaseReconstructionStep:
    """
    Shared utilities for reconstruction steps (FDK, SART).

    Handles:
        - Matching reconstruction size to input (match_input=True)
        - Resolving reco_dim from mm / px / geometry FOV
        - Geometry fitting (DOD) if needed
        - Optional cropping of extended slices back to original size
    """

    def __init__(
        self,
        opt: Any,
        reco_dim_mm: Optional[Union[Tuple[int, int, int], str]] = None,
        reco_dim_px: Optional[Tuple[int, int, int]] = None,
        reco_vx: Tuple[float, float, float] = (1.25, 5.0, 1.25),
        match_input: bool = False,
    ):
        self.opt = opt
        self.reco_dim_mm = reco_dim_mm
        self.reco_dim_px = reco_dim_px
        self.reco_vx = reco_vx
        self.match_input = match_input

    # ---- helpers ---------------------------------------------------------

    def _get_physical_size_from_input(self, md):
        """
        If match_input=True: infer reco_dim_mm from original CT dims, voxel size
        and stored extension_mm, then store it in self.reco_dim_mm.
        """
        init_shape = md.find("ct_dim")
        init_vx = md.find("ct_vx")
        up_mm, down_mm = md.find("extension_mm")

        reco_dim_mm = [
            init_shape[0] * init_vx[0],
            init_shape[1] * init_vx[1],
            init_shape[2] * init_vx[2] + up_mm + down_mm,
        ]
        self.reco_dim_mm = tuple(reco_dim_mm)

    def _prepare_reco_grid(self, md):
        """
        Resolve reco_dim and make sure geometry is adapted to the volume.
        """
        if self.match_input:
            self._get_physical_size_from_input(md)

        # Resolve reconstruction grid in pixels
        reco_dim = F.resolve_reco_dim(
            self.reco_dim_mm,
            self.reco_dim_px,
            self.reco_vx,
            self.opt.geometry,
        )

        # Geometry fit / fallback for DOD
        if getattr(self.opt.geometry, "DOD", None) is None:
            try:
                self.opt.geometry.fit_to_volume(md.find("ct_dim"), md.find("ct_vx"))
            except Exception:
                # default callback: average obese patient DOD
                self.opt.geometry.DOD = self.opt.geometry.bucky + 0.35

        return reco_dim

    def _crop_to_original_if_needed(self, result, md):
        """
        If match_input=True, remove the extended slices in Z based on extension_mm.
        """
        if not self.match_input:
            return result

        up_mm, down_mm = md.find("extension_mm")

        # Convert mm -> slices in reconstructed grid
        n_up = int(up_mm / self.reco_vx[2])
        n_down = int(down_mm / self.reco_vx[2])

        Z = result.shape[2]
        start = n_up
        stop = Z - n_down
        return result[:, :, start:stop]

    def _update_metadata(self, md, result):
        """
        Common metadata updates for reconstruction steps.
        """
        md.dim = result.shape
        md.voxel_size = self.reco_vx
        md.step_outputs[self.__class__.__name__] = {
            "kernel": type(self.opt).__name__,
            "geometry": getattr(self.opt, "geometry", None),
            "kernel_class": type(self.opt),
            "params": instance_to_constructor_dict(self),
        }
        return md


class FDK(_BaseReconstructionStep):
    """
    ChestXsim built-in FDK reconstruction step 
    """
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
        
        super().__init__(opt, reco_dim_mm, reco_dim_px, reco_vx, match_input)      
        self.filter_type = filter_type
        self.offset_filter = offset_filter
        self.axis = axis
        self.max_freq = max_freq
        self.padding = padding
  
    def __call__(self, ct_data: volumeData) -> volumeData:
        volume = ct_data.volume
        md = copy.deepcopy(ct_data.metadata)
        # resolve reconstruction grid dimensions
        reco_dim = self._prepare_reco_grid(md) 
        result = F.fdk(
            self.opt,
            volume,
            reco_dim=reco_dim,
            reco_vx_size=self.reco_vx,
            filter_type=self.filter_type,
            offset_filter=self.offset_filter,
            axis=self.axis,
            max_freq=self.max_freq,
            padding=self.padding,
          
        )
        result = self._crop_to_original_if_needed(result, md)
        md = self._update_metadata(md, result)
        return volumeData(volume=result, metadata=md)
       
class SART(_BaseReconstructionStep):
    """ 
    ChestXsim built-in SART reconstruction step 
    """
    def __init__(
        self,
        opt: Any,
        reco_dim_mm: Optional[Union[Tuple[int, int, int], str]] = None,
        reco_dim_px: Optional[Tuple[int, int, int]] = None,
        reco_vx: Tuple[float, float, float] = (1.25, 5.0, 1.25),
        match_input: bool = False,
        lamb: float = 1.0,
        n_iter: int = 20,
        eps: float = 1e-10,
        x0: Optional[Any] = None,
    ):
        super().__init__(opt, reco_dim_mm, reco_dim_px, reco_vx, match_input)
        self.lamb = lamb
        self.n_iter = n_iter
        self.eps = eps
        self.x0 = x0

    def __call__(self, ct_data: volumeData) -> volumeData:
        volume = ct_data.volume
        md = copy.deepcopy(ct_data.metadata)
        # resolve reconstruction grid dimensions
        reco_dim = self._prepare_reco_grid(md)
        result = F.sart(
            opt=self.opt,
            projections=volume,
            reco_dim=reco_dim,
            reco_vx_size=self.reco_vx,
            lamb=self.lamb,
            n_iter=self.n_iter,
            eps=self.eps,
            x_0=self.x0,
        )
        result = self._crop_to_original_if_needed(result, md)
        md = self._update_metadata(md, result)

        return volumeData(volume=result, metadata=md)

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
    
   