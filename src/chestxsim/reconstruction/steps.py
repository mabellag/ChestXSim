from chestxsim.core.data_containers  import  volumeData
from chestxsim.core.device import xp
from chestxsim.core.geometries import Geometry
import chestxsim.reconstruction.functional as F 
from typing import Union, Tuple, Optional, Dict 
from chestxsim.utility.filters import *
import copy 
import math 

import inspect

__all__= [ 
    "FDK",
    "SART",
    "FuxsimFDK",
    "RaptorFDK"
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
            round(init_shape[0] * init_vx[0]),
            round(init_shape[1] * init_vx[1]),
            round(init_shape[2] * init_vx[2] + up_mm + down_mm),
        ]
        self.reco_dim_mm = tuple(reco_dim_mm)
        # print( self.reco_dim_mm)

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
        n_up = math.ceil(up_mm / self.reco_vx[2])
        n_down = math.ceil(down_mm / self.reco_vx[2])
        return result[:, :, n_up:-n_down]

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
        if volume.ndim == 4 and volume.shape[-1] == 1:
            volume = xp.squeeze(volume, axis=-1)
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
        if volume.ndim == 4 and volume.shape[-1] == 1:
            volume = xp.squeeze(volume, axis=-1)
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

# -------------------------------------------------------------------------
#   FuxsimFDK – FuXSim_OP.reconstruct("FDK", ...)
class FuxsimFDK(_BaseReconstructionStep):
    """
    FDK step that uses the FuXSim executable backend via FuXSim_OP.reconstruct.

    Expected operator: FuXSim_OP (Tomo or CBCT), i.e.:

        params = { "filter": [ft_flag, offset_filter, ft_mode] }

        reco = fuxsim.reconstruct(
            method="FDK",
            sino=projections,
            reco_dim_xyz=(Nx, Ny, Nz),
            reco_vx_xyz=(vx, vy, vz),
            params=params,
        )
    """

    def __init__(
        self,
        opt: Any,
        reco_dim_mm: Union[Tuple[int, int, int], str] = None,
        reco_dim_px: Optional[Tuple[int, int, int]] = None,
        reco_vx: Tuple[float, float, float] = (1.25, 5.0, 1.25),
        match_input: bool = False,
        # FuXSim-specific filter knobs (mapped directly to "filter" param)
        filter_flag: int = 1,         # 0=off, 1=on
        offset_filter: float = 0.005, # same as before
        filter_mode: int = 1,         # FuXSim -fm value
        # Optional HU calibration for FuXSim (if you want later)
        hu: Optional[Tuple[int, str, float]] = None,  # [flag, path, kVp]
    ):
        super().__init__(opt, reco_dim_mm, reco_dim_px, reco_vx, match_input)
        self.filter_flag = filter_flag
        self.offset_filter = offset_filter
        self.filter_mode = filter_mode
        self.hu = hu

    def __call__(self, projections_data: volumeData) -> volumeData:
        """
        Input: projections in projections_data.volume
        Output: reconstructed volume via FuXSim FDK.
        """
        projections = projections_data.volume
        md = copy.deepcopy(projections_data.metadata)

        if projections.ndim == 4 and projections.shape[-1] == 1:
            projections = xp.squeeze(projections, axis=-1)

        # Resolve reco_dim (matches CT or user-provided)
        reco_dim = self._prepare_reco_grid(md)  # (Nx, Ny, Nz)

        # Build FuXSim params
        params: Dict[str, Any] = {
            "filter": [self.filter_flag, self.offset_filter, self.filter_mode]
        }
        if self.hu is not None:
            params["HU"] = list(self.hu)

        # Call FuXSim backend (self.opt must be a FuXSim_OP)
        result = self.opt.reconstruct(
            method="FDK",
            sino=projections,               # backend expects argument name 'sino'
            reco_dim_xyz=tuple(reco_dim),
            reco_vx_xyz=self.reco_vx,
            params=params,
        )

        result = self._crop_to_original_if_needed(result, md)
        md = self._update_metadata(md, result)
        return volumeData(volume=result, metadata=md)

#   RaptorFDK – Raptor_CBCT.reconstruct("FDK", ...)
from chestxsim.io.paths import CALIBRATION_DIR
class RaptorFDK:
    """
    Reconstruction step using the RapTor CBCT backend.

    - Does NOT extend _BaseReconstructionStep (RapTor chooses reco_dim & voxel size).
    - Always calls:

        opt.reconstruct(
            method="FDK",
            projections,  # passed as 'sino' internally
            reco_dim_xyz=None,
            reco_vx_xyz=None,
            params=params,
        )

    and reads final size/voxel from the operator (Raptor_CBCT uses info_reco.txt).
    """

    def __init__(self, opt: Any, params: Optional[Dict[str, Any]] = None):
        """
        opt:    Raptor_CBCT operator
        params: dict passed directly to Raptor_CBCT.reconstruct, e.g.:

            params = {
                "parker": 1,
                # "HU": [1, str(hu_table_path), 120],
                # "cupping": [1, str(cupping_file_path), 120],
            }
        """
        self.opt = opt
        self.params = params or {}
        self.hu_table_path = CALIBRATION_DIR / "hounsfield_func"

    def __call__(self, projections_data: volumeData) -> volumeData:
        projections = projections_data.volume
        md = copy.deepcopy(projections_data.metadata)

        if projections.ndim == 4 and projections.shape[-1] == 1:
            projections = xp.squeeze(projections, axis=-1)

        if "log" in self.params:
            v = self.params["log"]
            if isinstance(v, str):
                v = v.strip().lower() in ("true", "1", "yes", "y", "on")
            self.params["log"] = 1 if bool(v) else 0
        
        if "HU" in self.params:
            hu = self.params["HU"]

            # already explicit: [flag, path, kvp] -> keep as-is
            if isinstance(hu, (list, tuple)):
                pass
            else:
                if isinstance(hu, str):
                    hu = hu.strip().lower() in ("true", "1", "yes", "y", "on")
                hu = 1 if bool(hu) else 0

                if hu == 1:
                    voltage = self.params.get("voltage")

                    if voltage is None:
                        voltage = md.last("voltage")
                        if voltage is None:
                            voltage = md.find("voltage")

                    if voltage is None:
                        raise ValueError(
                            "HU=1 but no voltage found "
                            "(set params['voltage'] or ensure voltage exists in metadata)."
                        )
                    self.params["HU"] = [1, str(self.hu_table_path), voltage]
                else:
                    self.params["HU"] = [0, None, None]

        # Let RapTor decide reconstruction ROI and voxel size
        result = self.opt.reconstruct(
            method="FDK",
            sino=projections,          # Raptor_CBCT.reconstruct(sino=...)
            reco_dim_xyz=None,
            reco_vx_xyz=None,
            params=self.params,
        )

        # Raptor_CBCT updates these attributes from info_reco.txt
        reco_dim = getattr(self.opt, "reco_dim", result.shape)
        vx_size = getattr(self.opt, "vx_size", getattr(md, "voxel_size", None))

        md.dim = reco_dim
        md.voxel_size = vx_size
        md.step_outputs[self.__class__.__name__] = {
            "kernel": type(self.opt).__name__,
            "geometry": getattr(self.opt, "geometry", None),
            "kernel_class": type(self.opt),
            "params": instance_to_constructor_dict(self),
        }
        return volumeData(volume=result, metadata=md)