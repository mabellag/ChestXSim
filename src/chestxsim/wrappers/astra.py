
import astra
from chestxsim.utility import *
import numpy as np 
from typing import Optional, List, Tuple, Any
from chestxsim.core.geometries import TomoGeometry, CBCTGeometry
from chestxsim.core.device import xp
import cupy as cp
from enum import Enum

"""
    NOTES: 
    Coordinate system: [x,y,z] in ChestXsim         
        z top to bottom :: this aligns with detector rows :: detector height
        |
        | __ x side to side  :: this aligns with detector cols :: detector widht 
        /
    y anterior-posterior 

    Layouts and shapes:
    > ChestXsim layout:
        VOL 3D: [X,Y,Z]
        SINO 3D: [W,H,A]
        VOL 2D: [X,Y]
        SINO 2D: [A,D]

    > ASTRA layout:
        VOL 3D: [Z,Y,X]
        SINO 3D: [H,A,W] (rows, angles, cols)
        VOL 2D: [Y,X]
        SINO 2D: [A,D]

    Geometry Layout (per 3D projection) using cone_vec for Tomosynthesis 
    ( srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ ) according to astra 
    ---------------------------------
        vectors[i, 0:3]   : Source position:
                           (X = 0, Y = -SDD + DOD, Z = moving along step)
        vectors[i, 3:6]   : Detector center position:
                           (X = 0, Y = DOD, Z = adjusted to maintain magnification)
        vectors[i, 6:9]   : Vector from (0,0) to (0,1) pixel: horizontal detector axis (X)
        vectors[i, 9:12]  : Vector from (0,0) to (1,0) pixel: vertical detector axis (Z)
     
"""

class GeometryMode(Enum):
    CBCT3D = "cbct-3d"   # cone-beam volume <-> FDK
    CBCT2D = "cbct-2d"   # per-slice fan-beam or parallel <-> FBP

class Role(Enum):
    VOL  = "-vol"
    SINO = "-sino"    

CFG_KEYS = {
    (Role.VOL,  Role.SINO): dict(in_key='VolumeDataId',        out_key='ProjectionDataId'),
    (Role.SINO, Role.VOL ): dict(in_key='ProjectionDataId',    out_key='ReconstructionDataId'),
}

# =======================
# BASE OPERATOR 
# =======================
class Astra_OP:
    """
    Astra simulation base object.
    It implementsa a generic CUDA runner for projection, backprojection and fdk. 
    """
    def __init__(self):
        self.proj_size: List[int] = [1, 1] # number of pixels in the detector [width, height]
        self.px_size: List[float] = [1, 1] # size of pixels in the detector [width, height]
        self.nprojs: int 
      
       
    def create_vol_geom(self, vol_dim: Tuple[int, int, int], vx_size: Tuple[float, float, float]):
        """
        Create ASTRA 3D volume geometry. 
      
        vol_dim (tuple): Volume dimensions in voxels [x, y, z]
        vx_size (tuple): Voxel size in mm [x, y, z]
        
        Returns:
        ASTRA volume geometry
        """

        # ASTRA expects [rows, cols, slices] = [y, x, z] for geometry creation
        # But our volume is [x, y, z], so we need to swap x and y
        return astra.create_vol_geom(
            vol_dim[1], vol_dim[0], vol_dim[2],  # y, x, z dimensions
            -vol_dim[0] * vx_size[0] / 2, vol_dim[0] * vx_size[0] / 2,  # x bounds
            -vol_dim[1] * vx_size[1] / 2, vol_dim[1] * vx_size[1] / 2,  # y bounds
            -vol_dim[2] * vx_size[2] / 2, vol_dim[2] * vx_size[2] / 2)  # z bounds

    
    def create_proj_geom(self):
        """
        Create ASTRA projection geometry 3D.
        Implemented by subclases 
        """
        raise NotImplementedError
    
    def _run_astra_CUDA(
        self,
        algorithm_id: str,
        input_astra,                 # np.ndarray or cp.ndarray; already in ASTRA layout
        role: Role,                  # Role.VOL (input is volume) or Role.SINO (input is sinogram)
        vol_geom,                    # ASTRA volume geometry
        proj_geom,                   # ASTRA projection geometry
        out_shape_astra: tuple,      # output shape in ASTRA layout
        is3d: bool = True,
        extra_cfg: dict | None = None,
    ):
        """
        Minimal ASTRA  CUDA runner.

        - Assumes input is already in ASTRA layout (see table above).
        - Allocates output (NumPy), links input/output, wires cfg keys from `CFG_KEYS`.
        - Adds ProjectorId **only for 3D** CUDA algorithms.
        - Runs and returns output as CuPy array.
        """
        
        use_numpy_in  = isinstance(input_astra, np.ndarray)
        if not use_numpy_in:
            input_np = cp.asnumpy(input_astra).astype(np.float32, copy=False)
        else:
            input_np = input_astra.astype(np.float32, copy=False)

        out_np = np.zeros(out_shape_astra, dtype=np.float32)

        # Pick correct ASTRA link group 
        if is3d:
            if role is Role.VOL:
                in_dtype,  in_geom  = '-vol',    vol_geom
                out_dtype, out_geom = '-proj3d', proj_geom
            else:  # role is Role.SINO (we're providing a 3D sinogram)
                in_dtype,  in_geom  = '-proj3d', proj_geom
                out_dtype, out_geom = '-vol',    vol_geom
        else:
            if role is Role.VOL:
                in_dtype,  in_geom  = '-vol',  vol_geom
                out_dtype, out_geom = '-sino', proj_geom
            else:
                in_dtype,  in_geom  = '-sino', proj_geom
                out_dtype, out_geom = '-vol',  vol_geom
        # Link
        link = astra.data3d if is3d else astra.data2d
        in_id  = link.link(in_dtype,  in_geom,  input_np)
        out_id = link.link(out_dtype, out_geom, out_np)
        # Config
        cfg = astra.astra_dict(algorithm_id)
        keys = CFG_KEYS[(role, Role.SINO if role is Role.VOL else Role.VOL)]
        cfg[keys['in_key']]  = in_id
        cfg[keys['out_key']] = out_id
        cfg['ProjectorId'] = astra.create_projector('cuda3d' if is3d else 'cuda', proj_geom, vol_geom)
        if extra_cfg:
            cfg.update(extra_cfg)

        alg_id = astra.algorithm.create(cfg)
        try:
            astra.algorithm.run(alg_id)
        finally:
            astra.algorithm.delete(alg_id)
            link.delete(in_id)
            link.delete(out_id)

        return cp.asarray(out_np)

    # -- 3D high-level-
    def project(self, input_volume: cp.ndarray, vx_size: Tuple[float, float, float], is3d= True) -> cp.ndarray:
        """
        Compute forward projection using ASTRA with GPU.

        Parameters:
        input_volume (cp.ndarray): 3D input volume, shape [X, Y, Z]
        vx_size (tuple): Voxel size in mm (vx, vy, vz)

        Returns:
        cp.ndarray: 3D sinogram, shape [W, H, Angles]
        """
        if self.DOD is None:
            self.DOD = self.set_DOD(input_volume.shape, vx_size)

        # Create ASTRA volume and projection geometry
        vol_geom = self.create_vol_geom(input_volume.shape, vx_size)
        proj_geom = self.create_proj_geom()
        
        # Swap from [X, Y, Z] to [Z, Y, X] as expected by ASTRA and Link to 
        input_volume_astra = cp.swapaxes(input_volume, 0, 2).astype(cp.float32).copy()
        out_shape_astra = (self.proj_size[1], self.nprojs, self.proj_size[0]) # (H,A,W)
        sinogram_astra = self._run_astra_CUDA("FP3D_CUDA", 
                                              input_volume_astra,
                                              Role.VOL,
                                              vol_geom, proj_geom, 
                                              out_shape_astra,
                                              is3d)
        
        return cp.swapaxes(cp.swapaxes(sinogram_astra, 0, 2), 1, 2) # go back to chestxsim layout 
    
    def fdk(self, proj_data: cp.ndarray, reco_dim, reco_vx_size, is3d=True, **kwargs) -> cp.ndarray:
        sino = cp.swapaxes(cp.swapaxes(proj_data, 0, 1), 1, 2).astype(cp.float32, copy=False)
        vol_geom  = self.create_vol_geom(reco_dim, reco_vx_size)
        proj_geom = self.create_proj_geom()
        out_shape_astra = (reco_dim[2], reco_dim[1], reco_dim[0])  # (Z,Y,X)

        reco = self._run_astra_CUDA("FDK_CUDA", sino, Role.SINO, vol_geom, proj_geom, out_shape_astra, is3d=True)
        return cp.swapaxes(reco, 0, 2)

    def backproject(self, proj_data: cp.ndarray, reco_dim, reco_vx_size, is3d=True, **kwargs) -> cp.ndarray:
        sino = cp.swapaxes(cp.swapaxes(proj_data, 0, 1), 1, 2).astype(cp.float32, copy=False)
        vol_geom  = self.create_vol_geom(reco_dim, reco_vx_size)
        proj_geom = self.create_proj_geom()
        out_shape_astra = (reco_dim[2], reco_dim[1], reco_dim[0])  # (Z,Y,X)

        reco = self._run_astra_CUDA("BP3D_CUDA", sino, Role.SINO, vol_geom, proj_geom, out_shape_astra, is3d=True)
        return cp.swapaxes(reco, 0, 2)

           
class Astra_Tomo(Astra_OP):
    r"""Tomosynthesis simulation object.
    Can be used to execute Astra projection and backprojection operations.
    It can either be used by inputing numpy arrays
    with the methods forward_operator and adjoint_operator.
    """

    def __init__(self, geometry: Optional[TomoGeometry] = None):
        super().__init__()
        
        if geometry is not None:
            self.configure_from_geometry(geometry)
        else:
            # Default values if no geometry provided
            self.SDD: float = 1 # Source to detector distance in mm
            self.DOD: Optional[float] = None # Detector object distance in mm (will be calculated or set)
            self.step: float = 1 # Step size for tomosynthesis movement in mm
            
            self.bucky: float = 1  # Size of the bucky detector in mm
            # self.angles: Any = xp.array([0])  # Projection angles (default is single angle)
            

    # def _get_norm_weight(self):
    #     return 1/self.nprojs

    def set_DOD(self, ct_dim: Tuple[int, int, int], voxel_size: Tuple[float, float, float]) -> float:
        """Compute the detector-object distance based on volume dimensions.
        
        Parameters:
        ct_dim (tuple): Volume dimensions in voxels [x, y, z]
        voxel_size (tuple): Voxel size in mm [x, y, z]
        
        Returns:
        float: Calculated detector-object distance
        """
        return self.bucky + ct_dim[1] * voxel_size[1] / 2

    def create_proj_geom(self, DOD: Optional[float] = None):
        """Create projection geometry for tomosynthesis.
        
        Parameters:
        DOD (float, optional): Detector-object distance. If None, uses the current DOD value.
        
        Returns:
        Any: astra projection geometry object


        NOTES:
            vectors[i, 2] = (self.nprojs // 2 - i) * self.step  # Z — moves from top to bottom
                i= 0: this is initial position then source_z = ....

        """

        if DOD is not None:
            self.DOD = DOD
            
        if self.DOD is None:
            raise ValueError("DOD must be set either through compute_DOD or directly passed to create_proj_geom")

        # vectors = np.zeros((self.nprojs + 1, 12)) #vectors should be np to use in astra create proj geom
        vectors = np.zeros((self.nprojs, 12))
        FP = -self.DOD  # Focus plane at detector surface, can be changed to different focus plane
        
        # for i in range(self.nprojs + 1):
        for i in range(self.nprojs):
            # Source coordinates (for CBCT should change 0,1, but for tomo, only 2)
            vectors[i, 0] = 0
            vectors[i, 1] = -(self.SDD - self.DOD)
            vectors[i, 2] = (self.nprojs // 2 - i) * self.step
            
            # Center of detector
            vectors[i, 3] = 0
            vectors[i, 4] = self.DOD
            vectors[i, 5] = -(self.nprojs // 2 - i) * self.step * (self.DOD + FP) / (self.SDD - FP)
            
            # Vector from detector pixel (0,0) to (0,1)
            vectors[i, 6] = self.px_size[0]
            vectors[i, 7] = 0
            vectors[i, 8] = 0
            
            # Vector from detector pixel (0,0) to (1,0)
            vectors[i, 9] = 0
            vectors[i, 10] = 0
            vectors[i, 11] = self.px_size[1]
            
        return astra.create_proj_geom('cone_vec', self.proj_size[0], self.proj_size[1], vectors)

    def configure_from_dict(self,config: dict) -> "Astra_Tomo":
        self.proj_size = [config["geometry"]["detector_size"][0] // config["geometry"]["binning_proj"],
                         config["geometry"]["detector_size"][1] // config["geometry"]["binning_proj"]]
        self.px_size = [config["geometry"]["pixel_size"][0] * config["geometry"]["binning_proj"],
                       config["geometry"]["pixel_size"][1] * config["geometry"]["binning_proj"]]
        self.SDD = config["geometry"]["SDD"]
        self.bucky = config["geometry"]["bucky"]
        self.step = config["geometry"]["step"]
        self.nprojs = config["geometry"]["nsteps"]
        return self
    
    def configure_from_geometry(self, geometry: TomoGeometry)->"Astra_Tomo":
        self.geometry = geometry
        self.proj_size = [
            geometry.detector_size[0] // geometry.binning_proj,
            geometry.detector_size[1] // geometry.binning_proj
        ]
        self.px_size = [
            geometry.pixel_size[0] * geometry.binning_proj,
            geometry.pixel_size[1] * geometry.binning_proj
        ]
        
        # Set tomosynthesis-specific parameters
        self.SDD = geometry.SDD
        self.bucky = geometry.bucky
        self.step = geometry.step
        self.nprojs = geometry.nstep + 1
        self.DOD: Optional[float] = None  # Will be calculated when needed
        self.angles: Any = xp.array([0])  # Single angle for tomo
        
        return self

class Astra_CBCT(Astra_OP):
    
    def __init__(self, geometry: CBCTGeometry):
        super().__init__()
        if geometry is not None:
            self.configure_from_geometry(geometry)
        else:
            self.SDD: float = 1 # Source to detector distance in mm
            self.DOD: float = 1  # Detector to object distance in mm  
            self.step_angle: float = 1 # Angular step between projections in degrees
            self.init_angle: float = 0 # Initial angle in degrees
            self.nprojs: int = 360 # Number of projections

        self.angles: np.ndarray = np.array(range(360)) / 180 * np.pi  # Projection angles in radians
        self.geometry_mode: GeometryMode = GeometryMode.CBCT3D
        self.fbp_filter: str  = 'ram-lak'
        self.beam_geometry: str = 'fanflat' # parallel

    def set_geometry_mode(self, mode: GeometryMode):
        self.geometry_mode = mode

    def set_fbp_filter(self, name: str):
        self.fbp_filter = name  # 'ram-lak', 'hann', 'shepp-logan', 

    def set_beam_geometry(self, name:str):
        self.beam_geometry = name

    def create_proj_geom(self):
        """Create projection geometry for CBCT.
        
        Returns:
        astra projection geometry
        """
        if self.geometry_mode == GeometryMode.CBCT3D:
            return astra.create_proj_geom(
                'cone',
                self.px_size[1], self.px_size[0],
                self.proj_size[1], self.proj_size[0],
                self.angles,
                self.SDD - self.DOD,
                self.DOD
            )
        
        # ---- CBCT2D: 
        det_count = int(self.proj_size[0])                       
        det_spacing = float(self.px_size[0])                       
        angles    = np.asarray(self.angles, dtype=np.float32)    
        SOD       = float(self.SDD - self.DOD)
        ODD       = float(self.DOD)

        if self.beam_geometry == 'parallel':
            return astra.create_proj_geom('parallel', det_spacing, det_count, angles)

        if self.beam_geometry == 'fanflat':
            return astra.create_proj_geom('fanflat', det_spacing, det_count, angles, SOD, ODD)
            
    def _create_vol_geom_2d(self, X:int, Y:int, vx:float, vy:float):
        return astra.create_vol_geom(
            int(Y), int(X),
            -float(X)*float(vx)/2.0,  float(X)*float(vx)/2.0,
            -float(Y)*float(vy)/2.0,  float(Y)*float(vy)/2.0
        )

    def _parse_vx2d(self, vx_size):
        if len(vx_size) >= 2:
            return float(vx_size[0]), float(vx_size[1])
        raise ValueError("vx_size must be (vx, vy) or (vx, vy, vz)")
    
    def project(self, input_img: cp.ndarray, vx_size) -> cp.ndarray:
        # 3D mode delegates to parent 
        if self.geometry_mode == GeometryMode.CBCT3D:
            return super().project(input_img, vx_size)

        if input_img.ndim != 2:
            raise ValueError("CBCT2D project expects a single slice (X,Y).")
        
        X, Y = input_img.shape
        vx, vy = self._parse_vx2d(vx_size)

        A = int(len(self.angles))
        D = int(self.proj_size[0])

        vol_geom_2d  = self._create_vol_geom_2d(X, Y, vx, vy)
        proj_geom_2d = self.create_proj_geom()

        # convert to ASTRA layout [Y,X]
        img_astra = cp.swapaxes(input_img.astype(cp.float32, copy=False), 0, 1)
        out_shape_astra = (A, D)  # [angles, detector]

        sino_astra = self._run_astra_CUDA(
            "FP_CUDA",
            img_astra,
            Role.VOL,
            vol_geom_2d,
            proj_geom_2d,
            out_shape_astra,
            is3d=False 
        )
        return sino_astra 


    def backproject(self, proj_data: cp.ndarray, reco_dim, reco_vx_size, **kwargs) -> cp.ndarray:
        # 3d mode delegates to parent 
        if self.geometry_mode == GeometryMode.CBCT3D:
            return super().backproject(proj_data, reco_dim, reco_vx_size, **kwargs)

        if proj_data.ndim != 2:
            raise ValueError("CBCT2D backproject expects sino (A,D).")

        X, Y = reco_dim
        vx, vy = self._parse_vx2d(reco_vx_size)
        A, D = proj_data.shape

        vol_geom_2d  = self._create_vol_geom_2d(X, Y, vx, vy)
        proj_geom_2d = self.create_proj_geom()
        out_shape_astra = (Y, X)  # [rows, cols] in ASTRA

        # input already ASTRA layout [A,D]
        sino_astra = proj_data.astype(cp.float32, copy=False)

        vol_astra = self._run_astra_CUDA(
            "BP_CUDA",
            sino_astra,
            Role.SINO,
            vol_geom_2d,
            proj_geom_2d,
            out_shape_astra,
            is3d=False
        )

        # ASTRA -> [X,Y]
        return cp.swapaxes(vol_astra, 0, 1)

    def fbp(self, proj_data: cp.ndarray, reco_dim, reco_vx_size, **kwargs) -> cp.ndarray:
        # 3D mode delegates to parent (FDK)
        if self.geometry_mode == GeometryMode.CBCT3D:
            return super().fdk(proj_data, reco_dim, reco_vx_size, **kwargs)

        if proj_data.ndim != 2:
            raise ValueError("CBCT2D fbp expects sino (A,D).")

        X, Y = reco_dim
        vx, vy = self._parse_vx2d(reco_vx_size)
        A, D = proj_data.shape

        vol_geom_2d  = self._create_vol_geom_2d(X, Y, vx, vy)
        proj_geom_2d = self.create_proj_geom()
        out_shape_astra = (Y, X)

        sino_astra = proj_data.astype(cp.float32, copy=False)

        vol_astra = self._run_astra_CUDA(
            "FBP_CUDA",
            sino_astra,
            Role.SINO,
            vol_geom_2d,
            proj_geom_2d,
            out_shape_astra,
            is3d=False,
            extra_cfg={'FilterType': self.fbp_filter}
        )

        # ASTRA -> [X,Y]
        return cp.swapaxes(vol_astra, 0, 1)

    def configure_from_dict(self, config: dict)-> "Astra_CBCT":
        self.proj_size = [config["geometry"]["detector_size"][0] // config["geometry"]["binning_proj"],
                         config["geometry"]["detector_size"][1] // config["geometry"]["binning_proj"]]
        self.px_size = [config["geometry"]["pixel_size"][0] * config["geometry"]["binning_proj"],
                       config["geometry"]["pixel_size"][1] * config["geometry"]["binning_proj"]]
        self.SDD = config["geometry"]["SDD"]
        self.DOD = config["geometry"]["DOD"]
        self.step_angle = config["geometry"]["step_angle"]
        self.init_angle = config["geometry"]["init_angle"]
        self.nprojs = config["geometry"]["nprojs"]
        return self
    
    def configure_from_geometry(self, geometry: CBCTGeometry) ->"Astra_CBCT":
        self.geometry = geometry
        self.proj_size = [
            geometry.detector_size[0] // geometry.binning_proj,
            geometry.detector_size[1] // geometry.binning_proj
        ]
        self.px_size = [
            geometry.pixel_size[0] * geometry.binning_proj,
            geometry.pixel_size[1] * geometry.binning_proj
        ]
        
        self.SDD = geometry.SDD
        self.DOD = geometry.DOD
        self.step_angle = geometry.step_angle
        self.init_angle = geometry.init_angle
        self.nprojs = geometry.nprojs
        return self 