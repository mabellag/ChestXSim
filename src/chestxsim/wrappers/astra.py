"""
ASTRA-based implementations of the `GeometricOp` interface.

This module adapts ChestXsim layouts and geometries to ASTRA's data structures
and algorithms. It provides:

- `Astra_OP`: base class wrapping ASTRA projectors / reconstructors,
- concrete operators for different modalities:
    * `ASTRA_Tomo`        – 3D cone-beam DCT with custom `cone_vec` geometry,
    * `ASTRA_CBCT`        – 3D cone-beam CT with circular trajectory,
    * `ASTRA_PARALLELCT2D` – 2D parallel-beam CT,
    * `ASTRA_FANBEAMCT`   – 2D fan-beam CT.

Coordinate system (ChestXsim convention):
    [x, y, z]
    - z: top → bottom (detector rows / height)
    - x: left → right (detector cols / width)
    - y: anterior → posterior

Layout conventions:
    ChestXsim:
        VOL 3D : [X, Y, Z]
        SINO 3D: [W, H, A]
        VOL 2D : [X, Y]
        SINO 2D: [A, D]

    ASTRA:
        VOL 3D : [Z, Y, X]
        SINO 3D: [H, A, W]  (rows, angles, cols)
        VOL 2D : [Y, X]
        SINO 2D: [A, D]

Tomosynthesis `cone_vec` geometry per projection uses ASTRA's vector layout:
    (srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ)
where:
    - [srcX, srcY, srcZ] is the source position
    - [dX, dY, dZ] is the detector center
    - [uX, uY, uZ] is the horizontal detector axis (columns, X)
    - [vX, vY, vZ] is the vertical detector axis (rows, Z)
"""


import astra
import numpy as np 
from typing import Optional, Tuple, Dict, Type
from chestxsim.core.geometries import Geometry
from chestxsim.core.device import xp
from chestxsim.wrappers.base import GeometricOp, OpMod
from enum import Enum
from abc import ABC, abstractmethod


class Backend(Enum):
    GPU = "gpu"
    CPU = "cpu"


# ---- ASTRA HOOKS (ALGO SELECTION & LAYOUT HELPERS) -------------------------
# # Customizable behaviour points using hooks
class ASTRAHooks:
    """
    Helper object encapsulating ASTRA-specific behavior:

    - picks correct FP/BP/FDK/FBP algorithm IDs (CPU/GPU, 2D/3D),
    - defines conversions between ChestXsim <-> ASTRA layouts,
    - provides the correct ASTRA data API (data2d/data3d),
    - defines data-kind strings for linking NumPy arrays into ASTRA.

    Keeps backend/dimension-specific logic isolated from `Astra_OP`.
    """
    def __init__(self, backend: Backend, is3d: bool):
        self.backend = backend
        self.is3d = is3d

    #--- algorithm selection ---
    def forward_algo(self) -> str:
        if self.is3d:
            return 'FP3D_CUDA' # only 3d in gpu backend
        return 'FP_CUDA' if self.backend == Backend.GPU else 'FP'

    def backproj_algo(self) -> str:
        if self.is3d:
            return 'BP3D_CUDA'  # only 3d in gpu backend
        return 'BP_CUDA' if self.backend == Backend.GPU else 'BP'

    def recon_algo(self, astra_algo_id) -> str:
        # intended for any astra algorithm 
        if not isinstance(astra_algo_id, str) or not astra_algo_id:
            raise ValueError("Algorithm id must be a non-empty string.")
        
        # backend sanity check
        id_upper = astra_algo_id.upper()
        if "CUDA" in id_upper and self.backend != Backend.GPU:
            raise ValueError(f"Selected backend={self.backend.name} but algorithm '{astra_algo_id}' requires GPU.")
        
        # dimension sanity checkk
        is_3d_algo = ("3D" in id_upper) or (id_upper.startswith("FDK")) 
        if is_3d_algo and not self.is3d:
            raise ValueError(f"Algorithm '{astra_algo_id}' is 3D but geometry is 2D.")
        if (not is_3d_algo) and self.is3d and id_upper.startswith("FBP"):
            raise ValueError(f"Algorithm '{astra_algo_id}' is 2D (FBP) but geometry is 3D.")
        
        return astra_algo_id


    # --- shapes/layouts baked-in ---
    def proj_shape_astra(self,geometry: Geometry) -> Tuple[int, ...]:
        nA = geometry.nprojs
        W = geometry.W
        if self.is3d:
            H = geometry.H
            return (H, nA, W) 
        return (nA, W)

    def vol_shape_astra(self, dim_xyz: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.is3d:
            X, Y, Z = dim_xyz
            return (Z, Y, X)
        X, Y = dim_xyz
        return (Y, X)

    def to_astra_vol(self, v: xp.ndarray) -> xp.ndarray:
        v = v.astype(xp.float32, copy=False)
        return xp.swapaxes(v, 0, 2) if self.is3d else xp.swapaxes(v, 0, 1)

    def from_astra_vol(self, v: xp.ndarray) -> xp.ndarray:
        return xp.swapaxes(v, 0, 2) if self.is3d else xp.swapaxes(v, 0, 1)

    def to_astra_sino(self, s: xp.ndarray) -> xp.ndarray:
        if self.is3d:
            return xp.swapaxes(xp.swapaxes(s, 0, 1), 1, 2).astype(xp.float32, copy=False)
        return s.astype(xp.float32, copy=False) 

    def from_astra_sino(self, s: xp.ndarray) -> xp.ndarray:
        return xp.swapaxes(xp.swapaxes(s, 1, 2), 0, 1) if self.is3d else s

    # --- ASTRA data link + kinds baked-in ---
    def in_kind(self, role_vol: bool) -> str:
        return '-vol' if role_vol else ('-proj3d' if self.is3d else '-sino')

    def out_kind(self, role_vol: bool) -> str:
        return ('-proj3d' if self.is3d else '-sino') if role_vol else '-vol'
    
    @property
    def link_api(self):
        return astra.data3d if self.is3d else astra.data2d
    
    @property
    def proj_name(self):
        return None 

# ---- BASE ASTRA OPERATOR ---------------------------------------------------
class Astra_OP(GeometricOp, ABC):
    """
    Public API (dimension-aware via hooks):
      - project(vol_xyz, vx_xyz)
      - backproject(sino, reco_dim_xyz, reco_vx_xyz)
      - reconstruct(sino, reco_dim_xyz, reco_vx_xyz, filt=None)

    Base handles:
      • 2D vs 3D volume geometry box (since it's dimension-only)
      • common ASTRA run logic and array layouts via hooks

    Concrete ops handle:
      • create_proj_geom() (beam/modality-specific)
    """
    def __init__(self, geometry: Geometry, backend: Backend = Backend.GPU):
        super().__init__(geometry)  
        self.backend = backend
        self.hooks =  ASTRAHooks(backend=backend, is3d=geometry.is3d)

    # ----- Public API ----
    def project(self, vol_xyz: xp.ndarray, vx_xyz: Tuple[float, ...]):
        """Forward projection."""
        in_arr, vol_geom, proj_geom = self._prep_forward(vol_xyz, vx_xyz)
        out_shape = self.hooks.proj_shape_astra(self.geometry)
        algo = self.hooks.forward_algo()
        out = self._run(algo, in_arr, vol_geom, proj_geom,
                             out_shape, role_vol= True)
        return self.hooks.from_astra_sino(out)

    def backproject(self, sino, reco_dim_xyz: Tuple[int, ...], reco_vx_xyz: Tuple[float, ...]):
        """Adjoint/backprojection."""
        in_arr, vol_geom, proj_geom = self._prep_backward(sino, reco_dim_xyz, reco_vx_xyz)
        out_shape = self.hooks.vol_shape_astra(reco_dim_xyz)
        algo = self.hooks.backproj_algo()
        out = self._run(algo, in_arr, vol_geom, proj_geom,
                             out_shape, role_vol=False)
        return self.hooks.from_astra_vol(out)

    def reconstruct(self, 
                    method: str, # raw ASRTA algorithm name
                    sino, 
                    reco_dim_xyz, 
                    reco_vx_xyz, 
                    options: Optional[dict] = None, # goes into cfg['option']
                    iterations: Optional[int] = None
                    ):
        """Reconstruction (FDK or FBP depending on subclass)."""
        in_arr, vol_geom, proj_geom = self._prep_backward(sino, reco_dim_xyz, reco_vx_xyz)
        out_shape = self.hooks.vol_shape_astra(reco_dim_xyz)
        algo = self.hooks.recon_algo(method)
        cfg_extra = {}
        if options:
            cfg_extra['option'] = options

        out = self._run(algo, in_arr, vol_geom, proj_geom, out_shape, role_vol=False, extra_cfg=cfg_extra,  
                        run_kwargs={'iterations': iterations} if iterations is not None else None)
        return self.hooks.from_astra_vol(out)
    
    # --- create ASTRA volume geometry 
    def create_vol_geom(self, dim_xyz: Tuple[int, ...], vx_xyz: Tuple[float, ...]) -> int:
        if self.geometry.is3d:
            X, Y, Z = map(int, dim_xyz); vx, vy, vz = map(float, vx_xyz)
            return astra.create_vol_geom(
                Y, X, Z,
                -X*vx/2, X*vx/2,
                -Y*vy/2, Y*vy/2,
                -Z*vz/2, Z*vz/2
            )
        else:
            X, Y = map(int, dim_xyz); vx, vy = map(float, vx_xyz)
            return astra.create_vol_geom(
                Y, X,
                -X*vx/2, X*vx/2,
                -Y*vy/2, Y*vy/2
            )
    
    # --- create ASTRA projection geometry  (override by modality)
    @abstractmethod
    def create_proj_geom(self) -> int:
        raise NotImplementedError("Subclasses must implement create_proj_geom().")
    
     ## --- prepare volumes in ASTRA layout using hooks 
    def _prep_forward(self, vol_xyz, vx_xyz):
        if hasattr(self.geometry, "fit_to_volume"):
            self.geometry.fit_to_volume(vol_xyz.shape, vx_xyz)
        
        vol_geom  = self.create_vol_geom(vol_xyz.shape, vx_xyz)
        proj_geom = self.create_proj_geom()
        in_arr    = self.hooks.to_astra_vol(vol_xyz)
        return in_arr, vol_geom, proj_geom

    def _prep_backward(self, sino, reco_dim_xyz, reco_vx_xyz):
        vol_geom  = self.create_vol_geom(reco_dim_xyz, reco_vx_xyz)
        proj_geom = self.create_proj_geom()
        in_arr    = self.hooks.to_astra_sino(sino)
        return in_arr, vol_geom, proj_geom
    
    # --- unifed ASTRA runner 
    def _run(self, 
            algorithm_id: str, 
            input_astra, 
            vol_geom, 
            proj_geom,
            out_shape_astra, 
            role_vol: bool,  
            extra_cfg: dict | None = None,
            run_kwargs: dict | None = None
            ) -> xp.ndarray:
        
        """Minimal ASTRA executor"""
        link = self.hooks.link_api
        in_kind  = self.hooks.in_kind(role_vol)
        out_kind = self.hooks.out_kind(role_vol)
        proj_name = self.hooks.proj_name

        input_np = xp.asnumpy(input_astra) if isinstance(input_astra, xp.ndarray) else input_astra
        out_np   = np.zeros(out_shape_astra, dtype=np.float32)

        in_id  = link.link(in_kind,  vol_geom if role_vol else proj_geom, input_np)
        out_id = link.link(out_kind, proj_geom if role_vol else vol_geom, out_np)

        cfg = astra.astra_dict(algorithm_id)
        if role_vol:
            cfg['VolumeDataId']     = in_id
            cfg['ProjectionDataId'] = out_id
        
        else:
            cfg['ProjectionDataId']     = in_id
            cfg['ReconstructionDataId'] = out_id

        proj_id = None
        if proj_name is not None:
            proj_id = astra.create_projector(proj_name, proj_geom, vol_geom)
            cfg['ProjectorId'] = proj_id
                
        if extra_cfg:
            cfg.update(extra_cfg)

        alg_id = astra.algorithm.create(cfg)
        try:
            if run_kwargs:
                astra.algorithm.run(alg_id, **run_kwargs)  # e.g. iterations=100
            else:
                astra.algorithm.run(alg_id)
        # cleanup 
        finally:
            astra.algorithm.delete(alg_id)
            link.delete(in_id)
            link.delete(out_id)
            if proj_id is not None:
                astra.projector.delete(proj_id)

        return xp.asarray(out_np)
    
     
# ---- CONCRETE OPERATORS ----------------------------------------------------
class ASTRA_Tomo(Astra_OP):
    """
    Digital Chest Tomosynthesis (DCT) operator using ASTRA `cone_vec`.

    Implements:
        - fixed flat-panel detector,
        - X-ray source moving linearly along Z,
        - per-projection ASTRA vector geometry.
    """
    def create_proj_geom(self):
        g = self.geometry   
        W, H   = g.W, g.H
        pxW, pxH = g.pxW, g.pxH
        FP = - g.DOD
        vec = np.zeros((g.nprojs, 12), dtype=np.float32)
        for i in range(g.nprojs):
            z = (g.nprojs//2 - i)*g.step_mm
            vec[i,0:3]  = (0, -(g.SDD - g.DOD), z)
            vec[i,3:6]  = (0,  g.DOD, -z*(g.DOD + FP)/(g.SDD - FP))
            vec[i,6:9]  = (pxW, 0, 0)
            vec[i,9:12] = (0, 0, pxH)
        return astra.create_proj_geom('cone_vec', W, H, vec)
    
      
class ASTRA_CBCT(Astra_OP):
    """
    Cone-beam CT operator using ASTRA's built-in `cone` geometry.
    Circular source–detector trajectory around the object.
    """
    def create_proj_geom(self):
        g = self.geometry 
        W, H   = g.W, g.H
        pxW, pxH = g.pxW, g.pxH
        angles =  g.angles
        return astra.create_proj_geom(
                'cone',
                pxH, pxW,
                H, W,
                angles,
                g.SDD - g.DOD,
                g.DOD
            )

# class ASTRA_PARALLELCT3D(AstraOp):
#     """
#     """
#     def create_proj_geom(self):
#         g = self.geometry  # type CTGeom
#         W, H   = g.W, g.H
#         pxW, pxH = g.pxW, g.pxH
#         angles =  g.angles()
#         return astra.create_proj_geom(
#                 'parallel3d',
#                 H, W,
#                 pxH, pxW,
#                 angles,
#                 g.SDD - g.DOD,
#                 g.DOD
#             )
    
class ASTRA_PARALLELCT2D(Astra_OP):
    """2D parallel-beam CT operator using ASTRA `parallel` geometry."""
    def create_proj_geom(self):
        g = self.geometry 
        det_count, det_spacing= g.W, g.pxW
        angles= g.angles
        return astra.create_proj_geom('parallel', 
                                    det_spacing, 
                                    det_count,
                                    angles )
    
class ASTRA_FANBEAMCT(Astra_OP):
    """2D fan-beam CT operator using ASTRA `fanflat` geometry."""
    def create_proj_geom(self):
        g = self.geometry 
        det_count, det_spacing = g.W, g.pxW
        angles = g.angles
        return astra.create_proj_geom('fanflat', 
                                      det_spacing,
                                      det_count, 
                                      angles, 
                                      g.SOD,    # SOURCE-ORIGIN
                                      g.DOD)    # ORIGIN-DET
    

# ---- MODALITY REGISTRY (GEOMETRY → OPERATOR CLASS) -------------------------
MODALITY_REGISTRY: Dict[OpMod, Type[GeometricOp]] = {
    OpMod.DCT:        ASTRA_Tomo,
    OpMod.CBCT:       ASTRA_CBCT,
    OpMod.PARALLEL2D: ASTRA_PARALLELCT2D,
    OpMod.FANBEAM2D:  ASTRA_FANBEAMCT,
}
