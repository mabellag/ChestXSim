
# import astra
# from chestxsim.utility import *
# import numpy as np 
# from typing import Optional, List, Tuple, Any
# from chestxsim.core.geometries import TomoGeometry, CBCTGeometry
# from chestxsim.core.device import xp
# import cupy as cp

# """
#     NOTES: 
#     Coordinate system: [x,y,z] in ChestXsim         
#         z top to bottom :: this aligns with detector rows :: detector height
#         |
#         | __ x side to side  :: this aligns with detector cols :: detector widht 
#         /
#     y anterior-posterior 

#     ASTRA expects volume dimensions in order: [ rows , cols, slices] 
#     which means in our cordinates [cols, rows, slices] = [y, x, z]
#     Sinogram shape in astra = [detector_rows, angles, detector_cols] => [y, angles, x].

#     Geometry Layout (per projection):
#     ( srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ ) according to astra 
#     ---------------------------------
#         vectors[i, 0:3]   : Source position:
#                            (X = 0, Y = -SDD + DOD, Z = moving along step)
#         vectors[i, 3:6]   : Detector center position:
#                            (X = 0, Y = DOD, Z = adjusted to maintain magnification)
#         vectors[i, 6:9]   : Vector from (0,0) to (0,1) pixel: horizontal detector axis (X)
#         vectors[i, 9:12]  : Vector from (0,0) to (1,0) pixel: vertical detector axis (Z)
     
# """

# class Astra_OP:
#     r"""Astra simulation object.
#     Can be used to execute Astra projection and backprojection operations.
#     with the methods forward_operator and adjoint_operator. 
#     """
#     def __init__(self):
#         self.proj_size: List[int] = [1, 1] # number of pixels in the detector [width, height]
#         self.px_size: List[float] = [1, 1] # size of pixels in the detector [width, height]
#         #self._fov = self._compute_fov()
       
#     def create_vol_geom(self, vol_dim: Tuple[int, int, int], vx_size: Tuple[float, float, float]):
#         """Create volume geometry on demand based on input dimensions and voxel size.
#         Parameters:
#         vol_dim (tuple): Volume dimensions in voxels [x, y, z]
#         vx_size (tuple): Voxel size in mm [x, y, z]
        
#         Returns:
#         astra volume geometry
#         """

#         # ASTRA expects [rows, cols, slices] = [y, x, z] for geometry creation
#         # But our volume is [x, y, z], so we need to swap x and y
#         return astra.create_vol_geom(
#             vol_dim[1], vol_dim[0], vol_dim[2],  # y, x, z dimensions
#             -vol_dim[0] * vx_size[0] / 2, vol_dim[0] * vx_size[0] / 2,  # x bounds
#             -vol_dim[1] * vx_size[1] / 2, vol_dim[1] * vx_size[1] / 2,  # y bounds
#             -vol_dim[2] * vx_size[2] / 2, vol_dim[2] * vx_size[2] / 2)  # z bounds
    
#         # return astra.create_vol_geom(
#         #     vol_dim[1], vol_dim[0], vol_dim[2],  # y, x, z dimensions
#         #     -vol_dim[1] * vx_size[1] / 2, vol_dim[1] * vx_size[1] / 2,  # x bounds
#         #     -vol_dim[0] * vx_size[0] / 2, vol_dim[0] * vx_size[0] / 2,  # y bounds
#         #     -vol_dim[2] * vx_size[2] / 2, vol_dim[2] * vx_size[2] / 2)  # z bounds
    
#     def create_proj_geom(self):
#         """Create projection geometry - to be implemented by subclasses."""
#         raise NotImplementedError("Subclasses must implement this method")
    
#     def forward_operator(self, input_volume: cp.ndarray, vx_size: Tuple[float, float, float]) -> cp.ndarray:
#         """
#         Compute forward projection using ASTRA with GPU.

#         Parameters:
#         input_volume (cp.ndarray): 3D input volume, shape [X, Y, Z]
#         vx_size (tuple): Voxel size in mm (vx, vy, vz)

#         Returns:
#         cp.ndarray: 3D sinogram, shape [W, H, Angles]
#         """
#         if self.DOD is None:
#             self.DOD = self.set_DOD(input_volume.shape, vx_size)

#         # Create ASTRA volume and projection geometry
#         vol_geom = self.create_vol_geom(input_volume.shape, vx_size)
#         proj_geom = self.create_proj_geom()
        
#         # Swap from [X, Y, Z] to [Z, Y, X] as expected by ASTRA and Link to 
#         input_volume_astra = cp.swapaxes(input_volume, 0, 2).astype(cp.float32).copy()
#         vol_id = astra.data3d.link('-vol', vol_geom, input_volume_astra)
        

#         #  Create sinogram astra volume and link to 
#         # REVISAR SINO SHAPE 
#         sino_shape = (self.proj_size[1], self.nprojs, self.proj_size[0])  # [rows, angles, cols]
#         # sino_shape = (self.proj_size[1], self.nprojs, self.proj_size[0])  # [rows, angles, cols]
#         sinogram_astra = cp.zeros(sino_shape, dtype=cp.float32)
#         sino_id = astra.data3d.link('-sino', proj_geom, sinogram_astra)

#         # Create GPU forward projection algorithm and run it 
#         cfg = astra.astra_dict('FP3D_CUDA')
#         cfg['VolumeDataId'] = vol_id
#         cfg['ProjectionDataId'] = sino_id
#         cfg['ProjectorId'] = astra.create_projector('cuda3d', proj_geom, vol_geom)
#         alg_id = astra.algorithm.create(cfg)
#         astra.algorithm.run(alg_id)

#         # Cleanup
#         astra.algorithm.delete(alg_id)
#         astra.data3d.delete(vol_id)
#         astra.data3d.delete(sino_id)
       
#         # ASTRA sinogram is [rows, angles, cols], convert to [width, height, angles]
#         # i.e. from [rows, angles, cols] -> swapaxes(0,2) -> [cols, angles, rows] 
#         # then swapaxes(1,2) -> [cols, rows, angles] [WIDTH, HEIGHT ]
#         return cp.swapaxes(cp.swapaxes(sinogram_astra, 0, 2), 1, 2)
    

#     def adjoint_operator(self, proj_data: cp.ndarray, 
#                      reco_dim: Tuple[int, int, int], 
#                      reco_vx_size: Tuple[float, float, float],
#                      **kwargs) -> cp.ndarray:
#         """
#         Compute backprojection using ASTRA with GPU.

#         Parameters:
#         proj_data (cp.ndarray): 3D projection data in [W, H, Angles] format
#         reco_dim (tuple): Output volume shape [X, Y, Z] px 
#         reco_vx_size (tuple): Voxel size in mm [vx, vy, vz]

#         Returns:
#         cp.ndarray: Backprojected volume [X, Y, Z]

#         """

#         # print(proj_data.shape)
        
#         # Correct the shape of sinogram: [W, H, A] -> [H, A, W]
#         sino = xp.swapaxes(xp.swapaxes(proj_data, 0, 1), 1, 2).astype(xp.float32).copy()
#         # print(sino.shape)

#         vol_geom = self.create_vol_geom(reco_dim, reco_vx_size)
#         proj_geom = self.create_proj_geom()
#         # print(proj_geom)

#         # Allocate CuPy array for the output volume [Z, Y, X] (ASTRA layout)
#         vol_data = cp.zeros((reco_dim[2], reco_dim[1], reco_dim[0]), dtype=xp.float32)

#         # Link arrays to ASTRA
#         sino_id = astra.data3d.link('-sino', proj_geom, sino)
#         vol_id = astra.data3d.link('-vol', vol_geom, vol_data)

#         # Configure ASTRA backprojection
#         cfg = astra.astra_dict('FDK_CUDA') #BP3D_CUDA
#         cfg['ProjectionDataId'] = sino_id
#         cfg['ReconstructionDataId'] = vol_id 
#         cfg['ProjectorId'] = astra.create_projector('cuda3d', proj_geom, vol_geom)

#         alg_id = astra.algorithm.create(cfg)
#         astra.algorithm.run(alg_id)

#         # Cleanup
#         astra.algorithm.delete(alg_id)
#         astra.data3d.delete(vol_id)
#         astra.data3d.delete(sino_id)

#         # Convert ASTRA [Z, Y, X] to [X, Y, Z]
#         # return cp.swapaxes(vol_data, 0, 2)*1/self.nprojs
#         return cp.swapaxes(vol_data, 0, 2)
    
    
# class Astra_Tomo(Astra_OP):
#     r"""Tomosynthesis simulation object.
#     Can be used to execute Astra projection and backprojection operations.
#     It can either be used by inputing numpy arrays
#     with the methods forward_operator and adjoint_operator.
#     """

#     def __init__(self, geometry: Optional[TomoGeometry] = None):
#         super().__init__()
        
#         if geometry is not None:
#             self.configure_from_geometry(geometry)
#         else:
#             # Default values if no geometry provided
#             self.SDD: float = 1 # Source to detector distance in mm
#             self.DOD: Optional[float] = None # Detector object distance in mm (will be calculated or set)
#             self.step: float = 1 # Step size for tomosynthesis movement in mm
#             self.nprojs: int = 1 # Number of projections
#             self.bucky: float = 1  # Size of the bucky detector in mm
#             # self.angles: Any = xp.array([0])  # Projection angles (default is single angle)
            

#     # def _get_norm_weight(self):
#     #     return 1/self.nprojs

#     def set_DOD(self, ct_dim: Tuple[int, int, int], voxel_size: Tuple[float, float, float]) -> float:
#         """Compute the detector-object distance based on volume dimensions.
        
#         Parameters:
#         ct_dim (tuple): Volume dimensions in voxels [x, y, z]
#         voxel_size (tuple): Voxel size in mm [x, y, z]
        
#         Returns:
#         float: Calculated detector-object distance
#         """
#         return self.bucky + ct_dim[1] * voxel_size[1] / 2

#     def create_proj_geom(self, DOD: Optional[float] = None):
#         """Create projection geometry for tomosynthesis.
        
#         Parameters:
#         DOD (float, optional): Detector-object distance. If None, uses the current DOD value.
        
#         Returns:
#         Any: astra projection geometry object


#         NOTES:
#             vectors[i, 2] = (self.nprojs // 2 - i) * self.step  # Z — moves from top to bottom
#                 i= 0: this is initial position then source_z = ....

#         """

#         if DOD is not None:
#             self.DOD = DOD
            
#         if self.DOD is None:
#             raise ValueError("DOD must be set either through compute_DOD or directly passed to create_proj_geom")

#         # vectors = np.zeros((self.nprojs + 1, 12)) #vectors should be np to use in astra create proj geom
#         vectors = np.zeros((self.nprojs, 12))
#         FP = -self.DOD  # Focus plane at detector surface, can be changed to different focus plane
        
#         # for i in range(self.nprojs + 1):
#         for i in range(self.nprojs):
#             # Source coordinates (for CBCT should change 0,1, but for tomo, only 2)
#             vectors[i, 0] = 0
#             vectors[i, 1] = -(self.SDD - self.DOD)
#             vectors[i, 2] = (self.nprojs // 2 - i) * self.step
            
#             # Center of detector
#             vectors[i, 3] = 0
#             vectors[i, 4] = self.DOD
#             vectors[i, 5] = -(self.nprojs // 2 - i) * self.step * (self.DOD + FP) / (self.SDD - FP)
            
#             # Vector from detector pixel (0,0) to (0,1)
#             vectors[i, 6] = self.px_size[0]
#             vectors[i, 7] = 0
#             vectors[i, 8] = 0
            
#             # Vector from detector pixel (0,0) to (1,0)
#             vectors[i, 9] = 0
#             vectors[i, 10] = 0
#             vectors[i, 11] = self.px_size[1]
            
#         return astra.create_proj_geom('cone_vec', self.proj_size[0], self.proj_size[1], vectors)

#     def configure_from_dict(self,config: dict) -> "Astra_Tomo":
#         self.proj_size = [config["geometry"]["detector_size"][0] // config["geometry"]["binning_proj"],
#                          config["geometry"]["detector_size"][1] // config["geometry"]["binning_proj"]]
#         self.px_size = [config["geometry"]["pixel_size"][0] * config["geometry"]["binning_proj"],
#                        config["geometry"]["pixel_size"][1] * config["geometry"]["binning_proj"]]
#         self.SDD = config["geometry"]["SDD"]
#         self.bucky = config["geometry"]["bucky"]
#         self.step = config["geometry"]["step"]
#         self.nprojs = config["geometry"]["nsteps"]
#         return self
    
#     def configure_from_geometry(self, geometry: TomoGeometry)->"Astra_Tomo":
#         self.geometry = geometry
#         self.proj_size = [
#             geometry.detector_size[0] // geometry.binning_proj,
#             geometry.detector_size[1] // geometry.binning_proj
#         ]
#         self.px_size = [
#             geometry.pixel_size[0] * geometry.binning_proj,
#             geometry.pixel_size[1] * geometry.binning_proj
#         ]
        
#         # Set tomosynthesis-specific parameters
#         self.SDD = geometry.SDD
#         self.bucky = geometry.bucky
#         self.step = geometry.step
#         self.nprojs = geometry.nstep + 1
#         self.DOD: Optional[float] = None  # Will be calculated when needed
#         self.angles: Any = xp.array([0])  # Single angle for tomo
        
#         return self


# class Astra_CBCT(Astra_OP):
    
#     def __init__(self, geometry: CBCTGeometry):
#         super().__init__()
#         if geometry is not None:
#             self.configure_from_geometry(geometry)
#         else:
#             self.SDD: float = 1 # Source to detector distance in mm
#             self.DOD: float = 1  # Detector to object distance in mm  
#             self.step_angle: float = 1 # Angular step between projections in degrees
#             self.init_angle: float = 0 # Initial angle in degrees
#             self.nprojs: int = 360 # Number of projections

#         self.angles: np.ndarray = np.array(range(360)) / 180 * np.pi  # Projection angles in radians

#     def create_proj_geom(self):
#         """Create projection geometry for CBCT.
        
#         Returns:
#         astra projection geometry
#         """
#         return astra.create_proj_geom(
#             'cone', 
#             self.px_size[1], self.px_size[0],      # Detector pixel size (height, width)
#             self.proj_size[1], self.proj_size[0],  # Detector size (rows, cols)
#             self.angles, 
#             self.SDD - self.DOD,                   # Source to object distance
#             self.DOD                               # Object to detector distance
#         )
    

#     def configure_from_dict(self, config: dict)-> "Astra_CBCT":
#         self.proj_size = [config["geometry"]["detector_size"][0] // config["geometry"]["binning_proj"],
#                          config["geometry"]["detector_size"][1] // config["geometry"]["binning_proj"]]
#         self.px_size = [config["geometry"]["pixel_size"][0] * config["geometry"]["binning_proj"],
#                        config["geometry"]["pixel_size"][1] * config["geometry"]["binning_proj"]]
#         self.SDD = config["geometry"]["SDD"]
#         self.DOD = config["geometry"]["DOD"]
#         self.step_angle = config["geometry"]["step_angle"]
#         self.init_angle = config["geometry"]["init_angle"]
#         self.nprojs = config["geometry"]["nprojs"]
#         return self
    
#     def configure_from_geometry(self, geometry: CBCTGeometry) ->"Astra_CBCT":
#         self.geometry = geometry
#         self.proj_size = [
#             geometry.detector_size[0] // geometry.binning_proj,
#             geometry.detector_size[1] // geometry.binning_proj
#         ]
#         self.px_size = [
#             geometry.pixel_size[0] * geometry.binning_proj,
#             geometry.pixel_size[1] * geometry.binning_proj
#         ]
        
#         self.SDD = geometry.SDD
#         self.DOD = geometry.DOD
#         self.step_angle = geometry.step_angle
#         self.init_angle = geometry.init_angle
#         self.nprojs = geometry.nprojs
#         return self 