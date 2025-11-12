from dataclasses import dataclass, asdict
from typing import  Union, Tuple, List, ClassVar, Optional
import numpy as np
from enum import Enum


class Modality(Enum):
    TOMO= "TOMO" 
    CT3D = "CT3D"
    CT2D = "CT2D"

class BeamGeom(Enum):
    CONE = "cone"
    PARALLEL3D = 'parallel3d'
    FANFLAT = "fanflat"
    PARALLEL = "parallel"

@dataclass
class Geometry:
    "Base class to define acquisition geometries"
    # class varibales fixed  by class types
    modality: ClassVar[Modality]
    beam: ClassVar[BeamGeom]
    is3d: ClassVar[bool]

    # instance configurable attributes for all geometries
    detector_size: List[int]    # [W, H] pre-binning
    pixel_size: List[float]     # [pxW, pxH]
    binning_proj: int
  
    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Geometry":
        # validate class required arguments 
        """Construct from a plain dict (keys must match dataclass field names)."""
        return cls(**data)

    @property
    def W(self) -> int:
        # Always use index 0 (works for both 1D [W] and 2D [W, H])
        return int(self.detector_size[0]) // max(1, self.binning_proj)

    @property
    def H(self) -> int | None:
        # Only available when height exists (3D geometries)
        return (int(self.detector_size[1]) // max(1, self.binning_proj)
                if len(self.detector_size) > 1 else None)

    @property
    def pxW(self) -> float:
        return float(self.pixel_size[0]) * max(1, self.binning_proj)

    @property
    def pxH(self) -> float | None:
        return (float(self.pixel_size[1]) * max(1, self.binning_proj)
                if len(self.pixel_size) > 1 else None)
    
    def _validate_detector_dims(self):
        expected = 2 if self.is3d else 1

        # Normalize to tuple/list if single value provided
        if expected == 1:
            if isinstance(self.detector_size, (int, float)):
                self.detector_size = [self.detector_size]
            if isinstance(self.pixel_size, (int, float)):
                self.pixel_size = [self.pixel_size]

        if len(self.detector_size) != expected:
            raise ValueError(
                f"detector_size has {len(self.detector_size)} values, expected {expected} "
                f"({'W,H' if expected == 2 else 'W'})."
            )
        if len(self.pixel_size) != expected:
            raise ValueError(f"pixel_size has {len(self.pixel_size)} values, expected {expected}.")
        

@dataclass(kw_only=True)
class CircularTrajectory():
    """
    attributes for describing x-ray source moves in circular trajectory with detector
    """
    nprojs: int
    step_angle: Union[int, float] 
    init_angle: Union[int, float] 
    
    @property
    def angles(self) -> np.ndarray:
        return np.deg2rad(self.init_angle + self.step_angle * np.arange(self.nprojs)).astype(np.float32)

@dataclass(kw_only=True)
class LinearTrajectory:
    """ 
    attributes for describing x-ray source moves in a linear trajectory
    """
    nprojs: int
    step_mm: float


@dataclass(kw_only=True)
class FocusedBeam:
    """Attributes for geometries using a finite focal point (Cone/Fan)."""
    SDD: float
    DOD: Optional[float]=None # Detector - Object Distance 

    @property
    def SOD(self) -> float: # Source - object       
        return self.SDD - self.DOD


@dataclass 
class TomoGeometry(Geometry,LinearTrajectory,FocusedBeam):
    modality: ClassVar[Modality] = Modality.TOMO
    beam:  ClassVar[BeamGeom] = BeamGeom.CONE
    is3d: ClassVar[bool] = True
    bucky: float

    def __post_init__(self):
        self._validate_detector_dims()


    def fit_to_volume(self, vol_dim_xyz, vx_xyz)->None:
        """Personalize DOD based on this volume’s depth."""
        self.DOD = self.bucky + 0.5*vol_dim_xyz[2]*vx_xyz[2]
        
    

    # @property
    # def magnification(self):
    #     # source-detector distance over source-object distance 
    #     # tell us how many times larger the detector image is compared to the object 
    #     SOD = self.SDD - self.bucky 
    #     return  self.SDD / SOD

    # @property 
    # def fov(self)-> Tuple [float, float, float]:
    #     u_mm = self.W* self.pxW  # detector width
    #     v_mm= self.H* self.pxH  # detector height

    #     x_mm = u_mm/self.magnification
    #     z_mm= v_mm/self.magnification

    #     # compute y as the intersection between 
    #     s= self.nprojs*self.nprojs  # source displacement in mm
    #     y_mm = self.SDD/( 0.5*(s/z_mm)+1)
        
    #     return [round(val, 2) for val in[x_mm, y_mm, z_mm]]


@dataclass 
class CBCTGeometry(Geometry,CircularTrajectory,FocusedBeam):
    modality: ClassVar[Modality] = Modality.CT3D
    beam: ClassVar[BeamGeom] = BeamGeom.CONE
    is3d: ClassVar[bool] = True
    
    def __post_init__(self):
        self._validate_detector_dims()
        

    @property
    def magnification(self):
        # source-detector distance over source-object distance 
        # tell us how many times larger the detector image is compared to the object 
        return  self.SDD / self.SOD

    @property
    def fov(self) -> Tuple[float, float, float]:
        u_mm = self.W* self.pxW    # detector width
        v_mm= self.H* self.pxH  # detector height

        x_mm = u_mm/self.magnification # fov diameter
        y_mm = v_mm/self.magnification # fov height

        z_mm_extra= v_mm**2 / (4*self.magnification*(self.SOD + self.DOD))
        
        z_mm= y_mm - 2*z_mm_extra # chequear esto es correcto 

        return [round(val, 2) for val in[x_mm, y_mm, z_mm]]
    

@dataclass 
class CT2DFanGeometry(Geometry, CircularTrajectory, FocusedBeam):
    modality: ClassVar[Modality]  = Modality.CT2D
    beam: ClassVar[BeamGeom] = BeamGeom.FANFLAT
    is3d: ClassVar[bool] = False
    def __post_init__(self):
        self._validate_detector_dims()

@dataclass 
class CT2DParallelGeometry(Geometry, CircularTrajectory):
    modality: ClassVar[Modality] = Modality.CT2D
    beam: ClassVar[BeamGeom] = BeamGeom.PARALLEL
    is3d: ClassVar[bool] = False
    def __post_init__(self):
        self._validate_detector_dims()
    

### ===== FACTORY METHOD TO CREATE GEOMETRIES ====
GEOMETRY_REGISTRY: dict = {
    (Modality.CT3D, BeamGeom.CONE): CBCTGeometry,
    (Modality.TOMO, BeamGeom.CONE): TomoGeometry,
    (Modality.CT2D, BeamGeom.FANFLAT): CT2DFanGeometry, 
    (Modality.CT2D, BeamGeom.PARALLEL): CT2DParallelGeometry,
}

GEOMETRY_ID = {
    "CBCT": CBCTGeometry,
    "DCT": TomoGeometry,
    "FANBEAM": CT2DFanGeometry, 
    "PARALEL": CT2DParallelGeometry,
}

def create_geometry(modality: Modality, beam: BeamGeom, **kwargs)-> Geometry:
    """
    Factory function to create the correct Geometry instance 
    based on modality and beam type.
    """
    geom_class = GEOMETRY_REGISTRY.get((modality, beam))
    
    if geom_class is None:
        raise ValueError(f"Invalid/unsupported combination: {modality.value}/{beam.value}")

    instance = geom_class(
        **kwargs # detector_size, pixel_size, SOD, DOD, nprojs, etc.
    )
    
    # Final check that the class enforced the correct Modality/Beam
    if instance.modality != modality or instance.beam != beam:
         raise RuntimeError("Concrete geometry failed to enforce Modality/Beam.")
    
    return instance

def create_geometry_from_id(geom_id:str, **kwargs):
    geom_id = geom_id.strip().upper()
    geom_class = GEOMETRY_ID.get(geom_id)
    if geom_class is None:
        valid = ", ".join(GEOMETRY_ID.keys())
        raise ValueError(f"Invalid geometry_id '{geom_id}'. Valid options: {valid}")
    return geom_class(**kwargs)


































# codigo anterior::

# @dataclass
# class Geometry(ABC):
#     """
#     Abstract base class for acquisition geometries.
#     """
#     detector_size: Tuple[int, int] # width, height (pixels)
#     pixel_size: Tuple[float, float] #width, height (mm)
#     binning_proj: int 
#     SDD: Union[int, float]  # Source-to-detector distance (mm)
 

#     def to_dict(self)-> Dict[str, Any]:
#         return asdict(self)
    
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "Geometry":
#         return cls(**data)  #  pass dictionary keys as named arguments to the class constructor.

#     @abstractmethod
#     def fov(self) -> Tuple[float, float, float]:
#         """
#         Return the physical Field of View (FOV) of the system in mm.
#         Format: (x, y, z) where:
#             - x: lateral
#             - y: anterior-posterior
#             - z: axial (top-bottom)

                     
#                z top to bottom :: this aligns with detector hieght
#                |
#                | __ x side to side  :: this aligns with detector width
#               /
#             y anterior-posterior 
        
#         """
      
#         pass


# @dataclass
# class TomoGeometry(Geometry):
#     """
#     Geometry  for Digital Chest Tomosynthesis (DCT).
#     """
#     bucky: float 
#     step: float
#     nstep: int

#     @property
#     def modality(self):
#         return "Tomosynthesis"

#     @property
#     def magnification(self):
#         # source-detector distance over source-object distance 
#         # tell us how many times larger the detector image is compared to the object 
#         SOD = self.SDD - self.bucky 
#         return  self.SDD / SOD 


#     def fov(self) -> Tuple[float, float, float]:
       
#         u_mm = self.detector_size[0]* self.pixel_size[0]  # detector width
#         v_mm= self.detector_size[1]* self.pixel_size[1]  # detector height

#         x_mm = u_mm/self.magnification
#         z_mm= v_mm/self.magnification

#         # compute y as the intersection between 
#         s= self.nstep*self.step  # source displacement in mm
#         y_mm = self.SDD/( 0.5*(s/z_mm)+1)
        
#         return [round(val, 2) for val in[x_mm, y_mm, z_mm]]
    

# @dataclass 
# class CBCTGeometry(Geometry):
#     """
#     Geometry  for Cone-Beam CT acquisitions.
#     """
#     DOD: Union[int, float]   # Distance object–detector
#     nprojs: int 
#     step_angle: Union[int, float] 
#     init_angle: Union[int, float]

#     @property
#     def modality(self):
#         return "CBCT"

#     @property
#     def magnification(self):
#         # source-detector distance over source-object distance 
#         # tell us how many times larger the detector image is compared to the object 
#         SOD = self.SDD - self.DOD
#         return  self.SDD / SOD


#     def fov(self) -> Tuple[float, float, float]:
#         u_mm = self.detector_size[0]* self.pixel_size[0]  # detector width
#         v_mm= self.detector_size[1]* self.pixel_size[1]  # detector height

#         x_mm = u_mm/self.magnification # fov diameter
#         y_mm = v_mm/self.magnification # fov height

#         SOD = self.SDD - self.DOD
#         z_mm_extra= v_mm**2 / (4*self.magnification*(SOD + self.DOD))
        
#         z_mm= y_mm - 2*z_mm_extra # chequear esto es correcto 

#         return [round(val, 2) for val in[x_mm, x_mm, z_mm]]