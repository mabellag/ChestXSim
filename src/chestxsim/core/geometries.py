from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Any



@dataclass
class Geometry(ABC):
    """
    Abstract base class for acquisition geometries.
    """
    detector_size: Tuple[int, int] # width, height (pixels)
    pixel_size: Tuple[float, float] #width, height (mm)
    binning_proj: int 
    SDD: Union[int, float]  # Source-to-detector distance (mm)
 

    def to_dict(self)-> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Geometry":
        return cls(**data)  #  pass dictionary keys as named arguments to the class constructor.

    @abstractmethod
    def fov(self) -> Tuple[float, float, float]:
        """
        Return the physical Field of View (FOV) of the system in mm.
        Format: (x, y, z) where:
            - x: lateral
            - y: anterior-posterior
            - z: axial (top-bottom)

                     
               z top to bottom :: this aligns with detector hieght
               |
               | __ x side to side  :: this aligns with detector width
              /
            y anterior-posterior 
        
        """
      
        pass


@dataclass
class TomoGeometry(Geometry):
    """
    Geometry  for Digital Chest Tomosynthesis (DCT).
    """
    bucky: float 
    step: float
    nstep: int

    @property
    def modality(self):
        return "Tomosynthesis"

    @property
    def magnification(self):
        # source-detector distance over source-object distance 
        # tell us how many times larger the detector image is compared to the object 
        SOD = self.SDD - self.bucky 
        return  self.SDD / SOD 


    def fov(self) -> Tuple[float, float, float]:
       
        u_mm = self.detector_size[0]* self.pixel_size[0]  # detector width
        v_mm= self.detector_size[1]* self.pixel_size[1]  # detector height

        x_mm = u_mm/self.magnification
        z_mm= v_mm/self.magnification

        # compute y as the intersection between 
        s= self.nstep*self.step  # source displacement in mm
        y_mm = self.SDD/( 0.5*(s/z_mm)+1)
        
        return [round(val, 2) for val in[x_mm, y_mm, z_mm]]
    

@dataclass 
class CBCTGeometry(Geometry):
    """
    Geometry  for Cone-Beam CT acquisitions.
    """
    DOD: Union[int, float]   # Distance object–detector
    nprojs: int 
    step_angle: Union[int, float] 
    init_angle: Union[int, float]

    @property
    def modality(self):
        return "CBCT"

    @property
    def magnification(self):
        # source-detector distance over source-object distance 
        # tell us how many times larger the detector image is compared to the object 
        SOD = self.SDD - self.DOD
        return  self.SDD / SOD


    def fov(self) -> Tuple[float, float, float]:
        u_mm = self.detector_size[0]* self.pixel_size[0]  # detector width
        v_mm= self.detector_size[1]* self.pixel_size[1]  # detector height

        x_mm = u_mm/self.magnification # fov diameter
        y_mm = v_mm/self.magnification # fov height

        SOD = self.SDD - self.DOD
        z_mm_extra= v_mm**2 / (4*self.magnification*(SOD + self.DOD))
        
        z_mm= y_mm - 2*z_mm_extra # chequear esto es correcto 

        return [round(val, 2) for val in[x_mm, x_mm, z_mm]]